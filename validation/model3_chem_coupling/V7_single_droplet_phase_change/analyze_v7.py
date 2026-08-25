#!/usr/bin/env python3
"""Analyze V7 single-droplet phase-change validation output.

The script reads raw post-process arrays from D/*.dat.  Each row is treated as
physical coordinates followed by one field value; column 0 is x, column 1 is y,
and the final column is the field value.  Rows are sorted by coordinate, so the
analysis is not tied to MPI rank tiling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve()
VALIDATION_DIR = HERE.parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

try:
    from common import SK54_CTPHASE, SK54_CTFILE, load_solution
except Exception:  # pragma: no cover - keeps the analyzer usable without helper imports.
    SK54_CTFILE = "validation/model3_chem_coupling/mechanisms/yao_sk54.yaml"
    SK54_CTPHASE = "yao_sk54"
    load_solution = None


DAT_RE = re.compile(r"^(?P<var>.+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
SPECIES_RE = re.compile(r"^\s*(?P<idx>\d+)\s*Y_\{?(?P<name>[^}\s]+)\}?")
CASE_VALUE_RE = re.compile(r"""['"](?P<key>fluid_pp\((?P<fluid>\d+)\)%(?P<field>gamma|pi_inf|qv))['"]\s*:\s*(?P<value>[-+0-9.eE]+)""")
COMPOSITION_RE = re.compile(r"([A-Z][a-z]?)\s*:\s*([-+0-9.eE]+)")

NUM_FLUIDS = 3
CONT_BEG = 1
MOM_BEG = 4
E_IDX = 6
ADV_BEG = 7
INT_EN_BEG = 10
DEFAULT_FUEL_SPECIES_ID = 46
GAS_CONSTANT = 8314.46261815324
ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ar": 39.95,
}


@dataclass(frozen=True)
class Field:
    coords: List[Tuple[float, float]]
    values: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ctfile", default=SK54_CTFILE)
    parser.add_argument("--ctphase", default=SK54_CTPHASE)
    parser.add_argument("--fuel-species-id", type=int, default=DEFAULT_FUEL_SPECIES_ID)
    parser.add_argument("--closure-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--closure-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--negative-tol", type=float, default=-1.0e-12)
    parser.add_argument("--p2-p3-rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--p2-p3-abs-tol", type=float, default=1.0e-1)
    parser.add_argument("--evolution-tol", type=float, default=1.0e-8)
    return parser.parse_args()


def parse_dat_name(path: Path) -> Optional[Tuple[str, int]]:
    match = DAT_RE.match(path.name)
    if match is None:
        return None
    return match.group("var"), int(match.group("step"))


def data_roots(run_dir: Path) -> List[Path]:
    roots = []
    for candidate in (run_dir / "D", run_dir):
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def iter_dat_files(run_dir: Path) -> Iterable[Tuple[str, int, Path]]:
    for root in data_roots(run_dir):
        for path in root.glob("*.dat"):
            parsed = parse_dat_name(path)
            if parsed is not None:
                var, step = parsed
                yield var, step, path


def available_steps(run_dir: Path, var: str) -> List[int]:
    return sorted({step for file_var, step, _ in iter_dat_files(run_dir) if file_var == var})


def var_exists(run_dir: Path, var: str, step: int) -> bool:
    suffix = f".{step:06d}.dat"
    return any(file_var == var and file_step == step and path.name.endswith(suffix) for file_var, file_step, path in iter_dat_files(run_dir))


def read_field(run_dir: Path, var: str, step: int) -> Field:
    rows: List[Tuple[Tuple[float, float], float]] = []
    suffix = f".{step:06d}.dat"
    for file_var, file_step, path in iter_dat_files(run_dir):
        if file_var != var or file_step != step or not path.name.endswith(suffix):
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    value = float(parts[-1])
                except ValueError:
                    continue
                rows.append(((x, y), value))
    if not rows:
        raise FileNotFoundError(f"no {var} data found at step {step}")
    rows.sort(key=lambda item: (item[0][0], item[0][1]))
    return Field([row[0] for row in rows], [row[1] for row in rows])


def parse_species_from_indices(run_dir: Path) -> Dict[int, str]:
    path = run_dir / "indices.dat"
    species: Dict[int, str] = {}
    if not path.is_file():
        return species
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SPECIES_RE.match(line)
            if match:
                species[int(match.group("idx"))] = match.group("name")
    return species


def mechanism_candidates(run_dir: Path, ctfile: str) -> List[Path]:
    raw = Path(ctfile)
    return [
        raw,
        run_dir / ctfile,
        run_dir.parents[1] / ctfile if len(run_dir.parents) > 1 else raw,
        run_dir.parents[0] / "mechanisms" / raw.name,
        VALIDATION_DIR / "mechanisms" / raw.name,
    ]


def molecular_weights_from_yaml(run_dir: Path, ctfile: str) -> Dict[str, float]:
    path = next((candidate for candidate in mechanism_candidates(run_dir, ctfile) if candidate.is_file()), None)
    if path is None:
        return {}

    molecular_weights: Dict[str, float] = {}
    current_species: Optional[str] = None
    in_species_block = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "species:":
                in_species_block = True
                current_species = None
                continue
            if not in_species_block:
                continue
            if stripped.startswith("- name:"):
                current_species = stripped.split(":", 1)[1].strip().strip("'\"")
                continue
            if current_species and stripped.startswith("composition:"):
                composition = {element: float(count) for element, count in COMPOSITION_RE.findall(stripped)}
                weight = sum(ATOMIC_WEIGHTS[element] * count for element, count in composition.items() if element in ATOMIC_WEIGHTS)
                if weight > 0.0:
                    molecular_weights[current_species] = weight
                current_species = None
    return molecular_weights


def load_species_metadata(run_dir: Path, ctfile: str, ctphase: str) -> Tuple[Dict[int, str], Dict[str, float]]:
    species_by_var = parse_species_from_indices(run_dir)
    molecular_weights: Dict[str, float] = {}

    if load_solution is not None:
        try:
            sol = load_solution(ctfile, ctphase)
            molecular_weights = {name: float(mw) for name, mw in zip(sol.species_names, sol.molecular_weights)}
            if not species_by_var:
                species_beg = max(int(var.split(".")[1]) for var, _, _ in iter_dat_files(run_dir) if var.startswith("cons.")) - sol.n_species + 1
                species_by_var = {species_beg + idx: name for idx, name in enumerate(sol.species_names)}
        except Exception:
            pass

    if not molecular_weights:
        molecular_weights = molecular_weights_from_yaml(run_dir, ctfile)

    return species_by_var, molecular_weights


def parse_fluid_parameters(run_dir: Path) -> Dict[int, Dict[str, float]]:
    params = {
        1: {"gamma": 0.7407407407407407, "pi_inf": 835555555.5555555, "qv": -758060.0},
        2: {"gamma": 40.00000000000014, "pi_inf": 0.0, "qv": -234490.0},
        3: {"gamma": 2.5000000000000004, "pi_inf": 0.0, "qv": 0.0},
    }
    case_path = run_dir / "case.py"
    if not case_path.is_file():
        return params
    text = case_path.read_text(encoding="utf-8", errors="replace")
    for match in CASE_VALUE_RE.finditer(text):
        fluid = int(match.group("fluid"))
        field = match.group("field")
        params.setdefault(fluid, {})[field] = float(match.group("value"))
    return params


def median_spacing(values: Sequence[float]) -> float:
    unique = sorted(set(values))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return statistics.median(diffs) if diffs else 1.0


def cell_measure(coords: Sequence[Tuple[float, float]]) -> float:
    dx = median_spacing([coord[0] for coord in coords])
    dy = median_spacing([coord[1] for coord in coords])
    return dx * dy


def fsum(values: Sequence[float]) -> float:
    return math.fsum(values)


def max_abs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max(abs(x - y) for x, y in zip(a, b)) if a else math.nan


def field_integral(field: Field, measure: float) -> float:
    return fsum(field.values) * measure


def mixture_temperature(pressure: float, rho_g: float, y_values: Sequence[float], species_names: Sequence[str], molecular_weights: Dict[str, float]) -> float:
    if pressure <= 0.0 or rho_g <= 0.0:
        return math.nan
    if not molecular_weights:
        return math.nan
    inv_w = 0.0
    for species, y in zip(species_names, y_values):
        mw = molecular_weights.get(species)
        if mw and mw > 0.0:
            inv_w += y / mw
    if inv_w <= 0.0 or not math.isfinite(inv_w):
        return math.nan
    w_mix = 1.0 / inv_w
    return pressure * w_mix / (rho_g * GAS_CONSTANT)


def status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def info_status(value: str = "INFO") -> str:
    return value


def load_step(run_dir: Path, step: int, species_by_var: Dict[int, str], molecular_weights: Dict[str, float], fluid_params: Dict[int, Dict[str, float]]) -> Dict[str, object]:
    fields: Dict[str, Field] = {}
    required_vars = [f"cons.{idx}" for idx in range(1, 13)] + [f"cons.{idx}" for idx in species_by_var]
    for var in required_vars:
        fields[var] = read_field(run_dir, var, step)

    coords = fields["cons.1"].coords
    for var, field in fields.items():
        if field.coords != coords:
            raise ValueError(f"{var} coordinates differ from cons.1 at step {step}")

    measure = cell_measure(coords)
    arho = [fields[f"cons.{idx}"].values for idx in range(1, 4)]
    mom = [fields[f"cons.{idx}"].values for idx in range(MOM_BEG, E_IDX)]
    energy = fields[f"cons.{E_IDX}"].values
    alpha = [fields[f"cons.{idx}"].values for idx in range(ADV_BEG, ADV_BEG + NUM_FLUIDS)]
    int_en = [fields[f"cons.{idx}"].values for idx in range(INT_EN_BEG, INT_EN_BEG + NUM_FLUIDS)]
    species_vars = sorted(species_by_var)
    species_names = [species_by_var[var] for var in species_vars]
    rho_y = [fields[f"cons.{idx}"].values for idx in species_vars]

    gas_mass = [a2 + a3 for a2, a3 in zip(arho[1], arho[2])]
    alpha_g = [a2 + a3 for a2, a3 in zip(alpha[1], alpha[2])]
    rho_g_intrinsic = [gm / ag if ag > 1.0e-30 else math.nan for gm, ag in zip(gas_mass, alpha_g)]
    sum_rho_y = [fsum(cell_species) for cell_species in zip(*rho_y)]
    closure = [sr - gm for sr, gm in zip(sum_rho_y, gas_mass)]
    closure_abs = [abs(value) for value in closure]
    closure_rel = [abs(value) / max(abs(gm), 1.0e-30) for value, gm in zip(closure, gas_mass)]

    p2 = []
    p3 = []
    temperatures = []
    for cell in range(len(coords)):
        pressures = []
        for fluid_id, local_idx in ((2, 1), (3, 2)):
            alpha_i = alpha[local_idx][cell]
            if alpha_i <= 1.0e-30:
                pressures.append(math.nan)
                continue
            pars = fluid_params[fluid_id]
            pressure_i = ((int_en[local_idx][cell] - arho[local_idx][cell] * pars["qv"]) / alpha_i - pars["pi_inf"]) / pars["gamma"]
            pressures.append(pressure_i)
        p2.append(pressures[0])
        p3.append(pressures[1])
        finite_pressures = [p for p in pressures if math.isfinite(p)]
        p_g = fsum(finite_pressures) / len(finite_pressures) if finite_pressures else math.nan
        y_values = [values[cell] / gas_mass[cell] if gas_mass[cell] > 1.0e-30 else math.nan for values in rho_y]
        temperatures.append(mixture_temperature(p_g, rho_g_intrinsic[cell], y_values, species_names, molecular_weights))

    min_rho_y = math.inf
    min_y = math.inf
    min_species = ""
    min_cell = -1
    for name, values in zip(species_names, rho_y):
        for cell, value in enumerate(values):
            y = value / gas_mass[cell] if gas_mass[cell] > 1.0e-30 else math.nan
            if value < min_rho_y:
                min_rho_y = value
                min_species = name
                min_cell = cell
            if math.isfinite(y) and y < min_y:
                min_y = y

    return {
        "step": step,
        "coords": coords,
        "measure": measure,
        "arho": arho,
        "mom": mom,
        "energy": energy,
        "alpha": alpha,
        "int_en": int_en,
        "species_names": species_names,
        "rho_y": rho_y,
        "gas_mass": gas_mass,
        "alpha_g": alpha_g,
        "rho_g_intrinsic": rho_g_intrinsic,
        "sum_rho_y": sum_rho_y,
        "closure_abs": closure_abs,
        "closure_rel": closure_rel,
        "p2": p2,
        "p3": p3,
        "temperature": temperatures,
        "min_rho_y": min_rho_y,
        "min_y": min_y,
        "min_species": min_species,
        "min_cell": min_cell,
    }


def complete_steps(run_dir: Path, species_by_var: Dict[int, str]) -> Tuple[List[int], Dict[int, List[str]]]:
    candidate_steps = available_steps(run_dir, "cons.1")
    required = [f"cons.{idx}" for idx in range(1, 13)] + [f"cons.{idx}" for idx in species_by_var]
    complete: List[int] = []
    missing: Dict[int, List[str]] = {}
    for step in candidate_steps:
        absent = [var for var in required if not var_exists(run_dir, var, step)]
        if absent:
            missing[step] = absent
        else:
            complete.append(step)
    return complete, missing


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "V7_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    species_by_var, molecular_weights = load_species_metadata(run_dir, args.ctfile, args.ctphase)
    if not species_by_var:
        raise RuntimeError("Could not determine chemistry species variables from indices.dat or Cantera")
    fuel_var = next((var for var, name in species_by_var.items() if var - min(species_by_var) + 1 == args.fuel_species_id), None)
    fuel_name = species_by_var.get(fuel_var, f"species_id_{args.fuel_species_id}") if fuel_var is not None else f"species_id_{args.fuel_species_id}"
    if fuel_var is None:
        raise RuntimeError(f"fuel species id {args.fuel_species_id} is outside parsed species range")

    fluid_params = parse_fluid_parameters(run_dir)
    steps, missing = complete_steps(run_dir, species_by_var)

    if not steps:
        summary = [
            "# V7 Single-Droplet Phase-Change Analysis",
            "",
            "No complete saved conservative states were available yet.",
            f"Incomplete steps: {missing}",
        ]
        (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("\n".join(summary))
        return 2

    states = [load_step(run_dir, step, species_by_var, molecular_weights, fluid_params) for step in steps]
    first = states[0]
    final = states[-1]

    fuel_offset = sorted(species_by_var).index(fuel_var)
    rows = []
    max_abs_closure = 0.0
    max_rel_closure = 0.0
    worst_closure = (None, None)
    min_rho_y = math.inf
    min_y = math.inf
    min_species = ""
    min_where = (None, None)
    min_p = math.inf
    min_t = math.inf
    max_p2p3_rel = 0.0
    max_p2p3_abs = 0.0
    max_p2p3_where = (None, None)

    ml0 = field_integral(Field(first["coords"], first["arho"][0]), first["measure"])
    mv0 = field_integral(Field(first["coords"], first["arho"][1]), first["measure"])
    mf0 = field_integral(Field(first["coords"], first["rho_y"][fuel_offset]), first["measure"])

    for state in states:
        step = int(state["step"])
        measure = float(state["measure"])
        ml = field_integral(Field(state["coords"], state["arho"][0]), measure)
        mv = field_integral(Field(state["coords"], state["arho"][1]), measure)
        mf = field_integral(Field(state["coords"], state["rho_y"][fuel_offset]), measure)
        gas = fsum(state["gas_mass"]) * measure
        max_abs = max(state["closure_abs"])
        max_rel = max(state["closure_rel"])
        closure_cell = state["closure_abs"].index(max_abs)
        if max_abs > max_abs_closure:
            max_abs_closure = max_abs
            max_rel_closure = max_rel
            worst_closure = (step, closure_cell)
        if state["min_rho_y"] < min_rho_y:
            min_rho_y = state["min_rho_y"]
            min_y = state["min_y"]
            min_species = state["min_species"]
            min_where = (step, state["min_cell"])
        finite_p = [p for pair in zip(state["p2"], state["p3"]) for p in pair if math.isfinite(p)]
        finite_t = [t for t in state["temperature"] if math.isfinite(t)]
        if finite_p:
            min_p = min(min_p, min(finite_p))
        if finite_t:
            min_t = min(min_t, min(finite_t))
        for cell, (p2, p3) in enumerate(zip(state["p2"], state["p3"])):
            if math.isfinite(p2) and math.isfinite(p3):
                abs_err = abs(p2 - p3)
                rel = abs(p2 - p3) / max(abs(0.5 * (p2 + p3)), 1.0)
                if rel > max_p2p3_rel:
                    max_p2p3_rel = rel
                    max_p2p3_abs = abs_err
                    max_p2p3_where = (step, cell)
        pressure_values = [p for p in state["p3"] if math.isfinite(p)]
        temperature_values = [t for t in state["temperature"] if math.isfinite(t)]
        rows.append(
            {
                "step": step,
                "cell_measure": f"{measure:.17e}",
                "M_liquid": f"{ml:.17e}",
                "M_vapor_fluid": f"{mv:.17e}",
                f"M_{fuel_name}": f"{mf:.17e}",
                "M_gas_stored": f"{gas:.17e}",
                "delta_M_liquid": f"{ml - ml0:.17e}",
                "delta_M_vapor_fluid": f"{mv - mv0:.17e}",
                f"delta_M_{fuel_name}": f"{mf - mf0:.17e}",
                "minus_delta_liquid_minus_delta_vapor": f"{-(ml - ml0) - (mv - mv0):.17e}",
                "delta_vapor_minus_delta_fuel": f"{(mv - mv0) - (mf - mf0):.17e}",
                "max_abs_closure_error": f"{max_abs:.17e}",
                "max_rel_closure_error": f"{max_rel:.17e}",
                "min_rhoY": f"{state['min_rho_y']:.17e}",
                "min_Y": f"{state['min_y']:.17e}",
                "p_min_Pa": f"{min(pressure_values):.17e}" if pressure_values else "nan",
                "p_max_Pa": f"{max(pressure_values):.17e}" if pressure_values else "nan",
                "T_min_K": f"{min(temperature_values):.17e}" if temperature_values else "nan",
                "T_max_K": f"{max(temperature_values):.17e}" if temperature_values else "nan",
            }
        )

    profile_changes = {
        "liquid_arho": max_abs_delta(final["arho"][0], first["arho"][0]),
        "vapor_arho": max_abs_delta(final["arho"][1], first["arho"][1]),
        "fuel_rhoY": max_abs_delta(final["rho_y"][fuel_offset], first["rho_y"][fuel_offset]),
        "pressure_p3": max_abs_delta(final["p3"], first["p3"]),
        "mom_x": max_abs_delta(final["mom"][0], first["mom"][0]),
        "mom_y": max_abs_delta(final["mom"][1], first["mom"][1]),
    }
    shock_drop_evolved = max(profile_changes.values()) > args.evolution_tol if len(states) > 1 else False

    final_ml = field_integral(Field(final["coords"], final["arho"][0]), final["measure"])
    final_mv = field_integral(Field(final["coords"], final["arho"][1]), final["measure"])
    final_mf = field_integral(Field(final["coords"], final["rho_y"][fuel_offset]), final["measure"])
    dml = final_ml - ml0
    dmv = final_mv - mv0
    dmf = final_mf - mf0

    checks = [
        ("complete_saved_states", len(states) >= 2, f"{len(states)} complete saves", ">= 2"),
        ("cell_local_species_gas_closure_abs", max_abs_closure <= args.closure_abs_tol, f"{max_abs_closure:.6e}", f"<= {args.closure_abs_tol:.1e}"),
        ("cell_local_species_gas_closure_rel", max_rel_closure <= args.closure_rel_tol, f"{max_rel_closure:.6e}", f"<= {args.closure_rel_tol:.1e}"),
        ("minimum_rhoY", min_rho_y >= args.negative_tol, f"{min_rho_y:.6e}", f">= {args.negative_tol:.1e}"),
        ("minimum_Y", min_y >= args.negative_tol, f"{min_y:.6e}", f">= {args.negative_tol:.1e}"),
        ("positive_pressure", math.isfinite(min_p) and min_p > 0.0, f"{min_p:.6e}", "> 0"),
        ("positive_temperature", math.isfinite(min_t) and min_t > 0.0, f"{min_t:.6e}", "> 0"),
        (
            "p2_p3_consistency",
            max_p2p3_rel <= args.p2_p3_rel_tol or max_p2p3_abs <= args.p2_p3_abs_tol,
            f"{max_p2p3_rel:.6e} rel / {max_p2p3_abs:.6e} Pa",
            f"<= {args.p2_p3_rel_tol:.1e} rel or <= {args.p2_p3_abs_tol:.1e} Pa",
        ),
        ("shock_droplet_evolution", shock_drop_evolved, f"{max(profile_changes.values()):.6e}", f"> {args.evolution_tol:.1e}"),
    ]

    overall = all(item[1] for item in checks)

    csv_path = output_dir / "v7_time_history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "run_dir": str(run_dir),
        "steps": steps,
        "missing_incomplete_steps": {str(step): values for step, values in missing.items()},
        "species_by_var": {str(var): name for var, name in species_by_var.items()},
        "fuel_species": {"id": args.fuel_species_id, "name": fuel_name, "var": fuel_var},
        "profile_changes": profile_changes,
        "worst_closure": {"step": worst_closure[0], "cell": worst_closure[1], "abs": max_abs_closure, "rel": max_rel_closure},
        "minimum_species": {"name": min_species, "step": min_where[0], "cell": min_where[1], "rhoY": min_rho_y, "Y": min_y},
        "p2_p3": {"max_rel": max_p2p3_rel, "max_abs_Pa": max_p2p3_abs, "where": max_p2p3_where},
        "global_deltas_final_minus_step0": {
            "delta_M_liquid": dml,
            "delta_M_vapor_fluid": dmv,
            f"delta_M_{fuel_name}": dmf,
            "minus_delta_liquid_minus_delta_vapor": -dml - dmv,
            "delta_vapor_minus_delta_fuel": dmv - dmf,
        },
        "checks": {name: status(ok) for name, ok, _, _ in checks},
        "overall": status(overall),
    }
    (output_dir / "v7_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# V7 Single-Droplet Phase-Change Analysis",
        "",
        f"Run directory: `{run_dir}`",
        f"Complete saved steps analyzed: {steps}",
        f"Fuel species: `{fuel_name}` (id {args.fuel_species_id}, conservative variable cons.{fuel_var})",
        "",
        "The phase-change mass relation is reported relative to saved step 0. Because this",
        "case has non-periodic shock-droplet boundaries, global deltas are not treated as",
        "a closed-domain conservation proof after boundary fluxes have entered the run.",
        "",
        "| Component | Value | Criterion | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(f"| {name} | {value} | {criterion} | {status(ok)} |" for name, ok, value, criterion in checks)
    lines.extend(
        [
            "",
            "## Final Deltas vs Saved Step 0",
            "",
            f"- Delta liquid mass: `{dml:.12e}`",
            f"- Delta vapor-fluid mass: `{dmv:.12e}`",
            f"- Delta {fuel_name} mass: `{dmf:.12e}`",
            f"- `-Delta liquid - Delta vapor`: `{(-dml - dmv):.12e}`",
            f"- `Delta vapor - Delta fuel`: `{(dmv - dmf):.12e}`",
            "",
            "## Worst Locations",
            "",
            f"- Closure: step `{worst_closure[0]}`, cell `{worst_closure[1]}`, abs `{max_abs_closure:.12e}`, rel `{max_rel_closure:.12e}`",
            f"- Minimum species: `{min_species}`, step `{min_where[0]}`, cell `{min_where[1]}`, rhoY `{min_rho_y:.12e}`, Y `{min_y:.12e}`",
            f"- p2/p3 mismatch: step `{max_p2p3_where[0]}`, cell `{max_p2p3_where[1]}`, rel `{max_p2p3_rel:.12e}`, abs `{max_p2p3_abs:.12e} Pa`",
            "",
            "## Shock/Drop Evolution Evidence",
            "",
        ]
    )
    lines.extend(f"- {name}: `{value:.12e}`" for name, value in profile_changes.items())
    lines.extend(["", f"Overall V7 verdict: **{status(overall)}**"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
