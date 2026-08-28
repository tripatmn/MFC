#!/usr/bin/env python3
"""Compare paired V10 Model-3 species-diffusion OFF/ON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

HERE = Path(__file__).resolve()
CASE_DIR = HERE.parent
VALIDATION_DIR = HERE.parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from common import SK54_CTPHASE, SK54_CTFILE, SK54_FUEL, load_solution  # noqa: E402

DAT_RE = re.compile(r"^(?P<kind>[A-Za-z_]+)\.(?P<var_id>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
SPECIES_RE = re.compile(r"^\s*(?P<idx>\d+)\s*Y_\{?(?P<name>[^}\s]+)\}?")
CHEM_RANGE_RE = re.compile(r"^\s*\[(?P<beg>\d+)\s*,\s*(?P<end>\d+)\]\s*Chemistry\b")
PHASE_SPECIES_RE = re.compile(r"^\s*species:\s*\[(?P<species>.*)")
SPECIES_NAME_RE = re.compile(r"^\s*-\s*name:\s*(?P<name>[^,\s]+)")
COMPOSITION_RE = re.compile(r"([A-Z][a-z]?)\s*:\s*([-+0-9.eE]+)")
GAS_CONSTANT = 8314.46261815324
ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ar": 39.95,
}
NUM_FLUIDS = 3
CONT_BEG = 1
MOM_BEG = 4
E_IDX = 5
ADV_BEG = 6
INT_EN_BEG = 9


@dataclass(frozen=True)
class Field:
    coords: List[Tuple[float, float]]
    values: List[float]


@dataclass(frozen=True)
class DatFile:
    kind: str
    var_id: int
    rank: int
    step: int
    path: Path

    @property
    def var(self) -> str:
        return f"{self.kind}.{self.var_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-run-dir", type=Path, default=CASE_DIR / "run_diffusion_off")
    parser.add_argument("--on-run-dir", type=Path, default=CASE_DIR / "run_diffusion_on")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "analysis_out")
    parser.add_argument("--ctfile", default=SK54_CTFILE)
    parser.add_argument("--ctphase", default=SK54_CTPHASE)
    parser.add_argument("--fuel-species", default=SK54_FUEL)
    parser.add_argument("--closure-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--closure-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--conservation-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--negative-tol", type=float, default=-1.0e-12)
    parser.add_argument("--off-normalized-change-tol", type=float, default=1.0e-8)
    parser.add_argument("--off-to-on-signal-ratio-tol", type=float, default=1.0e-6)
    parser.add_argument("--small-safe-scale", type=float, default=1.0e-30)
    parser.add_argument("--diffusion-signal-tol", type=float, default=1.0e-12)
    return parser.parse_args()


def data_roots(run_dir: Path) -> List[Path]:
    return [path for path in (run_dir / "D", run_dir) if path.is_dir()]


def parse_dat_name(path: Path) -> DatFile | None:
    match = DAT_RE.match(path.name)
    if match is None:
        return None
    return DatFile(
        kind=match.group("kind"),
        var_id=int(match.group("var_id")),
        rank=int(match.group("rank")),
        step=int(match.group("step")),
        path=path,
    )


def iter_dat_files(run_dir: Path) -> Iterable[DatFile]:
    for root in data_roots(run_dir):
        for path in root.glob("*.dat"):
            parsed = parse_dat_name(path)
            if parsed is not None:
                yield parsed


def available_steps(run_dir: Path, var: str) -> List[int]:
    return sorted({item.step for item in iter_dat_files(run_dir) if item.var == var})


def var_exists(run_dir: Path, var: str, step: int) -> bool:
    return any(item.var == var and item.step == step for item in iter_dat_files(run_dir))


def read_field(run_dir: Path, var: str, step: int) -> Field:
    rows: List[Tuple[Tuple[float, float], float]] = []
    for item in iter_dat_files(run_dir):
        if item.var != var or item.step != step:
            continue
        with item.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1]) if len(parts) > 2 else 0.0
                    value = float(parts[-1])
                except ValueError:
                    continue
                rows.append(((x, y), value))
    if not rows:
        raise FileNotFoundError(f"No {var} data found at step {step} under {run_dir}")
    rows.sort(key=lambda item: (item[0][0], item[0][1]))
    return Field([row[0] for row in rows], [row[1] for row in rows])


def parse_species_from_indices(run_dir: Path) -> Dict[int, str]:
    species: Dict[int, str] = {}
    path = run_dir / "indices.dat"
    if not path.is_file():
        return species
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SPECIES_RE.match(line)
            if match:
                species[int(match.group("idx"))] = match.group("name")
    return species


def parse_chemistry_range_from_indices(run_dir: Path) -> Tuple[int, int] | None:
    path = run_dir / "indices.dat"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = CHEM_RANGE_RE.match(line)
            if match:
                return int(match.group("beg")), int(match.group("end"))
    return None


def mechanism_candidates(run_dir: Path, ctfile: str) -> List[Path]:
    raw = Path(ctfile)
    return [
        raw,
        run_dir / ctfile,
        HERE.parents[1] / ctfile,
        HERE.parents[1] / "mechanisms" / raw.name,
        VALIDATION_DIR / "mechanisms" / raw.name,
    ]


def parse_species_order_from_yaml(path: Path, ctphase: str) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != f"- name: {ctphase}":
            continue
        for candidate in lines[idx + 1:]:
            match = PHASE_SPECIES_RE.match(candidate)
            if match is None:
                continue
            block = match.group("species").strip()
            while "]" not in block:
                idx += 1
                block += " " + lines[idx].strip()
            block = block.split("]", 1)[0]
            return [item.strip().strip("'\"") for item in block.split(",") if item.strip()]
    return []


def parse_molecular_weights_from_yaml(path: Path) -> Dict[str, float]:
    molecular_weights: Dict[str, float] = {}
    current_species = ""
    in_species_block = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "species:":
            in_species_block = True
            current_species = ""
            continue
        if not in_species_block:
            continue
        match = SPECIES_NAME_RE.match(line)
        if match:
            current_species = match.group("name").strip("'\"")
            continue
        if current_species and stripped.startswith("composition:"):
            composition = {element: float(count) for element, count in COMPOSITION_RE.findall(stripped)}
            weight = sum(ATOMIC_WEIGHTS[element] * count for element, count in composition.items() if element in ATOMIC_WEIGHTS)
            if weight > 0.0:
                molecular_weights[current_species] = weight
            current_species = ""
    return molecular_weights


def load_mechanism_metadata(run_dir: Path, ctfile: str, ctphase: str) -> Tuple[List[str], Dict[str, float]]:
    for candidate in mechanism_candidates(run_dir, ctfile):
        if candidate.is_file():
            return parse_species_order_from_yaml(candidate, ctphase), parse_molecular_weights_from_yaml(candidate)
    return [], {}


def species_metadata(run_dir: Path, ctfile: str, ctphase: str) -> Tuple[Dict[int, str], Dict[str, float]]:
    species_by_var = parse_species_from_indices(run_dir)
    mechanism_species, molecular_weights = load_mechanism_metadata(run_dir, ctfile, ctphase)

    if not molecular_weights:
        try:
            sol = load_solution(ctfile, ctphase)
            molecular_weights = {name: float(mw) for name, mw in zip(sol.species_names, sol.molecular_weights)}
            if not mechanism_species:
                mechanism_species = list(sol.species_names)
        except Exception:
            pass

    if not species_by_var:
        chem_range = parse_chemistry_range_from_indices(run_dir)
        cons_ids = sorted({item.var_id for item in iter_dat_files(run_dir) if item.kind == "cons"})
        if chem_range is not None:
            species_beg, species_end = chem_range
        elif mechanism_species:
            species_beg = INT_EN_BEG + NUM_FLUIDS
            species_end = species_beg + len(mechanism_species) - 1
        elif cons_ids:
            raise RuntimeError(
                f"Cannot infer chemistry species names for {run_dir}: indices.dat is missing and mechanism metadata is unavailable. "
                f"Found conserved variable IDs {cons_ids[0]}..{cons_ids[-1]}."
            )
        else:
            raise RuntimeError(f"No conserved MFC D/*.dat files found under {run_dir}")

        names = mechanism_species
        if len(names) != species_end - species_beg + 1:
            names = [f"species_{idx - species_beg + 1}" for idx in range(species_beg, species_end + 1)]
        species_by_var = {species_beg + idx: name for idx, name in enumerate(names)}

    return species_by_var, molecular_weights


def cell_width(coords: Sequence[Tuple[float, float]]) -> float:
    xs = sorted({coord[0] for coord in coords})
    spacings = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if not spacings:
        return 1.0
    spacings.sort()
    return spacings[len(spacings) // 2]


def integral(values: Sequence[float], dx: float) -> float:
    return math.fsum(values) * dx


def finite_all(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def finite_min(values: Iterable[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return min(vals) if vals else math.nan


def finite_max(values: Iterable[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return max(vals) if vals else math.nan


def total_variation_periodic(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return math.fsum(abs(b - a) for a, b in zip(values, values[1:])) + abs(values[0] - values[-1])


def l2_gradient_periodic(values: Sequence[float], dx: float) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [(values[(idx + 1) % len(values)] - values[idx]) / dx for idx in range(len(values))]
    return math.sqrt(math.fsum(diff * diff for diff in diffs) / len(diffs))


def mixture_temperature(pressure: float, rho_g: float, y_values: Sequence[float], names: Sequence[str], molecular_weights: Dict[str, float]) -> float:
    inv_w = 0.0
    for name, y_value in zip(names, y_values):
        mw = molecular_weights.get(name)
        if mw:
            inv_w += y_value / mw
    if pressure <= 0.0 or rho_g <= 0.0 or inv_w <= 0.0:
        return math.nan
    return pressure / (rho_g * GAS_CONSTANT * inv_w)


def load_state(run_dir: Path, step: int, species_by_var: Dict[int, str], molecular_weights: Dict[str, float]) -> Dict[str, object]:
    arho = [read_field(run_dir, f"cons.{CONT_BEG + i}", step).values for i in range(NUM_FLUIDS)]
    mom = read_field(run_dir, f"cons.{MOM_BEG}", step).values
    energy = read_field(run_dir, f"cons.{E_IDX}", step).values
    alpha = [read_field(run_dir, f"cons.{ADV_BEG + i}", step).values for i in range(NUM_FLUIDS)]
    int_en = [read_field(run_dir, f"cons.{INT_EN_BEG + i}", step).values for i in range(NUM_FLUIDS)]
    coords = read_field(run_dir, f"cons.{CONT_BEG}", step).coords
    species_vars = sorted(species_by_var)
    rho_y = [read_field(run_dir, f"cons.{var_id}", step).values for var_id in species_vars]
    names = [species_by_var[var_id] for var_id in species_vars]
    dx = cell_width(coords)

    gas_mass = [a2 + a3 for a2, a3 in zip(arho[1], arho[2])]
    alpha_g = [a2 + a3 for a2, a3 in zip(alpha[1], alpha[2])]
    sum_rho_y = [math.fsum(rho_y_k[cell] for rho_y_k in rho_y) for cell in range(len(gas_mass))]
    closure_abs = [abs(sum_y - gas) for sum_y, gas in zip(sum_rho_y, gas_mass)]
    closure_rel = [err / max(abs(gas), 1.0e-30) for err, gas in zip(closure_abs, gas_mass)]
    y_species = [[value / max(gas_mass[cell], 1.0e-30) for cell, value in enumerate(rho_y_k)] for rho_y_k in rho_y]
    total_density = [a1 + a2 + a3 for a1, a2, a3 in zip(arho[0], arho[1], arho[2])]
    velocity = [m / max(rho, 1.0e-30) for m, rho in zip(mom, total_density)]
    p2 = [((ie2 - arho2 * -234490.0) / max(a2, 1.0e-30)) / 40.00000000000014 for ie2, arho2, a2 in zip(int_en[1], arho[1], alpha[1])]
    p3 = [(ie3 / max(a3, 1.0e-30)) / 2.5000000000000004 for ie3, a3 in zip(int_en[2], alpha[2])]
    pressure = [0.5 * (p2_cell + p3_cell) for p2_cell, p3_cell in zip(p2, p3)]
    p2p3_rel = [abs(p2_cell - p3_cell) / max(abs(p_cell), 1.0) for p2_cell, p3_cell, p_cell in zip(p2, p3, pressure)]
    temperature = []
    for cell, pressure_cell in enumerate(pressure):
        rho_g = gas_mass[cell] / max(alpha_g[cell], 1.0e-30)
        temperature.append(mixture_temperature(pressure_cell, rho_g, [ys[cell] for ys in y_species], names, molecular_weights))

    return {
        "coords": coords,
        "dx": dx,
        "arho": arho,
        "alpha": alpha,
        "energy": energy,
        "rho_y": rho_y,
        "y_species": y_species,
        "gas_mass": gas_mass,
        "closure_abs": closure_abs,
        "closure_rel": closure_rel,
        "velocity": velocity,
        "pressure": pressure,
        "temperature": temperature,
        "p2p3_rel": p2p3_rel,
    }


def complete_steps(run_dir: Path, species_vars: Sequence[int]) -> List[int]:
    required = [
        CONT_BEG,
        CONT_BEG + 1,
        CONT_BEG + 2,
        MOM_BEG,
        E_IDX,
        ADV_BEG,
        ADV_BEG + 1,
        ADV_BEG + 2,
        INT_EN_BEG,
        INT_EN_BEG + 1,
        INT_EN_BEG + 2,
    ]
    required += list(species_vars)
    return [step for step in available_steps(run_dir, f"cons.{CONT_BEG}") if all(var_exists(run_dir, f"cons.{var_id}", step) for var_id in required)]


def half_masks(coords: Sequence[Tuple[float, float]]) -> Tuple[List[bool], List[bool]]:
    xs = [coord[0] for coord in coords]
    midpoint = 0.5 * (min(xs) + max(xs))
    left = [x <= midpoint for x in xs]
    right = [not flag for flag in left]
    return left, right


def masked_integral(values: Sequence[float], mask: Sequence[bool], dx: float) -> float:
    return math.fsum(value for value, keep in zip(values, mask) if keep) * dx


def analyze_run(run_dir: Path, ctfile: str, ctphase: str) -> Dict[str, object]:
    species_by_var, molecular_weights = species_metadata(run_dir, ctfile, ctphase)
    species_vars = sorted(species_by_var)
    steps = complete_steps(run_dir, species_vars)
    if len(steps) < 2:
        raise RuntimeError(f"{run_dir} needs at least two complete saved states; found {steps}")

    states = {step: load_state(run_dir, step, species_by_var, molecular_weights) for step in steps}
    first = states[steps[0]]
    final = states[steps[-1]]
    dx = float(first["dx"])
    left_mask, right_mask = half_masks(first["coords"])

    species_indices = {name.upper(): idx for idx, name in enumerate(species_by_var[var_id] for var_id in species_vars)}
    species_integrals_initial = [integral(values, dx) for values in first["rho_y"]]
    max_species_integral_rel = 0.0
    max_closure_abs = 0.0
    max_closure_rel = 0.0
    min_rho_y = math.inf
    min_y = math.inf
    min_pressure = math.inf
    min_temperature = math.inf
    max_p2p3_rel = 0.0
    all_finite = True
    max_velocity = 0.0
    energy_integral_initial = integral(first["energy"], dx)
    max_energy_integral_rel = 0.0
    final_energy_integral_rel = 0.0
    rows: List[Dict[str, object]] = []

    for step, state in states.items():
        species_rels = []
        for idx, values in enumerate(state["rho_y"]):
            base = species_integrals_initial[idx]
            species_rels.append(abs(integral(values, dx) - base) / max(abs(base), 1.0e-30))
        max_species_integral_rel = max(max_species_integral_rel, max(species_rels))
        max_closure_abs = max(max_closure_abs, finite_max(state["closure_abs"]))
        max_closure_rel = max(max_closure_rel, finite_max(state["closure_rel"]))
        min_rho_y = min(min_rho_y, finite_min(value for values in state["rho_y"] for value in values))
        min_y = min(min_y, finite_min(value for values in state["y_species"] for value in values))
        min_pressure = min(min_pressure, finite_min(state["pressure"]))
        min_temperature = min(min_temperature, finite_min(state["temperature"]))
        max_p2p3_rel = max(max_p2p3_rel, finite_max(state["p2p3_rel"]))
        all_finite = all_finite and finite_all(value for values in state["rho_y"] for value in values)
        all_finite = all_finite and finite_all(state["pressure"]) and finite_all(state["temperature"])
        all_finite = all_finite and finite_all(state["energy"])
        max_velocity = max(max_velocity, finite_max(abs(value) for value in state["velocity"]))
        energy_integral = integral(state["energy"], dx)
        energy_integral_rel = abs(energy_integral - energy_integral_initial)/max(abs(energy_integral_initial), 1.0e-30)
        max_energy_integral_rel = max(max_energy_integral_rel, energy_integral_rel)
        if step == steps[-1]:
            final_energy_integral_rel = energy_integral_rel
        rows.append(
            {
                "step": step,
                "max_abs_closure": f"{finite_max(state['closure_abs']):.17e}",
                "max_rel_closure": f"{finite_max(state['closure_rel']):.17e}",
                "min_rhoY": f"{finite_min(value for values in state['rho_y'] for value in values):.17e}",
                "min_Y": f"{finite_min(value for values in state['y_species'] for value in values):.17e}",
                "T_min_K": f"{finite_min(state['temperature']):.17e}",
                "T_max_K": f"{finite_max(state['temperature']):.17e}",
                "p_min_Pa": f"{finite_min(state['pressure']):.17e}",
                "p_max_Pa": f"{finite_max(state['pressure']):.17e}",
                "max_p2p3_rel": f"{finite_max(state['p2p3_rel']):.17e}",
                "max_velocity": f"{finite_max(abs(value) for value in state['velocity']):.17e}",
                "energy_integral": f"{energy_integral:.17e}",
                "energy_integral_rel_err": f"{energy_integral_rel:.17e}",
            }
        )

    tracked: Dict[str, Dict[str, float]] = {}
    for name in (SK54_FUEL.upper(), "O2", "N2"):
        idx = species_indices[name]
        y0 = first["y_species"][idx]
        yf = final["y_species"][idx]
        rho_y0 = first["rho_y"][idx]
        rho_yf = final["rho_y"][idx]
        tracked[name] = {
            "initial_total": integral(rho_y0, dx),
            "final_total": integral(rho_yf, dx),
            "initial_left": masked_integral(rho_y0, left_mask, dx),
            "final_left": masked_integral(rho_yf, left_mask, dx),
            "initial_right": masked_integral(rho_y0, right_mask, dx),
            "final_right": masked_integral(rho_yf, right_mask, dx),
            "initial_tv": total_variation_periodic(y0),
            "final_tv": total_variation_periodic(yf),
            "initial_grad_l2": l2_gradient_periodic(y0, dx),
            "final_grad_l2": l2_gradient_periodic(yf, dx),
            "max_abs_profile_change": max(abs(a - b) for a, b in zip(y0, yf)),
            "max_abs_initial_profile": max(abs(value) for value in y0),
        }

    return {
        "run_dir": str(run_dir),
        "steps": steps,
        "species_by_var": species_by_var,
        "rows": rows,
        "max_species_integral_rel": max_species_integral_rel,
        "max_closure_abs": max_closure_abs,
        "max_closure_rel": max_closure_rel,
        "min_rho_y": min_rho_y,
        "min_y": min_y,
        "min_pressure": min_pressure,
        "min_temperature": min_temperature,
        "initial_pressure_min": finite_min(first["pressure"]),
        "initial_pressure_max": finite_max(first["pressure"]),
        "final_pressure_min": finite_min(final["pressure"]),
        "final_pressure_max": finite_max(final["pressure"]),
        "initial_temperature_min": finite_min(first["temperature"]),
        "initial_temperature_max": finite_max(first["temperature"]),
        "final_temperature_min": finite_min(final["temperature"]),
        "final_temperature_max": finite_max(final["temperature"]),
        "initial_energy_integral": energy_integral_initial,
        "final_energy_integral": integral(final["energy"], dx),
        "final_energy_integral_rel": final_energy_integral_rel,
        "max_energy_integral_rel": max_energy_integral_rel,
        "max_p2p3_rel": max_p2p3_rel,
        "all_finite": all_finite,
        "max_velocity": max_velocity,
        "tracked": tracked,
    }


def passfail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    args = parse_args()
    off = analyze_run(args.off_run_dir.resolve(), args.ctfile, args.ctphase)
    on = analyze_run(args.on_run_dir.resolve(), args.ctfile, args.ctphase)

    fuel = SK54_FUEL.upper()
    off_fuel = off["tracked"][fuel]
    on_fuel = on["tracked"][fuel]
    off_o2 = off["tracked"]["O2"]
    on_o2 = on["tracked"]["O2"]
    off_n2 = off["tracked"]["N2"]

    off_stationarity: Dict[str, Dict[str, float | bool]] = {}
    for name in (fuel, "O2", "N2"):
        off_item = off["tracked"][name]
        on_item = on["tracked"][name]
        normalized_change = off_item["max_abs_profile_change"]/max(off_item["max_abs_initial_profile"], args.small_safe_scale)
        off_to_on_signal_ratio = off_item["max_abs_profile_change"]/max(on_item["max_abs_profile_change"], args.small_safe_scale)
        off_stationarity[name] = {
            "raw_abs_change": off_item["max_abs_profile_change"],
            "normalized_change": normalized_change,
            "off_to_on_signal_ratio": off_to_on_signal_ratio,
            "pass": normalized_change <= args.off_normalized_change_tol
            and off_to_on_signal_ratio <= args.off_to_on_signal_ratio_tol,
        }
    off_profile_static = all(bool(item["pass"]) for item in off_stationarity.values())
    fuel_transfers_right = (
        on_fuel["final_left"] < on_fuel["initial_left"] - args.diffusion_signal_tol
        and on_fuel["final_right"] > on_fuel["initial_right"] + args.diffusion_signal_tol
    )
    o2_transfers_left = (
        on_o2["final_left"] > on_o2["initial_left"] + args.diffusion_signal_tol
        and on_o2["final_right"] < on_o2["initial_right"] - args.diffusion_signal_tol
    )
    fuel_broadens = on_fuel["final_grad_l2"] < on_fuel["initial_grad_l2"]
    o2_broadens = on_o2["final_grad_l2"] < on_o2["initial_grad_l2"]

    checks = {
        "off_all_outputs_finite": off["all_finite"],
        "on_all_outputs_finite": on["all_finite"],
        "off_species_conserved": off["max_species_integral_rel"] <= args.conservation_rel_tol,
        "on_species_conserved": on["max_species_integral_rel"] <= args.conservation_rel_tol,
        "off_species_gas_closure": off["max_closure_abs"] <= args.closure_abs_tol and off["max_closure_rel"] <= args.closure_rel_tol,
        "on_species_gas_closure": on["max_closure_abs"] <= args.closure_abs_tol and on["max_closure_rel"] <= args.closure_rel_tol,
        "off_species_nonnegative": off["min_rho_y"] >= args.negative_tol and off["min_y"] >= args.negative_tol,
        "on_species_nonnegative": on["min_rho_y"] >= args.negative_tol and on["min_y"] >= args.negative_tol,
        "off_pressure_temperature_positive": off["min_pressure"] > 0.0 and off["min_temperature"] > 0.0,
        "on_pressure_temperature_positive": on["min_pressure"] > 0.0 and on["min_temperature"] > 0.0,
        "off_total_energy_conserved": off["max_energy_integral_rel"] <= args.conservation_rel_tol,
        "on_total_energy_conserved": on["max_energy_integral_rel"] <= args.conservation_rel_tol,
        "off_p2_p3_consistent": off["max_p2p3_rel"] <= args.closure_rel_tol,
        "on_p2_p3_consistent": on["max_p2p3_rel"] <= args.closure_rel_tol,
        "off_profiles_static": off_profile_static,
        "diffusion_on_fuel_moves_high_to_low": fuel_transfers_right,
        "diffusion_on_o2_moves_high_to_low": o2_transfers_left,
        "diffusion_on_interface_broadens": fuel_broadens and o2_broadens,
    }
    verdict = all(checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, data in (("off", off), ("on", on)):
        with (args.output_dir / f"{label}_summary_by_step.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(data["rows"])
    inventory = {"off": off, "on": on, "checks": checks}
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# V10 Model-3 Species Diffusion",
        "",
        f"OFF run: `{args.off_run_dir.resolve()}`",
        f"ON run: `{args.on_run_dir.resolve()}`",
        "",
        "Periodic boundaries create two diffusion interfaces: the central patch interface and the wraparound boundary.",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {passfail(ok)} |" for name, ok in checks.items())
    lines.extend(
        [
            "",
            "| metric | OFF | ON |",
            "| --- | ---: | ---: |",
            f"| max species integral rel error | {off['max_species_integral_rel']:.6e} | {on['max_species_integral_rel']:.6e} |",
            f"| max abs species/gas closure | {off['max_closure_abs']:.6e} | {on['max_closure_abs']:.6e} |",
            f"| max rel species/gas closure | {off['max_closure_rel']:.6e} | {on['max_closure_rel']:.6e} |",
            f"| min rhoY | {off['min_rho_y']:.6e} | {on['min_rho_y']:.6e} |",
            f"| min Y | {off['min_y']:.6e} | {on['min_y']:.6e} |",
            f"| min pressure | {off['min_pressure']:.6e} | {on['min_pressure']:.6e} |",
            f"| initial pressure min/max | {off['initial_pressure_min']:.6e} / {off['initial_pressure_max']:.6e} | {on['initial_pressure_min']:.6e} / {on['initial_pressure_max']:.6e} |",
            f"| final pressure min/max | {off['final_pressure_min']:.6e} / {off['final_pressure_max']:.6e} | {on['final_pressure_min']:.6e} / {on['final_pressure_max']:.6e} |",
            f"| min temperature | {off['min_temperature']:.6e} | {on['min_temperature']:.6e} |",
            f"| initial temperature min/max | {off['initial_temperature_min']:.6e} / {off['initial_temperature_max']:.6e} | {on['initial_temperature_min']:.6e} / {on['initial_temperature_max']:.6e} |",
            f"| final temperature min/max | {off['final_temperature_min']:.6e} / {off['final_temperature_max']:.6e} | {on['final_temperature_min']:.6e} / {on['final_temperature_max']:.6e} |",
            f"| max total-energy integral rel error | {off['max_energy_integral_rel']:.6e} | {on['max_energy_integral_rel']:.6e} |",
            f"| final total-energy integral rel error | {off['final_energy_integral_rel']:.6e} | {on['final_energy_integral_rel']:.6e} |",
            f"| max p2/p3 rel mismatch | {off['max_p2p3_rel']:.6e} | {on['max_p2p3_rel']:.6e} |",
            f"| max velocity | {off['max_velocity']:.6e} | {on['max_velocity']:.6e} |",
            f"| fuel left mass change | {off_fuel['final_left'] - off_fuel['initial_left']:.6e} | {on_fuel['final_left'] - on_fuel['initial_left']:.6e} |",
            f"| fuel right mass change | {off_fuel['final_right'] - off_fuel['initial_right']:.6e} | {on_fuel['final_right'] - on_fuel['initial_right']:.6e} |",
            f"| O2 left mass change | {off_o2['final_left'] - off_o2['initial_left']:.6e} | {on_o2['final_left'] - on_o2['initial_left']:.6e} |",
            f"| O2 right mass change | {off_o2['final_right'] - off_o2['initial_right']:.6e} | {on_o2['final_right'] - on_o2['initial_right']:.6e} |",
            f"| max fuel profile change | {off_fuel['max_abs_profile_change']:.6e} | {on_fuel['max_abs_profile_change']:.6e} |",
            f"| max O2 profile change | {off_o2['max_abs_profile_change']:.6e} | {on_o2['max_abs_profile_change']:.6e} |",
            f"| max N2 profile change | {off_n2['max_abs_profile_change']:.6e} | {on['tracked']['N2']['max_abs_profile_change']:.6e} |",
            f"| fuel TV initial/final | {on_fuel['initial_tv']:.6e} | {on_fuel['final_tv']:.6e} |",
            f"| O2 TV initial/final | {on_o2['initial_tv']:.6e} | {on_o2['final_tv']:.6e} |",
            f"| fuel periodic gradient L2 initial/final | {on_fuel['initial_grad_l2']:.6e} | {on_fuel['final_grad_l2']:.6e} |",
            f"| O2 periodic gradient L2 initial/final | {on_o2['initial_grad_l2']:.6e} | {on_o2['final_grad_l2']:.6e} |",
            "",
            "| OFF stationarity metric | value | tolerance | result |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name in (fuel, "O2", "N2"):
        item = off_stationarity[name]
        lines.append(
            f"| {name} raw absolute change | {item['raw_abs_change']:.6e} | informational | - |"
        )
        lines.append(
            f"| {name} normalized OFF change | {item['normalized_change']:.6e} | "
            f"{args.off_normalized_change_tol:.6e} | {passfail(bool(item['normalized_change'] <= args.off_normalized_change_tol))} |"
        )
        lines.append(
            f"| {name} OFF/ON signal ratio | {item['off_to_on_signal_ratio']:.6e} | "
            f"{args.off_to_on_signal_ratio_tol:.6e} | "
            f"{passfail(bool(item['off_to_on_signal_ratio'] <= args.off_to_on_signal_ratio_tol))} |"
        )
    lines.extend(
        [
            "",
            "| tracked species | OFF initial total | OFF final total | ON initial total | ON final total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in (fuel, "O2", "N2"):
        off_item = off["tracked"][name]
        on_item = on["tracked"][name]
        lines.append(
            f"| {name} | {off_item['initial_total']:.6e} | {off_item['final_total']:.6e} | "
            f"{on_item['initial_total']:.6e} | {on_item['final_total']:.6e} |"
        )
    lines.extend(["", f"Overall V10 verdict: **{passfail(verdict)}**"])
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
