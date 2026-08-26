#!/usr/bin/env python3
"""Analyze V8 reacting shock/droplet validation output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve()
VALIDATION_DIR = HERE.parents[1]
V7_DIR = VALIDATION_DIR / "V7_single_droplet_phase_change"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))
if str(V7_DIR) not in sys.path:
    sys.path.insert(0, str(V7_DIR))

import analyze_v7 as v7  # noqa: E402


KEY_SPECIES = ("NC12H26", "O2", "OH", "H2O", "CO2", "CO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ctfile", default=v7.SK54_CTFILE)
    parser.add_argument("--ctphase", default=v7.SK54_CTPHASE)
    parser.add_argument("--fuel-species-id", type=int, default=v7.DEFAULT_FUEL_SPECIES_ID)
    parser.add_argument("--closure-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--closure-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--negative-tol", type=float, default=-1.0e-12)
    parser.add_argument("--p2-p3-rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--p2-p3-abs-tol", type=float, default=1.0e-1)
    parser.add_argument("--evolution-tol", type=float, default=1.0e-8)
    parser.add_argument("--chem-activity-tol", type=float, default=1.0e-12)
    return parser.parse_args()


def species_lookup(species_by_var: Dict[int, str]) -> Dict[str, int]:
    return {name.upper(): var for var, name in species_by_var.items()}


def species_integral(state: Dict[str, object], species_by_var: Dict[int, str], name: str) -> float:
    var = species_lookup(species_by_var).get(name.upper())
    if var is None:
        return math.nan
    offset = sorted(species_by_var).index(var)
    return v7.fsum(state["rho_y"][offset]) * state["measure"]


def species_profile(state: Dict[str, object], species_by_var: Dict[int, str], name: str) -> Optional[Sequence[float]]:
    var = species_lookup(species_by_var).get(name.upper())
    if var is None:
        return None
    offset = sorted(species_by_var).index(var)
    return state["rho_y"][offset]


def species_max_location(state: Dict[str, object], species_by_var: Dict[int, str], name: str) -> Tuple[float, Optional[int], Optional[Tuple[float, float]]]:
    values = species_profile(state, species_by_var, name)
    if not values:
        return math.nan, None, None
    cell = max(range(len(values)), key=lambda idx: values[idx])
    return values[cell], cell, state["coords"][cell]


def finite_values(values: Sequence[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def min_finite(values: Sequence[float]) -> float:
    finite = finite_values(values)
    return min(finite) if finite else math.nan


def max_finite(values: Sequence[float]) -> float:
    finite = finite_values(values)
    return max(finite) if finite else math.nan


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "V8_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    species_by_var, molecular_weights = v7.load_species_metadata(run_dir, args.ctfile, args.ctphase)
    if not species_by_var:
        raise RuntimeError("Could not determine chemistry species variables from indices.dat or Cantera/YAML")

    fluid_params = v7.parse_fluid_parameters(run_dir)
    steps, missing = v7.complete_steps(run_dir, species_by_var)
    if len(steps) < 2:
        lines = [
            "# V8 Reacting Shock/Droplet Analysis",
            "",
            f"Run directory: `{run_dir}`",
            f"Complete saved steps available: {steps}",
            f"Incomplete steps: {missing}",
            "",
            "Overall V8 verdict: **BLOCKED** until at least two complete saves exist.",
        ]
        (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 2

    states = [v7.load_step(run_dir, step, species_by_var, molecular_weights, fluid_params) for step in steps]
    first = states[0]
    final = states[-1]
    species_vars = sorted(species_by_var)
    fuel_var = next((var for var in species_vars if var - min(species_vars) + 1 == args.fuel_species_id), None)
    if fuel_var is None:
        raise RuntimeError(f"fuel species id {args.fuel_species_id} is outside parsed species range")
    fuel_name = species_by_var[fuel_var]
    fuel_offset = species_vars.index(fuel_var)

    ml0 = v7.fsum(first["arho"][0]) * first["measure"]
    mv0 = v7.fsum(first["arho"][1]) * first["measure"]
    mf0 = species_integral(first, species_by_var, fuel_name)
    mo20 = species_integral(first, species_by_var, "O2")
    initial_global_tmax = max_finite(first["temperature"])

    rows: List[Dict[str, object]] = []
    max_abs_closure = 0.0
    max_rel_closure = 0.0
    min_rho_y = math.inf
    min_y = math.inf
    min_species = ""
    min_where = (None, None)
    min_p = math.inf
    min_t = math.inf
    max_t = -math.inf
    max_p2p3_rel = 0.0
    max_p2p3_abs = 0.0
    max_p2p3_where = (None, None)
    max_product_locations = {}

    key_initial = {name: species_integral(first, species_by_var, name) for name in KEY_SPECIES}
    key_max_values = {name: -math.inf for name in KEY_SPECIES}

    for state in states:
        step = int(state["step"])
        measure = float(state["measure"])
        ml = v7.fsum(state["arho"][0]) * measure
        mv = v7.fsum(state["arho"][1]) * measure
        mf = species_integral(state, species_by_var, fuel_name)
        mo2 = species_integral(state, species_by_var, "O2")
        gas = v7.fsum(state["gas_mass"]) * measure
        max_abs = max(state["closure_abs"])
        max_rel = max(state["closure_rel"])
        max_abs_closure = max(max_abs_closure, max_abs)
        max_rel_closure = max(max_rel_closure, max_rel)

        if state["min_rho_y"] < min_rho_y:
            min_rho_y = state["min_rho_y"]
            min_y = state["min_y"]
            min_species = state["min_species"]
            min_where = (step, state["min_cell"])

        state_min_p = min_finite([p for pair in zip(state["p2"], state["p3"]) for p in pair])
        state_min_t = min_finite(state["temperature"])
        state_max_t = max_finite(state["temperature"])
        if math.isfinite(state_min_p):
            min_p = min(min_p, state_min_p)
        if math.isfinite(state_min_t):
            min_t = min(min_t, state_min_t)
        if math.isfinite(state_max_t):
            max_t = max(max_t, state_max_t)

        for cell, (p2, p3) in enumerate(zip(state["p2"], state["p3"])):
            if math.isfinite(p2) and math.isfinite(p3):
                abs_err = abs(p2 - p3)
                rel_err = abs_err / max(abs(0.5 * (p2 + p3)), 1.0)
                if rel_err > max_p2p3_rel:
                    max_p2p3_rel = rel_err
                    max_p2p3_abs = abs_err
                    max_p2p3_where = (step, cell)

        row: Dict[str, object] = {
            "step": step,
            "M_liquid": f"{ml:.17e}",
            "M_vapor_fluid": f"{mv:.17e}",
            f"M_{fuel_name}": f"{mf:.17e}",
            "M_O2": f"{mo2:.17e}",
            "M_gas_stored": f"{gas:.17e}",
            "delta_M_liquid": f"{ml - ml0:.17e}",
            "delta_M_vapor_fluid": f"{mv - mv0:.17e}",
            f"delta_M_{fuel_name}": f"{mf - mf0:.17e}",
            "global_delta_M_O2": f"{mo2 - mo20:.17e}",
            "fuel_consumption_vs_nonreacting_delta": f"{(mv - mv0) - (mf - mf0):.17e}",
            "max_abs_closure_error": f"{max_abs:.17e}",
            "max_rel_closure_error": f"{max_rel:.17e}",
            "min_rhoY": f"{state['min_rho_y']:.17e}",
            "min_Y": f"{state['min_y']:.17e}",
            "T_min_K": f"{state_min_t:.17e}",
            "T_max_K": f"{state_max_t:.17e}",
            "p_min_Pa": f"{state_min_p:.17e}",
        }
        for name in KEY_SPECIES:
            integral = species_integral(state, species_by_var, name)
            row[f"M_{name}"] = f"{integral:.17e}" if math.isfinite(integral) else "nan"
            max_value, cell, coord = species_max_location(state, species_by_var, name)
            if math.isfinite(max_value) and max_value > key_max_values[name]:
                key_max_values[name] = max_value
                max_product_locations[name] = (step, cell, coord)
        rows.append(row)

    with (output_dir / "v8_time_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_ml = v7.fsum(final["arho"][0]) * final["measure"]
    final_mv = v7.fsum(final["arho"][1]) * final["measure"]
    final_mf = species_integral(final, species_by_var, fuel_name)
    final_mo2 = species_integral(final, species_by_var, "O2")
    dml = final_ml - ml0
    dmv = final_mv - mv0
    dmf = final_mf - mf0
    dmo2 = final_mo2 - mo20

    product_increases = {}
    for name in ("OH", "H2O", "CO2", "CO"):
        initial = key_initial.get(name, math.nan)
        final_value = species_integral(final, species_by_var, name)
        product_increases[name] = final_value - initial if math.isfinite(initial) and math.isfinite(final_value) else math.nan

    fuel_consumption_indicator = dmv - dmf
    chemistry_activity_sources = [fuel_consumption_indicator, *product_increases.values()]
    chemistry_active = any(math.isfinite(value) and value > args.chem_activity_tol for value in chemistry_activity_sources)
    phase_change_active = dmv > args.chem_activity_tol and -dml > args.chem_activity_tol
    delta_global_tmax = max_t - initial_global_tmax

    profile_changes = {
        "liquid_arho": v7.max_abs_delta(final["arho"][0], first["arho"][0]),
        "vapor_arho": v7.max_abs_delta(final["arho"][1], first["arho"][1]),
        "fuel_rhoY": v7.max_abs_delta(final["rho_y"][fuel_offset], first["rho_y"][fuel_offset]),
        "pressure_p3": v7.max_abs_delta(final["p3"], first["p3"]),
        "mom_x": v7.max_abs_delta(final["mom"][0], first["mom"][0]),
        "mom_y": v7.max_abs_delta(final["mom"][1], first["mom"][1]),
    }
    shock_drop_evolved = max(value for value in profile_changes.values() if math.isfinite(value)) > args.evolution_tol

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
        ("phase_change_active", phase_change_active, f"dMv={dmv:.6e}, -dMl={-dml:.6e}", f"> {args.chem_activity_tol:.1e}"),
        ("chemistry_active", chemistry_active, f"fuel_deficit={fuel_consumption_indicator:.6e}", f"> {args.chem_activity_tol:.1e} or product growth"),
    ]
    overall = all(item[1] for item in checks)

    metrics = {
        "run_dir": str(run_dir),
        "steps": steps,
        "missing_incomplete_steps": {str(step): values for step, values in missing.items()},
        "fuel_species": fuel_name,
        "final_deltas": {
            "delta_M_liquid": dml,
            "delta_M_vapor_fluid": dmv,
            f"delta_M_{fuel_name}": dmf,
            "global_delta_M_O2": dmo2,
            "fuel_consumption_vs_nonreacting_delta": fuel_consumption_indicator,
            "product_increases": product_increases,
        },
        "max_species_values": {name: {"rhoY": key_max_values[name], "where": max_product_locations.get(name)} for name in KEY_SPECIES},
        "profile_changes": profile_changes,
        "temperature": {
            "initial_global_Tmax_K": initial_global_tmax,
            "maximum_T_over_run_K": max_t,
            "delta_global_Tmax_K": delta_global_tmax,
            "minimum_T_over_run_K": min_t,
        },
        "p2_p3": {"max_rel": max_p2p3_rel, "max_abs_Pa": max_p2p3_abs, "where": max_p2p3_where},
        "checks": {name: status(ok) for name, ok, _, _ in checks},
        "overall": status(overall),
    }
    (output_dir / "v8_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# V8 Reacting Shock/Droplet Analysis",
        "",
        f"Run directory: `{run_dir}`",
        f"Complete saved steps analyzed: {steps}",
        "",
        "V8 validates active reacting coupling in the shock/droplet calculation. It does not by itself classify sustained ignition.",
        "",
        "| Component | Value | Criterion | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(f"| {name} | {value} | {criterion} | {status(ok)} |" for name, ok, value, criterion in checks)
    lines.extend(
        [
            "",
            "## Reacting Indicators",
            "",
            f"- Delta liquid mass: `{dml:.12e}`",
            f"- Delta vapor-fluid mass: `{dmv:.12e}`",
            f"- Delta {fuel_name} mass: `{dmf:.12e}`",
            f"- Global domain delta O2 mass: `{dmo2:.12e}` (non-periodic boundary contaminated; informational only)",
            f"- Fuel consumption indicator `(Delta vapor - Delta fuel)`: `{fuel_consumption_indicator:.12e}`",
            f"- Initial global Tmax: `{initial_global_tmax:.12e} K`",
            f"- Maximum T over run: `{max_t:.12e} K`",
            f"- Delta global Tmax: `{delta_global_tmax:.12e} K` (descriptive; not used alone as chemistry evidence)",
            "",
            "## Product/Radical Growth",
            "",
        ]
    )
    for name, delta in product_increases.items():
        step, cell, coord = max_product_locations.get(name, (None, None, None))
        lines.append(f"- {name}: delta integral `{delta:.12e}`, max rhoY `{key_max_values[name]:.12e}` at step `{step}`, cell `{cell}`, coord `{coord}`")
    lines.extend(["", "## Shock/Drop Evolution Evidence", ""])
    lines.extend(f"- {name}: `{value:.12e}`" for name, value in profile_changes.items())
    lines.extend(["", f"Overall V8 verdict: **{status(overall)}**"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
