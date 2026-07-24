#!/usr/bin/env python3
"""Audit species admissibility directly from raw shock-droplet output.

The analyzer reads one saved state at a time from ``D/`` and/or ``p_all/``.
It never clips, normalizes, repairs, or writes simulation data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics


BASE_FIELDS = (
    "pressure",
    "liquid_alpha",
    "vapor_alpha",
    "air_alpha",
    "vapor_alpha_rho",
    "air_alpha_rho",
)
MASK_STATE_FIELDS = tuple(name for name in BASE_FIELDS if name != "pressure")
SCOPES = ("all_gas_mass", "valid_gas", "interface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gas-mass-floor", type=float, default=1.0e-8)
    parser.add_argument("--bounds-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "toolchain" / "main.py").is_file():
            return candidate
    raise RuntimeError(f"could not locate repository root above {start}")


def chemistry_configuration(run_dir: Path) -> tuple[str, str]:
    text = (run_dir / "case.py").read_text(errors="replace")
    file_match = re.search(r'^CTFILE\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    phase_match = re.search(r'^CTPHASE\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not file_match or not phase_match:
        raise RuntimeError("could not find CTFILE and CTPHASE in run case.py")
    return file_match.group(1), phase_match.group(1)


def mechanism_species(run_dir: Path) -> tuple[list[str], Path, str]:
    mechanism_name, phase_name = chemistry_configuration(run_dir)
    mechanism_path = repository_root(Path(__file__).resolve()) / mechanism_name
    document = yaml.safe_load(mechanism_path.read_text())
    for phase in document.get("phases", []):
        if phase.get("name") == phase_name:
            species = phase.get("species", [])
            if not species or not all(isinstance(name, str) for name in species):
                raise RuntimeError(f"phase {phase_name!r} has no explicit species order")
            return list(species), mechanism_path, phase_name
    raise RuntimeError(f"phase {phase_name!r} not found in {mechanism_path}")


def read_cons_field(run_dir: Path, cons_index: int, step: int) -> dict:
    """Use the base analyzer's raw-file discovery and loading conventions."""
    paths = raw.field_files(run_dir, "cons", cons_index, step).get(step, [])
    values: dict[tuple[float, float], float] = {}
    duplicates = 0
    rows = 0
    for path in paths:
        coords, vals = raw.read_raw_values(path)
        rows += int(vals.size)
        for coord, value in zip(coords, vals):
            key = (float(coord[0]), float(coord[1]))
            if key in values:
                duplicates += 1
                continue
            values[key] = float(value)
    return {
        "available": bool(values),
        "values": values,
        "paths": paths,
        "rows": rows,
        "duplicates": duplicates,
    }


def finite_extrema(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    return (
        float(np.min(finite)) if finite.size else math.nan,
        float(np.max(finite)) if finite.size else math.nan,
        int(values.size - finite.size),
    )


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: csv_value(row.get(key, "")) for key in columns} for row in rows)


def json_number(value: float | int) -> float | int | None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def state_fields(run_dir: Path, step: int) -> dict[str, dict]:
    return {name: raw.read_field(run_dir, name, step) for name in BASE_FIELDS}


def scope_masks(keys: list[tuple[float, float]], fields: dict, floor: float) -> tuple[dict[str, np.ndarray], dict]:
    context = gas_metrics.mask_context(fields, floor)
    all_gas = {
        key for key in keys
        if math.isfinite(context["gas_mass"].get(key, math.nan)) and context["gas_mass"].get(key, math.nan) > floor
    }
    masks = {
        "all_gas_mass": np.array([key in all_gas for key in keys], dtype=bool),
        "valid_gas": np.array([key in context["valid_gas_thermo"] for key in keys], dtype=bool),
        # Species fractions are defined only where the chemistry gas mass exceeds the floor.
        "interface": np.array([key in context["interface"] and key in all_gas for key in keys], dtype=bool),
    }
    return masks, context


def violation_severity(rhoy: float, y: float, tolerance: float) -> tuple[str | None, float]:
    if not math.isfinite(rhoy):
        return "rhoY_nonfinite", math.inf
    if not math.isfinite(y):
        return "Y_nonfinite", math.inf
    candidates = (
        ("rhoY_negative", max(-rhoy - tolerance, 0.0)),
        ("Y_negative", max(-y - tolerance, 0.0)),
        ("Y_above_one", max(y - 1.0 - tolerance, 0.0)),
    )
    return max(candidates, key=lambda item: item[1]) if any(value > 0.0 for _, value in candidates) else (None, 0.0)


def first_violation_record(
    save_index: int,
    step: int,
    time_s: float,
    keys: list[tuple[float, float]],
    fields: dict,
    context: dict,
    species: list[str],
    rhoy_all: np.ndarray,
    y_all: np.ndarray,
    sum_y: np.ndarray,
    all_gas_mask: np.ndarray,
    tolerance: float,
) -> dict | None:
    candidate: tuple[int, int, str, float] | None = None
    for cell_index in np.flatnonzero(all_gas_mask):
        for species_index in range(len(species)):
            kind, severity = violation_severity(
                float(rhoy_all[species_index, cell_index]),
                float(y_all[species_index, cell_index]),
                tolerance,
            )
            if kind is not None:
                candidate = (int(cell_index), species_index, kind, severity)
                break
        if candidate is not None:
            break
    if candidate is None:
        return None

    cell_index, species_index, kind, severity = candidate
    key = keys[cell_index]
    xs = sorted({item[0] for item in keys})
    ys = sorted({item[1] for item in keys})
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}
    valid_temperature = raw.reconstruct_temperature(fields, gas_mass_threshold=None)
    valid_gas = key in context["valid_gas_thermo"]
    temperature = valid_temperature["values"].get(key, math.nan) if valid_gas else math.nan
    selected_name = species[species_index]
    selected_rhoy = rhoy_all[species_index, cell_index]
    selected_y = y_all[species_index, cell_index]
    record = {
        "save_index": save_index,
        "step": step,
        "time_s": json_number(time_s),
        "violation_type": kind,
        "violation_severity": json_number(severity),
        "species": selected_name,
        "species_index_zero_based": species_index,
        "cell_indices": {"i": x_to_i[key[0]], "j": y_to_j[key[1]], "k": 0},
        "coordinates_m": {"x": key[0], "y": key[1]},
        "rhoY_i": json_number(selected_rhoy),
        "Y_i": json_number(selected_y),
        "sum_Y": json_number(sum_y[cell_index]),
        "gas_mass": json_number(context["gas_mass"].get(key, math.nan)),
        "liquid_alpha": json_number(fields["liquid_alpha"]["values"].get(key, math.nan)),
        "vapor_alpha": json_number(fields["vapor_alpha"]["values"].get(key, math.nan)),
        "air_alpha": json_number(fields["air_alpha"]["values"].get(key, math.nan)),
        "pressure": json_number(fields["pressure"]["values"].get(key, math.nan)),
        "valid_gas_temperature": json_number(temperature),
        "is_valid_gas_cell": valid_gas,
        "is_interface_cell": key in context["interface"],
        "all_species": {
            name: {"rhoY": json_number(rhoy_all[index, cell_index]), "Y": json_number(y_all[index, cell_index])}
            for index, name in enumerate(species)
        },
    }
    return record


def analyze_state(
    run_dir: Path,
    step: int,
    save_index: int,
    time_s: float,
    time_source: str,
    species: list[str],
    gas_mass_floor: float,
    tolerance: float,
) -> tuple[list[dict], list[dict], dict | None]:
    fields = state_fields(run_dir, step)
    if not all(fields[name]["available"] for name in MASK_STATE_FIELDS):
        missing = [name for name in MASK_STATE_FIELDS if not fields[name]["available"]]
        raise RuntimeError(f"save_index={save_index}: missing base fields: {missing}")

    common = set(fields[MASK_STATE_FIELDS[0]]["values"])
    for name in MASK_STATE_FIELDS[1:]:
        common &= set(fields[name]["values"])
    keys = sorted(common, key=lambda key: (key[1], key[0]))
    if not keys:
        raise RuntimeError(f"save_index={save_index}: no coordinate-matched base cells")
    pressure_available = fields["pressure"]["available"]
    if not pressure_available:
        # Initial raw output may omit primitive fields. Pressure is not part of
        # the gas-aware mask, so retain all cells and explicitly represent the
        # unavailable pressure/temperature context as nonfinite.
        fields["pressure"] = {
            "available": True,
            "values": {key: math.nan for key in keys},
            "stats": raw.missing_stats([], 0),
        }
    masks, context = scope_masks(keys, fields, gas_mass_floor)
    gas_mass = np.array([context["gas_mass"].get(key, math.nan) for key in keys], dtype=float)

    rhoy_all = np.full((len(species), len(keys)), np.nan, dtype=float)
    y_all = np.full_like(rhoy_all, np.nan)
    by_species_rows: list[dict] = []
    for species_index, species_name in enumerate(species):
        field = read_cons_field(run_dir, 13 + species_index, step)
        if not field["available"]:
            raise RuntimeError(f"save_index={save_index}: missing cons.{13 + species_index} for {species_name}")
        rhoy = np.array([field["values"].get(key, math.nan) for key in keys], dtype=float)
        y = np.full(len(keys), np.nan, dtype=float)
        eligible = masks["all_gas_mass"]
        y[eligible] = rhoy[eligible] / gas_mass[eligible]
        rhoy_all[species_index] = rhoy
        y_all[species_index] = y

        for scope in SCOPES:
            mask = masks[scope]
            rho_values = rhoy[mask]
            y_values = y[mask]
            rho_min, rho_max, rho_nonfinite = finite_extrema(rho_values)
            y_min, y_max, y_nonfinite = finite_extrema(y_values)
            by_species_rows.append({
                "save_index": save_index,
                "step": step,
                "time_s": time_s,
                "time_source": time_source,
                "scope": scope,
                "species_index": species_index + 1,
                "species": species_name,
                "cell_count": int(np.count_nonzero(mask)),
                "rhoY_min": rho_min,
                "rhoY_max": rho_max,
                "Y_min": y_min,
                "Y_max": y_max,
                "rhoY_below_negative_tolerance_count": int(np.count_nonzero(rho_values < -tolerance)),
                "Y_below_negative_tolerance_count": int(np.count_nonzero(y_values < -tolerance)),
                "Y_above_one_plus_tolerance_count": int(np.count_nonzero(y_values > 1.0 + tolerance)),
                "rhoY_nonfinite_count": rho_nonfinite,
                "Y_nonfinite_count": y_nonfinite,
            })

    sum_y = np.sum(y_all, axis=0)
    sum_y[np.any(~np.isfinite(y_all), axis=0)] = np.nan
    timeseries_rows: list[dict] = []
    for scope in SCOPES:
        mask = masks[scope]
        scope_sum = sum_y[mask]
        sum_min, sum_max, sum_nonfinite = finite_extrema(scope_sum)
        finite_sum = scope_sum[np.isfinite(scope_sum)]
        abs_error = np.abs(finite_sum - 1.0)
        scope_species = [row for row in by_species_rows if row["scope"] == scope]
        timeseries_rows.append({
            "save_index": save_index,
            "step": step,
            "time_s": time_s,
            "time_source": time_source,
            "scope": scope,
            "gas_mass_floor": gas_mass_floor,
            "bounds_tolerance": tolerance,
            "pressure_available": pressure_available,
            "cell_count": int(np.count_nonzero(mask)),
            "species_count": len(species),
            "sum_Y_min": sum_min,
            "sum_Y_max": sum_max,
            "sum_Y_max_abs_error": float(np.max(abs_error)) if abs_error.size else math.nan,
            "species_sum_violation_count": int(np.count_nonzero(abs_error > tolerance)),
            "sum_Y_nonfinite_count": sum_nonfinite,
            "rhoY_negative_count_all_species": sum(row["rhoY_below_negative_tolerance_count"] for row in scope_species),
            "Y_negative_count_all_species": sum(row["Y_below_negative_tolerance_count"] for row in scope_species),
            "Y_above_one_count_all_species": sum(row["Y_above_one_plus_tolerance_count"] for row in scope_species),
            "rhoY_nonfinite_count_all_species": sum(row["rhoY_nonfinite_count"] for row in scope_species),
            "Y_nonfinite_count_all_species": sum(row["Y_nonfinite_count"] for row in scope_species),
        })

    first = first_violation_record(
        save_index, step, time_s, keys, fields, context, species,
        rhoy_all, y_all, sum_y, masks["all_gas_mass"], tolerance,
    )
    return timeseries_rows, by_species_rows, first


def worst_species(rows: list[dict], field: str, mode: str) -> dict | None:
    finite_rows = [row for row in rows if math.isfinite(float(row[field]))]
    if not finite_rows:
        return None
    row = min(finite_rows, key=lambda item: item[field]) if mode == "min" else max(finite_rows, key=lambda item: item[field])
    return {key: row[key] for key in ("save_index", "time_s", "scope", "species", field)}


def write_summary(
    path: Path,
    run_dir: Path,
    out_dir: Path,
    species: list[str],
    mechanism_path: Path,
    phase_name: str,
    gas_mass_floor: float,
    tolerance: float,
    time_rows: list[dict],
    species_rows: list[dict],
    first: dict | None,
) -> None:
    all_rows = [row for row in species_rows if row["scope"] == "all_gas_mass"]
    negative_rhoy = sum(row["rhoY_below_negative_tolerance_count"] for row in all_rows)
    negative_y = sum(row["Y_below_negative_tolerance_count"] for row in all_rows)
    above_one = sum(row["Y_above_one_plus_tolerance_count"] for row in all_rows)
    all_time = [row for row in time_rows if row["scope"] == "all_gas_mass"]
    sum_violations = sum(row["species_sum_violation_count"] for row in all_time)
    max_sum_error = max((row["sum_Y_max_abs_error"] for row in all_time if math.isfinite(row["sum_Y_max_abs_error"])), default=math.nan)
    scope_lines: list[str] = []
    for scope in SCOPES:
        rows = [row for row in time_rows if row["scope"] == scope]
        finite_errors = [row["sum_Y_max_abs_error"] for row in rows if math.isfinite(row["sum_Y_max_abs_error"])]
        first_sum = next((row for row in rows if row["species_sum_violation_count"] > 0), None)
        scope_lines.extend((
            f"{scope}_rhoY_negative_occurrences: {sum(row['rhoY_negative_count_all_species'] for row in rows)}",
            f"{scope}_Y_negative_occurrences: {sum(row['Y_negative_count_all_species'] for row in rows)}",
            f"{scope}_Y_above_one_occurrences: {sum(row['Y_above_one_count_all_species'] for row in rows)}",
            f"{scope}_species_sum_violating_cells_over_saved_states: {sum(row['species_sum_violation_count'] for row in rows)}",
            f"{scope}_maximum_abs_sum_Y_error: {max(finite_errors, default=math.nan):.16e}",
            f"{scope}_earliest_species_sum_violation_save_index: {first_sum['save_index'] if first_sum else 'none'}",
        ))
    lines = [
        "Shock-droplet SK54 species admissibility audit",
        f"run_dir: {run_dir}",
        f"out_dir: {out_dir}",
        f"mechanism: {mechanism_path}",
        f"phase: {phase_name}",
        f"saved_states: {len(all_time)}",
        f"species_count: {len(species)}",
        f"gas_mass_definition: vapor_alpha_rho + air_alpha_rho",
        f"gas_mass_floor: {gas_mass_floor:.16e}",
        f"bounds_tolerance: {tolerance:.16e}",
        "interface_scope: existing 0.01 < liquid_alpha < 0.99 mask, restricted to gas_mass above floor",
        f"rhoY_negative_occurrences_all_gas_mass: {negative_rhoy}",
        f"Y_negative_occurrences_all_gas_mass: {negative_y}",
        f"Y_above_one_occurrences_all_gas_mass: {above_one}",
        f"species_sum_violating_cells_over_saved_states_all_gas_mass: {sum_violations}",
        f"maximum_abs_sum_Y_error_all_gas_mass: {max_sum_error:.16e}",
        *scope_lines,
        f"earliest_individual_species_violation: {json.dumps(first, sort_keys=True) if first is not None else 'none'}",
        f"most_negative_rhoY: {json.dumps(worst_species(all_rows, 'rhoY_min', 'min'), sort_keys=True)}",
        f"most_negative_Y: {json.dumps(worst_species(all_rows, 'Y_min', 'min'), sort_keys=True)}",
        f"largest_Y: {json.dumps(worst_species(all_rows, 'Y_max', 'max'), sort_keys=True)}",
        "caveat: saved output locates the earliest saved violation only; it does not identify the exact RK-stage source pathway.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze(run_dir: Path, out_dir: Path, gas_mass_floor: float, tolerance: float) -> None:
    if gas_mass_floor < 0.0 or not math.isfinite(gas_mass_floor):
        raise ValueError("--gas-mass-floor must be finite and nonnegative")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("--bounds-tolerance must be finite and nonnegative")
    species, mechanism_path, phase_name = mechanism_species(run_dir)
    steps = raw.available_steps(run_dir)
    if not steps:
        raise RuntimeError(f"no saved states found in {run_dir}")
    times = raw.infer_times(run_dir, steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_rows: list[dict] = []
    species_rows: list[dict] = []
    first: dict | None = None
    for save_index, step in enumerate(steps):
        time_s, time_source = times[step]
        state_time, state_species, state_first = analyze_state(
            run_dir, step, save_index, time_s, time_source, species, gas_mass_floor, tolerance,
        )
        time_rows.extend(state_time)
        species_rows.extend(state_species)
        if first is None and state_first is not None:
            first = state_first
        print(f"save_index={save_index} step={step} time_s={time_s:.8e} complete", flush=True)

    write_csv(out_dir / "species_bounds_timeseries.csv", time_rows)
    write_csv(out_dir / "species_bounds_by_species.csv", species_rows)
    (out_dir / "species_first_violation.json").write_text(json.dumps(first, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_summary(
        out_dir / "species_bounds_summary.txt", run_dir, out_dir, species, mechanism_path,
        phase_name, gas_mass_floor, tolerance, time_rows, species_rows, first,
    )
    print(f"wrote species audit to {out_dir}")


def main() -> None:
    args = parse_args()
    analyze(args.run_dir.resolve(), args.out_dir.resolve(), args.gas_mass_floor, args.bounds_tolerance)


if __name__ == "__main__":
    main()
