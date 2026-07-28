#!/usr/bin/env python3
"""Compare chemistry/diffusion variants for the Mach-3 SK54 shock-droplet case.

This standalone driver intentionally uses the raw ASCII ``D/`` saves and the
same lightweight raw-field readers as the existing shock-droplet analyzers.  It
does not depend on the previously generated ON/OFF or combined-analysis output
files, though it follows their plotting conventions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-c3-on-off")

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import yaml


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RUN_ROOT = REPO / "runs/shockdroplet_air_sk54/alpha_source_fix"
OUT = SCRIPT.parent
T_SAVE = 5.0e-8
GAS_MASS_FLOOR = 1.0e-8
BOUNDS_TOL = 1.0e-12
REL_DENOM_EPS = 1.0e-300

SOURCE_ON = RUN_ROOT / "full_1us_evap_only_consistency_gates"
RESTART_ON = RUN_ROOT / "restart18_to_2us_evap_only_consistency_gates"
OFF_RUN = RUN_ROOT / "full_2us_evap_only_chemistry_OFF"
C3_RUN = RUN_ROOT / "C3_diffusion_ON_reactions_ON_2p0us"
FUEL_OUT = OUT / "fuel_inventory_budget"
C3_FUEL_OUTPUT = OUT / "c3_fuel_inventory.csv"
C3_FUEL_INVENTORY_CSV = Path("/mnt/e/Mitansh/Research/Thesis/Coupling/C3_vs_chemistry_ON_OFF_0_to_2us/c3_fuel_inventory.csv")
C3_COMPACT_EXPORT = Path("/mnt/e/Mitansh/Research/Thesis/Coupling/C3_vs_chemistry_ON_OFF_0_to_2us/c3_export")

sys.path.insert(0, str(REPO / "examples/2D_dodecane_global_reduced"))
import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics

# This comparison is explicitly over existing raw ASCII D/ saves.  Avoid p_all/
# traversal because the restart trees can be large and are not needed here.
raw.raw_dirs = lambda run_dir: [Path(run_dir) / "D"] if (Path(run_dir) / "D").is_dir() else []

CASE_ORDER = [
    "CHEM_OFF_DIFF_OFF",
    "CHEM_ON_DIFF_OFF",
    "CHEM_ON_DIFF_ON_C3",
]
PANEL_ORDER = CASE_ORDER
CASE_LABEL = {
    "CHEM_OFF_DIFF_OFF": "chemistry OFF / diffusion OFF",
    "CHEM_ON_DIFF_OFF": "chemistry ON / diffusion OFF",
    "CHEM_ON_DIFF_ON_C3": "C3 chemistry ON / diffusion ON",
}
CASE_SHORT = {
    "CHEM_OFF_DIFF_OFF": "OFF/diffOFF",
    "CHEM_ON_DIFF_OFF": "ON/diffOFF",
    "CHEM_ON_DIFF_ON_C3": "C3 ON/diffON",
}

SPECIES_OF_INTEREST = ["NC12H26", "O2", "OH", "HO2", "H2O2", "H2O", "CO", "CO2"]
FRAME_TIMES_US = [0.50, 0.85, 1.00, 1.50, 1.90, 1.95, 2.00]
FRAME_VARIABLES = [
    "pressure",
    "valid_gas_temperature",
    "liquid_alpha",
    "NC12H26",
    "O2",
    "OH",
    "HO2",
    "H2O2",
    "CO",
    "CO2",
    "H2O",
]
DIFFERENCE_FRAME_TIMES_US = [1.50, 1.90]
DIFFERENCE_FRAME_VARIABLES = ["NC12H26", "O2", "OH", "valid_gas_temperature", "liquid_alpha"]
HYDRO_WAKE_TIMES_US = [1.50, 1.90]
HYDRO_WAKE_VARIABLES = ["vorticity", "velocity_magnitude", "pressure", "density"]
HYDRO_REQUIRED_C3_NPZ = ["u_velocity", "v_velocity", "pressure", "density", "liquid_alpha"]
HYDRO_DIFFERENTIATION_METHOD = "numpy.gradient with second-order centered interior and one-sided boundary differences"
MIXING_STATE_TIMES_US = [1.50, 1.90]
MIXING_STATE_FIELDS = [
    "NC12H26",
    "O2",
    "OH",
    "HO2",
    "H2O2",
    "valid_gas_temperature",
    "liquid_alpha",
    "pressure",
]
MIXING_RADICALS = ["OH", "HO2", "H2O2"]
MIXING_PHI_O2_FLOOR = 1.0e-12
MIXING_PHI_PLOT_CAP = 100.0
MIXING_PHI_LAYER_LOG_HALF_WIDTH = 0.05
MIXING_BANDS = [
    ("very_lean", -math.inf, 0.5, "phi < 0.5"),
    ("combustible_lean", 0.5, 0.8, "0.5 <= phi < 0.8"),
    ("near_stoichiometric", 0.8, 1.2, "0.8 <= phi <= 1.2"),
    ("moderately_rich", 1.2, 2.0, "1.2 < phi <= 2.0"),
    ("very_rich", 2.0, math.inf, "phi > 2.0"),
]
FUEL_REQUIRED_FIELDS = ["liquid_alpha_rho", "rhoY_NC12H26"]
FUEL_CASE_ORDER = ["OFF_NO_DIFF", "ON_NO_DIFF", "ON_WITH_DIFF_C3"]
FUEL_CASE_LABEL = {
    "OFF_NO_DIFF": "chemistry OFF / diffusion OFF",
    "ON_NO_DIFF": "chemistry ON / diffusion OFF",
    "ON_WITH_DIFF_C3": "chemistry ON / diffusion ON C3",
}
FUEL_CASE_TO_COMPARISON = {
    "OFF_NO_DIFF": "CHEM_OFF_DIFF_OFF",
    "ON_NO_DIFF": "CHEM_ON_DIFF_OFF",
    "ON_WITH_DIFF_C3": "CHEM_ON_DIFF_ON_C3",
}
FUEL_CASE_DIFFUSION = {"OFF_NO_DIFF": "F", "ON_NO_DIFF": "F", "ON_WITH_DIFF_C3": "T"}
FUEL_CASE_REACTIONS = {"OFF_NO_DIFF": "F", "ON_NO_DIFF": "T", "ON_WITH_DIFF_C3": "T"}
FUEL_REPORT_TIMES_US = [1.50, 1.90]
BASE_FIELDS = [
    "liquid_alpha_rho",
    "vapor_alpha_rho",
    "air_alpha_rho",
    "liquid_alpha",
    "vapor_alpha",
    "air_alpha",
    "pressure",
]
METRIC_FIELDS = BASE_FIELDS + [f"rhoY_{name}" for name in SPECIES_OF_INTEREST]
TREND_GROUPS = [
    ("liquid_mass_loss_fraction", "Liquid mass-loss fraction", "liquid_mass_loss_fraction.png"),
    ("vapor_mass", r"Vapor mass [kg m$^{-1}$]", "vapor_mass.png"),
    ("liquid_alpha_area_m2", r"Liquid area [m$^2$]", "liquid_area.png"),
    ("liquid_extent_x_m", "Streamwise liquid extent [m]", "liquid_extent_x.png"),
    ("liquid_extent_y_m", "Cross-stream liquid extent [m]", "liquid_extent_y.png"),
    ("valid_gas_Tmax_K", "Valid-gas Tmax [K]", "valid_gas_Tmax.png"),
    ("valid_gas_Tmean_K", "Valid-gas mean temperature [K]", "valid_gas_Tmean.png"),
    ("integrated_NC12H26", r"Integrated NC12H26 [kg m$^{-1}$]", "integrated_NC12H26.png"),
    ("integrated_O2", r"Integrated O2 [kg m$^{-1}$]", "integrated_O2.png"),
    ("integrated_OH", r"Integrated OH [kg m$^{-1}$]", "integrated_OH.png"),
    ("integrated_HO2", r"Integrated HO2 [kg m$^{-1}$]", "integrated_HO2.png"),
    ("integrated_H2O2", r"Integrated H2O2 [kg m$^{-1}$]", "integrated_H2O2.png"),
    ("integrated_CO", r"Integrated CO [kg m$^{-1}$]", "integrated_CO.png"),
    ("integrated_CO2", r"Integrated CO2 [kg m$^{-1}$]", "integrated_CO2.png"),
    ("integrated_H2O", r"Integrated H2O [kg m$^{-1}$]", "integrated_H2O.png"),
    ("mass_consistency_relative_error", "Integrated species/gas mass relative error", "mass_consistency.png"),
]
DIFF_METRICS = [key for key, _label, _file in TREND_GROUPS] + [
    "liquid_mass",
    "liquid_mass_loss",
    "valid_gas_cell_count",
    "species_negative_rhoY_count",
    "species_negative_Y_count",
    "species_Y_above_one_count",
    "species_sumY_min",
    "species_sumY_max",
    "species_sumY_max_abs_error",
    "species_sumY_violation_count",
]
EXPORT_CASE_ORDER = [
    "chemistry_OFF_diffusion_OFF",
    "chemistry_ON_diffusion_OFF",
    "chemistry_ON_diffusion_ON_C3",
]
EXPORT_CASE_LABEL = {
    "chemistry_OFF_diffusion_OFF": "chemistry OFF / diffusion OFF",
    "chemistry_ON_diffusion_OFF": "chemistry ON / diffusion OFF",
    "chemistry_ON_diffusion_ON_C3": "C3 chemistry ON / diffusion ON",
}
EXPORT_C3_CASE = "chemistry_ON_diffusion_ON_C3"


@dataclass(frozen=True)
class SaveItem:
    case: str
    run_dir: Path
    raw_step: int
    global_save: int
    origin: str


def ff(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def species_names() -> list[str]:
    mech = REPO / "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
    data = yaml.safe_load(mech.read_text())
    for phase in data.get("phases", []):
        if phase.get("name") == "yao_sk54":
            return list(phase["species"])
    raise RuntimeError(f"could not find yao_sk54 phase in {mech}")


def ensure_species_fields(names: list[str]) -> None:
    for idx, name in enumerate(names, start=1):
        raw.FIELDS.setdefault(f"rhoY_{name}", ("cons", 12 + idx))


def available_d_steps(run_dir: Path) -> set[int]:
    ddir = run_dir / "D"
    if not ddir.is_dir():
        return set()
    steps: set[int] = set()
    for path in ddir.glob("*.*.*.*.dat"):
        info = raw.raw_file_info(path)
        if info is not None:
            steps.add(info[3])
    return steps


def field_paths(run_dir: Path, field_name: str, step: int) -> list[Path]:
    kind, index = raw.FIELDS[field_name]
    return raw.field_files(run_dir, kind, index, step).get(step, [])


def field_available(run_dir: Path, field_name: str, step: int) -> bool:
    paths = field_paths(run_dir, field_name, step)
    return bool(paths) and any(path.stat().st_size > 0 for path in paths)


def item_for(case: str, global_save: int) -> SaveItem:
    if case == "CHEM_ON_DIFF_OFF":
        if global_save <= 18:
            return SaveItem(case, SOURCE_ON, global_save, global_save, "source")
        return SaveItem(case, RESTART_ON, global_save, global_save, "restart")
    if case == "CHEM_OFF_DIFF_OFF":
        return SaveItem(case, OFF_RUN, global_save, global_save, "fresh")
    if case == "CHEM_ON_DIFF_ON_C3":
        return SaveItem(case, C3_RUN, global_save, global_save, "fresh")
    raise ValueError(case)


def max_global_save_candidate() -> int:
    candidates = []
    for run_dir in [SOURCE_ON, RESTART_ON, OFF_RUN, C3_RUN]:
        steps = available_d_steps(run_dir)
        if steps:
            candidates.append(max(steps))
    return max(candidates) if candidates else -1


def save_has_fields(item: SaveItem, fields: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in fields if not field_available(item.run_dir, name, item.raw_step)]
    return not missing, missing


def common_saves(required_fields: list[str]) -> tuple[list[int], list[dict]]:
    max_save = max_global_save_candidate()
    common: list[int] = []
    rows: list[dict] = []
    for save in range(max_save + 1):
        all_ok = True
        row = {"global_save": save, "time_us": save * T_SAVE * 1e6}
        for case in CASE_ORDER:
            item = item_for(case, save)
            ok, missing = save_has_fields(item, required_fields)
            row[f"{case}_run_dir"] = str(item.run_dir.relative_to(REPO))
            row[f"{case}_raw_step"] = item.raw_step
            row[f"{case}_origin"] = item.origin
            row[f"{case}_has_required"] = "T" if ok else "F"
            row[f"{case}_missing"] = ",".join(missing)
            all_ok = all_ok and ok
        row["all_three_have_required"] = "T" if all_ok else "F"
        if all_ok:
            common.append(save)
        rows.append(row)
    return common, rows


def grid(field: dict):
    values = field.get("values", {})
    if not values:
        return None
    xs = np.array(sorted({key[0] for key in values}), dtype=float)
    ys = np.array(sorted({key[1] for key in values}), dtype=float)
    if xs.size == 0 or ys.size == 0:
        return None
    xi = {float(x): i for i, x in enumerate(xs)}
    yi = {float(y): i for i, y in enumerate(ys)}
    arr = np.full((ys.size, xs.size), np.nan)
    for (x, y), value in values.items():
        arr[yi[float(y)], xi[float(x)]] = float(value)
    return xs, ys, arr


def edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    mids = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (mids[0] - values[0])], mids, [values[-1] + (values[-1] - mids[-1])]))


def finite_values(field: dict) -> np.ndarray:
    vals = np.array(list(field.get("values", {}).values()), dtype=float)
    return vals[np.isfinite(vals)]


def read_fields(item: SaveItem, names: list[str]) -> dict[str, dict]:
    return {name: raw.read_field(item.run_dir, name, item.raw_step) for name in names}


def empty_field() -> dict:
    return {"available": False, "values": {}, "stats": raw.missing_stats([], 0)}


def read_available_fields(item: SaveItem, names: list[str]) -> dict[str, dict]:
    fields = {}
    for name in names:
        if name in raw.FIELDS and field_available(item.run_dir, name, item.raw_step):
            fields[name] = raw.read_field(item.run_dir, name, item.raw_step)
        else:
            fields[name] = empty_field()
    return fields


def valid_gas_temperature(fields: dict[str, dict]) -> dict:
    temp = raw.reconstruct_temperature(fields, gas_mass_threshold=None)
    masks = gas_metrics.mask_context(fields, GAS_MASS_FLOOR)
    vals = {
        key: value
        for key, value in temp.get("values", {}).items()
        if key in masks["valid_gas_thermo"]
    }
    return {
        "available": bool(vals),
        "values": vals,
        "stats": raw.stats_from_values(vals, available=bool(vals)),
    }


def valid_gas_keys(fields: dict[str, dict]) -> set[tuple[float, float]]:
    required = ["liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho", "pressure"]
    if not all(fields.get(name, empty_field()).get("available") for name in required):
        return set()
    return set(gas_metrics.mask_context(fields, GAS_MASS_FLOOR)["valid_gas_thermo"])


def finite_sum(field: dict) -> float:
    if not field.get("available"):
        return math.nan
    values = finite_values(field)
    return float(np.sum(values)) if values.size else math.nan


def compute_metrics_incremental(item: SaveItem, all_species: list[str], case_name: str | None = None) -> dict:
    """Compute compact per-save metrics while tolerating unavailable fields."""
    wanted = sorted(set(BASE_FIELDS + [f"rhoY_{s}" for s in sorted(set(all_species + SPECIES_OF_INTEREST))]))
    fields = read_available_fields(item, wanted)
    out_case = case_name or item.case
    row = {
        "case": out_case,
        "save": item.global_save,
        "global_save": item.global_save,
        "raw_step": item.raw_step,
        "origin": item.origin,
        "time_s": item.global_save * T_SAVE,
        "time_us": item.global_save * T_SAVE * 1e6,
        "run_dir": str(item.run_dir.relative_to(REPO)),
        "missing_fields": ",".join(name for name in METRIC_FIELDS if not fields.get(name, empty_field()).get("available")),
    }
    dx, dy, area = gas_metrics.estimate_cell_area(fields)
    row["cell_area_m2"] = area
    if fields["liquid_alpha_rho"].get("available") and math.isfinite(area):
        row["liquid_mass"] = finite_sum(fields["liquid_alpha_rho"]) * area
    else:
        row["liquid_mass"] = math.nan
    if fields["vapor_alpha_rho"].get("available") and math.isfinite(area):
        row["vapor_mass"] = finite_sum(fields["vapor_alpha_rho"]) * area
    else:
        row["vapor_mass"] = math.nan
    if fields["liquid_alpha"].get("available") and math.isfinite(area):
        liquid_keys = {
            key
            for key, value in fields["liquid_alpha"]["values"].items()
            if math.isfinite(value) and value >= 0.5
        }
        row["liquid_alpha_area_m2"] = len(liquid_keys) * area
        if liquid_keys:
            xs = [key[0] for key in liquid_keys]
            ys = [key[1] for key in liquid_keys]
            row["liquid_extent_x_m"] = max(xs) - min(xs)
            row["liquid_extent_y_m"] = max(ys) - min(ys)
        else:
            row["liquid_extent_x_m"] = math.nan
            row["liquid_extent_y_m"] = math.nan
    else:
        row["liquid_alpha_area_m2"] = math.nan
        row["liquid_extent_x_m"] = math.nan
        row["liquid_extent_y_m"] = math.nan
    temp = valid_gas_temperature(fields)
    temp_vals = finite_values(temp)
    row["valid_gas_Tmax_K"] = float(np.max(temp_vals)) if temp_vals.size else math.nan
    row["valid_gas_Tmean_K"] = float(np.mean(temp_vals)) if temp_vals.size else math.nan
    row["valid_gas_cell_count"] = int(temp_vals.size)
    for species in SPECIES_OF_INTEREST:
        field = fields.get(f"rhoY_{species}", empty_field())
        row[f"integrated_{species}"] = finite_sum(field) * area if field.get("available") and math.isfinite(area) else math.nan
        vals = finite_values(field)
        row[f"max_{species}"] = float(np.max(vals)) if vals.size else math.nan
    missing_species = [name for name in all_species if not fields.get(f"rhoY_{name}", empty_field()).get("available")]
    row["missing_species_fields"] = ",".join(missing_species)
    keys = valid_gas_keys(fields)
    if missing_species:
        row.update(species_diagnostics_unavailable(missing_species))
    elif keys and math.isfinite(area):
        row.update(species_bounds(fields, all_species, keys, area))
    else:
        row.update(species_diagnostics_unavailable([]))
    row["available"] = "T"
    return row


def species_bounds(
    fields: dict[str, dict],
    all_species: list[str],
    valid_keys: set[tuple[float, float]],
    area: float,
) -> dict:
    gas_mass = {
        key: fields["vapor_alpha_rho"]["values"].get(key, math.nan)
        + fields["air_alpha_rho"]["values"].get(key, math.nan)
        for key in valid_keys
    }
    sum_rhoY = {key: 0.0 for key in valid_keys}
    neg_rhoY = neg_Y = above_Y = 0
    min_rhoY = math.inf
    min_Y = math.inf
    max_Y = -math.inf
    max_Y_species = ""
    min_rhoY_species = ""
    for name in all_species:
        field = fields.get(f"rhoY_{name}")
        if not field or not field.get("available"):
            continue
        for key in valid_keys:
            rhoY = field["values"].get(key, math.nan)
            gm = gas_mass.get(key, math.nan)
            if not math.isfinite(rhoY) or not math.isfinite(gm) or gm <= GAS_MASS_FLOOR:
                continue
            y = rhoY / gm
            sum_rhoY[key] += rhoY
            if rhoY < min_rhoY:
                min_rhoY = rhoY
                min_rhoY_species = name
            if y < min_Y:
                min_Y = y
            if y > max_Y:
                max_Y = y
                max_Y_species = name
            neg_rhoY += int(rhoY < -BOUNDS_TOL)
            neg_Y += int(y < -BOUNDS_TOL)
            above_Y += int(y > 1.0 + BOUNDS_TOL)
    sumY = [
        sum_rhoY[key] / gas_mass[key]
        for key in valid_keys
        if math.isfinite(gas_mass.get(key, math.nan)) and gas_mass[key] > GAS_MASS_FLOOR
    ]
    gas_int = sum(v for v in gas_mass.values() if math.isfinite(v)) * area
    species_int = sum(sum_rhoY.values()) * area
    max_abs_sum = max((abs(v - 1.0) for v in sumY), default=math.nan)
    return {
        "species_negative_rhoY_count": neg_rhoY,
        "species_negative_Y_count": neg_Y,
        "species_Y_above_one_count": above_Y,
        "species_sumY_min": min(sumY) if sumY else math.nan,
        "species_sumY_max": max(sumY) if sumY else math.nan,
        "species_sumY_max_abs_error": max_abs_sum,
        "species_sumY_violation_count": sum(1 for v in sumY if abs(v - 1.0) > BOUNDS_TOL),
        "species_min_rhoY": min_rhoY if math.isfinite(min_rhoY) else math.nan,
        "species_min_rhoY_name": min_rhoY_species,
        "species_min_Y": min_Y if math.isfinite(min_Y) else math.nan,
        "species_max_Y": max_Y if math.isfinite(max_Y) else math.nan,
        "species_max_Y_name": max_Y_species,
        "integrated_valid_gas_mass": gas_int,
        "integrated_valid_gas_sum_rhoY": species_int,
        "mass_consistency_integrated_diff": species_int - gas_int,
        "mass_consistency_relative_error": (species_int - gas_int) / gas_int if gas_int else math.nan,
        "species_diagnostics_available": "T",
        "species_diagnostics_missing_reason": "",
    }


def species_diagnostics_unavailable(missing_species: list[str]) -> dict:
    return {
        "species_negative_rhoY_count": math.nan,
        "species_negative_Y_count": math.nan,
        "species_Y_above_one_count": math.nan,
        "species_sumY_min": math.nan,
        "species_sumY_max": math.nan,
        "species_sumY_max_abs_error": math.nan,
        "species_sumY_violation_count": math.nan,
        "species_min_rhoY": math.nan,
        "species_min_rhoY_name": "",
        "species_min_Y": math.nan,
        "species_max_Y": math.nan,
        "species_max_Y_name": "",
        "integrated_valid_gas_mass": math.nan,
        "integrated_valid_gas_sum_rhoY": math.nan,
        "mass_consistency_integrated_diff": math.nan,
        "mass_consistency_relative_error": math.nan,
        "species_diagnostics_available": "F",
        "species_diagnostics_missing_reason": (
            "missing species fields: " + ",".join(missing_species)
            if missing_species else "valid-gas mask or cell area unavailable"
        ),
    }


def compute_metrics(item: SaveItem, all_species: list[str]) -> dict:
    field_names = sorted(set(BASE_FIELDS + [f"rhoY_{s}" for s in sorted(set(all_species + SPECIES_OF_INTEREST))]))
    fields = read_fields(item, field_names)
    row = {
        "case": item.case,
        "case_label": CASE_LABEL[item.case],
        "global_save": item.global_save,
        "raw_step": item.raw_step,
        "origin": item.origin,
        "time_s": item.global_save * T_SAVE,
        "time_us": item.global_save * T_SAVE * 1e6,
        "run_dir": str(item.run_dir.relative_to(REPO)),
    }
    missing = [name for name in METRIC_FIELDS if not fields.get(name, {}).get("available")]
    row["missing_metric_fields"] = ",".join(missing)
    if missing:
        row["available"] = "F"
        return row
    masks = gas_metrics.mask_context(fields, GAS_MASS_FLOOR)
    dx, dy, area = gas_metrics.estimate_cell_area(fields)
    row["available"] = "T"
    row["cell_area_m2"] = area
    alpha_liq = fields["liquid_alpha"]["values"]
    liquid_keys = {
        key
        for key, value in alpha_liq.items()
        if math.isfinite(value) and value >= 0.5
    }
    row["liquid_alpha_area_m2"] = len(liquid_keys) * area if math.isfinite(area) else math.nan
    if liquid_keys:
        xs = [key[0] for key in liquid_keys]
        ys = [key[1] for key in liquid_keys]
        row["liquid_extent_x_m"] = max(xs) - min(xs)
        row["liquid_extent_y_m"] = max(ys) - min(ys)
    else:
        row["liquid_extent_x_m"] = math.nan
        row["liquid_extent_y_m"] = math.nan
    row["liquid_mass"] = fields["liquid_alpha_rho"]["stats"]["sum"] * area
    row["vapor_mass"] = fields["vapor_alpha_rho"]["stats"]["sum"] * area
    temp = valid_gas_temperature(fields)
    temp_values = finite_values(temp)
    row["valid_gas_cell_count"] = int(temp_values.size)
    row["valid_gas_Tmax_K"] = float(np.max(temp_values)) if temp_values.size else math.nan
    row["valid_gas_Tmean_K"] = float(np.mean(temp_values)) if temp_values.size else math.nan
    for species in SPECIES_OF_INTEREST:
        field = fields[f"rhoY_{species}"]
        row[f"integrated_{species}"] = field["stats"]["sum"] * area if field.get("available") else math.nan
        vals = finite_values(field)
        row[f"max_{species}"] = float(np.max(vals)) if vals.size else math.nan
    missing_species = [name for name in all_species if not fields.get(f"rhoY_{name}", {}).get("available")]
    row["missing_species_fields"] = ",".join(missing_species)
    if missing_species:
        row.update(species_diagnostics_unavailable(missing_species))
    else:
        row.update(species_bounds(fields, all_species, set(masks["valid_gas_thermo"]), area))
    return row


def add_liquid_loss(rows: list[dict]) -> None:
    initial: dict[str, float] = {}
    for case in CASE_ORDER:
        case_rows = [r for r in rows if r["case"] == case and r.get("available") == "T"]
        if case_rows:
            initial[case] = ff(case_rows[0].get("liquid_mass"))
    for row in rows:
        base = initial.get(row["case"], math.nan)
        liquid_mass = ff(row.get("liquid_mass"))
        if math.isfinite(base) and abs(base) > REL_DENOM_EPS and math.isfinite(liquid_mass):
            row["liquid_mass_loss"] = base - liquid_mass
            row["liquid_mass_loss_fraction"] = (base - liquid_mass) / base
        else:
            row["liquid_mass_loss"] = math.nan
            row["liquid_mass_loss_fraction"] = math.nan


def rel_diff(a: float, b: float) -> tuple[float, str]:
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan, "nonfinite"
    if abs(b) <= REL_DENOM_EPS:
        return math.nan, "zero_or_near_zero_denominator"
    return (a - b) / b, "ok"


def matched_differences(rows: list[dict]) -> list[dict]:
    by = {(r["case"], int(r["global_save"])): r for r in rows if r.get("available") == "T"}
    saves = sorted(
        save
        for save in {int(r["global_save"]) for r in rows}
        if all((case, save) in by for case in CASE_ORDER)
    )
    out: list[dict] = []
    pairs = [
        ("C3_minus_CHEM_ON_DIFF_OFF", "CHEM_ON_DIFF_ON_C3", "CHEM_ON_DIFF_OFF"),
        ("C3_minus_CHEM_OFF_DIFF_OFF", "CHEM_ON_DIFF_ON_C3", "CHEM_OFF_DIFF_OFF"),
        ("CHEM_ON_minus_CHEM_OFF", "CHEM_ON_DIFF_OFF", "CHEM_OFF_DIFF_OFF"),
    ]
    for save in saves:
        row = {"global_save": save, "time_us": save * T_SAVE * 1e6}
        for label, case_a, case_b in pairs:
            arow = by[(case_a, save)]
            brow = by[(case_b, save)]
            for metric in DIFF_METRICS:
                a = ff(arow.get(metric))
                b = ff(brow.get(metric))
                row[f"{label}_{metric}_abs"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
                rel, status = rel_diff(a, b)
                row[f"{label}_{metric}_rel"] = rel
                row[f"{label}_{metric}_rel_status"] = status
        out.append(row)
    return out


def load_frame_field(item: SaveItem, variable: str) -> tuple[dict, str, str, bool]:
    base_names = ["pressure", "liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"]
    if variable == "pressure":
        field = raw.read_field(item.run_dir, "pressure", item.raw_step)
        if field.get("available"):
            field = dict(field)
            field["values"] = {key: value / 1.0e6 for key, value in field["values"].items()}
        return field, "Pressure", "MPa", False
    if variable == "valid_gas_temperature":
        fields = read_fields(item, base_names)
        if not all(fields[name].get("available") for name in base_names):
            return {"available": False, "values": {}, "stats": raw.missing_stats([], 0)}, "Valid-gas temperature", "K", False
        return valid_gas_temperature(fields), "Valid-gas temperature", "K", False
    if variable == "liquid_alpha":
        return raw.read_field(item.run_dir, "liquid_alpha", item.raw_step), "Liquid alpha", "dimensionless", False
    field = raw.read_field(item.run_dir, f"rhoY_{variable}", item.raw_step)
    return field, f"{variable} partial density", r"kg m$^{-3}$", True


def liquid_alpha_field(item: SaveItem) -> dict:
    return raw.read_field(item.run_dir, "liquid_alpha", item.raw_step)


def choose_scale(fields: list[dict], variable: str) -> dict:
    vals = [finite_values(field) for field in fields]
    vals = [arr for arr in vals if arr.size]
    if not vals:
        return {"available": False, "log": False, "vmin": 0.0, "vmax": 1.0}
    allv = np.concatenate(vals)
    if variable == "liquid_alpha":
        return {"available": True, "log": False, "vmin": 0.0, "vmax": 1.0}
    use_log = False
    vmin = float(np.nanmin(allv))
    vmax = float(np.nanmax(allv))
    if variable in SPECIES_OF_INTEREST:
        pos = allv[np.isfinite(allv) & (allv > 0.0)]
        if pos.size:
            pmin = float(np.min(pos))
            pmax = float(np.max(pos))
            if pmax / max(pmin, 1.0e-300) >= 1.0e4 and pmax > 0.0:
                use_log = True
                vmin = max(pmin, pmax * 1.0e-10)
                vmax = pmax
            else:
                vmin = min(0.0, vmin)
                vmax = pmax
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return {"available": True, "log": use_log, "vmin": vmin, "vmax": vmax}


def draw_three_panel(save: int, variable: str, output_path: Path) -> tuple[bool, str, dict]:
    items = [item_for(case, save) for case in PANEL_ORDER]
    loaded = [load_frame_field(item, variable) for item in items]
    fields = [entry[0] for entry in loaded]
    if not all(field.get("available") and finite_values(field).size for field in fields):
        return False, "field unavailable/no finite values in at least one case", {}
    label, unit, is_species = loaded[0][1], loaded[0][2], loaded[0][3]
    scale = choose_scale(fields, variable)
    if not scale["available"]:
        return False, "no finite values for scale", {}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharex=True, sharey=True, constrained_layout=True)
    meshes = []
    for ax, case, item, field in zip(axes, PANEL_ORDER, items, fields):
        g = grid(field)
        if g is None:
            plt.close(fig)
            return False, f"{case} has no gridded {variable}", scale
        xs, ys, arr = g
        x_um = xs * 1.0e6
        y_um = ys * 1.0e6
        plot_arr = arr
        if scale["log"]:
            plot_arr = np.where(arr > 0.0, arr, np.nan)
            mesh = ax.pcolormesh(
                edges(x_um),
                edges(y_um),
                plot_arr,
                shading="auto",
                cmap="magma",
                norm=colors.LogNorm(vmin=scale["vmin"], vmax=scale["vmax"]),
            )
        else:
            mesh = ax.pcolormesh(
                edges(x_um),
                edges(y_um),
                plot_arr,
                shading="auto",
                cmap="viridis",
                vmin=scale["vmin"],
                vmax=scale["vmax"],
            )
        meshes.append(mesh)
        alpha = liquid_alpha_field(item)
        ag = grid(alpha)
        if ag is not None:
            axx, ayy, aa = ag
            if axx.size >= 2 and ayy.size >= 2 and np.nanmin(aa) <= 0.5 <= np.nanmax(aa):
                ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="white", linewidths=1.1)
                ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_title(CASE_LABEL[case], fontsize=10)
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y [µm]")
    cb = fig.colorbar(meshes[-1], ax=axes, pad=0.015)
    cb.set_label(label if unit == "dimensionless" else f"{label} [{unit}]")
    scale_label = "log" if scale["log"] else "linear"
    fig.suptitle(f"{label}, t = {save * T_SAVE * 1e6:.2f} µs, shared {scale_label} scale")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=210)
    plt.close(fig)
    scale["is_species"] = is_species
    return True, "", scale


def make_frame_manifest(dry_run: bool = False) -> list[dict]:
    rows: list[dict] = []
    for time_us in FRAME_TIMES_US:
        save = int(round(time_us / (T_SAVE * 1e6)))
        for variable in FRAME_VARIABLES:
            missing: list[str] = []
            for case in PANEL_ORDER:
                item = item_for(case, save)
                field_name = (
                    "pressure"
                    if variable == "pressure"
                    else "liquid_alpha"
                    if variable == "liquid_alpha"
                    else f"rhoY_{variable}"
                    if variable in SPECIES_OF_INTEREST
                    else ""
                )
                if variable == "valid_gas_temperature":
                    needed = ["pressure", "liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"]
                    ok, case_missing = save_has_fields(item, needed)
                    if not ok:
                        missing.append(f"{case}:{','.join(case_missing)}")
                elif not field_name or not field_available(item.run_dir, field_name, item.raw_step):
                    missing.append(f"{case}:{field_name or variable}")
            output = OUT / "comparison_frames" / variable / f"{variable}_save{save:03d}_t{time_us:.2f}us.png"
            if missing:
                rows.append({
                    "time_us": time_us,
                    "global_save": save,
                    "variable": variable,
                    "status": "skipped",
                    "reason": "; ".join(missing),
                    "file": "",
                })
                continue
            if dry_run:
                rows.append({
                    "time_us": time_us,
                    "global_save": save,
                    "variable": variable,
                    "status": "would_render",
                    "reason": "",
                    "file": str(output.relative_to(OUT)),
                })
            else:
                ok, reason, scale = draw_three_panel(save, variable, output)
                rows.append({
                    "time_us": time_us,
                    "global_save": save,
                    "variable": variable,
                    "status": "written" if ok else "skipped",
                    "reason": reason,
                    "file": str(output.relative_to(OUT)) if ok else "",
                    "scale": "log" if scale.get("log") else "linear",
                    "vmin": scale.get("vmin", math.nan),
                    "vmax": scale.get("vmax", math.nan),
                })
    return rows


def plot_metric(rows: list[dict], key: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    present_cases = [case for case in CASE_ORDER + EXPORT_CASE_ORDER if any(r.get("case") == case for r in rows)]
    for case in present_cases:
        rr = [r for r in rows if r["case"] == case and r.get("available") == "T" and math.isfinite(ff(r.get(key)))]
        label = CASE_SHORT.get(case, EXPORT_CASE_LABEL.get(case, case))
        ax.plot([ff(r["time_us"]) for r in rr], [ff(r[key]) for r in rr], "o-", ms=3, lw=1.3, label=label)
    ax.axvline(0.85, color="k", ls="--", lw=1.0)
    ax.text(0.855, 0.98, "shock exits domain", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="left", fontsize=8)
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    out = OUT / "trend_plots" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def make_trend_plots(rows: list[dict]) -> None:
    for key, ylabel, filename in TREND_GROUPS:
        plot_metric(rows, key, ylabel, filename)
    combined = [
        ("integrated_fuel_and_oxygen.png", ["integrated_NC12H26", "integrated_O2"], r"Integrated partial density [kg m$^{-1}$]"),
        ("integrated_OH_and_HO2.png", ["integrated_OH", "integrated_HO2"], r"Integrated partial density [kg m$^{-1}$]"),
        ("integrated_H2O2.png", ["integrated_H2O2"], r"Integrated partial density [kg m$^{-1}$]"),
        ("integrated_CO_CO2_H2O.png", ["integrated_CO", "integrated_CO2", "integrated_H2O"], r"Integrated partial density [kg m$^{-1}$]"),
        ("liquid_area_and_extents.png", ["liquid_alpha_area_m2", "liquid_extent_x_m", "liquid_extent_y_m"], "Area/extent [SI units]"),
    ]
    for filename, keys, ylabel in combined:
        fig, ax = plt.subplots(figsize=(8.0, 4.9), constrained_layout=True)
        present_cases = [case for case in CASE_ORDER + EXPORT_CASE_ORDER if any(r.get("case") == case for r in rows)]
        for case in present_cases:
            for key in keys:
                rr = [r for r in rows if r["case"] == case and r.get("available") == "T" and math.isfinite(ff(r.get(key)))]
                label = CASE_SHORT.get(case, EXPORT_CASE_LABEL.get(case, case))
                ax.plot(
                    [ff(r["time_us"]) for r in rr],
                    [ff(r[key]) for r in rr],
                    marker="o",
                    ms=2.5,
                    lw=1.1,
                    label=f"{label} {key.replace('integrated_', '')}",
                )
        ax.axvline(0.85, color="k", ls="--", lw=1.0)
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(OUT / "trend_plots" / filename, dpi=220)
        plt.close(fig)


def make_summary(rows: list[dict], diffs: list[dict], frame_manifest: list[dict]) -> None:
    complete = [r for r in rows if r.get("available") == "T"]
    final_by_case = {}
    for case in CASE_ORDER:
        rr = [r for r in complete if r["case"] == case]
        if rr:
            final_by_case[case] = max(rr, key=lambda r: int(r["global_save"]))
    lines = [
        "Three-case C3 / chemistry ON-OFF comparison summary",
        "====================================================",
        "",
        f"Output directory: {OUT}",
        f"Common metric rows: {len(diffs)} matched saves",
        "",
        "Final available states:",
    ]
    for case in CASE_ORDER:
        row = final_by_case.get(case)
        if row:
            lines.append(
                f"- {case}: save {row['global_save']} t={ff(row['time_us']):.2f} us, "
                f"liquid_loss_fraction={ff(row.get('liquid_mass_loss_fraction')):.8e}, "
                f"Tmax={ff(row.get('valid_gas_Tmax_K')):.8e} K, "
                f"sumY_range=[{ff(row.get('species_sumY_min')):.8e}, {ff(row.get('species_sumY_max')):.8e}]"
            )
        else:
            lines.append(f"- {case}: unavailable")
    lines += [
        "",
        "Frame status:",
        f"- would/wrote frames: {sum(1 for r in frame_manifest if r['status'] in {'written', 'would_render'})}",
        f"- skipped frames: {sum(1 for r in frame_manifest if r['status'] == 'skipped')}",
        "",
        "Caveats:",
        "- all three comparisons use the evap-only validation configuration;",
        "- rejected condensation proposals mean the phase-change history is not yet a final bidirectional physical result;",
        "- C3 uses the provisional gas-dominant diffusion interface mask;",
        "- successful C3 completion proves numerical viability under the gated configuration, not final physical validation;",
        "- the C3 post-process segmentation fault occurred after simulation completion and is unrelated to the raw-field analysis;",
        "- chemistry-limiter aggregate severity is unavailable unless present in logs.",
    ]
    (OUT / "three_case_final_summary.txt").write_text("\n".join(lines) + "\n")


def make_report(rows: list[dict], diffs: list[dict], frame_manifest: list[dict]) -> None:
    lines = [
        "# C3 chemistry/diffusion comparison, 0–2 µs",
        "",
        "Cases:",
        "",
        "- `CHEM_OFF_DIFF_OFF`: chemistry OFF / diffusion OFF.",
        "- `CHEM_ON_DIFF_OFF`: chemistry ON / diffusion OFF; saves 0–18 from source and 19 onward from restart.",
        "- `CHEM_ON_DIFF_ON_C3`: chemistry ON / diffusion ON C3.",
        "",
        "The valid-gas mask is `gas_alpha > 0.5`, `gas_mass > 1e-8`, and `alpha_liq < 0.5`.",
        "",
        "Do not infer ignition from isolated Tmax values. Product/radical production and thermal changes are reported separately.",
        "",
        "## Caveats",
        "",
        "- all three comparisons use the evap-only validation configuration;",
        "- rejected condensation proposals mean the phase-change history is not yet a final bidirectional physical result;",
        "- C3 uses the provisional gas-dominant diffusion interface mask;",
        "- successful C3 completion proves numerical viability under the gated configuration, not final physical validation;",
        "- the C3 post-process segmentation fault occurred after simulation completion and is unrelated to the raw-field analysis;",
        "- chemistry-limiter aggregate severity is unavailable unless present in logs.",
        "",
        "## Outputs",
        "",
        "- `three_case_timeseries.csv`",
        "- `three_case_matched_differences.csv`",
        "- `frame_manifest.csv`",
        "- `trend_plots/`",
        "- `comparison_frames/`",
        "",
        "## Availability",
        "",
        f"- matched metric saves: {len(diffs)}",
        f"- skipped requested frames: {sum(1 for r in frame_manifest if r['status'] == 'skipped')}",
        "",
        "## Final-state snapshot",
        "",
        "| case | save | time [µs] | liquid mass loss fraction | vapor mass | Tmax [K] | max |sumY-1| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASE_ORDER:
        rr = [r for r in rows if r["case"] == case and r.get("available") == "T"]
        if not rr:
            continue
        row = max(rr, key=lambda r: int(r["global_save"]))
        lines.append(
            f"| {case} | {row['global_save']} | {ff(row['time_us']):.2f} | "
            f"{ff(row.get('liquid_mass_loss_fraction')):.8e} | {ff(row.get('vapor_mass')):.8e} | "
            f"{ff(row.get('valid_gas_Tmax_K')):.8e} | {ff(row.get('species_sumY_max_abs_error')):.8e} |"
        )
    (OUT / "three_case_comparison_report.md").write_text("\n".join(lines) + "\n")


def full_domain_arrays(
    field: dict,
    alpha_field: dict,
    valid_by_presence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ag = grid(alpha_field)
    if ag is None:
        return None
    x, y, alpha_values = ag
    values = np.full_like(alpha_values, np.nan)
    valid_mask = np.zeros(alpha_values.shape, dtype=bool)
    field_values = field.get("values", {})
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            value = field_values.get((float(xx), float(yy)), math.nan)
            values[iy, ix] = value
            valid_mask[iy, ix] = (
                (float(xx), float(yy)) in field_values
                if valid_by_presence else math.isfinite(value)
            )
    return x, y, values, valid_mask, alpha_values


def full_domain_field(field: dict, alpha_field: dict) -> dict:
    arrays = full_domain_arrays(field, alpha_field)
    if arrays is None:
        return empty_field()
    x, y, values, _valid_mask, _alpha = arrays
    out_values = {
        (float(xx), float(yy)): float(values[iy, ix])
        for iy, yy in enumerate(y)
        for ix, xx in enumerate(x)
    }
    return {
        "available": True,
        "values": out_values,
        "stats": raw.stats_from_values(out_values, available=True),
    }


def mask_description(variable: str) -> str:
    if variable == "valid_gas_temperature":
        return "actual valid-gas mask: gas_alpha > 0.5, gas_mass > 1e-8, alpha_liq < 0.5"
    return "finite field values on the full liquid-alpha domain grid"


def save_field_npz(path: Path, field: dict, item: SaveItem, variable: str, alpha_field: dict) -> bool:
    arrays = full_domain_arrays(field, alpha_field, valid_by_presence=(variable == "valid_gas_temperature"))
    if arrays is None:
        return False
    x, y, values, valid_mask, alpha_values = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        values=values,
        time=np.array(item.global_save * T_SAVE),
        save=np.array(item.global_save),
        raw_step=np.array(item.raw_step),
        variable=np.array(variable),
        valid_mask=valid_mask,
        mask_description=np.array(mask_description(variable)),
        alpha_liq=alpha_values,
    )
    return True


def draw_single_export_field(field: dict, alpha: dict, variable: str, item: SaveItem, path: Path) -> bool:
    g = grid(field)
    if g is None:
        return False
    x, y, values = g
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    scale = choose_scale([field], variable)
    fig, ax = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    if scale["log"]:
        mesh = ax.pcolormesh(
            edges(x * 1.0e6),
            edges(y * 1.0e6),
            np.where(values > 0.0, values, np.nan),
            shading="auto",
            cmap="magma",
            norm=colors.LogNorm(vmin=scale["vmin"], vmax=scale["vmax"]),
        )
    else:
        mesh = ax.pcolormesh(
            edges(x * 1.0e6),
            edges(y * 1.0e6),
            values,
            shading="auto",
            cmap="viridis",
            vmin=scale["vmin"],
            vmax=scale["vmax"],
        )
    ag = grid(alpha)
    if ag is not None:
        axx, ayy, aa = ag
        if axx.size >= 2 and ayy.size >= 2 and np.nanmin(aa) <= 0.5 <= np.nanmax(aa):
            ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="white", linewidths=1.1)
            ax.contour(axx * 1.0e6, ayy * 1.0e6, aa, levels=[0.5], colors="black", linewidths=0.35)
    fig.colorbar(mesh, ax=ax, pad=0.02)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"C3 {variable}, save {item.global_save}, t={item.global_save * T_SAVE * 1e6:.2f} µs")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return True


def nearest_save(target_time_us: float, saves: list[int]) -> int | None:
    if not saves:
        return None
    return min(saves, key=lambda save: abs(save * T_SAVE * 1e6 - target_time_us))


def c3_export(output_dir: Path | None = None) -> None:
    out = output_dir or (OUT / "c3_export")
    all_species = species_names()
    ensure_species_fields(all_species)
    saves = sorted(available_d_steps(C3_RUN))
    rows: list[dict] = []
    time_rows: list[dict] = []
    for save in saves:
        item = SaveItem(EXPORT_C3_CASE, C3_RUN, save, save, "fresh")
        rows.append(compute_metrics_incremental(item, all_species, EXPORT_C3_CASE))
        time_rows.append({
            "case": EXPORT_C3_CASE,
            "save": save,
            "raw_step": save,
            "time_s": save * T_SAVE,
            "time_us": save * T_SAVE * 1e6,
            "run_dir": str(C3_RUN.relative_to(REPO)),
        })
    if rows:
        base = next((ff(r.get("liquid_mass")) for r in rows if math.isfinite(ff(r.get("liquid_mass")))), math.nan)
        for row in rows:
            liquid = ff(row.get("liquid_mass"))
            if math.isfinite(base) and abs(base) > REL_DENOM_EPS and math.isfinite(liquid):
                row["liquid_mass_loss"] = base - liquid
                row["liquid_mass_loss_fraction"] = (base - liquid) / base
            else:
                row["liquid_mass_loss"] = math.nan
                row["liquid_mass_loss_fraction"] = math.nan
    write_csv(out / "c3_timeseries.csv", rows)
    write_csv(out / "c3_time_map.csv", time_rows)
    write_csv(out / "c3_species_bounds_by_save.csv", [
        {key: row.get(key, "") for key in [
            "save",
            "global_save",
            "raw_step",
            "time_s",
            "time_us",
            "species_negative_rhoY_count",
            "species_negative_Y_count",
            "species_Y_above_one_count",
            "species_sumY_min",
            "species_sumY_max",
            "species_sumY_max_abs_error",
            "species_sumY_violation_count",
            "species_min_rhoY",
            "species_min_rhoY_name",
            "species_min_Y",
            "species_max_Y",
            "species_max_Y_name",
        ]}
        for row in rows
    ])
    write_csv(out / "c3_mass_consistency_by_save.csv", [
        {key: row.get(key, "") for key in [
            "global_save",
            "raw_step",
            "time_s",
            "time_us",
            "integrated_valid_gas_mass",
            "integrated_valid_gas_sum_rhoY",
            "mass_consistency_integrated_diff",
            "mass_consistency_relative_error",
        ]}
        for row in rows
    ])
    manifest: list[dict] = []
    for target_us in FRAME_TIMES_US:
        save = nearest_save(target_us, saves)
        if save is None:
            for variable in FRAME_VARIABLES:
                manifest.append({"target_time_us": target_us, "status": "skipped", "reason": "no C3 saves", "variable": variable})
            continue
        item = SaveItem(EXPORT_C3_CASE, C3_RUN, save, save, "fresh")
        actual_us = save * T_SAVE * 1e6
        alpha = raw.read_field(C3_RUN, "liquid_alpha", save) if field_available(C3_RUN, "liquid_alpha", save) else empty_field()
        for variable in FRAME_VARIABLES:
            field, _label, _unit, _is_species = load_frame_field(item, variable)
            row = {
                "target_time_us": target_us,
                "actual_time_us": actual_us,
                "time_mismatch_us": actual_us - target_us,
                "save": save,
                "raw_step": save,
                "variable": variable,
            }
            if not field.get("available") or finite_values(field).size == 0:
                row.update({"status": "skipped", "reason": "field unavailable/no finite values"})
                manifest.append(row)
                continue
            if not alpha.get("available"):
                row.update({"status": "skipped", "reason": "liquid alpha unavailable for full-domain export"})
                manifest.append(row)
                continue
            stem = f"{variable}_save{save:03d}_t{actual_us:.2f}us"
            png = out / "field_frames" / variable / f"{stem}.png"
            npz = out / "field_data" / variable / f"{stem}.npz"
            full_field = full_domain_field(field, alpha)
            png_ok = draw_single_export_field(full_field, alpha, variable, item, png)
            npz_ok = save_field_npz(npz, field, item, variable, alpha)
            row.update({
                "status": "written" if png_ok and npz_ok else "partial",
                "reason": "" if png_ok and npz_ok else "PNG or NPZ write failed",
                "png": str(png.relative_to(out)) if png_ok else "",
                "npz": str(npz.relative_to(out)) if npz_ok else "",
            })
            manifest.append(row)
    write_csv(out / "c3_frame_manifest.csv", manifest)
    make_export_trend_plots(rows, out / "trend_plots")
    summary = [
        "C3 compact export",
        "=================",
        "",
        f"run_dir: {C3_RUN}",
        f"output_dir: {out}",
        f"saves: {saves}",
        f"timeseries_rows: {len(rows)}",
        f"frame_manifest_rows: {len(manifest)}",
        "",
        "NPZ fields: x, y, values, time, save, raw_step, variable, valid_mask, alpha_liq.",
        "Raw D/ and p_all/ data are not copied.",
        "Unavailable pressure/temperature fields are skipped without aborting.",
    ]
    (out / "c3_export_summary.txt").write_text("\n".join(summary) + "\n")


def make_export_trend_plots(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, filename in TREND_GROUPS:
        rr = [row for row in rows if math.isfinite(ff(row.get(key)))]
        if not rr:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        ax.plot([ff(r["time_us"]) for r in rr], [ff(r[key]) for r in rr], "o-", ms=3, lw=1.3)
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def nearest_row(time_us: float, rows: list[dict]) -> dict | None:
    valid = [row for row in rows if math.isfinite(ff(row.get("time_us")))]
    if not valid:
        return None
    return min(valid, key=lambda row: abs(ff(row["time_us"]) - time_us))


def local_raw_rows_for_compare(all_species: list[str]) -> list[dict]:
    rows: list[dict] = []
    max_save = max_global_save_candidate()
    cases = [
        ("chemistry_OFF_diffusion_OFF", "CHEM_OFF_DIFF_OFF"),
        ("chemistry_ON_diffusion_OFF", "CHEM_ON_DIFF_OFF"),
    ]
    for _export_case, raw_case in cases:
        for save in range(max_save + 1):
            item = item_for(raw_case, save)
            if not available_d_steps(item.run_dir) or item.raw_step not in available_d_steps(item.run_dir):
                continue
            row = compute_metrics_incremental(item, all_species, _export_case)
            rows.append(row)
    add_export_liquid_loss(rows)
    return rows


def add_export_liquid_loss(rows: list[dict]) -> None:
    initial: dict[str, float] = {}
    for case in {row["case"] for row in rows}:
        rr = [row for row in rows if row["case"] == case and math.isfinite(ff(row.get("liquid_mass")))]
        if rr:
            initial[case] = ff(sorted(rr, key=lambda row: ff(row["time_us"]))[0]["liquid_mass"])
    for row in rows:
        base = initial.get(row["case"], math.nan)
        liquid = ff(row.get("liquid_mass"))
        if math.isfinite(base) and abs(base) > REL_DENOM_EPS and math.isfinite(liquid):
            row["liquid_mass_loss"] = base - liquid
            row["liquid_mass_loss_fraction"] = (base - liquid) / base
        else:
            row["liquid_mass_loss"] = math.nan
            row["liquid_mass_loss_fraction"] = math.nan


def c3_rows_from_export(path: Path) -> list[dict]:
    rows = read_csv_rows(path / "c3_timeseries.csv")
    for row in rows:
        row["case"] = EXPORT_C3_CASE
    return rows


def compare_from_c3_export(export_path: Path) -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    export_path = export_path.resolve()
    if not (export_path / "c3_timeseries.csv").is_file():
        raise RuntimeError(f"missing {export_path / 'c3_timeseries.csv'}")
    c3_rows = c3_rows_from_export(export_path)
    raw_rows = local_raw_rows_for_compare(all_species)
    by_case = {case: [row for row in raw_rows if row["case"] == case] for case in EXPORT_CASE_ORDER[:2]}
    matched_rows: list[dict] = []
    time_rows: list[dict] = []
    for c3 in c3_rows:
        time_us = ff(c3.get("time_us"))
        if not math.isfinite(time_us):
            continue
        c3_out = dict(c3)
        c3_out["case"] = EXPORT_C3_CASE
        time_rows.append(c3_out)
        for case in EXPORT_CASE_ORDER[:2]:
            row = nearest_row(time_us, by_case[case])
            if row is not None:
                out_row = dict(row)
                out_row["matched_to_C3_time_us"] = time_us
                out_row["time_mismatch_us"] = ff(row.get("time_us")) - time_us
                time_rows.append(out_row)
        matched_rows.append(make_export_diff_row(time_us, c3, by_case))
    write_csv(OUT / "three_case_timeseries.csv", time_rows)
    write_csv(OUT / "three_case_matched_differences.csv", matched_rows)
    frame_manifest = make_export_comparison_frames(export_path, by_case, c3_rows)
    write_csv(OUT / "frame_manifest.csv", frame_manifest)
    make_trend_plots(time_rows)
    make_export_summary(time_rows, matched_rows, frame_manifest, export_path)
    make_export_report(time_rows, matched_rows, frame_manifest, export_path)


def make_export_diff_row(time_us: float, c3: dict, by_case: dict[str, list[dict]]) -> dict:
    row = {"C3_time_us": time_us}
    cases = {
        EXPORT_C3_CASE: c3,
        "chemistry_OFF_diffusion_OFF": nearest_row(time_us, by_case["chemistry_OFF_diffusion_OFF"]),
        "chemistry_ON_diffusion_OFF": nearest_row(time_us, by_case["chemistry_ON_diffusion_OFF"]),
    }
    for case, case_row in cases.items():
        if case_row is None:
            continue
        row[f"{case}_time_us"] = ff(case_row.get("time_us"))
        row[f"{case}_time_mismatch_us"] = ff(case_row.get("time_us")) - time_us
    pairs = [
        ("C3_minus_chemistry_ON_diffusion_OFF", EXPORT_C3_CASE, "chemistry_ON_diffusion_OFF"),
        ("C3_minus_chemistry_OFF_diffusion_OFF", EXPORT_C3_CASE, "chemistry_OFF_diffusion_OFF"),
        ("chemistry_ON_minus_chemistry_OFF", "chemistry_ON_diffusion_OFF", "chemistry_OFF_diffusion_OFF"),
    ]
    for label, a_case, b_case in pairs:
        arow = cases.get(a_case)
        brow = cases.get(b_case)
        for metric in DIFF_METRICS:
            a = ff(arow.get(metric)) if arow else math.nan
            b = ff(brow.get(metric)) if brow else math.nan
            row[f"{label}_{metric}_abs"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            rel, status = rel_diff(a, b)
            row[f"{label}_{metric}_rel"] = rel
            row[f"{label}_{metric}_rel_status"] = status
    return row


def c3_npz_for(export_path: Path, variable: str, target_time_us: float) -> tuple[Path | None, dict]:
    manifest = read_csv_rows(export_path / "c3_frame_manifest.csv")
    candidates = [
        row for row in manifest
        if row.get("variable") == variable and row.get("status") == "written" and row.get("npz")
    ]
    if not candidates:
        return None, {}
    best = min(candidates, key=lambda row: abs(ff(row.get("actual_time_us")) - target_time_us))
    return export_path / best["npz"], best


def field_from_npz(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    x = np.array(data["x"], dtype=float)
    y = np.array(data["y"], dtype=float)
    values = np.array(data["values"], dtype=float)
    alpha = np.array(data["alpha_liq"], dtype=float)
    field_values = {}
    alpha_values = {}
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            field_values[(float(xx), float(yy))] = float(values[iy, ix])
            alpha_values[(float(xx), float(yy))] = float(alpha[iy, ix])
    return (
        {"available": True, "values": field_values, "stats": raw.stats_from_values(field_values, True)},
        {"available": True, "values": alpha_values, "stats": raw.stats_from_values(alpha_values, True)},
    )


def make_export_comparison_frames(export_path: Path, by_case: dict[str, list[dict]], c3_rows: list[dict]) -> list[dict]:
    manifest: list[dict] = []
    for target_us in FRAME_TIMES_US:
        for variable in FRAME_VARIABLES:
            npz, c3_meta = c3_npz_for(export_path, variable, target_us)
            if npz is None:
                manifest.append({"target_time_us": target_us, "variable": variable, "status": "skipped", "reason": "missing C3 NPZ"})
                continue
            c3_time = ff(c3_meta.get("actual_time_us"))
            c3_save = int(ff(c3_meta.get("save"))) if math.isfinite(ff(c3_meta.get("save"))) else -1
            raw_items = []
            missing = []
            for export_case, raw_case in [
                ("chemistry_OFF_diffusion_OFF", "CHEM_OFF_DIFF_OFF"),
                ("chemistry_ON_diffusion_OFF", "CHEM_ON_DIFF_OFF"),
            ]:
                row = nearest_row(c3_time, by_case[export_case])
                if row is None:
                    missing.append(f"{export_case}:no time row")
                    continue
                item = item_for(raw_case, int(ff(row["global_save"])))
                field, _label, _unit, _species = load_frame_field(item, variable)
                if not field.get("available") or finite_values(field).size == 0:
                    missing.append(f"{export_case}:{variable} unavailable")
                raw_items.append((export_case, item, field, liquid_alpha_field(item), row))
            if missing:
                manifest.append({
                    "target_time_us": target_us,
                    "variable": variable,
                    "status": "skipped",
                    "reason": "; ".join(missing),
                    "C3_actual_time_us": c3_time,
                    "C3_time_mismatch_us": c3_time - target_us,
                    "C3_save": c3_save,
                })
                continue
            c3_field, c3_alpha = field_from_npz(npz)
            output = OUT / "comparison_frames" / variable / f"{variable}_t{target_us:.2f}us.png"
            ok, reason = draw_export_three_panel(raw_items, c3_field, c3_alpha, c3_meta, variable, output)
            manifest.append({
                "target_time_us": target_us,
                "variable": variable,
                "status": "written" if ok else "skipped",
                "reason": reason,
                "file": str(output.relative_to(OUT)) if ok else "",
                "chemistry_OFF_actual_time_us": ff(raw_items[0][4].get("time_us")),
                "chemistry_OFF_time_mismatch_us": ff(raw_items[0][4].get("time_us")) - c3_time,
                "chemistry_OFF_save": raw_items[0][1].global_save,
                "chemistry_ON_actual_time_us": ff(raw_items[1][4].get("time_us")),
                "chemistry_ON_time_mismatch_us": ff(raw_items[1][4].get("time_us")) - c3_time,
                "chemistry_ON_save": raw_items[1][1].global_save,
                "C3_actual_time_us": c3_time,
                "C3_time_mismatch_us": c3_time - target_us,
                "C3_save": c3_save,
            })
    return manifest


def draw_export_three_panel(
    raw_items: list[tuple[str, SaveItem, dict, dict, dict]],
    c3_field: dict,
    c3_alpha: dict,
    c3_meta: dict,
    variable: str,
    output: Path,
) -> tuple[bool, str]:
    panels = [
        ("chemistry_OFF_diffusion_OFF", raw_items[0][2], raw_items[0][3], raw_items[0][1].global_save, ff(raw_items[0][4].get("time_us"))),
        ("chemistry_ON_diffusion_OFF", raw_items[1][2], raw_items[1][3], raw_items[1][1].global_save, ff(raw_items[1][4].get("time_us"))),
        (EXPORT_C3_CASE, c3_field, c3_alpha, int(ff(c3_meta.get("save"))), ff(c3_meta.get("actual_time_us"))),
    ]
    fields = [panel[1] for panel in panels]
    if not all(field.get("available") and finite_values(field).size for field in fields):
        return False, "missing finite panel data"
    scale = choose_scale(fields, variable)
    if variable == "valid_gas_temperature":
        for field in fields:
            field["values"] = {key: val for key, val in field["values"].items() if math.isfinite(val)}
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for ax, (case, field, alpha, save, time_us) in zip(axes, panels):
        g = grid(field)
        if g is None:
            plt.close(fig)
            return False, f"{case} has no gridded data"
        x, y, vals = g
        if scale["log"]:
            mesh = ax.pcolormesh(
                edges(x * 1.0e6),
                edges(y * 1.0e6),
                np.where(vals > 0.0, vals, np.nan),
                shading="auto",
                cmap="magma",
                norm=colors.LogNorm(vmin=scale["vmin"], vmax=scale["vmax"]),
            )
        else:
            mesh = ax.pcolormesh(
                edges(x * 1.0e6),
                edges(y * 1.0e6),
                vals,
                shading="auto",
                cmap="viridis",
                vmin=scale["vmin"],
                vmax=scale["vmax"],
            )
        ag = grid(alpha)
        if ag is not None:
            axx, ayy, avals = ag
            if np.nanmin(avals) <= 0.5 <= np.nanmax(avals):
                ax.contour(axx * 1.0e6, ayy * 1.0e6, avals, levels=[0.5], colors="white", linewidths=1.1)
                ax.contour(axx * 1.0e6, ayy * 1.0e6, avals, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_title(f"{EXPORT_CASE_LABEL[case]}\nsave {save}, t={time_us:.2f} µs", fontsize=9)
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y [µm]")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, pad=0.015)
    fig.suptitle(f"{variable}, shared finite-value color scale")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210)
    plt.close(fig)
    return True, ""


def make_export_summary(rows: list[dict], diffs: list[dict], frame_manifest: list[dict], export_path: Path) -> None:
    lines = [
        "Three-case comparison from compact C3 export",
        "============================================",
        "",
        f"C3 export: {export_path}",
        f"timeseries_rows: {len(rows)}",
        f"matched_difference_rows: {len(diffs)}",
        f"frame_rows: {len(frame_manifest)}",
        "",
        "Relative differences are blank/NaN when the denominator is zero, near zero, or nonfinite.",
        "",
        "Caveats:",
        "- C3 used temporary evap-only gates and a provisional diffusion mask.",
        "- Successful completion is numerical viability, not physical validation.",
        "- The C3 post-process segmentation fault occurred after simulation completion and is unrelated to raw-field analysis.",
        "- Maximum temperature alone does not prove ignition.",
    ]
    (OUT / "three_case_final_summary.txt").write_text("\n".join(lines) + "\n")


def make_export_report(rows: list[dict], diffs: list[dict], frame_manifest: list[dict], export_path: Path) -> None:
    lines = [
        "# Three-case comparison from C3 compact export",
        "",
        "Case names:",
        "",
        "- `chemistry_OFF_diffusion_OFF`",
        "- `chemistry_ON_diffusion_OFF`",
        "- `chemistry_ON_diffusion_ON_C3`",
        "",
        "C3 spatial panels read numerical `.npz` arrays, not PNG pixels. ON/OFF panels read raw local `D/` fields.",
        "",
        "Matching is by nearest physical time; time mismatches are reported in the CSV outputs.",
        "",
        "Caveats:",
        "",
        "- C3 used temporary evap-only gates and a provisional diffusion mask.",
        "- Successful completion is numerical viability, not physical validation.",
        "- The C3 post-process segmentation fault occurred after simulation completion and is unrelated to raw-field analysis.",
        "- Maximum temperature alone does not prove ignition.",
        "- Chemistry-limiter aggregate severity is unavailable unless present in logs.",
        "",
        f"C3 export path: `{export_path}`",
        f"Matched rows: {len(diffs)}",
        f"Frame rows: {len(frame_manifest)}",
    ]
    (OUT / "three_case_comparison_report.md").write_text("\n".join(lines) + "\n")


def run_full() -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    saves, _availability = common_saves(METRIC_FIELDS)
    if not saves:
        raise RuntimeError("no common saves with the metric field set available in all three cases")
    rows: list[dict] = []
    for save in saves:
        for case in CASE_ORDER:
            rows.append(compute_metrics(item_for(case, save), all_species))
    add_liquid_loss(rows)
    diffs = matched_differences(rows)
    frame_manifest = make_frame_manifest(dry_run=False)
    write_csv(OUT / "three_case_timeseries.csv", rows)
    write_csv(OUT / "three_case_matched_differences.csv", diffs)
    write_csv(OUT / "frame_manifest.csv", frame_manifest)
    make_trend_plots(rows)
    make_summary(rows, diffs, frame_manifest)
    make_report(rows, diffs, frame_manifest)


def dry_run() -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    examples = REPO / "examples/2D_dodecane_global_reduced"
    mech = REPO / "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
    print(f"script: {SCRIPT}")
    print(f"repo_from_parents4: {REPO}")
    print(f"examples_path: {examples} exists={examples.is_dir()}")
    print(f"mechanism_path: {mech} exists={mech.is_file()}")
    print("run directories:")
    for label, path in [
        ("CHEM_ON_DIFF_OFF source", SOURCE_ON),
        ("CHEM_ON_DIFF_OFF restart", RESTART_ON),
        ("CHEM_OFF_DIFF_OFF", OFF_RUN),
        ("CHEM_ON_DIFF_ON_C3", C3_RUN),
    ]:
        steps = sorted(available_d_steps(path))
        preview = f"{steps[:4]}...{steps[-4:]}" if len(steps) > 8 else str(steps)
        print(f"  {label}: exists={path.is_dir()} D_exists={(path / 'D').is_dir()} saves={preview}")
    saves, availability = common_saves(METRIC_FIELDS)
    print(f"common metric saves ({len(saves)}): {saves}")
    if availability:
        print("availability sample:")
        for row in availability[:3] + availability[-3:]:
            print(
                f"  save={row['global_save']} all={row['all_three_have_required']} "
                f"ONsrc/raw={row['CHEM_ON_DIFF_OFF_raw_step']} "
                f"OFF_missing={row['CHEM_OFF_DIFF_OFF_missing']} "
                f"C3_missing={row['CHEM_ON_DIFF_ON_C3_missing']}"
            )
    manifest = make_frame_manifest(dry_run=True)
    print("requested frame dry-run:")
    for row in manifest:
        print(
            f"  t={row['time_us']:.2f}us save={row['global_save']} "
            f"var={row['variable']} status={row['status']} reason={row.get('reason', '')}"
        )


def ensure_hydro_raw_fields() -> None:
    # MFC model_eqns=3, num_fluids=3, num_dims=2 primitive layout:
    # prim.1-3 are fluid partial densities, prim.4/5 are u/v, prim.6 is pressure.
    # Density for this diagnostic is reconstructed conservatively as cons.1+cons.2+cons.3.
    raw.FIELDS.setdefault("u_velocity", ("prim", 4))
    raw.FIELDS.setdefault("v_velocity", ("prim", 5))
    raw.FIELDS.setdefault("pressure", ("prim", 6))
    raw.FIELDS.setdefault("liquid_alpha_rho", ("cons", 1))
    raw.FIELDS.setdefault("vapor_alpha_rho", ("cons", 2))
    raw.FIELDS.setdefault("air_alpha_rho", ("cons", 3))
    raw.FIELDS.setdefault("liquid_alpha", ("cons", 7))


def hydrodynamic_mappings() -> dict[str, str]:
    ensure_hydro_raw_fields()
    return {
        "u_velocity": "prim.4",
        "v_velocity": "prim.5",
        "pressure": "prim.6",
        "density": "cons.1 + cons.2 + cons.3",
        "liquid_alpha": "cons.7",
    }


def raw_grid_field(item: SaveItem, field_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = raw.read_field(item.run_dir, field_name, item.raw_step)
    return grid_required(field, field_name)


def on_saves_with_hydro_fields() -> list[int]:
    ensure_hydro_raw_fields()
    out: list[int] = []
    for save in sorted(available_d_steps(SOURCE_ON) | available_d_steps(RESTART_ON)):
        item = item_for("CHEM_ON_DIFF_OFF", save)
        needed = ["u_velocity", "v_velocity", "pressure", "liquid_alpha_rho", "vapor_alpha_rho", "air_alpha_rho", "liquid_alpha"]
        if all(field_available(item.run_dir, name, item.raw_step) for name in needed):
            out.append(save)
    return out


def nearest_on_hydro_item(reference_time_us: float) -> SaveItem | None:
    saves = on_saves_with_hydro_fields()
    if not saves:
        return None
    save = min(saves, key=lambda idx: abs(idx * T_SAVE * 1.0e6 - reference_time_us))
    return item_for("CHEM_ON_DIFF_OFF", save)


def compute_vorticity(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    edge_order = 2 if x.size >= 3 and y.size >= 3 else 1
    dvdx = np.gradient(v, x, axis=1, edge_order=edge_order)
    dudy = np.gradient(u, y, axis=0, edge_order=edge_order)
    return dvdx - dudy


def on_hydro_state(item: SaveItem) -> dict:
    ensure_hydro_raw_fields()
    x_u, y_u, u = raw_grid_field(item, "u_velocity")
    x_v, y_v, v = raw_grid_field(item, "v_velocity")
    x_p, y_p, pressure_pa = raw_grid_field(item, "pressure")
    x_l, y_l, arho_liq = raw_grid_field(item, "liquid_alpha_rho")
    x_va, y_va, arho_vap = raw_grid_field(item, "vapor_alpha_rho")
    x_a, y_a, arho_air = raw_grid_field(item, "air_alpha_rho")
    x_alpha, y_alpha, alpha = raw_grid_field(item, "liquid_alpha")
    for name, xx, yy in [
        ("v_velocity", x_v, y_v),
        ("pressure", x_p, y_p),
        ("liquid_alpha_rho", x_l, y_l),
        ("vapor_alpha_rho", x_va, y_va),
        ("air_alpha_rho", x_a, y_a),
        ("liquid_alpha", x_alpha, y_alpha),
    ]:
        if not same_grid(x_u, y_u, xx, yy):
            raise RuntimeError(f"ON_NO_DIFF hydro grid mismatch for {name} at save {item.global_save}")
    density = arho_liq + arho_vap + arho_air
    return {
        "x": x_u,
        "y": y_u,
        "u_velocity": u,
        "v_velocity": v,
        "pressure": pressure_pa / 1.0e6,
        "density": density,
        "liquid_alpha": alpha,
        "velocity_magnitude": np.sqrt(u*u + v*v),
        "vorticity": compute_vorticity(x_u, y_u, u, v),
        "save": item.global_save,
        "time_us": item.global_save * T_SAVE * 1.0e6,
    }


def c3_hydro_npz_by_save(export_path: Path, save: int, variable: str) -> dict | None:
    for rec in c3_npz_records(export_path, variable):
        if int(rec["save"]) == int(save):
            return rec
    return None


def nearest_c3_hydro_reference(export_path: Path, requested_time_us: float) -> dict | None:
    pressure = nearest_c3_npz(export_path, "pressure", requested_time_us)
    if pressure is None:
        return None
    return pressure


def c3_hydro_missing_for_save(export_path: Path, save: int) -> list[str]:
    return [name for name in HYDRO_REQUIRED_C3_NPZ if c3_hydro_npz_by_save(export_path, save, name) is None]


def c3_hydro_state(export_path: Path, save: int) -> dict:
    records = {name: c3_hydro_npz_by_save(export_path, save, name) for name in HYDRO_REQUIRED_C3_NPZ}
    missing = [name for name, rec in records.items() if rec is None]
    if missing:
        raise RuntimeError(f"C3 compact export is missing hydrodynamic NPZ fields for save {save}: {','.join(missing)}")
    grids = {name: load_npz_grid(records[name]["path"]) for name in records}
    base = grids["u_velocity"]
    x = base["x"]
    y = base["y"]
    for name, data in grids.items():
        if not same_grid(x, y, data["x"], data["y"]):
            raise RuntimeError(f"C3 compact export hydro grid mismatch for {name} at save {save}")
    u = grids["u_velocity"]["values"]
    v = grids["v_velocity"]["values"]
    return {
        "x": x,
        "y": y,
        "u_velocity": u,
        "v_velocity": v,
        "pressure": grids["pressure"]["values"],
        "density": grids["density"]["values"],
        "liquid_alpha": grids["liquid_alpha"]["values"],
        "velocity_magnitude": np.sqrt(u*u + v*v),
        "vorticity": compute_vorticity(x, y, u, v),
        "save": save,
        "time_us": float(base["time_us"]),
    }


def finite_minmax(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return math.nan, math.nan
    return float(np.min(vals)), float(np.max(vals))


def plot_hydro_variable(requested_time_us: float, variable: str, on_state: dict, c3_state: dict, out_path: Path) -> dict:
    c3_x = c3_state["x"]
    c3_y = c3_state["y"]
    interpolation_used = not same_grid(on_state["x"], on_state["y"], c3_x, c3_y)
    on_values = interpolate_array(on_state["x"], on_state["y"], on_state[variable], c3_x, c3_y)
    on_alpha = interpolate_array(on_state["x"], on_state["y"], on_state["liquid_alpha"], c3_x, c3_y)
    c3_values = c3_state[variable]
    c3_alpha = c3_state["liquid_alpha"]
    diff = c3_values - on_values
    on_min, on_max = finite_minmax(on_values)
    c3_min, c3_max = finite_minmax(c3_values)
    diff_min, diff_max = finite_minmax(diff)
    finite_pair = np.concatenate([on_values[np.isfinite(on_values)], c3_values[np.isfinite(c3_values)]])
    if finite_pair.size == 0:
        raise RuntimeError(f"no finite hydrodynamic values for {variable} at {requested_time_us} us")
    if variable == "vorticity":
        lim = float(np.nanmax(np.abs(finite_pair)))
        norm_main = colors.TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim if lim > 0 else 1.0)
        cmap_main = "coolwarm"
    else:
        vmin = float(np.nanmin(finite_pair))
        vmax = float(np.nanmax(finite_pair))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm_main = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_main = "viridis"
    diff_lim = float(np.nanmax(np.abs(diff[np.isfinite(diff)]))) if np.any(np.isfinite(diff)) else 1.0
    if not math.isfinite(diff_lim) or diff_lim <= 0.0:
        diff_lim = 1.0
    titles = ["ON_NO_DIFF", "ON_WITH_DIFF_C3", "C3 - ON_NO_DIFF"]
    arrays = [on_values, c3_values, diff]
    alphas = [on_alpha, c3_alpha, c3_alpha]
    norms = [norm_main, norm_main, colors.TwoSlopeNorm(vcenter=0.0, vmin=-diff_lim, vmax=diff_lim)]
    cmaps = [cmap_main, cmap_main, "coolwarm"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True, sharex=True, sharey=True)
    for ax, arr, alpha, title, norm, cmap in zip(axes, arrays, alphas, titles, norms, cmaps):
        mesh = ax.pcolormesh(edges(c3_x*1e6), edges(c3_y*1e6), arr, shading="auto", cmap=cmap, norm=norm)
        if np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
            ax.contour(c3_x*1e6, c3_y*1e6, alpha, levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(c3_x*1e6, c3_y*1e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, pad=0.01)
    axes[0].set_ylabel("y [µm]")
    fig.suptitle(
        f"Hydrodynamic wake {variable}; requested {requested_time_us:.2f} µs\n"
        f"ON save {on_state['save']} t={on_state['time_us']:.3f} µs; "
        f"C3 save {c3_state['save']} t={c3_state['time_us']:.3f} µs; "
        f"Δt={c3_state['time_us'] - on_state['time_us']:.3e} µs"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {
        "requested_time_us": requested_time_us,
        "c3_time_us": c3_state["time_us"],
        "on_time_us": on_state["time_us"],
        "c3_save": c3_state["save"],
        "on_save": on_state["save"],
        "time_mismatch_us": c3_state["time_us"] - on_state["time_us"],
        "variable": variable,
        "on_min": on_min,
        "on_max": on_max,
        "c3_min": c3_min,
        "c3_max": c3_max,
        "difference_min": diff_min,
        "difference_max": diff_max,
        "interpolation_used": "T" if interpolation_used else "F",
        "differentiation_method": HYDRO_DIFFERENTIATION_METHOD if variable == "vorticity" else "not_applicable",
        "output_png": str(out_path),
        "status": "written",
        "reason": "",
    }


def hydrodynamic_wake_dry_run(export_path: Path) -> None:
    ensure_hydro_raw_fields()
    print("hydrodynamic wake comparison dry-run")
    print(f"  c3_export: {export_path} exists={export_path.is_dir()}")
    print("  verified mappings:")
    for key, value in hydrodynamic_mappings().items():
        print(f"    {key}: {value}")
    on_saves = on_saves_with_hydro_fields()
    print(f"  ON_NO_DIFF hydro saves: {on_saves[:4]}...{on_saves[-4:] if len(on_saves) > 4 else on_saves}")
    available_vars = sorted({p.parent.name for p in (export_path / "field_data").glob("*/*.npz")}) if (export_path / "field_data").is_dir() else []
    print(f"  C3 compact field variables: {available_vars}")
    for target in HYDRO_WAKE_TIMES_US:
        ref = nearest_c3_hydro_reference(export_path, target)
        if ref is None:
            print(f"  requested {target:.2f} us: BLOCKED missing C3 pressure NPZ reference")
            continue
        missing = c3_hydro_missing_for_save(export_path, int(ref["save"]))
        on_item = nearest_on_hydro_item(float(ref["time_us"]))
        on_status = "available" if on_item else "missing"
        status = "ready" if not missing and on_item else "blocked"
        print(
            f"  requested {target:.2f} us: status={status} c3_save={ref['save']} "
            f"c3_time={ref['time_us']:.3f} us on_status={on_status} "
            f"missing_c3_fields={','.join(missing) if missing else 'none'}"
        )


def make_hydrodynamic_wake_frames(export_path: Path, dry_run_only: bool = False) -> None:
    ensure_hydro_raw_fields()
    if dry_run_only:
        hydrodynamic_wake_dry_run(export_path)
        return
    manifest: list[dict] = []
    for target in HYDRO_WAKE_TIMES_US:
        ref = nearest_c3_hydro_reference(export_path, target)
        if ref is None:
            raise RuntimeError(f"missing C3 pressure NPZ reference near {target:.2f} us in {export_path}")
        missing = c3_hydro_missing_for_save(export_path, int(ref["save"]))
        if missing:
            raise RuntimeError(
                f"cannot make hydrodynamic wake frames for C3 save {ref['save']}: "
                f"missing compact C3 NPZ fields {','.join(missing)}. "
                "Regenerate/export C3 u_velocity, v_velocity, density, pressure, and liquid_alpha fields."
            )
        on_item = nearest_on_hydro_item(float(ref["time_us"]))
        if on_item is None:
            raise RuntimeError(f"missing ON_NO_DIFF hydrodynamic raw fields near C3 time {ref['time_us']:.3f} us")
        on_state = on_hydro_state(on_item)
        c3_state = c3_hydro_state(export_path, int(ref["save"]))
        out_dir = OUT / "hydrodynamic_wake_frames" / f"t{target:.2f}us"
        for variable in HYDRO_WAKE_VARIABLES:
            row = plot_hydro_variable(target, variable, on_state, c3_state, out_dir / f"{variable}.png")
            manifest.append(row)
    write_csv(OUT / "hydrodynamic_wake_manifest.csv", manifest)


def c3_npz_records(export_path: Path, variable: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted((export_path / "field_data").rglob("*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            var = str(np.asarray(data["variable"]).item())
            if var != variable:
                continue
            records.append({
                "path": path,
                "variable": var,
                "time_us": float(np.asarray(data["time"]).item()) * 1.0e6,
                "save": int(np.asarray(data["save"]).item()),
                "raw_step": int(np.asarray(data["raw_step"]).item()),
            })
        except Exception:
            continue
    return records


def nearest_c3_npz(export_path: Path, variable: str, requested_time_us: float) -> dict | None:
    records = c3_npz_records(export_path, variable)
    if not records:
        return None
    return min(records, key=lambda row: abs(row["time_us"] - requested_time_us))


def load_npz_grid(path: Path, variable: str = "", save: int | None = None) -> dict:
    if not isinstance(path, (str, os.PathLike)):
        keys = ",".join(str(key) for key in path.keys()) if isinstance(path, dict) else ""
        raise TypeError(
            "load_npz_grid expected a filesystem path"
            f"; variable={variable or 'unknown'}"
            f"; save={save if save is not None else 'unknown'}"
            f"; received_type={type(path).__name__}"
            f"; metadata_keys={keys or 'unavailable'}"
        )
    data = np.load(path, allow_pickle=False)
    return {
        "x": np.array(data["x"], dtype=float),
        "y": np.array(data["y"], dtype=float),
        "values": np.array(data["values"], dtype=float),
        "valid_mask": np.array(data["valid_mask"], dtype=bool),
        "alpha_liq": np.array(data["alpha_liq"], dtype=float),
        "time_us": float(np.asarray(data["time"]).item()) * 1.0e6,
        "save": int(np.asarray(data["save"]).item()),
        "raw_step": int(np.asarray(data["raw_step"]).item()),
        "variable": str(np.asarray(data["variable"]).item()),
        "mask_description": str(np.asarray(data["mask_description"]).item()) if "mask_description" in data else "",
    }


def load_c3_npz_record(record: dict, variable: str, save: int) -> dict:
    if not isinstance(record, dict) or "path" not in record:
        keys = ",".join(str(key) for key in record.keys()) if isinstance(record, dict) else ""
        raise TypeError(
            "C3 NPZ lookup did not return the established metadata record with a path key"
            f"; variable={variable}; save={save}"
            f"; received_type={type(record).__name__}"
            f"; metadata_keys={keys or 'unavailable'}"
        )
    return load_npz_grid(record["path"], variable=variable, save=save)


def on_saves_with_variable(variable: str) -> list[int]:
    saves = sorted(available_d_steps(SOURCE_ON) | available_d_steps(RESTART_ON))
    out: list[int] = []
    for save in saves:
        item = item_for("CHEM_ON_DIFF_OFF", save)
        if variable == "valid_gas_temperature":
            needed = ["pressure", "liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"]
            ok, _missing = save_has_fields(item, needed)
        elif variable == "liquid_alpha":
            ok = field_available(item.run_dir, "liquid_alpha", item.raw_step)
        else:
            ok = field_available(item.run_dir, f"rhoY_{variable}", item.raw_step)
        if ok:
            out.append(save)
    return out


def nearest_on_item(variable: str, reference_time_us: float) -> SaveItem | None:
    saves = on_saves_with_variable(variable)
    if not saves:
        return None
    save = min(saves, key=lambda idx: abs(idx * T_SAVE * 1.0e6 - reference_time_us))
    return item_for("CHEM_ON_DIFF_OFF", save)


def same_grid(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> bool:
    return (
        x0.shape == x1.shape and y0.shape == y1.shape and
        np.allclose(x0, x1, rtol=0.0, atol=1.0e-14) and
        np.allclose(y0, y1, rtol=0.0, atol=1.0e-14)
    )


def interpolate_array(src_x: np.ndarray, src_y: np.ndarray, src: np.ndarray, dst_x: np.ndarray, dst_y: np.ndarray) -> np.ndarray:
    if same_grid(src_x, src_y, dst_x, dst_y):
        return np.array(src, copy=True)
    tmp = np.full((src_y.size, dst_x.size), np.nan)
    for iy in range(src_y.size):
        row = src[iy, :]
        finite = np.isfinite(row)
        if np.count_nonzero(finite) >= 2:
            tmp[iy, :] = np.interp(dst_x, src_x[finite], row[finite], left=np.nan, right=np.nan)
    out = np.full((dst_y.size, dst_x.size), np.nan)
    for ix in range(dst_x.size):
        col = tmp[:, ix]
        finite = np.isfinite(col)
        if np.count_nonzero(finite) >= 2:
            out[:, ix] = np.interp(dst_y, src_y[finite], col[finite], left=np.nan, right=np.nan)
    return out


def nearest_index_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(src - value))) for value in dst], dtype=int)


def interpolate_mask(src_x: np.ndarray, src_y: np.ndarray, src: np.ndarray, dst_x: np.ndarray, dst_y: np.ndarray) -> np.ndarray:
    if same_grid(src_x, src_y, dst_x, dst_y):
        return np.array(src, copy=True)
    xi = nearest_index_map(src_x, dst_x)
    yi = nearest_index_map(src_y, dst_y)
    out = np.zeros((dst_y.size, dst_x.size), dtype=bool)
    for j, src_j in enumerate(yi):
        for i, src_i in enumerate(xi):
            out[j, i] = bool(src[src_j, src_i])
    return out


def on_full_domain_arrays(item: SaveItem, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    field, _label, _unit, _is_species = load_frame_field(item, variable)
    alpha = liquid_alpha_field(item)
    if not field.get("available") or not alpha.get("available"):
        return None
    return full_domain_arrays(field, alpha, valid_by_presence=(variable == "valid_gas_temperature"))


def draw_difference_frame(
    variable: str,
    requested_time_us: float,
    c3: dict,
    on_item: SaveItem,
    on_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
) -> tuple[bool, str, bool]:
    on_x, on_y, on_values, on_mask, on_alpha = on_arrays
    c3_x = c3["x"]
    c3_y = c3["y"]
    interpolation_used = not same_grid(on_x, on_y, c3_x, c3_y)
    on_values_i = interpolate_array(on_x, on_y, on_values, c3_x, c3_y)
    on_alpha_i = interpolate_array(on_x, on_y, on_alpha, c3_x, c3_y)
    on_mask_i = interpolate_mask(on_x, on_y, on_mask, c3_x, c3_y)
    c3_values = c3["values"]
    c3_mask = c3["valid_mask"]
    if variable == "valid_gas_temperature":
        mask = c3_mask & on_mask_i & np.isfinite(c3_values) & np.isfinite(on_values_i)
    else:
        mask = np.isfinite(c3_values) & np.isfinite(on_values_i) & c3_mask & on_mask_i
    diff = np.where(mask, c3_values - on_values_i, np.nan)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return False, "no finite overlapping difference values", interpolation_used
    vmax = float(np.nanmax(np.abs(finite)))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        edges(c3_x * 1.0e6),
        edges(c3_y * 1.0e6),
        diff,
        shading="auto",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(f"C3 - chemistry ON/diffusion OFF {variable}")
    if np.nanmin(c3["alpha_liq"]) <= 0.5 <= np.nanmax(c3["alpha_liq"]):
        ax.contour(c3_x * 1.0e6, c3_y * 1.0e6, c3["alpha_liq"], levels=[0.5], colors="black", linewidths=1.0)
    if np.nanmin(on_alpha_i) <= 0.5 <= np.nanmax(on_alpha_i):
        ax.contour(c3_x * 1.0e6, c3_y * 1.0e6, on_alpha_i, levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
    c3_time = c3["time_us"]
    on_time = on_item.global_save * T_SAVE * 1.0e6
    ax.set_title(
        f"{variable}: C3 - chemistry ON/diffusion OFF\n"
        f"C3 save {c3['save']} t={c3_time:.3f} µs; "
        f"ON save {on_item.global_save} t={on_time:.3f} µs; "
        f"Δt={on_time - c3_time:+.3e} µs"
    )
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True, "", interpolation_used


def make_difference_frames(export_path: Path) -> None:
    export_path = export_path.resolve()
    out_dir = OUT / "difference_frames"
    manifest: list[dict] = []
    for requested_time_us in DIFFERENCE_FRAME_TIMES_US:
        for variable in DIFFERENCE_FRAME_VARIABLES:
            record = nearest_c3_npz(export_path, variable, requested_time_us)
            row = {"variable": variable, "requested_time_us": requested_time_us}
            if record is None:
                row.update({"status": "skipped", "source_c3_npz": "", "output_png": "", "interpolation_used": ""})
                manifest.append(row)
                continue
            c3 = load_npz_grid(record["path"])
            on_item = nearest_on_item(variable, c3["time_us"])
            row.update({
                "c3_actual_time_us": c3["time_us"],
                "c3_save": c3["save"],
                "source_c3_npz": str(record["path"]),
            })
            if on_item is None:
                row.update({"status": "skipped", "reason": "no matching chemistry-ON raw save", "output_png": "", "interpolation_used": ""})
                manifest.append(row)
                continue
            on_arrays = on_full_domain_arrays(on_item, variable)
            on_time_us = on_item.global_save * T_SAVE * 1.0e6
            row.update({
                "on_actual_time_us": on_time_us,
                "on_save": on_item.global_save,
                "time_mismatch_us": on_time_us - c3["time_us"],
            })
            if on_arrays is None:
                row.update({"status": "skipped", "reason": "chemistry-ON field unavailable", "output_png": "", "interpolation_used": ""})
                manifest.append(row)
                continue
            output = out_dir / variable / f"{variable}_C3_minus_ON_t{requested_time_us:.2f}us.png"
            ok, reason, used_interp = draw_difference_frame(variable, requested_time_us, c3, on_item, on_arrays, output)
            row.update({
                "output_png": str(output.relative_to(OUT)) if ok else "",
                "interpolation_used": "T" if used_interp else "F",
                "status": "written" if ok else "skipped",
                "reason": reason,
            })
            manifest.append(row)
    write_csv(OUT / "difference_frame_manifest.csv", manifest)


def molecular_weight_from_composition(composition: dict[str, float]) -> float:
    atomic_weights = {
        "H": 1.00794,
        "C": 12.0107,
        "N": 14.0067,
        "O": 15.9994,
        "AR": 39.948,
        "Ar": 39.948,
    }
    total = 0.0
    for element, count in composition.items():
        if element not in atomic_weights:
            raise RuntimeError(f"atomic weight for {element!r} is not defined")
        total += atomic_weights[element] * float(count)
    return total


def mixing_stoich_coefficient() -> dict:
    mech = REPO / "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
    data = yaml.safe_load(mech.read_text())
    compositions = {
        species["name"]: species.get("composition", {})
        for species in data.get("species", [])
    }
    mw_o2 = molecular_weight_from_composition(compositions["O2"])
    mw_fuel = molecular_weight_from_composition(compositions["NC12H26"])
    coeff = 18.5 * mw_o2 / mw_fuel
    return {
        "mechanism": mech,
        "MW_O2_g_per_mol": mw_o2,
        "MW_NC12H26_g_per_mol": mw_fuel,
        "coefficient": coeff,
    }


def cell_area_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        return math.nan
    area = float(abs(np.median(dx)) * abs(np.median(dy)))
    return area if math.isfinite(area) and area > 0.0 else math.nan


def raw_species_full_arrays(item: SaveItem, species: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    field = raw.read_field(item.run_dir, f"rhoY_{species}", item.raw_step)
    alpha = liquid_alpha_field(item)
    if not field.get("available") or not alpha.get("available"):
        return None
    return full_domain_arrays(field, alpha)


def on_mixing_state(item: SaveItem) -> dict | None:
    fields = read_available_fields(item, [
        "liquid_alpha",
        "vapor_alpha",
        "air_alpha",
        "vapor_alpha_rho",
        "air_alpha_rho",
        "pressure",
        "rhoY_NC12H26",
        "rhoY_O2",
        "rhoY_OH",
        "rhoY_HO2",
        "rhoY_H2O2",
    ])
    required = [
        "liquid_alpha",
        "vapor_alpha",
        "air_alpha",
        "vapor_alpha_rho",
        "air_alpha_rho",
        "pressure",
        "rhoY_NC12H26",
        "rhoY_O2",
    ]
    if not all(fields[name].get("available") for name in required):
        return None
    x, y, alpha_liq = grid_required(fields["liquid_alpha"], "liquid_alpha")
    arrays: dict[str, np.ndarray] = {"alpha_liq": alpha_liq}
    for name in ["vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho", "pressure"]:
        gx, gy, arr = grid_required(fields[name], name)
        if not same_grid(x, y, gx, gy):
            raise RuntimeError(f"ON_NO_DIFF {name} grid does not match liquid_alpha")
        arrays[name] = arr
    for species in ["NC12H26", "O2", "OH", "HO2", "H2O2"]:
        field = fields.get(f"rhoY_{species}", empty_field())
        if field.get("available"):
            gx, gy, arr = grid_required(field, f"rhoY_{species}")
            if not same_grid(x, y, gx, gy):
                raise RuntimeError(f"ON_NO_DIFF rhoY_{species} grid does not match liquid_alpha")
            arrays[species] = arr
        else:
            arrays[species] = np.full_like(alpha_liq, np.nan)
    temp_field = valid_gas_temperature(fields)
    _tx, _ty, temp, temp_mask, _alpha = full_domain_arrays(
        temp_field,
        fields["liquid_alpha"],
        valid_by_presence=True,
    )
    gas_mass = arrays["vapor_alpha_rho"] + arrays["air_alpha_rho"]
    valid_gas = (
        np.isfinite(alpha_liq)
        & np.isfinite(arrays["vapor_alpha"])
        & np.isfinite(arrays["air_alpha"])
        & np.isfinite(gas_mass)
        & ((arrays["vapor_alpha"] + arrays["air_alpha"]) > 0.5)
        & (gas_mass > GAS_MASS_FLOOR)
        & (alpha_liq < 0.5)
        & temp_mask
    )
    arrays.update({
        "x": x,
        "y": y,
        "valid_gas_mask": valid_gas,
        "valid_gas_temperature": temp,
        "gas_mass": gas_mass,
        "pressure": arrays["pressure"] / 1.0e6,
        "time_us": item.global_save * T_SAVE * 1.0e6,
        "save": item.global_save,
    })
    return arrays


def c3_mixing_state(export_path: Path, save: int) -> dict:
    loaded = {}
    missing = []
    for variable in MIXING_STATE_FIELDS:
        record = c3_hydro_npz_by_save(export_path, save, variable)
        if record is None:
            missing.append(variable)
            continue
        loaded[variable] = load_c3_npz_record(record, variable, save)
    if missing:
        raise RuntimeError(f"C3 compact export is missing mixing-state fields for save {save}: {','.join(missing)}")
    base = loaded["liquid_alpha"]
    x = base["x"]
    y = base["y"]
    alpha_liq = base["alpha_liq"]
    out = {
        "x": x,
        "y": y,
        "alpha_liq": alpha_liq,
        "time_us": base["time_us"],
        "save": int(base["save"]),
    }
    for variable, data in loaded.items():
        if not same_grid(x, y, data["x"], data["y"]):
            raise RuntimeError(f"C3 {variable} grid does not match liquid_alpha grid")
        if data["alpha_liq"].shape != alpha_liq.shape:
            raise RuntimeError(f"C3 {variable} alpha_liq shape does not match liquid_alpha")
        out[variable] = data["values"]
    temp_data = loaded["valid_gas_temperature"]
    out["valid_gas_mask"] = temp_data["valid_mask"] & np.isfinite(temp_data["values"])
    out["gas_mass"] = None
    out["pressure"] = loaded["pressure"]["values"]
    return out


def compute_phi(state: dict, coeff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fuel = state["NC12H26"]
    o2 = state["O2"]
    mask = (
        np.array(state["valid_gas_mask"], dtype=bool)
        & np.isfinite(fuel)
        & np.isfinite(o2)
        & (fuel >= 0.0)
        & (o2 > MIXING_PHI_O2_FLOOR)
    )
    phi = np.full(fuel.shape, np.nan)
    phi[mask] = coeff * fuel[mask] / o2[mask]
    log_phi = np.full(fuel.shape, np.nan)
    finite_phi = np.isfinite(phi) & (phi > 0.0)
    log_phi[finite_phi] = np.log10(phi[finite_phi])
    return phi, log_phi, mask


def phi_band_mask(phi: np.ndarray, name: str) -> np.ndarray:
    for band_name, lo, hi, _label in MIXING_BANDS:
        if band_name != name:
            continue
        if band_name == "near_stoichiometric":
            return np.isfinite(phi) & (phi >= lo) & (phi <= hi)
        return np.isfinite(phi) & (phi >= lo) & (phi < hi)
    raise ValueError(name)


def mixing_band_stats(case: str, requested_time_us: float, state: dict, coeff: float) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    phi, log_phi, mask = compute_phi(state, coeff)
    area = cell_area_from_xy(state["x"], state["y"])
    valid_area = float(np.count_nonzero(mask) * area) if math.isfinite(area) else math.nan
    rows: list[dict] = []
    radicals = {name: state.get(name, np.full_like(phi, np.nan)) for name in MIXING_RADICALS}
    for band_name, _lo, _hi, label in MIXING_BANDS:
        band = phi_band_mask(phi, band_name) & mask
        band_area = float(np.count_nonzero(band) * area) if math.isfinite(area) else math.nan
        row = {
            "case": case,
            "requested_time_us": requested_time_us,
            "actual_time_us": state["time_us"],
            "save": state["save"],
            "band": band_name,
            "band_definition": label,
            "cell_count": int(np.count_nonzero(band)),
            "area_m2": band_area,
            "area_fraction_of_valid_gas": band_area / valid_area if valid_area and math.isfinite(valid_area) else math.nan,
            "gas_mass_weighted_fraction": math.nan,
            "gas_mass_weighted_status": "unavailable_without_C3_gas_density",
            "mean_temperature_K": float(np.nanmean(state["valid_gas_temperature"][band])) if np.any(band) else math.nan,
        }
        for radical, values in radicals.items():
            row[f"integrated_{radical}"] = (
                float(np.nansum(np.where(band, values, 0.0)) * area)
                if math.isfinite(area) else math.nan
            )
        rows.append(row)
    finite_phi = phi[mask & np.isfinite(phi)]
    summary = {
        "case": case,
        "requested_time_us": requested_time_us,
        "actual_time_us": state["time_us"],
        "save": state["save"],
        "valid_phi_cell_count": int(finite_phi.size),
        "valid_phi_area_m2": valid_area,
        "phi_0p5_to_2_area_m2": float(np.count_nonzero(mask & (phi >= 0.5) & (phi <= 2.0)) * area) if math.isfinite(area) else math.nan,
        "phi_0p8_to_1p2_area_m2": float(np.count_nonzero(mask & (phi >= 0.8) & (phi <= 1.2)) * area) if math.isfinite(area) else math.nan,
        "phi_max": float(np.nanmax(finite_phi)) if finite_phi.size else math.nan,
        "phi_p99p9": float(np.nanpercentile(finite_phi, 99.9)) if finite_phi.size else math.nan,
    }
    rows.append({**summary, "band": "all_valid_phi", "band_definition": "all finite valid-gas phi"})
    return rows, phi, log_phi, mask


def radical_phi_overlap_stats(
    case: str,
    requested_time_us: float,
    state: dict,
    phi: np.ndarray,
    mask: np.ndarray,
) -> list[dict]:
    area = cell_area_from_xy(state["x"], state["y"])
    xx, yy = np.meshgrid(state["x"], state["y"])
    log_phi = np.full(phi.shape, np.nan)
    positive_phi = mask & np.isfinite(phi) & (phi > 0.0)
    log_phi[positive_phi] = np.log10(phi[positive_phi])
    phi_layer = positive_phi & (np.abs(log_phi) <= MIXING_PHI_LAYER_LOG_HALF_WIDTH)
    layer_points = np.column_stack((xx[phi_layer], yy[phi_layer]))
    out = []
    for radical in MIXING_RADICALS:
        values = state.get(radical, np.full_like(phi, np.nan))
        finite = values[mask & np.isfinite(values)]
        if finite.size == 0:
            threshold = math.nan
            rich = np.zeros(phi.shape, dtype=bool)
        else:
            threshold = float(np.nanpercentile(finite, 99.0))
            rich = mask & np.isfinite(values) & (values >= threshold)
        rich_points = np.column_stack((xx[rich], yy[rich]))
        if rich_points.size and layer_points.size:
            distances = []
            for start in range(0, rich_points.shape[0], 256):
                diff = rich_points[start:start + 256, None, :] - layer_points[None, :, :]
                distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))
            dist = np.concatenate(distances)
            min_dist = float(np.min(dist))
            mean_dist = float(np.mean(dist))
        else:
            min_dist = math.nan
            mean_dist = math.nan
        out.append({
            "case": case,
            "requested_time_us": requested_time_us,
            "actual_time_us": state["time_us"],
            "save": state["save"],
            "radical": radical,
            "radical_99th_percentile_threshold": threshold,
            "radical_rich_cell_count": int(np.count_nonzero(rich)),
            "phi_layer_cell_count": int(np.count_nonzero(phi_layer)),
            "overlap_area_m2": float(np.count_nonzero(rich & phi_layer) * area) if math.isfinite(area) else math.nan,
            "min_distance_to_phi1_layer_m": min_dist,
            "mean_distance_to_phi1_layer_m": mean_dist,
            "phi_layer_definition": f"|log10(phi)| <= {MIXING_PHI_LAYER_LOG_HALF_WIDTH}",
        })
    return out


def draw_mixing_three_panel(
    out_path: Path,
    title: str,
    on_state: dict,
    c3_state: dict,
    on_values: np.ndarray,
    c3_values: np.ndarray,
    diff_values: np.ndarray,
    on_alpha: np.ndarray,
    c3_alpha: np.ndarray,
    cmap: str = "viridis",
    diverging: bool = False,
    contours: tuple[np.ndarray, list[float], str] | None = None,
    c3_contours: tuple[np.ndarray, list[float], str] | None = None,
    colorbar_label: str = "",
) -> None:
    finite_main = np.concatenate([
        on_values[np.isfinite(on_values)],
        c3_values[np.isfinite(c3_values)],
    ]) if np.isfinite(on_values).any() or np.isfinite(c3_values).any() else np.array([])
    if finite_main.size == 0:
        raise RuntimeError(f"no finite values for {title}")
    vmin = float(np.nanmin(finite_main))
    vmax = float(np.nanmax(finite_main))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    finite_diff = diff_values[np.isfinite(diff_values)]
    dmax = float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else 1.0
    if not math.isfinite(dmax) or dmax <= 0.0:
        dmax = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("ON_NO_DIFF", on_state, on_values, on_alpha, vmin, vmax, cmap),
        ("ON_WITH_DIFF_C3", c3_state, c3_values, c3_alpha, vmin, vmax, cmap),
        ("C3 - ON_NO_DIFF", c3_state, diff_values, c3_alpha, -dmax, dmax, "coolwarm"),
    ]
    meshes = []
    for ax, (label, state, values, alpha, lo, hi, cm) in zip(axes, panels):
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi) if cm == "coolwarm" else None
        mesh = ax.pcolormesh(
            edges(state["x"] * 1.0e6),
            edges(state["y"] * 1.0e6),
            values,
            shading="auto",
            cmap=cm,
            vmin=None if norm else lo,
            vmax=None if norm else hi,
            norm=norm,
        )
        meshes.append(mesh)
        if np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
            ax.contour(state["x"] * 1.0e6, state["y"] * 1.0e6, alpha, levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(state["x"] * 1.0e6, state["y"] * 1.0e6, alpha, levels=[0.5], colors="black", linewidths=0.3)
        if contours is not None and label == "ON_NO_DIFF":
            field, levels, color = contours
            ax.contour(state["x"] * 1.0e6, state["y"] * 1.0e6, field, levels=levels, colors=color, linewidths=0.9)
        if c3_contours is not None and label != "ON_NO_DIFF":
            field, levels, color = c3_contours
            ax.contour(state["x"] * 1.0e6, state["y"] * 1.0e6, field, levels=levels, colors=color, linewidths=0.9)
        ax.set_title(f"{label}\nsave {state['save']}, t={state['time_us']:.3f} µs", fontsize=9)
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y [µm]")
    fig.colorbar(meshes[0], ax=axes[:2], pad=0.015, label=colorbar_label)
    fig.colorbar(meshes[2], ax=axes[2], pad=0.015, label=f"C3 - ON {colorbar_label}")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def mixing_plot_set(
    out_dir: Path,
    requested_time_us: float,
    on_state: dict,
    c3_state: dict,
    coeff: float,
) -> list[dict]:
    on_phi, on_log_phi, on_mask = compute_phi(on_state, coeff)
    c3_phi, c3_log_phi, c3_mask = compute_phi(c3_state, coeff)
    on_interp = {
        "phi": interpolate_array(on_state["x"], on_state["y"], on_phi, c3_state["x"], c3_state["y"]),
        "log_phi": interpolate_array(on_state["x"], on_state["y"], on_log_phi, c3_state["x"], c3_state["y"]),
        "alpha": interpolate_array(on_state["x"], on_state["y"], on_state["alpha_liq"], c3_state["x"], c3_state["y"]),
        "mask": interpolate_mask(on_state["x"], on_state["y"], on_mask, c3_state["x"], c3_state["y"]),
    }
    rows = []
    plot_specs = [
        ("log10_phi", on_log_phi, c3_log_phi, "magma", r"log$_{10}(\phi)$"),
        ("phi_contours", np.minimum(on_phi, MIXING_PHI_PLOT_CAP), np.minimum(c3_phi, MIXING_PHI_PLOT_CAP), "viridis", "phi"),
        ("valid_gas_temperature", on_state["valid_gas_temperature"], c3_state["valid_gas_temperature"], "inferno", "K"),
        ("OH", on_state["OH"], c3_state["OH"], "magma", r"kg m$^{-3}$"),
        ("HO2", on_state["HO2"], c3_state["HO2"], "magma", r"kg m$^{-3}$"),
        ("H2O2", on_state["H2O2"], c3_state["H2O2"], "magma", r"kg m$^{-3}$"),
    ]
    for variable, on_values, c3_values, cmap, label in plot_specs:
        on_values_i = interpolate_array(on_state["x"], on_state["y"], on_values, c3_state["x"], c3_state["y"])
        if variable == "valid_gas_temperature":
            combined_mask = c3_mask & on_interp["mask"] & np.isfinite(c3_values) & np.isfinite(on_values_i)
        elif variable in MIXING_RADICALS:
            combined_mask = c3_mask & on_interp["mask"] & np.isfinite(c3_values) & np.isfinite(on_values_i)
        else:
            combined_mask = c3_mask & on_interp["mask"] & np.isfinite(c3_values) & np.isfinite(on_values_i)
        diff = np.where(combined_mask, c3_values - on_values_i, np.nan)
        on_plot = np.where(on_mask, on_values, np.nan)
        c3_plot = np.where(c3_mask, c3_values, np.nan)
        path = out_dir / f"{variable}.png"
        phi_levels = [0.5, 1.0, 2.0] if variable == "phi_contours" else [1.0]
        draw_mixing_three_panel(
            path,
            f"{variable} mixing-state comparison at requested t={requested_time_us:.2f} µs",
            on_state,
            c3_state,
            on_plot,
            c3_plot,
            diff,
            on_state["alpha_liq"],
            c3_state["alpha_liq"],
            cmap=cmap,
            contours=(on_phi, phi_levels, "cyan"),
            c3_contours=(c3_phi, phi_levels, "cyan"),
            colorbar_label=label,
        )
        rows.append({
            "requested_time_us": requested_time_us,
            "variable": variable,
            "output_png": str(path.relative_to(OUT)),
            "status": "written",
            "on_time_us": on_state["time_us"],
            "on_save": on_state["save"],
            "c3_time_us": c3_state["time_us"],
            "c3_save": c3_state["save"],
            "time_mismatch_us": on_state["time_us"] - c3_state["time_us"],
            "interpolation_used": "T" if not same_grid(on_state["x"], on_state["y"], c3_state["x"], c3_state["y"]) else "F",
        })
    return rows


def mixing_state_dry_run(export_path: Path) -> None:
    ensure_species_fields(species_names())
    coeff = mixing_stoich_coefficient()
    print("mixing-state analysis dry-run")
    print(f"repo_from_parents4: {REPO}")
    print(f"C3 export: {export_path} exists={export_path.is_dir()}")
    print(f"stoich coefficient s=18.5*MW_O2/MW_NC12H26 = {coeff['coefficient']:.10f}")
    print(f"MW_O2={coeff['MW_O2_g_per_mol']:.8f} g/mol")
    print(f"MW_NC12H26={coeff['MW_NC12H26_g_per_mol']:.8f} g/mol")
    print(f"O2 floor for phi denominator: {MIXING_PHI_O2_FLOOR:.3e}")
    print(f"phi plot cap: {MIXING_PHI_PLOT_CAP:.3e}")
    variables = sorted({record["variable"] for record in c3_npz_records(export_path, "NC12H26")})
    field_vars = sorted({path.parent.name for path in (export_path / "field_data").rglob("*.npz")})
    print(f"C3 field variables discovered: {field_vars}")
    for target in MIXING_STATE_TIMES_US:
        ref = nearest_c3_npz(export_path, "NC12H26", target)
        if ref is None:
            print(f"  t={target:.2f} us: missing C3 NC12H26 reference")
            continue
        missing = [var for var in MIXING_STATE_FIELDS if c3_hydro_npz_by_save(export_path, ref["save"], var) is None]
        validation_status = "not_checked"
        if not missing:
            try:
                sample_record = c3_hydro_npz_by_save(export_path, ref["save"], "NC12H26")
                sample = load_c3_npz_record(sample_record, "NC12H26", int(ref["save"]))
                validation_status = (
                    f"opened_NC12H26_npz shape={sample['values'].shape} "
                    f"path={sample_record['path']}"
                )
            except Exception as exc:
                validation_status = f"failed_to_open_required_npz: {exc}"
        on_item = nearest_on_item("NC12H26", ref["time_us"])
        print(
            f"  t={target:.2f} us: C3 save={ref['save']} time={ref['time_us']:.3f} us "
            f"missing_c3={','.join(missing) if missing else 'none'}"
        )
        print(f"    C3 NPZ validation: {validation_status}")
        if on_item is None:
            print("    ON_NO_DIFF: no matching save")
        else:
            print(
                f"    ON_NO_DIFF save={on_item.global_save} time={on_item.global_save * T_SAVE * 1e6:.3f} us "
                f"origin={on_item.origin}"
            )
    if variables:
        print(f"NC12H26 records discovered for variables: {variables}")
    print("gas-mass-weighted phi-band fractions: unavailable for compact C3 export without gas density")


def make_mixing_state_analysis(export_path: Path, dry_run_only: bool = False) -> None:
    export_path = export_path.resolve()
    if dry_run_only:
        mixing_state_dry_run(export_path)
        return
    ensure_species_fields(species_names())
    coeff = mixing_stoich_coefficient()
    out_root = OUT / "mixing_state_analysis"
    stats_rows: list[dict] = []
    manifest: list[dict] = []
    summary_lines = [
        "# Local mixing-state comparison",
        "",
        f"C3 compact export: `{export_path}`",
        f"Stoichiometric coefficient `s = 18.5 * MW_O2 / MW_NC12H26 = {coeff['coefficient']:.10f}`.",
        f"MW_O2 = {coeff['MW_O2_g_per_mol']:.8f} g/mol.",
        f"MW_NC12H26 = {coeff['MW_NC12H26_g_per_mol']:.8f} g/mol.",
        f"O2 floor for phi denominator: `{MIXING_PHI_O2_FLOOR:.3e}`.",
        f"Phi plotting cap: `{MIXING_PHI_PLOT_CAP:.3e}`; quantitative statistics use uncapped phi.",
        "",
        "Caveats:",
        "- This uses existing local data only; no new cluster export is required.",
        "- C3 compact export lacks gas-density fields, so gas-mass-weighted mixture-band fractions are unavailable.",
        "- Species fields are stored parent/species partial densities; phi uses their ratio, equivalent to Y_fuel/Y_O2 where both share the same gas mass.",
        "- Full Bilger mixture fraction and scalar dissipation are not calculated here; they require additional elemental/species and transport information.",
        "",
    ]
    for target in MIXING_STATE_TIMES_US:
        ref = nearest_c3_npz(export_path, "NC12H26", target)
        if ref is None:
            manifest.append({"requested_time_us": target, "status": "skipped", "reason": "missing C3 NC12H26"})
            continue
        c3_state = c3_mixing_state(export_path, int(ref["save"]))
        on_item = nearest_on_item("NC12H26", c3_state["time_us"])
        if on_item is None:
            manifest.append({"requested_time_us": target, "status": "skipped", "reason": "missing ON_NO_DIFF save"})
            continue
        on_state = on_mixing_state(on_item)
        if on_state is None:
            manifest.append({"requested_time_us": target, "status": "skipped", "reason": "missing ON_NO_DIFF fields"})
            continue
        out_dir = out_root / f"t{target:.2f}us"
        plot_rows = mixing_plot_set(out_dir, target, on_state, c3_state, coeff["coefficient"])
        manifest.extend(plot_rows)
        for case, state in [("ON_NO_DIFF", on_state), ("ON_WITH_DIFF_C3", c3_state)]:
            band_rows, phi, _log_phi, mask = mixing_band_stats(case, target, state, coeff["coefficient"])
            stats_rows.extend(band_rows)
            stats_rows.extend(radical_phi_overlap_stats(case, target, state, phi, mask))
        summary_lines.append(f"## Requested {target:.2f} µs")
        summary_lines.append("")
        summary_lines.append(
            f"- C3: save {c3_state['save']}, t={c3_state['time_us']:.3f} µs; "
            f"ON_NO_DIFF: save {on_state['save']}, t={on_state['time_us']:.3f} µs."
        )
        for case in ["ON_NO_DIFF", "ON_WITH_DIFF_C3"]:
            all_row = next(
                row for row in stats_rows
                if row.get("case") == case and row.get("requested_time_us") == target and row.get("band") == "all_valid_phi"
            )
            near = next(
                row for row in stats_rows
                if row.get("case") == case and row.get("requested_time_us") == target and row.get("band") == "near_stoichiometric"
            )
            useful = sum(
                ff(row.get("area_m2")) for row in stats_rows
                if row.get("case") == case
                and row.get("requested_time_us") == target
                and row.get("band") in {"combustible_lean", "near_stoichiometric", "moderately_rich"}
                and math.isfinite(ff(row.get("area_m2")))
            )
            summary_lines.append(
                f"- {case}: phi_p99.9={ff(all_row.get('phi_p99p9')):.6e}, "
                f"area(0.5<=phi<=2)={useful:.6e} m^2, "
                f"near-stoich area={ff(near.get('area_m2')):.6e} m^2."
            )
        summary_lines.append("")
    write_csv(out_root / "mixing_state_statistics.csv", stats_rows)
    write_csv(out_root / "mixing_state_manifest.csv", manifest)
    summary_lines.extend([
        "## Summary question framing",
        "",
        "- Does diffusion reduce very-rich fuel pockets? Compare the `very_rich` area fractions and phi percentiles.",
        "- Does diffusion increase the area of 0.5 <= phi <= 2.0 gas? Use the reported combustible/near/rich band areas.",
        "- Does diffusion broaden or shrink the near-stoichiometric layer? Use the near-stoich area and phi=1 contour frames.",
        "- Are OH, HO2, and H2O2 closer to phi=1 in C3? Use the radical 99th-percentile overlap and distance rows.",
        "- A visually weaker fuel wake can indicate dilution into a broader mixing layer; use phi bands rather than fuel field alone.",
        "",
        "All three comparisons use temporary evap-only gates and C3 uses a provisional diffusion mask; this is a local diagnostic, not final physical validation.",
    ])
    (out_root / "mixing_state_summary.md").write_text("\n".join(summary_lines) + "\n")




def fuel_case_paths() -> dict[str, list[Path]]:
    return {
        "OFF_NO_DIFF": [OFF_RUN],
        "ON_NO_DIFF": [SOURCE_ON, RESTART_ON],
        "ON_WITH_DIFF_C3": [],
    }


def fuel_item_for(case: str, save: int) -> SaveItem:
    item = item_for(FUEL_CASE_TO_COMPARISON[case], save)
    return SaveItem(case, item.run_dir, item.raw_step, item.global_save, item.origin)


def case_assignment_preview(case_path: Path, keys: list[str]) -> dict[str, str]:
    if not case_path.is_file():
        return {key: "case.py missing" for key in keys}
    namespace = {"__file__": str(case_path)}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(case_path.read_text(), str(case_path), "exec"), namespace)
        case = namespace.get("case")
        if isinstance(case, dict):
            return {key: repr(case.get(key, "not present")) for key in keys}
    except Exception:
        pass
    text = case_path.read_text()
    out: dict[str, str] = {}
    for key in keys:
        token = f'case["{key}"]'
        value = "not directly assigned in wrapper"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(token) and "=" in stripped:
                value = stripped.split("=", 1)[1].strip()
        out[key] = value
    return out


def validate_fuel_cases() -> None:
    missing: list[str] = []
    for case, paths in fuel_case_paths().items():
        for run_dir in paths:
            if not run_dir.is_dir():
                missing.append(f"{case}: missing run directory {run_dir}")
            elif not (run_dir / "D").is_dir():
                missing.append(f"{case}: missing raw D directory {run_dir / 'D'}")
    if not C3_FUEL_INVENTORY_CSV.is_file():
        missing.append(f"ON_WITH_DIFF_C3: missing compact C3 CSV {C3_FUEL_INVENTORY_CSV}")
    if missing:
        raise RuntimeError("fuel-inventory case unavailable:\n" + "\n".join(missing))


def fuel_available_saves(case: str) -> list[int]:
    if case == "ON_WITH_DIFF_C3":
        if not C3_FUEL_INVENTORY_CSV.is_file():
            return []
        return sorted(c3_parent_inventory_by_save().keys())
    candidates: set[int] = set()
    for run_dir in fuel_case_paths()[case]:
        candidates.update(available_d_steps(run_dir))
    saves: list[int] = []
    for save in sorted(candidates):
        item = fuel_item_for(case, save)
        if all(field_available(item.run_dir, field, item.raw_step) for field in FUEL_REQUIRED_FIELDS):
            saves.append(save)
    return saves


def fuel_common_saves() -> list[int]:
    save_sets = [set(fuel_available_saves(case)) for case in FUEL_CASE_ORDER]
    if not save_sets:
        return []
    return sorted(set.intersection(*save_sets))


def fuel_field_mappings() -> dict[str, tuple[str, int] | None]:
    ensure_species_fields(species_names())
    return {name: raw.FIELDS.get(name) for name in FUEL_REQUIRED_FIELDS}


def load_c3_parent_inventory_csv(path: Path = C3_FUEL_INVENTORY_CSV) -> list[dict]:
    required = [
        "save",
        "time_s",
        "time_us",
        "gas_parent_fuel_mass",
        "liquid_dodecane_mass",
        "combined_parent_dodecane_mass",
    ]
    if not path.is_file():
        raise RuntimeError(f"missing compact C3 fuel inventory CSV {path}")
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"C3 fuel inventory CSV {path} is missing columns: {','.join(missing)}")
        for raw_row in reader:
            row = {
                "case": "ON_WITH_DIFF_C3",
                "case_label": FUEL_CASE_LABEL["ON_WITH_DIFF_C3"],
                "diffusion": FUEL_CASE_DIFFUSION["ON_WITH_DIFF_C3"],
                "reactions": FUEL_CASE_REACTIONS["ON_WITH_DIFF_C3"],
                "save": int(float(raw_row["save"])),
                "raw_step": int(float(raw_row["save"])),
                "origin": "compact_c3_csv",
                "run_dir": str(path),
                "time_s": ff(raw_row["time_s"]),
                "time_us": ff(raw_row["time_us"]),
                "gas_parent_fuel_mass": ff(raw_row["gas_parent_fuel_mass"]),
                "liquid_dodecane_mass": ff(raw_row["liquid_dodecane_mass"]),
                "combined_parent_dodecane_mass": ff(raw_row["combined_parent_dodecane_mass"]),
                "nonfinite_cell_count": int(ff(raw_row.get("nonfinite_cell_count", 0)) or 0),
            }
            row["gas_fuel_mass"] = row["gas_parent_fuel_mass"]
            row["liquid_fuel_mass"] = row["liquid_dodecane_mass"]
            row["total_fuel_mass"] = row["combined_parent_dodecane_mass"]
            if not all(math.isfinite(ff(row[key])) for key in [
                "gas_parent_fuel_mass",
                "liquid_dodecane_mass",
                "combined_parent_dodecane_mass",
            ]):
                raise RuntimeError(f"C3 fuel inventory CSV {path} contains nonfinite inventory at save {row['save']}")
            rows.append(row)
    return sorted(rows, key=lambda r: int(r["save"]))


def c3_parent_inventory_by_save(path: Path = C3_FUEL_INVENTORY_CSV) -> dict[int, dict]:
    return {int(row["save"]): row for row in load_c3_parent_inventory_csv(path)}


def fuel_inventory_for_save(case: str, save: int) -> dict:
    if case == "ON_WITH_DIFF_C3":
        try:
            return dict(c3_parent_inventory_by_save()[save])
        except KeyError as exc:
            raise RuntimeError(f"C3 compact fuel inventory CSV has no save {save}") from exc
    item = fuel_item_for(case, save)
    fields = read_fields(item, FUEL_REQUIRED_FIELDS)
    data = c3_parent_inventory_grid(fields, save)
    liquid = data["liquid"]
    rhoY_fuel = data["rhoY_fuel"]
    weights = data["cell_area_weights"]
    finite_pair = np.isfinite(liquid) & np.isfinite(rhoY_fuel) & np.isfinite(weights)
    nonfinite_count = int(liquid.size - np.count_nonzero(finite_pair))
    gas_parent = float(np.sum(rhoY_fuel[finite_pair] * weights[finite_pair]))
    liquid_parent = float(np.sum(liquid[finite_pair] * weights[finite_pair]))
    combined_parent = gas_parent + liquid_parent
    if not all(math.isfinite(v) for v in [gas_parent, liquid_parent, combined_parent]):
        raise RuntimeError(
            f"{case} fuel inventory save {save}: nonfinite integrated mass "
            f"gas={gas_parent} liquid={liquid_parent} combined={combined_parent}"
        )
    return {
        "case": case,
        "case_label": FUEL_CASE_LABEL[case],
        "diffusion": FUEL_CASE_DIFFUSION[case],
        "reactions": FUEL_CASE_REACTIONS[case],
        "save": save,
        "raw_step": item.raw_step,
        "origin": item.origin,
        "run_dir": str(item.run_dir.relative_to(REPO)),
        "time_s": save * T_SAVE,
        "time_us": save * T_SAVE * 1e6,
        "gas_parent_fuel_mass": gas_parent,
        "liquid_dodecane_mass": liquid_parent,
        "combined_parent_dodecane_mass": combined_parent,
        # Backward-compatible aliases for earlier plot/CSV consumers.  These
        # are parent-dodecane inventories, not closed total-fuel budgets.
        "gas_fuel_mass": gas_parent,
        "liquid_fuel_mass": liquid_parent,
        "total_fuel_mass": combined_parent,
        "nonfinite_cell_count": nonfinite_count,
    }


def c3_parent_inventory_grid(fields: dict[str, dict], save: int) -> dict:
    x_liq, y_liq, liquid = grid_required(fields["liquid_alpha_rho"], "liquid_alpha_rho")
    x_fuel, y_fuel, rhoY_fuel = grid_required(fields["rhoY_NC12H26"], "rhoY_NC12H26")
    if not same_grid(x_liq, y_liq, x_fuel, y_fuel):
        raise RuntimeError(f"C3 fuel inventory save {save}: cons.1 and cons.58 grids do not match")
    if x_liq.size < 2 or y_liq.size < 2:
        raise RuntimeError(f"C3 fuel inventory save {save}: need at least two x/y coordinates to estimate cell area")
    dx_values = np.diff(x_liq)
    dy_values = np.diff(y_liq)
    finite_dx = dx_values[np.isfinite(dx_values) & (dx_values > 0.0)]
    finite_dy = dy_values[np.isfinite(dy_values) & (dy_values > 0.0)]
    if finite_dx.size == 0 or finite_dy.size == 0:
        raise RuntimeError(f"C3 fuel inventory save {save}: no finite positive coordinate spacing")
    dx = float(np.median(finite_dx))
    dy = float(np.median(finite_dy))
    area = dx * dy
    cell_area_weights = np.full(liquid.shape, area)
    if not np.all(np.isfinite(cell_area_weights)) or area <= 0.0:
        raise RuntimeError(f"C3 fuel inventory save {save}: invalid cell-area weights from dx={dx} dy={dy}")
    return {
        "x": x_liq,
        "y": y_liq,
        "liquid": liquid,
        "rhoY_fuel": rhoY_fuel,
        "dx_values": dx_values,
        "dy_values": dy_values,
        "dx": dx,
        "dy": dy,
        "cell_area_weights": cell_area_weights,
    }


def print_c3_inventory_grid_diagnostics(save: int, data: dict) -> None:
    x = data["x"]
    y = data["y"]
    liquid = data["liquid"]
    rhoY_fuel = data["rhoY_fuel"]
    dx_values = data["dx_values"]
    dy_values = data["dy_values"]
    weights = data["cell_area_weights"]
    finite_liq = int(np.count_nonzero(np.isfinite(liquid)))
    finite_fuel = int(np.count_nonzero(np.isfinite(rhoY_fuel)))
    finite_dx = dx_values[np.isfinite(dx_values)]
    finite_dy = dy_values[np.isfinite(dy_values)]
    finite_weights = weights[np.isfinite(weights)]
    print(
        f"C3_FUEL_INVENTORY_DIAG save={save} "
        f"liquid_shape={liquid.shape} rhoY_shape={rhoY_fuel.shape} "
        f"finite_liquid={finite_liq}/{liquid.size} finite_rhoY={finite_fuel}/{rhoY_fuel.size} "
        f"x_range=({float(np.nanmin(x)):.8e},{float(np.nanmax(x)):.8e}) "
        f"y_range=({float(np.nanmin(y)):.8e},{float(np.nanmax(y)):.8e}) "
        f"dx_minmax=({float(np.min(finite_dx)):.8e},{float(np.max(finite_dx)):.8e}) "
        f"dy_minmax=({float(np.min(finite_dy)):.8e},{float(np.max(finite_dy)):.8e}) "
        f"cell_area_minmax=({float(np.min(finite_weights)):.8e},{float(np.max(finite_weights)):.8e})"
    )


def c3_parent_inventory_for_save(run_dir: Path, save: int) -> dict:
    item = SaveItem("ON_WITH_DIFF_C3", run_dir, save, save, "fresh")
    fields = read_fields(item, ["liquid_alpha_rho", "rhoY_NC12H26"])
    data = c3_parent_inventory_grid(fields, save)
    print_c3_inventory_grid_diagnostics(save, data)
    liquid = data["liquid"]
    rhoY_fuel = data["rhoY_fuel"]
    weights = data["cell_area_weights"]
    finite_pair = np.isfinite(liquid) & np.isfinite(rhoY_fuel) & np.isfinite(weights)
    nonfinite_count = int(liquid.size - np.count_nonzero(finite_pair))
    gas_parent = float(np.sum(rhoY_fuel[finite_pair] * weights[finite_pair]))
    liquid_parent = float(np.sum(liquid[finite_pair] * weights[finite_pair]))
    combined_parent = gas_parent + liquid_parent
    if not all(math.isfinite(v) for v in [gas_parent, liquid_parent, combined_parent]):
        raise RuntimeError(
            f"C3 fuel inventory save {save}: nonfinite integrated mass "
            f"gas={gas_parent} liquid={liquid_parent} combined={combined_parent}"
        )
    return {
        "save": save,
        "time_s": save * T_SAVE,
        "time_us": save * T_SAVE * 1e6,
        "gas_parent_fuel_mass": gas_parent,
        "liquid_dodecane_mass": liquid_parent,
        "combined_parent_dodecane_mass": combined_parent,
        "nonfinite_cell_count": nonfinite_count,
    }


def c3_fuel_inventory_saves(run_dir: Path) -> list[int]:
    saves: list[int] = []
    for save in sorted(available_d_steps(run_dir)):
        if all(field_available(run_dir, field, save) for field in ["liquid_alpha_rho", "rhoY_NC12H26"]):
            saves.append(save)
    return saves


def export_c3_fuel_inventory(
    c3_raw_case: Path,
    output_file: Path,
    dry_run_only: bool = False,
    save_filter: int | None = None,
) -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    mappings = {
        "alpha_rho_liquid": raw.FIELDS.get("liquid_alpha_rho"),
        "rhoY_NC12H26": raw.FIELDS.get("rhoY_NC12H26"),
    }
    saves = c3_fuel_inventory_saves(c3_raw_case) if (c3_raw_case / "D").is_dir() else []
    if save_filter is not None:
        saves = [save for save in saves if save == save_filter]
    print("C3 parent-dodecane inventory export")
    print(f"  c3_raw_case: {c3_raw_case}")
    print(f"  D_exists: {(c3_raw_case / 'D').is_dir()}")
    print(f"  output_file: {output_file}")
    print(f"  save_filter: {save_filter if save_filter is not None else 'all'}")
    print("  mappings:")
    for name, mapping in mappings.items():
        print(f"    {name}: {mapping}")
    if saves:
        print(f"  available saves: {saves[:4]}...{saves[-4:] if len(saves) > 4 else saves}")
        print(f"  save range: {saves[0]} to {saves[-1]}")
        print(f"  time range: {saves[0]*T_SAVE*1e6:.2f} to {saves[-1]*T_SAVE*1e6:.2f} us")
    else:
        print("  available saves: []")
    if dry_run_only:
        print("  dry_run_only: no CSV written")
        return
    if not (c3_raw_case / "D").is_dir():
        raise RuntimeError(f"missing raw D directory under {c3_raw_case}")
    if not saves:
        raise RuntimeError("no C3 saves with liquid_alpha_rho and rhoY_NC12H26 available")
    rows = [c3_parent_inventory_for_save(c3_raw_case, save) for save in saves]
    initial = ff(rows[0].get("combined_parent_dodecane_mass")) if rows else math.nan
    for row in rows:
        gas = ff(row.get("gas_parent_fuel_mass"))
        liquid = ff(row.get("liquid_dodecane_mass"))
        combined = ff(row.get("combined_parent_dodecane_mass"))
        if math.isfinite(initial) and abs(initial) > REL_DENOM_EPS:
            row["gas_fraction_of_initial_combined"] = gas / initial if math.isfinite(gas) else math.nan
            row["liquid_fraction_of_initial_combined"] = liquid / initial if math.isfinite(liquid) else math.nan
            row["combined_fraction_of_initial"] = combined / initial if math.isfinite(combined) else math.nan
        else:
            row["gas_fraction_of_initial_combined"] = math.nan
            row["liquid_fraction_of_initial_combined"] = math.nan
            row["combined_fraction_of_initial"] = math.nan
        for key in [
            "gas_parent_fuel_mass",
            "liquid_dodecane_mass",
            "combined_parent_dodecane_mass",
            "gas_fraction_of_initial_combined",
            "liquid_fraction_of_initial_combined",
            "combined_fraction_of_initial",
        ]:
            if not math.isfinite(ff(row.get(key))):
                raise RuntimeError(f"C3 fuel inventory save {row['save']}: refusing to write nonfinite {key}")
    write_csv(output_file, rows)
    print(f"  wrote: {output_file}")
    print("  label: C3 parent-dodecane inventory only; not a closed conservation budget")


def fuel_inventory_rows(common_saves: list[int]) -> list[dict]:
    available_by_case = {case: fuel_available_saves(case) for case in FUEL_CASE_ORDER}
    initial: dict[str, float] = {}
    for case, saves in available_by_case.items():
        if not saves:
            initial[case] = math.nan
            continue
        initial_row = fuel_inventory_for_save(case, min(saves))
        initial[case] = ff(initial_row.get("combined_parent_dodecane_mass"))
    rows: list[dict] = []
    for save in common_saves:
        for case in FUEL_CASE_ORDER:
            row = fuel_inventory_for_save(case, save)
            base = initial.get(case, math.nan)
            gas = ff(row.get("gas_parent_fuel_mass"))
            liquid = ff(row.get("liquid_dodecane_mass"))
            combined = ff(row.get("combined_parent_dodecane_mass"))
            if math.isfinite(base) and abs(base) > REL_DENOM_EPS:
                row["gas_fraction_initial"] = gas / base if math.isfinite(gas) else math.nan
                row["liquid_fraction_initial"] = liquid / base if math.isfinite(liquid) else math.nan
                row["combined_parent_change_fraction"] = (combined - base) / base if math.isfinite(combined) else math.nan
                row["total_conservation_error"] = row["combined_parent_change_fraction"]
            else:
                row["gas_fraction_initial"] = math.nan
                row["liquid_fraction_initial"] = math.nan
                row["combined_parent_change_fraction"] = math.nan
                row["total_conservation_error"] = math.nan
            rows.append(row)
    return rows


def plot_fuel_metric(rows: list[dict], key: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for case in FUEL_CASE_ORDER:
        rr = [r for r in rows if r["case"] == case and math.isfinite(ff(r.get(key)))]
        ax.plot([ff(r["time_us"]) for r in rr], [ff(r[key]) for r in rr], "o-", ms=3, lw=1.25, label=FUEL_CASE_LABEL[case])
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = FUEL_OUT / "plots" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_fuel_differences(rows: list[dict]) -> None:
    by = {(r["case"], int(r["save"])): r for r in rows}
    saves = sorted({int(r["save"]) for r in rows})
    pairs = [
        ("ON_NO_DIFF - OFF_NO_DIFF", "ON_NO_DIFF", "OFF_NO_DIFF"),
        ("ON_WITH_DIFF_C3 - ON_NO_DIFF", "ON_WITH_DIFF_C3", "ON_NO_DIFF"),
    ]
    metrics = ["gas_parent_fuel_mass", "liquid_dodecane_mass", "combined_parent_dodecane_mass"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.4, 7.8), constrained_layout=True, sharex=True)
    for ax, metric in zip(axes, metrics):
        for label, case_a, case_b in pairs:
            xs: list[float] = []
            ys: list[float] = []
            for save in saves:
                a = by.get((case_a, save))
                b = by.get((case_b, save))
                if not a or not b:
                    continue
                av = ff(a.get(metric))
                bv = ff(b.get(metric))
                if math.isfinite(av) and math.isfinite(bv):
                    xs.append(save * T_SAVE * 1e6)
                    ys.append(av - bv)
            ax.plot(xs, ys, "o-", ms=3, lw=1.2, label=label)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time [µs]")
    path = FUEL_OUT / "plots" / "parent_dodecane_inventory_differences.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_fuel_inventory_plots(rows: list[dict]) -> None:
    plot_fuel_metric(rows, "gas_parent_fuel_mass", r"Gas NC12H26 parent fuel [kg m$^{-1}$]", "gas_fuel_inventory_vs_time.png")
    plot_fuel_metric(rows, "liquid_dodecane_mass", r"Liquid dodecane [kg m$^{-1}$]", "liquid_fuel_inventory_vs_time.png")
    plot_fuel_metric(rows, "combined_parent_dodecane_mass", r"Combined parent-dodecane inventory [kg m$^{-1}$]", "total_dodecane_inventory_vs_time.png")
    plot_fuel_metric(rows, "combined_parent_change_fraction", "Combined parent-dodecane change from initial", "total_fuel_conservation_error_vs_time.png")
    plot_fuel_differences(rows)


def fuel_row_at(rows: list[dict], case: str, save: int) -> dict | None:
    return next((r for r in rows if r["case"] == case and int(r["save"]) == save), None)


def make_fuel_inventory_summary(rows: list[dict], common_saves: list[int]) -> None:
    lines = [
        "# Parent-dodecane inventory budget",
        "",
        "This is an offline saved-state inventory over the full physical domain.",
        "It does not include reaction-source or boundary-flux integrals, so it is not a closed mass budget.",
        "Gas parent fuel is integrated from stored conservative `rhoY_NC12H26`, equivalent to",
        "`integral[(alpha_rho_vap + alpha_rho_air) * Y_NC12H26] dV`.",
        "Liquid dodecane is `integral(alpha_rho_liquid) dV`.",
        "In chemistry-ON cases, combined parent dodecane is not expected to remain constant because NC12H26 is chemically converted into products.",
        "Only `OFF_NO_DIFF` directly tests liquid-plus-vapor NC12H26 conservation using this inventory alone.",
        "C3 versus `ON_NO_DIFF` isolates how diffusion changes evaporation plus chemical consumption, but not those two mechanisms separately.",
        "",
        "## Cases",
    ]
    for case in FUEL_CASE_ORDER:
        paths = fuel_case_paths()[case]
        saves = fuel_available_saves(case)
        if case == "ON_WITH_DIFF_C3":
            path_text = str(C3_FUEL_INVENTORY_CSV)
        else:
            path_text = ", ".join(str(path.relative_to(REPO)) for path in paths)
        lines.append(f"- `{case}`: {FUEL_CASE_LABEL[case]}; {path_text}; saves={saves[:4]}...{saves[-4:] if saves else []}")
    if common_saves:
        lines += [
            "",
            "## Common time range",
            f"Common saves: {common_saves[:4]}...{common_saves[-4:]} ({len(common_saves)} saves)",
            f"Common time range: {common_saves[0]*T_SAVE*1e6:.2f} to {common_saves[-1]*T_SAVE*1e6:.2f} µs",
        ]
    else:
        lines += ["", "## Common time range", "No common saves with required fuel fields were found."]
    lines += ["", "## Requested common-time values"]
    for target in FUEL_REPORT_TIMES_US:
        if not common_saves:
            lines.append(f"- {target:.2f} µs: unavailable; no common saves.")
            continue
        save = min(common_saves, key=lambda s: abs(s * T_SAVE * 1e6 - target))
        mismatch = save * T_SAVE * 1e6 - target
        if abs(mismatch) > 0.5 * T_SAVE * 1e6 + 1e-12:
            lines.append(f"- {target:.2f} µs: unavailable within one save interval; nearest common save {save} at {save*T_SAVE*1e6:.2f} µs.")
            continue
        lines.append(f"- requested {target:.2f} µs; nearest common save {save} at {save*T_SAVE*1e6:.2f} µs:")
        for case in FUEL_CASE_ORDER:
            row = fuel_row_at(rows, case, save)
            if row is None:
                continue
            lines.append(
                f"  - {case}: gas_parent={ff(row['gas_parent_fuel_mass']):.8e}, "
                f"liquid={ff(row['liquid_dodecane_mass']):.8e}, "
                f"combined_parent={ff(row['combined_parent_dodecane_mass']):.8e}, "
                f"combined_change={ff(row['combined_parent_change_fraction']):.8e}"
            )
        off = fuel_row_at(rows, "OFF_NO_DIFF", save)
        on = fuel_row_at(rows, "ON_NO_DIFF", save)
        c3 = fuel_row_at(rows, "ON_WITH_DIFF_C3", save)
        if off and on:
            lines.append(
                f"  - OFF_NO_DIFF parent drift: {ff(off['combined_parent_change_fraction']):.8e}; "
                f"ON_NO_DIFF - OFF_NO_DIFF gas={ff(on['gas_parent_fuel_mass']) - ff(off['gas_parent_fuel_mass']):.8e}, "
                f"liquid={ff(on['liquid_dodecane_mass']) - ff(off['liquid_dodecane_mass']):.8e}."
            )
        if c3 and on:
            gas_diff = ff(c3["gas_parent_fuel_mass"]) - ff(on["gas_parent_fuel_mass"])
            liquid_diff = ff(c3["liquid_dodecane_mass"]) - ff(on["liquid_dodecane_mass"])
            combined_diff = ff(c3["combined_parent_dodecane_mass"]) - ff(on["combined_parent_dodecane_mass"])
            if math.isfinite(gas_diff) and gas_diff < 0.0:
                if math.isfinite(liquid_diff) and liquid_diff > 0.0 and math.isfinite(combined_diff) and combined_diff < 0.0:
                    accompaniment = "both more remaining liquid and less combined parent dodecane"
                elif math.isfinite(liquid_diff) and liquid_diff > 0.0:
                    accompaniment = "more remaining liquid"
                elif math.isfinite(combined_diff) and combined_diff < 0.0:
                    accompaniment = "less combined parent dodecane"
                else:
                    accompaniment = "neither more liquid nor less combined parent dodecane"
            else:
                accompaniment = "C3 gas parent fuel is not lower than ON_NO_DIFF at this time"
            lines.append(
                f"  - ON_WITH_DIFF_C3 - ON_NO_DIFF gas={gas_diff:.8e}, liquid={liquid_diff:.8e}, "
                f"combined_parent={combined_diff:.8e}; lower-C3-gas assessment: {accompaniment}."
            )
    lines += [
        "",
        "## Interpretation rules",
        "- Do not call gas-parent-fuel reduction `fuel loss` if combined parent dodecane remains conserved.",
        "- In chemistry-ON cases, combined parent dodecane can decrease through conversion of NC12H26 into products.",
        "- Material `OFF_NO_DIFF` combined-parent drift flags a conservation, boundary, clipping, or source-coupling issue.",
        "- C3 versus `ON_NO_DIFF` mixes diffusion effects on evaporation and chemical consumption unless source and boundary integrals are added.",
    ]
    FUEL_OUT.mkdir(parents=True, exist_ok=True)
    (FUEL_OUT / "fuel_inventory_budget_summary.md").write_text("\n".join(lines) + "\n")


def fuel_inventory_dry_run() -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    validate_fuel_cases()
    print("fuel inventory case paths:")
    for case, paths in fuel_case_paths().items():
        saves = fuel_available_saves(case)
        preview = f"{saves[:4]}...{saves[-4:]}" if len(saves) > 8 else str(saves)
        print(f"  {case}: {FUEL_CASE_LABEL[case]}")
        if case == "ON_WITH_DIFF_C3":
            print(f"    compact_csv: {C3_FUEL_INVENTORY_CSV}")
            print(f"      exists={C3_FUEL_INVENTORY_CSV.is_file()}")
            print("      source: copied HPC C3 parent-dodecane inventory CSV")
        for run_dir in paths:
            settings = case_assignment_preview(
                run_dir / "case.py",
                ["chem_params%diffusion", "chem_params%reactions", "chem_reaction_heat_enable", "t_stop", "t_save"],
            )
            print(f"    path: {run_dir}")
            print(f"      exists={run_dir.is_dir()} D_exists={(run_dir / 'D').is_dir()}")
            for key, value in settings.items():
                print(f"      case.py {key}: {value}")
        print(f"    stitched_by_existing_item_for={FUEL_CASE_TO_COMPARISON[case]}")
        print(f"    saves_with_required={preview}")
    print("field mappings:")
    for name, mapping in fuel_field_mappings().items():
        print(f"  {name}: {mapping}")
    common = fuel_common_saves()
    print("all_three_cases_present: T")
    print(f"chemistry_ON_no_diff_stitched: {SOURCE_ON} + {RESTART_ON}")
    print(f"common saves with required fields ({len(common)}): {common}")
    if common:
        t_min = common[0] * T_SAVE * 1e6
        t_max = common[-1] * T_SAVE * 1e6
        print(f"common time range: {t_min:.2f} to {t_max:.2f} us")
        print(f"common_times_extend_to_1p90us: {'T' if t_max >= 1.90 - 1.0e-12 else 'F'}")
    else:
        print("common time range: unavailable")
        print("common_times_extend_to_1p90us: F")
    print("formula: gas parent fuel = integral(rhoY_NC12H26) dV")
    print("liquid parent fuel = integral(alpha_rho_liquid) dV")
    print("combined parent dodecane = gas NC12H26 + liquid dodecane")
    print("no gas mask is applied to the full-domain gas parent-fuel integral")
    print("C3 values are read from the compact HPC CSV; local C3 raw files are not required for this budget")


def export_fuel_inventory_budget(dry_run_only: bool = False) -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    if dry_run_only:
        fuel_inventory_dry_run()
        return
    validate_fuel_cases()
    common = fuel_common_saves()
    if not common:
        raise RuntimeError("no common saves with fuel-inventory fields available in all three cases")
    rows = fuel_inventory_rows(common)
    FUEL_OUT.mkdir(parents=True, exist_ok=True)
    write_csv(FUEL_OUT / "fuel_inventory_budget.csv", rows)
    make_fuel_inventory_plots(rows)
    make_fuel_inventory_summary(rows, common)


DIFFUSION_SPECIES = ["NC12H26", "O2"]
DIFFUSION_TOTAL_DESCRIPTION = "sqrt(sum_k J_k^2) after zero-net correction and model-3 alpha-face weighting"


@dataclass(frozen=True)
class DiffusionVariableMap:
    liquid_alpha: tuple[str, int]
    vapor_alpha: tuple[str, int]
    air_alpha: tuple[str, int]
    vapor_alpha_rho: tuple[str, int]
    air_alpha_rho: tuple[str, int]
    pressure: tuple[str, int]
    species_prim_start: int
    species_indices: dict[str, int]


def c3_case_dict(case_path: Path) -> dict:
    namespace: dict = {}
    code = compile(case_path.read_text(), str(case_path), "exec")
    exec(code, namespace)
    case = namespace.get("case")
    if not isinstance(case, dict):
        raise RuntimeError(f"{case_path} did not define a case dictionary")
    return case


def diffusion_variable_map(case: dict, all_species: list[str]) -> DiffusionVariableMap:
    num_fluids = int(case.get("num_fluids", 3))
    if int(case.get("model_eqns", -1)) != 3:
        raise RuntimeError("diffusion diagnostic is currently defined for model_eqns == 3")
    species_indices = {name: all_species.index(name) + 1 for name in all_species}
    # MFC model-3 primitive layout used by s_compute_chemistry_diffusion_flux:
    #   1:num_fluids                 alpha_i rho_i
    #   num_fluids+1:...             velocities/pressure block
    #   advxb = num_fluids + num_dims + 2; for this 2D case advxb = 7
    #   chemxb = 12 + species_id     for SK54 primitive Y_i output
    advxb = num_fluids + int(case.get("num_dims", 2)) + 2
    chem_prim_start = 12
    return DiffusionVariableMap(
        liquid_alpha=("prim", advxb),
        vapor_alpha=("prim", advxb + 1),
        air_alpha=("prim", advxb + 2),
        vapor_alpha_rho=("prim", 2),
        air_alpha_rho=("prim", 3),
        pressure=("prim", 6),
        species_prim_start=chem_prim_start,
        species_indices=species_indices,
    )


def require_cantera():
    try:
        import cantera as ct  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on HPC environment
        raise RuntimeError("Cantera is required for --export-diffusion-diagnostics") from exc
    return ct


def grid_required(field: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = grid(field)
    if g is None:
        raise RuntimeError(f"field {name} is unavailable or cannot be gridded")
    return g


def read_mapped_field(run_dir: Path, mapping: tuple[str, int], save: int) -> dict:
    kind, index = mapping
    temp_name = f"__diffusion_diag_{kind}_{index}"
    previous = raw.FIELDS.get(temp_name)
    raw.FIELDS[temp_name] = (kind, index)
    try:
        return raw.read_field(run_dir, temp_name, save)
    finally:
        if previous is None:
            raw.FIELDS.pop(temp_name, None)
        else:
            raw.FIELDS[temp_name] = previous


def diffusion_read_save_fields(run_dir: Path, save: int, fmap: DiffusionVariableMap, all_species: list[str]) -> dict[str, dict]:
    fields = {
        "alpha_liq": read_mapped_field(run_dir, fmap.liquid_alpha, save),
        "alpha_vap": read_mapped_field(run_dir, fmap.vapor_alpha, save),
        "alpha_air": read_mapped_field(run_dir, fmap.air_alpha, save),
        "rho_vap": read_mapped_field(run_dir, fmap.vapor_alpha_rho, save),
        "rho_air": read_mapped_field(run_dir, fmap.air_alpha_rho, save),
        "pressure": read_mapped_field(run_dir, fmap.pressure, save),
    }
    for name in all_species:
        sid = fmap.species_indices[name]
        fields[f"Y_{name}"] = read_mapped_field(run_dir, ("prim", fmap.species_prim_start + sid), save)
    return fields


def arrays_from_diffusion_fields(fields: dict[str, dict], all_species: list[str]) -> dict[str, np.ndarray]:
    x, y, alpha_liq = grid_required(fields["alpha_liq"], "alpha_liq")
    out: dict[str, np.ndarray] = {"x": x, "y": y, "alpha_liq": alpha_liq}
    for key in ["alpha_vap", "alpha_air", "rho_vap", "rho_air", "pressure"]:
        xx, yy, arr = grid_required(fields[key], key)
        if not same_grid(x, y, xx, yy):
            raise RuntimeError(f"field {key} grid does not match liquid-alpha grid")
        out[key] = arr
    species_arrays = []
    missing = []
    for name in all_species:
        field = fields.get(f"Y_{name}", empty_field())
        if not field.get("available"):
            missing.append(name)
            continue
        xx, yy, arr = grid_required(field, f"Y_{name}")
        if not same_grid(x, y, xx, yy):
            raise RuntimeError(f"species field {name} grid does not match liquid-alpha grid")
        species_arrays.append(arr)
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += f", ... ({len(missing)} total)"
        raise RuntimeError(f"missing required SK54 species fields: {preview}")
    out["Y"] = np.stack(species_arrays, axis=0)
    return out


def diffusion_face_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_face = 0.5*(x[:-1] + x[1:])
    y_face = 0.5*(y[:-1] + y[1:])
    return x_face, y, x, y_face


def cantera_transport_state(gas, pressure: float, temperature: float, Y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    gas.TPY = temperature, pressure, Y
    diffusivities = np.asarray(gas.mix_diff_coeffs_mass, dtype=float)
    lambda_mix = float(gas.thermal_conductivity)
    mole_fractions = np.asarray(gas.X, dtype=float)
    enthalpies = np.asarray(gas.partial_molar_enthalpies, dtype=float) / np.asarray(gas.molecular_weights, dtype=float)
    mw = float(gas.mean_molecular_weight)
    return diffusivities, lambda_mix, mole_fractions, enthalpies, mw


def reconstruct_diffusion_fluxes(arr: dict[str, np.ndarray], species: list[str], mechanism: Path, phase: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ct = require_cantera()
    gas = ct.Solution(str(mechanism), phase)
    if list(gas.species_names) != species:
        raise RuntimeError("Cantera mechanism species order does not match parsed SK54 species order")
    molecular_weights = np.asarray(gas.molecular_weights, dtype=float)
    gas_constant = float(ct.gas_constant)
    x = arr["x"]
    y = arr["y"]
    alpha_liq = arr["alpha_liq"]
    alpha_g = arr["alpha_vap"] + arr["alpha_air"]
    rho_g_stored = arr["rho_vap"] + arr["rho_air"]
    pressure = arr["pressure"]
    Y = arr["Y"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_g_intrinsic = rho_g_stored/alpha_g
    eligible = (
        np.isfinite(alpha_g) & np.isfinite(alpha_liq) &
        (alpha_g >= 0.5) & (alpha_liq <= 0.5)
    )
    ny, nx = alpha_liq.shape
    ns = len(species)
    active_x = eligible[:, :-1] & eligible[:, 1:]
    active_y = eligible[:-1, :] & eligible[1:, :]
    Jx = np.zeros((ns, ny, max(nx - 1, 0)), dtype=float)
    Jy = np.zeros((ns, max(ny - 1, 0), nx), dtype=float)
    residuals: list[float] = []
    finite_temperature = []
    nonfinite_temperature_count = 0
    nonfinite_pressure_count = int(np.count_nonzero(~np.isfinite(pressure)))

    def compute_face(side0, side1, spacing: float) -> tuple[np.ndarray, float] | None:
        y0 = Y[:, side0[0], side0[1]]
        y1 = Y[:, side1[0], side1[1]]
        p0 = float(pressure[side0])
        p1 = float(pressure[side1])
        rho0 = float(rho_g_intrinsic[side0])
        rho1 = float(rho_g_intrinsic[side1])
        ag0 = float(alpha_g[side0])
        ag1 = float(alpha_g[side1])
        if not (np.all(np.isfinite(y0)) and np.all(np.isfinite(y1)) and math.isfinite(p0) and math.isfinite(p1)):
            return None
        if not (math.isfinite(rho0) and math.isfinite(rho1) and rho0 > 0.0 and rho1 > 0.0):
            return None
        mw0 = 1.0/np.sum(y0/molecular_weights)
        mw1 = 1.0/np.sum(y1/molecular_weights)
        rgas0 = gas_constant/mw0
        rgas1 = gas_constant/mw1
        t0 = p0/(rho0*rgas0)
        t1 = p1/(rho1*rgas1)
        if not (math.isfinite(t0) and math.isfinite(t1)):
            return None
        if t0 <= 1.0e-6 or t1 <= 1.0e-6 or t0 > 1.0e8 or t1 > 1.0e8:
            return None
        D0, _lam0, X0, _h0, _mw0_check = cantera_transport_state(gas, p0, t0, y0)
        D1, _lam1, X1, _h1, _mw1_check = cantera_transport_state(gas, p1, t1, y1)
        D_face = 0.5*(D0 + D1)
        X_face_grad = (X1 - X0)/spacing
        mw_face = 0.5*(mw0 + mw1)
        rho_face = 0.5*(rho0 + rho1)
        Y_face = 0.5*(y0 + y1)
        J = rho_face*D_face*molecular_weights/mw_face*X_face_grad
        rho_vic = float(np.sum(J))
        J = J - rho_vic*Y_face
        J = min(ag0, ag1)*J
        residuals.append(float(np.sum(J)))
        finite_temperature.extend([t0, t1])
        return J, rho_vic

    for j in range(ny):
        for i in range(nx - 1):
            if not active_x[j, i]:
                continue
            spacing = float(x[i + 1] - x[i])
            try:
                result = compute_face((j, i), (j, i + 1), spacing)
            except Exception:
                result = None
            if result is None:
                nonfinite_temperature_count += 1
                continue
            # MFC source-flux sign convention: flux_src_vf(chem_k) -= J_k.
            Jx[:, j, i] = -result[0]
    for j in range(ny - 1):
        for i in range(nx):
            if not active_y[j, i]:
                continue
            spacing = float(y[j + 1] - y[j])
            try:
                result = compute_face((j, i), (j + 1, i), spacing)
            except Exception:
                result = None
            if result is None:
                nonfinite_temperature_count += 1
                continue
            Jy[:, j, i] = -result[0]
    nc12 = species.index("NC12H26")
    o2 = species.index("O2")
    out = {
        "x": x,
        "y": y,
        "x_face": 0.5*(x[:-1] + x[1:]),
        "y_face": 0.5*(y[:-1] + y[1:]),
        "alpha_liq": alpha_liq,
        "alpha_g": alpha_g,
        "rho_g_stored": rho_g_stored,
        "rho_g_intrinsic": rho_g_intrinsic,
        "eligible_cell": eligible,
        "active_x": active_x,
        "active_y": active_y,
        "J_NC12H26_x": Jx[nc12],
        "J_NC12H26_y": Jy[nc12],
        "J_O2_x": Jx[o2],
        "J_O2_y": Jy[o2],
        "J_total_x": np.sqrt(np.sum(Jx*Jx, axis=0)),
        "J_total_y": np.sqrt(np.sum(Jy*Jy, axis=0)),
    }
    sumY = np.sum(Y, axis=0)
    inactive_x_zero = bool(np.all(out["J_total_x"][~active_x] == 0.0)) if active_x.size else True
    inactive_y_zero = bool(np.all(out["J_total_y"][~active_y] == 0.0)) if active_y.size else True
    finite_temperature_array = np.asarray(finite_temperature, dtype=float)
    stats = {
        "sumY_min": float(np.nanmin(sumY)),
        "sumY_max": float(np.nanmax(sumY)),
        "sumY_max_abs_error": float(np.nanmax(np.abs(sumY - 1.0))),
        "sumY_violation_count": float(np.count_nonzero(np.abs(sumY - 1.0) > BOUNDS_TOL)),
        "finite_pressure_fraction": float(np.count_nonzero(np.isfinite(pressure))/pressure.size),
        "finite_temperature_count": float(len(finite_temperature)),
        "temperature_min": float(np.min(finite_temperature_array)) if finite_temperature_array.size else math.nan,
        "temperature_max": float(np.max(finite_temperature_array)) if finite_temperature_array.size else math.nan,
        "nonfinite_or_invalid_active_face_count": float(nonfinite_temperature_count),
        "zero_net_J_max_abs_residual": float(max((abs(v) for v in residuals), default=0.0)),
        "inactive_x_zero_flux": 1.0 if inactive_x_zero else 0.0,
        "inactive_y_zero_flux": 1.0 if inactive_y_zero else 0.0,
        "nonfinite_pressure_count": float(nonfinite_pressure_count),
    }
    if nonfinite_temperature_count:
        raise RuntimeError(f"invalid active-face thermochemical state count: {nonfinite_temperature_count}")
    return out, stats


def percentile99p9(arr: np.ndarray) -> float:
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, 99.9))


def maxabs(arr: np.ndarray) -> float:
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))



def x_face_to_cell_abs(jx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    accum = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)
    if jx.size:
        magx = np.abs(jx)
        accum[:, :-1] += magx
        count[:, :-1] += np.isfinite(magx)
        accum[:, 1:] += magx
        count[:, 1:] += np.isfinite(magx)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = accum/count
    out[count == 0.0] = np.nan
    return out


def y_face_to_cell_abs(jy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    accum = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)
    if jy.size:
        magy = np.abs(jy)
        accum[:-1, :] += magy
        count[:-1, :] += np.isfinite(magy)
        accum[1:, :] += magy
        count[1:, :] += np.isfinite(magy)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = accum/count
    out[count == 0.0] = np.nan
    return out

def plot_diffusion_diagnostic(save_dir: Path, save: int, time_us: float, data: dict[str, np.ndarray]) -> Path:
    x = data["x"]
    y = data["y"]
    alpha = data["alpha_liq"]
    active_cell = np.zeros_like(alpha, dtype=float)
    if data["active_x"].size:
        active_cell[:, :-1] += data["active_x"].astype(float)
        active_cell[:, 1:] += data["active_x"].astype(float)
    if data["active_y"].size:
        active_cell[:-1, :] += data["active_y"].astype(float)
        active_cell[1:, :] += data["active_y"].astype(float)
    active_cell = np.where(active_cell > 0.0, active_cell, np.nan)
    nc12_mag = np.hypot(
        x_face_to_cell_abs(data["J_NC12H26_x"], alpha.shape),
        y_face_to_cell_abs(data["J_NC12H26_y"], alpha.shape),
    )
    o2_mag = np.hypot(
        x_face_to_cell_abs(data["J_O2_x"], alpha.shape),
        y_face_to_cell_abs(data["J_O2_y"], alpha.shape),
    )
    total_mag = np.hypot(
        x_face_to_cell_abs(data["J_total_x"], alpha.shape),
        y_face_to_cell_abs(data["J_total_y"], alpha.shape),
    )
    fields = [
        (active_cell, "cells adjacent to active faces", "viridis"),
        (nc12_mag, "|NC12H26 diffusion source-flux|", "magma"),
        (o2_mag, "|O2 diffusion source-flux|", "magma"),
        (total_mag, "sqrt(sum_k J_k^2) diffusion source-flux", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, (arr, title, cmap) in zip(axes.ravel(), fields):
        vals = arr[np.isfinite(arr)]
        if vals.size and title != "cells adjacent to active faces":
            vmax = float(np.nanmax(vals))
            mesh = ax.pcolormesh(edges(x*1e6), edges(y*1e6), arr, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        else:
            mesh = ax.pcolormesh(edges(x*1e6), edges(y*1e6), arr, shading="auto", cmap=cmap)
        if np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
            ax.contour(x*1e6, y*1e6, alpha, levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(x*1e6, y*1e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, pad=0.01)
    for ax in axes[-1, :]:
        ax.set_xlabel("x [µm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [µm]")
    fig.suptitle(f"C3 diffusion diagnostics, save {save:06d}, t={time_us:.2f} µs\ninterior saved-state reconstruction")
    png = save_dir / f"diffusion_diagnostics_save_{save:06d}.png"
    fig.savefig(png, dpi=210)
    plt.close(fig)
    return png


def save_diffusion_npz(save_dir: Path, save: int, time_s: float, data: dict[str, np.ndarray]) -> Path:
    path = save_dir / f"diffusion_diagnostics_save_{save:06d}.npz"
    np.savez_compressed(
        path,
        x=data["x"],
        y=data["y"],
        x_face=data["x_face"],
        y_face=data["y_face"],
        active_x=data["active_x"],
        active_y=data["active_y"],
        J_NC12H26_x=data["J_NC12H26_x"],
        J_NC12H26_y=data["J_NC12H26_y"],
        J_O2_x=data["J_O2_x"],
        J_O2_y=data["J_O2_y"],
        J_total_x=data["J_total_x"],
        J_total_y=data["J_total_y"],
        alpha_g=data["alpha_g"],
        alpha_liq=data["alpha_liq"],
        rho_g_stored=data["rho_g_stored"],
        rho_g_intrinsic=data["rho_g_intrinsic"],
        eligible_cell=data["eligible_cell"],
        time=np.array(time_s),
        save=np.array(save),
        description=np.array("interior saved-state reconstruction; boundaries and RK stages excluded; fluxes use MFC source-flux sign"),
    )
    return path



def diffusion_read_mask_fields(run_dir: Path, save: int, fmap: DiffusionVariableMap) -> dict[str, dict]:
    return {
        "alpha_liq": read_mapped_field(run_dir, fmap.liquid_alpha, save),
        "alpha_vap": read_mapped_field(run_dir, fmap.vapor_alpha, save),
        "alpha_air": read_mapped_field(run_dir, fmap.air_alpha, save),
    }


def arrays_from_mask_fields(fields: dict[str, dict]) -> dict[str, np.ndarray]:
    x, y, alpha_liq = grid_required(fields["alpha_liq"], "alpha_liq")
    out: dict[str, np.ndarray] = {"x": x, "y": y, "alpha_liq": alpha_liq}
    for key in ["alpha_vap", "alpha_air"]:
        xx, yy, arr = grid_required(fields[key], key)
        if not same_grid(x, y, xx, yy):
            raise RuntimeError(f"field {key} grid does not match liquid-alpha grid")
        out[key] = arr
    alpha_g = out["alpha_vap"] + out["alpha_air"]
    eligible = (
        np.isfinite(alpha_g) & np.isfinite(alpha_liq) &
        (alpha_g >= 0.5) & (alpha_liq <= 0.5)
    )
    out["alpha_g"] = alpha_g
    out["eligible_cell"] = eligible
    out["active_x"] = eligible[:, :-1] & eligible[:, 1:]
    out["active_y"] = eligible[:-1, :] & eligible[1:, :]
    out["x_face"] = 0.5*(x[:-1] + x[1:])
    out["y_face"] = 0.5*(y[:-1] + y[1:])
    return out


def x_face_fraction_to_cell(active_x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    accum = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)
    if active_x.size:
        vals = active_x.astype(float)
        accum[:, :-1] += vals
        count[:, :-1] += 1.0
        accum[:, 1:] += vals
        count[:, 1:] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = accum/count
    out[count == 0.0] = np.nan
    return out


def y_face_fraction_to_cell(active_y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    accum = np.zeros(shape, dtype=float)
    count = np.zeros(shape, dtype=float)
    if active_y.size:
        vals = active_y.astype(float)
        accum[:-1, :] += vals
        count[:-1, :] += 1.0
        accum[1:, :] += vals
        count[1:, :] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = accum/count
    out[count == 0.0] = np.nan
    return out


def save_diffusion_mask_npz(save_dir: Path, save: int, time_s: float, data: dict[str, np.ndarray]) -> Path:
    path = save_dir / f"diffusion_mask_save_{save:06d}.npz"
    np.savez_compressed(
        path,
        x=data["x"],
        y=data["y"],
        x_face=data["x_face"],
        y_face=data["y_face"],
        alpha_g=data["alpha_g"],
        alpha_liq=data["alpha_liq"],
        eligible_cell=data["eligible_cell"],
        active_x=data["active_x"],
        active_y=data["active_y"],
        save=np.array(save),
        time=np.array(time_s),
        description=np.array("interior saved-state reconstruction; boundaries and RK stages excluded"),
    )
    return path


def plot_diffusion_mask(save_dir: Path, save: int, time_us: float, data: dict[str, np.ndarray]) -> Path:
    x = data["x"]
    y = data["y"]
    alpha = data["alpha_liq"]
    panels = [
        (data["eligible_cell"].astype(float), "eligible cells", "viridis"),
        (x_face_fraction_to_cell(data["active_x"], alpha.shape), "active x-face fraction projected to cells", "viridis"),
        (y_face_fraction_to_cell(data["active_y"], alpha.shape), "active y-face fraction projected to cells", "viridis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, (arr, title, cmap) in zip(axes, panels):
        mesh = ax.pcolormesh(edges(x*1e6), edges(y*1e6), arr, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        if np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
            ax.contour(x*1e6, y*1e6, alpha, levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(x*1e6, y*1e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, pad=0.01)
    axes[0].set_ylabel("y [µm]")
    fig.suptitle(f"C3 diffusion mask, save {save:06d}, t={time_us:.2f} µs\ninterior saved-state reconstruction")
    png = save_dir / f"diffusion_mask_save_{save:06d}.png"
    fig.savefig(png, dpi=210)
    plt.close(fig)
    return png


def export_diffusion_mask_only(c3_raw_case: Path, saves: list[int], output_dir: Path, dry_run_only: bool = False) -> None:
    all_species = species_names()
    case_path = c3_raw_case / "case.py"
    case = c3_case_dict(case_path) if case_path.is_file() else {}
    fmap = diffusion_variable_map(case or {"model_eqns": 3, "num_fluids": 3, "num_dims": 2}, all_species)
    print("diffusion mask-only argument check")
    print(f"  c3_raw_case: {c3_raw_case}")
    print(f"  output_dir: {output_dir}")
    print(f"  saves: {saves}")
    print("  mapping:")
    print(f"    alpha_liq={fmap.liquid_alpha} alpha_vap={fmap.vapor_alpha} alpha_air={fmap.air_alpha}")
    print("  no Cantera transport properties are imported or evaluated in this mode")
    if dry_run_only:
        print("  dry_run_only: no raw fields read")
        return
    if not (c3_raw_case / "D").is_dir():
        raise RuntimeError(f"missing raw D directory under {c3_raw_case}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for save in saves:
        save_dir = output_dir / "diffusion_mask_diagnostics" / f"save_{save:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fields = diffusion_read_mask_fields(c3_raw_case, save, fmap)
        data = arrays_from_mask_fields(fields)
        time_s = save*T_SAVE
        npz = save_diffusion_mask_npz(save_dir, save, time_s, data)
        png = plot_diffusion_mask(save_dir, save, time_s*1e6, data)
        cell_count = int(data["eligible_cell"].size)
        x_face_count = int(data["active_x"].size)
        y_face_count = int(data["active_y"].size)
        eligible_count = int(np.count_nonzero(data["eligible_cell"]))
        active_x_count = int(np.count_nonzero(data["active_x"]))
        active_y_count = int(np.count_nonzero(data["active_y"]))
        manifest.append({
            "save": save,
            "physical_time_s": time_s,
            "physical_time_us": time_s*1e6,
            "eligible_cell_fraction": eligible_count/cell_count if cell_count else math.nan,
            "active_x_face_fraction": active_x_count/x_face_count if x_face_count else math.nan,
            "active_y_face_fraction": active_y_count/y_face_count if y_face_count else math.nan,
            "eligible_cell_count": eligible_count,
            "active_x_face_count": active_x_count,
            "active_y_face_count": active_y_count,
            "scope": "interior saved-state reconstruction; boundaries and RK stages excluded",
            "npz": str(npz),
            "png": str(png),
        })
    write_csv(output_dir / "diffusion_mask_manifest.csv", manifest)


def export_diffusion_diagnostics(c3_raw_case: Path, saves: list[int], output_dir: Path, dry_run_only: bool = False) -> None:
    all_species = species_names()
    ensure_species_fields(all_species)
    case_path = c3_raw_case / "case.py"
    case = c3_case_dict(case_path) if case_path.is_file() else {}
    fmap = diffusion_variable_map(case or {"model_eqns": 3, "num_fluids": 3, "num_dims": 2}, all_species)
    mechanism = REPO / str(case.get("cantera_file", "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"))
    phase = str(case.get("cantera_phase", "yao_sk54"))
    print("diffusion diagnostics argument check")
    print(f"  c3_raw_case: {c3_raw_case}")
    print(f"  output_dir: {output_dir}")
    print(f"  saves: {saves}")
    print(f"  mechanism: {mechanism} exists={mechanism.is_file()}")
    print(f"  phase: {phase}")
    print("  mapping:")
    print(f"    alpha_liq={fmap.liquid_alpha} alpha_vap={fmap.vapor_alpha} alpha_air={fmap.air_alpha}")
    print(f"    rho_vap={fmap.vapor_alpha_rho} rho_air={fmap.air_alpha_rho} pressure={fmap.pressure}")
    print(f"    primitive species Y index = {fmap.species_prim_start} + SK54 species_id")
    print(f"    NC12H26 species_id={fmap.species_indices['NC12H26']} prim_index={fmap.species_prim_start + fmap.species_indices['NC12H26']}")
    print(f"    O2 species_id={fmap.species_indices['O2']} prim_index={fmap.species_prim_start + fmap.species_indices['O2']}")
    if dry_run_only:
        print("  dry_run_only: no raw fields or Cantera transport properties read")
        return
    if not mechanism.is_file():
        raise RuntimeError(f"missing mechanism file {mechanism}")
    if not (c3_raw_case / "D").is_dir():
        raise RuntimeError(f"missing raw D directory under {c3_raw_case}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for save in saves:
        save_dir = output_dir / "diffusion_diagnostics" / f"save_{save:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fields = diffusion_read_save_fields(c3_raw_case, save, fmap, all_species)
        arrays = arrays_from_diffusion_fields(fields, all_species)
        data, stats = reconstruct_diffusion_fluxes(arrays, all_species, mechanism, phase)
        time_s = save*T_SAVE
        npz = save_diffusion_npz(save_dir, save, time_s, data)
        png = plot_diffusion_diagnostic(save_dir, save, time_s*1e6, data)
        active_x_fraction = float(np.count_nonzero(data["active_x"])/data["active_x"].size) if data["active_x"].size else math.nan
        active_y_fraction = float(np.count_nonzero(data["active_y"])/data["active_y"].size) if data["active_y"].size else math.nan
        manifest.append({
            "save": save,
            "physical_time_s": time_s,
            "physical_time_us": time_s*1e6,
            "active_x_face_fraction": active_x_fraction,
            "active_y_face_fraction": active_y_fraction,
            "max_abs_J_NC12H26_x": maxabs(data["J_NC12H26_x"]),
            "max_abs_J_NC12H26_y": maxabs(data["J_NC12H26_y"]),
            "p999_abs_J_NC12H26_x": percentile99p9(data["J_NC12H26_x"]),
            "p999_abs_J_NC12H26_y": percentile99p9(data["J_NC12H26_y"]),
            "max_abs_J_O2_x": maxabs(data["J_O2_x"]),
            "max_abs_J_O2_y": maxabs(data["J_O2_y"]),
            "p999_abs_J_O2_x": percentile99p9(data["J_O2_x"]),
            "p999_abs_J_O2_y": percentile99p9(data["J_O2_y"]),
            "max_J_total_x": maxabs(data["J_total_x"]),
            "max_J_total_y": maxabs(data["J_total_y"]),
            "p999_J_total_x": percentile99p9(data["J_total_x"]),
            "p999_J_total_y": percentile99p9(data["J_total_y"]),
            "sumY_min": stats["sumY_min"],
            "sumY_max": stats["sumY_max"],
            "sumY_max_abs_error": stats["sumY_max_abs_error"],
            "sumY_violation_count": int(stats["sumY_violation_count"]),
            "finite_pressure_fraction": stats["finite_pressure_fraction"],
            "temperature_min": stats["temperature_min"],
            "temperature_max": stats["temperature_max"],
            "finite_temperature_evaluation_count": int(stats["finite_temperature_count"]),
            "invalid_active_face_property_count": int(stats["nonfinite_or_invalid_active_face_count"]),
            "zero_net_J_max_abs_residual": stats["zero_net_J_max_abs_residual"],
            "inactive_x_zero_flux": "T" if stats["inactive_x_zero_flux"] == 1.0 else "F",
            "inactive_y_zero_flux": "T" if stats["inactive_y_zero_flux"] == 1.0 else "F",
            "mechanism_path": str(mechanism),
            "transport_model": "Cantera mixture-averaged / MFC chem_params%transport_model=1",
            "scope": "interior saved-state reconstruction; boundaries and RK stages excluded",
            "npz": str(npz),
            "png": str(png),
        })
    write_csv(output_dir / "diffusion_diagnostic_manifest.csv", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths, common saves, and requested frame availability")
    parser.add_argument("--export-c3", action="store_true", help="Export compact C3 CSV/PNG/NPZ artifacts from the HPC C3 raw D/ output")
    parser.add_argument("--c3-export-out", type=Path, default=OUT / "c3_export", help="Output directory for --export-c3")
    parser.add_argument("--compare-from-c3-export", type=Path, help="Compare local ON/OFF raw data against a compact C3 export directory")
    parser.add_argument("--make-difference-frames", action="store_true", help="Generate C3 minus chemistry-ON/diffusion-OFF difference maps from a C3 export")
    parser.add_argument("--make-hydrodynamic-wake-frames", action="store_true", help="Generate ON_NO_DIFF vs C3 hydrodynamic wake frames from compact C3 export")
    parser.add_argument("--make-mixing-state-analysis", action="store_true", help="Generate local ON_NO_DIFF vs C3 parent-fuel/O2 mixing-state analysis")
    parser.add_argument("--c3-export", type=Path, help="C3 compact export directory for C3 export-based frame modes")
    parser.add_argument("--export-diffusion-diagnostics", action="store_true", help="Reconstruct C3 saved-state model-3 chemistry-diffusion masks and species fluxes")
    parser.add_argument("--export-diffusion-mask-only", action="store_true", help="Export C3 saved-state model-3 chemistry-diffusion active masks without Cantera")
    parser.add_argument("--export-fuel-inventory-budget", action="store_true", help="Export full-domain C1/C2/C3 dodecane fuel-inventory budgets")
    parser.add_argument("--export-c3-fuel-inventory", action="store_true", help="Export C3-only parent-dodecane inventory CSV")
    parser.add_argument("--c3-raw-case", type=Path, default=C3_RUN, help="Raw C3 case directory for diffusion exports")
    parser.add_argument("--saves", type=int, nargs="+", default=[30, 38], help="C3 save indices for diffusion exports")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Output directory for diffusion exports")
    parser.add_argument("--output-file", type=Path, default=C3_FUEL_OUTPUT, help="Output CSV for --export-c3-fuel-inventory")
    parser.add_argument("--diffusion-dry-run", action="store_true", help="Check diffusion diagnostic arguments/mapping without requiring HPC raw fields")
    parser.add_argument("--fuel-inventory-dry-run", action="store_true", help="Check fuel-inventory case paths, mappings, and common saves without exporting")
    parser.add_argument("--c3-fuel-inventory-dry-run", action="store_true", help="Check C3-only fuel inventory inputs without writing CSV")
    parser.add_argument("--hydrodynamic-wake-dry-run", action="store_true", help="Check hydrodynamic wake frame inputs without rendering")
    parser.add_argument("--mixing-state-dry-run", action="store_true", help="Check mixing-state inputs without rendering")
    parser.add_argument("--c3-fuel-inventory-save", type=int, help="Restrict --export-c3-fuel-inventory to one save for lightweight testing")
    args = parser.parse_args()
    modes = sum(bool(v) for v in [
        args.dry_run,
        args.export_c3,
        args.compare_from_c3_export,
        args.make_difference_frames,
        args.make_hydrodynamic_wake_frames,
        args.make_mixing_state_analysis,
        args.export_diffusion_diagnostics,
        args.export_diffusion_mask_only,
        args.export_fuel_inventory_budget,
        args.export_c3_fuel_inventory,
    ])
    if modes > 1:
        parser.error("choose only one primary mode")
    if args.dry_run:
        dry_run()
    elif args.export_c3:
        c3_export(args.c3_export_out)
    elif args.compare_from_c3_export:
        compare_from_c3_export(args.compare_from_c3_export)
    elif args.make_difference_frames:
        if args.c3_export is None:
            parser.error("--make-difference-frames requires --c3-export PATH")
        make_difference_frames(args.c3_export)
    elif args.make_hydrodynamic_wake_frames:
        make_hydrodynamic_wake_frames(args.c3_export or C3_COMPACT_EXPORT, args.hydrodynamic_wake_dry_run)
    elif args.make_mixing_state_analysis:
        make_mixing_state_analysis(args.c3_export or C3_COMPACT_EXPORT, args.mixing_state_dry_run)
    elif args.export_diffusion_diagnostics:
        export_diffusion_diagnostics(args.c3_raw_case, args.saves, args.output_dir, args.diffusion_dry_run)
    elif args.export_diffusion_mask_only:
        export_diffusion_mask_only(args.c3_raw_case, args.saves, args.output_dir, args.diffusion_dry_run)
    elif args.export_fuel_inventory_budget:
        export_fuel_inventory_budget(args.fuel_inventory_dry_run)
    elif args.export_c3_fuel_inventory:
        export_c3_fuel_inventory(
            args.c3_raw_case,
            args.output_file,
            args.c3_fuel_inventory_dry_run,
            args.c3_fuel_inventory_save,
        )
    else:
        if args.diffusion_dry_run:
            parser.error("--diffusion-dry-run requires --export-diffusion-diagnostics or --export-diffusion-mask-only")
        if args.fuel_inventory_dry_run:
            parser.error("--fuel-inventory-dry-run requires --export-fuel-inventory-budget")
        if args.c3_fuel_inventory_dry_run:
            parser.error("--c3-fuel-inventory-dry-run requires --export-c3-fuel-inventory")
        if args.hydrodynamic_wake_dry_run:
            parser.error("--hydrodynamic-wake-dry-run requires --make-hydrodynamic-wake-frames")
        if args.mixing_state_dry_run:
            parser.error("--mixing-state-dry-run requires --make-mixing-state-analysis")
        run_full()


if __name__ == "__main__":
    main()
