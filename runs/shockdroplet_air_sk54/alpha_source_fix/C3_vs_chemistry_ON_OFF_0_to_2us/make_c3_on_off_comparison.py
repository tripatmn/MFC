#!/usr/bin/env python3
"""Compare chemistry/diffusion variants for the Mach-3 SK54 shock-droplet case.

This standalone driver intentionally uses the raw ASCII ``D/`` saves and the
same lightweight raw-field readers as the existing shock-droplet analyzers.  It
does not depend on the previously generated ON/OFF or combined-analysis output
files, though it follows their plotting conventions.
"""

from __future__ import annotations

import argparse
import csv
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


def load_npz_grid(path: Path) -> dict:
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
    parser.add_argument("--c3-export", type=Path, help="C3 compact export directory for --make-difference-frames")
    parser.add_argument("--export-diffusion-diagnostics", action="store_true", help="Reconstruct C3 saved-state model-3 chemistry-diffusion masks and species fluxes")
    parser.add_argument("--export-diffusion-mask-only", action="store_true", help="Export C3 saved-state model-3 chemistry-diffusion active masks without Cantera")
    parser.add_argument("--c3-raw-case", type=Path, default=C3_RUN, help="Raw C3 case directory for diffusion exports")
    parser.add_argument("--saves", type=int, nargs="+", default=[30, 38], help="C3 save indices for diffusion exports")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Output directory for diffusion exports")
    parser.add_argument("--diffusion-dry-run", action="store_true", help="Check diffusion diagnostic arguments/mapping without requiring HPC raw fields")
    args = parser.parse_args()
    modes = sum(bool(v) for v in [
        args.dry_run,
        args.export_c3,
        args.compare_from_c3_export,
        args.make_difference_frames,
        args.export_diffusion_diagnostics,
        args.export_diffusion_mask_only,
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
    elif args.export_diffusion_diagnostics:
        export_diffusion_diagnostics(args.c3_raw_case, args.saves, args.output_dir, args.diffusion_dry_run)
    elif args.export_diffusion_mask_only:
        export_diffusion_mask_only(args.c3_raw_case, args.saves, args.output_dir, args.diffusion_dry_run)
    else:
        if args.diffusion_dry_run:
            parser.error("--diffusion-dry-run requires --export-diffusion-diagnostics or --export-diffusion-mask-only")
        run_full()


if __name__ == "__main__":
    main()
