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


def valid_gas_temperature(fields: dict[str, dict]) -> dict:
    temp = raw.reconstruct_temperature(fields, gas_mass_threshold=None)
    masks = gas_metrics.mask_context(fields, GAS_MASS_FLOOR)
    vals = {
        key: value
        for key, value in temp.get("values", {}).items()
        if key in masks["valid_gas_thermo"] and math.isfinite(value)
    }
    return {
        "available": bool(vals),
        "values": vals,
        "stats": raw.stats_from_values(vals, available=bool(vals)),
    }


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
    for case in CASE_ORDER:
        rr = [r for r in rows if r["case"] == case and r.get("available") == "T" and math.isfinite(ff(r.get(key)))]
        ax.plot([ff(r["time_us"]) for r in rr], [ff(r[key]) for r in rr], "o-", ms=3, lw=1.3, label=CASE_SHORT[case])
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
        for case in CASE_ORDER:
            for key in keys:
                rr = [r for r in rows if r["case"] == case and r.get("available") == "T" and math.isfinite(ff(r.get(key)))]
                ax.plot(
                    [ff(r["time_us"]) for r in rr],
                    [ff(r[key]) for r in rr],
                    marker="o",
                    ms=2.5,
                    lw=1.1,
                    label=f"{CASE_SHORT[case]} {key.replace('integrated_', '')}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths, common saves, and requested frame availability")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run_full()


if __name__ == "__main__":
    main()
