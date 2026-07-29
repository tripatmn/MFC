#!/usr/bin/env python3
"""Direct compact export for C4 parallel-I/O Lustre restart saves.

This is the small C4-focused path for completed ``parallel_io = T`` runs whose
saved states live in ``restart_data/lustre_<save>.dat``.  It reuses the direct
Lustre reader pattern from git commit 26f832d3
(``analyze_tanabe_restart_lustre.py``): numeric Lustre files are read with
``numpy.fromfile(..., dtype=float64)`` and reshaped as MFC conservative fields.
The reader is generalized here using the C4 case dimensions and
``lustre_x_cb.dat``/``lustre_y_cb.dat``.

No MFC post_process, temporary cases, MPI rank arguments, or shell-outs are
used by this script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-c4-export-simple")

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RUN_ROOT = REPO / "runs/shockdroplet_air_sk54/alpha_source_fix"
DEFAULT_CASE = SCRIPT.parent
DEFAULT_OUT = SCRIPT.parent / "c4_export"
C3_HELPER = RUN_ROOT / "C3_vs_chemistry_ON_OFF_0_to_2us/make_c3_on_off_comparison.py"
T_SAVE = 5.0e-8
GAS_MASS_FLOOR = 1.0e-8
BOUNDS_TOL = 1.0e-12
REL_DENOM_EPS = 1.0e-300
LUSTRE_RE = re.compile(r"^lustre_(\d+)\.dat$")

# 0-based conservative variable positions for model_eqns=3, 3 fluids, 2-D.
VAR_ALPHA_RHO_LIQ = 0
VAR_ALPHA_RHO_VAP = 1
VAR_ALPHA_RHO_AIR = 2
VAR_MOM_X = 3
VAR_MOM_Y = 4
VAR_ENERGY = 5
VAR_ALPHA_LIQ = 6
VAR_ALPHA_VAP = 7
VAR_ALPHA_AIR = 8
VAR_EINT_LIQ = 9
VAR_EINT_VAP = 10
VAR_EINT_AIR = 11
SPECIES_OFFSET = 12

SPECIES_OF_INTEREST = ["NC12H26", "O2", "OH", "HO2", "H2O2", "H2O", "CO", "CO2"]
FRAME_TIMES_US = [0.50, 0.85, 1.00, 1.50, 1.90, 1.95, 2.00, 3.00, 4.00, 5.00]
FRAME_VARIABLES = [
    "pressure",
    "valid_gas_temperature",
    "liquid_alpha",
    "NC12H26",
    "O2",
    "OH",
    "HO2",
    "H2O2",
    "H2O",
    "CO",
    "CO2",
]
TREND_GROUPS = [
    ("liquid_mass_loss_fraction", "Liquid mass-loss fraction", "liquid_mass_loss_fraction.png"),
    ("vapor_mass", r"Vapor mass [kg m$^{-1}$]", "vapor_mass.png"),
    ("liquid_alpha_area_m2", r"Liquid area [m$^2$]", "liquid_area.png"),
    ("valid_gas_Tmax_K", "Valid-gas Tmax proxy [K]", "valid_gas_Tmax.png"),
    ("valid_gas_Tmean_K", "Valid-gas mean temperature proxy [K]", "valid_gas_Tmean.png"),
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

# These match the gas constants used by the existing raw shock-droplet analyzer.
AIR_GAMMA = 1.4
AIR_CV = 739.0
VAPOR_GAMMA = 1.025
VAPOR_CV = 1956.0
R_AIR = (AIR_GAMMA - 1.0) * AIR_CV
R_VAPOR = (VAPOR_GAMMA - 1.0) * VAPOR_CV


@dataclass(frozen=True)
class C4Grid:
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    area: np.ndarray


@dataclass(frozen=True)
class C4State:
    save: int
    time_s: float
    arr: np.ndarray
    grid: C4Grid


def load_c3_helper():
    spec = importlib.util.spec_from_file_location("c3_export_helper_for_c4", C3_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import helper: {C3_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_c3_helper()


def ff(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def load_case(case_dir: Path) -> dict:
    case_py = case_dir / "case.py"
    ns = runpy.run_path(str(case_py), run_name="__c4_case__")
    case = ns.get("case")
    if not isinstance(case, dict):
        raise RuntimeError(f"{case_py} did not define a case dictionary")
    return dict(case)


def species_names() -> list[str]:
    return base.species_names()


def species_index_0based(all_species: list[str], name: str) -> int:
    try:
        return SPECIES_OFFSET + all_species.index(name)
    except ValueError as exc:
        raise RuntimeError(f"species {name!r} not found in SK54 mechanism") from exc


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


def find_lustre_files(case_dir: Path) -> list[tuple[int, Path]]:
    restart_dir = case_dir / "restart_data"
    if not restart_dir.is_dir():
        raise RuntimeError(f"restart_data directory not found: {restart_dir}")
    found: list[tuple[int, Path]] = []
    for path in restart_dir.glob("lustre_*.dat"):
        match = LUSTRE_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    if not found:
        raise RuntimeError(f"no numeric restart_data/lustre_<save>.dat files found under {restart_dir}")
    return found


def selected_files(files: list[tuple[int, Path]], saves: list[int] | None) -> list[tuple[int, Path]]:
    if saves is None:
        return files
    by_save = dict(files)
    missing = [save for save in saves if save not in by_save]
    if missing:
        raise RuntimeError(f"requested saves unavailable: {missing}; available range {files[0][0]}..{files[-1][0]}")
    return [(save, by_save[save]) for save in saves]


def read_boundaries(path: Path, expected: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"required {name} boundary file not found: {path}")
    vals = np.fromfile(path, dtype=np.float64)
    if vals.size != expected:
        raise RuntimeError(f"{path} has {vals.size} float64 values; expected {expected}")
    if not np.all(np.isfinite(vals)) or np.any(np.diff(vals) <= 0.0):
        raise RuntimeError(f"{path} is nonfinite or non-monotone")
    return vals


def make_grid(case_dir: Path, case: dict) -> C4Grid:
    m = int(case["m"])
    n = int(case["n"])
    restart_dir = case_dir / "restart_data"
    x_cb = read_boundaries(restart_dir / "lustre_x_cb.dat", m + 2, "x")
    y_cb = read_boundaries(restart_dir / "lustre_y_cb.dat", n + 2, "y")
    x = 0.5 * (x_cb[:-1] + x_cb[1:])
    y = 0.5 * (y_cb[:-1] + y_cb[1:])
    dx = np.diff(x_cb)
    dy = np.diff(y_cb)
    area = dy[:, None] * dx[None, :]
    return C4Grid(x=x, y=y, dx=dx, dy=dy, area=area)


def expected_nvars(case: dict, all_species: list[str]) -> int:
    num_fluids = int(case.get("num_fluids", 3))
    num_dims = 2
    # model_eqns=3 layout: partial densities + momenta + total energy + alphas
    # + phasic internal energies + global chemistry species block.
    return num_fluids + num_dims + 1 + num_fluids + num_fluids + len(all_species)


def read_lustre_file(path: Path, case: dict, grid: C4Grid, all_species: list[str]) -> np.ndarray:
    nvars = expected_nvars(case, all_species)
    nx = grid.x.size
    ny = grid.y.size
    expected = nvars * ny * nx
    raw_values = np.fromfile(path, dtype=np.float64)
    if raw_values.size != expected:
        raise RuntimeError(f"{path} has {raw_values.size} float64 values; expected {expected} = {nvars}*{ny}*{nx}")
    # Reused reader convention from analyze_tanabe_restart_lustre.py.
    return raw_values.reshape((nvars, ny, nx))


def finite_integral(values: np.ndarray, area: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(values)
    if not np.any(mask):
        return math.nan, int(values.size)
    return float(np.sum(values[mask] * area[mask])), int(values.size - np.count_nonzero(mask))


def gas_temperature_proxy(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arho_v = arr[VAR_ALPHA_RHO_VAP]
    arho_a = arr[VAR_ALPHA_RHO_AIR]
    eint_v = arr[VAR_EINT_VAP]
    eint_a = arr[VAR_EINT_AIR]
    rho_g = arho_v + arho_a
    t_v = np.full_like(rho_g, np.nan, dtype=float)
    t_a = np.full_like(rho_g, np.nan, dtype=float)
    mask_v = np.isfinite(eint_v) & np.isfinite(arho_v) & (arho_v > 0.0)
    mask_a = np.isfinite(eint_a) & np.isfinite(arho_a) & (arho_a > 0.0)
    t_v[mask_v] = eint_v[mask_v] / (arho_v[mask_v] * VAPOR_CV)
    t_a[mask_a] = eint_a[mask_a] / (arho_a[mask_a] * AIR_CV)
    weighted = np.zeros_like(rho_g, dtype=float)
    weighted[mask_v] += arho_v[mask_v] * t_v[mask_v]
    weighted[mask_a] += arho_a[mask_a] * t_a[mask_a]
    t_g = np.full_like(rho_g, np.nan, dtype=float)
    mask = np.isfinite(weighted) & np.isfinite(rho_g) & (rho_g > 0.0)
    t_g[mask] = weighted[mask] / rho_g[mask]
    return t_g, rho_g


def valid_gas_mask(arr: np.ndarray, t_gas: np.ndarray) -> np.ndarray:
    alpha_liq = arr[VAR_ALPHA_LIQ]
    alpha_g = arr[VAR_ALPHA_VAP] + arr[VAR_ALPHA_AIR]
    gas_mass = arr[VAR_ALPHA_RHO_VAP] + arr[VAR_ALPHA_RHO_AIR]
    return (
        np.isfinite(alpha_liq)
        & np.isfinite(alpha_g)
        & np.isfinite(gas_mass)
        & np.isfinite(t_gas)
        & (alpha_g > 0.5)
        & (gas_mass > GAS_MASS_FLOOR)
        & (alpha_liq < 0.5)
    )


def species_stack(arr: np.ndarray, all_species: list[str]) -> np.ndarray:
    return arr[SPECIES_OFFSET:SPECIES_OFFSET + len(all_species)]


def compute_metrics(state: C4State, all_species: list[str]) -> dict:
    arr = state.arr
    area = state.grid.area
    t_gas, gas_mass = gas_temperature_proxy(arr)
    valid = valid_gas_mask(arr, t_gas)
    row: dict = {
        "case": "C4_EXPORT",
        "save": state.save,
        "global_save": state.save,
        "raw_step": state.save,
        "origin": "lustre_direct",
        "time_s": state.time_s,
        "time_us": state.time_s * 1.0e6,
        "source_layout": "parallel_lustre_direct",
        "temperature_note": "valid-gas temperature proxy from phasic internal energies in restart save",
        "cell_area_min_m2": float(np.nanmin(area)),
        "cell_area_max_m2": float(np.nanmax(area)),
    }
    liquid_mass, liquid_nonfinite = finite_integral(arr[VAR_ALPHA_RHO_LIQ], area)
    vapor_mass, vapor_nonfinite = finite_integral(arr[VAR_ALPHA_RHO_VAP], area)
    row["liquid_mass"] = liquid_mass
    row["vapor_mass"] = vapor_mass
    row["nonfinite_liquid_alpha_rho_cells"] = liquid_nonfinite
    row["nonfinite_vapor_alpha_rho_cells"] = vapor_nonfinite
    liquid_mask = np.isfinite(arr[VAR_ALPHA_LIQ]) & (arr[VAR_ALPHA_LIQ] >= 0.5)
    row["liquid_alpha_area_m2"] = float(np.sum(area[liquid_mask])) if np.any(liquid_mask) else 0.0
    if np.any(liquid_mask):
        yy, xx = np.where(liquid_mask)
        row["liquid_extent_x_m"] = float(state.grid.x[xx].max() - state.grid.x[xx].min())
        row["liquid_extent_y_m"] = float(state.grid.y[yy].max() - state.grid.y[yy].min())
    else:
        row["liquid_extent_x_m"] = math.nan
        row["liquid_extent_y_m"] = math.nan
    tvals = t_gas[valid]
    row["valid_gas_cell_count"] = int(tvals.size)
    row["valid_gas_Tmax_K"] = float(np.nanmax(tvals)) if tvals.size else math.nan
    row["valid_gas_Tmean_K"] = float(np.nanmean(tvals)) if tvals.size else math.nan
    for species in SPECIES_OF_INTEREST:
        idx = species_index_0based(all_species, species)
        mass, _nonfinite = finite_integral(arr[idx], area)
        row[f"integrated_{species}"] = mass
        vals = arr[idx]
        finite = vals[np.isfinite(vals)]
        row[f"max_{species}"] = float(np.max(finite)) if finite.size else math.nan
    stack = species_stack(arr, all_species)
    gas_valid = np.isfinite(gas_mass) & (gas_mass > GAS_MASS_FLOOR) & valid
    if np.any(gas_valid):
        y = np.full_like(stack, np.nan, dtype=float)
        y[:, gas_valid] = stack[:, gas_valid] / gas_mass[gas_valid]
        sum_rhoY = np.nansum(stack[:, gas_valid], axis=0)
        sumY = sum_rhoY / gas_mass[gas_valid]
        row["species_negative_rhoY_count"] = int(np.count_nonzero(stack[:, gas_valid] < -BOUNDS_TOL))
        row["species_negative_Y_count"] = int(np.count_nonzero(y[:, gas_valid] < -BOUNDS_TOL))
        row["species_Y_above_one_count"] = int(np.count_nonzero(y[:, gas_valid] > 1.0 + BOUNDS_TOL))
        row["species_sumY_min"] = float(np.nanmin(sumY))
        row["species_sumY_max"] = float(np.nanmax(sumY))
        row["species_sumY_max_abs_error"] = float(np.nanmax(np.abs(sumY - 1.0)))
        row["species_sumY_violation_count"] = int(np.count_nonzero(np.abs(sumY - 1.0) > BOUNDS_TOL))
        min_flat = int(np.nanargmin(stack[:, gas_valid]))
        species_i, _cell_i = np.unravel_index(min_flat, stack[:, gas_valid].shape)
        row["species_min_rhoY"] = float(np.nanmin(stack[:, gas_valid]))
        row["species_min_rhoY_name"] = all_species[species_i]
        max_y_flat = int(np.nanargmax(y[:, gas_valid]))
        max_y_species_i, _ = np.unravel_index(max_y_flat, y[:, gas_valid].shape)
        row["species_max_Y"] = float(np.nanmax(y[:, gas_valid]))
        row["species_max_Y_name"] = all_species[max_y_species_i]
        row["species_min_Y"] = float(np.nanmin(y[:, gas_valid]))
        row["integrated_valid_gas_mass"] = float(np.sum(gas_mass[gas_valid] * area[gas_valid]))
        row["integrated_valid_gas_sum_rhoY"] = float(np.sum(sum_rhoY * area[gas_valid]))
        row["mass_consistency_integrated_diff"] = row["integrated_valid_gas_sum_rhoY"] - row["integrated_valid_gas_mass"]
        row["mass_consistency_relative_error"] = (
            row["mass_consistency_integrated_diff"] / row["integrated_valid_gas_mass"]
            if abs(row["integrated_valid_gas_mass"]) > REL_DENOM_EPS else math.nan
        )
        row["species_diagnostics_available"] = "T"
        row["species_diagnostics_missing_reason"] = ""
    else:
        for key in [
            "species_negative_rhoY_count",
            "species_negative_Y_count",
            "species_Y_above_one_count",
            "species_sumY_min",
            "species_sumY_max",
            "species_sumY_max_abs_error",
            "species_sumY_violation_count",
            "species_min_rhoY",
            "species_min_Y",
            "species_max_Y",
            "integrated_valid_gas_mass",
            "integrated_valid_gas_sum_rhoY",
            "mass_consistency_integrated_diff",
            "mass_consistency_relative_error",
        ]:
            row[key] = math.nan
        row["species_min_rhoY_name"] = ""
        row["species_max_Y_name"] = ""
        row["species_diagnostics_available"] = "F"
        row["species_diagnostics_missing_reason"] = "no valid gas cells"
    row["available"] = "T"
    return row


def add_liquid_loss(rows: list[dict]) -> None:
    base_mass = next((ff(r.get("liquid_mass")) for r in rows if math.isfinite(ff(r.get("liquid_mass")))), math.nan)
    for row in rows:
        liquid = ff(row.get("liquid_mass"))
        if math.isfinite(base_mass) and abs(base_mass) > REL_DENOM_EPS and math.isfinite(liquid):
            row["liquid_mass_loss"] = base_mass - liquid
            row["liquid_mass_loss_fraction"] = (base_mass - liquid) / base_mass
        else:
            row["liquid_mass_loss"] = math.nan
            row["liquid_mass_loss_fraction"] = math.nan


def fuel_inventory_rows(rows: list[dict]) -> list[dict]:
    initial = next(
        (
            ff(row.get("integrated_NC12H26")) + ff(row.get("liquid_mass"))
            for row in rows
            if math.isfinite(ff(row.get("integrated_NC12H26"))) and math.isfinite(ff(row.get("liquid_mass")))
        ),
        math.nan,
    )
    out: list[dict] = []
    for row in rows:
        gas = ff(row.get("integrated_NC12H26"))
        liquid = ff(row.get("liquid_mass"))
        combined = gas + liquid if math.isfinite(gas) and math.isfinite(liquid) else math.nan
        denom_ok = math.isfinite(initial) and abs(initial) > REL_DENOM_EPS
        out.append({
            "save": row.get("global_save", row.get("save")),
            "raw_step": row.get("raw_step"),
            "time_s": row.get("time_s"),
            "time_us": row.get("time_us"),
            "gas_parent_fuel_mass": gas,
            "liquid_dodecane_mass": liquid,
            "combined_parent_dodecane_mass": combined,
            "gas_fraction_of_initial_combined": gas / initial if denom_ok and math.isfinite(gas) else math.nan,
            "liquid_fraction_of_initial_combined": liquid / initial if denom_ok and math.isfinite(liquid) else math.nan,
            "combined_fraction_of_initial": combined / initial if denom_ok and math.isfinite(combined) else math.nan,
        })
    return out


def field_array(state: C4State, variable: str, all_species: list[str]) -> tuple[np.ndarray | None, np.ndarray, str, str, np.ndarray]:
    arr = state.arr
    t_gas, _gas_mass = gas_temperature_proxy(arr)
    valid = valid_gas_mask(arr, t_gas)
    alpha = arr[VAR_ALPHA_LIQ]
    if variable == "pressure":
        return None, valid, "Pressure unavailable in conservative Lustre restart save", "Pa", alpha
    if variable == "valid_gas_temperature":
        return np.where(valid, t_gas, np.nan), valid, "Valid-gas temperature proxy", "K", alpha
    if variable == "liquid_alpha":
        return alpha, np.isfinite(alpha), "Liquid alpha", "dimensionless", alpha
    if variable in SPECIES_OF_INTEREST:
        values = arr[species_index_0based(all_species, variable)]
        return values, np.isfinite(values), f"{variable} partial density", r"kg m$^{-3}$", alpha
    raise ValueError(variable)


def edges(values: np.ndarray) -> np.ndarray:
    return base.edges(values)


def choose_scale(values: np.ndarray, variable: str) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"available": False, "log": False, "vmin": 0.0, "vmax": 1.0}
    if variable == "liquid_alpha":
        return {"available": True, "log": False, "vmin": 0.0, "vmax": 1.0}
    use_log = False
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if variable in SPECIES_OF_INTEREST:
        pos = finite[finite > 0.0]
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


def save_npz(path: Path, state: C4State, variable: str, values: np.ndarray, valid_mask: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=state.grid.x,
        y=state.grid.y,
        values=values,
        time=np.array(state.time_s),
        save=np.array(state.save),
        raw_step=np.array(state.save),
        variable=np.array(variable),
        valid_mask=valid_mask,
        mask_description=np.array("valid gas mask" if variable == "valid_gas_temperature" else "finite full-domain field mask"),
        alpha_liq=alpha,
    )


def draw_field(path: Path, state: C4State, variable: str, values: np.ndarray, alpha: np.ndarray, label: str, unit: str) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    scale = choose_scale(values, variable)
    if not scale["available"]:
        return False
    fig, ax = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    if scale["log"]:
        mesh = ax.pcolormesh(
            edges(state.grid.x * 1.0e6),
            edges(state.grid.y * 1.0e6),
            np.where(values > 0.0, values, np.nan),
            shading="auto",
            cmap="magma",
            norm=colors.LogNorm(vmin=scale["vmin"], vmax=scale["vmax"]),
        )
    else:
        mesh = ax.pcolormesh(
            edges(state.grid.x * 1.0e6),
            edges(state.grid.y * 1.0e6),
            values,
            shading="auto",
            cmap="viridis",
            vmin=scale["vmin"],
            vmax=scale["vmax"],
        )
    if np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
        ax.contour(state.grid.x * 1.0e6, state.grid.y * 1.0e6, alpha, levels=[0.5], colors="white", linewidths=1.1)
        ax.contour(state.grid.x * 1.0e6, state.grid.y * 1.0e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(label if unit == "dimensionless" else f"{label} [{unit}]")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"C4 {variable}, save {state.save}, t={state.time_s * 1e6:.2f} µs")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return True


def nearest_save(target_us: float, saves: list[int]) -> int | None:
    if not saves:
        return None
    return min(saves, key=lambda save: abs(save * T_SAVE * 1.0e6 - target_us))


def make_trend_plots(rows: list[dict], out_dir: Path) -> None:
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


def export_case(args: argparse.Namespace) -> None:
    case_dir = args.raw_case.resolve()
    out = args.c3_export_out.resolve()
    case = load_case(case_dir)
    all_species = species_names()
    files = selected_files(find_lustre_files(case_dir), args.saves)
    if not files:
        raise RuntimeError("zero saves selected")
    grid = make_grid(case_dir, case)
    rows: list[dict] = []
    time_rows: list[dict] = []
    frame_manifest: list[dict] = []
    frame_saves = {save for save, _ in files} if args.saves else {
        save for target in FRAME_TIMES_US for save in [nearest_save(target, [s for s, _ in files])] if save is not None
    }
    for save, path in files:
        print(f"[info] reading save {save}: {path}")
        state = C4State(save=save, time_s=save * T_SAVE, arr=read_lustre_file(path, case, grid, all_species), grid=grid)
        rows.append(compute_metrics(state, all_species))
        time_rows.append({
            "case": "C4_EXPORT",
            "save": save,
            "raw_step": save,
            "time_s": state.time_s,
            "time_us": state.time_s * 1.0e6,
            "source_layout": "restart_data/lustre direct",
            "source_file": str(path),
        })
        if save in frame_saves:
            for variable in FRAME_VARIABLES:
                values, valid_mask, label, unit, alpha = field_array(state, variable, all_species)
                row = {
                    "save": save,
                    "raw_step": save,
                    "actual_time_us": state.time_s * 1.0e6,
                    "variable": variable,
                }
                if values is None:
                    row.update({"status": "skipped", "reason": label})
                    frame_manifest.append(row)
                    continue
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    row.update({"status": "skipped", "reason": "no finite values"})
                    frame_manifest.append(row)
                    continue
                stem = f"{variable}_save{save:03d}_t{state.time_s * 1e6:.2f}us"
                npz = out / "field_data" / variable / f"{stem}.npz"
                png = out / "field_frames" / variable / f"{stem}.png"
                save_npz(npz, state, variable, values, valid_mask, alpha)
                png_ok = draw_field(png, state, variable, values, alpha, label, unit)
                row.update({
                    "status": "written" if png_ok else "partial",
                    "reason": "" if png_ok else "PNG write failed",
                    "npz": str(npz.relative_to(out)),
                    "png": str(png.relative_to(out)) if png_ok else "",
                })
                frame_manifest.append(row)
    add_liquid_loss(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "c4_timeseries.csv", rows)
    write_csv(out / "c4_time_map.csv", time_rows)
    write_csv(out / "c4_species_bounds_by_save.csv", [
        {key: row.get(key, "") for key in [
            "global_save", "raw_step", "time_s", "time_us",
            "species_negative_rhoY_count", "species_negative_Y_count", "species_Y_above_one_count",
            "species_sumY_min", "species_sumY_max", "species_sumY_max_abs_error",
            "species_sumY_violation_count", "species_min_rhoY", "species_min_rhoY_name",
            "species_min_Y", "species_max_Y", "species_max_Y_name",
            "species_diagnostics_available", "species_diagnostics_missing_reason",
        ]}
        for row in rows
    ])
    write_csv(out / "c4_mass_consistency_by_save.csv", [
        {key: row.get(key, "") for key in [
            "global_save", "raw_step", "time_s", "time_us",
            "integrated_valid_gas_mass", "integrated_valid_gas_sum_rhoY",
            "mass_consistency_integrated_diff", "mass_consistency_relative_error",
        ]}
        for row in rows
    ])
    write_csv(out / "c4_fuel_inventory.csv", fuel_inventory_rows(rows))
    write_csv(out / "c4_frame_manifest.csv", frame_manifest)
    make_trend_plots(rows, out / "trend_plots")
    summary = [
        "C4 direct-Lustre compact export",
        "================================",
        "",
        f"raw_case: {case_dir}",
        f"output_dir: {out}",
        f"saves: {[save for save, _ in files]}",
        f"timeseries_rows: {len(rows)}",
        f"frame_manifest_rows: {len(frame_manifest)}",
        "",
        "Reader reused from git history commit 26f832d3: find_lustre_files/read_lustre_file pattern from analyze_tanabe_restart_lustre.py.",
        "Pressure is unavailable in conservative Lustre restart saves and is skipped.",
        "Valid-gas temperature is a proxy reconstructed from phasic internal energies, as in the prior Lustre analyzer.",
        "No MFC post_process, temporary cases, or shell-outs are used.",
    ]
    (out / "c4_export_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"[done] wrote C4 compact export to {out}")


def dry_run(args: argparse.Namespace) -> None:
    case_dir = args.raw_case.resolve()
    case = load_case(case_dir)
    all_species = species_names()
    print(f"repo: {REPO}")
    print(f"raw_case: {case_dir}")
    print(f"output: {args.c3_export_out.resolve()}")
    print("prior reader: commit 26f832d3 analyze_tanabe_restart_lustre.py")
    print(f"case m,n,p: {case.get('m')}, {case.get('n')}, {case.get('p')}")
    print(f"expected nvars: {expected_nvars(case, all_species)}")
    print("verified mappings:")
    print("  alpha_rho_liquid = conservative variable 1 -> lustre index 0")
    print("  vapor_alpha_rho  = conservative variable 2 -> lustre index 1")
    print("  air_alpha_rho    = conservative variable 3 -> lustre index 2")
    print("  alpha_liquid     = conservative variable 7 -> lustre index 6")
    print("  alpha_vapor      = conservative variable 8 -> lustre index 7")
    print("  alpha_air        = conservative variable 9 -> lustre index 8")
    print(f"  rhoY_NC12H26     = conservative variable {species_index_0based(all_species, 'NC12H26') + 1}")
    print(f"  rhoY_O2          = conservative variable {species_index_0based(all_species, 'O2') + 1}")
    files = find_lustre_files(case_dir)
    selected = selected_files(files, args.saves)
    print(f"discovered_saves_count: {len(files)}")
    print(f"discovered_save_range: {files[0][0]}..{files[-1][0]}")
    print(f"selected_saves: {[save for save, _ in selected]}")
    grid = make_grid(case_dir, case)
    print(f"grid: nx={grid.x.size}, ny={grid.y.size}, dx=[{grid.dx.min():.8e},{grid.dx.max():.8e}], dy=[{grid.dy.min():.8e},{grid.dy.max():.8e}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-c4", action="store_true", help="Export compact C4 artifacts from restart_data/lustre saves")
    parser.add_argument("--export-c3", action="store_true", help="Backward-compatible alias for --export-c4")
    parser.add_argument("--raw-case", type=Path, default=DEFAULT_CASE, help="C4 case directory containing restart_data/lustre_*.dat")
    parser.add_argument("--c3-export-out", "--c4-export-out", dest="c3_export_out", type=Path, default=DEFAULT_OUT, help="Compact output directory")
    parser.add_argument("--saves", nargs="*", type=int, help="Optional explicit save indices, e.g. --saves 0 1")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery/mapping information without exporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.export_c4 or args.export_c3:
        export_case(args)
        return
    raise SystemExit("No action requested. Use --export-c4, --export-c3, or --dry-run.")


if __name__ == "__main__":
    main()
