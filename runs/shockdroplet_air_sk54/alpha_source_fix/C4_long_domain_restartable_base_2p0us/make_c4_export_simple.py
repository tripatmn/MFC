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
import math
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
SGM_EPS = 1.0e-16
PRESSURE_FLOOR_PA = 1.0e2
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
RHOY_VARIABLES = [f"rhoY_{species}" for species in SPECIES_OF_INTEREST]
FRAME_TIMES_US = [0.50, 0.85, 1.00, 1.50, 1.90, 1.95, 2.00, 3.00, 4.00, 5.00]
FRAME_VARIABLES = [
    "pressure",
    "valid_gas_temperature",
    "T_intrinsic_gas",
    "T_proxy_phasic_internal_energy",
    "liquid_alpha",
    *SPECIES_OF_INTEREST,
    *RHOY_VARIABLES,
]
TREND_GROUPS = [
    ("liquid_mass_loss_fraction", "Liquid mass-loss fraction", "liquid_mass_loss_fraction.png"),
    ("vapor_mass", r"Vapor mass [kg m$^{-1}$]", "vapor_mass.png"),
    ("liquid_alpha_area_m2", r"Liquid area [m$^2$]", "liquid_area.png"),
    ("pressure_valid_gas_max_Pa", "Valid-gas pressure max [Pa]", "pressure_valid_gas_max.png"),
    ("pressure_valid_gas_min_Pa", "Valid-gas pressure min [Pa]", "pressure_valid_gas_min.png"),
    ("valid_gas_Tmax_K", "Valid-gas Tmax, MFC primitive [K]", "valid_gas_Tmax.png"),
    ("valid_gas_Tmean_K", "Valid-gas mean temperature, MFC primitive [K]", "valid_gas_Tmean.png"),
    ("valid_gas_T_intrinsic_max_K", "Valid-gas intrinsic Tmax [K]", "valid_gas_T_intrinsic_max.png"),
    ("valid_gas_T_proxy_max_K", "Valid-gas proxy Tmax [K]", "valid_gas_T_proxy_max.png"),
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

# Retained only for the explicitly labeled phasic-internal-energy proxy.
AIR_CV = 739.0
VAPOR_CV = 1956.0


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


@dataclass(frozen=True)
class GasConfig:
    fluid_ids: tuple[int, ...]
    alpha_indices: tuple[int, ...]
    alpha_rho_indices: tuple[int, ...]


@dataclass(frozen=True)
class EosConfig:
    gammas: np.ndarray
    pi_infs: np.ndarray
    qvs: np.ndarray
    mpp_lim: bool


@dataclass(frozen=True)
class ChemistryConstants:
    molecular_weights: np.ndarray
    gas_constant: float
    mechanism: Path
    phase: str
    source: str


@dataclass(frozen=True)
class DerivedFields:
    rho: np.ndarray
    rho_g: np.ndarray
    alpha_g: np.ndarray
    pressure: np.ndarray
    y_raw: np.ndarray
    y_primitive: np.ndarray
    sum_y_raw: np.ndarray
    T_mfc_primitive: np.ndarray
    T_intrinsic_gas: np.ndarray
    T_proxy_phasic_internal_energy: np.ndarray
    valid_gas: np.ndarray


@dataclass
class ScaleStats:
    min: float = math.inf
    max: float = -math.inf
    pos_min: float = math.inf
    pos_max: float = -math.inf
    count: int = 0

    def add(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self.min = min(self.min, float(np.min(finite)))
        self.max = max(self.max, float(np.max(finite)))
        positive = finite[finite > 0.0]
        if positive.size:
            self.pos_min = min(self.pos_min, float(np.min(positive)))
            self.pos_max = max(self.pos_max, float(np.max(positive)))
        self.count += int(finite.size)


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


def truthy(value: Any) -> bool:
    return str(value).strip().upper() in {"T", "TRUE", "1", ".TRUE."}


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


def species_id_1based(all_species: list[str], name: str) -> int:
    return all_species.index(name) + 1


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


def gas_config_from_case(case: dict) -> GasConfig:
    num_fluids = int(case.get("num_fluids", 3))
    num_gas = int(case.get("chem_gas_num_fluids", 0) or 0)
    if num_gas > 0:
        ids = tuple(int(case[f"chem_gas_fluid_ids({idx})"]) for idx in range(1, num_gas + 1))
    else:
        ids = (int(case.get("chem_gas_fluid_id", 1)),)
    if not ids or any(fluid_id < 1 or fluid_id > num_fluids for fluid_id in ids):
        raise RuntimeError(f"invalid chemistry gas fluid ids {ids}; num_fluids={num_fluids}")
    alpha_rho = tuple(fluid_id - 1 for fluid_id in ids)
    alpha = tuple(VAR_ALPHA_LIQ + fluid_id - 1 for fluid_id in ids)
    return GasConfig(fluid_ids=ids, alpha_indices=alpha, alpha_rho_indices=alpha_rho)


def eos_config_from_case(case: dict) -> EosConfig:
    num_fluids = int(case.get("num_fluids", 3))
    gammas = np.array([float(case[f"fluid_pp({idx})%gamma"]) for idx in range(1, num_fluids + 1)], dtype=float)
    pi_infs = np.array([float(case[f"fluid_pp({idx})%pi_inf"]) for idx in range(1, num_fluids + 1)], dtype=float)
    qvs = np.array([float(case.get(f"fluid_pp({idx})%qv", 0.0)) for idx in range(1, num_fluids + 1)], dtype=float)
    return EosConfig(gammas=gammas, pi_infs=pi_infs, qvs=qvs, mpp_lim=truthy(case.get("mpp_lim", "F")))


def require_cantera_constants(case: dict, all_species: list[str]) -> ChemistryConstants:
    try:
        import cantera as ct  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cantera is required for exact T_mfc_primitive and T_intrinsic_gas export. "
            "Install/load the project Cantera environment or run only --dry-run."
        ) from exc
    mechanism = REPO / str(case.get("cantera_file", "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"))
    phase = str(case.get("cantera_phase", "yao_sk54"))
    gas = ct.Solution(str(mechanism), phase)
    if list(gas.species_names) != list(all_species):
        raise RuntimeError("Cantera mechanism species order does not match parsed SK54 species order")
    return ChemistryConstants(
        molecular_weights=np.asarray(gas.molecular_weights, dtype=float),
        gas_constant=float(ct.gas_constant),
        mechanism=mechanism,
        phase=phase,
        source="Cantera",
    )


def cantera_status(case: dict, all_species: list[str]) -> str:
    try:
        constants = require_cantera_constants(case, all_species)
    except Exception as exc:
        return f"unavailable: {exc}"
    return f"available: {constants.mechanism} phase={constants.phase}, species={len(constants.molecular_weights)}"


def finite_integral(values: np.ndarray, area: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(values)
    if not np.any(mask):
        return math.nan, int(values.size)
    return float(np.sum(values[mask] * area[mask])), int(values.size - np.count_nonzero(mask))


def species_stack(arr: np.ndarray, all_species: list[str]) -> np.ndarray:
    return arr[SPECIES_OFFSET:SPECIES_OFFSET + len(all_species)]


def chemistry_gas_fields(arr: np.ndarray, gas_cfg: GasConfig) -> tuple[np.ndarray, np.ndarray]:
    rho_g = np.zeros_like(arr[VAR_ALPHA_RHO_LIQ], dtype=float)
    alpha_g = np.zeros_like(arr[VAR_ALPHA_LIQ], dtype=float)
    for idx in gas_cfg.alpha_rho_indices:
        rho_g = rho_g + arr[idx]
    for idx in gas_cfg.alpha_indices:
        alpha_g = alpha_g + arr[idx]
    return rho_g, alpha_g


def gas_temperature_proxy(arr: np.ndarray) -> np.ndarray:
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
    return t_g


def mfc_style_primitive_y(stack: np.ndarray, rho_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho_ref = np.maximum(rho_g, SGM_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_raw = stack / rho_g[None, :, :]
        y_prim = stack / rho_ref[None, :, :]
    y_raw[:, ~(np.isfinite(rho_g) & (rho_g > 0.0))] = np.nan
    y_prim = np.clip(y_prim, 0.0, 1.0)
    sum_prim = np.sum(y_prim, axis=0)
    normalize = np.isfinite(sum_prim) & (sum_prim > 1.0)
    y_prim[:, normalize] = y_prim[:, normalize] / sum_prim[normalize]
    return y_raw, y_prim


def mixture_molecular_weight(y_primitive: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sum(y_primitive / molecular_weights[:, None, None], axis=0)
        mw = 1.0 / denom
    mw[~(np.isfinite(mw) & (mw > 0.0))] = np.nan
    return mw


def reconstruct_pressure(arr: np.ndarray, eos: EosConfig) -> tuple[np.ndarray, np.ndarray]:
    alpha_rho = arr[0:len(eos.gammas)].astype(float, copy=True)
    alpha = arr[VAR_ALPHA_LIQ:VAR_ALPHA_LIQ + len(eos.gammas)].astype(float, copy=True)
    if eos.mpp_lim:
        alpha_rho = np.maximum(alpha_rho, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_sum = np.sum(alpha, axis=0)
        good_alpha = np.isfinite(alpha_sum) & (alpha_sum > SGM_EPS)
        alpha[:, good_alpha] = alpha[:, good_alpha] / alpha_sum[good_alpha]
    rho = np.sum(alpha_rho, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = arr[VAR_MOM_X] / rho
        v = arr[VAR_MOM_Y] / rho
    dyn_p = 0.5 * (arr[VAR_MOM_X] * u + arr[VAR_MOM_Y] * v)
    gamma_mix = np.sum(alpha * eos.gammas[:, None, None], axis=0)
    pi_inf_mix = np.sum(alpha * eos.pi_infs[:, None, None], axis=0)
    qv_mix = np.sum(alpha_rho * eos.qvs[:, None, None], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure = (arr[VAR_ENERGY] - dyn_p - pi_inf_mix - qv_mix) / gamma_mix
    pressure = np.where(np.isfinite(pressure), np.maximum(pressure, PRESSURE_FLOOR_PA), np.nan)
    return pressure, rho


def derive_fields(
    state: C4State,
    all_species: list[str],
    gas_cfg: GasConfig,
    eos: EosConfig,
    chem: ChemistryConstants,
) -> DerivedFields:
    arr = state.arr
    rho_g, alpha_g = chemistry_gas_fields(arr, gas_cfg)
    pressure, rho = reconstruct_pressure(arr, eos)
    stack = species_stack(arr, all_species)
    y_raw, y_primitive = mfc_style_primitive_y(stack, rho_g)
    sum_y_raw = np.nansum(y_raw, axis=0)
    mw = mixture_molecular_weight(y_primitive, chem.molecular_weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_mfc = pressure * mw / (chem.gas_constant * rho_g)
        rho_g_intrinsic = rho_g / alpha_g
        T_intrinsic = pressure * mw / (chem.gas_constant * rho_g_intrinsic)
    T_mfc[~(np.isfinite(T_mfc) & (T_mfc > 0.0))] = np.nan
    T_intrinsic[~(np.isfinite(T_intrinsic) & (T_intrinsic > 0.0))] = np.nan
    T_proxy = gas_temperature_proxy(arr)
    valid = (
        np.isfinite(arr[VAR_ALPHA_LIQ])
        & np.isfinite(alpha_g)
        & np.isfinite(rho_g)
        & np.isfinite(T_mfc)
        & (alpha_g > 0.5)
        & (rho_g > GAS_MASS_FLOOR)
        & (arr[VAR_ALPHA_LIQ] < 0.5)
    )
    return DerivedFields(
        rho=rho,
        rho_g=rho_g,
        alpha_g=alpha_g,
        pressure=pressure,
        y_raw=y_raw,
        y_primitive=y_primitive,
        sum_y_raw=sum_y_raw,
        T_mfc_primitive=T_mfc,
        T_intrinsic_gas=T_intrinsic,
        T_proxy_phasic_internal_energy=T_proxy,
        valid_gas=valid,
    )


def compute_metrics(state: C4State, all_species: list[str], gas_cfg: GasConfig, eos: EosConfig, chem: ChemistryConstants) -> dict:
    arr = state.arr
    area = state.grid.area
    derived = derive_fields(state, all_species, gas_cfg, eos, chem)
    valid = derived.valid_gas
    row: dict = {
        "case": "C4_EXPORT",
        "save": state.save,
        "global_save": state.save,
        "raw_step": state.save,
        "origin": "lustre_direct",
        "time_s": state.time_s,
        "time_us": state.time_s * 1.0e6,
        "source_layout": "parallel_lustre_direct",
        "temperature_note": "T_mfc_primitive reconstructed from pressure, MW(Y_primitive), and configured chemistry gas density; proxy retained separately",
        "chemistry_gas_fluid_ids": ",".join(str(i) for i in gas_cfg.fluid_ids),
        "pressure_floor_Pa": PRESSURE_FLOOR_PA,
        "cell_area_min_m2": float(np.nanmin(area)),
        "cell_area_max_m2": float(np.nanmax(area)),
    }
    liquid_mass, liquid_nonfinite = finite_integral(arr[VAR_ALPHA_RHO_LIQ], area)
    vapor_mass, vapor_nonfinite = finite_integral(arr[VAR_ALPHA_RHO_VAP], area)
    row["liquid_mass"] = liquid_mass
    row["vapor_mass"] = vapor_mass
    row["nonfinite_liquid_alpha_rho_cells"] = liquid_nonfinite
    row["nonfinite_vapor_alpha_rho_cells"] = vapor_nonfinite
    gas_mass, gas_nonfinite = finite_integral(derived.rho_g, area)
    row["integrated_chemistry_gas_mass_full_domain"] = gas_mass
    row["nonfinite_chemistry_gas_mass_cells"] = gas_nonfinite
    liquid_mask = np.isfinite(arr[VAR_ALPHA_LIQ]) & (arr[VAR_ALPHA_LIQ] >= 0.5)
    row["liquid_alpha_area_m2"] = float(np.sum(area[liquid_mask])) if np.any(liquid_mask) else 0.0
    if np.any(liquid_mask):
        yy, xx = np.where(liquid_mask)
        row["liquid_extent_x_m"] = float(state.grid.x[xx].max() - state.grid.x[xx].min())
        row["liquid_extent_y_m"] = float(state.grid.y[yy].max() - state.grid.y[yy].min())
    else:
        row["liquid_extent_x_m"] = math.nan
        row["liquid_extent_y_m"] = math.nan
    row["valid_gas_cell_count"] = int(np.count_nonzero(valid))
    for label, values in [
        ("pressure_valid_gas", derived.pressure),
        ("valid_gas_T_mfc", derived.T_mfc_primitive),
        ("valid_gas_T_intrinsic", derived.T_intrinsic_gas),
        ("valid_gas_T_proxy", derived.T_proxy_phasic_internal_energy),
    ]:
        vals = values[valid]
        row[f"{label}_min_{'Pa' if 'pressure' in label else 'K'}"] = float(np.nanmin(vals)) if vals.size else math.nan
        row[f"{label}_max_{'Pa' if 'pressure' in label else 'K'}"] = float(np.nanmax(vals)) if vals.size else math.nan
        row[f"{label}_mean_{'Pa' if 'pressure' in label else 'K'}"] = float(np.nanmean(vals)) if vals.size else math.nan
    row["valid_gas_Tmax_K"] = row["valid_gas_T_mfc_max_K"]
    row["valid_gas_Tmean_K"] = row["valid_gas_T_mfc_mean_K"]
    for species in SPECIES_OF_INTEREST:
        sidx = all_species.index(species)
        cidx = SPECIES_OFFSET + sidx
        mass, _nonfinite = finite_integral(arr[cidx], area)
        row[f"integrated_{species}"] = mass
        vals = arr[cidx]
        finite = vals[np.isfinite(vals)]
        row[f"max_rhoY_{species}"] = float(np.max(finite)) if finite.size else math.nan
        yvals = derived.y_raw[sidx][valid]
        row[f"max_Y_raw_{species}"] = float(np.nanmax(yvals)) if yvals.size else math.nan
        row[f"min_Y_raw_{species}"] = float(np.nanmin(yvals)) if yvals.size else math.nan
        row[f"max_{species}"] = row[f"max_Y_raw_{species}"]
    stack = species_stack(arr, all_species)
    gas_valid = np.isfinite(derived.rho_g) & (derived.rho_g > GAS_MASS_FLOOR) & valid
    if np.any(gas_valid):
        y = derived.y_raw[:, gas_valid]
        sum_rhoY = np.sum(stack[:, gas_valid], axis=0)
        sumY = sum_rhoY / derived.rho_g[gas_valid]
        row["species_negative_rhoY_count"] = int(np.count_nonzero(stack[:, gas_valid] < -BOUNDS_TOL))
        row["species_negative_Y_count"] = int(np.count_nonzero(y < -BOUNDS_TOL))
        row["species_Y_above_one_count"] = int(np.count_nonzero(y > 1.0 + BOUNDS_TOL))
        row["species_sumY_min"] = float(np.nanmin(sumY))
        row["species_sumY_max"] = float(np.nanmax(sumY))
        row["species_sumY_max_abs_error"] = float(np.nanmax(np.abs(sumY - 1.0)))
        row["species_sumY_violation_count"] = int(np.count_nonzero(np.abs(sumY - 1.0) > BOUNDS_TOL))
        min_flat = int(np.nanargmin(stack[:, gas_valid]))
        species_i, _cell_i = np.unravel_index(min_flat, stack[:, gas_valid].shape)
        row["species_min_rhoY"] = float(np.nanmin(stack[:, gas_valid]))
        row["species_min_rhoY_name"] = all_species[species_i]
        max_y_flat = int(np.nanargmax(y))
        max_y_species_i, _ = np.unravel_index(max_y_flat, y.shape)
        row["species_max_Y"] = float(np.nanmax(y))
        row["species_max_Y_name"] = all_species[max_y_species_i]
        row["species_min_Y"] = float(np.nanmin(y))
        row["integrated_valid_gas_mass"] = float(np.sum(derived.rho_g[gas_valid] * area[gas_valid]))
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


def field_array(
    state: C4State,
    variable: str,
    all_species: list[str],
    derived: DerivedFields,
) -> tuple[np.ndarray | None, np.ndarray, str, str, np.ndarray, str]:
    arr = state.arr
    valid = derived.valid_gas
    alpha = arr[VAR_ALPHA_LIQ]
    if variable == "pressure":
        return derived.pressure, np.isfinite(derived.pressure), "Pressure reconstructed from model-3 conservative EOS", "Pa", alpha, "finite reconstructed pressure"
    if variable in {"valid_gas_temperature", "T_mfc_primitive"}:
        return (
            np.where(valid, derived.T_mfc_primitive, np.nan),
            valid,
            "T_mfc_primitive: p*MW(Y_primitive)/(R*rho_g)",
            "K",
            alpha,
            "valid gas mask",
        )
    if variable == "T_intrinsic_gas":
        return (
            np.where(valid, derived.T_intrinsic_gas, np.nan),
            valid,
            "T_intrinsic_gas: p*MW(Y_primitive)/(R*(rho_g/alpha_g))",
            "K",
            alpha,
            "valid gas mask",
        )
    if variable == "T_proxy_phasic_internal_energy":
        return (
            np.where(valid, derived.T_proxy_phasic_internal_energy, np.nan),
            valid,
            "Proxy gas temperature from phasic internal energies",
            "K",
            alpha,
            "valid gas mask; proxy quantity",
        )
    if variable == "liquid_alpha":
        return alpha, np.isfinite(alpha), "Liquid alpha", "dimensionless", alpha, "finite full-domain field mask"
    if variable.startswith("rhoY_"):
        species = variable[5:]
        values = arr[species_index_0based(all_species, species)]
        return values, np.isfinite(values), f"rhoY_{species} conservative partial density", r"kg m$^{-3}$", alpha, "finite full-domain rhoY mask"
    if variable in SPECIES_OF_INTEREST:
        sidx = all_species.index(variable)
        values = np.where(valid, derived.y_raw[sidx], np.nan)
        return values, valid, f"Y_{variable} raw mass fraction", "dimensionless", alpha, "valid gas mask; raw rhoY/rho_g"
    raise ValueError(variable)


def edges(values: np.ndarray) -> np.ndarray:
    return base.edges(values)


def choose_scale_from_stats(variable: str, stats: ScaleStats | None) -> dict | None:
    if stats is None or stats.count == 0:
        return None
    if variable == "liquid_alpha":
        return {"available": True, "log": False, "vmin": 0.0, "vmax": 1.0, "shared": True}
    use_log = False
    vmin = stats.min
    vmax = stats.max
    if variable in SPECIES_OF_INTEREST or variable.startswith("rhoY_"):
        if math.isfinite(stats.pos_min) and math.isfinite(stats.pos_max):
            if stats.pos_max / max(stats.pos_min, 1.0e-300) >= 1.0e4 and stats.pos_max > 0.0:
                use_log = True
                vmin = max(stats.pos_min, stats.pos_max * 1.0e-10)
                vmax = stats.pos_max
            else:
                vmin = min(0.0, vmin)
                vmax = stats.pos_max
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return {"available": True, "log": use_log, "vmin": vmin, "vmax": vmax, "shared": True}


def choose_scale(values: np.ndarray, variable: str, scale_override: dict | None = None) -> dict:
    if scale_override is not None:
        return scale_override
    stats = ScaleStats()
    stats.add(values)
    scale = choose_scale_from_stats(variable, stats)
    if scale is None:
        return {"available": False, "log": False, "vmin": 0.0, "vmax": 1.0, "shared": False}
    scale["shared"] = False
    return scale


def save_npz(
    path: Path,
    state: C4State,
    variable: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    alpha: np.ndarray,
    mask_description: str,
) -> None:
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
        mask_description=np.array(mask_description),
        alpha_liq=alpha,
    )


def draw_field(
    path: Path,
    state: C4State,
    variable: str,
    values: np.ndarray,
    alpha: np.ndarray,
    label: str,
    unit: str,
    scale_override: dict | None = None,
) -> tuple[bool, dict]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False, {"available": False, "log": False, "vmin": math.nan, "vmax": math.nan, "shared": False}
    scale = choose_scale(values, variable, scale_override)
    if not scale["available"]:
        return False, scale
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
    if np.any(np.isfinite(alpha)) and np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
        ax.contour(state.grid.x * 1.0e6, state.grid.y * 1.0e6, alpha, levels=[0.5], colors="white", linewidths=1.1)
        ax.contour(state.grid.x * 1.0e6, state.grid.y * 1.0e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(label if unit == "dimensionless" else f"{label} [{unit}]")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    shared = "shared scale" if scale.get("shared") else "frame scale"
    ax.set_title(f"C4 {variable}, save {state.save}, t={state.time_s * 1e6:.2f} µs ({shared})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return True, scale


def nearest_save(target_us: float, saves: list[int]) -> int | None:
    if not saves:
        return None
    return min(saves, key=lambda save: abs(save * T_SAVE * 1.0e6 - target_us))


def collect_frame_scales(
    files: list[tuple[int, Path]],
    frame_saves: set[int],
    case: dict,
    grid: C4Grid,
    all_species: list[str],
    gas_cfg: GasConfig,
    eos: EosConfig,
    chem: ChemistryConstants,
) -> dict[str, dict]:
    by_save = dict(files)
    stats = {variable: ScaleStats() for variable in FRAME_VARIABLES}
    for save in sorted(frame_saves):
        if save not in by_save:
            continue
        state = C4State(save=save, time_s=save * T_SAVE, arr=read_lustre_file(by_save[save], case, grid, all_species), grid=grid)
        derived = derive_fields(state, all_species, gas_cfg, eos, chem)
        for variable in FRAME_VARIABLES:
            values, _valid_mask, _label, _unit, _alpha, _mask_desc = field_array(state, variable, all_species, derived)
            if values is not None:
                stats[variable].add(values)
    return {
        variable: scale
        for variable, stat in stats.items()
        if (scale := choose_scale_from_stats(variable, stat)) is not None
    }


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
    gas_cfg = gas_config_from_case(case)
    eos = eos_config_from_case(case)
    chem = require_cantera_constants(case, all_species)
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
    shared_scales = collect_frame_scales(files, frame_saves, case, grid, all_species, gas_cfg, eos, chem)
    for save, path in files:
        print(f"[info] reading save {save}: {path}")
        state = C4State(save=save, time_s=save * T_SAVE, arr=read_lustre_file(path, case, grid, all_species), grid=grid)
        derived = derive_fields(state, all_species, gas_cfg, eos, chem)
        rows.append(compute_metrics(state, all_species, gas_cfg, eos, chem))
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
                values, valid_mask, label, unit, alpha, mask_desc = field_array(state, variable, all_species, derived)
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
                save_npz(npz, state, variable, values, valid_mask, alpha, mask_desc)
                png_ok, scale = draw_field(png, state, variable, values, alpha, label, unit, shared_scales.get(variable))
                row.update({
                    "status": "written" if png_ok else "partial",
                    "reason": "" if png_ok else "PNG write failed",
                    "npz": str(npz.relative_to(out)),
                    "png": str(png.relative_to(out)) if png_ok else "",
                    "scale_shared": "T" if scale.get("shared") else "F",
                    "scale_log": "T" if scale.get("log") else "F",
                    "scale_vmin": scale.get("vmin", math.nan),
                    "scale_vmax": scale.get("vmax", math.nan),
                    "mask_description": mask_desc,
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
        f"chemistry_gas_fluid_ids: {gas_cfg.fluid_ids}",
        f"Cantera mechanism: {chem.mechanism}",
        f"Cantera phase: {chem.phase}",
        "",
        "Reader reused from git history commit 26f832d3: find_lustre_files/read_lustre_file pattern from analyze_tanabe_restart_lustre.py.",
        "Pressure is exactly reconstructed from the model-3 conservative EOS using case fluid gamma/pi_inf/qv parameters and MFC's 100 Pa pressure floor.",
        "T_mfc_primitive is reconstructed as p*MW(Y_primitive)/(R*rho_g), where Y_primitive follows MFC clip-to-[0,1] and normalize-only-if-sum>1 behavior.",
        "T_intrinsic_gas uses p*MW(Y_primitive)/(R*(rho_g/alpha_g)) for gas-phase interpretation in diffuse-interface cells.",
        "T_proxy_phasic_internal_energy is retained as a labeled proxy only.",
        "Bare species frame variables are raw mass fractions Y_k=rhoY_k/rho_g; rhoY_<species> frame variables retain conservative partial-density exports.",
        "No MFC post_process, temporary cases, or shell-outs are used.",
    ]
    (out / "c4_export_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"[done] wrote C4 compact export to {out}")


def dry_run(args: argparse.Namespace) -> None:
    case_dir = args.raw_case.resolve()
    case = load_case(case_dir)
    all_species = species_names()
    gas_cfg = gas_config_from_case(case)
    eos = eos_config_from_case(case)
    print(f"repo: {REPO}")
    print(f"raw_case: {case_dir}")
    print(f"output: {args.c3_export_out.resolve()}")
    print("prior reader: commit 26f832d3 analyze_tanabe_restart_lustre.py")
    print(f"case m,n,p: {case.get('m')}, {case.get('n')}, {case.get('p')}")
    print(f"model_eqns: {case.get('model_eqns')} num_fluids: {case.get('num_fluids')}")
    print(f"expected nvars: {expected_nvars(case, all_species)}")
    print(f"chemistry gas fluid ids: {gas_cfg.fluid_ids}")
    print(f"chemistry gas alpha indices 1-based conservative: {[idx + 1 for idx in gas_cfg.alpha_indices]}")
    print(f"chemistry gas alpha_rho indices 1-based conservative: {[idx + 1 for idx in gas_cfg.alpha_rho_indices]}")
    print(f"mpp_lim: {eos.mpp_lim}")
    print("fluid EOS parameters:")
    for i, (gamma, pi_inf, qv) in enumerate(zip(eos.gammas, eos.pi_infs, eos.qvs), start=1):
        print(f"  fluid {i}: gamma={gamma:.16g} pi_inf={pi_inf:.16g} qv={qv:.16g}")
    print("verified mappings:")
    print("  alpha_rho_liquid = conservative variable 1 -> lustre index 0")
    print("  vapor_alpha_rho  = conservative variable 2 -> lustre index 1")
    print("  air_alpha_rho    = conservative variable 3 -> lustre index 2")
    print("  momentum_x       = conservative variable 4 -> lustre index 3")
    print("  momentum_y       = conservative variable 5 -> lustre index 4")
    print("  total_energy     = conservative variable 6 -> lustre index 5")
    print("  alpha_liquid     = conservative variable 7 -> lustre index 6")
    print("  alpha_vapor      = conservative variable 8 -> lustre index 7")
    print("  alpha_air        = conservative variable 9 -> lustre index 8")
    for species in SPECIES_OF_INTEREST:
        print(
            f"  rhoY_{species:<8} = species id {species_id_1based(all_species, species):2d} "
            f"-> conservative variable {species_index_0based(all_species, species) + 1}"
        )
    print(f"Cantera exact-temperature support: {cantera_status(case, all_species)}")
    try:
        files = find_lustre_files(case_dir)
    except RuntimeError as exc:
        print(f"restart_data discovery: unavailable in this environment ({exc})")
        files = []
    if files:
        selected = selected_files(files, args.saves)
        print(f"discovered_saves_count: {len(files)}")
        print(f"discovered_save_range: {files[0][0]}..{files[-1][0]}")
        print(f"selected_saves: {[save for save, _ in selected]}")
        try:
            grid = make_grid(case_dir, case)
        except RuntimeError as exc:
            print(f"grid: unavailable ({exc})")
        else:
            print(
                f"grid: nx={grid.x.size}, ny={grid.y.size}, "
                f"dx=[{grid.dx.min():.8e},{grid.dx.max():.8e}], "
                f"dy=[{grid.dy.min():.8e},{grid.dy.max():.8e}]"
            )
    else:
        print("discovered_saves_count: 0 (dry-run still verified case parsing and variable mappings)")


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
