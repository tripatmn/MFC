#!/usr/bin/env python3
"""Build advisor-ready raw shock-droplet plots, frames, and videos.

The script reads MFC raw ``D/`` and/or ``p_all`` output directly.  It uses the
same flat raw-reader helper path as ``analyze_shockdroplet_air_sk54.py`` and
``analyze_shockdroplet_air_sk54_gas_metrics.py`` when those files are present.
When ``D/`` is incomplete but hierarchical binary ``p_all/p*/<save>`` output is
available, it reads ``q_cons_vf*.dat`` directly.  It does not call MFC
post_process, ``mfc.sh``, restart/Lustre readers, or any pre-exported
``field_data`` folder.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-advisor-package")

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
RUN_ROOT = REPO / "runs/shockdroplet_air_sk54/alpha_source_fix"
DEFAULT_GAS_MASS_FLOOR = 1.0e-8
PRESSURE_FLOOR_PA = 100.0
R_UNIVERSAL_KJ_PER_KMOL_K = 8314.46261815324
SGM_EPS = 1.0e-16
REL_EPS = 1.0e-300

# 0-based model_eqns=3 conservative layout for three fluids, 2-D.
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

SPECIES_OF_INTEREST = ["NC12H26", "OH", "CO2", "H2O", "HO2", "H2O2", "O2"]
CONTOUR_FIELDS = ["valid_gas_temperature", "Y_NC12H26", "Y_OH", "Y_CO2", "Y_H2O"]
INTEGRATED_SPECIES = ["NC12H26", "OH", "CO2", "H2O", "HO2", "H2O2", "O2"]
OPTION_A_ZOOM_TMIN_US = 4.8
OPTION_A_ZOOM_TMAX_US = 6.3
DEFAULT_ZOOM_XLIM_UM = (4600.0, 5600.0)
DEFAULT_ZOOM_YLIM_UM = (100.0, 520.0)

FIELD_CONFIG = {
    "valid_gas_temperature": {
        "short": "temperature",
        "label": "T",
        "pretty": "Valid-gas temperature",
        "unit": "K",
        "scale": "linear",
        "vmin": 850.0,
        "vmax": 2200.0,
        "cmap": "inferno",
    },
    "Y_NC12H26": {
        "short": "nc12h26",
        "label": "Y_NC12H26",
        "pretty": "NC12H26 mass fraction",
        "unit": "-",
        "scale": "log",
        "vmin": 1.0e-8,
        "vmax": 1.0,
        "cmap": "magma",
    },
    "Y_OH": {
        "short": "oh",
        "label": "Y_OH",
        "pretty": "OH mass fraction",
        "unit": "-",
        "scale": "log",
        "vmin": 1.0e-12,
        "vmax": 1.0e-4,
        "cmap": "magma",
    },
    "Y_CO2": {
        "short": "co2",
        "label": "Y_CO2",
        "pretty": "CO2 mass fraction",
        "unit": "-",
        "scale": "log",
        "vmin": 1.0e-12,
        "vmax": 1.0e-3,
        "cmap": "magma",
    },
    "Y_H2O": {
        "short": "h2o",
        "label": "Y_H2O",
        "pretty": "H2O mass fraction",
        "unit": "-",
        "scale": "log",
        "vmin": 1.0e-10,
        "vmax": 1.0e-2,
        "cmap": "magma",
    },
}


@dataclass(frozen=True)
class RenderJob:
    field: str
    tmin_us: float | None
    tmax_us: float | None


@dataclass
class State:
    step: int
    save_index: int
    time_s: float
    time_source: str
    source_layout: str
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    arrays: dict[str, np.ndarray]
    y_arrays: dict[str, np.ndarray]
    rhoY_arrays: dict[str, np.ndarray]
    valid_gas: np.ndarray
    alpha_liq: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-label", default="shock-droplet")
    parser.add_argument("--option-a", action="store_true", help="T all saves; Y_NC12H26/Y_OH zoom over 4.8-6.3 us")
    parser.add_argument("--fields", nargs="+", choices=sorted(CONTOUR_FIELDS), help="Explicit fields for non-option-A mode")
    parser.add_argument("--tmin-us", type=float, default=None)
    parser.add_argument("--tmax-us", type=float, default=None)
    parser.add_argument("--zoom-xlim-um", type=float, nargs=2, metavar=("XMIN", "XMAX"), default=DEFAULT_ZOOM_XLIM_UM)
    parser.add_argument("--zoom-ylim-um", type=float, nargs=2, metavar=("YMIN", "YMAX"), default=DEFAULT_ZOOM_YLIM_UM)
    parser.add_argument("--selected-times-us", type=float, nargs="*", default=[])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected frames")
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--no-full-domain", action="store_true")
    parser.add_argument("--gas-mass-floor", type=float, default=DEFAULT_GAS_MASS_FLOOR)
    parser.add_argument("--skip-scalars", action="store_true", help="Render frames/manifest/README without scalar_timeseries")
    parser.add_argument("--skip-trends", action="store_true", help="Skip trend plot generation")
    parser.add_argument("--debug", action="store_true", help="Print per-frame progress with flush=True")
    parser.add_argument("--workers", type=int, default=1, help="Frame-render worker processes; default 1")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def debug_log(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "debug", False):
        log(message)


def truthy(value: Any) -> bool:
    return str(value).strip().upper() in {"T", "TRUE", "1", "YES"}


def load_case(run_dir: Path) -> dict:
    path = run_dir / "case.py"
    if not path.is_file():
        return {}
    import runpy
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = runpy.run_path(str(path))
    return dict(ns.get("case", {}))


def mechanism_path_from_case(case: dict) -> Path:
    rel = str(case.get("cantera_file", "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"))
    path = Path(rel)
    if not path.is_absolute():
        path = REPO / path
    return path


def phase_from_case(case: dict) -> str:
    return str(case.get("cantera_phase", "yao_sk54"))


def parse_inline_composition(text: str) -> dict[str, float]:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise RuntimeError(f"unsupported composition syntax: {text!r}")
    out: dict[str, float] = {}
    for part in text[1:-1].split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(":", 1)
        out[key.strip()] = float(value.strip())
    return out


def parse_mechanism_without_yaml(path: Path, phase: str) -> tuple[list[str], dict[str, dict[str, float]], str]:
    lines = path.read_text().splitlines()
    phase_species: list[str] | None = None
    for idx, line in enumerate(lines):
        if line.strip() == f"- name: {phase}":
            collected: list[str] = []
            for subline in lines[idx + 1:]:
                stripped = subline.strip()
                if stripped.startswith("kinetics:") or stripped.startswith("transport:"):
                    break
                if stripped.startswith("species:"):
                    collected.append(stripped.split("[", 1)[1] if "[" in stripped else "")
                    if "]" in stripped:
                        break
                    continue
                if collected:
                    collected.append(stripped)
                    if "]" in stripped:
                        break
            joined = " ".join(collected).replace("[", "").replace("]", "")
            phase_species = [item.strip() for item in joined.split(",") if item.strip()]
            break
    if phase_species is None:
        raise RuntimeError(f"could not find phase {phase!r} in {path}")
    compositions: dict[str, dict[str, float]] = {}
    in_species = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped == "species:":
            in_species = True
            continue
        if not in_species:
            continue
        if stripped.startswith("- name:"):
            current = stripped.split(":", 1)[1].strip()
            continue
        if current and stripped.startswith("composition:"):
            compositions[current] = parse_inline_composition(stripped.split(":", 1)[1].strip())
            current = None
    return phase_species, compositions, "fallback YAML parser"


def load_species_and_mw(case: dict) -> tuple[list[str], np.ndarray, str]:
    path = mechanism_path_from_case(case)
    phase = phase_from_case(case)
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text())
        phase_species = None
        for phase_entry in data.get("phases", []):
            if phase_entry.get("name") == phase:
                phase_species = list(phase_entry.get("species", []))
                break
        if phase_species is None:
            raise RuntimeError(f"could not find phase {phase!r} in {path}")
        compositions = {entry["name"]: dict(entry.get("composition", {})) for entry in data.get("species", [])}
        source = "yaml.safe_load"
    except Exception:
        phase_species, compositions, source = parse_mechanism_without_yaml(path, phase)
    atomic = {"H": 1.00794, "C": 12.0107, "O": 15.9994, "N": 14.0067, "Ar": 39.948}
    mw: list[float] = []
    missing = [sp for sp in phase_species if sp not in compositions]
    if missing:
        raise RuntimeError(f"mechanism missing species compositions: {missing}")
    for sp in phase_species:
        total = 0.0
        for element, count in compositions[sp].items():
            if element not in atomic:
                raise RuntimeError(f"unsupported element {element!r} in {sp}")
            total += atomic[element] * float(count)
        mw.append(total)
    return phase_species, np.array(mw, dtype=float), f"{path} ({source})"


def species_cons_index_1based(all_species: list[str], species: str) -> int:
    if species not in all_species:
        raise RuntimeError(f"species {species!r} not found in mechanism species list")
    return SPECIES_OFFSET + all_species.index(species) + 1


def gas_fluid_indices_from_case(case: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_fluids = int(case.get("num_fluids", 3) or 3)
    num_gas = int(case.get("chem_gas_num_fluids", 0) or 0)
    if num_gas > 0:
        ids = tuple(int(case[f"chem_gas_fluid_ids({idx})"]) for idx in range(1, num_gas + 1))
    else:
        ids = (int(case.get("chem_gas_fluid_id", 2)), int(case.get("chem_gas_fluid_id_2", 3))) if "chem_gas_fluid_id_2" in case else (2, 3)
    if any(fid < 1 or fid > num_fluids for fid in ids):
        raise RuntimeError(f"invalid chemistry gas fluid ids {ids}; num_fluids={num_fluids}")
    alpha_rho = tuple(fid for fid in ids)  # 1-based conservative variables.
    alpha = tuple(6 + fid for fid in ids)  # cons.7..9 are alpha_1..alpha_3.
    return alpha_rho, alpha


def eos_params_from_case(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    num_fluids = int(case.get("num_fluids", 3) or 3)
    gammas = np.array([float(case.get(f"fluid_pp({idx})%gamma", 0.0)) for idx in range(1, num_fluids + 1)])
    pi_infs = np.array([float(case.get(f"fluid_pp({idx})%pi_inf", 0.0)) for idx in range(1, num_fluids + 1)])
    qvs = np.array([float(case.get(f"fluid_pp({idx})%qv", 0.0)) for idx in range(1, num_fluids + 1)])
    return gammas, pi_infs, qvs, truthy(case.get("mpp_lim", "F"))


def available_pall_steps(run_dir: Path) -> list[int]:
    root = run_dir / "p_all"
    if not root.is_dir():
        return []
    steps: set[int] = set()
    for pdir in root.glob("p*"):
        if not pdir.is_dir() or pdir.name == "p_all":
            continue
        for savedir in pdir.iterdir():
            if savedir.is_dir() and savedir.name.isdigit():
                steps.add(int(savedir.name))
    return sorted(steps)


def available_flat_steps_fast(run_dir: Path) -> list[int]:
    # Avoid raw.available_steps() on hierarchical p_all trees: it performs many
    # recursive globs and can burn minutes on Lustre before printing anything.
    steps: set[int] = set()
    d_root = run_dir / "D"
    if d_root.is_dir():
        for path in d_root.rglob("*.dat"):
            info = raw.raw_file_info(path)
            if info is not None:
                steps.add(info[3])
    p_root = run_dir / "p_all"
    if p_root.is_dir():
        # Only support flat raw files directly under p_all here. Hierarchical
        # p_all/p*/<save>/q_cons_vf*.dat is handled by available_pall_steps().
        for path in p_root.glob("*.dat"):
            info = raw.raw_file_info(path)
            if info is not None:
                steps.add(info[3])
    return sorted(steps)


def all_steps(run_dir: Path) -> tuple[list[int], list[int], list[int]]:
    flat_steps = available_flat_steps_fast(run_dir)
    pall_steps = available_pall_steps(run_dir)
    return sorted(set(flat_steps) | set(pall_steps)), flat_steps, pall_steps


def read_fortran_vector(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data:
        return np.empty(0, dtype=float)
    if len(data) >= 8:
        n0 = struct.unpack("<i", data[:4])[0]
        n1 = struct.unpack("<i", data[-4:])[0]
        if 0 < n0 == n1 and n0 == len(data) - 8 and n0 % 8 == 0:
            return np.frombuffer(data[4:-4], dtype="<f8").astype(float, copy=True)
    if len(data) % 8 != 0:
        raise RuntimeError(f"{path} is neither a simple float64 vector nor one-record Fortran float64 data")
    return np.frombuffer(data, dtype="<f8").astype(float, copy=True)


def read_pall_field(run_dir: Path, step: int, stem: str, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    root = run_dir / "p_all"
    pieces: dict[tuple[float, float], float] = {}
    found = False
    for pdir in sorted(root.glob("p*")):
        savedir = pdir / str(step)
        path = savedir / f"{stem}{index}.dat"
        if not path.is_file():
            continue
        x = read_fortran_vector(savedir / "x_cb.dat") if (savedir / "x_cb.dat").is_file() else np.empty(0)
        y = read_fortran_vector(savedir / "y_cb.dat") if (savedir / "y_cb.dat").is_file() else np.empty(0)
        vals = read_fortran_vector(path)
        if x.size == 0:
            x = np.arange(vals.size, dtype=float)
        if y.size == 0:
            y = np.array([0.0], dtype=float)
        if vals.size != x.size * y.size:
            if vals.size == x.size and y.size == 1:
                pass
            else:
                raise RuntimeError(f"{path} has {vals.size} values; x*y={x.size}*{y.size}")
        arr = vals.reshape((y.size, x.size))
        for jj, yy in enumerate(y):
            for ii, xx in enumerate(x):
                pieces.setdefault((float(xx), float(yy)), float(arr[jj, ii]))
        found = True
    if not found:
        return None
    xs = np.array(sorted({key[0] for key in pieces}), dtype=float)
    ys = np.array(sorted({key[1] for key in pieces}), dtype=float)
    xi = {float(x): i for i, x in enumerate(xs)}
    yi = {float(y): i for i, y in enumerate(ys)}
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    for (x, y), value in pieces.items():
        grid[yi[float(y)], xi[float(x)]] = value
    return xs, ys, grid


def read_flat_generic(run_dir: Path, kind: str, index: int, step: int) -> dict:
    by_step = raw.field_files(run_dir, kind, index, step).get(step, [])
    values: dict[tuple[float, float], float] = {}
    for path in by_step:
        coords, vals = raw.read_raw_values(path)
        for coord, value in zip(coords, vals):
            values.setdefault((float(coord[0]), float(coord[1])), float(value))
    return {"available": bool(values), "values": values, "stats": raw.stats_from_values(values, available=bool(values)) if values else raw.missing_stats([], 0)}


def values_to_grid_or_none(field: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not field or not field.get("available"):
        return None
    return raw.values_to_grid(field["values"])


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (mid[0] - values[0])], mid, [values[-1] + (values[-1] - mid[-1])]))


def cell_area(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    dx = np.diff(edges(xs))
    dy = np.diff(edges(ys))
    return dy[:, None] * dx[None, :]


def mfc_primitive_y(rhoY_stack: np.ndarray, rho_g: np.ndarray) -> np.ndarray:
    rho_ref = np.maximum(rho_g, SGM_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = rhoY_stack / rho_ref[None, :, :]
    y = np.clip(y, 0.0, 1.0)
    sum_y = np.sum(y, axis=0)
    normalize = np.isfinite(sum_y) & (sum_y > 1.0)
    y[:, normalize] = y[:, normalize] / sum_y[normalize]
    return y


def mixture_mw(y: np.ndarray, mw: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sum(y / mw[:, None, None], axis=0)
        out = 1.0 / denom
    out[~(np.isfinite(out) & (out > 0.0))] = np.nan
    return out


def reconstruct_pressure_from_cons(cons: dict[int, np.ndarray], case: dict) -> np.ndarray | None:
    needed = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if not all(idx in cons for idx in needed):
        return None
    gammas, pi_infs, qvs, mpp_lim = eos_params_from_case(case)
    alpha_rho = np.stack([cons[1], cons[2], cons[3]])
    alpha = np.stack([cons[7], cons[8], cons[9]])
    if mpp_lim:
        alpha_rho = np.maximum(alpha_rho, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_sum = np.sum(alpha, axis=0)
        good = np.isfinite(alpha_sum) & (alpha_sum > SGM_EPS)
        alpha[:, good] = alpha[:, good] / alpha_sum[good]
    rho = np.sum(alpha_rho, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cons[4] / rho
        v = cons[5] / rho
        dyn = 0.5 * (cons[4] * u + cons[5] * v)
        gamma_mix = np.sum(alpha * gammas[:, None, None], axis=0)
        pi_mix = np.sum(alpha * pi_infs[:, None, None], axis=0)
        qv_mix = np.sum(alpha_rho * qvs[:, None, None], axis=0)
        p = (cons[6] - dyn - pi_mix - qv_mix) / gamma_mix
    return np.where(np.isfinite(p), np.maximum(p, PRESSURE_FLOOR_PA), np.nan)


def species_from_fields(fields: list[str]) -> set[str]:
    species: set[str] = set()
    for field in fields:
        if field.startswith("Y_"):
            species.add(field[2:])
        elif field.startswith("rhoY_"):
            species.add(field[5:])
    return species


def needed_cons_indices(all_species: list[str], fields: list[str], *, scalars: bool = False) -> set[int]:
    needed = {2, 3, 7, 8, 9}
    if "valid_gas_temperature" in fields or scalars:
        needed.update(range(1, 13 + len(all_species)))
    else:
        for species in species_from_fields(fields):
            needed.add(species_cons_index_1based(all_species, species))
    if scalars:
        for species in INTEGRATED_SPECIES:
            if species in all_species:
                needed.add(species_cons_index_1based(all_species, species))
    return needed


def make_state_from_arrays(
    step: int,
    save_index: int,
    time_s: float,
    time_source: str,
    source_layout: str,
    xs: np.ndarray,
    ys: np.ndarray,
    cons: dict[int, np.ndarray],
    prim_pressure: np.ndarray | None,
    raw_temperature: np.ndarray | None,
    case: dict,
    all_species: list[str],
    mw: np.ndarray,
    gas_mass_floor: float,
    temperature_required: bool,
) -> State:
    alpha_rho_g_idx, alpha_g_idx = gas_fluid_indices_from_case(case)
    if not all(idx in cons for idx in (*alpha_rho_g_idx, *alpha_g_idx, 7)):
        missing = [idx for idx in (*alpha_rho_g_idx, *alpha_g_idx, 7) if idx not in cons]
        raise RuntimeError(f"missing required conservative fields for gas/liquid masks: cons {missing}")
    rho_g = sum(cons[idx] for idx in alpha_rho_g_idx)
    alpha_g = sum(cons[idx] for idx in alpha_g_idx)
    alpha_liq = cons[7]
    pressure = prim_pressure if prim_pressure is not None else reconstruct_pressure_from_cons(cons, case)
    available_species_indices = {
        sp: species_cons_index_1based(all_species, sp)
        for sp in all_species
        if species_cons_index_1based(all_species, sp) in cons
    }
    all_species_available = len(available_species_indices) == len(all_species)
    temp_available = False
    if all_species_available and pressure is not None:
        rhoY_stack_arr = np.stack([cons[species_cons_index_1based(all_species, sp)] for sp in all_species])
        y_prim = mfc_primitive_y(rhoY_stack_arr, rho_g)
        mw_mix = mixture_mw(y_prim, mw)
        with np.errstate(divide="ignore", invalid="ignore"):
            temp = pressure * mw_mix / (R_UNIVERSAL_KJ_PER_KMOL_K * rho_g)
        temp_available = True
    elif raw_temperature is not None:
        temp = raw_temperature
        temp_available = True
    elif temperature_required:
        raise RuntimeError(
            "temperature unavailable: need all 54 conservative species plus pressure/EOS fields "
            "for C4 reconstruction, or raw-helper temperature from D/ output"
        )
    else:
        temp = np.full_like(rho_g, np.nan, dtype=float)
    base_valid = (
        np.isfinite(alpha_g)
        & np.isfinite(rho_g)
        & np.isfinite(alpha_liq)
        & (alpha_g > 0.5)
        & (rho_g > gas_mass_floor)
        & (alpha_liq < 0.5)
    )
    if temp_available:
        valid = base_valid & np.isfinite(temp) & (temp > 0.0)
    else:
        valid = base_valid
    arrays = {
        "valid_gas_temperature": np.where(valid, temp, np.nan),
        "rho_g": rho_g,
        "alpha_g": alpha_g,
        "liquid_alpha": alpha_liq,
    }
    if pressure is not None:
        arrays["pressure"] = pressure
    y_arrays: dict[str, np.ndarray] = {}
    rhoY_arrays: dict[str, np.ndarray] = {}
    for sp in SPECIES_OF_INTEREST:
        if sp not in all_species:
            continue
        idx = species_cons_index_1based(all_species, sp)
        if idx not in cons:
            continue
        rhoY = cons[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = rhoY / rho_g
        y_arrays[sp] = np.where(valid, y, np.nan)
        rhoY_arrays[sp] = rhoY
        arrays[f"Y_{sp}"] = y_arrays[sp]
        arrays[f"rhoY_{sp}"] = rhoY_arrays[sp]
    return State(step, save_index, time_s, time_source, source_layout, xs, ys, cell_area(xs, ys), arrays, y_arrays, rhoY_arrays, valid, alpha_liq)


def load_flat_state(run_dir: Path, step: int, save_index: int, time_s: float, time_source: str, case: dict, all_species: list[str], mw: np.ndarray, gas_mass_floor: float, fields: list[str], scalars: bool = False) -> State | None:
    try:
        fields = raw.read_step_fields(run_dir, step, gas_mass_floor)
    except Exception:
        return None
    grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("liquid_alpha", "vapor_alpha_rho", "air_alpha_rho", "rhoY_NC12H26"):
        gd = values_to_grid_or_none(fields.get(name, {}))
        if gd is not None:
            grids[name] = gd
    if "vapor_alpha_rho" not in grids or "air_alpha_rho" not in grids or "liquid_alpha" not in grids:
        return None
    xs, ys, _ = grids["vapor_alpha_rho"]
    cons: dict[int, np.ndarray] = {}
    for idx in sorted(needed_cons_indices(all_species, fields, scalars=scalars)):
        name = None
        if idx == 1:
            name = "liquid_alpha_rho"
        elif idx == 2:
            name = "vapor_alpha_rho"
        elif idx == 3:
            name = "air_alpha_rho"
        elif idx == 7:
            name = "liquid_alpha"
        elif idx == 8:
            name = "vapor_alpha"
        elif idx == 9:
            name = "air_alpha"
        elif idx >= 13:
            sp_i = idx - 13
            if 0 <= sp_i < len(all_species):
                name = f"rhoY_{all_species[sp_i]}"
        if name and name in fields and fields[name].get("available"):
            gd = values_to_grid_or_none(fields[name])
        else:
            # Read generic flat field if the base analyzer does not list it.
            gd = values_to_grid_or_none(read_flat_generic(run_dir, "cons", idx, step))
        if gd is not None:
            gx, gy, arr = gd
            if gx.shape == xs.shape and gy.shape == ys.shape and np.allclose(gx, xs) and np.allclose(gy, ys):
                cons[idx] = arr
    raw_temperature = None
    gd_t = values_to_grid_or_none(fields.get("temperature", {}))
    if gd_t is not None:
        gx, gy, arr = gd_t
        if gx.shape == xs.shape and gy.shape == ys.shape and np.allclose(gx, xs) and np.allclose(gy, ys):
            raw_temperature = arr
    prim_pressure = None
    gd_p = values_to_grid_or_none(fields.get("pressure", {}))
    if gd_p is not None:
        gx, gy, arr = gd_p
        if gx.shape == xs.shape and gy.shape == ys.shape and np.allclose(gx, xs) and np.allclose(gy, ys):
            prim_pressure = np.where(np.isfinite(arr), np.maximum(arr, PRESSURE_FLOOR_PA), np.nan)
    return make_state_from_arrays(step, save_index, time_s, time_source, "D_or_flat_raw", xs, ys, cons, prim_pressure, raw_temperature, case, all_species, mw, gas_mass_floor, "valid_gas_temperature" in fields or scalars)


def load_pall_state(run_dir: Path, step: int, save_index: int, time_s: float, time_source: str, case: dict, all_species: list[str], mw: np.ndarray, gas_mass_floor: float, fields: list[str], scalars: bool = False) -> State | None:
    cons: dict[int, np.ndarray] = {}
    xs_ref = ys_ref = None
    for idx in sorted(needed_cons_indices(all_species, fields, scalars=scalars)):
        gd = read_pall_field(run_dir, step, "q_cons_vf", idx)
        if gd is not None:
            xs, ys, arr = gd
            if xs_ref is None:
                xs_ref, ys_ref = xs, ys
            elif not (xs.shape == xs_ref.shape and ys.shape == ys_ref.shape and np.allclose(xs, xs_ref) and np.allclose(ys, ys_ref)):
                raise RuntimeError(f"p_all field cons.{idx} grid mismatch at save {step}")
            cons[idx] = arr
    if xs_ref is None or ys_ref is None:
        return None
    prim_pressure = None
    gd_p = read_pall_field(run_dir, step, "q_prim_vf", 6)
    if gd_p is not None:
        xs, ys, arr = gd_p
        if xs.shape == xs_ref.shape and ys.shape == ys_ref.shape and np.allclose(xs, xs_ref) and np.allclose(ys, ys_ref):
            prim_pressure = np.where(np.isfinite(arr), np.maximum(arr, PRESSURE_FLOOR_PA), np.nan)
    return make_state_from_arrays(step, save_index, time_s, time_source, "p_all_binary", xs_ref, ys_ref, cons, prim_pressure, None, case, all_species, mw, gas_mass_floor, "valid_gas_temperature" in fields or scalars)


def load_state(run_dir: Path, step: int, save_index: int, times: dict[int, tuple[float, str]], flat_steps: set[int], pall_steps: set[int], case: dict, all_species: list[str], mw: np.ndarray, gas_mass_floor: float, fields: list[str], scalars: bool = False) -> State:
    time_s, time_source = times.get(step, (math.nan, "missing"))
    if step in flat_steps:
        state = load_flat_state(run_dir, step, save_index, time_s, time_source, case, all_species, mw, gas_mass_floor, fields, scalars)
        if state is not None:
            return state
    if step in pall_steps:
        state = load_pall_state(run_dir, step, save_index, time_s, time_source, case, all_species, mw, gas_mass_floor, fields, scalars)
        if state is not None:
            return state
    raise RuntimeError(f"could not load save {step} from flat D/raw or p_all_binary")


def jobs_from_args(args: argparse.Namespace) -> list[RenderJob]:
    if args.option_a:
        return [
            RenderJob("valid_gas_temperature", None, None),
            RenderJob("Y_NC12H26", OPTION_A_ZOOM_TMIN_US, OPTION_A_ZOOM_TMAX_US),
            RenderJob("Y_OH", OPTION_A_ZOOM_TMIN_US, OPTION_A_ZOOM_TMAX_US),
        ]
    fields = args.fields or CONTOUR_FIELDS
    return [RenderJob(field, args.tmin_us, args.tmax_us) for field in fields]


def select_steps(steps: list[int], times: dict[int, tuple[float, str]], job: RenderJob, max_frames: int) -> list[int]:
    selected: list[int] = []
    for step in steps:
        time_s, _ = times.get(step, (math.nan, "missing"))
        time_us = time_s * 1.0e6 if math.isfinite(time_s) else math.nan
        if job.tmin_us is not None and (not math.isfinite(time_us) or time_us < job.tmin_us - 1.0e-12):
            continue
        if job.tmax_us is not None and (not math.isfinite(time_us) or time_us > job.tmax_us + 1.0e-12):
            continue
        selected.append(step)
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def draw_contour(path: Path, state: State, field: str, domain: str, case_label: str, xlim_um: tuple[float, float] | None, ylim_um: tuple[float, float] | None) -> None:
    if field not in state.arrays:
        raise RuntimeError(f"field {field!r} cannot be resolved; available arrays: {sorted(state.arrays)}")
    cfg = FIELD_CONFIG[field]
    values = state.arrays[field]
    plot_values = values.copy()
    if cfg["scale"] == "log":
        plot_values = np.where(plot_values > 0.0, plot_values, np.nan)
        mesh_kwargs = {"norm": colors.LogNorm(vmin=cfg["vmin"], vmax=cfg["vmax"]), "cmap": cfg["cmap"]}
    else:
        mesh_kwargs = {"vmin": cfg["vmin"], "vmax": cfg["vmax"], "cmap": cfg["cmap"]}
    finite = plot_values[np.isfinite(plot_values)]
    if finite.size == 0:
        raise RuntimeError(f"field {field!r} has no finite plottable values at save {state.step}")
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(edges(state.x * 1.0e6), edges(state.y * 1.0e6), plot_values, shading="auto", **mesh_kwargs)
    alpha = state.alpha_liq
    if alpha is not None and np.any(np.isfinite(alpha)) and np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
        ax.contour(state.x * 1.0e6, state.y * 1.0e6, alpha, levels=[0.5], colors="white", linewidths=1.1)
        ax.contour(state.x * 1.0e6, state.y * 1.0e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
    if xlim_um is not None:
        ax.set_xlim(xlim_um)
    if ylim_um is not None:
        ax.set_ylim(ylim_um)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(f"{cfg['label']} [{cfg['unit']}]")
    ax.set_title(f"{case_label}: {cfg['pretty']}, {domain}, t={state.time_s * 1e6:.3f} µs, save {state.step}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def finite_stats(arr: np.ndarray) -> tuple[float, float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    return float(np.nanmin(vals)), float(np.nanmax(vals)), float(np.nanmean(vals))


def compute_row(state: State) -> dict[str, Any]:
    row: dict[str, Any] = {
        "save_index": state.save_index,
        "step": state.step,
        "time_s": state.time_s,
        "time_us": state.time_s * 1.0e6 if math.isfinite(state.time_s) else math.nan,
        "time_source": state.time_source,
        "source_layout": state.source_layout,
        "valid_gas_cell_count": int(np.count_nonzero(state.valid_gas)),
    }
    temp = state.arrays["valid_gas_temperature"]
    tvals = temp[np.isfinite(temp)]
    row["valid_gas_temperature_max"] = float(np.nanmax(tvals)) if tvals.size else math.nan
    row["valid_gas_temperature_mean"] = float(np.nanmean(tvals)) if tvals.size else math.nan
    for sp in ["NC12H26", "OH", "CO2", "H2O"]:
        arr = state.y_arrays.get(sp)
        vals = arr[np.isfinite(arr)] if arr is not None else np.array([])
        row[f"max_Y_{sp}"] = float(np.nanmax(vals)) if vals.size else math.nan
    for sp in INTEGRATED_SPECIES:
        arr = state.rhoY_arrays.get(sp)
        if arr is None:
            row[f"integrated_{sp}"] = math.nan
        else:
            mask = np.isfinite(arr) & np.isfinite(state.area)
            row[f"integrated_{sp}"] = float(np.sum(arr[mask] * state.area[mask])) if np.any(mask) else math.nan
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def plot_trend(rows: list[dict[str, Any]], xkey: str, ykeys: list[str], labels: list[str], ylabel: str, path: Path, normalize: bool = False) -> None:
    if not rows:
        return
    t = np.array([float(row.get(xkey, math.nan)) for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for key, label in zip(ykeys, labels):
        vals = np.array([float(row.get(key, math.nan)) for row in rows])
        if normalize:
            finite = vals[np.isfinite(vals)]
            denom = float(np.nanmax(np.abs(finite))) if finite.size else math.nan
            if math.isfinite(denom) and denom > 0.0:
                vals = vals / denom
        ax.plot(t, vals, marker="o", ms=2.5, lw=1.4, label=label)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_trend_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    trend = out_dir / "trend_plots"
    plot_trend(rows, "time_us", ["valid_gas_temperature_max", "valid_gas_temperature_mean"], ["max", "mean"], "valid gas T [K]", trend / "valid_gas_temperature_max_mean.png")
    plot_trend(rows, "time_us", ["max_Y_NC12H26", "max_Y_OH", "max_Y_CO2", "max_Y_H2O"], ["NC12H26", "OH", "CO2", "H2O"], "max Y [-]", trend / "max_species_mass_fractions.png")
    plot_trend(rows, "time_us", [f"integrated_{sp}" for sp in INTEGRATED_SPECIES], INTEGRATED_SPECIES, r"integrated rhoY [kg m$^{-1}$]", trend / "integrated_species.png")
    plot_trend(rows, "time_us", ["integrated_OH", "integrated_HO2", "integrated_H2O2", "integrated_CO2", "integrated_H2O"], ["OH", "HO2", "H2O2", "CO2", "H2O"], "normalized integrated products/radicals [-]", trend / "normalized_products_radicals.png", normalize=True)


def assemble_mp4(frame_dir: Path, prefix: str, video_path: Path, fps: int) -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    pattern = frame_dir / f"{prefix}_%04d.png"
    command = [ffmpeg or "ffmpeg", "-y", "-framerate", str(fps), "-i", str(pattern), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(video_path)]
    if ffmpeg is None:
        return "ffmpeg_missing", " ".join(command)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown ffmpeg failure"
        return "ffmpeg_failed", tail
    return "mp4_written", str(video_path)


def nearest_step_for_time(steps: list[int], times: dict[int, tuple[float, str]], target_us: float) -> int | None:
    best = None
    best_dt = math.inf
    for step in steps:
        t, _ = times.get(step, (math.nan, "missing"))
        if not math.isfinite(t):
            continue
        dt = abs(t * 1.0e6 - target_us)
        if dt < best_dt:
            best_dt = dt
            best = step
    return best


def write_readme(out_dir: Path, args: argparse.Namespace, mechanism_source: str, flat_steps: list[int], pall_steps: list[int]) -> None:
    lines = [
        "Raw shock-droplet advisor package",
        f"run_dir: {args.run_dir.resolve()}",
        f"case_label: {args.case_label}",
        "",
        "I/O:",
        "  Uses analyze_shockdroplet_air_sk54.py raw helpers for flat D/ or flat p_all files.",
        "  Falls back to direct hierarchical p_all/p*/<save>/q_cons_vf*.dat binary reading when needed.",
        "  Does not call MFC post_process, mfc.sh, restart_data/lustre, or field_data exports.",
        f"  flat raw saves discovered: {len(flat_steps)}",
        f"  hierarchical p_all saves discovered: {len(pall_steps)}",
        "",
        "Main contour visuals are mass fractions Y_k, not rhoY_k.",
        "rhoY is used only for integrated scalar diagnostics.",
        "",
        "Valid gas mask: alpha_g > 0.5, rho_g > gas_mass_floor, alpha_liq < 0.5, finite positive T.",
        "Temperature fallback/reconstruction: p*MW(Y_primitive)/(R*rho_g), pressure floor 100 Pa, R=8314.46261815324.",
        f"Mechanism molecular weights: {mechanism_source}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n")


def write_summary(out_dir: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    lines = ["Advisor package key summary", f"states analyzed: {len(rows)}", f"frames written: {len(manifest)}"]
    if rows:
        first, last = rows[0], rows[-1]
        lines.extend([
            f"time range [us]: {first['time_us']} to {last['time_us']}",
            f"final valid_gas_temperature_max [K]: {last.get('valid_gas_temperature_max')}",
            f"final max_Y_NC12H26/OH/CO2/H2O: {last.get('max_Y_NC12H26')} / {last.get('max_Y_OH')} / {last.get('max_Y_CO2')} / {last.get('max_Y_H2O')}",
            f"final integrated_NC12H26/OH/CO2/H2O: {last.get('integrated_NC12H26')} / {last.get('integrated_OH')} / {last.get('integrated_CO2')} / {last.get('integrated_H2O')}",
        ])
    (out_dir / "key_summary.txt").write_text("\n".join(lines) + "\n")


def domains_from_args(args: argparse.Namespace) -> list[tuple[str, tuple[float, float] | None, tuple[float, float] | None]]:
    domains: list[tuple[str, tuple[float, float] | None, tuple[float, float] | None]] = []
    if not args.no_full_domain:
        domains.append(("full_domain", None, None))
    if not args.no_zoom:
        domains.append(("zoom", tuple(args.zoom_xlim_um), tuple(args.zoom_ylim_um)))
    if not domains:
        raise RuntimeError("both --no-full-domain and --no-zoom were specified; nothing to render")
    return domains


def steps_for_job(steps: list[int], times: dict[int, tuple[float, str]], job: RenderJob, args: argparse.Namespace) -> list[int]:
    if args.selected_times_us:
        out: list[int] = []
        for target_us in args.selected_times_us:
            step = nearest_step_for_time(steps, times, target_us)
            if step is not None and step not in out:
                out.append(step)
        return out[: args.max_frames] if args.max_frames > 0 else out
    return select_steps(steps, times, job, args.max_frames)


def write_manifest(out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "manifest.csv", manifest)


def render_task(task: tuple) -> dict[str, Any]:
    (
        run_dir, step, save_index, times, flat_steps, pall_steps, case, all_species, mw,
        gas_mass_floor, field, domain, xlim, ylim, case_label, out_dir, debug,
    ) = task
    state = load_state(run_dir, step, save_index, times, flat_steps, pall_steps, case, all_species, mw, gas_mass_floor, [field], False)
    cfg = FIELD_CONFIG[field]
    path = out_dir / "frames" / field / domain / f"{cfg['short']}_{domain}_{save_index:04d}.png"
    draw_contour(path, state, field, domain, case_label, xlim, ylim)
    if debug:
        log(f"rendered field={field} domain={domain} save={step} png={path}")
    return {
        "field": field,
        "domain": domain,
        "frame_index": save_index,
        "step": step,
        "time_s": state.time_s,
        "time_us": state.time_s * 1.0e6 if math.isfinite(state.time_s) else math.nan,
        "source_layout": state.source_layout,
        "png": str(path.relative_to(out_dir)),
        "video_status": "not_requested",
        "video_detail": "",
    }


def render_frames_fast(
    args: argparse.Namespace,
    run_dir: Path,
    out_dir: Path,
    steps: list[int],
    times: dict[int, tuple[float, str]],
    flat_set: set[int],
    pall_set: set[int],
    case: dict,
    all_species: list[str],
    mw: np.ndarray,
    jobs: list[RenderJob],
) -> list[dict[str, Any]]:
    log("frame stage start")
    domains = domains_from_args(args)
    manifest: list[dict[str, Any]] = []
    tasks: list[tuple] = []
    save_index_by_step = {step: i for i, step in enumerate(steps)}
    for job in jobs:
        selected = steps_for_job(steps, times, job, args)
        log(f"beginning field={job.field}; selected saves={selected}")
        if not selected:
            log(f"WARNING: no saves selected for field={job.field}; continuing")
            continue
        for domain, xlim, ylim in domains:
            (out_dir / "frames" / job.field / domain).mkdir(parents=True, exist_ok=True)
            for local_idx, step in enumerate(selected):
                tasks.append((
                    run_dir, step, local_idx, times, flat_set, pall_set, case, all_species, mw,
                    args.gas_mass_floor, job.field, domain, xlim, ylim, args.case_label, out_dir, args.debug,
                ))
    if args.workers > 1 and len(tasks) > 1:
        log(f"frame rendering with workers={args.workers}; tasks={len(tasks)}")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(render_task, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    manifest.append(future.result())
                except Exception as exc:
                    log(f"WARNING: frame task failed: {exc}")
    else:
        log(f"frame rendering serial; tasks={len(tasks)}")
        for task in tasks:
            try:
                manifest.append(render_task(task))
            except Exception as exc:
                field = task[10]
                if field == "valid_gas_temperature":
                    log(f"WARNING: temperature frame skipped: {exc}")
                    continue
                raise
    write_manifest(out_dir, manifest)
    log(f"frame stage end; frames={len(manifest)}")
    return manifest


def render(args: argparse.Namespace) -> list[dict[str, Any]]:
    log("entering render")
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    log(f"run-dir={run_dir}")
    log(f"out-dir={out_dir}")
    log(f"fields={args.fields}; option_a={args.option_a}; selected_times_us={args.selected_times_us}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log("output folders created")

    log("case/mechanism parse start")
    case = load_case(run_dir)
    all_species, mw, mechanism_source = load_species_and_mw(case)
    log(f"case/mechanism parse end; species={len(all_species)}; mechanism={mechanism_source}")

    log("save discovery start")
    steps, flat_steps, pall_steps = all_steps(run_dir)
    log(f"save discovery end; total={len(steps)} flat={len(flat_steps)} p_all={len(pall_steps)}")
    if not steps:
        raise RuntimeError(f"No raw D/ or p_all saves found in {run_dir}")
    times = raw.infer_times(run_dir, steps)
    first_t = times.get(steps[0], (math.nan, "missing"))[0]
    last_t = times.get(steps[-1], (math.nan, "missing"))[0]
    log(
        "time range: "
        f"{first_t * 1e6 if math.isfinite(first_t) else math.nan:.6g} to "
        f"{last_t * 1e6 if math.isfinite(last_t) else math.nan:.6g} us"
    )

    jobs = jobs_from_args(args)
    log(f"field list: {[job.field for job in jobs]}")
    for job in jobs:
        log(f"selected saves for {job.field}: {steps_for_job(steps, times, job, args)}")
    flat_set = set(flat_steps)
    pall_set = set(pall_steps)
    write_readme(out_dir, args, mechanism_source, flat_steps, pall_steps)
    log("README.txt written")

    fast_mode = bool(args.selected_times_us) or args.max_frames > 0 or args.skip_scalars or args.skip_trends
    manifest: list[dict[str, Any]] = []

    if fast_mode:
        log("fast-frame mode active: rendering frames before scalar/trend work")
        manifest = render_frames_fast(args, run_dir, out_dir, steps, times, flat_set, pall_set, case, all_species, mw, jobs)
        if args.skip_scalars and args.skip_trends:
            log("skip-scalars and skip-trends set; exiting after frame/manifest/README")
            write_summary(out_dir, [], manifest)
            return manifest
    else:
        log("standard mode: rendering frames before scalar/trend work to avoid silent startup delay")
        manifest = render_frames_fast(args, run_dir, out_dir, steps, times, flat_set, pall_set, case, all_species, mw, jobs)

    rows: list[dict[str, Any]] = []
    states_by_step: dict[int, State] = {}
    if not args.skip_scalars:
        log("scalar stage start")
        scalar_fields = CONTOUR_FIELDS
        for save_index, step in enumerate(steps):
            if args.debug:
                log(f"scalar loading save={step} index={save_index}")
            state = load_state(
                run_dir,
                step,
                save_index,
                times,
                flat_set,
                pall_set,
                case,
                all_species,
                mw,
                args.gas_mass_floor,
                scalar_fields,
                True,
            )
            states_by_step[step] = state
            rows.append(compute_row(state))
        write_csv(out_dir / "scalar_timeseries.csv", rows)
        log(f"scalar stage end; rows={len(rows)}")
    else:
        log("scalar stage skipped")

    if not args.skip_trends:
        log("trend stage start")
        if rows:
            make_trend_plots(rows, out_dir)
        else:
            log("WARNING: trends requested but no scalar rows are available")
        log("trend stage end")
    else:
        log("trend stage skipped")

    if not args.no_mp4:
        log("mp4 stage start")
        # MP4s for the fast frame stage are assembled here so slow video encoding
        # never delays first PNG/manifest output. Existing frames are serially encoded.
        videos_dir = out_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            cfg = FIELD_CONFIG[job.field]
            for domain, _xlim, _ylim in domains_from_args(args):
                frame_dir = out_dir / "frames" / job.field / domain
                if not any(frame_dir.glob("*.png")):
                    continue
                video_path = videos_dir / f"{cfg['short']}_{domain}.mp4"
                status, detail = assemble_mp4(frame_dir, f"{cfg['short']}_{domain}", video_path, args.fps)
                log(f"mp4 {job.field}/{domain}: {status} {detail}")
        log("mp4 stage end")
    else:
        log("mp4 stage skipped")

    write_summary(out_dir, rows, manifest)
    log(f"Wrote advisor package to {out_dir}")
    return manifest

def main() -> None:
    log("entering main")
    args = parse_args()
    log(f"parsed args: {args}")
    render(args)


if __name__ == "__main__":
    main()
