#!/usr/bin/env python3
"""Analyze Tanabe/SK54 MFC shared restart files directly.

This first-pass analyzer reads raw restart_data/lustre_<step>.dat files as
float64 arrays for the 128 x 128 restartable Tanabe case. It intentionally
avoids MFC post_process output and uses gas-masked proxy diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NVARS = 66
NX_POINTS = 129
NY_POINTS = 129

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

SPECIES_1BASED = {
    "OH": 5,
    "HO2": 6,
    "H2O2": 9,
    "O2": 10,
    "NC12H26": 46,
}

TRACKED_SPECIES = ("OH", "HO2", "H2O2", "O2", "NC12H26")
HOT_THRESHOLDS_K = (850.0, 1000.0, 1200.0, 1500.0, 2000.0)


@dataclass
class StateSummary:
    step: int
    time_s: float
    finite: bool
    nonfinite_count: int
    pressure_proxy_min: float
    pressure_proxy_max: float
    alpha_liq_min: float
    alpha_liq_max: float
    alpha_gas_min: float
    alpha_gas_max: float
    valid_gas_cells: int
    Tmax_gas_K: float
    T99_9_gas_K: float
    T99_gas_K: float
    T95_gas_K: float
    gas_cells_T_gt_850K: int
    gas_cells_T_gt_1000K: int
    gas_cells_T_gt_1200K: int
    gas_cells_T_gt_1500K: int
    gas_cells_T_gt_2000K: int
    Tmax_x_m: float
    Tmax_y_m: float
    Tmax_alpha_liq: float
    Tmax_alpha_vap: float
    Tmax_alpha_air: float
    Tmax_alpha_gas: float
    Tmax_rho_gas: float
    Tmax_Tgas_K: float
    Tmax_Y_O2: float
    Tmax_Y_NC12H26: float
    Tmax_Y_OH: float
    Tmax_Y_HO2: float
    Tmax_Y_H2O2: float
    D_eq_alpha05_m: float
    D_over_D0_alpha05: float
    D2_over_D02_alpha05: float
    D2_dense_core_m2: float
    D_eq_alpha_lo_m: float
    D_over_D0_alpha_lo: float
    D2_over_D02_alpha_lo: float
    Y_OH_max: float
    Y_OH_p99_9: float
    Y_OH_p99: float
    Y_OH_mean_gas: float
    Y_OH_integral_gas_m2: float
    Y_OH_min: float
    Y_HO2_max: float
    Y_HO2_p99_9: float
    Y_HO2_p99: float
    Y_HO2_mean_gas: float
    Y_HO2_integral_gas_m2: float
    Y_HO2_min: float
    Y_H2O2_max: float
    Y_H2O2_p99_9: float
    Y_H2O2_p99: float
    Y_H2O2_mean_gas: float
    Y_H2O2_integral_gas_m2: float
    Y_H2O2_min: float
    Y_O2_max: float
    Y_O2_p99_9: float
    Y_O2_p99: float
    Y_O2_mean_gas: float
    Y_O2_integral_gas_m2: float
    Y_O2_min: float
    Y_NC12H26_max: float
    Y_NC12H26_p99_9: float
    Y_NC12H26_p99: float
    Y_NC12H26_mean_gas: float
    Y_NC12H26_integral_gas_m2: float
    Y_NC12H26_min: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Tanabe/SK54 MFC restart_data/lustre_*.dat files directly."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--selected-steps", nargs="*", type=int, default=None)
    parser.add_argument("--alpha-gas-min", type=float, default=1.0e-6)
    parser.add_argument("--alpha-liq-max-for-gas", type=float, default=0.99)
    parser.add_argument("--alpha-liq-threshold", type=float, default=0.5)
    parser.add_argument("--d0", type=float, default=0.7e-3)
    parser.add_argument("--cv-v", type=float, default=1956.0)
    parser.add_argument("--cv-air", type=float, default=717.5)
    parser.add_argument("--t-save", type=float, default=1.0e-3)
    parser.add_argument("--domain-scale", type=float, default=6.0)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-contours", action="store_true")
    return parser.parse_args()


def find_lustre_files(run_dir: Path) -> list[tuple[int, Path]]:
    restart_dir = run_dir / "restart_data"
    if not restart_dir.is_dir():
        raise FileNotFoundError(f"restart_data directory not found: {restart_dir}")

    found: list[tuple[int, Path]] = []
    for path in restart_dir.glob("lustre_*.dat"):
        name = path.name
        if name in {"lustre_x_cb.dat", "lustre_y_cb.dat"}:
            continue
        match = re.fullmatch(r"lustre_(\d+)\.dat", name)
        if not match:
            continue
        found.append((int(match.group(1)), path))
    return sorted(found)


def read_lustre_file(path: Path) -> np.ndarray | None:
    expected = NVARS * NY_POINTS * NX_POINTS
    try:
        raw = np.fromfile(path, dtype=np.float64)
    except OSError as exc:
        print(f"[warn] could not read {path}: {exc}")
        return None
    if raw.size < expected:
        print(f"[warn] skipping {path}: size {raw.size} < expected {expected}")
        return None
    if raw.size != expected:
        print(f"[warn] {path}: size {raw.size} != expected {expected}; using first expected values")
        raw = raw[:expected]
    return raw.reshape((NVARS, NY_POINTS, NX_POINTS))


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def make_coords(d0: float, domain_scale: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    length = domain_scale * d0
    dx = length / (NX_POINTS - 1)
    dy = length / (NY_POINTS - 1)
    x = np.linspace(0.0, length, NX_POINTS)
    y = np.linspace(0.0, length, NY_POINTS)
    return x, y, dx, dy


def gas_temperature_proxy(arr: np.ndarray, cv_v: float, cv_air: float) -> tuple[np.ndarray, np.ndarray]:
    arho_v = arr[VAR_ALPHA_RHO_VAP]
    arho_a = arr[VAR_ALPHA_RHO_AIR]
    eint_v = arr[VAR_EINT_VAP]
    eint_a = arr[VAR_EINT_AIR]
    rho_gas = arho_v + arho_a

    t_v = safe_div(eint_v, arho_v * cv_v)
    t_a = safe_div(eint_a, arho_a * cv_air)
    weighted = np.zeros_like(rho_gas, dtype=np.float64)
    valid_v = np.isfinite(t_v) & np.isfinite(arho_v) & (arho_v > 0.0)
    valid_a = np.isfinite(t_a) & np.isfinite(arho_a) & (arho_a > 0.0)
    weighted[valid_v] += arho_v[valid_v] * t_v[valid_v]
    weighted[valid_a] += arho_a[valid_a] * t_a[valid_a]
    t_gas = safe_div(weighted, rho_gas)
    return t_gas, rho_gas


def species_var(species_name: str) -> int:
    return SPECIES_OFFSET + SPECIES_1BASED[species_name] - 1


def species_y(arr: np.ndarray, species_name: str, rho_gas: np.ndarray, valid_gas: np.ndarray) -> np.ndarray:
    rho_y = arr[species_var(species_name)]
    y = safe_div(rho_y, rho_gas)
    y[~valid_gas] = np.nan
    return y


def finite_values(field: np.ndarray) -> np.ndarray:
    return field[np.isfinite(field)]


def finite_percentile(field: np.ndarray, percentile: float) -> float:
    values = finite_values(field)
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def finite_max(field: np.ndarray) -> float:
    values = finite_values(field)
    if values.size == 0:
        return math.nan
    return float(np.max(values))


def finite_min(field: np.ndarray) -> float:
    values = finite_values(field)
    if values.size == 0:
        return math.nan
    return float(np.min(values))


def finite_mean(field: np.ndarray) -> float:
    values = finite_values(field)
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def species_stats(y_field: np.ndarray, cell_area: float) -> dict[str, float]:
    values = finite_values(y_field)
    if values.size == 0:
        return {
            "max": math.nan,
            "p99_9": math.nan,
            "p99": math.nan,
            "mean": math.nan,
            "integral": math.nan,
            "min": math.nan,
        }
    return {
        "max": float(np.max(values)),
        "p99_9": float(np.percentile(values, 99.9)),
        "p99": float(np.percentile(values, 99.0)),
        "mean": float(np.mean(values)),
        "integral": float(np.sum(values) * cell_area),
        "min": float(np.min(values)),
    }


def equivalent_diameter(alpha_liq: np.ndarray, threshold: float, cell_area: float) -> float:
    area = float(np.count_nonzero(np.isfinite(alpha_liq) & (alpha_liq > threshold)) * cell_area)
    if area <= 0.0:
        return math.nan
    return 2.0 * math.sqrt(area / math.pi)


def summarize_state(
    step: int,
    path: Path,
    arr: np.ndarray,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    cell_area: float,
) -> tuple[StateSummary, dict[str, np.ndarray]]:
    time_s = step * args.t_save
    alpha_liq = arr[VAR_ALPHA_LIQ]
    alpha_gas = arr[VAR_ALPHA_VAP] + arr[VAR_ALPHA_AIR]
    t_gas, rho_gas = gas_temperature_proxy(arr, args.cv_v, args.cv_air)
    valid_gas = (
        np.isfinite(t_gas)
        & np.isfinite(rho_gas)
        & (alpha_gas > args.alpha_gas_min)
        & (alpha_liq < args.alpha_liq_max_for_gas)
    )

    t_masked = np.where(valid_gas, t_gas, np.nan)
    if np.any(np.isfinite(t_masked)):
        flat_idx = int(np.nanargmax(t_masked))
        iy, ix = np.unravel_index(flat_idx, t_masked.shape)
        tmax = float(t_masked[iy, ix])
        tmax_x = float(x[ix])
        tmax_y = float(y[iy])
    else:
        iy = -1
        ix = -1
        tmax = math.nan
        tmax_x = math.nan
        tmax_y = math.nan

    t99_9 = finite_percentile(t_masked, 99.9)
    t99 = finite_percentile(t_masked, 99.0)
    t95 = finite_percentile(t_masked, 95.0)
    hot_counts = {
        int(threshold): int(np.count_nonzero(np.isfinite(t_masked) & (t_masked > threshold)))
        for threshold in HOT_THRESHOLDS_K
    }

    d05 = equivalent_diameter(alpha_liq, args.alpha_liq_threshold, cell_area)
    dlo = equivalent_diameter(alpha_liq, 1.0e-3, cell_area)

    y_fields = {
        name: species_y(arr, name, rho_gas, valid_gas)
        for name in TRACKED_SPECIES
    }
    y_stats = {name: species_stats(field, cell_area) for name, field in y_fields.items()}

    def at_tmax(field: np.ndarray) -> float:
        if iy < 0 or ix < 0:
            return math.nan
        value = field[iy, ix]
        return float(value) if np.isfinite(value) else math.nan

    summary = StateSummary(
        step=step,
        time_s=time_s,
        finite=bool(np.all(np.isfinite(arr))),
        nonfinite_count=int(np.size(arr) - np.count_nonzero(np.isfinite(arr))),
        pressure_proxy_min=finite_min(arr[VAR_ENERGY]),
        pressure_proxy_max=finite_max(arr[VAR_ENERGY]),
        alpha_liq_min=finite_min(alpha_liq),
        alpha_liq_max=finite_max(alpha_liq),
        alpha_gas_min=finite_min(alpha_gas),
        alpha_gas_max=finite_max(alpha_gas),
        valid_gas_cells=int(np.count_nonzero(valid_gas)),
        Tmax_gas_K=tmax,
        T99_9_gas_K=t99_9,
        T99_gas_K=t99,
        T95_gas_K=t95,
        gas_cells_T_gt_850K=hot_counts[850],
        gas_cells_T_gt_1000K=hot_counts[1000],
        gas_cells_T_gt_1200K=hot_counts[1200],
        gas_cells_T_gt_1500K=hot_counts[1500],
        gas_cells_T_gt_2000K=hot_counts[2000],
        Tmax_x_m=tmax_x,
        Tmax_y_m=tmax_y,
        Tmax_alpha_liq=at_tmax(alpha_liq),
        Tmax_alpha_vap=at_tmax(arr[VAR_ALPHA_VAP]),
        Tmax_alpha_air=at_tmax(arr[VAR_ALPHA_AIR]),
        Tmax_alpha_gas=at_tmax(alpha_gas),
        Tmax_rho_gas=at_tmax(rho_gas),
        Tmax_Tgas_K=at_tmax(t_masked),
        Tmax_Y_O2=at_tmax(y_fields["O2"]),
        Tmax_Y_NC12H26=at_tmax(y_fields["NC12H26"]),
        Tmax_Y_OH=at_tmax(y_fields["OH"]),
        Tmax_Y_HO2=at_tmax(y_fields["HO2"]),
        Tmax_Y_H2O2=at_tmax(y_fields["H2O2"]),
        D_eq_alpha05_m=d05,
        D_over_D0_alpha05=float(d05 / args.d0) if np.isfinite(d05) else math.nan,
        D2_over_D02_alpha05=float((d05 / args.d0) ** 2) if np.isfinite(d05) else math.nan,
        D2_dense_core_m2=float(d05 * d05) if np.isfinite(d05) else math.nan,
        D_eq_alpha_lo_m=dlo,
        D_over_D0_alpha_lo=float(dlo / args.d0) if np.isfinite(dlo) else math.nan,
        D2_over_D02_alpha_lo=float((dlo / args.d0) ** 2) if np.isfinite(dlo) else math.nan,
        Y_OH_max=y_stats["OH"]["max"],
        Y_OH_p99_9=y_stats["OH"]["p99_9"],
        Y_OH_p99=y_stats["OH"]["p99"],
        Y_OH_mean_gas=y_stats["OH"]["mean"],
        Y_OH_integral_gas_m2=y_stats["OH"]["integral"],
        Y_OH_min=y_stats["OH"]["min"],
        Y_HO2_max=y_stats["HO2"]["max"],
        Y_HO2_p99_9=y_stats["HO2"]["p99_9"],
        Y_HO2_p99=y_stats["HO2"]["p99"],
        Y_HO2_mean_gas=y_stats["HO2"]["mean"],
        Y_HO2_integral_gas_m2=y_stats["HO2"]["integral"],
        Y_HO2_min=y_stats["HO2"]["min"],
        Y_H2O2_max=y_stats["H2O2"]["max"],
        Y_H2O2_p99_9=y_stats["H2O2"]["p99_9"],
        Y_H2O2_p99=y_stats["H2O2"]["p99"],
        Y_H2O2_mean_gas=y_stats["H2O2"]["mean"],
        Y_H2O2_integral_gas_m2=y_stats["H2O2"]["integral"],
        Y_H2O2_min=y_stats["H2O2"]["min"],
        Y_O2_max=y_stats["O2"]["max"],
        Y_O2_p99_9=y_stats["O2"]["p99_9"],
        Y_O2_p99=y_stats["O2"]["p99"],
        Y_O2_mean_gas=y_stats["O2"]["mean"],
        Y_O2_integral_gas_m2=y_stats["O2"]["integral"],
        Y_O2_min=y_stats["O2"]["min"],
        Y_NC12H26_max=y_stats["NC12H26"]["max"],
        Y_NC12H26_p99_9=y_stats["NC12H26"]["p99_9"],
        Y_NC12H26_p99=y_stats["NC12H26"]["p99"],
        Y_NC12H26_mean_gas=y_stats["NC12H26"]["mean"],
        Y_NC12H26_integral_gas_m2=y_stats["NC12H26"]["integral"],
        Y_NC12H26_min=y_stats["NC12H26"]["min"],
    )
    fields = {"alpha_liq": alpha_liq, "Tgas": t_masked, **{f"Y_{k}": v for k, v in y_fields.items()}}
    return summary, fields


def choose_selected_steps(steps: list[int], requested: Iterable[int] | None) -> list[int]:
    if not steps:
        return []
    if requested:
        wanted = set(requested)
        return [step for step in steps if step in wanted]
    return sorted({steps[0], steps[len(steps) // 2], steps[-1]})


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def select_dense_core_k_fit(summaries: list[StateSummary]) -> dict[str, float | int | bool | None]:
    time = np.array([s.time_s for s in summaries], dtype=np.float64)
    d2 = np.array([s.D2_dense_core_m2 for s in summaries], dtype=np.float64)
    best: dict[str, float | int | bool | None] | None = None
    best_score: tuple[bool, float, float, int] | None = None

    for start in range(len(summaries)):
        for end in range(start + 9, len(summaries)):
            t_win = time[start : end + 1]
            d2_win = d2[start : end + 1]
            finite = np.isfinite(t_win) & np.isfinite(d2_win)
            if np.count_nonzero(finite) < 10:
                continue
            duration = float(np.nanmax(t_win[finite]) - np.nanmin(t_win[finite]))
            if duration < 0.020:
                continue
            fit = linear_fit(t_win, d2_win)
            slope = fit["slope"]
            r2 = fit["r2"]
            if not np.isfinite(slope) or not np.isfinite(r2):
                continue
            negative = slope < 0.0
            points = int(np.count_nonzero(finite))
            score = (negative, float(r2), duration, points)
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "sufficient": True,
                    "K_eff_dense_core_m2_s": float(-slope),
                    "slope_m2_s": float(slope),
                    "intercept_m2": float(fit["intercept"]),
                    "R2": float(r2),
                    "start_step": summaries[start].step,
                    "end_step": summaries[end].step,
                    "start_time_s": float(t_win[0]),
                    "end_time_s": float(t_win[-1]),
                    "duration_s": duration,
                    "point_count": points,
                    "negative_slope": negative,
                }

    if best is not None:
        return best
    return {
        "sufficient": False,
        "K_eff_dense_core_m2_s": math.nan,
        "slope_m2_s": math.nan,
        "intercept_m2": math.nan,
        "R2": math.nan,
        "start_step": None,
        "end_step": None,
        "start_time_s": math.nan,
        "end_time_s": math.nan,
        "duration_s": math.nan,
        "point_count": 0,
        "negative_slope": False,
    }


def plot_timeseries(summaries: list[StateSummary], out_dir: Path, dense_core_fit: dict[str, float | int | bool | None]) -> None:
    if not summaries:
        return
    time = np.array([s.time_s for s in summaries])

    def line_plot(filename: str, ys: list[tuple[str, np.ndarray]], ylabel: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        for label, values in ys:
            ax.plot(time, values, marker="o", linewidth=1.8, markersize=3.5, label=label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)

    line_plot("Tmax_gas_vs_time.png", [("Tmax gas proxy", np.array([s.Tmax_gas_K for s in summaries]))], "Tmax gas [K]", "Gas-masked Tmax proxy")
    line_plot(
        "T_robust_percentiles_vs_time.png",
        [
            ("Tmax", np.array([s.Tmax_gas_K for s in summaries])),
            ("T99.9", np.array([s.T99_9_gas_K for s in summaries])),
            ("T99", np.array([s.T99_gas_K for s in summaries])),
            ("T95", np.array([s.T95_gas_K for s in summaries])),
        ],
        "Gas temperature proxy [K]",
        "Robust gas-temperature percentiles",
    )
    line_plot(
        "hot_cell_counts_vs_time.png",
        [
            (">850 K", np.array([s.gas_cells_T_gt_850K for s in summaries])),
            (">1000 K", np.array([s.gas_cells_T_gt_1000K for s in summaries])),
            (">1200 K", np.array([s.gas_cells_T_gt_1200K for s in summaries])),
            (">1500 K", np.array([s.gas_cells_T_gt_1500K for s in summaries])),
            (">2000 K", np.array([s.gas_cells_T_gt_2000K for s in summaries])),
        ],
        "Gas-cell count",
        "Hot gas-cell counts",
    )
    line_plot("D_over_D0_vs_time.png", [("alpha_liq > 0.5", np.array([s.D_over_D0_alpha05 for s in summaries])), ("alpha_liq > 1e-3", np.array([s.D_over_D0_alpha_lo for s in summaries]))], "D / D0", "Equivalent droplet diameter")
    line_plot("D2_over_D02_vs_time.png", [("alpha_liq > 0.5", np.array([s.D2_over_D02_alpha05 for s in summaries])), ("alpha_liq > 1e-3", np.array([s.D2_over_D02_alpha_lo for s in summaries]))], "D2 / D0^2", "Equivalent D2 regression")
    plot_dense_core_fit(summaries, dense_core_fit, out_dir)
    line_plot(
        "species_max_vs_time.png",
        [
            ("OH", np.array([s.Y_OH_max for s in summaries])),
            ("HO2", np.array([s.Y_HO2_max for s in summaries])),
            ("H2O2", np.array([s.Y_H2O2_max for s in summaries])),
            ("O2", np.array([s.Y_O2_max for s in summaries])),
            ("NC12H26", np.array([s.Y_NC12H26_max for s in summaries])),
        ],
        "Max gas-masked Yk",
        "Gas-masked species maxima",
    )
    line_plot(
        "species_p999_vs_time.png",
        [
            ("OH", np.array([s.Y_OH_p99_9 for s in summaries])),
            ("HO2", np.array([s.Y_HO2_p99_9 for s in summaries])),
            ("H2O2", np.array([s.Y_H2O2_p99_9 for s in summaries])),
            ("O2", np.array([s.Y_O2_p99_9 for s in summaries])),
            ("NC12H26", np.array([s.Y_NC12H26_p99_9 for s in summaries])),
        ],
        "Gas-masked Yk p99.9",
        "Robust gas-masked species p99.9",
    )


def plot_dense_core_fit(
    summaries: list[StateSummary],
    dense_core_fit: dict[str, float | int | bool | None],
    out_dir: Path,
) -> None:
    time = np.array([s.time_s for s in summaries])
    d2 = np.array([s.D2_dense_core_m2 for s in summaries])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ax.plot(time, d2, marker="o", linewidth=1.8, markersize=3.5, label="dense core D2")
    if dense_core_fit.get("sufficient"):
        t0 = float(dense_core_fit["start_time_s"])
        t1 = float(dense_core_fit["end_time_s"])
        slope = float(dense_core_fit["slope_m2_s"])
        intercept = float(dense_core_fit["intercept_m2"])
        mask = np.isfinite(time) & (time >= t0) & (time <= t1)
        ax.plot(time[mask], slope * time[mask] + intercept, "--", linewidth=2.0, label="best fit")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Dense-core D2 [m2]")
    ax.set_title("Dense-core D2 with best linear fit")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "D2_dense_core_with_best_fit.png", dpi=300)
    plt.close(fig)


def plot_contour(field: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    finite = field[np.isfinite(field)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_csv(summaries: list[StateSummary], path: Path) -> None:
    if not summaries:
        return
    fields = list(asdict(summaries[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def first_threshold(summaries: list[StateSummary], threshold: float) -> dict[str, float | int | None]:
    for summary in summaries:
        if np.isfinite(summary.Tmax_gas_K) and summary.Tmax_gas_K >= threshold:
            return {"step": summary.step, "time_s": summary.time_s, "Tmax_gas_K": summary.Tmax_gas_K}
    return {"step": None, "time_s": None, "Tmax_gas_K": None}


def first_delta_threshold(
    summaries: list[StateSummary],
    field_name: str,
    baseline: float,
    delta: float,
) -> dict[str, float | int | None]:
    threshold = baseline + delta
    for summary in summaries:
        value = getattr(summary, field_name)
        if np.isfinite(value) and value >= threshold:
            return {"step": summary.step, "time_s": summary.time_s, field_name: value, "threshold_K": threshold}
    return {"step": None, "time_s": None, field_name: None, "threshold_K": threshold}


def max_dtdt(summaries: list[StateSummary]) -> float:
    best = math.nan
    for a, b in zip(summaries, summaries[1:]):
        dt = b.time_s - a.time_s
        if dt > 0 and np.isfinite(a.Tmax_gas_K) and np.isfinite(b.Tmax_gas_K):
            slope = (b.Tmax_gas_K - a.Tmax_gas_K) / dt
            if not np.isfinite(best) or slope > best:
                best = slope
    return best


def species_normalization_warnings(summaries: list[StateSummary]) -> list[str]:
    warnings: list[str] = []
    for name in TRACKED_SPECIES:
        max_values = np.array([getattr(s, f"Y_{name}_max") for s in summaries], dtype=np.float64)
        min_values = np.array([getattr(s, f"Y_{name}_min") for s in summaries], dtype=np.float64)
        max_value = finite_max(max_values)
        min_value = finite_min(min_values)
        if np.isfinite(max_value) and max_value > 1.0:
            warnings.append(f"Y_{name} max exceeds 1.0: {max_value:.6e}")
        if np.isfinite(min_value) and min_value < -1.0e-8:
            warnings.append(f"Y_{name} min is below -1e-8: {min_value:.6e}")
    return warnings


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = find_lustre_files(args.run_dir)
    if args.max_states is not None:
        files = files[: args.max_states]
    if not files:
        raise SystemExit(f"No restart_data/lustre_<step>.dat files found under {args.run_dir}")

    x, y, dx, dy = make_coords(args.d0, args.domain_scale)
    cell_area = dx * dy
    selected_steps = choose_selected_steps([step for step, _ in files], args.selected_steps)
    summaries: list[StateSummary] = []

    for step, path in files:
        print(f"[info] reading step {step}: {path}")
        arr = read_lustre_file(path)
        if arr is None:
            continue
        summary, fields = summarize_state(step, path, arr, args, x, y, cell_area)
        summaries.append(summary)

        if not args.skip_contours and step in selected_steps:
            plot_contour(fields["alpha_liq"], args.out_dir / f"contour_alpha_liq_{step}.png", f"alpha_liq step {step}")
            plot_contour(fields["Tgas"], args.out_dir / f"contour_Tgas_{step}.png", f"Gas T proxy step {step}", cmap="inferno")
            for name in ("OH", "HO2", "H2O2", "NC12H26"):
                plot_contour(fields[f"Y_{name}"], args.out_dir / f"contour_Y_{name}_{step}.png", f"Y_{name} step {step}", cmap="magma")

    summaries.sort(key=lambda item: item.step)
    write_csv(summaries, args.out_dir / "tanabe_restart_by_state.csv")
    dense_core_fit = select_dense_core_k_fit(summaries)
    if not args.skip_plots:
        plot_timeseries(summaries, args.out_dir, dense_core_fit)

    first_induction = first_threshold(summaries, 820.0)
    total_ignition = first_threshold(summaries, 2000.0)
    final_time = summaries[-1].time_s if summaries else math.nan
    initial_tmax = summaries[0].Tmax_gas_K if summaries else math.nan
    initial_t999 = summaries[0].T99_9_gas_K if summaries else math.nan
    first_stage_delta20_tmax = first_delta_threshold(summaries, "Tmax_gas_K", initial_tmax, 20.0)
    first_stage_delta20_t999 = first_delta_threshold(summaries, "T99_9_gas_K", initial_t999, 20.0)
    absolute_820_unreliable = bool(summaries and first_induction["step"] == summaries[0].step)
    species_warnings = species_normalization_warnings(summaries)
    summary_json = {
        "run_dir": str(args.run_dir),
        "state_count": len(summaries),
        "first_step": summaries[0].step if summaries else None,
        "last_step": summaries[-1].step if summaries else None,
        "final_time_s": final_time,
        "ambient_temperature_K": 800.0,
        "initial_Tmax_gas_K": initial_tmax,
        "initial_T99_9_gas_K": initial_t999,
        "first_induction_absolute_Tmax_ge_820K": first_induction,
        "first_induction_absolute_820K_unreliable": absolute_820_unreliable,
        "first_stage_delta20_Tmax": first_stage_delta20_tmax,
        "first_stage_delta20_T99_9": first_stage_delta20_t999,
        "total_ignition_Tmax_ge_2000K": total_ignition,
        "max_dTmax_dt_K_s": max_dtdt(summaries),
        "hot_ignition_by_0p15s": bool(total_ignition["time_s"] is not None and total_ignition["time_s"] <= 0.15),
        "dense_core_D2_fit": dense_core_fit,
        "recommended_K_label": "K_eff_dense_core, not classical K_burn, unless the fit window and R2 support a D2-law interpretation",
        "species_normalization_warnings": species_warnings,
        "assumptions": {
            "nvars": NVARS,
            "shape": [NVARS, NY_POINTS, NX_POINTS],
            "dtype": "float64",
            "time_s": "step * t_save",
            "t_save_s": args.t_save,
            "domain_length_m": args.domain_scale * args.d0,
            "cv_v_J_kg_K": args.cv_v,
            "cv_air_J_kg_K": args.cv_air,
            "temperature_proxy": "mass-weighted vapor/air cv temperature, gas-masked only",
        },
    }
    with (args.out_dir / "tanabe_restart_summary.json").open("w") as f:
        json.dump(summary_json, f, indent=2, allow_nan=True)

    lines = [
        "Tanabe/SK54 restart_data analyzer summary",
        f"Run dir: {args.run_dir}",
        f"States analyzed: {len(summaries)}",
        f"Step range: {summary_json['first_step']} .. {summary_json['last_step']}",
        f"Final inferred time: {final_time:.6e} s",
        f"Initial Tmax gas proxy: {initial_tmax:.6e} K",
        f"Initial T99.9 gas proxy: {initial_t999:.6e} K",
        f"Absolute first induction proxy Tmax>=820 K: {first_induction}",
        f"Absolute 820 K threshold unreliable because it fires at first state: {absolute_820_unreliable}",
        f"Baseline-corrected first-stage proxy Tmax>=initial+20 K: {first_stage_delta20_tmax}",
        f"Baseline-corrected first-stage proxy T99.9>=initial+20 K: {first_stage_delta20_t999}",
        f"Total ignition proxy Tmax>=2000 K: {total_ignition}",
        f"Max dTmax/dt: {summary_json['max_dTmax_dt_K_s']:.6e} K/s",
        f"Hot ignition by 0.15 s: {summary_json['hot_ignition_by_0p15s']}",
        "",
        "Dense-core D2 fit",
        f"  K_eff_dense_core: {dense_core_fit['K_eff_dense_core_m2_s']}",
        f"  slope: {dense_core_fit['slope_m2_s']}",
        f"  R2: {dense_core_fit['R2']}",
        f"  window: {dense_core_fit['start_time_s']} .. {dense_core_fit['end_time_s']} s",
        f"  points: {dense_core_fit['point_count']}",
        "  Label guidance: call this K_eff unless the chosen window has strong D2-law behavior; do not overclaim K_burn.",
        "",
        "Species normalization warnings:",
        *(f"  - {warning}" for warning in species_warnings),
        "",
        "Temperature is a gas-only proxy; liquid-dominated cells are excluded.",
        "The alpha_liq > 1e-3 diameter is a liquid-footprint diagnostic, not the K_eff/K_burn basis.",
    ]
    (args.out_dir / "tanabe_restart_summary.txt").write_text("\n".join(lines) + "\n")

    print(f"[done] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
