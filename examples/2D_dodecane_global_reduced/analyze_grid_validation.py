#!/usr/bin/env python3
"""Analyze clean dodecane grid-validation raw D/ outputs.

This is an offline raw-output analyzer. It does not run MFC or post_process.

Example:
  python examples/2D_dodecane_global_reduced/analyze_grid_validation.py \
    --case low,nonreacting,/path/to/dodecane_grid_low_nonreacting \
    --case low,burning,/path/to/dodecane_grid_low_burning \
    --out grid_validation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np


RAW_RE = re.compile(r"^(?P<kind>[^.]+)\.(?P<index>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
SPECIES_FIELDS = {
    "C12H26": ("cons", 13),
    "O2": ("cons", 14),
    "CO2": ("cons", 16),
    "H2O": ("cons", 17),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="GRID,TYPE,RUN_DIR",
        help="Completed run folder. TYPE must be nonreacting or burning.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("grid_validation_summary.csv"),
        help="Per-run summary CSV path.",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=None,
        help="Pairwise burning/nonreacting comparison CSV path.",
    )
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=None,
        help="Per-timestep metrics CSV path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for grid-validation PNG plots and figure summary.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--no-snapshots", action="store_true", help="Skip raw field snapshot PNGs.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fit-frac-min", type=float, default=0.80)
    parser.add_argument("--fit-frac-max", type=float, default=0.95)
    return parser.parse_args()


def parse_case_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(",", 2)
    if len(parts) != 3:
        raise SystemExit(f"Invalid --case {spec!r}; expected GRID,TYPE,RUN_DIR")
    grid, case_type, run_dir = (part.strip() for part in parts)
    if case_type not in {"nonreacting", "burning"}:
        raise SystemExit(f"Invalid case type {case_type!r}; expected nonreacting or burning")
    return grid, case_type, Path(run_dir)


def ddir(run_dir: Path) -> Path:
    return run_dir if run_dir.name == "D" else run_dir / "D"


def raw_files(root: Path, kind: str, index: int) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for path in ddir(root).glob(f"{kind}.{index}.*.*.dat"):
        match = RAW_RE.match(path.name)
        if match and match.group("kind") == kind and int(match.group("index")) == index:
            out.setdefault(int(match.group("step")), []).append(path)
    return {step: sorted(paths) for step, paths in out.items()}


def common_steps(root: Path, fields: Iterable[tuple[str, int]]) -> list[int]:
    step_sets = []
    for kind, index in fields:
        steps = set(raw_files(root, kind, index))
        if not steps:
            return []
        step_sets.append(steps)
    return sorted(set.intersection(*step_sets))


def read_field(root: Path, kind: str, index: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    values: dict[tuple[float, float], float] = {}
    for path in raw_files(root, kind, index).get(step, []):
        arr = np.loadtxt(path)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for x, y, value in arr[:, [0, 1, arr.shape[1] - 1]]:
            key = (float(x), float(y))
            if key not in values:
                values[key] = float(value)
    if not values:
        raise FileNotFoundError(f"{root}: missing {kind}.{index} at step {step}")
    coords = np.array(list(values.keys()), dtype=float)
    vals = np.array([values[(float(x), float(y))] for x, y in coords], dtype=float)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order], vals[order]


def grid_metrics(coords: np.ndarray) -> tuple[int, int, float, float]:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dxs = np.diff(np.sort(xs))
    dys = np.diff(np.sort(ys))
    dx = float(np.median(dxs[dxs > 0])) if np.any(dxs > 0) else float("nan")
    dy = float(np.median(dys[dys > 0])) if np.any(dys > 0) else float("nan")
    return int(len(xs)), int(len(ys)), dx, dy


def alpha_bounds(alpha: np.ndarray) -> tuple[float, float]:
    finite = alpha[np.isfinite(alpha)]
    bg = float(np.nanpercentile(finite, 1.0))
    core = float(np.nanpercentile(finite, 99.0))
    if not core > bg:
        bg = float(np.nanmin(finite))
        core = float(np.nanmax(finite))
    if not core > bg:
        core = bg + 1.0
    return bg, core


def fit_window(time: np.ndarray, value: np.ndarray, threshold_norm: np.ndarray, fit_min: float, fit_max: float) -> tuple[float, float, int, int]:
    mask = (
        np.isfinite(time)
        & np.isfinite(value)
        & np.isfinite(threshold_norm)
        & (threshold_norm >= fit_min)
        & (threshold_norm <= fit_max)
    )
    idx = np.flatnonzero(mask)
    if idx.size < 4:
        return float("nan"), float("nan"), -1, -1
    t = time[idx]
    y = value[idx]
    slope, intercept = np.linalg.lstsq(np.column_stack([t, np.ones_like(t)]), y, rcond=None)[0]
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return float(-slope), float(r2), int(idx[0]), int(idx[-1])


def endpoint_sum(root: Path, field: tuple[str, int], step: int) -> float:
    return float(np.sum(read_field(root, field[0], field[1], step)[1]))


def analyze_case(
    grid: str,
    case_type: str,
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    required = [("cons", 1), ("cons", 2), ("cons", 7)]
    steps = common_steps(root, required)
    if not steps:
        raise SystemExit(f"No required raw fields found for {root}")

    first_coords, first_alpha = read_field(root, "cons", 7, steps[0])
    nx, ny, dx, dy = grid_metrics(first_coords)
    d_a = dx * dy
    bg, core = alpha_bounds(first_alpha)

    time = np.array(steps, dtype=float)
    sim_path = root / "simulation.inp"
    if root.name == "D":
        sim_path = root.parent / "simulation.inp"
    if sim_path.is_file():
        for line in sim_path.read_text(errors="replace").splitlines():
            stripped = line.split("!", 1)[0].strip()
            if stripped.lower().startswith("dt"):
                time *= float(stripped.split("=", 1)[1].strip().rstrip(","))
                break

    d2_threshold = []
    liquid_mass = []
    vapor_mass = []
    alpha_weighted = []
    pressure_max = []
    pressure_mean = []
    has_pressure = bool(common_steps(root, [("prim", 6)]))
    has_species = case_type == "burning" and bool(common_steps(root, SPECIES_FIELDS.values()))
    species_sums = {name: [] for name in SPECIES_FIELDS}

    for step in steps:
        _, alpha = read_field(root, "cons", 7, step)
        _, ar_liq = read_field(root, "cons", 1, step)
        _, ar_vap = read_field(root, "cons", 2, step)
        area = float(np.count_nonzero(alpha >= args.threshold) * d_a)
        d2_threshold.append((2.0 * math.sqrt(area / math.pi)) ** 2 if area > 0.0 else 0.0)
        liquid_mass.append(float(np.sum(ar_liq) * d_a))
        vapor_mass.append(float(np.sum(ar_vap) * d_a))
        weighted = np.clip((alpha - bg) / (core - bg), 0.0, 1.0)
        alpha_weighted.append(float(np.sum(weighted) * d_a))
        if has_pressure and step in raw_files(root, "prim", 6):
            _, pressure = read_field(root, "prim", 6, step)
            pressure_max.append(float(np.nanmax(pressure)))
            pressure_mean.append(float(np.nanmean(pressure)))
        else:
            pressure_max.append(float("nan"))
            pressure_mean.append(float("nan"))
        if has_species:
            for name, field in SPECIES_FIELDS.items():
                species_sums[name].append(endpoint_sum(root, field, step))

    d2_threshold = np.asarray(d2_threshold)
    liquid_mass = np.asarray(liquid_mass)
    vapor_mass = np.asarray(vapor_mass)
    alpha_weighted = np.asarray(alpha_weighted)
    pressure_max_arr = np.asarray(pressure_max)
    pressure_mean_arr = np.asarray(pressure_mean)
    threshold_norm = d2_threshold / d2_threshold[0] if d2_threshold[0] else np.full_like(d2_threshold, np.nan)
    mass_norm = liquid_mass / liquid_mass[0] if liquid_mass[0] else np.full_like(liquid_mass, np.nan)
    weighted_norm = alpha_weighted / alpha_weighted[0] if alpha_weighted[0] else np.full_like(alpha_weighted, np.nan)

    k_mass, r2_mass, fit_i0, fit_i1 = fit_window(time, mass_norm, threshold_norm, args.fit_frac_min, args.fit_frac_max)
    k_threshold, r2_threshold, _, _ = fit_window(time, threshold_norm, threshold_norm, args.fit_frac_min, args.fit_frac_max)

    row: dict[str, float | int | str] = {
        "grid_label": grid,
        "case_type": case_type,
        "run_dir": str(root),
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "fit_step_start": int(steps[fit_i0]) if fit_i0 >= 0 else "",
        "fit_step_end": int(steps[fit_i1]) if fit_i1 >= 0 else "",
        "D2_threshold_first": float(d2_threshold[0]),
        "D2_threshold_last": float(d2_threshold[-1]),
        "D2_threshold_final_norm": float(threshold_norm[-1]),
        "mass_equiv_D2_norm_first": float(mass_norm[0]),
        "mass_equiv_D2_norm_last": float(mass_norm[-1]),
        "alpha_weighted_D2_norm_first": float(weighted_norm[0]),
        "alpha_weighted_D2_norm_last": float(weighted_norm[-1]),
        "liquid_alpha_rho_first": float(liquid_mass[0]),
        "liquid_alpha_rho_last": float(liquid_mass[-1]),
        "liquid_alpha_rho_delta": float(liquid_mass[-1] - liquid_mass[0]),
        "vapor_alpha_rho_first": float(vapor_mass[0]),
        "vapor_alpha_rho_last": float(vapor_mass[-1]),
        "vapor_alpha_rho_delta": float(vapor_mass[-1] - vapor_mass[0]),
        "K_mass": k_mass,
        "R2_mass": r2_mass,
        "K_threshold": k_threshold,
        "R2_threshold": r2_threshold,
        "pressure_final_max": float(pressure_max_arr[-1]) if pressure_max_arr.size else "",
        "pressure_final_mean": float(pressure_mean_arr[-1]) if pressure_mean_arr.size else "",
    }

    if has_species:
        for name in SPECIES_FIELDS:
            row[f"{name}_delta"] = species_sums[name][-1] - species_sums[name][0]
    else:
        for name in SPECIES_FIELDS:
            row[f"{name}_delta"] = ""

    ts_rows: list[dict[str, float | int | str]] = []
    for i, step in enumerate(steps):
        ts_row: dict[str, float | int | str] = {
            "grid_label": grid,
            "case_type": case_type,
            "run_dir": str(root),
            "step": int(step),
            "time": float(time[i]),
            "D2_threshold": float(d2_threshold[i]),
            "D2_threshold_norm": float(threshold_norm[i]),
            "mass_equiv_D2_norm": float(mass_norm[i]),
            "alpha_weighted_D2_norm": float(weighted_norm[i]),
            "liquid_alpha_rho_integral": float(liquid_mass[i]),
            "vapor_alpha_rho_integral": float(vapor_mass[i]),
            "pressure_max": float(pressure_max_arr[i]) if pressure_max_arr.size else "",
            "pressure_mean": float(pressure_mean_arr[i]) if pressure_mean_arr.size else "",
        }
        if has_species:
            for name in SPECIES_FIELDS:
                ts_row[f"{name}_sum"] = float(species_sums[name][i])
                ts_row[f"{name}_delta"] = float(species_sums[name][i] - species_sums[name][0])
        else:
            for name in SPECIES_FIELDS:
                ts_row[f"{name}_sum"] = ""
                ts_row[f"{name}_delta"] = ""
        ts_rows.append(ts_row)

    return row, ts_rows


def comparison_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    out = []
    by_grid = sorted({str(row["grid_label"]) for row in rows})
    for grid in by_grid:
        non = next((row for row in rows if row["grid_label"] == grid and row["case_type"] == "nonreacting"), None)
        burn = next((row for row in rows if row["grid_label"] == grid and row["case_type"] == "burning"), None)
        if not non or not burn:
            continue
        non_reg = float(non["liquid_alpha_rho_first"]) - float(non["liquid_alpha_rho_last"])
        burn_reg = float(burn["liquid_alpha_rho_first"]) - float(burn["liquid_alpha_rho_last"])
        k_non = float(non["K_mass"])
        k_burn = float(burn["K_mass"])
        out.append({
            "grid_label": grid,
            "K_mass_burning": k_burn,
            "K_mass_nonreacting": k_non,
            "K_mass_burning_over_nonreacting": k_burn / k_non if np.isfinite(k_non) and k_non != 0.0 else "",
            "liquid_regression_nonreacting": non_reg,
            "liquid_regression_burning": burn_reg,
            "liquid_regression_percent_increase": 100.0*(burn_reg - non_reg)/non_reg if non_reg != 0.0 else "",
            "burning_K_mass_gt_nonreacting": bool(k_burn > k_non) if np.isfinite(k_burn) and np.isfinite(k_non) else "",
        })
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grid_sort_key(grid: str) -> tuple[int, str]:
    order = {"low": 0, "mid": 1, "high": 2}
    return order.get(grid, 99), grid


def load_plot_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def rows_by_case(ts_rows: list[dict[str, float | int | str]]) -> dict[tuple[str, str], list[dict[str, float | int | str]]]:
    out: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in ts_rows:
        out.setdefault((str(row["grid_label"]), str(row["case_type"])), []).append(row)
    return out


def arr(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) if row[key] != "" else np.nan for row in rows], dtype=float)


def savefig(plt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_grid_metric_overlay(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], case_type: str, plots_dir: Path) -> Path:
    metrics = [
        ("D2_threshold_norm", "threshold D2_norm"),
        ("mass_equiv_D2_norm", "mass-equivalent D2_norm"),
        ("alpha_weighted_D2_norm", "alpha-weighted D2_norm"),
    ]
    grids = sorted({grid for grid, ctype in by_case if ctype == case_type}, key=grid_sort_key)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        for grid in grids:
            rows = by_case[(grid, case_type)]
            ax.plot(arr(rows, "time"), arr(rows, metric), linewidth=2.0, label=grid)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(title="grid", fontsize=8)
    axes[0].set_title(f"{case_type.capitalize()} grid overlay: D2 metrics")
    axes[-1].set_xlabel("time")
    path = plots_dir / f"{case_type}_grid_overlay_d2_metrics.png"
    savefig(plt, path)
    return path


def plot_burning_vs_nonreacting_by_grid(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], plots_dir: Path) -> Path:
    grids = sorted({grid for grid, _ in by_case}, key=grid_sort_key)
    fig, axes = plt.subplots(len(grids), 1, figsize=(9.5, 3.4*max(len(grids), 1)), sharex=True)
    if len(grids) == 1:
        axes = [axes]
    for ax, grid in zip(axes, grids):
        for case_type, color in (("nonreacting", "#4c78a8"), ("burning", "#f58518")):
            rows = by_case.get((grid, case_type))
            if not rows:
                continue
            ax.plot(arr(rows, "time"), arr(rows, "mass_equiv_D2_norm"), color=color, linewidth=2.4, label=f"{case_type}: mass")
            ax.plot(arr(rows, "time"), arr(rows, "D2_threshold_norm"), color=color, linestyle=":", linewidth=1.8, label=f"{case_type}: threshold")
        ax.set_title(f"{grid} grid: burning vs nonreacting")
        ax.set_ylabel("D2_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time")
    path = plots_dir / "burning_vs_nonreacting_by_grid.png"
    savefig(plt, path)
    return path


def plot_budget_all_grids(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], plots_dir: Path, field: str, ylabel: str, filename: str) -> Path:
    plt.figure(figsize=(9.5, 6.2))
    for grid, case_type in sorted(by_case, key=lambda item: (grid_sort_key(item[0]), item[1])):
        rows = by_case[(grid, case_type)]
        linestyle = "-" if case_type == "burning" else "--"
        label = f"{grid} {case_type}"
        plt.plot(arr(rows, "time"), arr(rows, field), linestyle=linestyle, linewidth=2.0, label=label)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.title(ylabel + " across all grids")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = plots_dir / filename
    savefig(plt, path)
    return path


def plot_pressure_all_grids(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    for grid, case_type in sorted(by_case, key=lambda item: (grid_sort_key(item[0]), item[1])):
        rows = by_case[(grid, case_type)]
        linestyle = "-" if case_type == "burning" else "--"
        label = f"{grid} {case_type}"
        axes[0].plot(arr(rows, "time"), arr(rows, "pressure_mean"), linestyle=linestyle, linewidth=2.0, label=label)
        axes[1].plot(arr(rows, "time"), arr(rows, "pressure_max"), linestyle=linestyle, linewidth=2.0, label=label)
    axes[0].set_ylabel("pressure mean")
    axes[1].set_ylabel("pressure max")
    axes[1].set_xlabel("time")
    axes[0].set_title("Pressure diagnostics across all grids")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    path = plots_dir / "pressure_mean_max_all_grids.png"
    savefig(plt, path)
    return path


def plot_burning_species(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], plots_dir: Path) -> Path:
    species = ["C12H26", "O2", "CO2", "H2O"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    for ax, name in zip(axes.ravel(), species):
        for grid in sorted({grid for grid, ctype in by_case if ctype == "burning"}, key=grid_sort_key):
            rows = by_case.get((grid, "burning"))
            if rows:
                ax.plot(arr(rows, "time"), arr(rows, f"{name}_delta"), linewidth=2.0, label=grid)
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.set_title(name)
        ax.set_ylabel("raw sum delta")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    fig.suptitle("Burning species deltas across grids")
    path = plots_dir / "burning_species_deltas_all_grids.png"
    savefig(plt, path)
    return path


def plot_primary_summary(plt, by_case: dict[tuple[str, str], list[dict[str, float | int | str]]], plots_dir: Path) -> Path:
    plt.figure(figsize=(9.5, 6.2))
    colors = {"low": "#4c78a8", "mid": "#54a24b", "high": "#e45756"}
    for grid, case_type in sorted(by_case, key=lambda item: (grid_sort_key(item[0]), item[1])):
        rows = by_case[(grid, case_type)]
        linestyle = "-" if case_type == "burning" else "--"
        plt.plot(
            arr(rows, "time"),
            arr(rows, "mass_equiv_D2_norm"),
            color=colors.get(grid),
            linestyle=linestyle,
            linewidth=2.4,
            label=f"{grid} {case_type}",
        )
    plt.xlabel("time")
    plt.ylabel("mass-equivalent D2_norm")
    plt.title("Primary metric summary: mass-equivalent D2_norm across grids")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    path = plots_dir / "primary_mass_equiv_D2_norm_summary.png"
    savefig(plt, path)
    return path


def field_to_image(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    x_to_i = {float(x): i for i, x in enumerate(xs)}
    y_to_j = {float(y): i for i, y in enumerate(ys)}
    image = np.full((len(ys), len(xs)), np.nan)
    for (x, y), value in zip(coords, values):
        image[y_to_j[float(y)], x_to_i[float(x)]] = value
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    return image, extent


def make_field_snapshots(plt, case_specs: list[tuple[str, str, Path]], plots_dir: Path) -> list[Path]:
    burning = [spec for spec in case_specs if spec[1] == "burning"]
    if not burning:
        return []
    selected = next((spec for spec in burning if spec[0] == "low"), burning[0])
    grid, case_type, root = selected
    steps = common_steps(root, [("cons", 7), ("cons", 8), ("prim", 6)])
    if len(steps) < 3:
        return []
    selected_steps = [steps[0], steps[len(steps)//2], steps[-1]]
    fields = [
        ("liquid_alpha", "cons", 7, "viridis"),
        ("vapor_alpha", "cons", 8, "magma"),
        ("pressure", "prim", 6, "plasma"),
    ]
    out = []
    snap_dir = plots_dir / "field_snapshots"
    for step in selected_steps:
        for label, kind, index, cmap in fields:
            coords, values = read_field(root, kind, index, step)
            image, extent = field_to_image(coords, values)
            plt.figure(figsize=(6.2, 5.4))
            im = plt.imshow(image, origin="lower", extent=extent, cmap=cmap, aspect="equal")
            plt.colorbar(im, label=label)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"{grid} {case_type}: {label}, step {step}")
            path = snap_dir / f"{grid}_{case_type}_{label}_step_{step:06d}.png"
            savefig(plt, path)
            out.append(path)
    return out


def write_figure_summary(path: Path, generated: list[Path], snapshots: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Grid-Validation Figure Summary\n\n")
        handle.write("- `nonreacting_grid_overlay_d2_metrics.png`: nonreacting low/mid/high overlay for threshold, mass-equivalent, and alpha-weighted D2_norm.\n")
        handle.write("- `burning_grid_overlay_d2_metrics.png`: burning low/mid/high overlay for the same three D2 metrics.\n")
        handle.write("- `burning_vs_nonreacting_by_grid.png`: per-grid comparison of burning and nonreacting mass-equivalent D2_norm with threshold D2_norm as secondary dotted curves.\n")
        handle.write("- `liquid_alpha_rho_integral_all_grids.png`: liquid alpha_rho integral histories for all grid/case combinations.\n")
        handle.write("- `vapor_alpha_rho_integral_all_grids.png`: vapor alpha_rho integral histories for all grid/case combinations.\n")
        handle.write("- `pressure_mean_max_all_grids.png`: pressure mean and max histories for all available grid/case combinations.\n")
        handle.write("- `burning_species_deltas_all_grids.png`: burning-grid species sum deltas for C12H26, O2, CO2, and H2O.\n")
        handle.write("- `primary_mass_equiv_D2_norm_summary.png`: primary validation metric summary across all grids.\n")
        if snapshots:
            handle.write("- `field_snapshots/`: early/mid/late raw-field PNGs for liquid alpha, vapor alpha, and pressure from the selected baseline burning grid.\n")
        handle.write("\nGenerated files:\n")
        for item in generated + snapshots:
            handle.write(f"- `{item.name if item.parent == path.parent else item.relative_to(path.parent)}`\n")


def generate_plots(
    ts_rows: list[dict[str, float | int | str]],
    case_specs: list[tuple[str, str, Path]],
    plots_dir: Path,
    make_snapshots: bool,
) -> list[Path]:
    plt = load_plot_backend()
    by_case = rows_by_case(ts_rows)
    generated = [
        plot_grid_metric_overlay(plt, by_case, "nonreacting", plots_dir),
        plot_grid_metric_overlay(plt, by_case, "burning", plots_dir),
        plot_burning_vs_nonreacting_by_grid(plt, by_case, plots_dir),
        plot_budget_all_grids(
            plt,
            by_case,
            plots_dir,
            "liquid_alpha_rho_integral",
            "liquid alpha_rho integral",
            "liquid_alpha_rho_integral_all_grids.png",
        ),
        plot_budget_all_grids(
            plt,
            by_case,
            plots_dir,
            "vapor_alpha_rho_integral",
            "vapor alpha_rho integral",
            "vapor_alpha_rho_integral_all_grids.png",
        ),
        plot_pressure_all_grids(plt, by_case, plots_dir),
        plot_burning_species(plt, by_case, plots_dir),
        plot_primary_summary(plt, by_case, plots_dir),
    ]
    snapshots = make_field_snapshots(plt, case_specs, plots_dir) if make_snapshots else []
    summary_path = plots_dir / "FIGURE_SUMMARY.md"
    write_figure_summary(summary_path, generated, snapshots)
    return generated + snapshots + [summary_path]


def main() -> None:
    args = parse_args()
    if not args.case:
        raise SystemExit("Provide at least one --case GRID,TYPE,RUN_DIR")
    case_specs = [parse_case_spec(spec) for spec in args.case]
    rows = []
    ts_rows = []
    for spec in case_specs:
        row, case_ts_rows = analyze_case(*spec, args)
        rows.append(row)
        ts_rows.extend(case_ts_rows)
    write_csv(args.out, rows)
    comparison_path = args.comparison_out or args.out.with_name(args.out.stem + "_comparison.csv")
    comparisons = comparison_rows(rows)
    write_csv(comparison_path, comparisons)
    timeseries_path = args.timeseries_out or args.out.with_name(args.out.stem + "_timeseries.csv")
    write_csv(timeseries_path, ts_rows)
    print(f"summary_csv={args.out}")
    print(f"comparison_csv={comparison_path}")
    print(f"timeseries_csv={timeseries_path}")
    if not args.no_plots:
        plots_dir = args.plots_dir or args.out.with_name("grid_validation_plots")
        generated = generate_plots(ts_rows, case_specs, plots_dir, not args.no_snapshots)
        print(f"plots_dir={plots_dir}")
        for path in generated:
            print(f"plot={path}")


if __name__ == "__main__":
    main()
