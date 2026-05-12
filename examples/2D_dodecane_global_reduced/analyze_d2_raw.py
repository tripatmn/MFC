#!/usr/bin/env python3
"""Raw D/ analyzer for 2D equivalent-circle droplet D^2(t)."""

import argparse
import csv
import math
import pathlib
import re
import sys
from collections import deque

import numpy as np


RAW_RE = re.compile(r"^(?P<kind>[^.]+)\.(?P<index>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")


def data_dir(path):
    path = pathlib.Path(path)
    return path if path.name == "D" else path / "D"


def run_dir(path):
    path = pathlib.Path(path)
    return path.parent if path.name == "D" else path


def parse_dt(root):
    sim_path = run_dir(root) / "simulation.inp"
    if not sim_path.is_file():
        raise SystemExit(f"Missing simulation.inp beside D/: {sim_path}")
    for line in sim_path.read_text(errors="replace").splitlines():
        stripped = line.split("!", 1)[0].strip()
        if stripped.lower().startswith("dt"):
            _, value = stripped.split("=", 1)
            return float(value.strip().rstrip(","))
    raise SystemExit(f"Could not find fixed dt in {sim_path}")


def raw_files(root, kind, index):
    ddir = data_dir(root)
    files = []
    for path in ddir.glob(f"{kind}.{index}.*.*.dat"):
        match = RAW_RE.match(path.name)
        if match and int(match.group("index")) == index and match.group("kind") == kind:
            files.append((int(match.group("step")), int(match.group("rank")), path))
    return sorted(files)


def available_steps(root, kind, index):
    return sorted({step for step, _, _ in raw_files(root, kind, index)})


def read_shard(path):
    arr = np.loadtxt(path)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(f"{path} has {arr.shape[1]} columns; expected x y value")
    return arr[:, [0, 1, arr.shape[1] - 1]]


def read_field(root, kind, index, step):
    paths = [path for s, _, path in raw_files(root, kind, index) if s == step]
    if not paths:
        raise FileNotFoundError(f"No files for {kind}.{index} step {step}")

    values = {}
    duplicate_count = 0
    duplicate_max_abs_diff = 0.0
    finite = True

    for path in paths:
        arr = read_shard(path)
        finite = finite and bool(np.isfinite(arr).all())
        for x, y, value in arr:
            key = (float(x), float(y))
            value = float(value)
            if key in values:
                duplicate_count += 1
                duplicate_max_abs_diff = max(duplicate_max_abs_diff, abs(values[key] - value))
                continue
            values[key] = value

    if not values:
        raise ValueError(f"No coordinate rows found for {kind}.{index} step {step}")

    coords = np.array(list(values.keys()), dtype=float)
    vals = np.array([values[(float(x), float(y))] for x, y in coords], dtype=float)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return {
        "coords": coords[order],
        "values": vals[order],
        "duplicates": duplicate_count,
        "duplicate_max_abs_diff": duplicate_max_abs_diff,
        "finite": finite,
        "files": paths,
    }


def median_spacing(unique_coords):
    if len(unique_coords) < 2:
        return 1.0
    diffs = np.diff(np.sort(unique_coords))
    diffs = diffs[diffs > 0.0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def grid_metrics(coords):
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dx = median_spacing(xs)
    dy = median_spacing(ys)
    return xs, ys, dx, dy


def largest_component_area(coords, mask, dx, dy):
    if not bool(mask.any()):
        return 0.0, 0.0

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    x_to_i = {float(x): i for i, x in enumerate(xs)}
    y_to_j = {float(y): j for j, y in enumerate(ys)}

    occupied = set()
    for (x, y), is_liquid in zip(coords, mask):
        if is_liquid:
            occupied.add((x_to_i[float(x)], y_to_j[float(y)]))

    largest = 0
    while occupied:
        start = occupied.pop()
        count = 1
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            for nxt in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if nxt in occupied:
                    occupied.remove(nxt)
                    queue.append(nxt)
                    count += 1
        largest = max(largest, count)

    total = int(mask.sum())
    area = float(largest * dx * dy)
    fraction = float(largest / total) if total else 0.0
    return area, fraction


def estimate_alpha_bounds(alpha0):
    finite = alpha0[np.isfinite(alpha0)]
    if len(finite) == 0:
        return 0.0, 1.0
    alpha_bg = float(np.nanpercentile(finite, 1.0))
    alpha_core = float(np.nanpercentile(finite, 99.0))
    if not alpha_core > alpha_bg:
        alpha_bg = float(np.nanmin(finite))
        alpha_core = float(np.nanmax(finite))
    if not alpha_core > alpha_bg:
        alpha_core = alpha_bg + 1.0
    return alpha_bg, alpha_core


def analyze(root, args):
    dt = parse_dt(root)
    steps = available_steps(root, "cons", args.alpha_index)
    rho_steps = set(available_steps(root, "cons", args.alpha_rho_index))
    steps = [step for step in steps if step in rho_steps]
    if not steps:
        raise SystemExit("No common alpha and alpha_rho timesteps found")

    first_alpha = read_field(root, "cons", args.alpha_index, steps[0])
    _, _, dx, dy = grid_metrics(first_alpha["coords"])
    dA = dx * dy
    alpha_bg, alpha_core = estimate_alpha_bounds(first_alpha["values"])

    rows = []
    duplicate_total = 0
    duplicate_steps = []
    duplicate_max_abs_diff = 0.0

    for step in steps:
        alpha_field = read_field(root, "cons", args.alpha_index, step)
        alpha_rho_field = read_field(root, "cons", args.alpha_rho_index, step)
        alpha = alpha_field["values"]
        alpha_rho = alpha_rho_field["values"]

        step_duplicates = alpha_field["duplicates"] + alpha_rho_field["duplicates"]
        if step_duplicates:
            duplicate_steps.append(step)
        duplicate_total += step_duplicates
        duplicate_max_abs_diff = max(
            duplicate_max_abs_diff,
            alpha_field["duplicate_max_abs_diff"],
            alpha_rho_field["duplicate_max_abs_diff"],
        )

        finite = bool(np.isfinite(alpha).all() and np.isfinite(alpha_rho).all())
        areas = {}
        for cutoff in (0.1, 0.5, 0.9):
            areas[cutoff] = float(np.count_nonzero(alpha >= cutoff) * dA)

        primary_mask = alpha >= args.threshold
        primary_area = float(np.count_nonzero(primary_mask) * dA)
        equivalent_diameter = 2.0 * math.sqrt(primary_area / math.pi) if primary_area > 0.0 else 0.0
        d2 = equivalent_diameter * equivalent_diameter

        weighted = np.clip((alpha - alpha_bg) / (alpha_core - alpha_bg), 0.0, 1.0)
        component_area, component_fraction = largest_component_area(
            alpha_field["coords"], primary_mask, dx, dy
        )

        rows.append({
            "timestep": step,
            "time": step * dt,
            "liquid_area_threshold_0p1": areas[0.1],
            "liquid_area_threshold_0p5": areas[0.5],
            "liquid_area_threshold_0p9": areas[0.9],
            "liquid_alpha_integral": float(np.nansum(alpha) * dA),
            "liquid_alpha_weighted_area": float(np.nansum(weighted) * dA),
            "liquid_alpha_rho_integral": float(np.nansum(alpha_rho) * dA),
            "equivalent_diameter": equivalent_diameter,
            "D2": d2,
            "n_liquid_cells": int(np.count_nonzero(primary_mask)),
            "alpha_liq_min": float(np.nanmin(alpha)),
            "alpha_liq_max": float(np.nanmax(alpha)),
            "finite": finite,
            "largest_component_area": component_area,
            "largest_component_fraction": component_fraction,
        })

    return {
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "alpha_bg": alpha_bg,
        "alpha_core": alpha_core,
        "rows": rows,
        "duplicates": {
            "count": duplicate_total,
            "steps": duplicate_steps,
            "max_abs_diff": duplicate_max_abs_diff,
        },
    }


def fit_d2(rows, args):
    finite_rows = [row for row in rows if row["finite"] and np.isfinite(row["D2"]) and row["D2"] > 0.0]
    if not finite_rows:
        return {"status": "insufficient fit points", "reason": "no positive finite D2 values", "n_fit": 0}

    d2_initial = finite_rows[0]["D2"]
    selected = []
    for row in finite_rows:
        frac = row["D2"] / d2_initial
        if frac < args.fit_frac_min or frac > args.fit_frac_max:
            continue
        if args.fit_step_min is not None and row["timestep"] < args.fit_step_min:
            continue
        if args.fit_step_max is not None and row["timestep"] > args.fit_step_max:
            continue
        selected.append(row)

    if len(selected) < 4:
        return {
            "status": "insufficient fit points",
            "reason": "fewer than 4 points in selected fit window",
            "n_fit": len(selected),
            "d2_initial": d2_initial,
        }

    t = np.array([row["time"] for row in selected], dtype=float)
    d2 = np.array([row["D2"] for row in selected], dtype=float)
    A = np.column_stack([t, np.ones_like(t)])
    slope, intercept = np.linalg.lstsq(A, d2, rcond=None)[0]
    pred = slope * t + intercept
    ss_res = float(np.sum((d2 - pred) ** 2))
    ss_tot = float(np.sum((d2 - np.mean(d2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    stderr_slope = float("nan")
    if len(selected) > 2:
        sigma2 = ss_res / (len(selected) - 2)
        stderr_slope = math.sqrt(sigma2 / float(np.sum((t - np.mean(t)) ** 2)))

    return {
        "status": "fit ok",
        "n_fit": len(selected),
        "d2_initial": d2_initial,
        "fit_step_min": selected[0]["timestep"],
        "fit_step_max": selected[-1]["timestep"],
        "fit_t_min": selected[0]["time"],
        "fit_t_max": selected[-1]["time"],
        "slope": float(slope),
        "intercept": float(intercept),
        "K": float(-slope),
        "R2": r2,
        "stderr_slope": stderr_slope,
    }


def write_csv(path, rows):
    fields = [
        "timestep",
        "time",
        "liquid_area_threshold_0p1",
        "liquid_area_threshold_0p5",
        "liquid_area_threshold_0p9",
        "liquid_alpha_integral",
        "liquid_alpha_weighted_area",
        "liquid_alpha_rho_integral",
        "equivalent_diameter",
        "D2",
        "n_liquid_cells",
        "alpha_liq_min",
        "alpha_liq_max",
        "finite",
        "largest_component_area",
        "largest_component_fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, root, args, analysis, fit):
    dup = analysis["duplicates"]
    with path.open("w") as f:
        f.write("Raw D^2(t) analyzer summary\n")
        f.write(f"root={pathlib.Path(root)}\n")
        f.write(f"alpha_liq=cons.{args.alpha_index}\n")
        f.write(f"alpha_rho_liq=cons.{args.alpha_rho_index}\n")
        f.write(f"primary_threshold={args.threshold:.16g}\n")
        f.write(f"dt={analysis['dt']:.16e}\n")
        f.write(f"dx={analysis['dx']:.16e}\n")
        f.write(f"dy={analysis['dy']:.16e}\n")
        f.write(f"alpha_bg_initial={analysis['alpha_bg']:.16e}\n")
        f.write(f"alpha_core_initial={analysis['alpha_core']:.16e}\n")
        f.write(f"duplicate_coordinate_rows={dup['count']}\n")
        f.write(f"duplicate_steps={dup['steps']}\n")
        f.write(f"duplicate_max_abs_diff={dup['max_abs_diff']:.16e}\n")
        f.write("\nFit\n")
        f.write(f"status={fit['status']}\n")
        for key in (
            "reason",
            "n_fit",
            "d2_initial",
            "fit_step_min",
            "fit_step_max",
            "fit_t_min",
            "fit_t_max",
            "slope",
            "intercept",
            "K",
            "R2",
            "stderr_slope",
        ):
            if key in fit:
                f.write(f"{key}={fit[key]}\n")
        f.write("\nLimitations\n")
        f.write(
            "This is a 2D equivalent-circle cross-sectional diameter, "
            "not a true 3D spherical droplet diameter.\n"
        )
        f.write(
            "The reported K is therefore a 2D Cartesian cross-section trend "
            "metric unless a separate physical mapping is defined.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze raw MFC D/ output and fit a 2D equivalent-circle D^2(t) law."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Run directory containing D/ or the D/ directory itself.",
    )
    parser.add_argument("--fit-frac-min", type=float, default=0.2)
    parser.add_argument("--fit-frac-max", type=float, default=0.9)
    parser.add_argument("--fit-step-min", type=int)
    parser.add_argument("--fit-step-max", type=int)
    parser.add_argument("--alpha-index", type=int, default=7)
    parser.add_argument("--alpha-rho-index", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="d2_raw_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.root)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    analysis = analyze(root, args)
    fit = fit_d2(analysis["rows"], args)

    csv_path = outdir / "d2_timeseries.csv"
    summary_path = outdir / "d2_fit_summary.txt"
    write_csv(csv_path, analysis["rows"])
    write_summary(summary_path, root, args, analysis, fit)

    dup_count = analysis["duplicates"]["count"]
    if dup_count:
        print(f"WARNING duplicate coordinate rows detected: {dup_count}", file=sys.stderr)
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    print(f"fit_status={fit['status']}")


if __name__ == "__main__":
    main()
