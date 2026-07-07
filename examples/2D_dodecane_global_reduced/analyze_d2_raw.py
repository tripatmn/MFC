#!/usr/bin/env python3
"""Raw D/ analyzer for 2D equivalent-circle droplet D^2(t)."""

import argparse
import csv
import math
import pathlib
import re
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def parse_dt_optional(root):
    try:
        return parse_dt(root)
    except SystemExit:
        return float("nan")


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


def safe_div(num, den, floor=1.0e-14):
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > floor)
    out[mask] = num[mask] / den[mask]
    return out


def mask_area(mask, dA):
    return float(np.count_nonzero(mask) * dA)


def mask_centroid(coords, mask):
    valid = mask & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.mean(coords[valid, 0])), float(np.mean(coords[valid, 1]))


def mask_extents(coords, mask):
    valid = mask & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.min(coords[valid, 0])),
        float(np.max(coords[valid, 0])),
        float(np.min(coords[valid, 1])),
        float(np.max(coords[valid, 1])),
    )


def velocity_stats(mask, speed):
    valid = mask & np.isfinite(speed)
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.mean(speed[valid])), float(np.max(speed[valid]))


def read_optional_values(root, kind, index, step, coords_ref):
    try:
        field = read_field(root, kind, index, step)
    except (FileNotFoundError, ValueError):
        return None, None
    if len(field["coords"]) != len(coords_ref) or not np.allclose(field["coords"], coords_ref):
        return None, field
    return field["values"], field


def point_displacement(row0, row1, prefix):
    x0 = row0.get(f"{prefix}_centroid_x_m", float("nan"))
    y0 = row0.get(f"{prefix}_centroid_y_m", float("nan"))
    x1 = row1.get(f"{prefix}_centroid_x_m", float("nan"))
    y1 = row1.get(f"{prefix}_centroid_y_m", float("nan"))
    if not all(np.isfinite(value) for value in (x0, y0, x1, y1)):
        return float("nan"), float("nan"), float("nan")
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    return dx, dy, float(math.hypot(dx, dy))


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
    dt = parse_dt_optional(root) if args.t_save is not None else parse_dt(root)
    time_source = "t_save_override" if args.t_save is not None else "simulation_dt"
    steps = available_steps(root, "cons", args.alpha_index)
    rho_steps = set(available_steps(root, "cons", args.alpha_rho_index))
    steps = [step for step in steps if step in rho_steps]
    if not steps:
        raise SystemExit("No common alpha and alpha_rho timesteps found")

    first_alpha = read_field(root, "cons", args.alpha_index, steps[0])
    xs, ys, dx, dy = grid_metrics(first_alpha["coords"])
    dA = dx * dy
    alpha_bg, alpha_core = estimate_alpha_bounds(first_alpha["values"])

    rows = []
    duplicate_total = 0
    duplicate_steps = []
    duplicate_max_abs_diff = 0.0

    for step in steps:
        alpha_field = read_field(root, "cons", args.alpha_index, step)
        alpha_rho_field = read_field(root, "cons", args.alpha_rho_index, step)
        coords = alpha_field["coords"]
        alpha = alpha_field["values"]
        alpha_rho = alpha_rho_field["values"]
        save_index = steps.index(step)
        time_value = save_index * args.t_save if args.t_save is not None else step * dt

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
        dense_core_mask = alpha > args.alpha_liq_threshold
        footprint_mask = alpha > args.footprint_threshold
        gas_mask = alpha < 0.1
        primary_area = float(np.count_nonzero(primary_mask) * dA)
        equivalent_diameter = 2.0 * math.sqrt(primary_area / math.pi) if primary_area > 0.0 else 0.0
        d2 = equivalent_diameter * equivalent_diameter
        d_over_d0 = equivalent_diameter / args.d0 if args.d0 else float("nan")
        d2_over_d02 = d2 / (args.d0*args.d0) if args.d0 else float("nan")
        dense_core_area = mask_area(dense_core_mask, dA)
        footprint_area = mask_area(footprint_mask, dA)
        dense_core_centroid = mask_centroid(coords, dense_core_mask)
        footprint_centroid = mask_centroid(coords, footprint_mask)
        dense_extents = mask_extents(coords, dense_core_mask)
        footprint_extents = mask_extents(coords, footprint_mask)
        _, dense_component_fraction = largest_component_area(coords, dense_core_mask, dx, dy)
        _, footprint_component_fraction = largest_component_area(coords, footprint_mask, dx, dy)

        weighted = np.clip((alpha - alpha_bg) / (alpha_core - alpha_bg), 0.0, 1.0)
        component_area, component_fraction = largest_component_area(
            alpha_field["coords"], primary_mask, dx, dy
        )

        velocity_duplicate_count = 0
        optional_cons = {}
        for index in (1, 2, 3, 4, 5):
            values, field = read_optional_values(root, "cons", index, step, coords)
            optional_cons[index] = values
            if field is not None:
                velocity_duplicate_count += field["duplicates"]
        arho_liq = optional_cons[1]
        arho_vap = optional_cons[2]
        arho_air = optional_cons[3]
        mom_x = optional_cons[4]
        mom_y = optional_cons[5]
        if all(values is not None for values in (arho_liq, arho_vap, arho_air, mom_x, mom_y)):
            rho_mix = arho_liq + arho_vap + arho_air
            u = safe_div(mom_x, rho_mix)
            v = safe_div(mom_y, rho_mix)
            speed = np.sqrt(u*u + v*v)
            dense_speed = velocity_stats(dense_core_mask, speed)
            footprint_speed = velocity_stats(footprint_mask, speed)
            gas_speed = velocity_stats(gas_mask, speed)
            global_speed_max = float(np.nanmax(speed)) if np.any(np.isfinite(speed)) else float("nan")
            finite = finite and bool(np.isfinite(rho_mix).all())
        else:
            dense_speed = (float("nan"), float("nan"))
            footprint_speed = (float("nan"), float("nan"))
            gas_speed = (float("nan"), float("nan"))
            global_speed_max = float("nan")

        rows.append({
            "timestep": step,
            "save_index": save_index,
            "time": time_value,
            "liquid_area_threshold_0p1": areas[0.1],
            "liquid_area_threshold_0p5": areas[0.5],
            "liquid_area_threshold_0p9": areas[0.9],
            "liquid_alpha_integral": float(np.nansum(alpha) * dA),
            "liquid_alpha_weighted_area": float(np.nansum(weighted) * dA),
            "liquid_alpha_rho_integral": float(np.nansum(alpha_rho) * dA),
            "equivalent_diameter": equivalent_diameter,
            "D2": d2,
            "D_over_D0": d_over_d0,
            "D2_over_D02": d2_over_d02,
            "n_liquid_cells": int(np.count_nonzero(primary_mask)),
            "alpha_liq_min": float(np.nanmin(alpha)),
            "alpha_liq_max": float(np.nanmax(alpha)),
            "finite": finite,
            "largest_component_area": component_area,
            "largest_component_fraction": component_fraction,
            "dense_core_area_m2": dense_core_area,
            "footprint_area_m2": footprint_area,
            "dense_core_centroid_x_m": dense_core_centroid[0],
            "dense_core_centroid_y_m": dense_core_centroid[1],
            "footprint_centroid_x_m": footprint_centroid[0],
            "footprint_centroid_y_m": footprint_centroid[1],
            "dense_core_xmin_m": dense_extents[0],
            "dense_core_xmax_m": dense_extents[1],
            "dense_core_ymin_m": dense_extents[2],
            "dense_core_ymax_m": dense_extents[3],
            "footprint_xmin_m": footprint_extents[0],
            "footprint_xmax_m": footprint_extents[1],
            "footprint_ymin_m": footprint_extents[2],
            "footprint_ymax_m": footprint_extents[3],
            "dense_core_largest_component_fraction": dense_component_fraction,
            "footprint_largest_component_fraction": footprint_component_fraction,
            "dense_core_speed_mean": dense_speed[0],
            "dense_core_speed_max": dense_speed[1],
            "footprint_speed_mean": footprint_speed[0],
            "footprint_speed_max": footprint_speed[1],
            "gas_speed_mean": gas_speed[0],
            "gas_speed_max": gas_speed[1],
            "global_speed_max": global_speed_max,
        })
        duplicate_total += velocity_duplicate_count

    return {
        "dt": dt,
        "t_save": args.t_save,
        "time_source": time_source,
        "dx": dx,
        "dy": dy,
        "x_min": float(np.min(xs)),
        "x_max": float(np.max(xs)),
        "y_min": float(np.min(ys)),
        "y_max": float(np.max(ys)),
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
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(analysis, args):
    rows = analysis["rows"]
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    dense_dx, dense_dy, dense_dist = point_displacement(first, last, "dense_core")
    foot_dx, foot_dy, foot_dist = point_displacement(first, last, "footprint")
    global_speeds = np.array([row.get("global_speed_max", float("nan")) for row in rows], dtype=float)
    max_global_speed = float(np.nanmax(global_speeds)) if np.any(np.isfinite(global_speeds)) else float("nan")
    tol_x = 0.5*analysis["dx"]
    tol_y = 0.5*analysis["dy"]
    footprint_boundary = any(
        (
            np.isfinite(row["footprint_xmin_m"]) and row["footprint_xmin_m"] <= analysis["x_min"] + tol_x
        )
        or (
            np.isfinite(row["footprint_xmax_m"]) and row["footprint_xmax_m"] >= analysis["x_max"] - tol_x
        )
        or (
            np.isfinite(row["footprint_ymin_m"]) and row["footprint_ymin_m"] <= analysis["y_min"] + tol_y
        )
        or (
            np.isfinite(row["footprint_ymax_m"]) and row["footprint_ymax_m"] >= analysis["y_max"] - tol_y
        )
        for row in rows
    )
    dense_connected = all(
        np.isfinite(row["dense_core_largest_component_fraction"])
        and row["dense_core_largest_component_fraction"] >= 0.999
        for row in rows
        if row["dense_core_area_m2"] > 0.0
    )
    return {
        "state_count": len(rows),
        "final_time_s": last["time"],
        "initial_D_over_D0": first.get("D_over_D0", float("nan")),
        "final_D_over_D0": last.get("D_over_D0", float("nan")),
        "initial_D2_over_D02": first.get("D2_over_D02", float("nan")),
        "final_D2_over_D02": last.get("D2_over_D02", float("nan")),
        "dense_core_centroid_displacement_x_m": dense_dx,
        "dense_core_centroid_displacement_y_m": dense_dy,
        "dense_core_centroid_displacement_m": dense_dist,
        "dense_core_centroid_displacement_over_D0": dense_dist/args.d0 if args.d0 and np.isfinite(dense_dist) else float("nan"),
        "footprint_centroid_displacement_x_m": foot_dx,
        "footprint_centroid_displacement_y_m": foot_dy,
        "footprint_centroid_displacement_m": foot_dist,
        "footprint_centroid_displacement_over_D0": foot_dist/args.d0 if args.d0 and np.isfinite(foot_dist) else float("nan"),
        "max_global_speed": max_global_speed,
        "footprint_reaches_domain_boundary": bool(footprint_boundary),
        "dense_core_remains_one_connected_component": bool(dense_connected),
    }


def line_plot(path, rows, series, ylabel, title):
    if not rows:
        return
    time = np.array([row["time"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    for label, key in series:
        ax.plot(time, np.array([row.get(key, float("nan")) for row in rows], dtype=float), marker="o", linewidth=1.8, markersize=3.5, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def contour_overlay(path, coords, field, title, threshold, footprint_threshold, marker=None, speed=False, contour_field=None):
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    shape = (len(ys), len(xs))
    x_to_i = {float(x): i for i, x in enumerate(xs)}
    y_to_j = {float(y): j for j, y in enumerate(ys)}
    grid = np.full(shape, np.nan, dtype=float)
    for (x, y), value in zip(coords, field):
        grid[y_to_j[float(y)], x_to_i[float(x)]] = value
    contour_grid = grid
    if contour_field is not None:
        contour_grid = np.full(shape, np.nan, dtype=float)
        for (x, y), value in zip(coords, contour_field):
            contour_grid[y_to_j[float(y)], x_to_i[float(x)]] = value
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=150)
    finite = grid[np.isfinite(grid)]
    if speed:
        vmin = 0.0
        vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
        cmap = "plasma"
    else:
        vmin = 0.0
        vmax = 1.0
        cmap = "viridis"
    if vmax <= vmin:
        vmax = vmin + 1.0
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    if np.any(np.isfinite(contour_grid)):
        ax.contour(contour_grid, levels=[footprint_threshold, threshold], colors=["white", "red"], linewidths=[1.1, 1.3], origin="lower")
    if marker is not None and all(np.isfinite(value) for value in marker):
        mx = int(np.argmin(np.abs(xs - marker[0])))
        my = int(np.argmin(np.abs(ys - marker[1])))
        ax.scatter(mx, my, marker="x", color="cyan", s=45, linewidths=1.4)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_plots(outdir, root, args, rows):
    if not rows:
        return
    if args.d0:
        line_plot(outdir / "D2_over_D02_vs_time.png", rows, [("D2/D0^2", "D2_over_D02")], "D2 / D0^2", "Dense-core D2 regression")
        line_plot(outdir / "D_over_D0_vs_time.png", rows, [("D/D0", "D_over_D0")], "D / D0", "Dense-core equivalent diameter")
    line_plot(
        outdir / "liquid_area_vs_time.png",
        rows,
        [("dense core", "dense_core_area_m2"), ("footprint", "footprint_area_m2")],
        "Area [m2]",
        "Liquid area diagnostics",
    )
    line_plot(
        outdir / "liquid_centroid_x_vs_time.png",
        rows,
        [("dense core", "dense_core_centroid_x_m"), ("footprint", "footprint_centroid_x_m")],
        "Centroid x [m]",
        "Liquid centroid x",
    )
    line_plot(
        outdir / "liquid_centroid_y_vs_time.png",
        rows,
        [("dense core", "dense_core_centroid_y_m"), ("footprint", "footprint_centroid_y_m")],
        "Centroid y [m]",
        "Liquid centroid y",
    )
    line_plot(
        outdir / "liquid_extent_x_vs_time.png",
        rows,
        [("dense xmin", "dense_core_xmin_m"), ("dense xmax", "dense_core_xmax_m"), ("footprint xmin", "footprint_xmin_m"), ("footprint xmax", "footprint_xmax_m")],
        "x extent [m]",
        "Liquid x extents",
    )
    line_plot(
        outdir / "velocity_stats_vs_time.png",
        rows,
        [
            ("dense mean", "dense_core_speed_mean"),
            ("dense max", "dense_core_speed_max"),
            ("footprint mean", "footprint_speed_mean"),
            ("footprint max", "footprint_speed_max"),
            ("gas mean", "gas_speed_mean"),
            ("gas max", "gas_speed_max"),
        ],
        "Speed proxy [m/s]",
        "Velocity proxy diagnostics",
    )

    for row in (rows[0], rows[-1]):
        step = row["timestep"]
        alpha_field = read_field(root, "cons", args.alpha_index, step)
        contour_overlay(
            outdir / f"contour_alpha_liq_overlay_{row['save_index']:06d}.png",
            alpha_field["coords"],
            alpha_field["values"],
            f"alpha_liq overlay save {row['save_index']}",
            args.alpha_liq_threshold,
            args.footprint_threshold,
            marker=(row["dense_core_centroid_x_m"], row["dense_core_centroid_y_m"]),
        )
        arho_liq, _ = read_optional_values(root, "cons", 1, step, alpha_field["coords"])
        arho_vap, _ = read_optional_values(root, "cons", 2, step, alpha_field["coords"])
        arho_air, _ = read_optional_values(root, "cons", 3, step, alpha_field["coords"])
        mom_x, _ = read_optional_values(root, "cons", 4, step, alpha_field["coords"])
        mom_y, _ = read_optional_values(root, "cons", 5, step, alpha_field["coords"])
        if all(values is not None for values in (arho_liq, arho_vap, arho_air, mom_x, mom_y)):
            rho_mix = arho_liq + arho_vap + arho_air
            speed = np.sqrt(safe_div(mom_x, rho_mix)**2 + safe_div(mom_y, rho_mix)**2)
            contour_overlay(
                outdir / f"contour_speed_overlay_{row['save_index']:06d}.png",
                alpha_field["coords"],
                speed,
                f"speed overlay save {row['save_index']}",
                args.alpha_liq_threshold,
                args.footprint_threshold,
                marker=(row["dense_core_centroid_x_m"], row["dense_core_centroid_y_m"]),
                speed=True,
                contour_field=alpha_field["values"],
            )


def write_summary(path, root, args, analysis, fit):
    dup = analysis["duplicates"]
    run_summary = summarize_run(analysis, args)
    with path.open("w") as f:
        f.write("Raw D^2(t) analyzer summary\n")
        f.write(f"root={pathlib.Path(root)}\n")
        f.write(f"alpha_liq=cons.{args.alpha_index}\n")
        f.write(f"alpha_rho_liq=cons.{args.alpha_rho_index}\n")
        f.write(f"primary_threshold={args.threshold:.16g}\n")
        f.write(f"alpha_liq_threshold={args.alpha_liq_threshold:.16g}\n")
        f.write(f"footprint_threshold={args.footprint_threshold:.16g}\n")
        f.write(f"dt={analysis['dt']:.16e}\n")
        f.write(f"time_source={analysis['time_source']}\n")
        if analysis["t_save"] is not None:
            f.write(f"t_save_override={analysis['t_save']:.16e}\n")
        f.write(f"dx={analysis['dx']:.16e}\n")
        f.write(f"dy={analysis['dy']:.16e}\n")
        f.write(f"alpha_bg_initial={analysis['alpha_bg']:.16e}\n")
        f.write(f"alpha_core_initial={analysis['alpha_core']:.16e}\n")
        f.write(f"duplicate_coordinate_rows={dup['count']}\n")
        f.write(f"duplicate_steps={dup['steps']}\n")
        f.write(f"duplicate_max_abs_diff={dup['max_abs_diff']:.16e}\n")
        f.write("\nRun summary\n")
        for key, value in run_summary.items():
            f.write(f"{key}={value}\n")
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
    parser.add_argument("--t-save", type=float)
    parser.add_argument("--d0", type=float)
    parser.add_argument("--alpha-liq-threshold", type=float, default=0.5)
    parser.add_argument("--footprint-threshold", type=float, default=1.0e-3)
    parser.add_argument("--out", default="d2_raw_analysis")
    args = parser.parse_args()
    if args.threshold == 0.5:
        args.threshold = args.alpha_liq_threshold
    return args


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
    make_plots(outdir, root, args, analysis["rows"])

    dup_count = analysis["duplicates"]["count"]
    if dup_count:
        print(f"WARNING duplicate coordinate rows detected: {dup_count}", file=sys.stderr)
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    print(f"fit_status={fit['status']}")


if __name__ == "__main__":
    main()
