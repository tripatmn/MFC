#!/usr/bin/env python3
"""Create shock-droplet streamwise and cross-stream line-cut diagnostics.

This reads the same raw ASCII ``D/`` and ``p_all/`` files used by the
shock-droplet analyzers. It does not require MFC post_process and does not run
MFC. The output is designed for inspecting where pressure, valid gas
temperature, fuel vapor, and radical/product species sit relative to the
droplet and shock.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-shockdroplet-linecuts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics


SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO2", "H2O")
HYDRO_FIELDS = ("liquid_alpha", "gas_alpha", "pressure", "temperature_valid_gas")
SPECIES_FIELDS = tuple(f"rhoY_{species}" for species in SPECIES)
FIELD_LABELS = {
    "liquid_alpha": r"$\alpha_\ell$",
    "gas_alpha": r"$\alpha_g$",
    "pressure": "Pressure [Pa]",
    "temperature_valid_gas": "Valid gas T [K]",
    "gas_mass": r"$\alpha\rho_g$",
    **{f"rhoY_{species}": rf"$\rho Y_{{{species}}}$" for species in SPECIES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run folder containing raw D/ or p_all/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for line-cut plots and CSVs")
    parser.add_argument("--times-us", type=float, nargs="*", default=None, help="Optional target saved times in microseconds")
    parser.add_argument("--gas-mass-floor", type=float, default=1.0e-8)
    parser.add_argument(
        "--compare-run-dir",
        type=Path,
        default=None,
        help="Reserved for later grid-comparison mode. Single-run plotting is implemented now.",
    )
    return parser.parse_args()


def finite_float(value: float | int | str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_derived_fields(fields: dict[str, dict], gas_mass_floor: float) -> dict[str, dict]:
    masks = gas_metrics.mask_context(fields, gas_mass_floor)
    fields["temperature_valid_gas"] = gas_metrics.reconstruct_valid_gas_temperature(fields, masks["valid_gas_thermo"])

    keys = set()
    for name in ("liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"):
        field = fields.get(name)
        if field and field["available"]:
            keys = set(field["values"]) if not keys else keys & set(field["values"])
    gas_alpha_values = {}
    gas_mass_values = {}
    for key in keys:
        gas_alpha_values[key] = fields["vapor_alpha"]["values"][key] + fields["air_alpha"]["values"][key]
        gas_mass_values[key] = fields["vapor_alpha_rho"]["values"][key] + fields["air_alpha_rho"]["values"][key]
    fields["gas_alpha"] = {
        "available": bool(gas_alpha_values),
        "values": gas_alpha_values,
        "stats": raw.stats_from_values(gas_alpha_values, available=bool(gas_alpha_values)),
    }
    fields["gas_mass"] = {
        "available": bool(gas_mass_values),
        "values": gas_mass_values,
        "stats": raw.stats_from_values(gas_mass_values, available=bool(gas_mass_values)),
    }
    return fields


def grid_for(fields: dict[str, dict], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    field = fields.get(name)
    if not field or not field["available"]:
        return None
    return raw.values_to_grid(field["values"])


def estimate_cell_geometry(fields: dict[str, dict]) -> tuple[float, float, float]:
    return gas_metrics.estimate_cell_area(fields)


def droplet_geometry(fields: dict[str, dict]) -> dict[str, float | str]:
    alpha = fields.get("liquid_alpha")
    if not alpha or not alpha["available"]:
        return {
            "xc": math.nan,
            "yc": math.nan,
            "radius": math.nan,
            "area": math.nan,
            "method": "missing_liquid_alpha",
        }

    dx, dy, cell_area = estimate_cell_geometry(fields)
    coords = []
    weights = []
    area_alpha = 0.0
    area_threshold = 0.0
    for (x, y), value in alpha["values"].items():
        if not math.isfinite(value):
            continue
        clipped = min(max(value, 0.0), 1.0)
        if clipped > 0.0:
            coords.append((x, y))
            weights.append(clipped)
            if math.isfinite(cell_area):
                area_alpha += clipped * cell_area
        if value > 0.5 and math.isfinite(cell_area):
            area_threshold += cell_area

    if coords and sum(weights) > 0.0:
        arr = np.array(coords, dtype=float)
        w = np.array(weights, dtype=float)
        xc = float(np.average(arr[:, 0], weights=w))
        yc = float(np.average(arr[:, 1], weights=w))
    else:
        xc, yc, _ = raw.centroid_from_alpha(alpha)

    area = area_alpha if area_alpha > 0.0 else area_threshold
    radius = math.sqrt(area / math.pi) if area > 0.0 else math.nan
    return {"xc": xc, "yc": yc, "radius": radius, "area": area, "method": "alpha_weighted_area"}


def finite_max_key(fields: dict[str, dict], name: str) -> tuple[float, tuple[float, float] | None]:
    field = fields.get(name)
    if not field or not field["available"]:
        return math.nan, None
    best = -math.inf
    best_key = None
    for key, value in field["values"].items():
        if math.isfinite(value) and value > best:
            best = value
            best_key = key
    if best_key is None:
        return math.nan, None
    return float(best), best_key


def local(fields: dict[str, dict], name: str, key: tuple[float, float] | None) -> float:
    if key is None:
        return math.nan
    field = fields.get(name)
    if not field or not field["available"]:
        return math.nan
    return float(field["values"].get(key, math.nan))


def classify_location(fields: dict[str, dict], key: tuple[float, float] | None) -> str:
    if key is None:
        return "missing"
    alpha_liq = local(fields, "liquid_alpha", key)
    gas_alpha = local(fields, "gas_alpha", key)
    if not math.isfinite(alpha_liq):
        return "unknown"
    if alpha_liq > 0.99:
        return "liquid_dominated"
    if 0.01 < alpha_liq < 0.99:
        return "interface"
    if math.isfinite(gas_alpha) and gas_alpha > 0.9 and alpha_liq < 0.1:
        return "gas_dominant"
    return "other"


def select_steps(run_dir: Path, times_us: list[float] | None) -> list[tuple[int, int, float, str]]:
    steps = raw.available_steps(run_dir)
    times = raw.infer_times(run_dir, steps)
    if not times_us:
        return [(ordinal, step, *times.get(step, (math.nan, "missing"))) for ordinal, step in enumerate(steps)]

    selected = []
    used_steps = set()
    for target_us in times_us:
        target_s = target_us * 1.0e-6
        best = None
        best_dt = math.inf
        for ordinal, step in enumerate(steps):
            time_s, source = times.get(step, (math.nan, "missing"))
            if not math.isfinite(time_s):
                continue
            diff = abs(time_s - target_s)
            if diff < best_dt:
                best = (ordinal, step, time_s, source)
                best_dt = diff
        if best is not None and best[1] not in used_steps:
            selected.append(best)
            used_steps.add(best[1])
    return selected


def nearest_index(values: np.ndarray, target: float) -> int | None:
    if values.size == 0 or not math.isfinite(target):
        return None
    return int(np.nanargmin(np.abs(values - target)))


def unique_cuts(cuts: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for label, value in cuts:
        if not math.isfinite(value):
            continue
        if any(abs(value - old) <= max(abs(value), abs(old), 1.0) * 1.0e-10 for _old_label, old in out):
            continue
        out.append((label, value))
    return out


def linecut_data(
    fields: dict[str, dict],
    orientation: str,
    cuts: list[tuple[str, float]],
    field_names: tuple[str, ...],
) -> tuple[list[dict], dict[str, float]]:
    base = grid_for(fields, "pressure") or grid_for(fields, "liquid_alpha")
    if base is None:
        return [], {}
    xs, ys, _grid = base
    rows: list[dict] = []
    actual_positions: dict[str, float] = {}
    for cut_label, target in cuts:
        if orientation == "streamwise":
            idx = nearest_index(ys, target)
            if idx is None:
                continue
            actual = float(ys[idx])
            coord_values = xs
            coord_name = "x_m"
            fixed_name = "y_m"
        else:
            idx = nearest_index(xs, target)
            if idx is None:
                continue
            actual = float(xs[idx])
            coord_values = ys
            coord_name = "y_m"
            fixed_name = "x_m"
        actual_positions[cut_label] = actual

        field_grids = {name: grid_for(fields, name) for name in field_names}
        for n, coord in enumerate(coord_values):
            row = {"cut_label": cut_label, coord_name: float(coord), fixed_name: actual}
            for name, grid_data in field_grids.items():
                if grid_data is None:
                    row[name] = math.nan
                    continue
                gx, gy, grid = grid_data
                if orientation == "streamwise":
                    gidx = nearest_index(gy, actual)
                    row[name] = float(grid[gidx, n]) if gidx is not None and n < grid.shape[1] else math.nan
                else:
                    gidx = nearest_index(gx, actual)
                    row[name] = float(grid[n, gidx]) if gidx is not None and n < grid.shape[0] else math.nan
            rows.append(row)
    return rows, actual_positions


def write_cut_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_cut(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row["cut_label"]), []).append(row)
    return out


def plot_hydro(path: Path, rows: list[dict], orientation: str, time_s: float, geom: dict[str, float | str]) -> None:
    by_cut = rows_by_cut(rows)
    coord = "x_m" if orientation == "streamwise" else "y_m"
    xlabel = "x [um]" if orientation == "streamwise" else "y [um]"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), squeeze=False)
    for ax, field_name in zip(axes.ravel(), HYDRO_FIELDS):
        for label, cut_rows in by_cut.items():
            x = np.array([finite_float(row.get(coord)) * 1.0e6 for row in cut_rows])
            y = np.array([finite_float(row.get(field_name)) for row in cut_rows])
            ax.plot(x, y, linewidth=1.7, label=label)
        ax.set_title(FIELD_LABELS[field_name])
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        mark_droplet(ax, orientation, geom)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{orientation.replace('_', ' ').title()} hydro cuts, t = {time_s:.3e} s")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_species(path: Path, rows: list[dict], orientation: str, time_s: float, geom: dict[str, float | str]) -> None:
    by_cut = rows_by_cut(rows)
    coord = "x_m" if orientation == "streamwise" else "y_m"
    xlabel = "x [um]" if orientation == "streamwise" else "y [um]"
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, field_name in zip(axes.ravel(), SPECIES_FIELDS):
        ax.axis("on")
        positive_seen = False
        for label, cut_rows in by_cut.items():
            x = np.array([finite_float(row.get(coord)) * 1.0e6 for row in cut_rows])
            y = np.array([finite_float(row.get(field_name)) for row in cut_rows])
            positive_seen = positive_seen or bool(np.any(np.isfinite(y) & (y > 0.0)))
            y_plot = np.where(np.isfinite(y) & (y > 0.0), y, 1.0e-30)
            ax.plot(x, y_plot, linewidth=1.7, label=label)
        if positive_seen:
            ax.set_yscale("log")
        ax.set_title(FIELD_LABELS[field_name])
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25, which="both")
        mark_droplet(ax, orientation, geom)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{orientation.replace('_', ' ').title()} species cuts, t = {time_s:.3e} s")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def mark_droplet(ax: plt.Axes, orientation: str, geom: dict[str, float | str]) -> None:
    xc = finite_float(geom.get("xc"))
    yc = finite_float(geom.get("yc"))
    radius = finite_float(geom.get("radius"))
    center = xc if orientation == "streamwise" else yc
    if math.isfinite(center):
        ax.axvline(center * 1.0e6, color="0.35", linestyle="--", linewidth=0.9, alpha=0.8)
    if math.isfinite(center) and math.isfinite(radius):
        ax.axvline((center - radius) * 1.0e6, color="0.45", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.axvline((center + radius) * 1.0e6, color="0.45", linestyle=":", linewidth=0.9, alpha=0.8)


def state_summary(
    fields: dict[str, dict],
    step: int,
    save_index: int,
    time_s: float,
    geom: dict[str, float | str],
    x_positions: dict[str, float],
    y_positions: dict[str, float],
) -> dict:
    row = {
        "save_index": save_index,
        "step": step,
        "time_s": time_s,
        "xc": geom["xc"],
        "yc": geom["yc"],
        "radius": geom["radius"],
        "liquid_area": geom["area"],
        "geometry_method": geom["method"],
        "raw_Tmax_in_liquid_dominated": False,
    }
    raw_tmax, raw_tmax_key = finite_max_key(fields, "temperature")
    row["raw_Tmax"] = raw_tmax
    row["raw_Tmax_location"] = classify_location(fields, raw_tmax_key)
    row["raw_Tmax_in_liquid_dominated"] = row["raw_Tmax_location"] == "liquid_dominated"
    for species in ("OH", "HO2", "CO2", "H2O"):
        value, key = finite_max_key(fields, f"rhoY_{species}")
        row[f"{species}_max"] = value
        row[f"{species}_max_x"] = key[0] if key is not None else math.nan
        row[f"{species}_max_y"] = key[1] if key is not None else math.nan
        row[f"{species}_max_region"] = classify_location(fields, key)
    row["selected_x_cuts_m"] = x_positions
    row["selected_y_cuts_m"] = y_positions
    return row


def analyze(run_dir: Path, out_dir: Path, times_us: list[float] | None, gas_mass_floor: float) -> list[dict]:
    stream_dir = out_dir / "streamwise"
    cross_dir = out_dir / "cross_stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)

    selected = select_steps(run_dir, times_us)
    summaries: list[dict] = []
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "gas_mass_floor": gas_mass_floor,
        "requested_times_us": times_us,
        "comparison_mode": "TODO: --compare-run-dir is accepted but not implemented for plotting yet.",
    }

    for save_index, step, time_s, time_source in selected:
        fields = raw.read_step_fields(run_dir, step, gas_mass_floor)
        fields = build_derived_fields(fields, gas_mass_floor)
        geom = droplet_geometry(fields)

        oh_max, oh_key = finite_max_key(fields, "rhoY_OH")
        ho2_max, ho2_key = finite_max_key(fields, "rhoY_HO2")
        shock_x, shock_method = raw.shock_location_from_pressure(fields["pressure"])
        xc = finite_float(geom["xc"])
        yc = finite_float(geom["yc"])
        radius = finite_float(geom["radius"])

        y_cuts = unique_cuts(
            [
                ("yc", yc),
                ("yc_plus_R", yc + radius),
                ("yc_minus_R", yc - radius),
                ("OH_max_y", oh_key[1] if oh_key else math.nan),
                ("HO2_max_y", ho2_key[1] if ho2_key else math.nan),
            ]
        )
        x_cuts = unique_cuts(
            [
                ("xc", xc),
                ("xc_minus_R", xc - radius),
                ("xc_plus_R", xc + radius),
                ("shock_front", shock_x),
                ("OH_max_x", oh_key[0] if oh_key else math.nan),
                ("HO2_max_x", ho2_key[0] if ho2_key else math.nan),
            ]
        )

        all_fields = HYDRO_FIELDS + SPECIES_FIELDS
        stream_rows, y_positions = linecut_data(fields, "streamwise", y_cuts, all_fields)
        cross_rows, x_positions = linecut_data(fields, "cross_stream", x_cuts, all_fields)

        tag = f"t{save_index:04d}"
        write_cut_csv(stream_dir / f"streamwise_cut_data_{tag}.csv", stream_rows)
        write_cut_csv(cross_dir / f"cross_stream_cut_data_{tag}.csv", cross_rows)
        if stream_rows:
            plot_hydro(stream_dir / f"streamwise_hydro_{tag}.png", stream_rows, "streamwise", time_s, geom)
            plot_species(stream_dir / f"streamwise_species_{tag}.png", stream_rows, "streamwise", time_s, geom)
        if cross_rows:
            plot_hydro(cross_dir / f"cross_stream_hydro_{tag}.png", cross_rows, "cross_stream", time_s, geom)
            plot_species(cross_dir / f"cross_stream_species_{tag}.png", cross_rows, "cross_stream", time_s, geom)

        summary = state_summary(fields, step, save_index, time_s, geom, x_positions, y_positions)
        summary["time_source"] = time_source
        summary["shock_front_x"] = shock_x
        summary["shock_front_method"] = shock_method
        summary["OH_max_value_for_cut"] = oh_max
        summary["HO2_max_value_for_cut"] = ho2_max
        summaries.append(summary)

    (out_dir / "linecut_metadata.json").write_text(
        json.dumps({"metadata": metadata, "states": summaries}, indent=2, allow_nan=True) + "\n"
    )
    write_summary(out_dir / "linecut_summary.txt", summaries, selected)
    return summaries


def write_summary(path: Path, summaries: list[dict], selected: list[tuple[int, int, float, str]]) -> None:
    lines = [
        "Shock-droplet line-cut diagnostics",
        f"saved states selected: {len(selected)}",
        f"saved states analyzed: {len(summaries)}",
        "",
    ]
    if not selected:
        lines.extend(
            [
                "No raw saved states were found. Expected raw D/ and/or p_all/ folders in the run directory.",
                "",
            ]
        )
        path.write_text("\n".join(lines))
        return
    for item in summaries:
        lines.extend(
            [
                f"save_index={item['save_index']} step={item['step']} time={item['time_s']:.6e} s",
                f"  droplet centroid/radius: ({item['xc']}, {item['yc']}) / {item['radius']} m",
                f"  streamwise y cuts: {item['selected_y_cuts_m']}",
                f"  cross-stream x cuts: {item['selected_x_cuts_m']}",
                f"  OH max: {item['OH_max']} at ({item['OH_max_x']}, {item['OH_max_y']}) region={item['OH_max_region']}",
                f"  HO2 max: {item['HO2_max']} at ({item['HO2_max_x']}, {item['HO2_max_y']}) region={item['HO2_max_region']}",
                f"  CO2 max: {item['CO2_max']} at ({item['CO2_max_x']}, {item['CO2_max_y']}) region={item['CO2_max_region']}",
                f"  H2O max: {item['H2O_max']} at ({item['H2O_max_x']}, {item['H2O_max_y']}) region={item['H2O_max_region']}",
                f"  raw Tmax: {item['raw_Tmax']} region={item['raw_Tmax_location']}",
            ]
        )
        if item["raw_Tmax_in_liquid_dominated"]:
            lines.append("  WARNING: raw Tmax is liquid-dominated; do not interpret it as physical gas temperature.")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.compare_run_dir is not None:
        print("Note: --compare-run-dir is accepted for future comparison mode; single-run plotting is used now.")
    summaries = analyze(args.run_dir.resolve(), args.out_dir.resolve(), args.times_us, args.gas_mass_floor)
    print(f"Analyzed {len(summaries)} saved states from {args.run_dir}")
    print(f"Wrote line cuts to {args.out_dir}")


if __name__ == "__main__":
    main()
