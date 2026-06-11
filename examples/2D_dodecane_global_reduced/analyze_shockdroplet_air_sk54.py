#!/usr/bin/env python3
"""Analyze raw shock-droplet SK54 sanity output without MFC post_process.

The script reads the ASCII raw ``D/`` and/or ``p_all/`` files written by MFC and
creates compact PNG frame panels plus a CSV/text summary. It is intentionally
conservative about memory: each field is loaded one saved state at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-shockdroplet")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_RE = re.compile(r"^(?P<kind>[^.]+)\.(?P<index>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")

SK54_SPECIES_INDEX = {
    "N2": 1,
    "OH": 5,
    "HO2": 6,
    "H2O": 8,
    "H2O2": 9,
    "O2": 10,
    "CO2": 19,
    "NC12H26": 46,
}

FIELDS = {
    "pressure": ("prim", 6),
    "liquid_alpha_rho": ("cons", 1),
    "vapor_alpha_rho": ("cons", 2),
    "air_alpha_rho": ("cons", 3),
    "mixture_density": ("prim", 1),
    "liquid_alpha": ("cons", 7),
    "vapor_alpha": ("cons", 8),
    "air_alpha": ("cons", 9),
}

for species, species_index in SK54_SPECIES_INDEX.items():
    FIELDS[f"rhoY_{species}"] = ("cons", 12 + species_index)

PLOT_FIELDS = (
    "pressure",
    "temperature",
    "liquid_alpha",
    "vapor_alpha_rho",
    "rhoY_NC12H26",
    "rhoY_O2",
    "rhoY_CO2",
    "rhoY_H2O",
    "rhoY_OH",
    "rhoY_HO2",
    "rhoY_H2O2",
)

AIR_GAMMA = 1.4
AIR_CV = 739.0
VAPOR_GAMMA = 1.025
VAPOR_CV = 1956.0
R_AIR = (AIR_GAMMA - 1.0) * AIR_CV
R_VAPOR = (VAPOR_GAMMA - 1.0) * VAPOR_CV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="MFC run folder containing D/ and logs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for PNG/CSV/text diagnostics")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frame PNGs; 0 means all saved states")
    return parser.parse_args()


def raw_dirs(run_dir: Path) -> list[Path]:
    return [path for path in (run_dir / "D", run_dir / "p_all") if path.is_dir()]


def raw_file_info(path: Path) -> tuple[str, int, int, int] | None:
    match = RAW_RE.match(path.name)
    if not match:
        return None
    return (
        match.group("kind"),
        int(match.group("index")),
        int(match.group("rank")),
        int(match.group("step")),
    )


def field_files(run_dir: Path, kind: str, index: int, step: int | None = None) -> dict[int, list[Path]]:
    by_step: dict[int, list[Path]] = {}
    for root in raw_dirs(run_dir):
        for path in root.rglob(f"{kind}.{index}.*.*.dat"):
            info = raw_file_info(path)
            if info is None:
                continue
            file_kind, file_index, _rank, file_step = info
            if file_kind != kind or file_index != index:
                continue
            if step is not None and file_step != step:
                continue
            by_step.setdefault(file_step, []).append(path)
    return {key: sorted(paths) for key, paths in by_step.items()}


def available_steps(run_dir: Path) -> list[int]:
    steps: set[int] = set()
    for kind, index in FIELDS.values():
        steps.update(field_files(run_dir, kind, index))
    return sorted(steps)


def read_raw_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        arr = np.loadtxt(path)
    except (OSError, ValueError):
        return np.empty((0, 2)), np.empty(0)
    if arr.size == 0:
        return np.empty((0, 2)), np.empty(0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return np.empty((0, 2)), np.empty(0)
    return arr[:, :2].astype(float, copy=False), arr[:, -1].astype(float, copy=False)


def read_field(run_dir: Path, name: str, step: int) -> dict:
    kind, index = FIELDS[name]
    paths = field_files(run_dir, kind, index, step).get(step, [])
    values: dict[tuple[float, float], float] = {}
    row_count = 0
    duplicate_count = 0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    for path in paths:
        coords, vals = read_raw_values(path)
        row_count += int(vals.size)
        if vals.size:
            nan_count += int(np.count_nonzero(np.isnan(vals)))
            posinf_count += int(np.count_nonzero(vals == np.inf))
            neginf_count += int(np.count_nonzero(vals == -np.inf))
        for coord, value in zip(coords, vals):
            key = (float(coord[0]), float(coord[1]))
            if key in values:
                duplicate_count += 1
                continue
            values[key] = float(value)

    if not values:
        return {"available": False, "values": {}, "stats": missing_stats(paths, row_count)}
    vals = np.fromiter(values.values(), dtype=float, count=len(values))
    finite = vals[np.isfinite(vals)]
    stats = {
        "available": True,
        "files": len(paths),
        "row_count": row_count,
        "unique_count": len(values),
        "duplicates": duplicate_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": int(vals.size - finite.size),
        "min": float(np.min(finite)) if finite.size else math.nan,
        "max": float(np.max(finite)) if finite.size else math.nan,
        "mean": float(np.mean(finite)) if finite.size else math.nan,
        "sum": float(np.sum(finite)) if finite.size else math.nan,
    }
    return {"available": True, "values": values, "stats": stats}


def missing_stats(paths: list[Path], row_count: int) -> dict:
    return {
        "available": False,
        "files": len(paths),
        "row_count": row_count,
        "unique_count": 0,
        "duplicates": 0,
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "nonfinite_count": 0,
        "min": math.nan,
        "max": math.nan,
        "mean": math.nan,
        "sum": math.nan,
    }


def read_step_fields(run_dir: Path, step: int) -> dict[str, dict]:
    fields = {}
    for name in FIELDS:
        fields[name] = read_field(run_dir, name, step)
    temperature = reconstruct_temperature(fields)
    fields["temperature"] = temperature
    return fields


def reconstruct_temperature(fields: dict[str, dict]) -> dict:
    required = ("pressure", "vapor_alpha_rho", "air_alpha_rho")
    if not all(fields[name]["available"] for name in required):
        return {"available": False, "values": {}, "stats": missing_stats([], 0)}
    keys = set(fields["pressure"]["values"])
    keys &= set(fields["vapor_alpha_rho"]["values"])
    keys &= set(fields["air_alpha_rho"]["values"])
    values: dict[tuple[float, float], float] = {}
    for key in keys:
        pressure = fields["pressure"]["values"][key]
        rho_v = fields["vapor_alpha_rho"]["values"][key]
        rho_air = fields["air_alpha_rho"]["values"][key]
        rho_g = rho_v + rho_air
        if not all(math.isfinite(v) for v in (pressure, rho_v, rho_air, rho_g)) or pressure <= 0.0 or rho_g <= 1.0e-12:
            values[key] = math.nan
            continue
        r_mix = (rho_v * R_VAPOR + rho_air * R_AIR) / rho_g
        values[key] = pressure / (rho_g * r_mix) if r_mix > 0.0 else math.nan
    vals = np.fromiter(values.values(), dtype=float, count=len(values)) if values else np.empty(0)
    finite = vals[np.isfinite(vals)]
    stats = {
        "available": bool(values),
        "files": 0,
        "row_count": len(values),
        "unique_count": len(values),
        "duplicates": 0,
        "nan_count": int(np.count_nonzero(np.isnan(vals))),
        "posinf_count": int(np.count_nonzero(vals == np.inf)),
        "neginf_count": int(np.count_nonzero(vals == -np.inf)),
        "nonfinite_count": int(vals.size - finite.size),
        "min": float(np.min(finite)) if finite.size else math.nan,
        "max": float(np.max(finite)) if finite.size else math.nan,
        "mean": float(np.mean(finite)) if finite.size else math.nan,
        "sum": float(np.sum(finite)) if finite.size else math.nan,
    }
    return {"available": bool(values), "values": values, "stats": stats}


def parse_run_time_inf(run_dir: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    path = run_dir / "run_time.inf"
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        nums = [float(tok) for tok in NUM_RE.findall(stripped)]
        if len(nums) >= 3:
            out[int(nums[0])] = (float(nums[2]), float(nums[1]))
    return out


def parse_scalar_from_run_text(run_dir: Path, name: str) -> float | None:
    text = ""
    for path in (run_dir / "case.py", run_dir / "simulation.inp"):
        if path.is_file():
            text += "\n" + path.read_text(errors="replace")
    number = NUM_RE.pattern
    patterns = [
        rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*({number})',
        rf"\b{re.escape(name)}\b[^0-9+\-.]*({number})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def infer_times(run_dir: Path, steps: list[int]) -> dict[int, tuple[float, str]]:
    run_times = parse_run_time_inf(run_dir)
    t_save = parse_scalar_from_run_text(run_dir, "t_save")
    times = {}
    for ordinal, step in enumerate(steps):
        if step in run_times and math.isfinite(run_times[step][0]) and run_times[step][0] > 0.0:
            times[step] = (run_times[step][0], "run_time.inf")
        elif t_save is not None:
            times[step] = (ordinal * t_save, "inferred_save_index_t_save")
        else:
            times[step] = (math.nan, "missing")
    return times


def values_to_grid(values: dict[tuple[float, float], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not values:
        return None
    xs = np.array(sorted({key[0] for key in values}), dtype=float)
    ys = np.array(sorted({key[1] for key in values}), dtype=float)
    if xs.size == 0 or ys.size == 0:
        return None
    x_index = {float(x): i for i, x in enumerate(xs)}
    y_index = {float(y): i for i, y in enumerate(ys)}
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    for (x, y), value in values.items():
        grid[y_index[float(y)], x_index[float(x)]] = value
    return xs, ys, grid


def plot_panel(fields: dict[str, dict], names: list[str], titles: list[str], out_path: Path, time_s: float) -> None:
    ncols = 2
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name, title in zip(axes.ravel(), names, titles):
        ax.axis("on")
        field = fields.get(name)
        if not field or not field["available"]:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        grid_data = values_to_grid(field["values"])
        if grid_data is None:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        xs, ys, grid = grid_data
        extent = [float(xs.min() * 1.0e6), float(xs.max() * 1.0e6), float(ys.min() * 1.0e6), float(ys.max() * 1.0e6)]
        image = ax.imshow(grid, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if math.isfinite(time_s):
        fig.suptitle(f"Shock-droplet SK54 diagnostics, t = {time_s:.3e} s", fontsize=14)
    else:
        fig.suptitle("Shock-droplet SK54 diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def centroid_from_alpha(field: dict) -> tuple[float, float, float]:
    if not field["available"]:
        return math.nan, math.nan, math.nan
    coords = []
    weights = []
    for (x, y), value in field["values"].items():
        if math.isfinite(value) and value > 0.5:
            coords.append((x, y))
            weights.append(value)
    if not coords:
        return math.nan, math.nan, math.nan
    coords_arr = np.array(coords, dtype=float)
    weights_arr = np.array(weights, dtype=float)
    x_c = float(np.average(coords_arr[:, 0], weights=weights_arr))
    y_c = float(np.average(coords_arr[:, 1], weights=weights_arr))
    return x_c, y_c, float(np.sum(weights_arr))


def shock_location_from_pressure(field: dict) -> tuple[float, str]:
    if not field["available"]:
        return math.nan, "missing_pressure"
    grid_data = values_to_grid(field["values"])
    if grid_data is None:
        return math.nan, "empty_pressure"
    xs, _ys, grid = grid_data
    finite = np.where(np.isfinite(grid), grid, np.nan)
    if finite.size == 0 or not np.isfinite(finite).any():
        return math.nan, "nonfinite_pressure"
    p_col = np.nanmean(finite, axis=0)
    if p_col.size < 3 or not np.isfinite(p_col).any():
        return math.nan, "insufficient_pressure_profile"
    grad = np.abs(np.gradient(p_col, xs))
    if not np.isfinite(grad).any():
        return math.nan, "nonfinite_pressure_gradient"
    idx = int(np.nanargmax(grad))
    return float(xs[idx]), "max_abs_dmeanp_dx"


def row_for_state(step: int, save_index: int, time_s: float, time_source: str, fields: dict[str, dict]) -> dict:
    row = {
        "save_index": save_index,
        "step": step,
        "time_s": time_s,
        "time_source": time_source,
    }
    for name in PLOT_FIELDS:
        stats = fields.get(name, {"stats": missing_stats([], 0)})["stats"]
        prefix = name
        row[f"{prefix}_min"] = stats["min"]
        row[f"{prefix}_max"] = stats["max"]
        row[f"{prefix}_mean"] = stats["mean"]
        row[f"{prefix}_sum"] = stats["sum"]
        row[f"{prefix}_nonfinite_count"] = stats["nonfinite_count"]
    for name in ("mixture_density", "liquid_alpha_rho", "vapor_alpha_rho", "air_alpha_rho"):
        stats = fields.get(name, {"stats": missing_stats([], 0)})["stats"]
        row[f"{name}_min"] = stats["min"]
        row[f"{name}_max"] = stats["max"]
        row[f"{name}_sum"] = stats["sum"]
    row["total_nonfinite_count"] = sum(fields[name]["stats"]["nonfinite_count"] for name in fields)
    row["liquid_alpha_centroid_x"] = centroid_from_alpha(fields["liquid_alpha"])[0]
    row["liquid_alpha_centroid_y"] = centroid_from_alpha(fields["liquid_alpha"])[1]
    row["shock_front_x"] = shock_location_from_pressure(fields["pressure"])[0]
    row["shock_front_source"] = shock_location_from_pressure(fields["pressure"])[1]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
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


def write_summary(path: Path, rows: list[dict], run_dir: Path, png_count: int) -> None:
    lines = [
        "Shock-droplet air/SK54 raw diagnostics",
        f"run_dir: {run_dir}",
        f"saved states analyzed: {len(rows)}",
        f"PNG frame files written: {png_count}",
    ]
    if rows:
        final = rows[-1]
        lines.extend([
            "",
            "Final saved state:",
            f"  save_index: {final['save_index']}",
            f"  step: {final['step']}",
            f"  time_s: {final['time_s']} ({final['time_source']})",
            f"  pressure_min/max: {final['pressure_min']} / {final['pressure_max']} Pa",
            f"  temperature_min/max: {final['temperature_min']} / {final['temperature_max']} K",
            f"  total_nonfinite_count: {final['total_nonfinite_count']}",
            f"  shock_front_x: {final['shock_front_x']} m ({final['shock_front_source']})",
            f"  liquid_alpha_centroid: ({final['liquid_alpha_centroid_x']}, {final['liquid_alpha_centroid_y']}) m",
            f"  rhoY_NC12H26_sum: {final['rhoY_NC12H26_sum']}",
            f"  rhoY_O2_sum: {final['rhoY_O2_sum']}",
            f"  rhoY_CO2_sum: {final['rhoY_CO2_sum']}",
            f"  rhoY_H2O_sum: {final['rhoY_H2O_sum']}",
            f"  rhoY_OH_sum: {final['rhoY_OH_sum']}",
            f"  rhoY_HO2_sum: {final['rhoY_HO2_sum']}",
            f"  rhoY_H2O2_sum: {final['rhoY_H2O2_sum']}",
        ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = available_steps(run_dir)
    times = infer_times(run_dir, steps)
    rows: list[dict] = []
    png_count = 0
    frame_limit = len(steps) if args.max_frames <= 0 else min(args.max_frames, len(steps))

    for save_index, step in enumerate(steps):
        time_s, time_source = times.get(step, (math.nan, "missing"))
        fields = read_step_fields(run_dir, step)
        row = row_for_state(step, save_index, time_s, time_source, fields)
        rows.append(row)

        if save_index < frame_limit:
            hydro_names = ["pressure", "temperature", "liquid_alpha", "vapor_alpha_rho"]
            hydro_titles = ["Pressure [Pa]", "Gas temperature estimate [K]", "Liquid alpha", "Vapor alpha-rho"]
            chem_names = ["rhoY_NC12H26", "rhoY_O2", "rhoY_CO2", "rhoY_H2O", "rhoY_OH", "rhoY_HO2", "rhoY_H2O2"]
            chem_titles = ["NC12H26 rhoY", "O2 rhoY", "CO2 rhoY", "H2O rhoY", "OH rhoY", "HO2 rhoY", "H2O2 rhoY"]
            plot_panel(fields, hydro_names, hydro_titles, out_dir / f"frame_{save_index:06d}_hydro.png", time_s)
            plot_panel(fields, chem_names, chem_titles, out_dir / f"frame_{save_index:06d}_chem.png", time_s)
            png_count += 2

    write_csv(out_dir / "shockdroplet_summary.csv", rows)
    write_summary(out_dir / "shockdroplet_key_summary.txt", rows, run_dir, png_count)
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "saved_states": len(rows),
        "png_count": png_count,
        "field_indices": FIELDS,
        "temperature_reconstruction": {
            "formula": "T = p / ((rho_v + rho_air) * R_mix)",
            "R_mix": "(rho_v*R_vapor + rho_air*R_air)/(rho_v + rho_air)",
            "R_air": R_AIR,
            "R_vapor": R_VAPOR,
        },
    }
    (out_dir / "shockdroplet_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Analyzed {len(rows)} saved states from {run_dir}")
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
