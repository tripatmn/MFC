#!/usr/bin/env python3
"""Make single-field raw shock-droplet contour frames/videos.

This reads MFC raw ASCII ``D/`` and/or ``p_all/`` output directly through the
same helper path used by ``analyze_shockdroplet_air_sk54.py``.  It deliberately
avoids MFC post_process, restart_data/lustre files, and pre-exported field_data
folders.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-single-field")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics

DEFAULT_GAS_MASS_FLOOR = 1.0e-8
OPTION_A_ZOOM_TMIN_US = 4.8
OPTION_A_ZOOM_TMAX_US = 6.3
DEFAULT_ZOOM_XLIM_UM = (4600.0, 5600.0)
DEFAULT_ZOOM_YLIM_UM = (100.0, 520.0)

FIELD_CONFIG = {
    "valid_gas_temperature": {
        "folder": "temperature",
        "prefix": "temperature",
        "label": "Valid-gas temperature",
        "unit": "K",
        "scale": "linear",
        "vmin": 850.0,
        "vmax": 2200.0,
        "cmap": "inferno",
        "description": "raw reconstructed temperature masked by gas_alpha > 0.5, gas_mass > floor, alpha_liq < 0.5",
    },
    "rhoY_NC12H26": {
        "folder": "nc12h26",
        "prefix": "nc12h26",
        "label": "NC12H26 rhoY",
        "unit": r"kg m$^{-3}$",
        "scale": "log",
        "vmin": 1.0e-8,
        "vmax": 1.0e-1,
        "cmap": "magma",
        "description": "conservative species partial density rhoY_NC12H26",
    },
    "rhoY_OH": {
        "folder": "oh",
        "prefix": "oh",
        "label": "OH rhoY",
        "unit": r"kg m$^{-3}$",
        "scale": "log",
        "vmin": 1.0e-12,
        "vmax": 1.0e-5,
        "cmap": "magma",
        "description": "conservative species partial density rhoY_OH",
    },
    "rhoY_CO2": {
        "folder": "co2",
        "prefix": "co2",
        "label": "CO2 rhoY",
        "unit": r"kg m$^{-3}$",
        "scale": "log",
        "vmin": 1.0e-12,
        "vmax": 1.0e-4,
        "cmap": "magma",
        "description": "conservative species partial density rhoY_CO2",
    },
    "rhoY_H2O": {
        "folder": "h2o",
        "prefix": "h2o",
        "label": "H2O rhoY",
        "unit": r"kg m$^{-3}$",
        "scale": "log",
        "vmin": 1.0e-10,
        "vmax": 1.0e-2,
        "cmap": "magma",
        "description": "conservative species partial density rhoY_H2O",
    },
}


@dataclass(frozen=True)
class RenderJob:
    field: str
    tmin_us: float | None
    tmax_us: float | None
    xlim_um: tuple[float, float] | None
    ylim_um: tuple[float, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="MFC run folder containing raw D/ or p_all output")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for frame sequences, MP4s, and manifest")
    parser.add_argument("--case-label", default="shock-droplet", help="Short label shown in frame titles and README")
    parser.add_argument("--option-a", action="store_true", help="Render default temperature/full and late-packet NC12H26/OH zooms")
    parser.add_argument("--fields", nargs="+", choices=sorted(FIELD_CONFIG), help="Explicit fields to render when --option-a is not used")
    parser.add_argument("--tmin-us", type=float, default=None, help="Minimum physical time in microseconds for explicit fields")
    parser.add_argument("--tmax-us", type=float, default=None, help="Maximum physical time in microseconds for explicit fields")
    parser.add_argument("--xlim-um", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="Plot x-limits in microns")
    parser.add_argument("--ylim-um", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="Plot y-limits in microns")
    parser.add_argument("--fps", type=int, default=8, help="MP4 frame rate when ffmpeg is available")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum selected frames per rendered field; 0 means all")
    parser.add_argument("--no-mp4", action="store_true", help="Only write PNG frames and skip MP4 assembly")
    parser.add_argument("--gas-mass-floor", type=float, default=DEFAULT_GAS_MASS_FLOOR, help="Valid-gas temperature gas-mass floor")
    return parser.parse_args()


def jobs_from_args(args: argparse.Namespace) -> list[RenderJob]:
    if args.option_a:
        return [
            RenderJob("valid_gas_temperature", None, None, None, None),
            RenderJob("rhoY_NC12H26", OPTION_A_ZOOM_TMIN_US, OPTION_A_ZOOM_TMAX_US, DEFAULT_ZOOM_XLIM_UM, DEFAULT_ZOOM_YLIM_UM),
            RenderJob("rhoY_OH", OPTION_A_ZOOM_TMIN_US, OPTION_A_ZOOM_TMAX_US, DEFAULT_ZOOM_XLIM_UM, DEFAULT_ZOOM_YLIM_UM),
        ]
    fields = args.fields or ["valid_gas_temperature"]
    xlim = tuple(args.xlim_um) if args.xlim_um else None
    ylim = tuple(args.ylim_um) if args.ylim_um else None
    return [RenderJob(field, args.tmin_us, args.tmax_us, xlim, ylim) for field in fields]


def select_steps(steps: list[int], times: dict[int, tuple[float, str]], job: RenderJob, max_frames: int) -> list[int]:
    selected = []
    for step in steps:
        time_s, _source = times.get(step, (math.nan, "missing"))
        time_us = time_s * 1.0e6 if math.isfinite(time_s) else math.nan
        if job.tmin_us is not None and (not math.isfinite(time_us) or time_us < job.tmin_us - 1.0e-12):
            continue
        if job.tmax_us is not None and (not math.isfinite(time_us) or time_us > job.tmax_us + 1.0e-12):
            continue
        selected.append(step)
    if max_frames and max_frames > 0:
        selected = selected[:max_frames]
    return selected


def add_valid_gas_temperature(fields: dict[str, dict], gas_mass_floor: float) -> None:
    masks = gas_metrics.mask_context(fields, gas_mass_floor)
    fields["valid_gas_temperature"] = gas_metrics.reconstruct_valid_gas_temperature(fields, masks["valid_gas_thermo"])


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    if values.size == 1:
        half = 0.5
        return np.array([values[0] - half, values[0] + half], dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))


def resolve_field(fields: dict[str, dict], name: str) -> dict:
    if name not in fields or not fields[name].get("available", False):
        available = sorted(key for key, value in fields.items() if value.get("available", False))
        raise KeyError(f"Requested field {name!r} is unavailable. Available fields: {available}")
    return fields[name]


def grid_for_field(fields: dict[str, dict], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = resolve_field(fields, name)
    grid_data = raw.values_to_grid(field["values"])
    if grid_data is None:
        available = sorted(key for key, value in fields.items() if value.get("available", False))
        raise RuntimeError(f"Requested field {name!r} could not be gridded. Available fields: {available}")
    return grid_data


def alpha_grid(fields: dict[str, dict], xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    field = fields.get("liquid_alpha")
    if not field or not field.get("available", False):
        return None
    grid_data = raw.values_to_grid(field["values"])
    if grid_data is None:
        return None
    ax, ay, alpha = grid_data
    if ax.shape == xs.shape and ay.shape == ys.shape and np.allclose(ax, xs) and np.allclose(ay, ys):
        return alpha
    return None


def finite_count_in_window(xs: np.ndarray, ys: np.ndarray, values: np.ndarray, job: RenderJob) -> int:
    mask = np.isfinite(values)
    if job.xlim_um is not None:
        x_um = xs * 1.0e6
        mask &= (x_um[None, :] >= job.xlim_um[0]) & (x_um[None, :] <= job.xlim_um[1])
    if job.ylim_um is not None:
        y_um = ys * 1.0e6
        mask &= (y_um[:, None] >= job.ylim_um[0]) & (y_um[:, None] <= job.ylim_um[1])
    return int(np.count_nonzero(mask))


def draw_frame(
    out_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    alpha: np.ndarray | None,
    job: RenderJob,
    time_s: float,
    step: int,
    case_label: str,
) -> None:
    cfg = FIELD_CONFIG[job.field]
    x_edges = edges(xs * 1.0e6)
    y_edges = edges(ys * 1.0e6)
    plot_values = values.copy()
    if cfg["scale"] == "log":
        plot_values = np.where(plot_values > 0.0, plot_values, np.nan)
        norm = colors.LogNorm(vmin=cfg["vmin"], vmax=cfg["vmax"])
        mesh_kwargs = {"norm": norm, "cmap": cfg["cmap"]}
    else:
        mesh_kwargs = {"vmin": cfg["vmin"], "vmax": cfg["vmax"], "cmap": cfg["cmap"]}

    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, plot_values, shading="auto", **mesh_kwargs)
    if alpha is not None and np.any(np.isfinite(alpha)) and np.nanmin(alpha) <= 0.5 <= np.nanmax(alpha):
        ax.contour(xs * 1.0e6, ys * 1.0e6, alpha, levels=[0.5], colors="white", linewidths=1.1)
        ax.contour(xs * 1.0e6, ys * 1.0e6, alpha, levels=[0.5], colors="black", linewidths=0.35)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(f"{cfg['label']} [{cfg['unit']}]")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal", adjustable="box")
    if job.xlim_um is not None:
        ax.set_xlim(job.xlim_um)
    if job.ylim_um is not None:
        ax.set_ylim(job.ylim_um)
    time_label = f"t={time_s * 1.0e6:.3f} µs" if math.isfinite(time_s) else "t=unknown"
    window_label = "zoom" if job.xlim_um or job.ylim_um else "full domain"
    ax.set_title(f"{case_label}: {cfg['label']}, {time_label}, save {step} ({window_label})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def assemble_mp4(frame_dir: Path, prefix: str, out_path: Path, fps: int) -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    pattern = frame_dir / f"{prefix}_%04d.png"
    command = [
        ffmpeg or "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    if ffmpeg is None:
        return "ffmpeg_missing", " ".join(command)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return "ffmpeg_failed", proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown ffmpeg error"
    return "mp4_written", str(out_path)


def write_readme(out_dir: Path, args: argparse.Namespace, jobs: list[RenderJob], steps: list[int], times: dict[int, tuple[float, str]]) -> None:
    first_time = times.get(steps[0], (math.nan, "missing"))[0] if steps else math.nan
    last_time = times.get(steps[-1], (math.nan, "missing"))[0] if steps else math.nan
    lines = [
        "Raw MFC shock-droplet single-field videos",
        f"run_dir: {args.run_dir.resolve()}",
        f"case_label: {args.case_label}",
        f"raw saves found: {len(steps)}",
        f"first/last time [us]: {first_time * 1.0e6 if math.isfinite(first_time) else math.nan} / {last_time * 1.0e6 if math.isfinite(last_time) else math.nan}",
        "",
        "I/O:",
        "  Reads raw ASCII D/ and/or p_all directly via analyze_shockdroplet_air_sk54.py helpers.",
        "  Does not call MFC post_process, mfc.sh, restart_data/lustre, or field_data exports.",
        "",
        "Fields:",
    ]
    for job in jobs:
        cfg = FIELD_CONFIG[job.field]
        lines.append(f"  - {job.field}: {cfg['description']}; scale={cfg['scale']} [{cfg['vmin']}, {cfg['vmax']}]")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n")


def render(args: argparse.Namespace) -> list[dict[str, object]]:
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    steps = raw.available_steps(run_dir)
    if not steps:
        roots = [str(path) for path in raw.raw_dirs(run_dir)]
        raise RuntimeError(f"No raw D/ or p_all saves found in {run_dir}; raw roots discovered: {roots}")
    times = raw.infer_times(run_dir, steps)
    jobs = jobs_from_args(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    write_readme(out_dir, args, jobs, steps, times)

    first_time = times.get(steps[0], (math.nan, "missing"))[0]
    last_time = times.get(steps[-1], (math.nan, "missing"))[0]
    print(f"Raw saves found: {len(steps)}")
    print(f"Available time range: {first_time * 1.0e6 if math.isfinite(first_time) else math.nan:.6g} to {last_time * 1.0e6 if math.isfinite(last_time) else math.nan:.6g} us")

    for job in jobs:
        cfg = FIELD_CONFIG[job.field]
        selected = select_steps(steps, times, job, args.max_frames)
        if not selected:
            raise RuntimeError(f"No saves selected for {job.field}; requested tmin/tmax={job.tmin_us}/{job.tmax_us} us")
        field_dir = out_dir / cfg["folder"]
        frame_dir = field_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering {job.field}: {len(selected)} frames, first/last save {selected[0]} / {selected[-1]}")
        for frame_idx, step in enumerate(selected):
            time_s, time_source = times.get(step, (math.nan, "missing"))
            fields = raw.read_step_fields(run_dir, step, args.gas_mass_floor)
            add_valid_gas_temperature(fields, args.gas_mass_floor)
            xs, ys, values = grid_for_field(fields, job.field)
            alpha = alpha_grid(fields, xs, ys)
            rendered_finite_count = finite_count_in_window(xs, ys, values, job)
            if rendered_finite_count == 0:
                raise RuntimeError(f"Selected field {job.field!r} has no finite values in the requested window at save {step}")
            frame_path = frame_dir / f"{cfg['prefix']}_{frame_idx:04d}.png"
            draw_frame(frame_path, xs, ys, values, alpha, job, time_s, step, args.case_label)
            manifest.append({
                "field": job.field,
                "folder": cfg["folder"],
                "frame_index": frame_idx,
                "raw_save": step,
                "time_s": time_s,
                "time_us": time_s * 1.0e6 if math.isfinite(time_s) else math.nan,
                "time_source": time_source,
                "png": str(frame_path.relative_to(out_dir)),
                "xlim_um": "" if job.xlim_um is None else f"{job.xlim_um[0]} {job.xlim_um[1]}",
                "ylim_um": "" if job.ylim_um is None else f"{job.ylim_um[0]} {job.ylim_um[1]}",
                "finite_values_in_window": rendered_finite_count,
                "mp4_status": "not_requested" if args.no_mp4 else "pending",
                "mp4_detail": "",
            })
        if not args.no_mp4:
            status, detail = assemble_mp4(frame_dir, cfg["prefix"], field_dir / f"{cfg['prefix']}.mp4", args.fps)
            for row in manifest:
                if row["field"] == job.field:
                    row["mp4_status"] = status
                    row["mp4_detail"] = detail
            if status == "ffmpeg_missing":
                print(f"ffmpeg not found for {job.field}; run manually: {detail}")
            elif status == "ffmpeg_failed":
                print(f"ffmpeg failed for {job.field}: {detail}")
            else:
                print(f"Wrote {detail}")

    manifest_path = out_dir / "manifest.csv"
    columns = [
        "field", "folder", "frame_index", "raw_save", "time_s", "time_us", "time_source", "png",
        "xlim_um", "ylim_um", "finite_values_in_window", "mp4_status", "mp4_detail",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    render(args)


if __name__ == "__main__":
    main()
