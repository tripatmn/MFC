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
    Tmax_x_m: float
    Tmax_y_m: float
    D_eq_alpha05_m: float
    D_over_D0_alpha05: float
    D2_over_D02_alpha05: float
    D_eq_alpha_lo_m: float
    D_over_D0_alpha_lo: float
    D2_over_D02_alpha_lo: float
    Y_OH_max: float
    Y_HO2_max: float
    Y_H2O2_max: float
    Y_O2_max: float
    Y_NC12H26_max: float


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
        tmax = math.nan
        tmax_x = math.nan
        tmax_y = math.nan

    d05 = equivalent_diameter(alpha_liq, args.alpha_liq_threshold, cell_area)
    dlo = equivalent_diameter(alpha_liq, 1.0e-3, cell_area)

    y_fields = {
        name: species_y(arr, name, rho_gas, valid_gas)
        for name in ("OH", "HO2", "H2O2", "O2", "NC12H26")
    }

    summary = StateSummary(
        step=step,
        time_s=time_s,
        finite=bool(np.all(np.isfinite(arr))),
        nonfinite_count=int(np.size(arr) - np.count_nonzero(np.isfinite(arr))),
        pressure_proxy_min=float(np.nanmin(arr[VAR_ENERGY])),
        pressure_proxy_max=float(np.nanmax(arr[VAR_ENERGY])),
        alpha_liq_min=float(np.nanmin(alpha_liq)),
        alpha_liq_max=float(np.nanmax(alpha_liq)),
        alpha_gas_min=float(np.nanmin(alpha_gas)),
        alpha_gas_max=float(np.nanmax(alpha_gas)),
        valid_gas_cells=int(np.count_nonzero(valid_gas)),
        Tmax_gas_K=tmax,
        Tmax_x_m=tmax_x,
        Tmax_y_m=tmax_y,
        D_eq_alpha05_m=d05,
        D_over_D0_alpha05=float(d05 / args.d0) if np.isfinite(d05) else math.nan,
        D2_over_D02_alpha05=float((d05 / args.d0) ** 2) if np.isfinite(d05) else math.nan,
        D_eq_alpha_lo_m=dlo,
        D_over_D0_alpha_lo=float(dlo / args.d0) if np.isfinite(dlo) else math.nan,
        D2_over_D02_alpha_lo=float((dlo / args.d0) ** 2) if np.isfinite(dlo) else math.nan,
        Y_OH_max=float(np.nanmax(y_fields["OH"])) if np.any(np.isfinite(y_fields["OH"])) else math.nan,
        Y_HO2_max=float(np.nanmax(y_fields["HO2"])) if np.any(np.isfinite(y_fields["HO2"])) else math.nan,
        Y_H2O2_max=float(np.nanmax(y_fields["H2O2"])) if np.any(np.isfinite(y_fields["H2O2"])) else math.nan,
        Y_O2_max=float(np.nanmax(y_fields["O2"])) if np.any(np.isfinite(y_fields["O2"])) else math.nan,
        Y_NC12H26_max=float(np.nanmax(y_fields["NC12H26"])) if np.any(np.isfinite(y_fields["NC12H26"])) else math.nan,
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


def plot_timeseries(summaries: list[StateSummary], out_dir: Path) -> None:
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
    line_plot("D_over_D0_vs_time.png", [("alpha_liq > 0.5", np.array([s.D_over_D0_alpha05 for s in summaries])), ("alpha_liq > 1e-3", np.array([s.D_over_D0_alpha_lo for s in summaries]))], "D / D0", "Equivalent droplet diameter")
    line_plot("D2_over_D02_vs_time.png", [("alpha_liq > 0.5", np.array([s.D2_over_D02_alpha05 for s in summaries])), ("alpha_liq > 1e-3", np.array([s.D2_over_D02_alpha_lo for s in summaries]))], "D2 / D0^2", "Equivalent D2 regression")
    line_plot(
        "species_max_vs_time.png",
        [
            ("OH", np.array([s.Y_OH_max for s in summaries])),
            ("HO2", np.array([s.Y_HO2_max for s in summaries])),
            ("H2O2", np.array([s.Y_H2O2_max for s in summaries])),
            ("NC12H26", np.array([s.Y_NC12H26_max for s in summaries])),
        ],
        "Max gas-masked Yk",
        "Gas-masked species maxima",
    )


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


def max_dtdt(summaries: list[StateSummary]) -> float:
    best = math.nan
    for a, b in zip(summaries, summaries[1:]):
        dt = b.time_s - a.time_s
        if dt > 0 and np.isfinite(a.Tmax_gas_K) and np.isfinite(b.Tmax_gas_K):
            slope = (b.Tmax_gas_K - a.Tmax_gas_K) / dt
            if not np.isfinite(best) or slope > best:
                best = slope
    return best


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
    if not args.skip_plots:
        plot_timeseries(summaries, args.out_dir)

    first_induction = first_threshold(summaries, 820.0)
    total_ignition = first_threshold(summaries, 2000.0)
    final_time = summaries[-1].time_s if summaries else math.nan
    summary_json = {
        "run_dir": str(args.run_dir),
        "state_count": len(summaries),
        "first_step": summaries[0].step if summaries else None,
        "last_step": summaries[-1].step if summaries else None,
        "final_time_s": final_time,
        "ambient_temperature_K": 800.0,
        "first_induction_Tmax_ge_820K": first_induction,
        "total_ignition_Tmax_ge_2000K": total_ignition,
        "max_dTmax_dt_K_s": max_dtdt(summaries),
        "hot_ignition_by_0p15s": bool(total_ignition["time_s"] is not None and total_ignition["time_s"] <= 0.15),
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
        f"First induction proxy Tmax>=820 K: {first_induction}",
        f"Total ignition proxy Tmax>=2000 K: {total_ignition}",
        f"Max dTmax/dt: {summary_json['max_dTmax_dt_K_s']:.6e} K/s",
        f"Hot ignition by 0.15 s: {summary_json['hot_ignition_by_0p15s']}",
        "",
        "Temperature is a gas-only proxy; liquid-dominated cells are excluded.",
    ]
    (args.out_dir / "tanabe_restart_summary.txt").write_text("\n".join(lines) + "\n")

    print(f"[done] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
