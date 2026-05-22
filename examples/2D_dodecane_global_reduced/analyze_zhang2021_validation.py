#!/usr/bin/env python3
"""Analyze Zhang-style 1 mm dodecane validation raw D/ outputs.

This is an offline raw-output analyzer. It does not run MFC or post_process.

Example:
  python examples/2D_dodecane_global_reduced/analyze_zhang2021_validation.py \
    --evap runs/zhang2021_validation/evap_1mm \
    --burning runs/zhang2021_validation/burning_1mm \
    --out runs/zhang2021_validation/zhang2021_validation_summary.csv \
    --plots-dir runs/zhang2021_validation/plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_grid_validation import (
    SPECIES_FIELDS,
    analyze_case,
    comparison_rows,
    load_plot_backend,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evap", type=Path, default=None, help="Nonreacting evaporation run folder or D/ folder.")
    parser.add_argument("--burning", type=Path, default=None, help="Burning run folder or D/ folder.")
    parser.add_argument("--out", type=Path, default=Path("zhang2021_validation_summary.csv"))
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fit-frac-min", type=float, default=0.80)
    parser.add_argument("--fit-frac-max", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def timeseries_path(summary_path: Path) -> Path:
    return summary_path.with_name("zhang2021_validation_timeseries.csv")


def fit_summary_path(summary_path: Path) -> Path:
    return summary_path.with_name("zhang2021_validation_fit_summary.txt")


def default_plots_dir(summary_path: Path) -> Path:
    return summary_path.with_name("zhang2021_validation_plots")


def arr(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def write_fit_summary(path: Path, summaries: list[dict[str, float | int | str]], comparisons: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Zhang 2021-style dodecane validation fit summary\n")
        handle.write("Primary metric: mass-equivalent D2_norm from liquid alpha_rho.\n")
        handle.write("Secondary metric: threshold D2_norm from alpha_liq >= 0.5.\n")
        handle.write("Fit window is selected by threshold D2_norm.\n\n")
        for row in summaries:
            handle.write(f"[{row['case_type']}]\n")
            handle.write(f"run_dir = {row['run_dir']}\n")
            handle.write(f"steps = {row['first_step']}..{row['last_step']}\n")
            handle.write(f"mass_equiv_D2_norm_final = {float(row['mass_equiv_D2_norm_last']):.16e}\n")
            handle.write(f"threshold_D2_norm_final = {float(row['D2_threshold_final_norm']):.16e}\n")
            handle.write(f"K_mass = {float(row['K_mass']):.16e}\n")
            handle.write(f"R2_mass = {float(row['R2_mass']):.16e}\n")
            handle.write(f"K_threshold = {float(row['K_threshold']):.16e}\n")
            handle.write(f"R2_threshold = {float(row['R2_threshold']):.16e}\n")
            handle.write(f"liquid_alpha_rho_delta = {float(row['liquid_alpha_rho_delta']):.16e}\n")
            handle.write(f"vapor_alpha_rho_delta = {float(row['vapor_alpha_rho_delta']):.16e}\n")
            for name in SPECIES_FIELDS:
                value = row.get(f"{name}_delta", "")
                if value != "":
                    handle.write(f"{name}_delta = {float(value):.16e}\n")
            handle.write("\n")
        for row in comparisons:
            handle.write("[burning_vs_nonreacting]\n")
            for key, value in row.items():
                handle.write(f"{key} = {value}\n")


def make_plots(plots_dir: Path, grouped_rows: dict[str, list[dict[str, float | int | str]]]) -> list[Path]:
    plt = load_plot_backend()
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    styles = {
        "nonreacting": {"label": "Nonreacting", "color": "tab:blue"},
        "burning": {"label": "Burning", "color": "tab:red"},
    }

    metric_specs = [
        ("D2_threshold_norm", "Threshold D2 / D0^2", "threshold_d2_norm.png"),
        ("mass_equiv_D2_norm", "Mass-equivalent D2 / D0^2", "mass_equiv_d2_norm.png"),
        ("alpha_weighted_D2_norm", "Alpha-weighted D2 / D0^2", "alpha_weighted_d2_norm.png"),
        ("liquid_alpha_rho_integral", "Liquid alpha_rho integral", "liquid_alpha_rho_integral.png"),
        ("vapor_alpha_rho_integral", "Vapor alpha_rho integral", "vapor_alpha_rho_integral.png"),
        ("pressure_mean", "Pressure mean [Pa]", "pressure_mean.png"),
        ("pressure_max", "Pressure max [Pa]", "pressure_max.png"),
    ]
    for key, ylabel, filename in metric_specs:
        plt.figure(figsize=(7.0, 4.5))
        for case_type, rows in grouped_rows.items():
            y = arr(rows, key)
            if not np.isfinite(y).any():
                continue
            plt.plot(arr(rows, "time"), y, linewidth=2.0, **styles[case_type])
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(f"Zhang-style validation: {ylabel}")
        plt.legend()
        plt.tight_layout()
        path = plots_dir / filename
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    if "burning" in grouped_rows:
        species = ["C12H26", "O2", "CO2", "H2O"]
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True)
        rows = grouped_rows["burning"]
        for ax, name in zip(axes.ravel(), species):
            key = f"{name}_delta"
            y = arr(rows, key)
            ax.plot(arr(rows, "time"), y, linewidth=2.0)
            ax.set_title(name)
            ax.set_ylabel("delta sum")
            ax.grid(True, alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")
        fig.suptitle("Burning species deltas")
        fig.tight_layout()
        path = plots_dir / "burning_species_deltas.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if {"nonreacting", "burning"}.issubset(grouped_rows):
        plt.figure(figsize=(7.0, 4.5))
        for case_type in ("nonreacting", "burning"):
            rows = grouped_rows[case_type]
            plt.plot(arr(rows, "time"), arr(rows, "mass_equiv_D2_norm"), linewidth=2.4, **styles[case_type])
            plt.plot(
                arr(rows, "time"),
                arr(rows, "D2_threshold_norm"),
                color=styles[case_type]["color"],
                linestyle=":",
                linewidth=1.8,
                label=f"{styles[case_type]['label']} threshold",
            )
        plt.xlabel("Time [s]")
        plt.ylabel("D2 / D0^2")
        plt.title("Zhang-style burning/nonreacting comparison")
        plt.legend()
        plt.tight_layout()
        path = plots_dir / "burning_vs_nonreacting_d2_norm.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def main() -> None:
    args = parse_args()
    if args.evap is None and args.burning is None:
        raise SystemExit("Provide --evap, --burning, or both.")

    summaries: list[dict[str, float | int | str]] = []
    timeseries: list[dict[str, float | int | str]] = []
    grouped_rows: dict[str, list[dict[str, float | int | str]]] = {}

    if args.evap is not None:
        summary, rows = analyze_case("zhang2021", "nonreacting", args.evap, args)
        summaries.append(summary)
        timeseries.extend(rows)
        grouped_rows["nonreacting"] = rows
    if args.burning is not None:
        summary, rows = analyze_case("zhang2021", "burning", args.burning, args)
        summaries.append(summary)
        timeseries.extend(rows)
        grouped_rows["burning"] = rows

    comparisons = comparison_rows(summaries)
    write_csv(args.out, summaries)
    write_csv(timeseries_path(args.out), timeseries)
    if comparisons:
        write_csv(args.out.with_name("zhang2021_validation_comparison.csv"), comparisons)
    write_fit_summary(fit_summary_path(args.out), summaries, comparisons)

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = make_plots(args.plots_dir or default_plots_dir(args.out), grouped_rows)

    print(f"summary_csv={args.out}")
    print(f"timeseries_csv={timeseries_path(args.out)}")
    print(f"fit_summary={fit_summary_path(args.out)}")
    if comparisons:
        print(f"comparison_csv={args.out.with_name('zhang2021_validation_comparison.csv')}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
