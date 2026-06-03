#!/usr/bin/env python3
"""Analyze raw D/ output from the 0.25 mm quiescent dodecane pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cantera as ct
except Exception:  # pragma: no cover
    ct = None

from analyze_quiescent_burning_validation import (
    analyze_case,
    comparison_rows,
    ddir,
    fit_case,
    make_plots,
    run_dir,
    write_csv,
    write_fit_summary,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evap",
        type=Path,
        default=repo / "runs" / "quiescent_025mm_pilot" / "evap",
        help="Nonreacting evaporation run directory or D/ directory.",
    )
    parser.add_argument(
        "--burning",
        type=Path,
        default=None,
        help="Optional burning run directory or D/ directory.",
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=repo / "examples" / "1D_dodecane_global_smoke" / "dodecane_global_1step.yaml",
        help="Cantera mechanism for reconstructed burning temperatures.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "runs" / "quiescent_025mm_pilot" / "analysis",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fit-frac-min", type=float, default=0.80)
    parser.add_argument("--fit-frac-max", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evap is None and args.burning is None:
        raise SystemExit("Provide --evap, --burning, or both.")

    gas = None
    if ct is not None and args.mechanism.is_file():
        try:
            gas = ct.Solution(str(args.mechanism))
        except Exception:
            gas = None

    summaries: list[dict] = []
    timeseries: list[dict] = []
    grouped: dict[str, list[dict]] = {}

    if args.evap is not None and ddir(args.evap).is_dir():
        summary, rows = analyze_case("nonreacting", args.evap, args.threshold, None)
        summaries.append(summary)
        timeseries.extend(rows)
        grouped["nonreacting"] = rows
    elif args.evap is not None:
        raise SystemExit(f"Evaporation D/ directory not found: {ddir(args.evap)}")

    if args.burning is not None and ddir(args.burning).is_dir():
        summary, rows = analyze_case("burning", args.burning, args.threshold, gas)
        summaries.append(summary)
        timeseries.extend(rows)
        grouped["burning"] = rows
    elif args.burning is not None:
        raise SystemExit(f"Burning D/ directory not found: {ddir(args.burning)}")

    fits: list[dict] = []
    for summary in summaries:
        fits.extend(fit_case(summary, grouped[summary["case"]], args.fit_frac_min, args.fit_frac_max))
    comparisons = comparison_rows(summaries, fits)

    out_dir = args.out_dir
    write_csv(out_dir / "quiescent_025mm_pilot_summary.csv", summaries)
    write_csv(out_dir / "quiescent_025mm_pilot_timeseries.csv", timeseries)
    write_csv(out_dir / "quiescent_025mm_pilot_K_table.csv", fits)
    write_csv(out_dir / "quiescent_025mm_pilot_comparison.csv", comparisons)
    write_fit_summary(out_dir / "quiescent_025mm_pilot_fit_summary.txt", summaries, fits)
    plot_paths = [] if args.no_plots or "burning" not in grouped else make_plots(out_dir, grouped, fits)

    print(f"out_dir={out_dir}")
    print(f"summary_csv={out_dir / 'quiescent_025mm_pilot_summary.csv'}")
    print(f"timeseries_csv={out_dir / 'quiescent_025mm_pilot_timeseries.csv'}")
    print(f"k_table_csv={out_dir / 'quiescent_025mm_pilot_K_table.csv'}")
    if comparisons:
        print(f"comparison_csv={out_dir / 'quiescent_025mm_pilot_comparison.csv'}")
    print(f"fit_summary={out_dir / 'quiescent_025mm_pilot_fit_summary.txt'}")
    for path in plot_paths:
        print(f"plot={path}")
    for row in fits:
        if row["metric"] == "mass_equivalent" and row["method"] == "manual_threshold_0.95_0.80":
            print(
                f"{row['case']}_K_mass="
                f"{row.get('K_m2_s', '')} m^2/s, "
                f"{row.get('K_mm2_s', '')} mm^2/s, "
                f"R2={row.get('R2', '')}"
            )


if __name__ == "__main__":
    main()
