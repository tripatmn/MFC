"""Command-line interface for mfc-post."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .analysis import analyze_case
from .inspect import inspect_case
from .plotting import PLOT_SETS, plot_history
from .process import process_case
from .render import RENDER_FIELDS, render_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfc-post", description="Inspect MFC output without combining source families.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inventory output sources in an MFC case directory")
    inspect_parser.add_argument("case", help="path to an MFC case/run directory")
    inspect_parser.add_argument("--format", choices=("text", "json"), default="text")
    inspect_parser.add_argument("--mechanism", help="explicit Cantera YAML when run metadata is not self-contained")
    inspect_parser.add_argument("--phase", help="explicit mechanism phase name")
    process_parser = subparsers.add_parser("process", help="compute the initial conservative inventory and quality slice")
    process_parser.add_argument("case", help="path to an MFC case/run directory")
    process_parser.add_argument("--execution", choices=("auto", "serial", "mpi"), default="auto")
    process_parser.add_argument("--source", choices=("p_all", "lustre_shared", "lustre_per_process"))
    process_parser.add_argument("--index-start", type=int, help="first saved index to include")
    process_parser.add_argument("--index-stop", type=int, help="exclusive saved-index upper bound")
    process_parser.add_argument("--output", help="output directory (default: CASE/mfc_post_output)")
    process_parser.add_argument("--mechanism", help="explicit Cantera YAML when run metadata is not self-contained")
    process_parser.add_argument("--phase", help="explicit mechanism phase name")
    render_parser = subparsers.add_parser(
        "render", help="render selected full-domain physical fields without computing scalar history"
    )
    render_parser.add_argument("case", help="path to an MFC case/run directory")
    render_parser.add_argument(
        "--selected-times-us", required=True,
        help="comma-separated requested physical times in microseconds",
    )
    render_parser.add_argument(
        "--fields", default=",".join(RENDER_FIELDS),
        help="comma-separated symbolic fields",
    )
    render_parser.add_argument("--out-dir", required=True, help="new or empty output directory")
    render_parser.add_argument("--execution", choices=("auto", "serial", "mpi"), default="auto")
    render_parser.add_argument("--mechanism", help="explicit Cantera YAML when metadata is incomplete")
    render_parser.add_argument("--phase", help="explicit mechanism phase name")
    render_parser.add_argument("--no-zoom", action="store_true", help="render only full-domain frames")
    render_parser.add_argument("--no-mp4", action="store_true", help="explicitly disable MP4 output")
    render_parser.add_argument("--skip-scalars", action="store_true", help="skip scalar-history processing")
    render_parser.add_argument("--skip-trends", action="store_true", help="skip trend processing")
    analyze_parser = subparsers.add_parser(
        "analyze", help="compute an MPI-compatible scalar history from raw p_all saves"
    )
    analyze_parser.add_argument("case", help="path to an MFC case/run directory")
    selection = analyze_parser.add_mutually_exclusive_group()
    selection.add_argument("--selected-times-us", help="comma-separated requested times; nearest saves are used")
    selection.add_argument("--time-range-us", help="inclusive START,STOP range in microseconds")
    analyze_parser.add_argument("--stride", type=int, default=1, help="retain every Nth selected save")
    analyze_parser.add_argument("--out-dir", required=True, help="new or empty output directory")
    analyze_parser.add_argument("--execution", choices=("auto", "serial", "mpi"), default="auto")
    analyze_parser.add_argument("--mechanism", help="explicit Cantera YAML when metadata is incomplete")
    analyze_parser.add_argument("--phase", help="explicit mechanism phase name")
    plot_parser = subparsers.add_parser(
        "plot", help="render trend PNGs from scalar_timeseries.csv without reading p_all"
    )
    plot_parser.add_argument("input", help="scalar_timeseries.csv or its analysis directory")
    plot_selection = plot_parser.add_mutually_exclusive_group()
    plot_selection.add_argument("--fields", help="comma-separated scalar CSV columns")
    plot_selection.add_argument("--plot-set", choices=tuple(PLOT_SETS), default=None)
    plot_parser.add_argument("--out-dir", help="new or empty output directory (default: INPUT/trend_plots)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_case(args.case, mechanism=args.mechanism, phase=args.phase)
            if args.format == "json":
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            else:
                print(format_text(result))
            return 0
        if args.command == "process":
            result = process_case(
                args.case, execution=args.execution, source_family=args.source,
                index_start=args.index_start, index_stop=args.index_stop, output=args.output,
                mechanism=args.mechanism, phase=args.phase,
            )
            if result is not None:
                print(
                    f"mfc-post: processed {len(result['records'])} state(s) from "
                    f"{result['source']['family']} -> {result['output_directory']}"
                )
            return 0
        if args.command == "render":
            result = render_case(
                args.case,
                selected_times_us=_comma_floats(args.selected_times_us),
                fields=_comma_strings(args.fields), out_dir=args.out_dir,
                execution=args.execution, mechanism=args.mechanism, phase=args.phase,
                no_zoom=args.no_zoom, no_mp4=args.no_mp4,
                skip_scalars=args.skip_scalars, skip_trends=args.skip_trends,
            )
            if result is not None:
                print(
                    f"mfc-post: rendered {len(result['frames'])} frame(s) -> "
                    f"{result['output_directory']}", flush=True,
                )
            return 0
        if args.command == "analyze":
            time_range = _time_range(args.time_range_us) if args.time_range_us else None
            result = analyze_case(
                args.case,
                selected_times_us=(
                    _comma_floats(args.selected_times_us) if args.selected_times_us else None
                ),
                time_range_us=time_range, stride=args.stride, out_dir=args.out_dir,
                execution=args.execution, mechanism=args.mechanism, phase=args.phase,
            )
            if result is not None:
                print(
                    f"mfc-post: analyzed {len(result['rows'])} state(s) -> "
                    f"{result['output_directory']}", flush=True,
                )
            return 0
        if args.command == "plot":
            result = plot_history(
                args.input,
                fields=_comma_strings(args.fields) if args.fields else None,
                plot_set=args.plot_set, out_dir=args.out_dir,
            )
            print(
                f"mfc-post: plotted {len(result['files'])} trend(s) -> "
                f"{result['output_directory']}", flush=True,
            )
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mfc-post: {exc}", file=sys.stderr)
        return 2
    return 1


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in _comma_strings(value))
    except ValueError as exc:
        raise ValueError(f"invalid --selected-times-us value: {value!r}") from exc


def _time_range(value: str) -> tuple[float, float]:
    parsed = _comma_floats(value)
    if len(parsed) != 2:
        raise ValueError(f"--time-range-us requires exactly START,STOP; got {value!r}")
    return parsed


def format_text(result: dict[str, Any]) -> str:
    metadata = result["run_metadata"]
    lines = [
        f"MFC output inspection: {result['case']}",
        f"Metadata: {', '.join(result['metadata_sources']) or 'none'}",
        f"Model: equations={metadata['parameters'].get('model_eqns', '?')}, fluids={metadata['parameters'].get('num_fluids', '?')}",
        f"Grid: dimensions={metadata.get('dimensions')}, shape={_shape(metadata.get('grid_shape'))}",
        f"Equation layout: {result['equation_layout']['total_size']} fields; species begin={result['equation_layout']['species_begin']}, count={result['equation_layout']['species_count']}",
        "",
    ]
    if not result["sources"]:
        lines.append("Sources: none detected")
    for source in result["sources"]:
        timeline = source["timeline"]
        times = [value for value in timeline["physical_times"] if value is not None]
        lines.extend([
            f"[{source['family']}] {source['path']}",
            f"  layout: {source['layout']}",
            f"  states: {len(timeline['saved_indices'])}; saved indices: {_indices(timeline['saved_indices'])}",
            f"  physical time: {_range(times)}; cadence={timeline.get('cadence')}; basis={timeline['time_basis']}",
            f"  grid: {_grid(source.get('grid'))}",
            f"  precision/downsampled: {source.get('precision') or 'ambiguous'} / {source.get('downsampled')}",
            f"  fields ({len(source['fields'])}): {', '.join(source['fields'][:16])}{' ...' if len(source['fields']) > 16 else ''}",
            f"  restart: {source['restart_suitability']}",
            f"  post-process: {source['postprocess_suitability']}",
            f"  missing/ambiguous: {', '.join(source['missing_metadata']) or 'none'}",
        ])
        for warning in timeline.get("warnings", []) + source.get("warnings", []):
            lines.append(f"  WARNING: {warning}")
        lines.append("")
    if result["conflicts"]:
        lines.append("Conflicts and overlaps (not resolved):")
        lines.extend(f"  WARNING: {item['message']}" for item in result["conflicts"])
        lines.append("")
    recommendation = result.get("recommendation")
    if recommendation:
        lines.append(f"Recommended source: {recommendation['source']} ({recommendation['path']})")
        lines.append(f"  {recommendation['reason']}; score={recommendation['score']}")
    for warning in result.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines).rstrip()


def _indices(values: list[int]) -> str:
    if len(values) <= 12:
        return ", ".join(str(value) for value in values)
    return f"{', '.join(str(value) for value in values[:6])}, ..., {', '.join(str(value) for value in values[-3:])}"


def _range(values: list[float]) -> str:
    return "unavailable" if not values else f"{min(values):.9g} .. {max(values):.9g}"


def _shape(value: list[int] | None) -> str:
    return "unknown" if not value else "x".join(str(item) for item in value)


def _grid(grid: dict[str, Any] | None) -> str:
    if not grid:
        return "unknown"
    suffix = f"; nonuniform={','.join(grid['nonuniform_axes'])}" if grid.get("nonuniform_axes") else ""
    return f"{grid['dimensions']}D {_shape(grid['shape'])}{suffix}"
