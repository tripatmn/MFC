#!/usr/bin/env python3
"""Summarize Haehn/RSBI AQSS diagnostics from MFC stdout/stderr logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
KEYVAL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_%]*)\s*=\s*(" + FLOAT_RE + r"|[A-Za-z_][A-Za-z0-9_+\-.]*)")
TSTEP_RE = re.compile(r"(?:t_step|timestep|time step)\s*[=:]?\s*(\d+)", re.IGNORECASE)
EXIT_RE = re.compile(r"(?:exit|return)\s+code\s*[=:]?\s*(-?\d+)", re.IGNORECASE)
GRID_PATTERNS = [
    re.compile(r"\bm\s*[=:]\s*(\d+).*?\bn\s*[=:]\s*(\d+)(?:.*?\bp\s*[=:]\s*(\d+))?", re.IGNORECASE),
    re.compile(r"\bgrid\b.*?(\d+)\s*[xX]\s*(\d+)(?:\s*[xX]\s*(\d+))?", re.IGNORECASE),
]

AQSS_TAGS = (
    "AQSS_DIAG",
    "AQSS_PRE_REPAIR",
    "AQSS_PRE_ABORT",
    "AQSS_POST_ABORT",
    "AQSS_ABORT_DETAIL",
    "AQSS_ABORT_SPECIES",
    "AQSS_ABORT_THERMO",
)

FAILURE_MARKERS = (
    "AQSS_ABORT",
    "s_mpi_abort",
    "MPI_ABORT",
    "segmentation fault",
    "floating invalid",
    "traceback",
    "fatal",
    "error:",
    "nan",
)

COMPLETION_MARKERS = (
    "completed successfully",
    "normal termination",
    "program terminated",
    "simulation complete",
    "run complete",
)


def parse_value(raw: str):
    text = raw.strip()
    try:
        if any(c in text for c in ".EeDd"):
            return float(text.replace("D", "E").replace("d", "e"))
        return int(text)
    except ValueError:
        return text


def parse_keyvals(line: str) -> dict[str, object]:
    return {key: parse_value(value) for key, value in KEYVAL_RE.findall(line)}


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def collect_logs(run_dir: Path, out_glob: str) -> list[Path]:
    files = set(run_dir.glob(out_glob))
    files.update(run_dir.glob("*.err"))
    files.update(run_dir.glob("*.out"))
    return sorted(path for path in files if path.is_file())


def write_csv(path: Path, rows: list[dict[str, object]], preferred: list[str]) -> None:
    keys = set()
    for row in rows:
        keys.update(row)
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys.difference(fieldnames)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reduce_diag(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        step = to_int(row.get("t_step"))
        if step is not None:
            grouped[step].append(row)

    reduced = []
    for step in sorted(grouped):
        group = grouped[step]

        def max_field(name: str) -> float | None:
            vals = [to_float(row.get(name)) for row in group]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            return max(vals) if vals else None

        def min_field(name: str) -> float | None:
            vals = [to_float(row.get(name)) for row in group]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            return min(vals) if vals else None

        def sum_field(name: str) -> float | None:
            vals = [to_float(row.get(name)) for row in group]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            return sum(vals) if vals else None

        reduced.append(
            {
                "t_step": step,
                "rank_count": len({row.get("rank") for row in group if row.get("rank") is not None}),
                "diag_count": len(group),
                "nsub_max": max_field("nsub"),
                "max_pre_sumY_err": max_field("max_pre_sumY_err"),
                "max_post_sumY_err": max_field("max_post_sumY_err"),
                "minY": min_field("minY"),
                "heat_pos_sum": sum_field("heat_pos_sum"),
                "heat_neg_sum": sum_field("heat_neg_sum"),
                "heat_abs_sum": sum_field("heat_abs_sum"),
                "species_mass_before": sum_field("species_mass_before"),
                "species_mass_after": sum_field("species_mass_after"),
            }
        )
    return reduced


def summarize(log_files, diag_rows, reduced_rows, repair_rows, abort_rows, progress_steps, exit_codes, grid_hits, failure_lines, completed) -> str:
    failed = bool(abort_rows or failure_lines or any(code != 0 for code in exit_codes))
    status = "failed" if failed else ("completed" if completed or any(code == 0 for code in exit_codes) else "unknown")

    first_abort = min(abort_rows, key=lambda row: (str(row.get("source", "")), int(row.get("line", 0)))) if abort_rows else None
    first_heat_step = None
    max_heat_abs = None
    for row in reduced_rows:
        heat = to_float(row.get("heat_abs_sum"))
        step = to_int(row.get("t_step"))
        if heat is not None:
            max_heat_abs = heat if max_heat_abs is None else max(max_heat_abs, heat)
            if heat > 0 and step is not None:
                first_heat_step = step if first_heat_step is None else min(first_heat_step, step)

    min_candidates = []
    for row in diag_rows + repair_rows + abort_rows:
        for key in ("minY", "minY_before", "minY_after"):
            value = to_float(row.get(key))
            if value is not None:
                min_candidates.append(value)
    worst_min_y = min(min_candidates) if min_candidates else None

    pre_errs = [to_float(row.get("max_pre_sumY_err")) for row in diag_rows]
    post_errs = [to_float(row.get("max_post_sumY_err")) for row in diag_rows]
    pre_errs = [v for v in pre_errs if v is not None]
    post_errs = [v for v in post_errs if v is not None]

    chemistry_activity = bool(first_heat_step is not None or (max_heat_abs is not None and max_heat_abs > 0))
    classification = "chemistry activity present" if chemistry_activity else "interface-stability only / no heat activity seen"

    lines = [
        "# Haehn/RSBI AQSS Log Summary",
        "",
        f"- Run status: **{status}**",
        f"- Logs parsed: {len(log_files)}",
        f"- Exit codes: {exit_codes if exit_codes else 'not found'}",
        f"- Grid size: {', '.join(grid_hits) if grid_hits else 'not found'}",
        f"- Last progress step: {max(progress_steps) if progress_steps else 'not found'}",
        f"- First nonzero heat step: {first_heat_step if first_heat_step is not None else 'none found'}",
        f"- Max heat_abs_sum: {max_heat_abs if max_heat_abs is not None else 'not found'}",
        f"- Worst minY: {worst_min_y if worst_min_y is not None else 'not found'}",
        f"- Worst pre-closure error: {max(pre_errs) if pre_errs else 'not found'}",
        f"- Worst post-closure error: {max(post_errs) if post_errs else 'not found'}",
        f"- Pre-repair count: {len(repair_rows)}",
        f"- Classification: **{classification}**",
        "",
    ]

    if first_abort:
        lines.extend(
            [
                "## First Abort",
                "",
                f"- Source: `{first_abort.get('source')}` line {first_abort.get('line')}",
                f"- Tag: `{first_abort.get('tag')}`",
                f"- Step: {first_abort.get('t_step', 'not found')}",
                f"- Reason: {first_abort.get('reason', 'not found')}",
                f"- Species: {first_abort.get('species', 'not found')}",
                "",
            ]
        )
    else:
        lines.extend(["## First Abort", "", "No AQSS abort line found.", ""])

    if failure_lines:
        lines.extend(["## Failure Markers", ""])
        lines.extend(f"- {line}" for line in failure_lines[:20])
        if len(failure_lines) > 20:
            lines.append(f"- ... {len(failure_lines) - 20} more")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-glob", default="*.out")
    parser.add_argument("--output-dir", default="analysis_out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else run_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_files = collect_logs(run_dir, args.out_glob)
    diag_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    abort_rows: list[dict[str, object]] = []
    progress_steps: list[int] = []
    exit_codes: list[int] = []
    grid_hits: list[str] = []
    failure_lines: list[str] = []
    completed = False

    for log_file in log_files:
        rel_source = str(log_file.relative_to(run_dir)) if log_file.is_relative_to(run_dir) else str(log_file)
        try:
            stream = log_file.open(errors="replace")
        except OSError:
            continue
        with stream:
            for line_no, line in enumerate(stream, start=1):
                text = line.strip()
                lower = text.lower()

                if any(marker in lower for marker in COMPLETION_MARKERS):
                    completed = True
                if any(marker.lower() in lower for marker in FAILURE_MARKERS):
                    failure_lines.append(f"{rel_source}:{line_no}: {text[:180]}")

                exit_match = EXIT_RE.search(text)
                if exit_match:
                    exit_codes.append(int(exit_match.group(1)))

                step_match = TSTEP_RE.search(text)
                if step_match:
                    progress_steps.append(int(step_match.group(1)))

                for pattern in GRID_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        dims = [part for part in match.groups() if part is not None]
                        grid_hits.append(" x ".join(dims))
                        break

                for tag in AQSS_TAGS:
                    if tag in text:
                        row = {"source": rel_source, "line": line_no, "tag": tag}
                        row.update(parse_keyvals(text))
                        if tag == "AQSS_DIAG":
                            diag_rows.append(row)
                        elif tag == "AQSS_PRE_REPAIR":
                            repair_rows.append(row)
                        else:
                            abort_rows.append(row)
                        break

    reduced_rows = reduce_diag(diag_rows)

    write_csv(output_dir / "aqss_diag_by_rank.csv", diag_rows, ["source", "line", "rank", "t_step", "nsub"])
    write_csv(output_dir / "aqss_diag_reduced.csv", reduced_rows, ["t_step", "rank_count", "diag_count"])
    write_csv(output_dir / "aqss_pre_repairs.csv", repair_rows, ["source", "line", "rank", "t_step", "i", "j", "k", "min_species"])
    write_csv(output_dir / "aqss_aborts.csv", abort_rows, ["source", "line", "tag", "rank", "t_step", "reason", "i", "j", "k", "species"])

    summary = summarize(
        log_files,
        diag_rows,
        reduced_rows,
        repair_rows,
        abort_rows,
        progress_steps,
        exit_codes,
        sorted(set(grid_hits)),
        failure_lines,
        completed,
    )
    (output_dir / "summary.md").write_text(summary + "\n")

    print(f"Wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
