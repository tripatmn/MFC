#!/usr/bin/env python3
"""Lightweight post-run checks for SK54 diffusion test cases.

This reads saved ASCII D/ output only. It never modifies solver output.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/home/tripatmn/mfc_work/MFC_alpha_diag")
sys.path.insert(0, str(REPO / "examples/2D_dodecane_global_reduced"))
import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics

raw.raw_dirs = lambda run_dir: [Path(run_dir) / "D"] if (Path(run_dir) / "D").is_dir() else []
MECH = REPO / "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
PHASE = "yao_sk54"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def species_names() -> list[str]:
    doc = yaml.safe_load(MECH.read_text())
    for phase in doc.get("phases", []):
        if phase.get("name") == PHASE:
            return list(phase["species"])
    raise RuntimeError(f"phase {PHASE!r} not found in {MECH}")


def available_steps(run_dir: Path) -> list[int]:
    d = run_dir / "D"
    if not d.is_dir():
        raise FileNotFoundError(f"raw output directory is missing: {d}")
    return sorted({int(path.name.split(".")[3]) for path in d.glob("cons.1.*.*.dat") if path.stat().st_size > 0})


def steps_with_fields(run_dir: Path, steps: list[int], names: list[str]) -> tuple[list[int], dict[int, list[str]]]:
    complete: list[int] = []
    missing_by_step: dict[int, list[str]] = {}
    for step in steps:
        missing = [name for name in names if not raw.read_field(run_dir, name, step)["available"]]
        if missing:
            missing_by_step[step] = missing
        else:
            complete.append(step)
    return complete, missing_by_step


def field(run_dir: Path, kind: str, index: int, step: int) -> dict:
    name = f"tmp_{kind}_{index}"
    raw.FIELDS[name] = (kind, index)
    return raw.read_field(run_dir, name, step)


def parse_run_time(run_dir: Path) -> list[dict]:
    path = run_dir / "run_time.inf"
    if not path.exists():
        return []
    rows = []
    num = re.compile(r"^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)(?:\s+([-+0-9.Ee]+))?")
    runtime_re = re.compile(r"^\s*Run-time:\s*([-+0-9.Ee]+)s\s*$", re.I)
    runtime_seconds = math.nan
    for line in path.read_text(errors="replace").splitlines():
        runtime_match = runtime_re.match(line)
        if runtime_match:
            runtime_seconds = float(runtime_match.group(1))
            continue
        m = num.match(line)
        if m:
            dt_raw = m.group(2)
            time_raw = m.group(3)
            dt_value = float(dt_raw)
            time_value = float(time_raw)
            dt_precision_limited = dt_value == 0.0 and dt_raw in {"0.000000", "0.0000000"}
            time_precision_limited = time_value == 0.0 and time_raw in {"0.000000", "0.0000000"}
            rows.append({
                "t_step": int(m.group(1)),
                "dt_s": math.nan if dt_precision_limited else dt_value,
                "physical_time_s": math.nan if time_precision_limited else time_value,
                "dt_raw": dt_raw,
                "physical_time_raw": time_raw,
                "ICFL_max": float(m.group(4)),
                "extra_stability_column_1": float(m.group(5)) if m.group(5) is not None else math.nan,
                "dt_note": "precision_limited_zero_in_run_time_inf" if dt_precision_limited else "",
                "physical_time_note": "precision_limited_zero_in_run_time_inf" if time_precision_limited else "",
                "run_time_seconds_final": math.nan,
            })
    if rows and math.isfinite(runtime_seconds):
        rows[-1]["run_time_seconds_final"] = runtime_seconds
    return rows


def grep_failures(run_dir: Path) -> str:
    pats = re.compile(
        r"("
        r"\bNaN\b|"
        r"\bInf\b|"
        r"nonfinite|"
        r"\babort(?:ed)?\b|"
        r"segmentation fault|"
        r"floating[- ]point exception|"
        r"undefined reference|"
        r"invalid .*property|"
        r"property failure|"
        r"diffusion .*failure|"
        r"chemistry .*failure|"
        r"negative .*state|"
        r"unphysical .*state|"
        r"insufficient fuel|"
        r"\bptxas fatal\b|"
        r"\bnvfortran-.*error\b|"
        r"\bnvc\+\+-.*error\b|"
        r"constant memory.*exceed"
        r")",
        re.I,
    )
    chunks = []
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {"", ".txt", ".log", ".out", ".err", ".inf"}:
            try:
                for i, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
                    if pats.search(line):
                        chunks.append(f"{path.name}:{i}:{line}")
            except OSError:
                pass
    return "\n".join(chunks) + ("\n" if chunks else "")


def analyze(run_dir: Path, out_dir: Path, gas_floor: float, tol: float) -> None:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    names = species_names()
    candidate_steps = available_steps(run_dir)
    if not candidate_steps:
        raise RuntimeError(f"no saved states found in {run_dir / 'D'}")
    required_fields = [
        "liquid_alpha",
        "vapor_alpha",
        "air_alpha",
        "vapor_alpha_rho",
        "air_alpha_rho",
        "pressure",
    ]
    steps, missing_by_step = steps_with_fields(run_dir, candidate_steps, required_fields)
    if not steps:
        details = "; ".join(
            f"save {step}: {', '.join(missing)}" for step, missing in sorted(missing_by_step.items())
        )
        raise RuntimeError(f"no saves with all required output fields in {run_dir / 'D'}; {details}")
    rows = []
    for step in steps:
        fields = {
            "liquid_alpha": raw.read_field(run_dir, "liquid_alpha", step),
            "vapor_alpha": raw.read_field(run_dir, "vapor_alpha", step),
            "air_alpha": raw.read_field(run_dir, "air_alpha", step),
            "vapor_alpha_rho": raw.read_field(run_dir, "vapor_alpha_rho", step),
            "air_alpha_rho": raw.read_field(run_dir, "air_alpha_rho", step),
            "pressure": raw.read_field(run_dir, "pressure", step),
        }
        species_fields = {name: field(run_dir, "cons", 12 + sid, step) for sid, name in enumerate(names, start=1)}
        missing_species = [name for name, f in species_fields.items() if not f["available"]]
        if missing_species:
            preview = ", ".join(missing_species[:8])
            if len(missing_species) > 8:
                preview += f", ... ({len(missing_species)} total)"
            raise RuntimeError(f"save {step} is missing required species fields: {preview}")
        masks = gas_metrics.mask_context(fields, gas_floor)
        dx, dy, area = gas_metrics.estimate_cell_area(fields)
        keys = set(masks["valid_gas_thermo"])
        gas_mass = {
            key: fields["vapor_alpha_rho"]["values"].get(key, math.nan) + fields["air_alpha_rho"]["values"].get(key, math.nan)
            for key in keys
        }
        sum_rhoY = {key: 0.0 for key in keys}
        neg_rhoY = neg_Y = above_Y = 0
        min_rhoY = math.inf
        min_Y = math.inf
        max_Y = -math.inf
        worst_species = ""
        for name, f in species_fields.items():
            for key in keys:
                rhoY = f["values"].get(key, math.nan)
                gm = gas_mass.get(key, math.nan)
                if not math.isfinite(rhoY) or not math.isfinite(gm) or gm <= gas_floor:
                    continue
                Y = rhoY / gm
                sum_rhoY[key] += rhoY
                if rhoY < min_rhoY:
                    min_rhoY, worst_species = rhoY, name
                min_Y = min(min_Y, Y)
                max_Y = max(max_Y, Y)
                neg_rhoY += int(rhoY < -tol)
                neg_Y += int(Y < -tol)
                above_Y += int(Y > 1.0 + tol)
        sumY = [sum_rhoY[key] / gas_mass[key] for key in keys if math.isfinite(gas_mass[key]) and gas_mass[key] > gas_floor]
        gas_int = sum(gas_mass[key] for key in keys if math.isfinite(gas_mass[key])) * area
        species_int = sum(sum_rhoY.values()) * area
        rows.append({
            "save": step,
            "time_s_nominal": step * 5.0e-8,
            "time_us_nominal": step * 0.05,
            "available": "T",
            "valid_gas_cells": len(keys),
            "negative_rhoY_count": neg_rhoY,
            "negative_Y_count": neg_Y,
            "Y_above_one_count": above_Y,
            "sumY_min": min(sumY) if sumY else math.nan,
            "sumY_max": max(sumY) if sumY else math.nan,
            "sumY_violation_count": sum(1 for v in sumY if abs(v - 1.0) > tol),
            "min_rhoY": min_rhoY,
            "min_Y": min_Y,
            "max_Y": max_Y,
            "worst_min_rhoY_species": worst_species,
            "integrated_valid_gas_mass": gas_int,
            "integrated_valid_gas_sum_rhoY": species_int,
            "integrated_sum_rhoY_minus_gas_mass": species_int - gas_int,
            "relative_integrated_error": (species_int - gas_int) / gas_int if gas_int else math.nan,
        })
    write_csv(out_dir / "species_and_mass_consistency_by_save.csv", rows)
    write_csv(out_dir / "timestep_history.csv", parse_run_time(run_dir))
    (out_dir / "diffusion_failure_messages.txt").write_text(grep_failures(run_dir))
    summary = [
        f"run_dir: {run_dir}",
        f"candidate_conservative_saves: {candidate_steps}",
        f"analyzed_saves: {steps}",
        f"skipped_incomplete_saves: {sorted(missing_by_step)}",
        "missing_required_fields_by_skipped_save: " + (
            "; ".join(f"{step}={','.join(missing)}" for step, missing in sorted(missing_by_step.items()))
            if missing_by_step else "none"
        ),
        f"rows: {len(rows)}",
        "chemistry_limiter_activity: grep-based only; explicit aggregate limiter counters are not recorded unless the run log contains TEMP_CHEM diagnostics.",
        f"failure_message_lines: {len((out_dir / 'diffusion_failure_messages.txt').read_text().splitlines())}",
    ]
    (out_dir / "analysis_summary.txt").write_text("\n".join(summary) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gas-mass-floor", type=float, default=1.0e-8)
    parser.add_argument("--bounds-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis_diffusion_lightweight"
    analyze(args.run_dir, out_dir, args.gas_mass_floor, args.bounds_tolerance)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
