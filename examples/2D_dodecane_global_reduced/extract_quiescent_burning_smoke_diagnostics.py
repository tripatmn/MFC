#!/usr/bin/env python3
"""Extract compact diagnostics from a 0.25 mm quiescent burning smoke run.

This script is intended to run in place on HPC. It reads only logs plus a few
selected raw states from D/ or p_all/ and writes a small diagnostic bundle.
It does not run MFC and does not copy raw fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np


RAW_RE = re.compile(r"^(?P<kind>[^.]+)\.(?P<index>\d+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")

FIELDS = {
    "pressure": ("prim", 6),
    "liquid_alpha_rho": ("cons", 1),
    "vapor_alpha_rho": ("cons", 2),
    "air_alpha_rho": ("cons", 3),
    "liquid_alpha": ("cons", 7),
    "vapor_alpha": ("cons", 8),
    "air_alpha": ("cons", 9),
    "rhoY_C12H26": ("cons", 13),
    "rhoY_O2": ("cons", 14),
    "rhoY_CO2": ("cons", 16),
    "rhoY_H2O": ("cons", 17),
}

STATE_FIELD_ORDER = (
    "pressure",
    "liquid_alpha_rho",
    "vapor_alpha_rho",
    "air_alpha_rho",
    "liquid_alpha",
    "vapor_alpha",
    "air_alpha",
    "rhoY_C12H26",
    "rhoY_O2",
    "rhoY_CO2",
    "rhoY_H2O",
)

KEYWORDS = (
    "nan",
    "inf",
    "error",
    "abort",
    "cancel",
    "killed",
    "segmentation",
    "floating",
    "exception",
    "failed",
    "stuck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run folder, e.g. runs/.../burning_smoke")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for compact diagnostics")
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Use every Nth saved state for optional history plots; selected first/previous/last states are always read.",
    )
    parser.add_argument(
        "--max-history-states",
        type=int,
        default=3,
        help="Maximum saved states to read for optional raw-field plots and trends.",
    )
    return parser.parse_args()


def clean_line(line: str) -> str:
    return line.encode("utf-8", errors="replace").decode("utf-8", errors="replace").rstrip()


def parse_numeric_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        nums = [float(tok) for tok in NUM_RE.findall(stripped)]
        if len(nums) < 3:
            continue
        rows.append({
            "step": int(nums[0]),
            "dt": float(nums[1]),
            "time": float(nums[2]),
            "icfl_max": float(nums[3]) if len(nums) > 3 else math.nan,
        })
    return rows


def scan_text_file(path: Path, tail_lines: int = 80) -> dict:
    hits = []
    tail = deque(maxlen=tail_lines)
    line_count = 0
    last_numeric = None
    try:
        with path.open("r", errors="replace") as handle:
            for line_count, line in enumerate(handle, start=1):
                line = clean_line(line)
                tail.append(line)
                low = line.lower()
                if any(keyword in low for keyword in KEYWORDS):
                    hits.append({"line": line_count, "text": line[:240]})
                stripped = line.strip()
                if stripped and stripped[0].isdigit():
                    nums = [float(tok) for tok in NUM_RE.findall(stripped)]
                    if len(nums) >= 3:
                        last_numeric = {
                            "step": int(nums[0]),
                            "dt": float(nums[1]),
                            "time": float(nums[2]),
                            "icfl_max": float(nums[3]) if len(nums) > 3 else math.nan,
                        }
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "line_count": line_count,
        "keyword_hits": hits[-40:],
        "last_numeric_row": last_numeric,
        "tail": list(tail),
    }


def collect_log_summary(run_dir: Path) -> dict:
    patterns = ["*.out", "*.err", "run_time.inf", "pre_time_data.dat", "run_time_data.dat", "*time*.dat"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(run_dir.glob(pattern)))
    unique_paths = sorted(set(paths))

    files = [scan_text_file(path) for path in unique_paths]
    run_time_rows = parse_numeric_rows(run_dir / "run_time.inf")
    last_run_time = run_time_rows[-1] if run_time_rows else None
    nonzero_dt = [float(row["dt"]) for row in run_time_rows if float(row["dt"]) > 0.0]
    nonzero_time = [float(row["time"]) for row in run_time_rows if float(row["time"]) > 0.0]
    min_dt = min((float(row["dt"]) for row in run_time_rows), default=math.nan)
    max_dt = max((float(row["dt"]) for row in run_time_rows), default=math.nan)
    return {
        "files": files,
        "run_time_rows": run_time_rows,
        "last_run_time": last_run_time,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "min_dt_nonzero": min(nonzero_dt, default=math.nan),
        "max_dt_nonzero": max(nonzero_dt, default=math.nan),
        "max_time_nonzero": max(nonzero_time, default=math.nan),
        "run_time_inf_dt_rounded_to_zero": bool(run_time_rows and not nonzero_dt),
        "run_time_inf_time_rounded_to_zero": bool(run_time_rows and not nonzero_time),
        "n_run_time_rows": len(run_time_rows),
    }


def raw_dirs(run_dir: Path) -> list[Path]:
    out = []
    for name in ("D", "p_all"):
        path = run_dir / name
        if path.is_dir():
            out.append(path)
    return out


def iter_raw_files(root: Path, kind: str | None = None, index: int | None = None) -> Iterable[Path]:
    pattern = "*.dat" if kind is None or index is None else f"{kind}.{index}.*.*.dat"
    yield from root.rglob(pattern)


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
    by_step: dict[int, list[Path]] = defaultdict(list)
    for root in raw_dirs(run_dir):
        for path in iter_raw_files(root, kind, index):
            info = raw_file_info(path)
            if info is None:
                continue
            p_kind, p_index, _rank, p_step = info
            if p_kind != kind or p_index != index:
                continue
            if step is not None and p_step != step:
                continue
            by_step[p_step].append(path)
    return {key: sorted(paths) for key, paths in by_step.items()}


def available_steps(run_dir: Path) -> dict[str, list[int]]:
    out = {}
    for name, (kind, index) in FIELDS.items():
        out[name] = sorted(field_files(run_dir, kind, index).keys())
    return out


def directory_size(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    count = 0
    size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            count += 1
            size += file_path.stat().st_size
    return count, size


def read_raw_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        arr = np.loadtxt(path)
    except ValueError:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    return arr[:, [0, 1]].astype(float, copy=False), arr[:, -1].astype(float, copy=False)


def read_field(run_dir: Path, name: str, step: int) -> dict:
    kind, index = FIELDS[name]
    paths = field_files(run_dir, kind, index, step).get(step, [])
    values: dict[tuple[float, float], float] = {}
    duplicates = 0
    duplicate_max_abs_diff = 0.0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    row_count = 0

    for path in paths:
        coords, vals = read_raw_values(path)
        row_count += int(vals.size)
        if vals.size:
            nan_count += int(np.count_nonzero(np.isnan(vals)))
            posinf_count += int(np.count_nonzero(vals == np.inf))
            neginf_count += int(np.count_nonzero(vals == -np.inf))
        for (x, y), value in zip(coords, vals):
            key = (float(x), float(y))
            value = float(value)
            if key in values:
                duplicates += 1
                if math.isfinite(value) and math.isfinite(values[key]):
                    duplicate_max_abs_diff = max(duplicate_max_abs_diff, abs(values[key] - value))
                continue
            values[key] = value

    if not values:
        return {
            "available": False,
            "files": len(paths),
            "row_count": row_count,
            "values": {},
            "stats": missing_stats(),
        }

    vals_array = np.fromiter(values.values(), dtype=float, count=len(values))
    finite = vals_array[np.isfinite(vals_array)]
    stats = {
        "available": True,
        "files": len(paths),
        "row_count": row_count,
        "unique_count": len(values),
        "duplicates": duplicates,
        "duplicate_max_abs_diff": duplicate_max_abs_diff,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": int(vals_array.size - finite.size),
        "min": float(np.min(finite)) if finite.size else math.nan,
        "max": float(np.max(finite)) if finite.size else math.nan,
        "mean": float(np.mean(finite)) if finite.size else math.nan,
        "sum": float(np.sum(finite)) if finite.size else math.nan,
    }
    return {"available": True, "files": len(paths), "row_count": row_count, "values": values, "stats": stats}


def missing_stats() -> dict:
    return {
        "available": False,
        "files": 0,
        "row_count": 0,
        "unique_count": 0,
        "duplicates": 0,
        "duplicate_max_abs_diff": math.nan,
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "nonfinite_count": 0,
        "min": math.nan,
        "max": math.nan,
        "mean": math.nan,
        "sum": math.nan,
    }


def grid_area_from_values(values: dict[tuple[float, float], float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    coords = np.array(list(values.keys()), dtype=float)
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dx = median_positive_spacing(xs)
    dy = median_positive_spacing(ys)
    return dx, dy, dx * dy


def median_positive_spacing(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    diffs = np.diff(np.sort(values))
    diffs = diffs[diffs > 0.0]
    return float(np.median(diffs)) if diffs.size else math.nan


def combined_field_stats(fields: dict[str, dict], names: tuple[str, ...]) -> dict:
    if not all(fields[name]["available"] for name in names):
        return missing_stats()
    keys = set(fields[names[0]]["values"])
    for name in names[1:]:
        keys &= set(fields[name]["values"])
    if not keys:
        return missing_stats()
    vals = np.array([sum(fields[name]["values"][key] for name in names) for key in keys], dtype=float)
    finite = vals[np.isfinite(vals)]
    return {
        "available": True,
        "files": 0,
        "row_count": int(vals.size),
        "unique_count": int(vals.size),
        "duplicates": 0,
        "duplicate_max_abs_diff": math.nan,
        "nan_count": int(np.count_nonzero(np.isnan(vals))),
        "posinf_count": int(np.count_nonzero(vals == np.inf)),
        "neginf_count": int(np.count_nonzero(vals == -np.inf)),
        "nonfinite_count": int(vals.size - finite.size),
        "min": float(np.min(finite)) if finite.size else math.nan,
        "max": float(np.max(finite)) if finite.size else math.nan,
        "mean": float(np.mean(finite)) if finite.size else math.nan,
        "sum": float(np.sum(finite)) if finite.size else math.nan,
    }


def time_row_is_usable(step: int, row: dict) -> bool:
    time = float(row.get("time", math.nan))
    return math.isfinite(time) and (step == 0 or time > 0.0)


def state_time(step: int, time_by_step: dict[int, dict], fixed_dt: float | None, adaptive_dt: bool) -> tuple[float, float, str]:
    if step in time_by_step and time_row_is_usable(step, time_by_step[step]):
        row = time_by_step[step]
        return float(row["time"]), float(row["dt"]), "run_time.inf"
    lower = [s for s in time_by_step if s <= step and time_row_is_usable(s, time_by_step[s])]
    if lower:
        nearest = max(lower)
        row = time_by_step[nearest]
        return float(row["time"]), float(row["dt"]), f"nearest_run_time_step_{nearest}"
    if fixed_dt is not None:
        source = "approx_initial_dt_adaptive" if adaptive_dt else "simulation.inp_dt"
        return float(step * fixed_dt), fixed_dt, source
    return math.nan, math.nan, "missing"


def read_text_if_exists(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def run_text(run_dir: Path) -> str:
    return "\n".join(
        read_text_if_exists(path)
        for path in (run_dir / "simulation.inp", run_dir / "case.py")
    )


def parse_fixed_dt(run_dir: Path) -> float | None:
    sim_path = run_dir / "simulation.inp"
    if not sim_path.is_file():
        return None
    for line in sim_path.read_text(errors="replace").splitlines():
        stripped = line.split("!", 1)[0].strip()
        if stripped.lower().startswith("dt"):
            try:
                return float(stripped.split("=", 1)[1].strip().rstrip(","))
            except (IndexError, ValueError):
                return None
    return None


def parse_adaptive_dt(run_dir: Path) -> bool:
    text = run_text(run_dir).lower()
    match = re.search(r"cfl_adap_dt[^a-z0-9_]*['\"]?([tf])['\"]?", text)
    return bool(match and match.group(1) == "t")


def parse_d0_mm(run_dir: Path) -> tuple[float, str]:
    text = run_text(run_dir)
    radius_match = re.search(
        r"patch_icpp\(2\)%radius[^0-9+\-.]*(%s)" % NUM_RE.pattern,
        text,
        flags=re.IGNORECASE,
    )
    if radius_match:
        return 2.0*float(radius_match.group(1))*1.0e3, "patch_icpp(2)%radius"

    d0_match = re.search(r"\bD0\s*=\s*(%s)" % NUM_RE.pattern, text)
    if d0_match:
        return float(d0_match.group(1))*1.0e3, "case.py D0"

    mm_match = re.search(r"\b(0\.25|\.25)\s*mm\b", text, flags=re.IGNORECASE)
    if mm_match:
        return 0.25, "case.py text 0.25 mm"

    path_match = re.search(r"025mm|0\.25", str(run_dir), flags=re.IGNORECASE)
    if path_match:
        return 0.25, "run-dir name 0.25 mm"

    return math.nan, "missing"


def selected_steps(steps_by_field: dict[str, list[int]]) -> list[int]:
    candidates = []
    for field in ("liquid_alpha_rho", "pressure", "liquid_alpha"):
        candidates.extend(steps_by_field.get(field, []))
    steps = sorted(set(candidates))
    if not steps:
        return []
    out = [steps[0]]
    if len(steps) > 2:
        out.append(steps[-2])
    if steps[-1] not in out:
        out.append(steps[-1])
    return out


def history_steps(steps_by_field: dict[str, list[int]], stride: int, max_states: int) -> list[int]:
    base = sorted(set(steps_by_field.get("liquid_alpha_rho", [])) | set(steps_by_field.get("pressure", [])))
    if not base:
        return []
    base = base[::max(stride, 1)]
    if len(base) > max_states:
        take = np.linspace(0, len(base) - 1, max_states, dtype=int)
        base = [base[i] for i in take]
    return sorted(set(base) | set(selected_steps(steps_by_field)))


def analyze_state(run_dir: Path, step: int, time_by_step: dict[int, dict], fixed_dt: float | None, adaptive_dt: bool) -> tuple[dict, dict]:
    fields = {name: read_field(run_dir, name, step) for name in STATE_FIELD_ORDER}
    reference = next((fields[name]["values"] for name in ("liquid_alpha_rho", "liquid_alpha", "pressure") if fields[name]["available"]), {})
    dx, dy, d_area = grid_area_from_values(reference)
    time, dt, time_source = state_time(step, time_by_step, fixed_dt, adaptive_dt)

    row = {
        "step": step,
        "time": time,
        "dt": dt,
        "time_source": time_source,
        "dx": dx,
        "dy": dy,
        "cell_area": d_area,
    }

    detail = {"step": step, "fields": {}}
    for name in STATE_FIELD_ORDER:
        stats = dict(fields[name]["stats"])
        detail["fields"][name] = stats
        prefix = short_prefix(name)
        row[f"{prefix}_available"] = stats["available"]
        row[f"{prefix}_min"] = stats["min"]
        row[f"{prefix}_max"] = stats["max"]
        row[f"{prefix}_mean"] = stats["mean"]
        row[f"{prefix}_sum"] = stats["sum"]
        row[f"{prefix}_nonfinite_count"] = stats["nonfinite_count"]
        if math.isfinite(d_area):
            row[f"{prefix}_integral"] = stats["sum"] * d_area if stats["available"] else math.nan
        else:
            row[f"{prefix}_integral"] = math.nan

    for label, names in (
        ("gas_density", ("vapor_alpha_rho", "air_alpha_rho")),
        ("mixture_density", ("liquid_alpha_rho", "vapor_alpha_rho", "air_alpha_rho")),
    ):
        stats = combined_field_stats(fields, names)
        detail["fields"][label] = stats
        row[f"{label}_available"] = stats["available"]
        row[f"{label}_min"] = stats["min"]
        row[f"{label}_max"] = stats["max"]
        row[f"{label}_mean"] = stats["mean"]
        row[f"{label}_sum"] = stats["sum"]
        row[f"{label}_nonfinite_count"] = stats["nonfinite_count"]
        row[f"{label}_integral"] = stats["sum"] * d_area if stats["available"] and math.isfinite(d_area) else math.nan

    row["finite_loaded_fields"] = all(
        not bool(detail["fields"][name].get("nonfinite_count", 0))
        for name in detail["fields"]
        if detail["fields"][name].get("available")
    )
    return row, detail


def short_prefix(name: str) -> str:
    return {
        "pressure": "pressure",
        "liquid_alpha_rho": "liq_arho",
        "vapor_alpha_rho": "vap_arho",
        "air_alpha_rho": "air_arho",
        "liquid_alpha": "liq_alpha",
        "vapor_alpha": "vap_alpha",
        "air_alpha": "air_alpha",
        "rhoY_C12H26": "rhoY_C12H26",
        "rhoY_O2": "rhoY_O2",
        "rhoY_CO2": "rhoY_CO2",
        "rhoY_H2O": "rhoY_H2O",
    }[name]


def enrich_d2(rows: list[dict], d0_mm: float) -> None:
    if not rows:
        return
    first_mass = rows[0].get("liq_arho_integral", math.nan)
    d0_squared = d0_mm*d0_mm if math.isfinite(d0_mm) else math.nan
    first_time = rows[0].get("time", math.nan)
    for row in rows:
        mass = row.get("liq_arho_integral", math.nan)
        row["D2_mass_norm"] = mass / first_mass if first_mass and math.isfinite(first_mass) else math.nan
        row["D2_mass_mm2"] = d0_squared*row["D2_mass_norm"] if math.isfinite(d0_squared) and math.isfinite(row["D2_mass_norm"]) else math.nan
        elapsed = row.get("time", math.nan) - first_time if math.isfinite(row.get("time", math.nan)) and math.isfinite(first_time) else math.nan
        if elapsed and math.isfinite(elapsed) and elapsed > 0.0 and math.isfinite(row["D2_mass_mm2"]):
            row["K_mass_mm2_s_cumulative"] = -(row["D2_mass_mm2"] - d0_squared)/elapsed
        else:
            row["K_mass_mm2_s_cumulative"] = math.nan

        alpha_sum = row.get("liq_alpha_sum", math.nan)
        area = row.get("cell_area", math.nan)
        if math.isfinite(alpha_sum) and math.isfinite(area):
            row["liquid_alpha_integral"] = alpha_sum * area
        else:
            row["liquid_alpha_integral"] = math.nan


def final_mass_k(rows: list[dict], d0_mm: float) -> dict:
    if len(rows) < 2 or not math.isfinite(d0_mm):
        return {"status": "insufficient", "reason": "need at least two states and D0"}
    first = rows[0]
    last = rows[-1]
    t0 = float(first.get("time", math.nan))
    t1 = float(last.get("time", math.nan))
    d20 = float(first.get("D2_mass_mm2", math.nan))
    d21 = float(last.get("D2_mass_mm2", math.nan))
    if not all(math.isfinite(v) for v in (t0, t1, d20, d21)) or t1 <= t0:
        return {"status": "insufficient", "reason": "invalid time or D2 values"}
    return {
        "status": "ok",
        "D0_mm": d0_mm,
        "D0_squared_mm2": d0_mm*d0_mm,
        "step_initial": int(first["step"]),
        "step_final": int(last["step"]),
        "time_initial_s": t0,
        "time_final_s": t1,
        "time_delta_s": t1 - t0,
        "time_source_initial": first.get("time_source", ""),
        "time_source_final": last.get("time_source", ""),
        "D2_initial_mm2": d20,
        "D2_final_mm2": d21,
        "D2_mass_norm_final": float(last.get("D2_mass_norm", math.nan)),
        "K_mass_mm2_s": -(d21 - d20)/(t1 - t0),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_log_text(path: Path, log_summary: dict) -> None:
    with path.open("w") as handle:
        handle.write("Smoke diagnostic log summary\n")
        handle.write(f"n_log_files={len(log_summary['files'])}\n")
        handle.write(f"n_run_time_rows={log_summary['n_run_time_rows']}\n")
        handle.write(f"last_run_time={log_summary['last_run_time']}\n")
        handle.write(f"min_dt={log_summary['min_dt']}\n")
        handle.write(f"max_dt={log_summary['max_dt']}\n")
        handle.write("\nKeyword hits\n")
        for file_info in log_summary["files"]:
            hits = file_info.get("keyword_hits", [])
            if not hits:
                continue
            handle.write(f"\n{file_info['path']}\n")
            for hit in hits:
                handle.write(f"  line {hit['line']}: {hit['text']}\n")
        handle.write("\nTails\n")
        for file_info in log_summary["files"]:
            handle.write(f"\n== {file_info['path']} ==\n")
            for line in file_info.get("tail", []):
                handle.write(f"{line}\n")


def time_history_for_plot(run_time_rows: list[dict], fixed_dt: float | None, adaptive_dt: bool) -> tuple[list[dict], str]:
    usable = [
        {
            "step": int(row["step"]),
            "time": float(row["time"]),
            "dt": float(row["dt"]),
        }
        for row in run_time_rows
        if time_row_is_usable(int(row["step"]), row) and float(row["dt"]) > 0.0
    ]
    if usable:
        return usable, "run_time.inf"
    if fixed_dt is None:
        return [], "missing"
    source = "approx_initial_dt_adaptive" if adaptive_dt else "simulation.inp_dt"
    return [
        {
            "step": int(row["step"]),
            "time": int(row["step"])*fixed_dt,
            "dt": fixed_dt,
        }
        for row in run_time_rows
    ], source


def maybe_write_plots(out_dir: Path, rows: list[dict], run_time_rows: list[dict], fixed_dt: float | None, adaptive_dt: bool) -> tuple[list[str], str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return [], "matplotlib_unavailable"

    paths: list[str] = []
    plot_time_rows, time_plot_source = time_history_for_plot(run_time_rows, fixed_dt, adaptive_dt)
    if plot_time_rows:
        times = np.array([row["time"] for row in plot_time_rows], dtype=float)
        dts = np.array([row["dt"] for row in plot_time_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, dts, marker=".", linewidth=1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("dt [s]")
        ax.set_title(f"Timestep history ({time_plot_source})")
        ax.grid(True, alpha=0.3)
        path = out_dir / "smoke_dt_history.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if len(rows) >= 2:
        times = np.array([row.get("time", math.nan) for row in rows], dtype=float)
        plot_specs = [
            ("pressure_min", "pressure_max", "Pressure extrema", "Pressure [Pa]", "smoke_pressure_extrema.png"),
            ("liq_arho_integral", "vap_arho_integral", "Liquid/vapor alpha_rho integrals", "Integral", "smoke_alpha_rho_integrals.png"),
            ("rhoY_C12H26_integral", "rhoY_O2_integral", "Fuel/O2 species integrals", "Integral", "smoke_species_fuel_o2.png"),
            ("rhoY_CO2_integral", "rhoY_H2O_integral", "Product species integrals", "Integral", "smoke_species_products.png"),
        ]
        for y1, y2, title, ylabel, filename in plot_specs:
            if not any(math.isfinite(float(row.get(y1, math.nan))) for row in rows):
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(times, [row.get(y1, math.nan) for row in rows], marker="o", label=y1)
            ax.plot(times, [row.get(y2, math.nan) for row in rows], marker="o", label=y2)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            path = out_dir / filename
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths, time_plot_source


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_summary = collect_log_summary(run_dir)
    steps_by_field = available_steps(run_dir)
    chosen_steps = selected_steps(steps_by_field)
    trend_steps = history_steps(steps_by_field, args.history_stride, args.max_history_states)
    fixed_dt = parse_fixed_dt(run_dir)
    adaptive_dt = parse_adaptive_dt(run_dir)
    d0_mm, d0_source = parse_d0_mm(run_dir)
    time_by_step = {int(row["step"]): row for row in log_summary["run_time_rows"]}

    state_rows = []
    state_details = []
    for step in chosen_steps:
        row, detail = analyze_state(run_dir, step, time_by_step, fixed_dt, adaptive_dt)
        state_rows.append(row)
        state_details.append(detail)
    enrich_d2(state_rows, d0_mm)

    trend_rows = []
    for step in trend_steps:
        row, _detail = analyze_state(run_dir, step, time_by_step, fixed_dt, adaptive_dt)
        trend_rows.append(row)
    enrich_d2(trend_rows, d0_mm)

    d_count, d_size = directory_size(run_dir / "D")
    p_all_count, p_all_size = directory_size(run_dir / "p_all")
    plots, time_plot_source = maybe_write_plots(out_dir, trend_rows, log_summary["run_time_rows"], fixed_dt, adaptive_dt)

    notes = []
    if all(not steps_by_field[name] for name in ("rhoY_C12H26", "rhoY_O2", "rhoY_CO2", "rhoY_H2O")):
        notes.append("chemistry/species raw fields are unavailable; this is expected for nonreacting chemistry-off cases")
    if log_summary["run_time_inf_dt_rounded_to_zero"] or log_summary["run_time_inf_time_rounded_to_zero"]:
        notes.append(
            "run_time.inf dt/time columns appear rounded to zero; saved-state times use "
            "an approximate initial-dt mapping when no usable log time is available"
        )

    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "raw_dirs_present": [str(path) for path in raw_dirs(run_dir)],
        "D_file_count": d_count,
        "D_bytes": d_size,
        "p_all_file_count": p_all_count,
        "p_all_bytes": p_all_size,
        "fields": {name: {"kind": kind, "index": index, "steps": steps_by_field[name]} for name, (kind, index) in FIELDS.items()},
        "selected_steps": chosen_steps,
        "trend_steps": trend_steps,
        "fixed_dt_from_simulation_inp": fixed_dt,
        "cfl_adap_dt": adaptive_dt,
        "D0_mm": d0_mm,
        "D0_source": d0_source,
        "mass_d2_estimate": final_mass_k(state_rows, d0_mm),
        "last_run_time": log_summary["last_run_time"],
        "min_dt": log_summary["min_dt"],
        "max_dt": log_summary["max_dt"],
        "min_dt_nonzero": log_summary["min_dt_nonzero"],
        "max_dt_nonzero": log_summary["max_dt_nonzero"],
        "max_time_nonzero": log_summary["max_time_nonzero"],
        "run_time_inf_dt_rounded_to_zero": log_summary["run_time_inf_dt_rounded_to_zero"],
        "run_time_inf_time_rounded_to_zero": log_summary["run_time_inf_time_rounded_to_zero"],
        "time_plot_source": time_plot_source,
        "state_details": state_details,
        "plots": plots,
        "notes": notes,
    }

    summary_path = out_dir / "smoke_diagnostics_summary.json"
    by_state_path = out_dir / "smoke_diagnostics_by_state.csv"
    trend_path = out_dir / "smoke_diagnostics_trend.csv"
    log_path = out_dir / "smoke_diagnostics_log_summary.txt"

    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
    write_csv(by_state_path, state_rows)
    write_csv(trend_path, trend_rows)
    write_log_text(log_path, log_summary)

    print(f"summary={summary_path}")
    print(f"by_state_csv={by_state_path}")
    print(f"trend_csv={trend_path}")
    print(f"log_summary={log_path}")
    for path in plots:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
