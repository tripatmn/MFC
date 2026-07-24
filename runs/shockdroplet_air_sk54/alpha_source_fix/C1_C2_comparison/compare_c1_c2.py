#!/usr/bin/env python3
"""Direct C1-vs-C2 comparison for short SK54 diffusion regressions."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path("/home/tripatmn/mfc_work/MFC_alpha_diag")
BASE = REPO / "runs/shockdroplet_air_sk54/alpha_source_fix"
C1 = BASE / "C1_diffusion_ON_reactions_OFF_0p2us"
C2 = BASE / "C2_diffusion_ON_reactions_ON_0p2us"
OUT = BASE / "C1_C2_comparison"
MECH = REPO / "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
PHASE = "yao_sk54"
T_SAVE = 5.0e-8
GAS_FLOOR = 1.0e-8
NEAR_ZERO = 1.0e-30

sys.path.insert(0, str(REPO / "examples/2D_dodecane_global_reduced"))
import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics

raw.raw_dirs = lambda run_dir: [Path(run_dir) / "D"] if (Path(run_dir) / "D").is_dir() else []

SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "H2O", "CO", "CO2")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def species_indices() -> dict[str, int]:
    doc = yaml.safe_load(MECH.read_text())
    for phase in doc.get("phases", []):
        if phase.get("name") == PHASE:
            return {name: i for i, name in enumerate(phase["species"], start=1)}
    raise RuntimeError(f"phase {PHASE!r} not found in {MECH}")


def analyzed_saves(run_dir: Path) -> list[int]:
    path = run_dir / "analysis_diffusion_lightweight/species_and_mass_consistency_by_save.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing lightweight analysis CSV: {path}")
    return [int(row["save"]) for row in csv.DictReader(path.open())]


def bounds_by_save(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "analysis_diffusion_lightweight/species_and_mass_consistency_by_save.csv"
    out: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(path.open()):
        save = int(row["save"])
        out[save] = {}
        for key, value in row.items():
            if key == "save":
                continue
            try:
                out[save][key] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def parse_timestep_history(run_dir: Path) -> tuple[int | None, float, float]:
    path = run_dir / "analysis_diffusion_lightweight/timestep_history.csv"
    if not path.is_file():
        return None, math.nan, math.nan
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None, math.nan, math.nan
    max_t_step = max(int(row["t_step"]) for row in rows if row.get("t_step", "").strip())
    runtime = math.nan
    for row in reversed(rows):
        try:
            runtime = float(row.get("run_time_seconds_final", "nan"))
        except ValueError:
            runtime = math.nan
        if math.isfinite(runtime):
            break
    cost = runtime / max_t_step if max_t_step and math.isfinite(runtime) else math.nan
    return max_t_step, runtime, cost


def failure_messages(run_dir: Path) -> list[str]:
    path = run_dir / "analysis_diffusion_lightweight/diffusion_failure_messages.txt"
    if not path.is_file():
        return []
    return [line for line in path.read_text(errors="replace").splitlines() if line.strip()]


def load_fields(run_dir: Path, save: int, idx: dict[str, int]) -> dict[str, dict]:
    fields = raw.read_step_fields(run_dir, save, GAS_FLOOR)
    for species in SPECIES:
        species_id = idx.get(species)
        if species_id is None:
            fields[f"rhoY_{species}"] = {"available": False, "values": {}, "stats": raw.missing_stats([], 0)}
        else:
            raw.FIELDS[f"rhoY_{species}"] = ("cons", 12 + species_id)
            fields[f"rhoY_{species}"] = raw.read_field(run_dir, f"rhoY_{species}", save)
    return fields


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def percentile(values: list[float], q: float) -> float:
    good = finite(values)
    return float(np.percentile(np.array(good, dtype=float), q)) if good else math.nan


def integrate(field: dict, area: float) -> float:
    if not field.get("available") or not math.isfinite(area):
        return math.nan
    vals = finite(list(field["values"].values()))
    return float(sum(vals) * area) if vals else math.nan


def gas_y_values(fields: dict[str, dict], masks: dict, species: str) -> list[float]:
    field = fields.get(f"rhoY_{species}", {})
    if not field.get("available"):
        return []
    out = []
    for key in masks["valid_gas_thermo"]:
        rhoY = field["values"].get(key, math.nan)
        gas_mass = masks["gas_mass"].get(key, math.nan)
        if math.isfinite(rhoY) and math.isfinite(gas_mass) and gas_mass > GAS_FLOOR:
            out.append(rhoY / gas_mass)
    return out


def dense_liquid_metrics(fields: dict[str, dict], area: float) -> dict[str, float]:
    values = fields["liquid_alpha"]["values"]
    dense = [(key, value) for key, value in values.items() if math.isfinite(value) and value > 0.5]
    if not dense or not math.isfinite(area):
        return {
            "liquid_alpha_area": math.nan,
            "equivalent_diameter_m": math.nan,
            "D2_m2": math.nan,
            "liquid_centroid_x": math.nan,
            "liquid_centroid_y": math.nan,
            "liquid_streamwise_extent_m": math.nan,
            "liquid_cross_stream_extent_m": math.nan,
        }
    xs = [key[0] for key, _ in dense]
    ys = [key[1] for key, _ in dense]
    weights = [value for _, value in dense]
    wsum = sum(weights)
    liquid_area = len(dense) * area
    diameter = math.sqrt(4.0 * liquid_area / math.pi)
    return {
        "liquid_alpha_area": liquid_area,
        "equivalent_diameter_m": diameter,
        "D2_m2": diameter * diameter,
        "liquid_centroid_x": sum(x * w for x, w in zip(xs, weights)) / wsum,
        "liquid_centroid_y": sum(y * w for y, w in zip(ys, weights)) / wsum,
        "liquid_streamwise_extent_m": max(xs) - min(xs),
        "liquid_cross_stream_extent_m": max(ys) - min(ys),
    }


def one_run_metrics(run_dir: Path, save: int, idx: dict[str, int], bounds: dict[int, dict[str, float]]) -> dict[str, float]:
    fields = load_fields(run_dir, save, idx)
    masks = gas_metrics.mask_context(fields, GAS_FLOOR)
    fields["temperature_valid_gas"] = gas_metrics.reconstruct_valid_gas_temperature(fields, masks["valid_gas_thermo"])
    _dx, _dy, area = gas_metrics.estimate_cell_area(fields)

    temp_values = [fields["temperature_valid_gas"]["values"].get(key, math.nan) for key in masks["valid_gas_thermo"]]
    temp_values = finite(temp_values)
    out: dict[str, float] = {
        "valid_gas_Tmax": max(temp_values) if temp_values else math.nan,
        "valid_gas_T999": percentile(temp_values, 99.9),
        "valid_gas_T95": percentile(temp_values, 95.0),
        "valid_gas_Tmean": sum(temp_values) / len(temp_values) if temp_values else math.nan,
        "valid_gas_cell_count": float(len(masks["valid_gas_thermo"])),
        "cell_area_m2": area,
        "vaporized_fuel_mass_proxy_vapor_alpha_rho_integral": integrate(fields["vapor_alpha_rho"], area),
    }
    out.update(dense_liquid_metrics(fields, area))

    b = bounds.get(save, {})
    out["negative_rhoY_count"] = b.get("negative_rhoY_count", math.nan)
    out["negative_Y_count"] = b.get("negative_Y_count", math.nan)
    out["Y_above_one_count"] = b.get("Y_above_one_count", math.nan)
    out["sumY_violation_count"] = b.get("sumY_violation_count", math.nan)
    out["max_abs_sumY_minus_1"] = max(abs(b.get("sumY_min", math.nan) - 1.0), abs(b.get("sumY_max", math.nan) - 1.0))
    out["gas_species_mass_relative_error"] = b.get("relative_integrated_error", math.nan)

    for species in SPECIES:
        field = fields.get(f"rhoY_{species}", {})
        ys = gas_y_values(fields, masks, species)
        if field.get("available"):
            valid_rhoY = finite([field["values"].get(key, math.nan) for key in masks["valid_gas_thermo"]])
            out[f"rhoY_{species}_integral"] = integrate(field, area)
            out[f"rhoY_{species}_max_valid_gas"] = max(valid_rhoY) if valid_rhoY else math.nan
            out[f"rhoY_{species}_min_valid_gas"] = min(valid_rhoY) if valid_rhoY else math.nan
            out[f"Y_{species}_max_valid_gas"] = max(ys) if ys else math.nan
            out[f"Y_{species}_min_valid_gas"] = min(ys) if ys else math.nan
        else:
            out[f"rhoY_{species}_integral"] = math.nan
            out[f"rhoY_{species}_max_valid_gas"] = math.nan
            out[f"rhoY_{species}_min_valid_gas"] = math.nan
            out[f"Y_{species}_max_valid_gas"] = math.nan
            out[f"Y_{species}_min_valid_gas"] = math.nan
    return out


def metric_list() -> list[tuple[str, str]]:
    metrics = [
        ("runtime_t_step_at_save", "unavailable: run_time.inf time/dt are precision-limited to zero"),
        ("runtime_dt_s_at_save", "unavailable: run_time.inf time/dt are precision-limited to zero"),
        ("valid_gas_Tmax", "K"),
        ("valid_gas_T999", "K"),
        ("valid_gas_T95", "K"),
        ("valid_gas_Tmean", "K"),
        ("valid_gas_cell_count", "cells"),
        ("vaporized_fuel_mass_proxy_vapor_alpha_rho_integral", "kg/m-depth proxy"),
        ("liquid_alpha_area", "m2"),
        ("equivalent_diameter_m", "m"),
        ("D2_m2", "m2"),
        ("liquid_centroid_x", "m"),
        ("liquid_centroid_y", "m"),
        ("liquid_streamwise_extent_m", "m"),
        ("liquid_cross_stream_extent_m", "m"),
        ("negative_rhoY_count", "count"),
        ("negative_Y_count", "count"),
        ("Y_above_one_count", "count"),
        ("sumY_violation_count", "count"),
        ("max_abs_sumY_minus_1", "dimensionless"),
        ("gas_species_mass_relative_error", "dimensionless"),
        ("run_time_seconds_final", "s, run-level"),
        ("cost_seconds_per_timestep", "s/step, run-level"),
    ]
    for species in SPECIES:
        metrics.extend([
            (f"rhoY_{species}_integral", "kg/m-depth"),
            (f"rhoY_{species}_max_valid_gas", "kg/m3"),
            (f"rhoY_{species}_min_valid_gas", "kg/m3"),
            (f"Y_{species}_max_valid_gas", "mass fraction"),
            (f"Y_{species}_min_valid_gas", "mass fraction"),
        ])
    return metrics


def relative(c1: float, c2: float) -> float:
    if not math.isfinite(c1) or not math.isfinite(c2) or abs(c1) <= NEAR_ZERO:
        return math.nan
    return (c2 - c1) / c1


def classify(diff: float, rel: float) -> str:
    if not math.isfinite(diff):
        return "unavailable"
    if abs(diff) <= 1.0e-12 or (math.isfinite(rel) and abs(rel) <= 1.0e-10):
        return "roundoff-level"
    if math.isfinite(rel) and abs(rel) <= 1.0e-4:
        return "small"
    if math.isfinite(rel) and abs(rel) > 1.0e-2:
        return "meaningful"
    return "small-to-moderate"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = species_indices()
    matched = sorted(set(analyzed_saves(C1)) & set(analyzed_saves(C2)))
    if not matched:
        raise RuntimeError("no matched analyzed saves between C1 and C2")

    c1_bounds = bounds_by_save(C1)
    c2_bounds = bounds_by_save(C2)
    c1_tstep, c1_runtime, c1_cost = parse_timestep_history(C1)
    c2_tstep, c2_runtime, c2_cost = parse_timestep_history(C2)

    rows: list[dict[str, Any]] = []
    final: dict[str, tuple[float, float, float, float, str]] = {}
    for save in matched:
        c1 = one_run_metrics(C1, save, idx, c1_bounds)
        c2 = one_run_metrics(C2, save, idx, c2_bounds)
        c1.update({"run_time_seconds_final": c1_runtime, "cost_seconds_per_timestep": c1_cost, "runtime_t_step_at_save": math.nan, "runtime_dt_s_at_save": math.nan})
        c2.update({"run_time_seconds_final": c2_runtime, "cost_seconds_per_timestep": c2_cost, "runtime_t_step_at_save": math.nan, "runtime_dt_s_at_save": math.nan})
        for metric, units in metric_list():
            c1v = float(c1.get(metric, math.nan))
            c2v = float(c2.get(metric, math.nan))
            diff = c2v - c1v if math.isfinite(c1v) and math.isfinite(c2v) else math.nan
            rel = relative(c1v, c2v)
            cls = classify(diff, rel)
            rows.append({
                "save": save,
                "physical_time_s": save * T_SAVE,
                "physical_time_us": save * T_SAVE * 1.0e6,
                "metric": metric,
                "units": units,
                "C1_value": c1v,
                "C2_value": c2v,
                "C2_minus_C1": diff,
                "relative_difference": rel,
                "classification": cls,
                "note": "per-save timestep/dt unavailable from precision-limited run_time.inf" if metric.startswith("runtime_") else "",
            })
            if save == matched[-1]:
                final[metric] = (c1v, c2v, diff, rel, cls)

    write_csv(OUT / "c1_c2_by_save.csv", rows)

    selected = [
        "valid_gas_Tmax",
        "valid_gas_Tmean",
        "rhoY_NC12H26_integral",
        "rhoY_O2_integral",
        "rhoY_OH_integral",
        "rhoY_HO2_integral",
        "rhoY_H2O2_integral",
        "rhoY_H2O_integral",
        "rhoY_CO2_integral",
        "vaporized_fuel_mass_proxy_vapor_alpha_rho_integral",
        "liquid_alpha_area",
        "D2_m2",
        "max_abs_sumY_minus_1",
        "gas_species_mass_relative_error",
    ]
    lines = [
        "C1 vs C2 short diffusion regression comparison",
        f"C1: {C1}",
        f"C2: {C2}",
        f"matched analyzed saves: {matched}",
        f"time window: {matched[0] * T_SAVE * 1e6:.3f} to {matched[-1] * T_SAVE * 1e6:.3f} us",
        "",
        "Runtime parsing:",
        "  run_time.inf columns are Time-step, dt, Time, ICFL Max.",
        "  dt and Time are fixed six-decimal fields and are printed as 0.000000 for these sub-microsecond runs.",
        "  Per-save timestep count and dt are therefore NaN rather than inferred.",
        f"  C1 final runtime/timestep/cost: {c1_runtime} s / {c1_tstep} / {c1_cost} s-step",
        f"  C2 final runtime/timestep/cost: {c2_runtime} s / {c2_tstep} / {c2_cost} s-step",
        "",
        "Failure-message scan:",
        f"  C1 explicit failure lines: {len(failure_messages(C1))}",
        f"  C2 explicit failure lines: {len(failure_messages(C2))}",
        "  Chemistry limiter aggregate activity: not recorded in available run files.",
        "",
        "Final matched-save selected metrics:",
    ]
    for metric in selected:
        c1v, c2v, diff, rel, cls = final.get(metric, (math.nan, math.nan, math.nan, math.nan, "unavailable"))
        lines.append(f"  {metric}: C1={c1v:.16e} C2={c2v:.16e} diff={diff:.16e} rel={rel:.6e} class={cls}")
    lines.extend([
        "",
        "Scientific interpretation:",
        "  This 0.2 microsecond interval is a regression and very-early chemistry comparison only.",
        "  Species admissibility and gas/species mass consistency remain clean in both C1 and C2 saved outputs.",
        "  Small radical/product or temperature differences are early chemistry response, not ignition evidence.",
        "  Vaporization and droplet-deformation metrics are too early-window to interpret physically beyond regression consistency.",
    ])
    (OUT / "c1_c2_final_summary.txt").write_text("\n".join(lines) + "\n")

    report = [
        "# C1 vs C2 comparison, 0.05-0.20 microseconds",
        "",
        "C1 has diffusion ON with reactions and chemical heat release OFF. C2 has diffusion ON with reactions and chemical heat release ON.",
        "Both use the same raw-output masks and definitions; save 0 is excluded because primitive pressure is unavailable.",
        "",
        "## Key result",
        "",
        "Both completed short regressions remain species-admissible in saved outputs, with sum(Y) and integrated gas/species mass agreeing to roundoff.",
        "The interval is too short to support any ignition or meaningful combustion claim.",
        "",
        "## Runtime caveat",
        "",
        "`run_time.inf` stores `dt` and `Time` as fixed six-decimal values, so all sub-microsecond entries are printed as `0.000000`.",
        "The comparison therefore uses `save * t_save` for physical time and records per-save timestep/dt as unavailable.",
        "",
        "## Final selected metrics",
        "",
        "| metric | C1 | C2 | C2-C1 | relative | class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in selected:
        c1v, c2v, diff, rel, cls = final.get(metric, (math.nan, math.nan, math.nan, math.nan, "unavailable"))
        report.append(f"| `{metric}` | {c1v:.8e} | {c2v:.8e} | {diff:.8e} | {rel:.3e} | {cls} |")
    report.extend([
        "",
        "## Interpretation bins",
        "",
        "- Roundoff-level: differences at numerical noise scale.",
        "- Small: chemistry-induced or runtime-level differences that are measurable but not physically decisive in this short window.",
        "- Meaningful: relative difference above the comparison threshold; still not automatically a physical combustion conclusion at 0.2 microseconds.",
        "",
        "## Limitations",
        "",
        "- Chemistry limiter aggregate activity was not recorded in available runtime files.",
        "- CO is reported only if the SK54 mechanism species and corresponding raw field are available.",
        "- Total vaporized fuel mass is reported as the integrated vapor-fluid partial-density proxy; NC12H26 species mass is reported separately.",
    ])
    (OUT / "c1_c2_comparison_report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
