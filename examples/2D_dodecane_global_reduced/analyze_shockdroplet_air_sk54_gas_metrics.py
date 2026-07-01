#!/usr/bin/env python3
"""Gas/interface-aware diagnostics for raw shock-droplet SK54 output.

This extends ``analyze_shockdroplet_air_sk54.py`` with masks that avoid
interpreting liquid-dominated cells as gas thermodynamic states. It reads raw
``D/`` and/or ``p_all/`` files directly and does not require MFC post_process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-shockdroplet-gas")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_shockdroplet_air_sk54 as raw


SPECIES = ("NC12H26", "O2", "CO2", "H2O", "OH", "HO2", "H2O2")
PANEL_FIELDS = ("temperature_valid_gas", "rhoY_NC12H26", "rhoY_O2", "rhoY_OH", "rhoY_HO2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="MFC run folder containing raw D/ or p_all/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output diagnostics folder")
    parser.add_argument("--gas-mass-floor", type=float, default=1.0e-8)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all saved states")
    return parser.parse_args()


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def stats(values: list[float]) -> dict[str, float | int]:
    good = finite(values)
    return {
        "count": len(good),
        "min": min(good) if good else math.nan,
        "max": max(good) if good else math.nan,
        "mean": sum(good) / len(good) if good else math.nan,
        "sum": sum(good) if good else 0.0,
    }


def estimate_cell_area(fields: dict[str, dict]) -> tuple[float, float, float]:
    """Estimate dx, dy, and cell area from raw field coordinates."""
    for name in ("pressure", "liquid_alpha", "rhoY_O2"):
        field = fields.get(name)
        if not field or not field["available"]:
            continue
        values = field["values"]
        xs = np.array(sorted({key[0] for key in values}), dtype=float)
        ys = np.array(sorted({key[1] for key in values}), dtype=float)
        if xs.size < 2 or ys.size < 2:
            continue
        dxs = np.diff(xs)
        dys = np.diff(ys)
        dxs = dxs[np.isfinite(dxs) & (dxs > 0.0)]
        dys = dys[np.isfinite(dys) & (dys > 0.0)]
        if dxs.size and dys.size:
            dx = float(np.median(dxs))
            dy = float(np.median(dys))
            return dx, dy, dx * dy
    return math.nan, math.nan, math.nan


def mask_context(fields: dict[str, dict], gas_mass_floor: float) -> dict:
    required = ("liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho", "pressure")
    if not all(fields[name]["available"] for name in required):
        return {
            "keys": [],
            "gas_dominant": set(),
            "interface": set(),
            "liquid_dominated": set(),
            "valid_gas_thermo": set(),
            "gas_alpha": {},
            "gas_mass": {},
            "warnings": ["missing mask input fields"],
        }

    keys = set(fields["liquid_alpha"]["values"])
    for name in required[1:]:
        keys &= set(fields[name]["values"])
    gas_dominant = set()
    interface = set()
    liquid_dominated = set()
    valid_gas_thermo = set()
    gas_alpha_by_key = {}
    gas_mass_by_key = {}
    warnings = []

    for key in keys:
        alpha_liq = fields["liquid_alpha"]["values"][key]
        alpha_vap = fields["vapor_alpha"]["values"][key]
        alpha_air = fields["air_alpha"]["values"][key]
        arho_vap = fields["vapor_alpha_rho"]["values"][key]
        arho_air = fields["air_alpha_rho"]["values"][key]
        gas_alpha = alpha_vap + alpha_air
        gas_mass = arho_vap + arho_air
        gas_alpha_by_key[key] = gas_alpha
        gas_mass_by_key[key] = gas_mass
        if not all(math.isfinite(v) for v in (alpha_liq, alpha_vap, alpha_air, arho_vap, arho_air, gas_alpha, gas_mass)):
            continue
        if gas_alpha > 0.9 and alpha_liq < 0.1:
            gas_dominant.add(key)
        if 0.01 < alpha_liq < 0.99:
            interface.add(key)
        if alpha_liq > 0.99:
            liquid_dominated.add(key)
        if gas_alpha > 0.5 and gas_mass > gas_mass_floor and alpha_liq < 0.5:
            valid_gas_thermo.add(key)
    return {
        "keys": keys,
        "gas_dominant": gas_dominant,
        "interface": interface,
        "liquid_dominated": liquid_dominated,
        "valid_gas_thermo": valid_gas_thermo,
        "gas_alpha": gas_alpha_by_key,
        "gas_mass": gas_mass_by_key,
        "warnings": warnings,
    }


def field_values(fields: dict[str, dict], name: str, keys: set | None = None) -> list[float]:
    field = fields.get(name)
    if not field or not field["available"]:
        return []
    values = field["values"]
    if keys is None:
        return list(values.values())
    return [values[key] for key in keys if key in values]


def masked_stats(fields: dict[str, dict], name: str, keys: set) -> dict[str, float | int]:
    return stats(field_values(fields, name, keys))


def extrema_location(fields: dict[str, dict], name: str, keys: set) -> tuple[float, tuple[float, float] | None]:
    field = fields.get(name)
    if not field or not field["available"]:
        return math.nan, None
    best = -math.inf
    best_key = None
    for key in keys:
        value = field["values"].get(key, math.nan)
        if math.isfinite(value) and value > best:
            best = value
            best_key = key
    if best_key is None:
        return math.nan, None
    return float(best), best_key


def local(fields: dict[str, dict], name: str, key: tuple[float, float] | None) -> float:
    if key is None:
        return math.nan
    field = fields.get(name)
    if not field or not field["available"]:
        return math.nan
    return float(field["values"].get(key, math.nan))


def add_local_context(row: dict, fields: dict[str, dict], masks: dict, prefix: str, key: tuple[float, float] | None) -> None:
    row[f"{prefix}_x"] = key[0] if key is not None else math.nan
    row[f"{prefix}_y"] = key[1] if key is not None else math.nan
    row[f"{prefix}_alpha_liq"] = local(fields, "liquid_alpha", key)
    row[f"{prefix}_gas_alpha"] = masks["gas_alpha"].get(key, math.nan) if key is not None else math.nan
    row[f"{prefix}_gas_mass"] = masks["gas_mass"].get(key, math.nan) if key is not None else math.nan
    row[f"{prefix}_pressure"] = local(fields, "pressure", key)
    for species in SPECIES:
        row[f"{prefix}_rhoY_{species}"] = local(fields, f"rhoY_{species}", key)


def reconstruct_valid_gas_temperature(fields: dict[str, dict], valid_keys: set) -> dict:
    raw_temp = fields.get("temperature")
    if not raw_temp or not raw_temp["available"]:
        return {"available": False, "values": {}, "stats": raw.missing_stats([], 0)}
    values = {key: raw_temp["values"][key] for key in valid_keys if key in raw_temp["values"]}
    return {
        "available": bool(values),
        "values": values,
        "stats": raw.stats_from_values(values, available=bool(values)),
    }


def liquid_centroid(fields: dict[str, dict]) -> tuple[float, float]:
    x, y, _w = raw.centroid_from_alpha(fields["liquid_alpha"])
    return x, y


def row_for_state(step: int, save_index: int, time_s: float, time_source: str, fields: dict[str, dict], gas_mass_floor: float) -> dict:
    masks = mask_context(fields, gas_mass_floor)
    fields["temperature_valid_gas"] = reconstruct_valid_gas_temperature(fields, masks["valid_gas_thermo"])
    row: dict[str, float | int | str] = {
        "save_index": save_index,
        "step": step,
        "time_s": time_s,
        "time_source": time_source,
        "gas_mass_floor": gas_mass_floor,
        "total_cell_count": len(masks["keys"]),
        "gas_dominant_cell_count": len(masks["gas_dominant"]),
        "interface_cell_count": len(masks["interface"]),
        "liquid_dominated_cell_count": len(masks["liquid_dominated"]),
        "valid_gas_thermo_cell_count": len(masks["valid_gas_thermo"]),
        "total_nonfinite_count": sum(field["stats"]["nonfinite_count"] for field in fields.values() if "stats" in field),
    }
    dx, dy, cell_area = estimate_cell_area(fields)
    row["dx_m"] = dx
    row["dy_m"] = dy
    row["cell_area_m2"] = cell_area
    total_cells = max(len(masks["keys"]), 1)
    row["gas_dominant_area_fraction"] = len(masks["gas_dominant"]) / total_cells
    row["interface_area_fraction"] = len(masks["interface"]) / total_cells
    row["liquid_dominated_area_fraction"] = len(masks["liquid_dominated"]) / total_cells
    row["valid_gas_thermo_area_fraction"] = len(masks["valid_gas_thermo"]) / total_cells
    cx, cy = liquid_centroid(fields)
    row["liquid_alpha_centroid_x"] = cx
    row["liquid_alpha_centroid_y"] = cy

    for mask_name, keys in (("all", None), ("gas_dominant", masks["gas_dominant"]), ("interface", masks["interface"])):
        pstats = stats(field_values(fields, "pressure", keys))
        row[f"pressure_{mask_name}_min"] = pstats["min"]
        row[f"pressure_{mask_name}_max"] = pstats["max"]

    raw_tmax, raw_tkey = extrema_location(fields, "temperature", set(fields.get("temperature", {}).get("values", {})))
    valid_tmax, valid_tkey = extrema_location(fields, "temperature_valid_gas", masks["valid_gas_thermo"])
    row["raw_temperature_max"] = raw_tmax
    row["raw_temperature_max_in_liquid_dominated"] = bool(raw_tkey in masks["liquid_dominated"]) if raw_tkey is not None else False
    row["valid_gas_temperature_min"] = fields["temperature_valid_gas"]["stats"]["min"]
    row["valid_gas_temperature_max"] = valid_tmax
    add_local_context(row, fields, masks, "valid_gas_Tmax_cell", valid_tkey)
    add_local_context(row, fields, masks, "raw_Tmax_cell", raw_tkey)

    for species in SPECIES:
        field_name = f"rhoY_{species}"
        whole = stats(field_values(fields, field_name))
        gas = masked_stats(fields, field_name, masks["gas_dominant"])
        interface = masked_stats(fields, field_name, masks["interface"])
        row[f"{field_name}_sum"] = whole["sum"]
        row[f"{field_name}_gas_dominant_sum"] = gas["sum"]
        row[f"{field_name}_interface_sum"] = interface["sum"]
        row[f"{field_name}_volume_integral"] = whole["sum"] * cell_area if math.isfinite(cell_area) else math.nan
        row[f"{field_name}_gas_dominant_volume_integral"] = (
            gas["sum"] * cell_area if math.isfinite(cell_area) else math.nan
        )
        row[f"{field_name}_interface_volume_integral"] = (
            interface["sum"] * cell_area if math.isfinite(cell_area) else math.nan
        )
        row[f"{field_name}_mean"] = whole["mean"]
        row[f"{field_name}_gas_dominant_mean"] = gas["mean"]
        row[f"{field_name}_interface_mean"] = interface["mean"]
        row[f"{field_name}_gas_dominant_max"] = gas["max"]
        row[f"{field_name}_interface_max"] = interface["max"]

    for species in ("OH", "HO2"):
        field_name = f"rhoY_{species}"
        gas_max, gas_key = extrema_location(fields, field_name, masks["gas_dominant"])
        int_max, int_key = extrema_location(fields, field_name, masks["interface"])
        row[f"{field_name}_gas_dominant_max_location_value"] = gas_max
        add_local_context(row, fields, masks, f"{field_name}_gas_dominant_max_cell", gas_key)
        row[f"{field_name}_interface_max_location_value"] = int_max
        add_local_context(row, fields, masks, f"{field_name}_interface_max_cell", int_key)

    o2_initial = None
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


def plot_timeseries(rows: list[dict], out_dir: Path) -> int:
    if not rows:
        return 0
    count = 0
    t = np.array([float(row["time_s"]) * 1.0e6 for row in rows])
    plots = [
        ("valid_gas_temperature_max", "Valid gas Tmax [K]", "valid_gas_temperature_max.png"),
        ("rhoY_OH_sum", "Domain-summed rhoY_OH", "OH_sum.png"),
        ("rhoY_HO2_sum", "Domain-summed rhoY_HO2", "HO2_sum.png"),
        ("rhoY_NC12H26_sum", "Domain-summed rhoY_NC12H26", "NC12H26_sum.png"),
        ("rhoY_O2_sum", "Domain-summed rhoY_O2", "O2_sum.png"),
        ("rhoY_CO2_sum", "Domain-summed rhoY_CO2", "CO2_sum.png"),
        ("rhoY_H2O_sum", "Domain-summed rhoY_H2O", "H2O_sum.png"),
    ]
    for key, ylabel, name in plots:
        y = np.array([float(row.get(key, math.nan)) for row in rows])
        if not np.isfinite(y).any():
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(t, y, marker="o", linewidth=1.8)
        ax.set_xlabel("Time [us]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=220)
        plt.close(fig)
        count += 1

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label in (("rhoY_CO2_sum", "CO2"), ("rhoY_H2O_sum", "H2O")):
        y = np.array([float(row.get(key, math.nan)) for row in rows])
        if np.isfinite(y).any():
            ax.plot(t, y, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Domain-summed rhoY")
    ax.set_title("Product species growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "CO2_H2O_sum.png", dpi=220)
    plt.close(fig)
    count += 1
    return count


def plot_final_fields(fields: dict[str, dict], out_dir: Path, step: int, time_s: float) -> int:
    names = ["temperature_valid_gas", "rhoY_NC12H26", "rhoY_O2", "rhoY_OH", "rhoY_HO2", "rhoY_CO2", "rhoY_H2O"]
    titles = ["Valid gas T [K]", "NC12H26 rhoY", "O2 rhoY", "OH rhoY", "HO2 rhoY", "CO2 rhoY", "H2O rhoY"]
    raw.plot_panel(fields, names, titles, out_dir / f"final_gas_chem_fields_step_{step:06d}.png", time_s)
    return 1


def write_summary(path: Path, rows: list[dict], run_dir: Path, png_count: int, warnings: list[str]) -> None:
    lines = [
        "Shock-droplet gas/interface metrics",
        f"run_dir: {run_dir}",
        f"saved states analyzed: {len(rows)}",
        f"PNG files written: {png_count}",
        "",
        "Masks:",
        "  gas_dominant: gas_alpha > 0.9 and alpha_liq < 0.1",
        "  interface: 0.01 < alpha_liq < 0.99",
        "  liquid_dominated: alpha_liq > 0.99",
        "  valid_gas_thermo: gas_alpha > 0.5, gas_mass > gas_mass_floor, alpha_liq < 0.5",
        "  raw temperature is unsafe near liquid/interface cells and is reported only as a warning aid.",
    ]
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in warnings)
    if rows:
        first = rows[0]
        final = rows[-1]
        lines.extend([
            "",
            "Final saved state:",
            f"  step/time: {final['step']} / {float(final['time_s']):.6e} s ({final['time_source']})",
            f"  pressure_all_min/max: {final['pressure_all_min']} / {final['pressure_all_max']} Pa",
            f"  pressure_gas_dominant_min/max: {final['pressure_gas_dominant_min']} / {final['pressure_gas_dominant_max']} Pa",
            f"  pressure_interface_min/max: {final['pressure_interface_min']} / {final['pressure_interface_max']} Pa",
            f"  total_nonfinite_count: {final['total_nonfinite_count']}",
            f"  gas_dominant/interface/valid_gas counts: {final['gas_dominant_cell_count']} / {final['interface_cell_count']} / {final['valid_gas_thermo_cell_count']}",
            f"  valid_gas_temperature_max: {final['valid_gas_temperature_max']} K",
            f"  valid_gas_Tmax location: ({final['valid_gas_Tmax_cell_x']}, {final['valid_gas_Tmax_cell_y']}) m",
            f"  valid_gas_Tmax local alpha_liq/gas_alpha/gas_mass/p: {final['valid_gas_Tmax_cell_alpha_liq']} / {final['valid_gas_Tmax_cell_gas_alpha']} / {final['valid_gas_Tmax_cell_gas_mass']} / {final['valid_gas_Tmax_cell_pressure']}",
            f"  raw_temperature_max: {final['raw_temperature_max']} K",
            f"  raw_Tmax_in_liquid_dominated: {final['raw_temperature_max_in_liquid_dominated']}",
            f"  NC12H26 sum initial/final/delta: {first['rhoY_NC12H26_sum']} / {final['rhoY_NC12H26_sum']} / {final['rhoY_NC12H26_sum'] - first['rhoY_NC12H26_sum']}",
            f"  O2 sum initial/final/delta: {first['rhoY_O2_sum']} / {final['rhoY_O2_sum']} / {final['rhoY_O2_sum'] - first['rhoY_O2_sum']}",
            f"  CO2 sum initial/final/delta: {first['rhoY_CO2_sum']} / {final['rhoY_CO2_sum']} / {final['rhoY_CO2_sum'] - first['rhoY_CO2_sum']}",
            f"  H2O sum initial/final/delta: {first['rhoY_H2O_sum']} / {final['rhoY_H2O_sum']} / {final['rhoY_H2O_sum'] - first['rhoY_H2O_sum']}",
            f"  OH sum initial/final/delta: {first['rhoY_OH_sum']} / {final['rhoY_OH_sum']} / {final['rhoY_OH_sum'] - first['rhoY_OH_sum']}",
            f"  HO2 sum initial/final/delta: {first['rhoY_HO2_sum']} / {final['rhoY_HO2_sum']} / {final['rhoY_HO2_sum'] - first['rhoY_HO2_sum']}",
            f"  H2O2 sum initial/final/delta: {first['rhoY_H2O2_sum']} / {final['rhoY_H2O2_sum']} / {final['rhoY_H2O2_sum'] - first['rhoY_H2O2_sum']}",
        ])
    else:
        lines.extend(["", "No saved raw states were found."])
    path.write_text("\n".join(lines) + "\n")


def analyze(run_dir: Path, out_dir: Path, gas_mass_floor: float, max_frames: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = raw.available_steps(run_dir)
    times = raw.infer_times(run_dir, steps)
    rows: list[dict] = []
    warnings: list[str] = []
    if not steps:
        warnings.append("No raw D/ or p_all saved states found.")

    frame_limit = len(steps) if max_frames <= 0 else min(max_frames, len(steps))
    final_fields = None
    final_step = -1
    final_time = math.nan
    for save_index, step in enumerate(steps):
        time_s, time_source = times.get(step, (math.nan, "missing"))
        fields = raw.read_step_fields(run_dir, step, gas_mass_floor)
        row = row_for_state(step, save_index, time_s, time_source, fields, gas_mass_floor)
        rows.append(row)
        final_fields = fields
        final_step = step
        final_time = time_s
        if bool(row.get("raw_temperature_max_in_liquid_dominated")):
            warnings.append(f"raw Tmax occurs in liquid-dominated cell at save_index={save_index}, step={step}")

        if save_index < frame_limit:
            raw.plot_panel(
                fields,
                list(PANEL_FIELDS),
                ["Valid gas T [K]", "NC12H26 rhoY", "O2 rhoY", "OH rhoY", "HO2 rhoY"],
                out_dir / f"frame_{save_index:06d}_gas_metrics.png",
                time_s,
            )

    write_csv(out_dir / "gas_metrics_timeseries.csv", rows)
    png_count = plot_timeseries(rows, out_dir)
    if final_fields is not None:
        png_count += plot_final_fields(final_fields, out_dir, final_step, final_time)
    write_summary(out_dir / "gas_metrics_key_summary.txt", rows, run_dir, png_count + min(frame_limit, len(steps)), warnings)
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "gas_mass_floor": gas_mass_floor,
        "saved_states": len(rows),
        "masks": {
            "gas_dominant": "gas_alpha > 0.9 and alpha_liq < 0.1",
            "interface": "0.01 < alpha_liq < 0.99",
            "liquid_dominated": "alpha_liq > 0.99",
            "valid_gas_thermo": "gas_alpha > 0.5 and gas_mass > gas_mass_floor and alpha_liq < 0.5",
        },
        "temperature_formula": raw.reconstruct_temperature.__doc__ or "T = p/(rho_g R_mix) from base analyzer",
        "warnings": warnings,
    }
    (out_dir / "gas_metrics_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return rows


def main() -> None:
    args = parse_args()
    rows = analyze(args.run_dir.resolve(), args.out_dir.resolve(), args.gas_mass_floor, args.max_frames)
    print(f"Analyzed {len(rows)} saved states from {args.run_dir}")
    print(f"Wrote gas metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
