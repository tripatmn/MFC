#!/usr/bin/env python3
"""Two-zone chemistry budgets for reacting shock-droplet SK54 output.

The script reads raw MFC ``D/`` and/or ``p_all/`` output directly, using the
same field mapping as the shock-droplet analyzers. It compares chemistry
species budgets in an interface shell against a narrow gas window surrounding
the detected shock front.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mfc-shockdroplet-zones")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_shockdroplet_air_sk54 as raw
import analyze_shockdroplet_air_sk54_gas_metrics as gas_metrics


SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO2", "H2O")
ZONES = ("interface", "shock_front_gas", "gas_dominant")
ZONE_LABELS = {
    "interface": "Interface shell",
    "shock_front_gas": "Shock-front gas",
    "gas_dominant": "Whole gas-dominant",
}
ZONE_COLORS = {
    "interface": "#d62728",
    "shock_front_gas": "#9467bd",
    "gas_dominant": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run folder containing raw D/ or p_all/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for budget CSV/plots")
    parser.add_argument(
        "--shock-half-width-um",
        type=float,
        default=20.0,
        help="Half-width of shock-front gas window in microns.",
    )
    parser.add_argument("--gas-mass-floor", type=float, default=1.0e-8)
    return parser.parse_args()


def finite_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_fields(run_dir: Path, step: int, gas_mass_floor: float) -> dict[str, dict]:
    fields = raw.read_step_fields(run_dir, step, gas_mass_floor)
    masks = gas_metrics.mask_context(fields, gas_mass_floor)
    fields["temperature_valid_gas"] = gas_metrics.reconstruct_valid_gas_temperature(fields, masks["valid_gas_thermo"])

    keys = set()
    for name in ("liquid_alpha", "vapor_alpha", "air_alpha", "vapor_alpha_rho", "air_alpha_rho"):
        field = fields.get(name)
        if field and field["available"]:
            keys = set(field["values"]) if not keys else keys & set(field["values"])
    gas_alpha_values = {}
    gas_mass_values = {}
    for key in keys:
        gas_alpha_values[key] = fields["vapor_alpha"]["values"][key] + fields["air_alpha"]["values"][key]
        gas_mass_values[key] = fields["vapor_alpha_rho"]["values"][key] + fields["air_alpha_rho"]["values"][key]
    fields["gas_alpha"] = {
        "available": bool(gas_alpha_values),
        "values": gas_alpha_values,
        "stats": raw.stats_from_values(gas_alpha_values, available=bool(gas_alpha_values)),
    }
    fields["gas_mass"] = {
        "available": bool(gas_mass_values),
        "values": gas_mass_values,
        "stats": raw.stats_from_values(gas_mass_values, available=bool(gas_mass_values)),
    }
    return fields


def zone_masks(fields: dict[str, dict], shock_x: float, shock_half_width_m: float, gas_mass_floor: float) -> dict[str, set]:
    masks = gas_metrics.mask_context(fields, gas_mass_floor)
    interface = set(masks["interface"])
    gas_dominant = set(masks["gas_dominant"])
    if math.isfinite(shock_x):
        shock = {key for key in gas_dominant if abs(key[0] - shock_x) <= shock_half_width_m}
    else:
        shock = set()
    return {
        "interface": interface,
        "shock_front_gas": shock,
        "gas_dominant": gas_dominant,
    }


def sum_species(fields: dict[str, dict], species: str, keys: set) -> float:
    field = fields.get(f"rhoY_{species}")
    if not field or not field["available"]:
        return math.nan
    total = 0.0
    any_value = False
    for key in keys:
        value = field["values"].get(key, math.nan)
        if math.isfinite(value):
            total += value
            any_value = True
    return total if any_value else math.nan


def analyze_state(
    run_dir: Path,
    step: int,
    save_index: int,
    time_s: float,
    time_source: str,
    shock_half_width_m: float,
    gas_mass_floor: float,
) -> dict:
    fields = build_fields(run_dir, step, gas_mass_floor)
    shock_x, shock_method = raw.shock_location_from_pressure(fields["pressure"])
    zones = zone_masks(fields, shock_x, shock_half_width_m, gas_mass_floor)
    dx, dy, cell_area = gas_metrics.estimate_cell_area(fields)
    total_cells = max(len(fields.get("pressure", {}).get("values", {})), 1)

    row = {
        "save_index": save_index,
        "step": step,
        "time_s": time_s,
        "time_source": time_source,
        "shock_front_x": shock_x,
        "shock_front_method": shock_method,
        "shock_half_width_m": shock_half_width_m,
        "dx_m": dx,
        "dy_m": dy,
        "cell_area_m2": cell_area,
        "total_cell_count": total_cells,
    }
    for zone_name in ZONES:
        count = len(zones[zone_name])
        row[f"{zone_name}_cell_count"] = count
        row[f"{zone_name}_area_fraction"] = count / total_cells
        for species in SPECIES:
            raw_sum = sum_species(fields, species, zones[zone_name])
            row[f"rhoY_{species}_{zone_name}_sum"] = raw_sum
            row[f"rhoY_{species}_{zone_name}_volume_integral"] = (
                raw_sum * cell_area if math.isfinite(raw_sum) and math.isfinite(cell_area) else math.nan
            )
    return row


def available_rows(run_dir: Path, shock_half_width_m: float, gas_mass_floor: float) -> list[dict]:
    steps = raw.available_steps(run_dir)
    times = raw.infer_times(run_dir, steps)
    rows = []
    for save_index, step in enumerate(steps):
        time_s, time_source = times.get(step, (math.nan, "missing"))
        rows.append(analyze_state(run_dir, step, save_index, time_s, time_source, shock_half_width_m, gas_mass_floor))
    return rows


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


def series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([finite_float(row.get(key)) for row in rows], dtype=float)


def time_axis(rows: list[dict]) -> tuple[np.ndarray, str]:
    t = series(rows, "time_s")
    if np.isfinite(t).any():
        return t * 1.0e6, "Time [us]"
    return np.arange(len(rows), dtype=float), "Save index"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
        }
    )


def plot_species_group(rows: list[dict], out_path: Path, species_group: tuple[str, ...], title: str) -> None:
    if not rows:
        return
    x, xlabel = time_axis(rows)
    fig, axes = plt.subplots(len(species_group), 1, figsize=(8.0, 3.1 * len(species_group)), squeeze=False)
    for ax, species in zip(axes.ravel(), species_group):
        for zone in ZONES:
            y = series(rows, f"rhoY_{species}_{zone}_volume_integral")
            ax.plot(x, y, linewidth=1.9, label=ZONE_LABELS[zone], color=ZONE_COLORS[zone])
        ax.set_title(species)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\sum \rho Y_k \Delta x \Delta y$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def integrate_time(rows: list[dict], key: str) -> float:
    y = series(rows, key)
    t = series(rows, "time_s")
    if len(rows) >= 2 and np.isfinite(t).all():
        return float(np.trapz(np.nan_to_num(y, nan=0.0), t))
    return float(np.nansum(y))


def summarize(rows: list[dict]) -> dict:
    summary = {"saved_states": len(rows), "species": {}}
    if not rows:
        return summary

    for species in SPECIES:
        zone_integrals = {
            zone: integrate_time(rows, f"rhoY_{species}_{zone}_volume_integral")
            for zone in ("interface", "shock_front_gas")
        }
        denom = zone_integrals["interface"] + zone_integrals["shock_front_gas"]
        summary["species"][species] = {
            "interface_time_integral": zone_integrals["interface"],
            "shock_front_gas_time_integral": zone_integrals["shock_front_gas"],
            "interface_fraction_of_interface_plus_shock": (
                zone_integrals["interface"] / denom if denom > 0.0 else math.nan
            ),
            "shock_front_gas_fraction_of_interface_plus_shock": (
                zone_integrals["shock_front_gas"] / denom if denom > 0.0 else math.nan
            ),
        }

    for species in ("OH", "HO2"):
        candidates = []
        for zone in ZONES:
            y = series(rows, f"rhoY_{species}_{zone}_volume_integral")
            finite = np.isfinite(y)
            if finite.any():
                idxs = np.flatnonzero(finite)
                idx = idxs[int(np.nanargmax(y[finite]))]
                candidates.append((float(y[idx]), zone, idx))
        if candidates:
            value, zone, idx = max(candidates, key=lambda item: item[0])
            summary[f"peak_{species}"] = {
                "zone": zone,
                "value": value,
                "time_s": finite_float(rows[idx].get("time_s")),
                "time_us": finite_float(rows[idx].get("time_s")) * 1.0e6,
            }

    for species in ("CO2", "H2O"):
        final = rows[-1]
        candidates = []
        for zone in ZONES:
            candidates.append((finite_float(final.get(f"rhoY_{species}_{zone}_volume_integral")), zone))
        value, zone = max(candidates, key=lambda item: -math.inf if not math.isfinite(item[0]) else item[0])
        summary[f"final_{species}_dominant_zone"] = {"zone": zone, "value": value}

    for species in ("OH", "HO2"):
        item = summary["species"][species]
        iface = item["interface_fraction_of_interface_plus_shock"]
        shock = item["shock_front_gas_fraction_of_interface_plus_shock"]
        if math.isfinite(iface) and iface >= 0.6:
            label = "mostly interface-localized"
        elif math.isfinite(shock) and shock >= 0.6:
            label = "mostly shock-front-gas-localized"
        elif math.isfinite(iface):
            label = "mixed interface/shock-front gas"
        else:
            label = "insufficient signal"
        summary[f"{species}_late_radical_localization"] = label
    return summary


def plot_fraction_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    labels = []
    interface = []
    shock = []
    for species in SPECIES:
        i_val = integrate_time(rows, f"rhoY_{species}_interface_volume_integral")
        s_val = integrate_time(rows, f"rhoY_{species}_shock_front_gas_volume_integral")
        denom = i_val + s_val
        labels.append(species)
        interface.append(i_val / denom if denom > 0.0 else 0.0)
        shock.append(s_val / denom if denom > 0.0 else 0.0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(x, interface, label="Interface shell", color=ZONE_COLORS["interface"])
    ax.bar(x, shock, bottom=interface, label="Shock-front gas", color=ZONE_COLORS["shock_front_gas"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction of interface + shock-front budget")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Interface vs shock-front gas budget fractions")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_text_summary(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "Two-zone shock-droplet chemistry budget",
        f"saved states analyzed: {summary.get('saved_states', 0)}",
        "",
    ]
    if not rows:
        lines.extend(["No raw saved states were found. Expected D/ and/or p_all/ in the run directory.", ""])
        path.write_text("\n".join(lines))
        return

    for species in ("OH", "HO2"):
        peak = summary.get(f"peak_{species}", {})
        lines.append(
            f"peak {species}: zone={peak.get('zone')} time={peak.get('time_us')} us value={peak.get('value')}"
        )
    for species in ("CO2", "H2O"):
        final = summary.get(f"final_{species}_dominant_zone", {})
        lines.append(f"final {species} dominant zone: {final.get('zone')} value={final.get('value')}")
    lines.append("")
    lines.append("Interface vs shock-front gas time-integrated fractions:")
    for species in SPECIES:
        item = summary["species"][species]
        lines.append(
            f"  {species}: interface={item['interface_fraction_of_interface_plus_shock']} "
            f"shock_front_gas={item['shock_front_gas_fraction_of_interface_plus_shock']}"
        )
    lines.append("")
    lines.append(f"OH localization: {summary.get('OH_late_radical_localization')}")
    lines.append(f"HO2 localization: {summary.get('HO2_late_radical_localization')}")
    path.write_text("\n".join(lines) + "\n")


def analyze(run_dir: Path, out_dir: Path, shock_half_width_um: float, gas_mass_floor: float) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_half_width_m = shock_half_width_um * 1.0e-6
    rows = available_rows(run_dir, shock_half_width_m, gas_mass_floor)
    write_csv(out_dir / "zone_budget_timeseries.csv", rows)
    summary = summarize(rows)
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "shock_half_width_um": shock_half_width_um,
        "gas_mass_floor": gas_mass_floor,
        "zones": {
            "interface": "0.01 < alpha_liq < 0.99",
            "shock_front_gas": "gas_alpha > 0.9, alpha_liq < 0.1, abs(x - shock_front_x) <= shock_half_width",
            "gas_dominant": "gas_alpha > 0.9 and alpha_liq < 0.1",
        },
    }
    (out_dir / "zone_budget_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, allow_nan=True) + "\n"
    )
    write_text_summary(out_dir / "zone_budget_summary.txt", rows, summary)
    setup_style()
    plot_species_group(rows, out_dir / "OH_HO2_by_zone.png", ("OH", "HO2"), "Radical budgets by zone")
    plot_species_group(rows, out_dir / "CO2_H2O_by_zone.png", ("CO2", "H2O"), "Product budgets by zone")
    plot_species_group(rows, out_dir / "NC12H26_O2_by_zone.png", ("NC12H26", "O2"), "Fuel and oxidizer budgets by zone")
    plot_fraction_summary(rows, out_dir / "interface_vs_shock_fraction_by_species.png")
    return rows


def main() -> None:
    args = parse_args()
    rows = analyze(args.run_dir.resolve(), args.out_dir.resolve(), args.shock_half_width_um, args.gas_mass_floor)
    print(f"Analyzed {len(rows)} saved states from {args.run_dir}")
    print(f"Wrote zone budgets to {args.out_dir}")


if __name__ == "__main__":
    main()
