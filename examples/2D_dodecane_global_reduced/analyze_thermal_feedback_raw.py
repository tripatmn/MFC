#!/usr/bin/env python3
"""Offline thermal-feedback diagnostics for raw MFC D/ output.

This reads existing formatted raw output only. It does not require
post_process and does not assume a new simulation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cantera as ct
except Exception:  # pragma: no cover - reported at runtime
    ct = None


SPECIES = ("c12h26", "o2", "n2", "co2", "h2o")
FIELD_MAP = {
    "liquid_alpha_rho": "cons.1",
    "vapor_alpha_rho": "cons.2",
    "air_alpha_rho": "cons.3",
    "momentum_x": "cons.4",
    "momentum_y": "cons.5",
    "total_energy": "cons.6",
    "pressure": "prim.6",
    "liquid_alpha": "cons.7",
    "vapor_alpha": "cons.8",
    "air_alpha": "cons.9",
    "liquid_partial_energy": "cons.10",
    "vapor_partial_energy": "cons.11",
    "air_partial_energy": "cons.12",
    "rhoY_c12h26": "cons.13",
    "rhoY_o2": "cons.14",
    "rhoY_n2": "cons.15",
    "rhoY_co2": "cons.16",
    "rhoY_h2o": "cons.17",
    "Y_c12h26": "prim.13",
    "Y_o2": "prim.14",
    "Y_n2": "prim.15",
    "Y_co2": "prim.16",
    "Y_h2o": "prim.17",
}


@dataclass(frozen=True)
class RawCase:
    name: str
    root: Path
    mechanism: Path | None = None
    reactive: bool = False

    @property
    def ddir(self) -> Path:
        return self.root / "D" if (self.root / "D").is_dir() else self.root


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_rate = Path(__file__).resolve().parent / "dodecane_global_1step_rate1000.yaml"
    parser = argparse.ArgumentParser(
        description="Analyze raw D/ thermal-feedback diagnostics for clean dodecane pilots."
    )
    parser.add_argument(
        "--nonreact",
        type=Path,
        default=repo / "runs" / "dodecane_d2_clean_evap_pilot_v2",
        help="Nonreacting pilot root or D/ directory.",
    )
    parser.add_argument(
        "--rate1000",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Rebuilt rate1000 pilot root or D/ directory.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline burning root or D/ directory, if preserved.",
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=default_rate,
        help="Cantera mechanism used for reactive reconstruction.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--all-steps", action="store_true", help="Print all per-step rows to CSV/JSON.")
    return parser.parse_args()


def split_field(field: str) -> tuple[str, int]:
    prefix, idx = field.split(".")
    return prefix, int(idx)


def file_index(ddir: Path, prefix: str, idx: int) -> dict[int, list[Path]]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.{idx}\.\d+\.(\d+)\.dat$")
    out: dict[int, list[Path]] = {}
    for path in ddir.glob(f"{prefix}.{idx}.*.*.dat"):
        match = pattern.match(path.name)
        if match:
            out.setdefault(int(match.group(1)), []).append(path)
    return {step: sorted(paths) for step, paths in out.items()}


def available_fields(ddir: Path, prefix: str) -> dict[int, list[int]]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.\d+\.(\d+)\.dat$")
    out: dict[int, set[int]] = {}
    for path in ddir.glob(f"{prefix}.*.*.*.dat"):
        match = pattern.match(path.name)
        if match:
            idx = int(match.group(1))
            step = int(match.group(2))
            out.setdefault(idx, set()).add(step)
    return {idx: sorted(steps) for idx, steps in sorted(out.items())}


def common_steps(ddir: Path, fields: Iterable[str]) -> list[int]:
    step_sets = []
    for field in fields:
        prefix, idx = split_field(field)
        steps = set(file_index(ddir, prefix, idx))
        if not steps:
            return []
        step_sets.append(steps)
    return sorted(set.intersection(*step_sets))


def read_field(ddir: Path, field: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix, idx = split_field(field)
    paths = file_index(ddir, prefix, idx).get(step, [])
    if not paths:
        raise FileNotFoundError(f"{ddir}: missing {field} at step {step}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for path in paths:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
        vals.append(arr[:, 2])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(vals)


def cell_area(ddir: Path) -> float:
    steps = common_steps(ddir, ["cons.1"])
    if not steps:
        return 1.0
    x, y, _ = read_field(ddir, "cons.1", steps[0])
    ux = np.unique(np.round(x, 15))
    uy = np.unique(np.round(y, 15))
    dx = float(np.min(np.diff(ux))) if len(ux) > 1 else 1.0
    dy = float(np.min(np.diff(uy))) if len(uy) > 1 else 1.0
    return abs(dx * dy)


def mass_fractions(rho_g: np.ndarray, rho_y: list[np.ndarray]) -> np.ndarray:
    y = np.zeros((len(SPECIES), rho_g.size), dtype=float)
    valid = np.isfinite(rho_g) & (rho_g > 1e-300)
    for i, rho_y_i in enumerate(rho_y):
        raw = np.zeros_like(rho_g)
        raw[valid] = rho_y_i[valid] / rho_g[valid]
        raw = np.where(np.isfinite(raw), raw, 0.0)
        y[i] = np.clip(raw, 0.0, 1.0)
    ysum = np.sum(y, axis=0)
    renorm = ysum > 1.0
    y[:, renorm] /= ysum[renorm]
    return y


def temperature_from_pressure(gas: "ct.Solution", pressure: np.ndarray, rho_g: np.ndarray, y: np.ndarray) -> np.ndarray:
    mw = np.asarray(gas.molecular_weights, dtype=float)
    inv_w = np.sum(y / mw[:, None], axis=0)
    mix_w = np.divide(1.0, inv_w, out=np.full_like(inv_w, np.nan), where=inv_w > 0.0)
    return np.divide(
        pressure * mix_w,
        ct.gas_constant * rho_g,
        out=np.full_like(pressure, np.nan),
        where=(rho_g > 1e-300) & np.isfinite(mix_w),
    )


def heat_release_for_cells(
    gas: "ct.Solution",
    rho_g: np.ndarray,
    temp_for_rates: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qdot = np.zeros_like(rho_g)
    omega_mass = np.zeros((len(SPECIES), rho_g.size), dtype=float)
    mw = np.asarray(gas.molecular_weights, dtype=float)
    active_idx = np.flatnonzero(active)
    for idx in active_idx:
        try:
            gas.TDY = float(temp_for_rates[idx]), float(rho_g[idx]), y[:, idx]
            wdot = gas.net_production_rates
            h = gas.partial_molar_enthalpies
        except Exception:
            continue
        qdot[idx] = -float(np.dot(h, wdot))
        omega_mass[:, idx] = wdot * mw
    return qdot, omega_mass


def species_endpoint_sums(ddir: Path) -> dict[str, float | int] | None:
    steps = common_steps(ddir, [f"cons.{idx}" for idx in range(13, 18)])
    if not steps:
        return None
    out: dict[str, float | int] = {"species_first_step": steps[0], "species_last_step": steps[-1]}
    for name, idx in zip(SPECIES, range(13, 18)):
        first = read_field(ddir, f"cons.{idx}", steps[0])[2]
        last = read_field(ddir, f"cons.{idx}", steps[-1])[2]
        out[f"{name}_first_step0_sum"] = float(np.sum(first))
        out[f"{name}_last_sum"] = float(np.sum(last))
        out[f"{name}_delta_from_step0"] = float(np.sum(last) - np.sum(first))
    return out


def summarize_array(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    data = values if mask is None else values[mask]
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def analyze_reactive_case(case: RawCase, gas: "ct.Solution") -> tuple[dict, list[dict]]:
    ddir = case.ddir
    d_a = cell_area(ddir)
    required = [
        "cons.1",
        "cons.2",
        "cons.3",
        "cons.7",
        "cons.8",
        "cons.9",
        "prim.6",
        "cons.13",
        "cons.14",
        "cons.15",
        "cons.16",
        "cons.17",
    ]
    steps = common_steps(ddir, required)
    rows = []
    for step in steps:
        x, ycoord, ar_liq = read_field(ddir, "cons.1", step)
        _, _, ar_vap = read_field(ddir, "cons.2", step)
        _, _, ar_air = read_field(ddir, "cons.3", step)
        _, _, alpha_liq = read_field(ddir, "cons.7", step)
        _, _, alpha_vap = read_field(ddir, "cons.8", step)
        _, _, alpha_air = read_field(ddir, "cons.9", step)
        _, _, pressure = read_field(ddir, "prim.6", step)
        rho_y = [read_field(ddir, f"cons.{idx}", step)[2] for idx in range(13, 18)]

        rho_g = ar_vap + ar_air
        alpha_g = alpha_vap + alpha_air
        yk = mass_fractions(rho_g, rho_y)
        temp = temperature_from_pressure(gas, pressure, rho_g, yk)

        interface = (alpha_liq > 0.01) & (alpha_liq < 0.99)
        interface_loose = (alpha_liq > 1e-4) & (alpha_liq < 0.9999)
        gas_mask = (alpha_g > 0.01) & np.isfinite(temp) & (rho_g > 1e-300)
        gas_interface = interface & gas_mask
        gas_interface_loose = interface_loose & gas_mask
        reactive = (
            (yk[0] > 1e-14)
            & (yk[1] > 1e-14)
            & np.isfinite(temp)
            & (temp > 0.0)
            & (rho_g > 1e-300)
            & (alpha_g > 0.01)
        )
        temp_for_rates = np.clip(temp, 250.0, 3000.0)
        qdot, omega_mass = heat_release_for_cells(gas, rho_g, temp_for_rates, yk, reactive)
        qdot_pos = np.maximum(qdot, 0.0)

        fuel_o2_overlap = yk[0] * yk[1]
        co2_sum = float(np.sum(rho_y[3]))
        h2o_sum = float(np.sum(rho_y[4]))
        qdot_global = float(np.sum(qdot) * d_a)
        qdot_pos_global = float(np.sum(qdot_pos) * d_a)
        qdot_near = float(np.sum(qdot[interface]) * d_a)
        qdot_pos_near = float(np.sum(qdot_pos[interface]) * d_a)
        qdot_gas_near = float(np.sum(qdot[gas_interface]) * d_a)
        qdot_pos_gas_near = float(np.sum(qdot_pos[gas_interface]) * d_a)
        qdot_loose = float(np.sum(qdot[interface_loose]) * d_a)
        qdot_pos_gas_loose = float(np.sum(qdot_pos[gas_interface_loose]) * d_a)

        rows.append(
            {
                "case": case.name,
                "step": step,
                "n_cells": int(ar_liq.size),
                "n_interface": int(np.count_nonzero(interface)),
                "n_interface_loose": int(np.count_nonzero(interface_loose)),
                "n_gas_cells": int(np.count_nonzero(gas_mask)),
                "n_gas_interface": int(np.count_nonzero(gas_interface)),
                "n_gas_interface_loose": int(np.count_nonzero(gas_interface_loose)),
                "n_reactive": int(np.count_nonzero(reactive)),
                "liquid_alpha_rho_integral": float(np.sum(ar_liq) * d_a),
                "vapor_alpha_rho_integral": float(np.sum(ar_vap) * d_a),
                "liquid_alpha_integral": float(np.sum(alpha_liq) * d_a),
                "vapor_alpha_integral": float(np.sum(alpha_vap) * d_a),
                "rho_g_integral": float(np.sum(rho_g) * d_a),
                "fuel_sum": float(np.sum(rho_y[0])),
                "o2_sum": float(np.sum(rho_y[1])),
                "n2_sum": float(np.sum(rho_y[2])),
                "co2_sum": co2_sum,
                "h2o_sum": h2o_sum,
                "fuel_o2_overlap": float(np.sum(fuel_o2_overlap) * d_a),
                "fuel_o2_overlap_interface": float(np.sum(fuel_o2_overlap[interface]) * d_a),
                "fuel_o2_overlap_interface_loose": float(np.sum(fuel_o2_overlap[interface_loose]) * d_a),
                "qdot_global": qdot_global,
                "qdot_positive_global": qdot_pos_global,
                "qdot_interface": qdot_near,
                "qdot_positive_interface": qdot_pos_near,
                "qdot_gas_interface": qdot_gas_near,
                "qdot_positive_gas_interface": qdot_pos_gas_near,
                "qdot_interface_loose": qdot_loose,
                "qdot_positive_gas_interface_loose": qdot_pos_gas_loose,
                "qdot_positive_interface_fraction": qdot_pos_near / qdot_pos_global if qdot_pos_global > 0 else None,
                "qdot_positive_gas_interface_fraction": qdot_pos_gas_near / qdot_pos_global if qdot_pos_global > 0 else None,
                "qdot_positive_gas_interface_loose_fraction": qdot_pos_gas_loose / qdot_pos_global if qdot_pos_global > 0 else None,
                "omega_c12h26_sum": float(np.sum(omega_mass[0]) * d_a),
                "omega_o2_sum": float(np.sum(omega_mass[1]) * d_a),
                "omega_n2_sum": float(np.sum(omega_mass[2]) * d_a),
                "omega_co2_sum": float(np.sum(omega_mass[3]) * d_a),
                "omega_h2o_sum": float(np.sum(omega_mass[4]) * d_a),
                "T_global_min": summarize_array(temp)["min"],
                "T_global_max": summarize_array(temp)["max"],
                "T_global_mean": summarize_array(temp)["mean"],
                "T_gas_min": summarize_array(temp, gas_mask)["min"],
                "T_gas_max": summarize_array(temp, gas_mask)["max"],
                "T_gas_mean": summarize_array(temp, gas_mask)["mean"],
                "T_interface_min": summarize_array(temp, interface)["min"],
                "T_interface_max": summarize_array(temp, interface)["max"],
                "T_interface_mean": summarize_array(temp, interface)["mean"],
                "T_gas_interface_min": summarize_array(temp, gas_interface)["min"],
                "T_gas_interface_max": summarize_array(temp, gas_interface)["max"],
                "T_gas_interface_mean": summarize_array(temp, gas_interface)["mean"],
                "T_interface_loose_min": summarize_array(temp, interface_loose)["min"],
                "T_interface_loose_max": summarize_array(temp, interface_loose)["max"],
                "T_interface_loose_mean": summarize_array(temp, interface_loose)["mean"],
                "T_gas_interface_loose_min": summarize_array(temp, gas_interface_loose)["min"],
                "T_gas_interface_loose_max": summarize_array(temp, gas_interface_loose)["max"],
                "T_gas_interface_loose_mean": summarize_array(temp, gas_interface_loose)["mean"],
                "qdot_max": float(np.nanmax(qdot)) if qdot.size else None,
                "qdot_interface_max": float(np.nanmax(qdot[interface])) if np.any(interface) else None,
                "qdot_gas_interface_max": float(np.nanmax(qdot[gas_interface])) if np.any(gas_interface) else None,
            }
        )

    summary = summarize_case(case, rows)
    species_sums = species_endpoint_sums(ddir)
    if species_sums:
        summary.update(species_sums)
    summary["temperature_reconstruction"] = (
        "succeeded: ideal-gas chemistry temperature from prim.6 pressure, "
        "rho_g=sum(cons.2,cons.3), and Cantera mixture molecular weight"
    )
    summary["full_internal_energy_temperature"] = (
        "not attempted: current raw fields include model-3 total and partial energies, "
        "but an unambiguous gas-only internal energy consistent with Cantera reference "
        "states is not directly available without mirroring MFC internals"
    )
    return summary, rows


def analyze_nonreactive_case(case: RawCase) -> tuple[dict, list[dict]]:
    ddir = case.ddir
    d_a = cell_area(ddir)
    required = ["cons.1", "cons.2", "cons.3", "cons.7", "cons.8", "prim.6"]
    steps = common_steps(ddir, required)
    rows = []
    r_air = (1.4 - 1.0) * 739.0
    for step in steps:
        _, _, ar_liq = read_field(ddir, "cons.1", step)
        _, _, ar_vap = read_field(ddir, "cons.2", step)
        _, _, ar_air = read_field(ddir, "cons.3", step)
        _, _, alpha_liq = read_field(ddir, "cons.7", step)
        _, _, alpha_vap = read_field(ddir, "cons.8", step)
        _, _, pressure = read_field(ddir, "prim.6", step)
        rho_g = ar_vap + ar_air
        t_proxy = np.divide(
            pressure,
            r_air * rho_g,
            out=np.full_like(pressure, np.nan),
            where=rho_g > 1e-300,
        )
        interface = (alpha_liq > 0.01) & (alpha_liq < 0.99)
        rows.append(
            {
                "case": case.name,
                "step": step,
                "n_cells": int(ar_liq.size),
                "n_interface": int(np.count_nonzero(interface)),
                "liquid_alpha_rho_integral": float(np.sum(ar_liq) * d_a),
                "vapor_alpha_rho_integral": float(np.sum(ar_vap) * d_a),
                "liquid_alpha_integral": float(np.sum(alpha_liq) * d_a),
                "vapor_alpha_integral": float(np.sum(alpha_vap) * d_a),
                "rho_g_integral": float(np.sum(rho_g) * d_a),
                "T_proxy_global_min": summarize_array(t_proxy)["min"],
                "T_proxy_global_max": summarize_array(t_proxy)["max"],
                "T_proxy_global_mean": summarize_array(t_proxy)["mean"],
                "T_proxy_interface_min": summarize_array(t_proxy, interface)["min"],
                "T_proxy_interface_max": summarize_array(t_proxy, interface)["max"],
                "T_proxy_interface_mean": summarize_array(t_proxy, interface)["mean"],
            }
        )
    summary = summarize_case(case, rows)
    summary["temperature_reconstruction"] = (
        "not applicable: nonreacting raw output has no species fields; reported "
        "temperature is an air-constant pressure/rho_g proxy only"
    )
    return summary, rows


def summarize_case(case: RawCase, rows: list[dict]) -> dict:
    cons = available_fields(case.ddir, "cons")
    prim = available_fields(case.ddir, "prim")
    summary = {
        "case": case.name,
        "root": str(case.root),
        "D": str(case.ddir),
        "exists": case.ddir.is_dir(),
        "n_steps": len(rows),
        "cons_indices": sorted(cons),
        "prim_indices": sorted(prim),
        "field_map": FIELD_MAP,
    }
    if not rows:
        return summary
    first = rows[0]
    last = rows[-1]
    summary.update(
        {
            "first_step": first["step"],
            "last_step": last["step"],
            "liquid_alpha_rho_first": first.get("liquid_alpha_rho_integral"),
            "liquid_alpha_rho_last": last.get("liquid_alpha_rho_integral"),
            "vapor_alpha_rho_first": first.get("vapor_alpha_rho_integral"),
            "vapor_alpha_rho_last": last.get("vapor_alpha_rho_integral"),
            "liquid_alpha_first": first.get("liquid_alpha_integral"),
            "liquid_alpha_last": last.get("liquid_alpha_integral"),
            "vapor_alpha_first": first.get("vapor_alpha_integral"),
            "vapor_alpha_last": last.get("vapor_alpha_integral"),
        }
    )
    if "qdot_positive_global" in last:
        max_qdot_row = max(rows, key=lambda r: abs(r.get("qdot_positive_global") or 0.0))
        summary.update(
            {
                "fuel_first": first["fuel_sum"],
                "fuel_last": last["fuel_sum"],
                "o2_first": first["o2_sum"],
                "o2_last": last["o2_sum"],
                "co2_first": first["co2_sum"],
                "co2_last": last["co2_sum"],
                "h2o_first": first["h2o_sum"],
                "h2o_last": last["h2o_sum"],
                "T_interface_initial": {
                    "min": first["T_interface_min"],
                    "max": first["T_interface_max"],
                    "mean": first["T_interface_mean"],
                },
                "T_interface_final": {
                    "min": last["T_interface_min"],
                    "max": last["T_interface_max"],
                    "mean": last["T_interface_mean"],
                },
                "T_gas_interface_initial": {
                    "min": first["T_gas_interface_min"],
                    "max": first["T_gas_interface_max"],
                    "mean": first["T_gas_interface_mean"],
                },
                "T_gas_interface_final": {
                    "min": last["T_gas_interface_min"],
                    "max": last["T_gas_interface_max"],
                    "mean": last["T_gas_interface_mean"],
                },
                "T_global_final": {
                    "min": last["T_global_min"],
                    "max": last["T_global_max"],
                    "mean": last["T_global_mean"],
                },
                "T_gas_final": {
                    "min": last["T_gas_min"],
                    "max": last["T_gas_max"],
                    "mean": last["T_gas_mean"],
                },
                "qdot_positive_global_final": last["qdot_positive_global"],
                "qdot_positive_interface_final": last["qdot_positive_interface"],
                "qdot_positive_interface_fraction_final": last["qdot_positive_interface_fraction"],
                "qdot_positive_gas_interface_final": last["qdot_positive_gas_interface"],
                "qdot_positive_gas_interface_fraction_final": last["qdot_positive_gas_interface_fraction"],
                "qdot_positive_gas_interface_loose_fraction_final": last["qdot_positive_gas_interface_loose_fraction"],
                "max_qdot_positive_step": max_qdot_row["step"],
                "max_qdot_positive_global": max_qdot_row["qdot_positive_global"],
                "max_qdot_positive_interface": max_qdot_row["qdot_positive_interface"],
                "max_qdot_positive_interface_fraction": max_qdot_row["qdot_positive_interface_fraction"],
                "max_qdot_positive_gas_interface": max_qdot_row["qdot_positive_gas_interface"],
                "max_qdot_positive_gas_interface_fraction": max_qdot_row["qdot_positive_gas_interface_fraction"],
                "fuel_o2_overlap_final": last["fuel_o2_overlap"],
                "fuel_o2_overlap_interface_final": last["fuel_o2_overlap_interface"],
                "omega_final": {
                    name: last[f"omega_{name}_sum"] for name in SPECIES
                },
            }
        )
    return summary


def write_outputs(out_json: Path | None, out_csv: Path | None, payload: dict, rows: list[dict]) -> None:
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if out_csv and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def print_summary(payload: dict) -> None:
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))
    print("\nDiagnosis:")
    rate = payload["summaries"].get("rate1000")
    non = payload["summaries"].get("nonreact")
    if not rate or rate.get("n_steps", 0) == 0:
        print("- Rebuilt rate1000 raw output was not analyzable.")
        return
    print("- T_chem reconstruction: " + str(rate.get("temperature_reconstruction")))
    print("- Full internal-energy Cantera inversion: " + str(rate.get("full_internal_energy_temperature")))
    qfrac = rate.get("qdot_positive_interface_fraction_final")
    if qfrac is not None:
        print(f"- Final positive heat-release overlap with strict interface band: {qfrac:.6g}")
    print(
        "- Final gas-interface T_chem mean: "
        f"{rate.get('T_gas_interface_final', {}).get('mean')} K; "
        "gas-filtered global final mean: "
        f"{rate.get('T_gas_final', {}).get('mean')} K"
    )
    def delta(name: str) -> float | None:
        key = f"{name}_delta_from_step0"
        return rate.get(key)
    print(
        "- Final species signs: "
        f"fuel {delta('c12h26'):+.6e}, "
        f"O2 {delta('o2'):+.6e}, "
        f"CO2 {delta('co2'):+.6e}, "
        f"H2O {delta('h2o'):+.6e}"
    )
    if non:
        print(
            "- Phase budget comparison, final rate1000 minus nonreacting: "
            f"liquid alpha_rho "
            f"{rate.get('liquid_alpha_rho_last') - non.get('liquid_alpha_rho_last'):+.6e}, "
            f"vapor alpha_rho "
            f"{rate.get('vapor_alpha_rho_last') - non.get('vapor_alpha_rho_last'):+.6e}"
        )


def main() -> int:
    args = parse_args()
    cases = [
        RawCase("nonreact", args.nonreact, reactive=False),
        RawCase("rate1000", args.rate1000, mechanism=args.mechanism, reactive=True),
    ]
    if args.baseline is not None:
        cases.insert(1, RawCase("baseline", args.baseline, mechanism=args.mechanism, reactive=True))

    gas = None
    if any(case.reactive for case in cases):
        if ct is None:
            raise SystemExit("Cantera is not importable; run with the project venv or install cantera.")
        gas = ct.Solution(str(args.mechanism))
        missing = [name for name in SPECIES if name not in gas.species_names]
        if missing:
            raise SystemExit(f"Mechanism missing required species: {missing}")
        gas = ct.Solution(str(args.mechanism), gas.name)

    summaries = {}
    all_rows: list[dict] = []
    limitations = []
    for case in cases:
        if not case.ddir.is_dir():
            limitations.append(f"{case.name}: raw D directory not found at {case.ddir}")
            continue
        if case.reactive:
            summary, rows = analyze_reactive_case(case, gas)
        else:
            summary, rows = analyze_nonreactive_case(case)
        summaries[case.name] = summary
        all_rows.extend(rows if args.all_steps else [rows[0], rows[-1]] if rows else [])

    if args.baseline is None:
        limitations.append("baseline burning raw D/ was not provided/found; comparison uses nonreacting vs rebuilt rate1000 only")

    payload = {
        "summaries": summaries,
        "limitations": limitations,
        "field_index_notes": {
            "model_eqns": 3,
            "chem_gas_fluid_ids": [2, 3],
            "rho_g": "cons.2 + cons.3",
            "interface_mask": "0.01 < cons.7(liquid alpha) < 0.99",
            "gas_interface_mask": "interface_mask and cons.8+cons.9 > 0.01",
            "temperature_method": "T = p * W_mix / (R_universal * rho_g), with Y_k=rhoY_k/rho_g clipped like MFC chemistry source",
            "heat_release_proxy": "-sum_k(h_k(T) * omega_k), using T clipped to MFC chemistry limits 250..3000 K, integrated over 2D cell area",
        },
    }
    write_outputs(args.out_json, args.out_csv, payload, all_rows)
    print_summary(payload)
    if limitations:
        print("\nLimitations:")
        for item in limitations:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
