#!/usr/bin/env python3
"""Lightweight post-run inventory and scalar summaries for validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from common import SK54_CTPHASE, SK54_CTFILE, load_solution


DAT_RE = re.compile(r"^(?P<var>.+)\.(?P<rank>\d+)\.(?P<step>\d+)\.dat$")
PROBE_RE = re.compile(r"^probe(?P<probe>\d+)_prim\.dat$")

CASE_CHECKS = {
    "V1": [
        "final reconstructed T vs Cantera at 100 us within max(2 K, 1%)",
        "final H2/O2/OH/H2O mass fractions vs Cantera within max(1e-5, 2%)",
        "max_abs(sum(Y)-1) <= 1e-10 and min(Y) >= -1e-12",
    ],
    "V2": [
        "ignition metric vs Cantera within max(2 us, 10%)",
        "T(t) and pressure histories within 2% before and through ignition",
        "relative abs(sum(rhoY)-rho_g_stored) <= 1e-10",
        "relative total-E drift <= 1e-10 and relative abs(p2-p3) <= 1e-10",
    ],
    "V3": [
        "delta(alpha_rho_2) >= 0",
        "-delta(alpha_rho_1) = delta(alpha_rho_2) = delta(rhoY_NC12H26)",
        "non-fuel species integrated masses unchanged to <= 1e-13 relative",
    ],
    "V4": [
        "paired with V3: V4 consumes freshly introduced fuel in first post-step state",
        "V4 product/radical integral exceeds V3 by > 1e-12",
        "relative abs(sum(rhoY)-rho_g_stored) <= 1e-10",
    ],
    "V5": [
        "L_inf abs(sum(rhoY)-((alpha*rho)_2+(alpha*rho)_3)) <= 1e-10",
        "integrated closure error <= 1e-12 relative to integrated gas mass",
        "each globally integrated species conserved to <= 1e-12 relative",
    ],
}

V1_SPECIES = ("H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2")
V1_MOLECULAR_WEIGHTS = (2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, 34.014, 39.95, 28.014)
V1_COMPARE_SPECIES = ("H2", "O2", "OH", "H2O")
GAS_CONSTANT = 8314.46261815324
V2_COMPARE_SPECIES = ("NC12H26", "O2", "OH", "H2O")
V2_KEY_SPECIES = ("NC12H26", "O2", "OH")
V2_EOS = {
    2: {"gamma": 40.00000000000014, "pi_inf": 0.0, "qv": -234490.0},
    3: {"gamma": 2.5000000000000004, "pi_inf": 0.0, "qv": 0.0},
}


def parse_name(path: Path) -> tuple[str, int, int] | None:
    match = DAT_RE.match(path.name)
    if not match:
        return None
    return match.group("var"), int(match.group("rank")), int(match.group("step"))


def dat_files(run_dir: Path) -> list[Path]:
    roots = [run_dir / "D", run_dir]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.glob("*.dat")))
    return sorted(set(files))


def probe_files(run_dir: Path) -> list[Path]:
    roots = [run_dir / "D", run_dir]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(path for path in sorted(root.glob("probe*_prim.dat")) if PROBE_RE.match(path.name))
    return sorted(set(files))


def read_last_column(path: Path) -> dict[str, float]:
    count = 0
    total = 0.0
    min_value = math.inf
    max_value = -math.inf
    finite = True
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                value = float(parts[-1])
            except (ValueError, IndexError):
                continue
            count += 1
            total += value
            min_value = min(min_value, value)
            max_value = max(max_value, value)
            finite = finite and math.isfinite(value)
    if count == 0:
        return {"count": 0, "sum": math.nan, "min": math.nan, "max": math.nan, "finite": False}
    return {"count": count, "sum": total, "min": min_value, "max": max_value, "finite": finite}


def read_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                values.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    return values


def load_var_values(run_dir: Path, var: str, step: int) -> list[float]:
    pattern = f"{var}.*.{step:06d}.dat"
    values: list[float] = []
    for root in (run_dir / "D", run_dir):
        if not root.is_dir():
            continue
        for path in sorted(root.glob(pattern)):
            values.extend(read_values(path))
    return values


def steps_for_var(run_dir: Path, var: str) -> list[int]:
    steps = set()
    for path in dat_files(run_dir):
        parsed = parse_name(path)
        if parsed and parsed[0] == var:
            steps.add(parsed[2])
    return sorted(steps)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def reconstruct_temperature(rho: float, pressure: float, ys: list[float], molecular_weights: list[float] | tuple[float, ...]) -> float:
    inv_mw = sum(y / mw for y, mw in zip(ys, molecular_weights))
    if rho <= 0.0 or inv_mw <= 0.0:
        return math.nan
    mix_molecular_weight = 1.0 / inv_mw
    return pressure * mix_molecular_weight / (rho * GAS_CONSTANT)


def analyze_v1(run_dir: Path) -> tuple[dict, list[dict]]:
    reference_path = Path(__file__).resolve().parent / "V1_cantera_reference" / "metrics.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing V1 Cantera reference: {reference_path}")
    with reference_path.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)

    steps = steps_for_var(run_dir, "prim.1")
    if not steps:
        raise FileNotFoundError(f"No prim.1 fields found under {run_dir / 'D'}")
    final_step = steps[-1]

    rho = load_var_values(run_dir, "prim.1", final_step)
    pressure = load_var_values(run_dir, "prim.3", final_step)
    species_by_name = {
        name: load_var_values(run_dir, f"prim.{5 + idx}", final_step)
        for idx, name in enumerate(V1_SPECIES)
    }
    field_lengths = {len(rho), len(pressure), *(len(values) for values in species_by_name.values())}
    if len(field_lengths) != 1 or 0 in field_lengths:
        raise ValueError(f"Incomplete or mismatched V1 final fields at output step {final_step}: lengths={sorted(field_lengths)}")

    final_temperatures = []
    for cell_idx in range(len(rho)):
        ys = [species_by_name[name][cell_idx] for name in V1_SPECIES]
        final_temperatures.append(reconstruct_temperature(rho[cell_idx], pressure[cell_idx], ys, V1_MOLECULAR_WEIGHTS))

    max_abs_sum_y_err = 0.0
    min_y = math.inf
    for step in steps:
        step_species = [load_var_values(run_dir, f"prim.{5 + idx}", step) for idx in range(len(V1_SPECIES))]
        if any(len(values) != len(step_species[0]) for values in step_species):
            raise ValueError(f"Incomplete V1 species fields at output step {step}")
        for cell_values in zip(*step_species):
            sum_y = sum(cell_values)
            max_abs_sum_y_err = max(max_abs_sum_y_err, abs(sum_y - 1.0))
            min_y = min(min_y, min(cell_values))

    mfc_final = {
        "T_K": mean(final_temperatures),
        "rho_kg_m3": mean(rho),
        "pressure_Pa": mean(pressure),
        "species": {name: mean(species_by_name[name]) for name in V1_COMPARE_SPECIES},
    }

    checks = []
    ref_t = reference["T_final_K"]
    temp_tol = max(2.0, 0.01 * abs(ref_t))
    temp_err = abs(mfc_final["T_K"] - ref_t)
    checks.append({"criterion": "final_T", "mfc": mfc_final["T_K"], "reference": ref_t, "tolerance": temp_tol, "error": temp_err, "pass": temp_err <= temp_tol})

    for name in V1_COMPARE_SPECIES:
        ref_y = reference["final_mass_fractions"][name]
        mfc_y = mfc_final["species"][name]
        tol = max(1.0e-5, 0.02 * abs(ref_y))
        err = abs(mfc_y - ref_y)
        checks.append({"criterion": f"final_Y_{name}", "mfc": mfc_y, "reference": ref_y, "tolerance": tol, "error": err, "pass": err <= tol})

    checks.append(
        {
            "criterion": "max_abs_sumY_minus_1",
            "mfc": max_abs_sum_y_err,
            "reference": 0.0,
            "tolerance": 1.0e-10,
            "error": max_abs_sum_y_err,
            "pass": max_abs_sum_y_err <= 1.0e-10,
        }
    )
    checks.append(
        {
            "criterion": "min_Y",
            "mfc": min_y,
            "reference": 0.0,
            "tolerance": -1.0e-12,
            "error": min_y,
            "pass": min_y >= -1.0e-12,
        }
    )

    summary = {
        "case": "V1",
        "final_output_step": final_step,
        "reference": reference,
        "mfc_final": mfc_final,
        "max_abs_sumY_minus_1": max_abs_sum_y_err,
        "min_Y": min_y,
        "pass": all(check["pass"] for check in checks),
    }
    return summary, checks


def read_scalar_from_input(run_dir: Path, name: str) -> float | None:
    for input_name in ("simulation.inp", "post_process.inp"):
        path = run_dir / input_name
        if not path.is_file():
            continue
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*([^!\s]+)", re.I)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
    return None


def interpolate(x: list[float], y: list[float], xq: float) -> float:
    if not x:
        return math.nan
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]
    for idx in range(1, len(x)):
        if x[idx] >= xq:
            frac = (xq - x[idx - 1]) / (x[idx] - x[idx - 1])
            return y[idx - 1] + frac * (y[idx] - y[idx - 1])
    return y[-1]


def crossing_time(times: list[float], values: list[float], threshold: float) -> float | None:
    for idx in range(1, len(times)):
        y0 = values[idx - 1] - threshold
        y1 = values[idx] - threshold
        if y0 == 0.0:
            return times[idx - 1]
        if y0 * y1 <= 0.0 and y1 != y0:
            frac = -y0 / (y1 - y0)
            return times[idx - 1] + frac * (times[idx] - times[idx - 1])
    return None


def load_reference_history(case: str) -> tuple[list[dict[str, float]], dict]:
    base = Path(__file__).resolve().parent / f"{case}_cantera_reference"
    history_path = base / "history.csv"
    metrics_path = base / "metrics.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing {case} Cantera history: {history_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {case} Cantera metrics: {metrics_path}")
    with history_path.open("r", encoding="utf-8") as handle:
        history = [{name: float(value) if value else math.nan for name, value in row.items()} for row in csv.DictReader(handle)]
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return history, metrics


def species_mole_fraction(ys: dict[str, float], molecular_weights: dict[str, float], species_name: str) -> float:
    denom = sum(value / molecular_weights[name] for name, value in ys.items() if name in molecular_weights)
    if denom <= 0.0:
        return math.nan
    return ys[species_name] / molecular_weights[species_name] / denom


def status_row(metric: str, mfc, reference, error, tolerance, status: str, note: str = "") -> dict:
    return {
        "metric": metric,
        "mfc": mfc,
        "reference": reference,
        "error": error,
        "tolerance": tolerance,
        "status": status,
        "note": note,
    }


def pass_fail(error: float, tolerance: float) -> str:
    if not math.isfinite(error):
        return "BLOCKED"
    return "PASS" if error <= tolerance else "FAIL"


def format_metric_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def analyze_v2(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    gas = load_solution(SK54_CTFILE, SK54_CTPHASE)
    species_names = tuple(gas.species_names)
    molecular_weights = {name: float(gas.molecular_weights[idx]) for idx, name in enumerate(species_names)}
    species_field = {name: 12 + idx for idx, name in enumerate(species_names)}
    history, reference = load_reference_history("V2")
    ref_times = [row["time_s"] for row in history]

    dt = read_scalar_from_input(run_dir, "dt")
    if dt is None:
        raise ValueError("Could not read dt from V2 simulation.inp")

    steps = steps_for_var(run_dir, "cons.1")
    if not steps:
        raise FileNotFoundError(f"No cons.1 fields found under {run_dir / 'D'}")

    required_fields = {
        "alpha_rho_1": "cons.1",
        "alpha_rho_2": "cons.2",
        "alpha_rho_3": "cons.3",
        "total_E": "cons.5",
        "alpha_1": "cons.6",
        "alpha_2": "cons.7",
        "alpha_3": "cons.8",
        "int_en_2": "cons.10",
        "int_en_3": "cons.11",
    }
    missing = []
    for step in (steps[0], steps[-1]):
        for var in required_fields.values():
            if not load_var_values(run_dir, var, step):
                missing.append(f"{var}@{step}")
        for field_id in species_field.values():
            if not load_var_values(run_dir, f"cons.{field_id}", step):
                missing.append(f"cons.{field_id}@{step}")
    if missing:
        raise FileNotFoundError(f"V2 output is missing required normal fields: {', '.join(missing[:12])}")

    trajectory = []
    max_species_closure = 0.0
    min_y = math.inf
    min_rho_y = math.inf
    max_energy_drift = 0.0
    max_p23_mismatch = 0.0
    e0 = None

    for step in steps:
        arho2 = load_var_values(run_dir, "cons.2", step)
        arho3 = load_var_values(run_dir, "cons.3", step)
        alpha2 = load_var_values(run_dir, "cons.7", step)
        alpha3 = load_var_values(run_dir, "cons.8", step)
        total_e = load_var_values(run_dir, "cons.5", step)
        int2 = load_var_values(run_dir, "cons.10", step)
        int3 = load_var_values(run_dir, "cons.11", step)
        rho_y_by_name = {
            name: load_var_values(run_dir, f"cons.{species_field[name]}", step)
            for name in species_names
        }
        cell_count = len(arho2)
        if any(len(values) != cell_count for values in [arho3, alpha2, alpha3, total_e, int2, int3, *rho_y_by_name.values()]):
            raise ValueError(f"Incomplete V2 fields at step {step}")

        cell_temperatures = []
        cell_pressures = []
        cell_p2 = []
        cell_p3 = []
        species_means = defaultdict(float)
        x_oh_values = []
        closure_values = []

        for cell_idx in range(cell_count):
            rho_g_stored = arho2[cell_idx] + arho3[cell_idx]
            alpha_g = alpha2[cell_idx] + alpha3[cell_idx]
            rho_g_intrinsic = rho_g_stored / alpha_g if alpha_g > 0.0 else math.nan
            ys = {
                name: rho_y_by_name[name][cell_idx] / rho_g_stored if rho_g_stored > 0.0 else math.nan
                for name in species_names
            }
            sum_rho_y = sum(rho_y_by_name[name][cell_idx] for name in species_names)
            closure = abs(sum_rho_y - rho_g_stored) / max(abs(rho_g_stored), 1.0e-30)
            closure_values.append(closure)
            max_species_closure = max(max_species_closure, closure)
            min_y = min(min_y, min(ys.values()))
            min_rho_y = min(min_rho_y, min(rho_y_by_name[name][cell_idx] for name in species_names))

            p2 = ((int2[cell_idx] - arho2[cell_idx] * V2_EOS[2]["qv"]) / alpha2[cell_idx] - V2_EOS[2]["pi_inf"]) / V2_EOS[2]["gamma"]
            p3 = ((int3[cell_idx] - arho3[cell_idx] * V2_EOS[3]["qv"]) / alpha3[cell_idx] - V2_EOS[3]["pi_inf"]) / V2_EOS[3]["gamma"]
            p_g = (alpha2[cell_idx] * p2 + alpha3[cell_idx] * p3) / alpha_g
            temperature = reconstruct_temperature(
                rho_g_intrinsic,
                p_g,
                [ys[name] for name in species_names],
                [molecular_weights[name] for name in species_names],
            )
            cell_p2.append(p2)
            cell_p3.append(p3)
            cell_pressures.append(p_g)
            cell_temperatures.append(temperature)
            max_p23_mismatch = max(max_p23_mismatch, abs(p2 - p3) / max(abs(p_g), 1.0))
            for name in V2_COMPARE_SPECIES:
                species_means[name] += ys[name]
            x_oh_values.append(species_mole_fraction(ys, molecular_weights, "OH"))

        for name in species_means:
            species_means[name] /= cell_count
        e_mean = mean(total_e)
        if e0 is None:
            e0 = e_mean
        max_energy_drift = max(max_energy_drift, abs(e_mean - e0) / max(abs(e0), 1.0e-300))

        trajectory.append(
            {
                "step": step,
                "time_s": step * dt,
                "T_K": mean(cell_temperatures),
                "P_Pa": mean(cell_pressures),
                "P2_Pa": mean(cell_p2),
                "P3_Pa": mean(cell_p3),
                "rho_g_stored": mean([a + b for a, b in zip(arho2, arho3)]),
                "alpha_g": mean([a + b for a, b in zip(alpha2, alpha3)]),
                "total_E": e_mean,
                "max_species_closure": max(closure_values),
                "min_Y": min_y,
                "min_rhoY": min_rho_y,
                "X_OH": mean(x_oh_values),
                **{f"Y_{name}": species_means[name] for name in V2_COMPARE_SPECIES},
            }
        )

    times = [row["time_s"] for row in trajectory]
    temperatures = [row["T_K"] for row in trajectory]
    pressures = [row["P_Pa"] for row in trajectory]
    x_oh = [row["X_OH"] for row in trajectory]
    peak_oh = max(x_oh)
    t_half_peak_oh = crossing_time(times, x_oh, 0.5 * peak_oh) if peak_oh > 0.0 else None
    t_tplus400 = crossing_time(times, temperatures, temperatures[0] + 400.0)

    max_t_rel = 0.0
    max_p_rel = 0.0
    for row in trajectory:
        tref = interpolate(ref_times, [hrow["T_K"] for hrow in history], row["time_s"])
        pref = interpolate(ref_times, [hrow["P_Pa"] for hrow in history], row["time_s"])
        max_t_rel = max(max_t_rel, abs(row["T_K"] - tref) / max(abs(tref), 1.0e-300))
        max_p_rel = max(max_p_rel, abs(row["P_Pa"] - pref) / max(abs(pref), 1.0e-300))

    final = trajectory[-1]
    ref_final = history[-1]
    checks = []
    if t_half_peak_oh is None:
        checks.append(status_row("t_half_peak_OH_s", "missing", reference["t_half_peak_X_OH_s"], "missing", max(2.0e-6, 0.1 * reference["t_half_peak_X_OH_s"]), "BLOCKED", "MFC OH never crossed half its peak."))
    else:
        tol = max(2.0e-6, 0.1 * reference["t_half_peak_X_OH_s"])
        err = abs(t_half_peak_oh - reference["t_half_peak_X_OH_s"])
        checks.append(status_row("t_half_peak_OH_s", t_half_peak_oh, reference["t_half_peak_X_OH_s"], err, tol, pass_fail(err, tol), "Linear interpolation across 1 us saved cadence."))
    peak_oh_err = abs(peak_oh - reference["peak_X_OH"])
    checks.append(
        status_row(
            "peak_X_OH",
            peak_oh,
            reference["peak_X_OH"],
            peak_oh_err,
            max(1.0e-6, 0.1 * reference["peak_X_OH"]),
            pass_fail(peak_oh_err, max(1.0e-6, 0.1 * reference["peak_X_OH"])),
        )
    )
    if t_tplus400 is None:
        checks.append(status_row("t_Tplus400_s", "missing", reference["t_Tplus400_s"], "missing", max(2.0e-6, 0.1 * reference["t_Tplus400_s"]), "BLOCKED", "MFC temperature never crossed T0+400 K."))
    else:
        tol = max(2.0e-6, 0.1 * reference["t_Tplus400_s"])
        err = abs(t_tplus400 - reference["t_Tplus400_s"])
        checks.append(status_row("t_Tplus400_s", t_tplus400, reference["t_Tplus400_s"], err, tol, pass_fail(err, tol), "Linear interpolation across 1 us saved cadence."))

    for metric, mfc_value, ref_value, tol in (
        ("final_T_K", final["T_K"], reference["T_final_K"], max(2.0, 0.01 * abs(reference["T_final_K"]))),
        ("max_rel_T_history", max_t_rel, 0.0, 0.02),
        ("max_rel_P_history", max_p_rel, 0.0, 0.02),
    ):
        err = abs(mfc_value - ref_value) if metric == "final_T_K" else mfc_value
        checks.append(status_row(metric, mfc_value, ref_value, err, tol, pass_fail(err, tol)))

    for name in ("NC12H26", "O2"):
        ref_value = ref_final[f"Y_{name}"]
        mfc_value = final[f"Y_{name}"]
        tol = max(5.0e-5, 0.1 * abs(ref_value))
        err = abs(mfc_value - ref_value)
        checks.append(status_row(f"final_Y_{name}", mfc_value, ref_value, err, tol, pass_fail(err, tol)))

    for metric, value, tol in (
        ("max_species_closure_error", max_species_closure, 1.0e-10),
        ("max_total_E_drift", max_energy_drift, 1.0e-10),
        ("max_p2_p3_mismatch", max_p23_mismatch, 1.0e-10),
    ):
        checks.append(status_row(metric, value, 0.0, value, tol, pass_fail(value, tol)))
    checks.append(status_row("min_Y", min_y, 0.0, min_y, -1.0e-12, "PASS" if min_y >= -1.0e-12 else "FAIL"))
    checks.append(status_row("min_rhoY", min_rho_y, 0.0, min_rho_y, -1.0e-12, "PASS" if min_rho_y >= -1.0e-12 else "FAIL"))

    field_mapping = {
        "rho_g_stored": "cons.2 + cons.3",
        "alpha_g": "cons.7 + cons.8",
        "alpha_rho_2": "cons.2",
        "alpha_rho_3": "cons.3",
        "alpha_2": "cons.7",
        "alpha_3": "cons.8",
        "total_E": "cons.5",
        "int_en_2": "cons.10",
        "int_en_3": "cons.11",
        "species_rhoY": "cons.(11 + SK54 species id); e.g. NC12H26=cons.57, O2=cons.21, OH=cons.16, H2O=cons.19",
        "pressure": "p_i = ((int_en_i - alpha_rho_i*qv_i)/alpha_i - pi_inf_i)/gamma_i, alpha-weighted over fluids 2 and 3",
        "temperature": "T = p_g * W(Y) / (rho_g_intrinsic * Ru)",
    }
    summary = {
        "case": "V2",
        "field_mapping": field_mapping,
        "post_process_rerun_required": False,
        "saved_cadence_s": times[1] - times[0] if len(times) > 1 else math.nan,
        "interpolation": "linear interpolation was used for ignition crossing metrics across the 1 us saved cadence",
        "final_output_step": steps[-1],
        "mfc_final": final,
        "reference_metrics": reference,
        "max_species_closure_error": max_species_closure,
        "min_Y": min_y,
        "min_rhoY": min_rho_y,
        "max_total_E_drift": max_energy_drift,
        "max_p2_p3_mismatch": max_p23_mismatch,
        "max_rel_T_history": max_t_rel,
        "max_rel_P_history": max_p_rel,
        "t_half_peak_OH_s": t_half_peak_oh,
        "t_Tplus400_s": t_tplus400,
        "overall_status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
    }
    return summary, checks, trajectory


def read_probe(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                values = [float(value) for value in parts[:4]]
            except ValueError:
                continue
            rows.append({"time_s": values[0], "rho": values[1], "vel_x": values[2], "pressure_Pa": values[3]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(prog="analyze_outputs")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case", default="unknown")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "validation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = []
    rows = []
    by_step: dict[int, list[dict]] = defaultdict(list)
    for path in dat_files(run_dir):
        parsed = parse_name(path)
        record = {"path": str(path), "name": path.name, "parsed": parsed is not None}
        if parsed:
            var, rank, step = parsed
            stats = read_last_column(path)
            record.update({"var": var, "rank": rank, "step": step, **stats})
            rows.append(record)
            by_step[step].append(record)
        inventory.append(record)

    with (output_dir / "inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)

    with (output_dir / "field_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["step", "rank", "var", "count", "sum", "min", "max", "finite", "path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["step"], item["var"], item["rank"])):
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    probe_rows = []
    for path in probe_files(run_dir):
        match = PROBE_RE.match(path.name)
        probe_id = int(match.group("probe")) if match else -1
        for row in read_probe(path):
            probe_rows.append({"probe": probe_id, **row, "path": str(path)})

    with (output_dir / "probe_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["probe", "time_s", "rho", "vel_x", "pressure_Pa", "path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_rows)

    quantitative_summary = None
    quantitative_checks = []
    if args.case.upper() == "V1":
        quantitative_summary, quantitative_checks = analyze_v1(run_dir)
        with (output_dir / "v1_quantitative_results.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": quantitative_summary, "checks": quantitative_checks}, handle, indent=2, sort_keys=True)
        with (output_dir / "v1_pass_fail.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["criterion", "mfc", "reference", "tolerance", "error", "pass"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quantitative_checks)
    elif args.case.upper() == "V2":
        quantitative_summary, quantitative_checks, trajectory = analyze_v2(run_dir)
        with (output_dir / "v2_quantitative_results.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": quantitative_summary, "checks": quantitative_checks}, handle, indent=2, sort_keys=True)
        with (output_dir / "v2_pass_fail.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["metric", "mfc", "reference", "error", "tolerance", "status", "note"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quantitative_checks)
        with (output_dir / "v2_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(trajectory[0]) if trajectory else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Validation Output Summary\n\n")
        handle.write(f"- Case label: `{args.case}`\n")
        handle.write(f"- Run directory: `{run_dir}`\n")
        handle.write(f"- Parsed `.dat` fields: {len(rows)}\n")
        handle.write(f"- Parsed probe rows: {len(probe_rows)}\n")
        handle.write(f"- Timesteps found: {', '.join(str(step) for step in sorted(by_step)) or 'none'}\n\n")
        checks = CASE_CHECKS.get(args.case.upper())
        if checks:
            handle.write("## Quantitative Checks\n\n")
            for check in checks:
                handle.write(f"- {check}\n")
            handle.write("\n")
        if quantitative_summary is not None and args.case.upper() == "V1":
            handle.write("## V1 PASS/FAIL\n\n")
            handle.write(f"- Overall: {'PASS' if quantitative_summary['pass'] else 'FAIL'}\n")
            handle.write(f"- Final reconstructed T: {quantitative_summary['mfc_final']['T_K']:.12g} K\n")
            handle.write(f"- Reference final T: {quantitative_summary['reference']['T_final_K']:.12g} K\n")
            handle.write(f"- max_abs(sum(Y)-1): {quantitative_summary['max_abs_sumY_minus_1']:.12e}\n")
            handle.write(f"- min(Y): {quantitative_summary['min_Y']:.12e}\n\n")
            handle.write("| criterion | MFC | reference | tolerance | error | pass |\n")
            handle.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
            for check in quantitative_checks:
                handle.write(
                    f"| {check['criterion']} | {check['mfc']:.12g} | {check['reference']:.12g} | "
                    f"{check['tolerance']:.12g} | {check['error']:.12g} | {check['pass']} |\n"
                )
            handle.write("\n")
        elif quantitative_summary is not None and args.case.upper() == "V2":
            handle.write("## V2 Field Mapping\n\n")
            for name, mapping in quantitative_summary["field_mapping"].items():
                handle.write(f"- `{name}`: {mapping}\n")
            handle.write("\n")
            handle.write("## V2 PASS/FAIL\n\n")
            handle.write(f"- Overall: {quantitative_summary['overall_status']}\n")
            handle.write(f"- Post-process rerun required: {quantitative_summary['post_process_rerun_required']}\n")
            handle.write(f"- Ignition interpolation: {quantitative_summary['interpolation']}\n")
            handle.write(f"- Final reconstructed T: {quantitative_summary['mfc_final']['T_K']:.12g} K\n")
            handle.write(f"- Reference final T: {quantitative_summary['reference_metrics']['T_final_K']:.12g} K\n")
            handle.write(f"- max species closure error: {quantitative_summary['max_species_closure_error']:.12e}\n")
            handle.write(f"- max total-E drift: {quantitative_summary['max_total_E_drift']:.12e}\n")
            handle.write(f"- max p2/p3 mismatch: {quantitative_summary['max_p2_p3_mismatch']:.12e}\n")
            handle.write(f"- min Y: {quantitative_summary['min_Y']:.12e}\n")
            handle.write(f"- min rhoY: {quantitative_summary['min_rhoY']:.12e}\n\n")
            handle.write("| metric | MFC | Cantera/reference | error | tolerance | status |\n")
            handle.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
            for check in quantitative_checks:
                handle.write(
                    f"| {check['metric']} | {format_metric_value(check['mfc'])} | "
                    f"{format_metric_value(check['reference'])} | {format_metric_value(check['error'])} | "
                    f"{format_metric_value(check['tolerance'])} | {check['status']} |\n"
                )
            handle.write("\n")
        handle.write("This helper summarizes available post-process ASCII tables only. ")
        handle.write("Operator-internal quantities such as phase-change delta mass or face flux equality ")
        handle.write("must be inferred from saved fields unless solver diagnostics are enabled in a later stage.\n")

    print(f"Wrote {output_dir / 'inventory.json'}")
    print(f"Wrote {output_dir / 'field_summary.csv'}")
    print(f"Wrote {output_dir / 'probe_summary.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
