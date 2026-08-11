"""Trend plotting from an existing scalar-timeseries CSV only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


FIELD_LABELS = {
    "integrated_rhoY_NC12H26": ("Integrated NC12H26 inventory", "Conservative inventory"),
    "integrated_rhoY_OH": ("Integrated OH inventory", "Conservative inventory"),
    "integrated_rhoY_HO2": ("Integrated HO2 inventory", "Conservative inventory"),
    "integrated_rhoY_H2O2": ("Integrated H2O2 inventory", "Conservative inventory"),
    "integrated_rhoY_CO": ("Integrated CO inventory", "Conservative inventory"),
    "integrated_rhoY_CO2": ("Integrated CO2 inventory", "Conservative inventory"),
    "integrated_rhoY_H2O": ("Integrated H2O inventory", "Conservative inventory"),
    "max_valid_gas_temperature_K": ("Maximum valid-gas temperature", "Temperature (K)"),
    "mean_valid_gas_temperature_K": ("Mean valid-gas temperature", "Temperature (K)"),
    "max_Y_NC12H26": ("Maximum gas-phase Y(NC12H26)", "Mass fraction"),
    "max_Y_OH": ("Maximum gas-phase Y(OH)", "Mass fraction"),
    "max_Y_CO2": ("Maximum gas-phase Y(CO2)", "Mass fraction"),
    "max_Y_H2O": ("Maximum gas-phase Y(H2O)", "Mass fraction"),
    "hot_gas_area_above_1200K": ("Hot gas above 1200 K", "Area / volume"),
    "hot_gas_area_above_1500K": ("Hot gas above 1500 K", "Area / volume"),
    "combustible_area": ("Combustible gas", "Area / volume"),
    "near_stoichiometric_area": ("Near-stoichiometric gas", "Area / volume"),
    "hot_combustible_overlap_area": ("Hot-combustible overlap", "Area / volume"),
    "hot_near_stoich_overlap_area": ("Hot-near-stoichiometric overlap", "Area / volume"),
}

PLOT_SETS = {
    "species": tuple(name for name in FIELD_LABELS if name.startswith("integrated_rhoY_")),
    "thermal": (
        "max_valid_gas_temperature_K", "mean_valid_gas_temperature_K",
        "hot_gas_area_above_1200K", "hot_gas_area_above_1500K",
    ),
    "mixing": (
        "max_Y_NC12H26", "max_Y_OH", "max_Y_CO2", "max_Y_H2O",
        "combustible_area", "near_stoichiometric_area",
        "hot_combustible_overlap_area", "hot_near_stoich_overlap_area",
    ),
    "all": tuple(FIELD_LABELS),
}
PLOT_SETS["default"] = PLOT_SETS["all"]


def plot_history(
    input_path: str | Path,
    fields: Iterable[str] | None = None,
    plot_set: str | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    csv_path = _csv_path(input_path)
    destination = Path(out_dir).expanduser().resolve() if out_dir else csv_path.parent / "trend_plots"
    requested = tuple(fields or ())
    if requested and plot_set:
        raise ValueError("--fields and --plot-set are mutually exclusive")
    if not requested:
        selected_set = plot_set or "default"
        if selected_set not in PLOT_SETS:
            raise ValueError(f"unknown plot set {selected_set!r}; choose from {sorted(PLOT_SETS)}")
        requested = PLOT_SETS[selected_set]
    unknown = [name for name in requested if name not in FIELD_LABELS]
    if unknown:
        raise ValueError(f"unknown scalar plot fields: {unknown}")

    print(f"mfc-post plot: reading scalar history {csv_path}", flush=True)
    rows, columns = _read_csv(csv_path)
    missing = [name for name in ("time_us", *requested) if name not in columns]
    if missing:
        raise ValueError(f"scalar history is missing required columns: {missing}")
    _prepare_output(destination)
    rows.sort(key=lambda row: (float(row["time_us"]), int(row["saved_index"])))
    times = np.asarray([float(row["time_us"]) for row in rows])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for name in requested:
        print(f"mfc-post plot: rendering {name}", flush=True)
        values = np.asarray([_float_or_nan(row[name]) for row in rows])
        title, ylabel = FIELD_LABELS[name]
        if name.startswith("integrated_rhoY_"):
            ylabel = f"{ylabel} ({rows[0].get('integrated_rhoY_unit') or 'case units'})"
        elif name.endswith("area") or "_area_" in name:
            ylabel = f"{ylabel} ({rows[0].get('spatial_measure_unit') or 'case units'})"
        figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        axis.plot(times, values, marker="o", linewidth=1.6, markersize=3.5)
        axis.set(title=title, xlabel="Physical time (microseconds)", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        files.append(str(path))

    manifest = {
        "schema_version": "mfc-post.plot/v1",
        "input_csv": str(csv_path),
        "input_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "time_range_us": [float(times.min()), float(times.max())],
        "fields": list(requested),
        "files": files,
        "data_access": "CSV only; plot does not inspect or read p_all",
    }
    _atomic_json(destination / "plot_manifest.json", manifest)
    print(f"mfc-post plot: completion: wrote {len(files)} PNG(s) to {destination}", flush=True)
    return {"output_directory": str(destination), **manifest}


def _csv_path(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / "scalar_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"scalar-timeseries CSV does not exist: {path}")
    return path


def _read_csv(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"scalar history contains no records: {path}")
    return rows, columns


def _prepare_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
