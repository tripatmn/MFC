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
    "liquid_area_alpha_gt_0p5": ("Liquid area", "Area / volume"),
    "liquid_area_ratio_A_A0": ("Liquid area ratio", "A / A0"),
    "liquid_NC12H26_inventory": ("Liquid NC12H26", "Conservative inventory"),
    "vapor_NC12H26_inventory": ("Gas NC12H26", "Conservative inventory"),
    "total_NC12H26_inventory": ("Total NC12H26", "Conservative inventory"),
    **{
        f"gas_mass_weighted_Y_{name}": (
            f"Gas-mass-weighted Y({name})", "Gas-mass-weighted mass fraction"
        )
        for name in ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO", "CO2", "H2O")
    },
}

TREND_SPECS = {
    "valid_gas_temperature_max": {
        "fields": ("max_valid_gas_temperature_K",),
        "title": "Maximum valid-gas temperature", "ylabel": "Temperature (K)",
    },
    "products_CO2_H2O": {
        "fields": ("gas_mass_weighted_Y_CO2", "gas_mass_weighted_Y_H2O"),
        "title": "Gas-phase products", "ylabel": "Gas-mass-weighted mass fraction",
    },
    "radicals_OH_HO2_H2O2": {
        "fields": ("gas_mass_weighted_Y_OH", "gas_mass_weighted_Y_HO2", "gas_mass_weighted_Y_H2O2"),
        "title": "Gas-phase radicals", "ylabel": "Gas-mass-weighted mass fraction",
    },
    "hot_combustible_overlap_area": {
        "fields": ("hot_combustible_overlap_area",),
        "title": "Hot-combustible overlap", "ylabel": "Area / volume",
    },
    "hot_near_stoich_overlap_area": {
        "fields": ("hot_near_stoich_overlap_area",),
        "title": "Hot-near-stoichiometric overlap", "ylabel": "Area / volume",
    },
    "near_stoichiometric_area": {
        "fields": ("near_stoichiometric_area",),
        "title": "Near-stoichiometric gas area", "ylabel": "Area / volume",
    },
    "liquid_area_ratio_A_A0": {
        "fields": ("liquid_area_ratio_A_A0",),
        "title": "Liquid area ratio", "ylabel": "A / A0",
    },
    "dodecane_inventory": {
        "fields": (
            "liquid_NC12H26_inventory", "vapor_NC12H26_inventory",
            "total_NC12H26_inventory",
        ),
        "title": "Dodecane inventory", "ylabel": "Conservative inventory",
    },
}

PLOT_SETS = {
    "thermal": ("valid_gas_temperature_max",),
    "species": ("products_CO2_H2O", "radicals_OH_HO2_H2O2", "dodecane_inventory"),
    "mixing": (
        "hot_combustible_overlap_area", "hot_near_stoich_overlap_area",
        "near_stoichiometric_area", "liquid_area_ratio_A_A0",
    ),
    "all": tuple(TREND_SPECS),
    "default": tuple(TREND_SPECS),
}


def plot_history(
    input_path: str | Path,
    fields: Iterable[str] | None = None,
    plot_set: str | None = None,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    csv_path = _csv_path(input_path)
    destination = Path(out_dir).expanduser().resolve() if out_dir else csv_path.parent / "trend_plots"
    requested_fields = tuple(fields or ())
    if requested_fields and plot_set:
        raise ValueError("--fields and --plot-set are mutually exclusive")
    if requested_fields:
        unknown = [name for name in requested_fields if name not in FIELD_LABELS]
        if unknown:
            raise ValueError(f"unknown scalar plot fields: {unknown}")
        specs = tuple(_individual_spec(name) for name in requested_fields)
    else:
        selected_set = plot_set or "default"
        if selected_set not in PLOT_SETS:
            raise ValueError(f"unknown plot set {selected_set!r}; choose from {sorted(PLOT_SETS)}")
        specs = tuple({"name": name, **TREND_SPECS[name]} for name in PLOT_SETS[selected_set])
    if not specs:
        raise ValueError("plot selection produced zero artifacts")
    needed_fields = tuple(dict.fromkeys(field for spec in specs for field in spec["fields"]))

    print(f"mfc-post plot: reading scalar history {csv_path}", flush=True)
    rows, columns = _read_csv(csv_path)
    missing = [name for name in ("saved_index", "time_us", *needed_fields) if name not in columns]
    if missing:
        raise ValueError(f"scalar history is missing required columns: {missing}")
    for row_number, row in enumerate(rows, 2):
        row["_mfc_time_us"] = _parse_float(row["time_us"], "time_us", row_number)
        row["_mfc_saved_index"] = _parse_int(row["saved_index"], "saved_index", row_number)
        row["_mfc_csv_row"] = row_number
    rows.sort(key=lambda row: (row["_mfc_time_us"], row["_mfc_saved_index"]))
    times = np.asarray([row["_mfc_time_us"] for row in rows])
    series = {
        name: np.asarray([
            _parse_float(row[name], name, row["_mfc_csv_row"])
            for row in rows
        ])
        for name in needed_fields
    }
    print(f"mfc-post plot: output directory: {destination}", flush=True)
    _prepare_output(destination, overwrite)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    plots = []
    for position, spec in enumerate(specs, 1):
        print(
            f"mfc-post plot: progress: plot {position}/{len(specs)}: {spec['name']}",
            flush=True,
        )
        ylabel = spec["ylabel"]
        if all(
            name.startswith("integrated_rhoY_") or name.endswith("_NC12H26_inventory")
            for name in spec["fields"]
        ):
            ylabel = f"{ylabel} ({rows[0].get('integrated_rhoY_unit') or 'case units'})"
        elif all(name.endswith("area") or "_area_" in name for name in spec["fields"]):
            ylabel = f"{ylabel} ({rows[0].get('spatial_measure_unit') or 'case units'})"
        path = destination / f"{spec['name']}.png"
        figure = None
        try:
            figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            for name in spec["fields"]:
                label = _series_label(name)
                axis.plot(
                    times, series[name], marker="o", linewidth=1.6, markersize=3.5,
                    label=label,
                )
            if len(spec["fields"]) > 1:
                axis.legend(frameon=False)
            axis.set(
                title=spec["title"], xlabel="Physical time (microseconds)", ylabel=ylabel,
            )
            axis.grid(True, alpha=0.3)
            figure.savefig(path, dpi=180)
        except Exception as exc:
            raise RuntimeError(f"failed to render trend {spec['name']} to {path}: {exc}") from exc
        finally:
            if figure is not None:
                plt.close(figure)
        files.append(str(path))
        plots.append({"name": spec["name"], "fields": list(spec["fields"]), "path": str(path)})

    if not files:
        raise RuntimeError("plot produced zero PNG artifacts; no manifest written")

    manifest = {
        "schema_version": "mfc-post.plot/v1",
        "input_csv": str(csv_path),
        "input_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "time_range_us": [float(times.min()), float(times.max())],
        "fields": list(needed_fields),
        "plots": plots,
        "files": files,
        "data_access": "CSV only; plot does not inspect or read p_all",
    }
    _atomic_json(destination / "plot_manifest.json", manifest)
    print(f"mfc-post plot: completion: wrote {len(files)} PNG(s) to {destination}", flush=True)
    return {"output_directory": str(destination), **manifest}


def _individual_spec(name: str) -> dict:
    aliases = {
        "max_valid_gas_temperature_K": "valid_gas_temperature_max",
        "hot_combustible_overlap_area": "hot_combustible_overlap_area",
        "hot_near_stoich_overlap_area": "hot_near_stoich_overlap_area",
    }
    title, ylabel = FIELD_LABELS[name]
    return {
        "name": aliases.get(name, name), "fields": (name,),
        "title": title, "ylabel": ylabel,
    }


def _series_label(name: str) -> str:
    if name.startswith("integrated_rhoY_"):
        return name.removeprefix("integrated_rhoY_")
    if name.startswith("gas_mass_weighted_Y_"):
        return name.removeprefix("gas_mass_weighted_Y_")
    inventory_labels = {
        "liquid_NC12H26_inventory": "liquid",
        "vapor_NC12H26_inventory": "vapor / gas",
        "total_NC12H26_inventory": "total",
    }
    if name in inventory_labels:
        return inventory_labels[name]
    return FIELD_LABELS[name][0]


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


def _prepare_output(destination: Path, overwrite: bool) -> None:
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination}; use --overwrite to replace "
            "mfc-post plot outputs"
        )
    if destination.exists() and overwrite:
        generated_names = {
            "plot_manifest.json",
            *(f"{name}.png" for name in TREND_SPECS),
            *(f"{name}.png" for name in FIELD_LABELS),
            "valid_gas_temperature_max.png",
        }
        for name in generated_names:
            path = destination / name
            if path.is_file():
                path.unlink()
    destination.mkdir(parents=True, exist_ok=True)


def _parse_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid numeric value for {field} at CSV row {row_number}: {value!r}"
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(
            f"non-finite numeric value for {field} at CSV row {row_number}: {value!r}"
        )
    return parsed


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer value for {field} at CSV row {row_number}: {value!r}"
        ) from exc


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
