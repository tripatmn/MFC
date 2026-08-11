"""Clean, selected-state contour rendering from raw p_all output."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .execution import ExecutionContext
from .inspect import inspect_case
from .models import RunMetadata
from .process import _transformation_provenance
from .reconstruction import FieldRegistry, Model3Configuration, reconstruct_model3
from .sources import PAllSource


RENDER_FIELDS = ("temperature", "OH", "NC12H26", "O2", "phi", "alpha_liq")
DEFAULT_PHI_LEVELS = (0.5, 1.0, 2.0)
O2_FLOOR = 1.0e-12

FIELD_STYLE = {
    "temperature": ("Temperature", "Temperature [K]", "inferno"),
    "OH": ("OH mass fraction", "OH mass fraction", "magma"),
    "NC12H26": ("NC12H26 mass fraction", "NC12H26 mass fraction", "viridis"),
    "O2": ("O2 mass fraction", "O2 mass fraction", "viridis"),
    "phi": ("Equivalence ratio", "Equivalence ratio", "plasma"),
    "alpha_liq": ("Liquid volume fraction", "Liquid volume fraction", "Blues"),
}


def render_case(
    case_path: str | Path,
    selected_times_us: Iterable[float] | None = None,
    fields: Iterable[str] = RENDER_FIELDS,
    out_dir: str | Path | None = None,
    execution: str = "auto",
    mechanism: str | Path | None = None,
    phase: str | None = None,
    time_range_us: tuple[float, float] | None = None,
    stride: int = 1,
    overwrite: bool = False,
    no_mp4: bool = False,
    overlay: tuple[str, str] | None = None,
    overlay_levels: Iterable[float] = DEFAULT_PHI_LEVELS,
) -> dict[str, Any] | None:
    context = ExecutionContext.create(execution)
    root = Path(case_path).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve() if out_dir else root / "render_clean"
    requested_fields = tuple(str(value).strip() for value in fields if str(value).strip())
    if not requested_fields:
        raise ValueError("--fields requires at least one field")
    if stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if selected_times_us is not None and time_range_us is not None:
        raise ValueError("--selected-times-us and --time-range-us are mutually exclusive")

    _progress(context, f"startup: case path: {root}")
    inspection = inspect_case(root, mechanism=mechanism, phase=phase)
    metadata = _metadata(inspection)
    report = next((item for item in inspection["sources"] if item["family"] == "p_all"), None)
    if report is None or not report["timeline"]["saved_indices"]:
        raise ValueError(f"clean rendering requires raw p_all saves; none were found in {root}")
    if metadata.dimensions != 2:
        raise ValueError(f"PNG contour rendering requires a 2D case, got {metadata.dimensions}D")
    config = Model3Configuration.from_metadata(metadata)
    canonical_fields = tuple(_canonical_symbol(value, config) for value in requested_fields)
    if len(set(canonical_fields)) != len(canonical_fields):
        raise ValueError("--fields resolves to duplicate fields")
    canonical_overlay = None
    if overlay is not None:
        if len(overlay) != 2:
            raise ValueError("--overlay requires BASE,CONTOUR")
        canonical_overlay = tuple(_canonical_symbol(value, config) for value in overlay)
        if canonical_overlay[0] == canonical_overlay[1]:
            raise ValueError("--overlay base and contour fields must differ")
    levels = tuple(float(value) for value in overlay_levels)
    if canonical_overlay and (not levels or any(not np.isfinite(value) for value in levels)):
        raise ValueError("--overlay-levels requires one or more finite values")
    if canonical_overlay and len(set(levels)) != len(levels):
        raise ValueError("--overlay-levels contains duplicates")
    levels = tuple(sorted(levels))

    selections = _select_times(
        report["timeline"],
        tuple(float(value) for value in selected_times_us) if selected_times_us is not None else None,
        time_range_us, stride,
    )
    if not selections:
        raise ValueError("render selection produced zero saved states")
    time_tokens = [_time_token(item["actual_time_us"]) for item in selections]
    if len(set(time_tokens)) != len(time_tokens):
        raise ValueError(
            "selected saves collide at the deterministic 0.01 us filename precision; "
            "narrow the selection or increase --stride"
        )
    _progress(context, f"startup: discovered p_all path: {report['path']}")
    _progress(context, f"startup: saves discovered: {len(report['timeline']['saved_indices'])}")
    _progress(context, f"startup: saves selected after filters: {len(selections)}")
    _progress(context, f"startup: output directory: {destination}")
    _prepare_output(context, destination, overwrite)

    source = PAllSource(root, metadata)
    raw_fields = tuple(str(item["name"]) for item in metadata.equation_layout)
    load_symbols = tuple(dict.fromkeys(canonical_fields + (canonical_overlay or ())))
    ranges = {symbol: [float("inf"), float("-inf")] for symbol in load_symbols}
    all_sources: set[str] = set()
    cached: dict[int, dict[str, Any]] = {}

    for position, selection in enumerate(selections, 1):
        saved_index = int(selection["saved_index"])
        _progress(context, f"progress: limits {position}/{len(selections)}: saved_index={saved_index}")
        assembled = _load_assembled(source, saved_index, raw_fields, load_symbols, config, context)
        if context.rank == 0:
            all_sources.update(assembled["source_files"])
            for symbol in load_symbols:
                plotted, _ = _plot_data(symbol, assembled)
                observed = _range(plotted)
                if observed[0] is not None:
                    ranges[symbol][0] = min(ranges[symbol][0], observed[0])
                    ranges[symbol][1] = max(ranges[symbol][1], observed[1])
            if len(selections) == 1:
                cached[saved_index] = assembled

    limits = None
    limit_error = None
    if context.rank == 0:
        try:
            limits = {symbol: _final_limits(symbol, values) for symbol, values in ranges.items()}
        except Exception as exc:
            limit_error = f"{type(exc).__name__}: {exc}"
    limit_errors = [item for item in context.comm.allgather(limit_error) if item]
    if limit_errors:
        raise RuntimeError("rendering failed while determining batch limits: " + limit_errors[0])

    frames: list[dict[str, Any]] = []
    for position, selection in enumerate(selections, 1):
        saved_index = int(selection["saved_index"])
        _progress(context, f"progress: render {position}/{len(selections)}: saved_index={saved_index}")
        assembled = cached.get(saved_index) if context.rank == 0 else None
        if len(selections) != 1:
            assembled = _load_assembled(source, saved_index, raw_fields, load_symbols, config, context)
        root_error = None
        if context.rank == 0:
            try:
                counts = _mask_counts(assembled)
                for symbol in canonical_fields:
                    frames.append(_render_field(
                        destination, selection, symbol, assembled, limits[symbol], config, counts
                    ))
                if canonical_overlay:
                    frames.append(_render_overlay(
                        destination, selection, canonical_overlay, levels, assembled,
                        limits[canonical_overlay[0]], counts,
                    ))
            except Exception as exc:
                root_error = (
                    f"worker_rank=0 source=p_all saved_index={saved_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        render_errors = [item for item in context.comm.allgather(root_error) if item]
        if render_errors:
            raise RuntimeError("rendering failed; no manifest written:\n" + "\n".join(render_errors))

    if context.rank != 0:
        return None
    if not frames:
        raise RuntimeError("render produced zero PNG artifacts; no manifest written")
    provenance = _provenance(config, all_sources, limits, canonical_overlay, levels)
    manifest = {
        "schema_version": "mfc-post.render-clean/v1",
        "case": str(root),
        "source": {"family": "p_all", "path": report["path"], "layout": report["layout"]},
        "selection_policy": "nearest requested saves or inclusive time range, followed by stride",
        "selections": selections,
        "fields": list(canonical_fields),
        "field_limits": {
            symbol: {"minimum": value[0], "maximum": value[1]} for symbol, value in limits.items()
        },
        "overlay": (
            {"base": canonical_overlay[0], "contour": canonical_overlay[1], "levels": list(levels)}
            if canonical_overlay else None
        ),
        "frames": frames,
        "options": {
            "clean": True, "static_png_only": True, "no_mp4": bool(no_mp4),
            "execution": {"mode": context.mode, "mpi_size": context.size},
        },
        "warnings": inspection["warnings"] + report["warnings"],
    }
    _atomic_json(destination / "manifest.json", manifest)
    _atomic_json(destination / "provenance.json", provenance)
    _progress(context, f"completion: artifact count: {len(frames)}")
    _progress(context, f"completion: manifest.json: {destination / 'manifest.json'}")
    _progress(context, f"completion: provenance.json: {destination / 'provenance.json'}")
    manifest["output_directory"] = str(destination)
    return manifest


def _metadata(inspection: dict[str, Any]) -> RunMetadata:
    raw = inspection["run_metadata"]
    return RunMetadata(
        case_path=raw["case_path"], parameters=raw["parameters"], dimensions=raw["dimensions"],
        grid_shape=tuple(raw["grid_shape"]) if raw["grid_shape"] else None,
        equation_layout=tuple(raw["equation_layout"]), species_names=tuple(raw["species_names"]),
        missing=tuple(raw["missing"]), warnings=tuple(raw["warnings"]),
    )


def _select_times(timeline, selected_times_us=None, time_range_us=None, stride=1):
    available = [
        (int(index), float(time) * 1.0e6)
        for index, time in zip(timeline["saved_indices"], timeline["physical_times"])
        if time is not None and np.isfinite(time)
    ]
    if not available:
        raise ValueError("p_all timeline has no determinable physical times")
    selections = []
    if selected_times_us is not None:
        if not selected_times_us or any(not np.isfinite(value) for value in selected_times_us):
            raise ValueError("--selected-times-us requires one or more finite values")
        seen = set()
        for requested in selected_times_us:
            index, actual = min(
                available, key=lambda item: (abs(item[1] - requested), item[1], item[0])
            )
            if index in seen:
                continue
            seen.add(index)
            selections.append({
                "requested_time_us": requested, "saved_index": index,
                "actual_time_us": actual, "time_error_us": actual - requested,
            })
    elif time_range_us is not None:
        start, stop = (float(value) for value in time_range_us)
        if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
            raise ValueError("--time-range-us must be finite START,STOP with STOP >= START")
        selections = [
            {"requested_time_us": actual, "saved_index": index, "actual_time_us": actual, "time_error_us": 0.0}
            for index, actual in available if start <= actual <= stop
        ]
    else:
        selections = [
            {"requested_time_us": actual, "saved_index": index, "actual_time_us": actual, "time_error_us": 0.0}
            for index, actual in available
        ]
    selections = selections[::stride]
    for ordinal, selection in enumerate(selections, 1):
        selection["selection_ordinal"] = ordinal
    return selections


def _canonical_symbol(symbol: str, config: Model3Configuration) -> str:
    cleaned = symbol.strip()
    aliases = {
        "temperature": "temperature", "T": "temperature",
        "phi": "phi", "equivalence_ratio": "phi",
        "alpha_liq": "alpha_liq", "alpha[liquid]": "alpha_liq",
    }
    if cleaned in aliases:
        return aliases[cleaned]
    requested = cleaned[2:-1] if cleaned.startswith("Y[") and cleaned.endswith("]") else cleaned
    standard = next(
        (name for name in ("OH", "NC12H26", "O2") if name.casefold() == requested.casefold()),
        None,
    )
    if standard is None:
        raise ValueError("clean render supports species fields OH, NC12H26, and O2")
    matches = [name for name in config.species_names if name.casefold() == standard.casefold()]
    if len(matches) != 1:
        raise ValueError(f"requested render field/species {cleaned!r} is absent or ambiguous")
    return standard


def _load_assembled(source, saved_index, raw_fields, symbols, config, context):
    local_pieces = []
    local_error = None
    try:
        partitions = source.partition_ids(saved_index)
        for partition in partitions[context.rank::context.size]:
            state = source.read_partition(saved_index, partition, raw_fields)
            physical = reconstruct_model3(state, config)
            local_pieces.append(_piece(state, physical, symbols, config))
    except Exception as exc:
        local_error = (
            f"worker_rank={context.rank} source=p_all saved_index={saved_index}: "
            f"{type(exc).__name__}: {exc}"
        )
    state_errors = [item for item in context.comm.allgather(local_error) if item]
    if state_errors:
        raise RuntimeError("rendering failed while loading state:\n" + "\n".join(sorted(state_errors)))
    gathered = context.comm.gather(local_pieces, root=0)
    if context.rank != 0:
        return None
    pieces = [piece for group in gathered for piece in group]
    if not pieces:
        raise RuntimeError(f"saved index {saved_index}: no p_all partitions were loaded")
    assembled = _assemble(pieces, symbols)
    assembled["source_files"] = sorted({path for piece in pieces for path in piece["source_files"]})
    return assembled


def _piece(state, physical, symbols, config):
    if state.grid is None or not state.grid.bounds.get("x") or not state.grid.bounds.get("y"):
        raise ValueError(f"saved index {state.saved_index}: partition lacks 2D physical coordinates")
    registry = FieldRegistry(physical, config)
    fields = {
        "temperature": np.asarray(registry.resolve("temperature").values, dtype=np.float64),
        "temperature.raw": np.asarray(registry.resolve("temperature.raw").values, dtype=np.float64),
        "pressure": np.asarray(registry.resolve("pressure").values, dtype=np.float64),
        "pressure.raw": np.asarray(registry.resolve("pressure.raw").values, dtype=np.float64),
    }
    for symbol in symbols:
        if symbol in {"OH", "NC12H26", "O2"}:
            fields[symbol] = np.asarray(registry.resolve(f"Y[{symbol}]").values, dtype=np.float64)
        elif symbol == "alpha_liq":
            fields[symbol] = np.asarray(registry.resolve("alpha[liquid]").values, dtype=np.float64)
    if "phi" in symbols:
        fuel = fields.get("NC12H26")
        oxygen = fields.get("O2")
        if fuel is None:
            fuel = np.asarray(registry.resolve("Y[NC12H26]").values, dtype=np.float64)
        if oxygen is None:
            oxygen = np.asarray(registry.resolve("Y[O2]").values, dtype=np.float64)
        phi = np.full(fuel.shape, np.nan)
        valid = np.isfinite(fuel) & np.isfinite(oxygen) & (fuel >= 0.0) & (oxygen > O2_FLOOR)
        phi[valid] = _stoich_coefficient(config) * fuel[valid] / oxygen[valid]
        fields["phi"] = phi
    masks = {name: np.asarray(field.values, dtype=bool) for name, field in physical.masks.items()}
    return {
        "x_bounds": np.asarray(state.grid.bounds["x"], dtype=np.float64),
        "y_bounds": np.asarray(state.grid.bounds["y"], dtype=np.float64),
        "fields": fields, "masks": masks,
        "source_files": sorted({
            field.provenance.source_path for field in state.fields.values() if field.provenance
        }),
    }


def _assemble(pieces, symbols):
    x_bounds = _merge_bounds(piece["x_bounds"] for piece in pieces)
    y_bounds = _merge_bounds(piece["y_bounds"] for piece in pieces)
    shape = (len(y_bounds) - 1, len(x_bounds) - 1)
    names = set(symbols) | {"temperature", "temperature.raw", "pressure", "pressure.raw"}
    fields = {name: np.full(shape, np.nan) for name in names}
    mask_names = set().union(*(piece["masks"] for piece in pieces))
    masks = {name: np.zeros(shape, dtype=bool) for name in mask_names}
    assigned = np.zeros(shape, dtype=bool)
    for piece in pieces:
        ix = _bound_index(x_bounds, piece["x_bounds"][0])
        iy = _bound_index(y_bounds, piece["y_bounds"][0])
        nx, ny = len(piece["x_bounds"]) - 1, len(piece["y_bounds"]) - 1
        target = np.s_[iy:iy + ny, ix:ix + nx]
        if np.any(assigned[target]):
            raise ValueError("p_all spatial partitions overlap while assembling the full domain")
        for name in names:
            values = np.asarray(piece["fields"][name])
            if values.shape != (ny, nx):
                raise ValueError(f"partition field {name} has shape {values.shape}, expected {(ny, nx)}")
            fields[name][target] = values
        for name in mask_names:
            masks[name][target] = piece["masks"][name]
        assigned[target] = True
    if not np.all(assigned):
        raise ValueError(f"p_all spatial assembly contains {np.count_nonzero(~assigned)} uncovered cells")
    return {"x_bounds": x_bounds, "y_bounds": y_bounds, "fields": fields, "masks": masks}


def _plot_data(symbol, assembled):
    fields, masks = assembled["fields"], assembled["masks"]
    values = fields[symbol]
    if symbol == "temperature":
        mask = masks["mask.chemistry_valid"] & masks["mask.gas_dominated"]
        policy = "mask.chemistry_valid AND mask.gas_dominated; chemistry-clipped temperature"
    elif symbol in {"OH", "NC12H26", "O2", "phi"}:
        mask = masks["mask.valid"] & masks["mask.gas_dominated"] & np.isfinite(values)
        policy = "mask.valid AND mask.gas_dominated"
    else:
        mask = masks["mask.valid"]
        policy = "mask.valid"
    return np.where(mask, values, np.nan), policy


def _render_field(destination, selection, symbol, assembled, limits, config, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted, mask_policy = _plot_data(symbol, assembled)
    raw_values = assembled["fields"]["temperature.raw"] if symbol == "temperature" else assembled["fields"][symbol]
    title, colorbar_label, cmap_name = FIELD_STYLE[symbol]
    folder = destination / symbol
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}_t{_time_token(selection['actual_time_us'])}us.png"
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#eeeeee")
        image = axis.pcolormesh(
            assembled["x_bounds"] * 1.0e6, assembled["y_bounds"] * 1.0e6,
            np.ma.masked_invalid(plotted), cmap=cmap, vmin=limits[0], vmax=limits[1],
            shading="flat", rasterized=True,
        )
        figure.colorbar(image, ax=axis, pad=0.02).set_label(colorbar_label)
        _clean_axes(axis, f"{title}, t = {selection['actual_time_us']:.2f} us")
        figure.savefig(path, dpi=180)
    except Exception as exc:
        raise RuntimeError(f"failed to render {symbol} to {path}: {exc}") from exc
    finally:
        if figure is not None:
            plt.close(figure)
    return {
        **selection, "kind": "field", "field": symbol, "path": str(path),
        "raw_range": _range_dict(raw_values), "plotted_range": _range_dict(plotted),
        "color_limits": {"minimum": limits[0], "maximum": limits[1]},
        "mask_policy": mask_policy, "counts": counts,
        "species_equation_index": (
            config.equation_indices[f"species_density[{_actual_species(config, symbol)}]"]
            if symbol in {"OH", "NC12H26", "O2"} else None
        ),
    }


def _render_overlay(destination, selection, overlay, levels, assembled, base_limits, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, contour = overlay
    base_values, base_policy = _plot_data(base, assembled)
    contour_values, contour_policy = _plot_data(contour, assembled)
    folder = destination / f"overlay_{base}_{contour}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{base}_{contour}_t{_time_token(selection['actual_time_us'])}us.png"
    contour_range = _range(contour_values)
    drawn_levels = [
        value for value in levels
        if contour_range[0] is not None and contour_range[0] <= value <= contour_range[1]
    ]
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
        cmap = plt.get_cmap(FIELD_STYLE[base][2]).copy()
        cmap.set_bad("#eeeeee")
        image = axis.pcolormesh(
            assembled["x_bounds"] * 1.0e6, assembled["y_bounds"] * 1.0e6,
            np.ma.masked_invalid(base_values), cmap=cmap,
            vmin=base_limits[0], vmax=base_limits[1], shading="flat", rasterized=True,
        )
        figure.colorbar(image, ax=axis, pad=0.02).set_label(FIELD_STYLE[base][1])
        if drawn_levels:
            x_centers = (assembled["x_bounds"][:-1] + assembled["x_bounds"][1:]) * 0.5e6
            y_centers = (assembled["y_bounds"][:-1] + assembled["y_bounds"][1:]) * 0.5e6
            axis.contour(
                x_centers, y_centers, np.ma.masked_invalid(contour_values),
                levels=drawn_levels, colors="white", linewidths=1.0,
            )
        _clean_axes(
            axis,
            f"{FIELD_STYLE[base][0]} with {FIELD_STYLE[contour][0].lower()} contours, "
            f"t = {selection['actual_time_us']:.2f} us",
        )
        figure.savefig(path, dpi=180)
    except Exception as exc:
        raise RuntimeError(f"failed to render overlay {base}+{contour} to {path}: {exc}") from exc
    finally:
        if figure is not None:
            plt.close(figure)
    return {
        **selection, "kind": "overlay", "base_field": base, "overlay_field": contour,
        "path": str(path), "levels_requested": list(levels), "levels_drawn": drawn_levels,
        "base_color_limits": {"minimum": base_limits[0], "maximum": base_limits[1]},
        "base_mask_policy": base_policy, "overlay_mask_policy": contour_policy, "counts": counts,
    }


def _clean_axes(axis, title):
    axis.set_title(title)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_aspect("equal", adjustable="box")


def _mask_counts(assembled):
    masks, fields = assembled["masks"], assembled["fields"]
    valid = masks["mask.valid"]
    return {
        "total": int(valid.size), "valid": int(np.count_nonzero(valid)),
        "gas_dominated": int(np.count_nonzero(masks["mask.gas_dominated"])),
        "liquid_dominated": int(np.count_nonzero(masks["mask.liquid_dominated"])),
        "interface": int(np.count_nonzero(masks["mask.interface"])),
        "invalid": int(np.count_nonzero(~valid)),
        "temperature_clipped": int(np.count_nonzero(
            np.isfinite(fields["temperature.raw"]) & (fields["temperature.raw"] != fields["temperature"])
        )),
        "pressure_floored": int(np.count_nonzero(
            np.isfinite(fields["pressure.raw"]) & (fields["pressure.raw"] != fields["pressure"])
        )),
    }


def _final_limits(symbol, observed):
    low, high = observed
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"field {symbol} has no finite plotted cells in the selected batch")
    if high == low:
        delta = max(abs(low) * 1.0e-6, 1.0e-15)
        return low - delta, high + delta
    return low, high


def _range(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return (None, None) if not finite.size else (float(np.min(finite)), float(np.max(finite)))


def _range_dict(values):
    low, high = _range(values)
    return {"minimum": low, "maximum": high}


def _time_token(time_us):
    return f"{time_us:07.2f}".replace("-", "m").replace(".", "p")


def _stoich_coefficient(config):
    weights = dict(zip(config.species_names, config.molecular_weights))
    names = {
        requested: next(name for name in config.species_names if name.casefold() == requested.casefold())
        for requested in ("O2", "NC12H26")
    }
    return 18.5 * weights[names["O2"]] / weights[names["NC12H26"]]


def _actual_species(config, requested):
    matches = [name for name in config.species_names if name.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"required render species {requested!r} is absent or ambiguous")
    return matches[0]


def _bound_index(global_bounds, value):
    matches = np.flatnonzero(np.isclose(global_bounds, value, rtol=1.0e-12, atol=1.0e-15))
    if len(matches) != 1:
        raise ValueError(f"partition boundary {value} was not uniquely located in the global grid")
    return int(matches[0])


def _merge_bounds(groups):
    ordered = np.sort(np.concatenate(list(groups)))
    merged = []
    for value in ordered:
        if merged and np.isclose(merged[-1], value, rtol=1.0e-12, atol=1.0e-15):
            merged[-1] = 0.5 * (merged[-1] + float(value))
        else:
            merged.append(float(value))
    return np.asarray(merged, dtype=np.float64)


def _prepare_output(context, destination, overwrite):
    error = None
    if context.rank == 0:
        try:
            if destination.exists() and not destination.is_dir():
                raise FileExistsError(f"output path exists and is not a directory: {destination}")
            if destination.exists() and any(destination.iterdir()) and not overwrite:
                raise FileExistsError(
                    f"output directory is not empty: {destination}; use --overwrite to replace "
                    "mfc-post render outputs"
                )
            if destination.exists() and overwrite:
                for name in (*FIELD_STYLE, "manifest.json", "provenance.json"):
                    path = destination / name
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                for path in destination.glob("overlay_*"):
                    if path.is_dir():
                        shutil.rmtree(path)
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [item for item in context.comm.allgather(error) if item]
    if errors:
        raise RuntimeError(errors[0])


def _provenance(config, source_files, limits, overlay, levels):
    return {
        "source_files": sorted(source_files),
        "mechanism": {"path": config.mechanism_path, "phase": config.mechanism_phase},
        "species_mapping": [
            {
                "name": name,
                "equation_index": config.equation_indices[f"species_density[{name}]"],
                "molecular_weight_kg_per_kmol": weight,
            }
            for name, weight in zip(config.species_names, config.molecular_weights)
        ],
        "eos_inputs": [asdict(item) for item in config.eos],
        "mask_thresholds": asdict(config.thresholds),
        "transformations": _transformation_provenance(config),
        "coordinate_conversion": "MFC coordinates multiplied by 1e6 for micrometer plot axes",
        "batch_color_limits": {
            symbol: {"minimum": value[0], "maximum": value[1]} for symbol, value in limits.items()
        },
        "render_policy": {
            "clean_static_png": True,
            "temperature": "chemistry-clipped temperature; chemistry-valid AND gas-dominated",
            "species_and_phi": "gas-phase fields; valid AND gas-dominated",
            "alpha_liq": "solver-reconstructed liquid volume fraction over valid cells",
            "masked_color": "light gray; no in-image mask legend or provenance annotation",
            "overlay": (
                {"base": overlay[0], "contour": overlay[1], "levels": list(levels)}
                if overlay else None
            ),
        },
    }


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _progress(context, message):
    if context.rank == 0:
        print(f"mfc-post render: {message}", flush=True)
