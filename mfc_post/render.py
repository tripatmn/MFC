"""Clean, selected-state contour rendering from supported raw MFC output."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .execution import ExecutionContext, split_range
from .inspect import inspect_case
from .models import RunMetadata
from .process import _transformation_provenance
from .reconstruction import FieldRegistry, Model3Configuration, reconstruct_model3
from .source_selection import select_raw_source
from .sources import LustreSharedSource, PAllSource


RENDER_FIELDS = ("temperature", "OH", "NC12H26", "O2", "phi", "alpha_liq")
DEFAULT_PHI_LEVELS = (1.0,)
O2_FLOOR = 1.0e-12
LIQUID_CONTOUR_LEVEL = 0.5
LIQUID_CONTOUR_COLOR = "#00a6c8"
LIQUID_CONTOUR_WIDTH = 1.1
RENDER_MASKS = {
    "strict_gas": "legacy field-specific valid-gas mask",
    "nonliquid": "alpha_liq <= 0.5 AND isfinite(field)",
}

FIELD_STYLE = {
    "temperature": ("Temperature", "Temperature [K]", "inferno"),
    "OH": ("OH mass fraction", "OH mass fraction", "magma"),
    "HO2": ("HO2 mass fraction", "HO2 mass fraction", "magma"),
    "H2O2": ("H2O2 mass fraction", "H2O2 mass fraction", "magma"),
    "NC12H26": ("NC12H26 mass fraction", "NC12H26 mass fraction", "viridis"),
    "O2": ("O2 mass fraction", "O2 mass fraction", "viridis"),
    "CO2": ("CO2 mass fraction", "CO2 mass fraction", "viridis"),
    "H2O": ("H2O mass fraction", "H2O mass fraction", "viridis"),
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
    temperature_mask: str = "nonliquid",
    source_family: str = "auto",
    field_limits: Mapping[str, tuple[float, float]] | None = None,
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
    if temperature_mask not in RENDER_MASKS:
        raise ValueError(
            f"unknown render mask {temperature_mask!r}; choose from {tuple(RENDER_MASKS)}"
        )

    _progress(context, f"startup: case path: {root}")
    inspection = inspect_case(root, mechanism=mechanism, phase=phase)
    metadata = _metadata(inspection)
    source, report = select_raw_source(root, metadata, inspection, source_family)
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
    manual_limits = _canonical_limits(
        field_limits or {}, config, canonical_fields, canonical_overlay,
    )

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
    _progress(context, f"startup: selected source family: {report['family']}")
    _progress(context, f"startup: selected source path: {report['path']}")
    _progress(context, f"startup: saves discovered: {len(report['timeline']['saved_indices'])}")
    _progress(context, f"startup: saves selected after filters: {len(selections)}")
    _progress(context, f"startup: render mask: {temperature_mask}")
    _progress(
        context,
        "startup: manual field limits: "
        + (", ".join(
            f"{name}={bounds[0]:g}:{bounds[1]:g}"
            for name, bounds in manual_limits.items()
        ) or "none"),
    )
    _progress(context, f"startup: output directory: {destination}")
    _prepare_output(context, destination, overwrite)

    raw_fields = tuple(str(item["name"]) for item in metadata.equation_layout)
    # alpha_liq is a visual context field for every render, even when it is not
    # itself requested as a plotted field.
    load_symbols = tuple(dict.fromkeys(
        canonical_fields + (canonical_overlay or ()) + ("alpha_liq",)
    ))
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
                plotted, _ = _plot_data(symbol, assembled, temperature_mask)
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
            limits = {
                symbol: (
                    manual_limits[symbol]
                    if symbol in manual_limits else _final_limits(symbol, values)
                )
                for symbol, values in ranges.items()
            }
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
                        destination, selection, symbol, assembled, limits[symbol], config, counts,
                        temperature_mask, manual_limits,
                    ))
                if canonical_overlay:
                    frames.append(_render_overlay(
                        destination, selection, canonical_overlay, levels, assembled,
                        limits[canonical_overlay[0]], counts, temperature_mask, manual_limits,
                    ))
            except Exception as exc:
                root_error = (
                    f"worker_rank=0 source={source.family} saved_index={saved_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        render_errors = [item for item in context.comm.allgather(root_error) if item]
        if render_errors:
            raise RuntimeError("rendering failed; no manifest written:\n" + "\n".join(render_errors))

    if context.rank != 0:
        return None
    if not frames:
        raise RuntimeError("render produced zero PNG artifacts; no manifest written")
    provenance = _provenance(
        config, all_sources, limits, canonical_overlay, levels, temperature_mask, frames,
        report, metadata, selections, manual_limits,
    )
    manifest = {
        "schema_version": "mfc-post.render-clean/v1",
        "case": str(root),
        "source": {
            "family": report["family"], "path": report["path"], "layout": report["layout"],
        },
        "selection_policy": "nearest requested saves or inclusive time range, followed by stride",
        "selections": selections,
        "fields": list(canonical_fields),
        "temperature_mask": {
            "mode": temperature_mask,
            "definition": RENDER_MASKS[temperature_mask],
            "alpha_liq_threshold": LIQUID_CONTOUR_LEVEL,
        },
        "render_mask": {
            "mode": temperature_mask,
            "definition": RENDER_MASKS[temperature_mask],
            "alpha_liq_threshold": LIQUID_CONTOUR_LEVEL,
        },
        "field_limits": {
            symbol: {
                "minimum": value[0], "maximum": value[1],
                "mode": "manual" if symbol in manual_limits else "batch",
            }
            for symbol, value in limits.items()
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
    matches = [name for name in config.species_names if name.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"requested render field/species {cleaned!r} is absent or ambiguous")
    return matches[0]


def _canonical_limits(raw_limits, config, fields, overlay):
    result = {}
    color_fields = set(fields)
    if overlay:
        color_fields.add(overlay[0])
    for requested, raw_bounds in raw_limits.items():
        symbol = _canonical_symbol(str(requested), config)
        if symbol not in color_fields:
            raise ValueError(
                f"--field-limits field {requested!r} is not a requested field or overlay base"
            )
        if symbol in result:
            raise ValueError(f"--field-limits resolves to duplicate field {symbol!r}")
        try:
            low, high = (float(value) for value in raw_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--field-limits for {requested!r} requires numeric VMIN,VMAX"
            ) from exc
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(
                f"--field-limits for {requested!r} requires finite VMAX > VMIN; "
                f"got {low!r}, {high!r}"
            )
        result[symbol] = (low, high)
    return result


def _load_assembled(source, saved_index, raw_fields, symbols, config, context):
    local_pieces = []
    local_error = None
    try:
        if isinstance(source, PAllSource):
            partitions = source.partition_ids(saved_index)
            for partition in partitions[context.rank::context.size]:
                state = source.read_partition(saved_index, partition, raw_fields)
                physical = reconstruct_model3(state, config)
                local_pieces.append(_piece(state, physical, symbols, config))
            global_grid = None
        elif isinstance(source, LustreSharedSource):
            global_grid = source.read_grid(saved_index)
            cells = int(np.prod(global_grid.shape, dtype=np.int64))
            start, stop = split_range(cells, context.size, context.rank)
            state = source.read_chunk(
                saved_index, start, stop, raw_fields,
                mpi_comm=context.comm if context.mpi is not None else None,
            )
            physical = reconstruct_model3(state, config)
            local_pieces.append(_flat_piece(
                state, physical, symbols, config, start, stop,
            ))
        else:
            raise ValueError(f"unsupported render source {source.family}")
    except Exception as exc:
        local_error = (
            f"worker_rank={context.rank} source={source.family} saved_index={saved_index}: "
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
        raise RuntimeError(
            f"saved index {saved_index}: no {source.family} partitions/chunks were loaded"
        )
    assembled = (
        _assemble(pieces, symbols)
        if isinstance(source, PAllSource)
        else _assemble_flat(pieces, symbols, global_grid)
    )
    assembled["source_files"] = sorted({path for piece in pieces for path in piece["source_files"]})
    return assembled


def _piece(state, physical, symbols, config):
    if state.grid is None or not state.grid.bounds.get("x") or not state.grid.bounds.get("y"):
        raise ValueError(f"saved index {state.saved_index}: partition lacks 2D physical coordinates")
    fields, masks, source_files = _derived_piece_data(state, physical, symbols, config)
    return {
        "x_bounds": np.asarray(state.grid.bounds["x"], dtype=np.float64),
        "y_bounds": np.asarray(state.grid.bounds["y"], dtype=np.float64),
        "fields": fields, "masks": masks, "source_files": source_files,
    }


def _flat_piece(state, physical, symbols, config, start, stop):
    fields, masks, source_files = _derived_piece_data(state, physical, symbols, config)
    return {
        "start": int(start), "stop": int(stop),
        "fields": {name: np.asarray(values).reshape(-1) for name, values in fields.items()},
        "masks": {name: np.asarray(values).reshape(-1) for name, values in masks.items()},
        "source_files": source_files,
    }


def _derived_piece_data(state, physical, symbols, config):
    registry = FieldRegistry(physical, config)
    fields = {
        "temperature": np.asarray(registry.resolve("temperature").values, dtype=np.float64),
        "temperature.raw": np.asarray(registry.resolve("temperature.raw").values, dtype=np.float64),
        "pressure": np.asarray(registry.resolve("pressure").values, dtype=np.float64),
        "pressure.raw": np.asarray(registry.resolve("pressure.raw").values, dtype=np.float64),
    }
    for symbol in symbols:
        if symbol in config.species_names:
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
    return fields, masks, sorted({
            field.provenance.source_path for field in state.fields.values() if field.provenance
        })


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


def _assemble_flat(pieces, symbols, grid):
    if grid is None or not grid.bounds.get("x") or not grid.bounds.get("y"):
        raise ValueError("shared Lustre render requires global x/y boundary files")
    cell_count = int(np.prod(grid.shape, dtype=np.int64))
    canonical_shape = tuple(reversed(grid.shape))
    names = set(symbols) | {"temperature", "temperature.raw", "pressure", "pressure.raw"}
    fields = {name: np.full(cell_count, np.nan) for name in names}
    mask_names = set().union(*(piece["masks"] for piece in pieces))
    masks = {name: np.zeros(cell_count, dtype=bool) for name in mask_names}
    assigned = np.zeros(cell_count, dtype=bool)
    for piece in pieces:
        start, stop = piece["start"], piece["stop"]
        if start < 0 or stop < start or stop > cell_count:
            raise ValueError(f"invalid shared Lustre assembled range [{start}, {stop})")
        target = np.s_[start:stop]
        if np.any(assigned[target]):
            raise ValueError("shared Lustre chunks overlap while assembling the full domain")
        for name in names:
            values = np.asarray(piece["fields"][name]).reshape(-1)
            if values.size != stop - start:
                raise ValueError(
                    f"shared Lustre field {name} has {values.size} values for range [{start}, {stop})"
                )
            fields[name][target] = values
        for name in mask_names:
            masks[name][target] = np.asarray(piece["masks"][name]).reshape(-1)
        assigned[target] = True
    if not np.all(assigned):
        raise ValueError(
            f"shared Lustre assembly contains {np.count_nonzero(~assigned)} uncovered cells"
        )
    return {
        "x_bounds": np.asarray(grid.bounds["x"], dtype=np.float64),
        "y_bounds": np.asarray(grid.bounds["y"], dtype=np.float64),
        "fields": {
            name: values.reshape(canonical_shape, order="C") for name, values in fields.items()
        },
        "masks": {
            name: values.reshape(canonical_shape, order="C") for name, values in masks.items()
        },
    }


def _plot_data(symbol, assembled, temperature_mask="nonliquid"):
    fields, masks = assembled["fields"], assembled["masks"]
    values = fields[symbol]
    if symbol == "alpha_liq":
        mask = masks["mask.valid"] & np.isfinite(values)
        policy = "mask.valid AND isfinite(alpha_liq); liquid context field exemption"
    elif temperature_mask == "nonliquid":
        alpha_liq = fields["alpha_liq"]
        mask = (alpha_liq <= LIQUID_CONTOUR_LEVEL) & np.isfinite(values)
        policy = "alpha_liq <= 0.5 AND isfinite(field)"
    elif temperature_mask == "strict_gas":
        if symbol == "temperature":
            mask = masks["mask.chemistry_valid"] & masks["mask.gas_dominated"]
            policy = "mask.chemistry_valid AND mask.gas_dominated; chemistry-clipped temperature"
        else:
            mask = masks["mask.valid"] & masks["mask.gas_dominated"] & np.isfinite(values)
            policy = "mask.valid AND mask.gas_dominated AND isfinite(field)"
    else:
        raise ValueError(f"unknown render mask {temperature_mask!r}")
    return np.where(mask, values, np.nan), policy


def _field_style(symbol):
    return FIELD_STYLE.get(
        symbol, (f"{symbol} mass fraction", f"{symbol} mass fraction", "viridis"),
    )


def _render_field(
    destination, selection, symbol, assembled, limits, config, counts, temperature_mask,
    manual_limits,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted, mask_policy = _plot_data(symbol, assembled, temperature_mask)
    raw_values = assembled["fields"]["temperature.raw"] if symbol == "temperature" else assembled["fields"][symbol]
    title, colorbar_label, cmap_name = _field_style(symbol)
    output_symbol = _output_symbol(symbol, temperature_mask, manual_limits)
    folder = destination / output_symbol
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{output_symbol}_t{_time_token(selection['actual_time_us'])}us.png"
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
        _draw_liquid_underlay(axis, assembled)
        cmap = plt.get_cmap(cmap_name).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        image = axis.pcolormesh(
            assembled["x_bounds"] * 1.0e6, assembled["y_bounds"] * 1.0e6,
            np.ma.masked_invalid(plotted), cmap=cmap, vmin=limits[0], vmax=limits[1],
            shading="flat", rasterized=True,
        )
        figure.colorbar(image, ax=axis, pad=0.02).set_label(colorbar_label)
        liquid_contour_drawn = _draw_liquid_contour(axis, assembled)
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
        "color_limits": {
            "minimum": limits[0], "maximum": limits[1],
            "mode": "manual" if symbol in manual_limits else "batch",
        },
        "mask_policy": mask_policy, "counts": counts,
        "plot_cell_counts": _plot_cell_counts(plotted),
        "render_mask": temperature_mask if symbol != "alpha_liq" else "valid",
        "temperature_mask": temperature_mask if symbol == "temperature" else None,
        "liquid_context": _liquid_context_record(liquid_contour_drawn),
        "species_equation_index": (
            config.equation_indices[f"species_density[{_actual_species(config, symbol)}]"]
            if symbol in config.species_names else None
        ),
    }


def _render_overlay(
    destination, selection, overlay, levels, assembled, base_limits, counts, temperature_mask,
    manual_limits,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, contour = overlay
    base_values, base_policy = _plot_data(base, assembled, temperature_mask)
    contour_values, contour_policy = _plot_data(contour, assembled, temperature_mask)
    base_output = _output_symbol(base, temperature_mask, manual_limits)
    contour_output = _output_symbol(contour, temperature_mask, manual_limits)
    folder = destination / f"overlay_{base_output}_{contour_output}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (
        f"{base_output}_{contour_output}_t{_time_token(selection['actual_time_us'])}us.png"
    )
    contour_range = _range(contour_values)
    drawn_levels = [
        value for value in levels
        if contour_range[0] is not None and contour_range[0] <= value <= contour_range[1]
    ]
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
        _draw_liquid_underlay(axis, assembled)
        cmap = plt.get_cmap(_field_style(base)[2]).with_extremes(
            bad=(0.0, 0.0, 0.0, 0.0),
        )
        image = axis.pcolormesh(
            assembled["x_bounds"] * 1.0e6, assembled["y_bounds"] * 1.0e6,
            np.ma.masked_invalid(base_values), cmap=cmap,
            vmin=base_limits[0], vmax=base_limits[1], shading="flat", rasterized=True,
        )
        figure.colorbar(image, ax=axis, pad=0.02).set_label(_field_style(base)[1])
        if drawn_levels:
            x_centers = (assembled["x_bounds"][:-1] + assembled["x_bounds"][1:]) * 0.5e6
            y_centers = (assembled["y_bounds"][:-1] + assembled["y_bounds"][1:]) * 0.5e6
            axis.contour(
                x_centers, y_centers, np.ma.masked_invalid(contour_values),
                levels=drawn_levels, colors="white", linewidths=1.0,
            )
        liquid_contour_drawn = _draw_liquid_contour(axis, assembled)
        contour_description = (
            f"{_field_style(contour)[0].lower()} = {levels[0]:g} contour"
            if len(levels) == 1 else f"{_field_style(contour)[0].lower()} contours"
        )
        _clean_axes(
            axis,
            f"{_field_style(base)[0]} with {contour_description}, "
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
        "base_color_limits": {
            "minimum": base_limits[0], "maximum": base_limits[1],
            "mode": "manual" if base in manual_limits else "batch",
        },
        "base_mask_policy": base_policy, "overlay_mask_policy": contour_policy, "counts": counts,
        "base_plot_cell_counts": _plot_cell_counts(base_values),
        "overlay_plot_cell_counts": _plot_cell_counts(contour_values),
        "render_mask": temperature_mask,
        "temperature_mask": temperature_mask if "temperature" in overlay else None,
        "liquid_context": _liquid_context_record(liquid_contour_drawn),
    }


def _output_symbol(symbol, temperature_mask, manual_limits):
    name = f"temperature_{temperature_mask}" if symbol == "temperature" else symbol
    if symbol in manual_limits:
        low, high = manual_limits[symbol]
        prefix = "T" if symbol == "temperature" else "L"
        name += f"_{prefix}{_limit_token(low)}_{_limit_token(high)}"
    return name


def _limit_token(value):
    return f"{value:g}".replace("+", "").replace("-", "m").replace(".", "p")


def _plot_cell_counts(plotted):
    plotted_count = int(np.count_nonzero(np.isfinite(plotted)))
    total = int(np.asarray(plotted).size)
    return {"total": total, "plotted": plotted_count, "masked": total - plotted_count}


def _draw_liquid_underlay(axis, assembled):
    """Show reconstructed liquid fraction wherever the gas field is masked."""
    import matplotlib.colors as colors

    alpha = np.asarray(assembled["fields"]["alpha_liq"], dtype=np.float64)
    valid = np.asarray(assembled["masks"]["mask.valid"], dtype=bool)
    liquid_cmap = colors.LinearSegmentedColormap.from_list(
        "mfc_post_liquid_context", ("#f4f9fb", "#9bd8e5"),
    ).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    axis.pcolormesh(
        assembled["x_bounds"] * 1.0e6, assembled["y_bounds"] * 1.0e6,
        np.ma.masked_invalid(np.where(valid, alpha, np.nan)),
        cmap=liquid_cmap, vmin=0.0, vmax=1.0, shading="flat", rasterized=True,
    )


def _draw_liquid_contour(axis, assembled):
    alpha = np.asarray(assembled["fields"]["alpha_liq"], dtype=np.float64)
    valid = np.asarray(assembled["masks"]["mask.valid"], dtype=bool)
    alpha = np.where(valid, alpha, np.nan)
    alpha_range = _range(alpha)
    drawn = (
        alpha_range[0] is not None
        and alpha_range[0] <= LIQUID_CONTOUR_LEVEL <= alpha_range[1]
        and alpha_range[0] < alpha_range[1]
    )
    if drawn:
        x_centers = (assembled["x_bounds"][:-1] + assembled["x_bounds"][1:]) * 0.5e6
        y_centers = (assembled["y_bounds"][:-1] + assembled["y_bounds"][1:]) * 0.5e6
        axis.contour(
            x_centers, y_centers, np.ma.masked_invalid(alpha),
            levels=(LIQUID_CONTOUR_LEVEL,), colors=LIQUID_CONTOUR_COLOR,
            linewidths=LIQUID_CONTOUR_WIDTH,
        )
    return drawn


def _liquid_context_record(contour_drawn):
    record = {
        "underlay": "reconstructed alpha_liq on valid cells beneath masked field values",
        "contour": {
            "field": "alpha_liq", "level": LIQUID_CONTOUR_LEVEL,
            "color": LIQUID_CONTOUR_COLOR, "linewidth": LIQUID_CONTOUR_WIDTH,
        },
    }
    if contour_drawn is not None:
        record["contour"]["drawn"] = bool(contour_drawn)
    return record


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
                for path in destination.glob("temperature_*"):
                    if path.is_dir():
                        shutil.rmtree(path)
                for name in FIELD_STYLE:
                    for path in destination.glob(f"{name}_L*"):
                        if path.is_dir():
                            shutil.rmtree(path)
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [item for item in context.comm.allgather(error) if item]
    if errors:
        raise RuntimeError(errors[0])


def _provenance(
    config, source_files, limits, overlay, levels, temperature_mask, frames,
    report, metadata, selections, manual_limits,
):
    return {
        "source_family": report["family"], "source_path": report["path"],
        "source": {
            "family": report["family"], "path": report["path"],
            "files": sorted(source_files),
        },
        "timeline": {
            "time_basis": report["timeline"]["time_basis"],
            "t_save": metadata.parameters.get("t_save"),
            "records": [
                {
                    "saved_index": item["saved_index"],
                    "physical_time": item["actual_time_us"] * 1.0e-6,
                }
                for item in selections
            ],
            "warnings": report["timeline"].get("warnings", []),
        },
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
            symbol: {
                "minimum": value[0], "maximum": value[1],
                "mode": "manual" if symbol in manual_limits else "batch",
            }
            for symbol, value in limits.items()
        },
        "manual_field_limits": {
            symbol: {"minimum": value[0], "maximum": value[1]}
            for symbol, value in manual_limits.items()
        },
        "render_policy": {
            "clean_static_png": True,
            "temperature_mask": {
                "mode": temperature_mask,
                "definition": RENDER_MASKS[temperature_mask],
                "alpha_liq_threshold": LIQUID_CONTOUR_LEVEL,
                "temperature_field": "chemistry-clipped temperature",
                "cell_counts": _temperature_cell_counts(frames),
            },
            "scalar_mask": {
                "mode": temperature_mask,
                "definition": RENDER_MASKS[temperature_mask],
                "alpha_liq_threshold": LIQUID_CONTOUR_LEVEL,
                "cell_counts": _scalar_cell_counts(frames),
                "alpha_liq_exemption": "alpha_liq itself uses mask.valid AND finite values",
            },
            "alpha_liq": "solver-reconstructed liquid volume fraction over valid cells",
            "masked_regions": (
                "transparent field pixels reveal a pale alpha_liq underlay; invalid cells remain "
                "transparent; no in-image mask legend or provenance annotation"
            ),
            "liquid_context": _liquid_context_record(None),
            "overlay": (
                {"base": overlay[0], "contour": overlay[1], "levels": list(levels)}
                if overlay else None
            ),
        },
    }


def _temperature_cell_counts(frames):
    records = []
    for frame in frames:
        if frame.get("field") == "temperature":
            counts, role = frame["plot_cell_counts"], "field"
        elif frame.get("base_field") == "temperature":
            counts, role = frame["base_plot_cell_counts"], "overlay_base"
        elif frame.get("overlay_field") == "temperature":
            counts, role = frame["overlay_plot_cell_counts"], "overlay_contour"
        else:
            continue
        records.append({
            "saved_index": frame["saved_index"], "actual_time_us": frame["actual_time_us"],
            "role": role, **counts,
        })
    return records


def _scalar_cell_counts(frames):
    records = []
    for frame in frames:
        if frame["kind"] == "field":
            entries = ((frame["field"], "field", frame["plot_cell_counts"]),)
        else:
            entries = (
                (frame["base_field"], "overlay_base", frame["base_plot_cell_counts"]),
                (frame["overlay_field"], "overlay_contour", frame["overlay_plot_cell_counts"]),
            )
        for field, role, counts in entries:
            records.append({
                "saved_index": frame["saved_index"],
                "actual_time_us": frame["actual_time_us"],
                "field": field, "role": role, **counts,
            })
    return records


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _progress(context, message):
    if context.rank == 0:
        print(f"mfc-post render: {message}", flush=True)
