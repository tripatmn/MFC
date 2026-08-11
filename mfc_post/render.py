"""Selected-state, full-domain rendering without full-history scalar work."""

from __future__ import annotations

import csv
import json
import os
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


RENDER_FIELDS = (
    "temperature", "Y[NC12H26]", "Y[OH]", "Y[CO2]", "Y[H2O]",
)


def render_case(
    case_path: str | Path,
    selected_times_us: Iterable[float],
    fields: Iterable[str] = RENDER_FIELDS,
    out_dir: str | Path | None = None,
    execution: str = "auto",
    mechanism: str | Path | None = None,
    phase: str | None = None,
    no_zoom: bool = False,
    no_mp4: bool = False,
    skip_scalars: bool = False,
    skip_trends: bool = False,
) -> dict[str, Any] | None:
    context = ExecutionContext.create(execution)
    root = Path(case_path).expanduser().resolve()
    requested_times = tuple(float(value) for value in selected_times_us)
    requested_fields = tuple(str(value).strip() for value in fields if str(value).strip())
    if not requested_times or any(not np.isfinite(value) for value in requested_times):
        raise ValueError("--selected-times-us requires one or more finite values")
    if not requested_fields:
        raise ValueError("--fields requires at least one symbolic field")
    if len(set(requested_fields)) != len(requested_fields):
        raise ValueError("--fields contains duplicate symbolic fields")
    unsupported = [
        name for name in requested_fields
        if name != "temperature" and not (name.startswith("Y[") and name.endswith("]"))
    ]
    if unsupported:
        raise ValueError(f"unsupported render fields: {unsupported}; use temperature or Y[SPECIES]")
    destination = Path(out_dir).expanduser().resolve() if out_dir else root / "mfc_post_render"

    _progress(context, f"discovery: inspecting {root}")
    inspection = inspect_case(root, mechanism=mechanism, phase=phase)
    metadata = _metadata(inspection)
    report = next((item for item in inspection["sources"] if item["family"] == "p_all"), None)
    if report is None or not report["timeline"]["saved_indices"]:
        raise ValueError(f"selected-state rendering requires raw p_all saves; none were found in {root}")
    config = Model3Configuration.from_metadata(metadata)
    if metadata.dimensions != 2:
        raise ValueError(f"PNG field rendering currently requires a 2D case, got {metadata.dimensions}D")
    source = PAllSource(root, metadata)
    selections = _select_times(report["timeline"], requested_times)
    requested_fields = tuple(_canonical_symbol(symbol, config) for symbol in requested_fields)
    if len(set(requested_fields)) != len(requested_fields):
        raise ValueError("--fields resolves to duplicate canonical fields")
    _progress(
        context,
        "time selection: " + ", ".join(
            f"{item['requested_time_us']:.9g} us -> save {item['saved_index']} "
            f"at {item['actual_time_us']:.9g} us" for item in selections
        ),
    )

    output_error = None
    if context.rank == 0:
        try:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {destination}; choose a distinct --out-dir "
                    "for each serial/MPI run"
                )
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
    errors = [item for item in context.comm.allgather(output_error) if item]
    if errors:
        raise RuntimeError(errors[0])

    raw_fields = tuple(str(item["name"]) for item in metadata.equation_layout)
    frames: list[dict[str, Any]] = []
    all_sources: set[str] = set()
    for selection in selections:
        saved_index = int(selection["saved_index"])
        _progress(context, f"field derivation: saved index {saved_index}")
        local_pieces: list[dict[str, Any]] = []
        local_error = None
        try:
            partitions = source.partition_ids(saved_index)
            for partition in partitions[context.rank::context.size]:
                state = source.read_partition(saved_index, partition, raw_fields)
                physical = reconstruct_model3(state, config)
                registry = FieldRegistry(physical, config)
                local_pieces.append(_piece(state, physical, registry, requested_fields))
        except Exception as exc:
            local_error = (
                f"worker_rank={context.rank} source=p_all saved_index={saved_index}: "
                f"{type(exc).__name__}: {exc}"
            )
        state_errors = [item for item in context.comm.allgather(local_error) if item]
        if state_errors:
            raise RuntimeError("rendering failed; no manifest written:\n" + "\n".join(sorted(state_errors)))
        gathered = context.comm.gather(local_pieces, root=0)
        root_error = None
        if context.rank == 0:
            try:
                pieces = [piece for group in gathered for piece in group]
                if not pieces:
                    raise RuntimeError(f"saved index {saved_index}: no p_all partitions were loaded")
                assembled = _assemble(pieces, requested_fields)
                all_sources.update(path for piece in pieces for path in piece["source_files"])
                counts = _mask_counts(assembled)
                for symbol in requested_fields:
                    _progress(context, f"rendering: saved index {saved_index}, field {symbol}")
                    frame = _render_field(destination, selection, symbol, assembled, config, counts)
                    frames.append(frame)
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
    manifest = {
        "schema_version": "mfc-post.render/v1",
        "case": str(root),
        "source": {"family": "p_all", "path": report["path"], "layout": report["layout"]},
        "selection_policy": "nearest available physical time; ties choose earlier physical time then saved index",
        "selections": selections,
        "frames": frames,
        "options": {
            "full_domain": True,
            "no_zoom": bool(no_zoom), "no_mp4": bool(no_mp4),
            "skip_scalars": bool(skip_scalars), "skip_trends": bool(skip_trends),
            "scalar_history_computed": False, "trends_computed": False, "mp4_computed": False,
            "execution": {"mode": context.mode, "mpi_size": context.size},
        },
        "provenance": _provenance(config, all_sources),
        "warnings": inspection["warnings"] + report["warnings"],
    }
    _atomic_json(destination / "manifest.json", manifest)
    _atomic_json(destination / "provenance.json", manifest["provenance"])
    _write_manifest_csv(destination / "frames.csv", frames)
    _progress(context, f"completion: rendered {len(frames)} PNG(s) in {destination}")
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


def _select_times(timeline: dict[str, Any], requested_times_us: tuple[float, ...]) -> list[dict[str, Any]]:
    available = [
        (int(index), float(time) * 1.0e6)
        for index, time in zip(timeline["saved_indices"], timeline["physical_times"])
        if time is not None and np.isfinite(time)
    ]
    if not available:
        raise ValueError("p_all timeline has no determinable physical times")
    result = []
    for ordinal, requested in enumerate(requested_times_us, 1):
        index, actual = min(available, key=lambda item: (abs(item[1] - requested), item[1], item[0]))
        result.append({
            "request_ordinal": ordinal,
            "requested_time_us": requested, "saved_index": index, "actual_time_us": actual,
            "time_error_us": actual - requested,
        })
    return result


def _canonical_symbol(symbol: str, config: Model3Configuration) -> str:
    if symbol == "temperature":
        return symbol
    requested = symbol[2:-1] if symbol.startswith("Y[") and symbol.endswith("]") else None
    matches = [name for name in config.species_names if requested and name.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"requested render species {requested or symbol!r} is absent or ambiguous")
    return f"Y[{matches[0]}]"


def _piece(state, physical, registry, symbols: tuple[str, ...]) -> dict[str, Any]:
    if state.grid is None or not state.grid.bounds.get("x") or not state.grid.bounds.get("y"):
        raise ValueError(f"saved index {state.saved_index}: partition lacks 2D physical coordinates")
    fields: dict[str, np.ndarray] = {}
    for symbol in symbols:
        fields[symbol] = np.asarray(registry.resolve(symbol).values, dtype=np.float64)
    fields["temperature.raw"] = np.asarray(registry.resolve("temperature.raw").values, dtype=np.float64)
    fields["pressure"] = np.asarray(registry.resolve("pressure").values, dtype=np.float64)
    fields["pressure.raw"] = np.asarray(registry.resolve("pressure.raw").values, dtype=np.float64)
    masks = {
        name: np.asarray(field.values, dtype=bool)
        for name, field in physical.masks.items()
    }
    return {
        "x_bounds": np.asarray(state.grid.bounds["x"], dtype=np.float64),
        "y_bounds": np.asarray(state.grid.bounds["y"], dtype=np.float64),
        "fields": fields, "masks": masks,
        "source_files": sorted({
            field.provenance.source_path for field in state.fields.values() if field.provenance
        }),
    }


def _assemble(pieces: list[dict[str, Any]], symbols: tuple[str, ...]) -> dict[str, Any]:
    x_bounds = _merge_bounds(piece["x_bounds"] for piece in pieces)
    y_bounds = _merge_bounds(piece["y_bounds"] for piece in pieces)
    shape = (len(y_bounds) - 1, len(x_bounds) - 1)
    names = set(symbols) | {"temperature.raw", "pressure", "pressure.raw"}
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


def _bound_index(global_bounds: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(np.isclose(global_bounds, value, rtol=1.0e-12, atol=1.0e-15))
    if len(matches) != 1:
        raise ValueError(f"partition boundary {value} was not uniquely located in the global grid")
    return int(matches[0])


def _merge_bounds(groups: Iterable[np.ndarray]) -> np.ndarray:
    ordered = np.sort(np.concatenate(list(groups)))
    merged: list[float] = []
    for value in ordered:
        if merged and np.isclose(merged[-1], value, rtol=1.0e-12, atol=1.0e-15):
            merged[-1] = 0.5 * (merged[-1] + float(value))
        else:
            merged.append(float(value))
    return np.asarray(merged, dtype=np.float64)


def _mask_counts(assembled: dict[str, Any]) -> dict[str, int]:
    masks, fields = assembled["masks"], assembled["fields"]
    valid = masks["mask.valid"]
    raw_t, plotted_t = fields["temperature.raw"], fields["temperature"]
    raw_p, plotted_p = fields["pressure.raw"], fields["pressure"]
    return {
        "total": int(valid.size),
        "valid": int(np.count_nonzero(valid)),
        "gas_dominated": int(np.count_nonzero(masks["mask.gas_dominated"])),
        "liquid_dominated": int(np.count_nonzero(masks["mask.liquid_dominated"])),
        "interface": int(np.count_nonzero(masks["mask.interface"])),
        "invalid": int(np.count_nonzero(~valid)),
        "thermodynamically_invalid": int(np.count_nonzero(masks["mask.thermodynamically_invalid"])),
        "chemistry_valid": int(np.count_nonzero(masks["mask.chemistry_valid"])),
        "temperature_clipped": int(np.count_nonzero(
            np.isfinite(raw_t) & np.isfinite(plotted_t) & (raw_t != plotted_t)
        )),
        "pressure_floored": int(np.count_nonzero(
            np.isfinite(raw_p) & np.isfinite(plotted_p) & (raw_p != plotted_p)
        )),
    }


def _render_field(destination, selection, symbol, assembled, config, counts) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fields, masks = assembled["fields"], assembled["masks"]
    values = fields[symbol]
    if symbol == "temperature":
        raw_values = fields["temperature.raw"]
        plot_mask = masks["mask.chemistry_valid"] & masks["mask.gas_dominated"]
        mask_policy = "mask.chemistry_valid AND mask.gas_dominated; plotted temperature is chemistry-clipped"
        definition = "Chemistry temperature clipped to [chem_T_min, chem_T_max]; raw q_T range reported separately"
        units, cmap = "K", "inferno"
    else:
        raw_values = values
        plot_mask = masks["mask.valid"] & masks["mask.gas_dominated"]
        species = symbol[2:-1]
        mask_policy = "mask.valid AND mask.gas_dominated"
        definition = f"Gas-phase mass fraction Y[{species}], not conservative rhoY[{species}]"
        units, cmap = "1", "viridis"
    plotted = np.where(plot_mask, values, np.nan)
    raw_range = _range(raw_values)
    plotted_range = _range(plotted)
    if plotted_range[0] is None:
        raise ValueError(f"saved index {selection['saved_index']}: {symbol} has no finite plotted cells")
    vmin, vmax = plotted_range
    if vmax == vmin:
        delta = max(abs(vmin) * 1.0e-6, 1.0e-15)
        vmin, vmax = vmin - delta, vmax + delta

    background = np.zeros(values.shape, dtype=np.int8)
    background[masks["mask.interface"]] = 1
    background[masks["mask.liquid_dominated"]] = 2
    background[~masks["mask.valid"]] = 3
    figure, axis = plt.subplots(figsize=(12.0, 4.2), constrained_layout=True)
    axis.pcolormesh(
        assembled["x_bounds"], assembled["y_bounds"], background,
        cmap=ListedColormap(["white", "#f2d18b", "#9db7c9", "#666666"]),
        vmin=-0.5, vmax=3.5, shading="flat", rasterized=True,
    )
    image = axis.pcolormesh(
        assembled["x_bounds"], assembled["y_bounds"], np.ma.masked_invalid(plotted),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(f"{symbol} [{units}]")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{symbol} — actual t={selection['actual_time_us']:.6g} μs "
        f"(requested {selection['requested_time_us']:.6g} μs, save {selection['saved_index']})"
    )
    axis.text(
        0.01, -0.19, f"Definition: {definition}\nMask: {mask_policy}",
        transform=axis.transAxes, fontsize=8, va="top",
    )
    axis.legend(
        handles=[
            Patch(facecolor="#f2d18b", label="interface"),
            Patch(facecolor="#9db7c9", label="liquid-dominated"),
            Patch(facecolor="#666666", label="invalid"),
        ],
        loc="upper right", fontsize=8, framealpha=0.9,
    )
    filename = (
        f"request_{selection['request_ordinal']:03d}_{_safe_name(symbol)}_"
        f"requested_{selection['requested_time_us']:.9g}us_"
        f"save_{selection['saved_index']:06d}_actual_{selection['actual_time_us']:.6g}us.png"
    )
    path = destination / filename
    figure.savefig(path, dpi=180, transparent=False)
    plt.close(figure)
    return {
        **selection, "field": symbol, "path": str(path), "units": units,
        "definition": definition, "mask_policy": mask_policy,
        "raw_range_basis": (
            "unclipped reconstructed temperature.raw over all cells"
            if symbol == "temperature" else "solver-bounded gas mass fraction over all cells before render masking"
        ),
        "raw_range": {"minimum": raw_range[0], "maximum": raw_range[1]},
        "plotted_range": {"minimum": plotted_range[0], "maximum": plotted_range[1]},
        "color_limits": {"minimum": vmin, "maximum": vmax},
        "counts": counts,
        "species_equation_index": (
            config.equation_indices[f"species_density[{symbol[2:-1]}]"]
            if symbol.startswith("Y[") else None
        ),
    }


def _range(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def _safe_name(symbol: str) -> str:
    return symbol.replace("[", "_").replace("]", "").replace(".", "_")


def _provenance(config: Model3Configuration, source_files: set[str]) -> dict[str, Any]:
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
        "coordinate_units": "m (audited SI chemistry-case convention; the MFC flow solver itself is unit-agnostic)",
        "render_policy": {
            "full_domain": True,
            "temperature": "chemistry-clipped temperature, masked by chemistry-valid AND gas-dominated",
            "species": "solver-bounded gas Y_k, masked by valid AND gas-dominated",
            "background": "interface=gold, liquid-dominated=blue-gray, invalid=dark gray",
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_manifest_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    rows = []
    for frame in frames:
        rows.append({
            "requested_time_us": frame["requested_time_us"], "actual_time_us": frame["actual_time_us"],
            "saved_index": frame["saved_index"], "field": frame["field"], "path": frame["path"],
            "raw_min": frame["raw_range"]["minimum"], "raw_max": frame["raw_range"]["maximum"],
            "plotted_min": frame["plotted_range"]["minimum"], "plotted_max": frame["plotted_range"]["maximum"],
            **frame["counts"],
        })
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _progress(context: ExecutionContext, message: str) -> None:
    if context.rank == 0:
        print(f"mfc-post render: {message}", flush=True)
