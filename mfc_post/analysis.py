"""MPI-compatible scalar histories from partition-local Model-3 states."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .execution import ExecutionContext
from .heat_release import CHECK_ATOL_W_M3, CHECK_RTOL, CanteraHeatRelease
from .inspect import inspect_case
from .config import load_parameters
from .mechanism import load_cantera_mechanism
from .models import RunMetadata, State
from .process import _load_local, _partition_plan, _transformation_provenance
from .reconstruction import Model3Configuration, reconstruct_model3
from .source_selection import select_raw_source


INTEGRATED_SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO", "CO2", "H2O")
PRESENTATION_SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO", "CO2", "H2O")
MAX_SPECIES = ("NC12H26", "OH", "CO2", "H2O")
O2_FLOOR = 1.0e-12
COMBUSTIBLE_PHI = (0.5, 2.0)
NEAR_STOICH_PHI = (0.8, 1.2)
HOT_THRESHOLDS = (1200.0, 1500.0)


@dataclass
class ScalarPartial:
    sums: dict[str, float] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    quality_maxima: dict[str, float] = field(default_factory=dict)
    source_files: set[str] = field(default_factory=set)
    heat_release_maxima: list[tuple[float, float, float]] = field(default_factory=list)


def analyze_case(
    case_path: str | Path,
    selected_times_us: Iterable[float] | None = None,
    time_range_us: tuple[float, float] | None = None,
    stride: int = 1,
    out_dir: str | Path | None = None,
    execution: str = "auto",
    mechanism: str | Path | None = None,
    phase: str | None = None,
    overwrite: bool = False,
    source_family: str = "auto",
    compute_heat_release: str | None = None,
) -> dict[str, Any] | None:
    context = ExecutionContext.create(execution)
    root = Path(case_path).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve() if out_dir else root / "mfc_post_analysis"
    if stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if selected_times_us is not None and time_range_us is not None:
        raise ValueError("--selected-times-us and --time-range-us are mutually exclusive")
    if compute_heat_release not in {None, "cantera"}:
        raise ValueError("--compute-heat-release supports only 'cantera'")

    _progress(context, f"startup: case path: {root}")
    loaded_mechanism = None
    if compute_heat_release == "cantera":
        params, _ = load_parameters(root)
        loaded_mechanism = load_cantera_mechanism(root, params, mechanism, phase)
        _progress(
            context,
            "startup: Cantera mechanism: "
            f"{loaded_mechanism.metadata.path}, phase={loaded_mechanism.metadata.phase}, "
            f"species={len(loaded_mechanism.metadata.species_names)}",
        )
    inspection = inspect_case(
        root, mechanism=mechanism, phase=phase,
        resolved_mechanism=loaded_mechanism.metadata if loaded_mechanism else None,
    )
    metadata = _metadata(inspection)
    source, report = select_raw_source(root, metadata, inspection, source_family)
    discovered_count = len(report["timeline"]["saved_indices"])
    _progress(context, f"startup: selected source family: {report['family']}")
    _progress(context, f"startup: selected source path: {report['path']}")
    _progress(context, f"startup: saves discovered: {discovered_count}")
    config = Model3Configuration.from_metadata(metadata)
    _validate_species(config, metadata)
    if compute_heat_release and metadata.dimensions != 2:
        raise ValueError("Cantera heat-release analysis currently requires a 2D case")
    heat_release = (
        CanteraHeatRelease(config, loaded_mechanism.gas)
        if loaded_mechanism is not None else None
    )
    if heat_release is not None:
        _progress(context, "startup: heat release: Cantera TDY evaluation enabled")
    indices, selection = _select_indices(
        report["timeline"], selected_times_us, time_range_us, stride
    )
    _progress(context, f"startup: saves selected after filters: {len(indices)}")
    _progress(context, f"startup: output directory: {destination}")
    _prepare_output(context, destination, overwrite)

    strategy = context.strategy(len(indices))
    raw_fields = tuple(str(item["name"]) for item in metadata.equation_layout)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if strategy == "state":
        state_comm = context.mpi.COMM_SELF
        for local_position, saved_index in enumerate(indices[context.rank::context.size]):
            try:
                _save_progress(
                    context, context.rank + local_position * context.size + 1,
                    len(indices), saved_index,
                    state_parallel=True,
                    diagnostic=compute_heat_release,
                )
                states = _load_local(source, saved_index, raw_fields, "serial", context)
                partial = _accumulate(states, config, heat_release)
                reduced = _reduce(
                    partial, state_comm, context.mpi, metadata.dimensions or 1,
                    heat_release_enabled=heat_release is not None,
                )
                rows.append(_row(saved_index, report["timeline"], reduced))
            except Exception as exc:
                errors.append(_error(context, report["family"], saved_index, exc))
        all_errors = [item for group in context.comm.allgather(errors) for item in group]
        if all_errors:
            raise RuntimeError("analysis failed; no output written:\n" + "\n".join(sorted(all_errors)))
        gathered = context.comm.gather(rows, root=0)
        if context.rank == 0:
            rows = [row for group in gathered for row in group]
    else:
        for position, saved_index in enumerate(indices, 1):
            _save_progress(
                context, position, len(indices), saved_index,
                state_parallel=False,
                diagnostic=compute_heat_release,
            )
            local_error = None
            partial = None
            try:
                states = _load_local(source, saved_index, raw_fields, strategy, context)
                partial = _accumulate(states, config, heat_release)
            except Exception as exc:
                local_error = _error(context, report["family"], saved_index, exc)
            state_errors = [item for item in context.comm.allgather(local_error) if item]
            if state_errors:
                raise RuntimeError("analysis failed; no output written:\n" + "\n".join(sorted(state_errors)))
            reduced = _reduce(
                partial, context.comm, context.mpi, metadata.dimensions or 1,
                heat_release_enabled=heat_release is not None,
            )
            if context.rank == 0:
                rows.append(_row(saved_index, report["timeline"], reduced))

    if context.rank != 0:
        return None
    if not rows:
        raise RuntimeError("analysis produced zero scalar records; no output written")
    rows.sort(key=lambda row: (row["physical_time_s"], row["saved_index"]))
    _add_presentation_columns(rows)
    source_files = sorted(set().union(*(set(row.pop("source_files")) for row in rows)))
    quality = [
        {"saved_index": row["saved_index"], "time_us": row["time_us"], **row.pop("quality")}
        for row in rows
    ]
    if compute_heat_release == "cantera":
        mismatches = sum(item.get("heat_release_check_mismatch_cells", 0) for item in quality)
        sign_mismatches = sum(
            item.get("heat_release_check_sign_mismatch_cells", 0) for item in quality
        )
        if mismatches or sign_mismatches:
            _progress(
                context,
                "WARNING: Cantera HRR verification disagreement: "
                f"magnitude_cells={mismatches}, sign_cells={sign_mismatches}; "
                "see quality.json",
            )
    provenance = _provenance(
        root, report, config, context, strategy, indices, selection, stride,
        source_files, source, metadata, rows, heat_release,
    )
    _write_csv(
        destination / "scalar_timeseries.csv", rows,
        include_heat_release=compute_heat_release == "cantera",
    )
    _atomic_json(destination / "quality.json", {
        "schema_version": "mfc-post.quality/v1",
        "source_family": report["family"],
        "quality": quality,
    })
    _atomic_json(destination / "provenance.json", provenance)
    result = {
        "output_directory": str(destination), "rows": rows,
        "quality": quality, "provenance": provenance,
    }
    _progress(context, f"completion: row count: {len(rows)}")
    _progress(context, f"completion: scalar_timeseries.csv: {destination / 'scalar_timeseries.csv'}")
    _progress(context, f"completion: quality.json: {destination / 'quality.json'}")
    _progress(context, f"completion: provenance.json: {destination / 'provenance.json'}")
    return result


def _metadata(inspection: dict[str, Any]) -> RunMetadata:
    raw = inspection["run_metadata"]
    return RunMetadata(
        case_path=raw["case_path"], parameters=raw["parameters"], dimensions=raw["dimensions"],
        grid_shape=tuple(raw["grid_shape"]) if raw["grid_shape"] else None,
        equation_layout=tuple(raw["equation_layout"]), species_names=tuple(raw["species_names"]),
        missing=tuple(raw["missing"]), warnings=tuple(raw["warnings"]),
    )


def _validate_species(config: Model3Configuration, metadata: RunMetadata) -> None:
    required = set(INTEGRATED_SPECIES) | set(MAX_SPECIES) | {"O2"}
    by_casefold: dict[str, list[str]] = {}
    for name in config.species_names:
        by_casefold.setdefault(name.casefold(), []).append(name)
    invalid = [name for name in sorted(required) if len(by_casefold.get(name.casefold(), [])) != 1]
    if invalid:
        raise ValueError(f"required analysis species are absent or ambiguous: {invalid}")
    fuel_id = int(metadata.parameters.get("fuel_species_id", 0))
    if not 1 <= fuel_id <= len(config.species_names):
        raise ValueError(
            "fuel_species_id is required to identify the designated liquid as NC12H26"
        )
    if config.species_names[fuel_id - 1].casefold() != "nc12h26":
        raise ValueError(
            "fuel_species_id does not resolve to NC12H26; liquid and total dodecane "
            "inventories would be ambiguous"
        )


def _select_indices(timeline, selected_times_us, time_range_us, stride):
    available = [
        (int(index), float(time) * 1.0e6)
        for index, time in zip(timeline["saved_indices"], timeline["physical_times"])
        if time is not None and np.isfinite(time)
    ]
    if not available:
        raise ValueError("p_all timeline has no determinable physical times")
    selection: dict[str, Any]
    if selected_times_us is not None:
        requested = tuple(float(value) for value in selected_times_us)
        if not requested or any(not np.isfinite(value) for value in requested):
            raise ValueError("--selected-times-us requires finite values")
        mappings = []
        chosen: set[int] = set()
        for value in requested:
            index, actual = min(available, key=lambda item: (abs(item[1] - value), item[1], item[0]))
            mappings.append({
                "requested_time_us": value, "saved_index": index,
                "actual_time_us": actual, "time_error_us": actual - value,
            })
            chosen.add(index)
        candidates = [index for index, _ in available if index in chosen]
        selection = {"mode": "nearest_selected_times", "mappings": mappings}
    elif time_range_us is not None:
        start, stop = (float(value) for value in time_range_us)
        if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
            raise ValueError("--time-range-us must be finite START,STOP with STOP >= START")
        candidates = [index for index, time in available if start <= time <= stop]
        selection = {"mode": "inclusive_time_range", "start_us": start, "stop_us": stop}
    else:
        candidates = [index for index, _ in available]
        selection = {"mode": "all_available_saves"}
    indices = candidates[::stride]
    if not indices:
        raise ValueError("time selection and stride produced no saved states")
    selection["stride"] = stride
    selection["saved_indices"] = indices
    return indices, selection


def _prepare_output(context: ExecutionContext, destination: Path, overwrite: bool) -> None:
    error = None
    if context.rank == 0:
        try:
            if destination.exists() and not destination.is_dir():
                raise FileExistsError(f"output path exists and is not a directory: {destination}")
            if destination.exists() and any(destination.iterdir()) and not overwrite:
                raise FileExistsError(
                    f"output directory is not empty: {destination}; use --overwrite to replace "
                    "mfc-post analyze outputs"
                )
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [item for item in context.comm.allgather(error) if item]
    if errors:
        raise RuntimeError(errors[0])


def _accumulate(
    states: Iterable[State], config: Model3Configuration,
    heat_release: CanteraHeatRelease | None = None,
) -> ScalarPartial:
    result = ScalarPartial()
    stoich = _stoich_coefficient(config)
    species_names = _species_lookup(config)
    for state in states:
        if state.grid is None or state.grid.cell_measures is None:
            raise ValueError(f"saved index {state.saved_index}: cell measures are unavailable")
        weights = np.asarray(state.grid.cell_measures, dtype=np.float64).reshape(-1)
        physical = reconstruct_model3(state, config)
        fields = {name: np.asarray(field.values, dtype=np.float64).reshape(-1) for name, field in physical.fields.items()}
        masks = {name: np.asarray(field.values, dtype=bool).reshape(-1) for name, field in physical.masks.items()}
        if any(values.size != weights.size for values in fields.values()):
            raise ValueError(f"saved index {state.saved_index}: reconstructed field/grid size mismatch")
        finite_weights = np.isfinite(weights) & (weights > 0.0)
        valid_gas = masks["mask.chemistry_valid"] & masks["mask.gas_dominated"] & finite_weights
        if heat_release is not None:
            hrr = heat_release.evaluate(state, physical)
            _add(result.sums, "integrated_heat_release_rate_net_W_per_m", hrr.net)
            _add(result.sums, "integrated_heat_release_rate_positive_W_per_m", hrr.positive)
            _add(result.sums, "integrated_heat_release_rate_negative_W_per_m", hrr.negative)
            _add(result.sums, "area_positive_heat_release_rate", hrr.positive_area)
            result.counts["heat_release_evaluated_cells"] = (
                result.counts.get("heat_release_evaluated_cells", 0) + hrr.evaluated_cells
            )
            result.counts["heat_release_check_mismatch_cells"] = (
                result.counts.get("heat_release_check_mismatch_cells", 0)
                + hrr.check_mismatch_cells
            )
            result.counts["heat_release_check_sign_mismatch_cells"] = (
                result.counts.get("heat_release_check_sign_mismatch_cells", 0)
                + hrr.check_sign_mismatch_cells
            )
            result.quality_maxima["heat_release_check_max_abs_error_W_m3"] = max(
                result.quality_maxima.get("heat_release_check_max_abs_error_W_m3", 0.0),
                hrr.check_max_abs_error_W_m3,
            )
            result.quality_maxima["heat_release_check_max_relative_error"] = max(
                result.quality_maxima.get("heat_release_check_max_relative_error", 0.0),
                hrr.check_max_relative_error,
            )
            if hrr.maximum is not None:
                result.heat_release_maxima.append((hrr.maximum, hrr.x_max, hrr.y_max))
        for species in INTEGRATED_SPECIES:
            values = fields[f"rhoY[{species_names[species]}]"]
            mask = finite_weights & np.isfinite(values)
            _add(result.sums, f"integrated_rhoY_{species}", np.sum(values[mask] * weights[mask], dtype=np.float64))
        gas_density = fields["gas_density"]
        gas_mass_mask = valid_gas & np.isfinite(gas_density) & (gas_density > 0.0)
        _add(
            result.sums, "valid_gas_mass",
            np.sum(gas_density[gas_mass_mask] * weights[gas_mass_mask], dtype=np.float64),
        )
        for species in PRESENTATION_SPECIES:
            values = fields[f"Y[{species_names[species]}]"]
            mask = gas_mass_mask & np.isfinite(values)
            _add(
                result.sums, f"valid_gas_species_mass_{species}",
                np.sum(gas_density[mask] * values[mask] * weights[mask], dtype=np.float64),
            )
        raw_liquid = np.asarray(
            state.fields[f"partial_density[{config.liquid_fluid_id}]"].values,
            dtype=np.float64,
        ).reshape(-1)
        liquid_mass_mask = finite_weights & np.isfinite(raw_liquid)
        _add(
            result.sums, "liquid_NC12H26_inventory",
            np.sum(raw_liquid[liquid_mass_mask] * weights[liquid_mass_mask], dtype=np.float64),
        )
        alpha_liq = fields[f"alpha[{config.liquid_fluid_id}]"]
        liquid_area_mask = masks["mask.valid"] & finite_weights & np.isfinite(alpha_liq) & (alpha_liq > 0.5)
        _add(
            result.sums, "liquid_area_alpha_gt_0p5",
            np.sum(weights[liquid_area_mask], dtype=np.float64),
        )
        temperature = fields["temperature"]
        tmask = valid_gas & np.isfinite(temperature)
        _add(result.sums, "valid_gas_temperature_weighted_sum", np.sum(temperature[tmask] * weights[tmask], dtype=np.float64))
        _add(result.sums, "valid_gas_area", np.sum(weights[tmask], dtype=np.float64))
        _maximum(result.maxima, "max_valid_gas_temperature_K", temperature[tmask])
        for species in MAX_SPECIES:
            values = fields[f"Y[{species_names[species]}]"]
            _maximum(result.maxima, f"max_Y_{species}", values[valid_gas & np.isfinite(values)])
        hot = {}
        for threshold in HOT_THRESHOLDS:
            hot[threshold] = tmask & (temperature > threshold)
            _add(result.sums, f"hot_gas_area_above_{int(threshold)}K", np.sum(weights[hot[threshold]], dtype=np.float64))
        fuel = fields[f"Y[{species_names['NC12H26']}]"]
        oxygen = fields[f"Y[{species_names['O2']}]"]
        phi_valid = valid_gas & np.isfinite(fuel) & np.isfinite(oxygen) & (fuel >= 0.0) & (oxygen > O2_FLOOR)
        phi = np.full(weights.shape, np.nan)
        phi[phi_valid] = stoich * fuel[phi_valid] / oxygen[phi_valid]
        combustible = phi_valid & (phi >= COMBUSTIBLE_PHI[0]) & (phi <= COMBUSTIBLE_PHI[1])
        near = phi_valid & (phi >= NEAR_STOICH_PHI[0]) & (phi <= NEAR_STOICH_PHI[1])
        for name, mask in (
            ("combustible_area", combustible), ("near_stoichiometric_area", near),
            ("hot_combustible_overlap_area", hot[1200.0] & combustible),
            ("hot_near_stoich_overlap_area", hot[1200.0] & near),
        ):
            _add(result.sums, name, np.sum(weights[mask], dtype=np.float64))
        _quality(result, state, physical, weights, phi_valid)
        result.source_files.update(
            field.provenance.source_path for field in state.fields.values() if field.provenance
        )
    return result


def _quality(result, state, physical, weights, phi_valid):
    raw = {name: np.asarray(field.values, dtype=np.float64).reshape(-1) for name, field in state.fields.items()}
    derived = {name: np.asarray(field.values, dtype=np.float64).reshape(-1) for name, field in physical.fields.items()}
    masks = {name: np.asarray(field.values, dtype=bool).reshape(-1) for name, field in physical.masks.items()}
    valid = masks["mask.valid"]
    alpha_sum = np.sum([raw[name] for name in raw if name.startswith("volume_fraction[")], axis=0)
    species_sum = np.sum([raw[name] for name in raw if name.startswith("species_density[")], axis=0)
    species_residual = species_sum - derived["gas_density"]
    for name, mask in (
        ("total_cells", np.ones(valid.shape, dtype=bool)), ("valid_cells", valid),
        ("gas_dominated_cells", masks["mask.gas_dominated"]),
        ("interface_cells", masks["mask.interface"]),
        ("liquid_dominated_cells", masks["mask.liquid_dominated"]),
        ("invalid_cells", ~valid),
        ("thermodynamically_invalid_cells", masks["mask.thermodynamically_invalid"]),
        ("chemistry_valid_cells", masks["mask.chemistry_valid"]),
        ("valid_phi_cells", phi_valid),
        ("temperature_clipped_cells", np.isfinite(derived["temperature.raw"]) & (derived["temperature.raw"] != derived["temperature"])),
        ("pressure_floored_cells", np.isfinite(derived["pressure.raw"]) & (derived["pressure.raw"] != derived["pressure"])),
    ):
        result.counts[name] = result.counts.get(name, 0) + int(np.count_nonzero(mask))
    _maximum(result.quality_maxima, "max_abs_species_closure_kg_m3", np.abs(species_residual[np.isfinite(species_residual)]))
    _maximum(result.quality_maxima, "max_abs_volume_fraction_closure", np.abs(alpha_sum[np.isfinite(alpha_sum)] - 1.0))


def _add(mapping: dict[str, float], name: str, value) -> None:
    mapping[name] = mapping.get(name, 0.0) + float(value)


def _maximum(mapping: dict[str, float], name: str, values) -> None:
    values = np.asarray(values)
    if values.size:
        mapping[name] = max(mapping.get(name, float("-inf")), float(np.max(values)))


def _reduce(
    partial: ScalarPartial, comm, mpi, dimensions: int,
    heat_release_enabled: bool = False,
) -> dict[str, Any] | None:
    sum_op, max_op = (mpi.SUM, mpi.MAX) if mpi else (None, None)
    sum_names = sorted(set().union(*comm.allgather(set(partial.sums))))
    max_names = sorted(set().union(*comm.allgather(set(partial.maxima))))
    count_names = sorted(set().union(*comm.allgather(set(partial.counts))))
    quality_names = sorted(set().union(*comm.allgather(set(partial.quality_maxima))))
    sums = {name: comm.reduce(partial.sums.get(name, 0.0), op=sum_op, root=0) for name in sum_names}
    maxima = {name: comm.reduce(partial.maxima.get(name, float("-inf")), op=max_op, root=0) for name in max_names}
    counts = {name: comm.reduce(partial.counts.get(name, 0), op=sum_op, root=0) for name in count_names}
    quality = {name: comm.reduce(partial.quality_maxima.get(name, 0.0), op=max_op, root=0) for name in quality_names}
    files = sorted(set().union(*comm.allgather(partial.source_files)))
    heat_release_candidates = (
        [
            candidate
            for group in comm.allgather(partial.heat_release_maxima)
            for candidate in group
        ]
        if heat_release_enabled else []
    )
    if comm.rank != 0:
        return None
    spatial_unit = {1: "m", 2: "m2", 3: "m3"}.get(dimensions, "case_measure")
    integrated_unit = {1: "kg/m2", 2: "kg/m", 3: "kg"}.get(dimensions, "case_mass_measure")
    internal_sums = {
        "valid_gas_temperature_weighted_sum", "valid_gas_area", "valid_gas_mass",
        *(f"valid_gas_species_mass_{name}" for name in PRESENTATION_SPECIES),
    }
    record = {
        **{name: float(value) for name, value in sums.items() if name not in internal_sums},
        **{name: (float(value) if np.isfinite(value) else None) for name, value in maxima.items()},
        "mean_valid_gas_temperature_K": (
            float(sums["valid_gas_temperature_weighted_sum"] / sums["valid_gas_area"])
            if sums.get("valid_gas_area", 0.0) > 0.0 else None
        ),
        **{
            f"gas_mass_weighted_Y_{name}": (
                float(sums.get(f"valid_gas_species_mass_{name}", 0.0) / sums["valid_gas_mass"])
                if sums.get("valid_gas_mass", 0.0) > 0.0 else None
            )
            for name in PRESENTATION_SPECIES
        },
        "vapor_NC12H26_inventory": float(sums.get("integrated_rhoY_NC12H26", 0.0)),
        "total_NC12H26_inventory": float(
            sums.get("liquid_NC12H26_inventory", 0.0)
            + sums.get("integrated_rhoY_NC12H26", 0.0)
        ),
        "spatial_measure_unit": spatial_unit,
        "integrated_rhoY_unit": integrated_unit,
        "quality": {**{name: int(value) for name, value in counts.items()}, **{name: float(value) for name, value in quality.items()}},
        "source_files": files,
    }
    if any(name.startswith("integrated_heat_release_rate_") for name in sums):
        candidate = max(
            heat_release_candidates,
            key=lambda item: (item[0], -item[1], -item[2]),
            default=None,
        )
        record.update({
            "max_heat_release_rate_W_m3": candidate[0] if candidate else None,
            "x_max_heat_release_rate_m": candidate[1] if candidate else None,
            "y_max_heat_release_rate_m": candidate[2] if candidate else None,
        })
    return record


def _row(saved_index, timeline, reduced):
    position = timeline["saved_indices"].index(saved_index)
    physical_time = timeline["physical_times"][position]
    return {
        "saved_index": saved_index,
        "simulation_step": timeline["simulation_steps"][position] if timeline["simulation_steps"] else None,
        "physical_time_s": physical_time, "time_us": physical_time * 1.0e6,
        **reduced,
    }


def _add_presentation_columns(rows: list[dict[str, Any]]) -> None:
    reference_area = rows[0].get("liquid_area_alpha_gt_0p5")
    for row in rows:
        row["liquid_area_ratio_A_A0"] = (
            float(row["liquid_area_alpha_gt_0p5"] / reference_area)
            if reference_area is not None and reference_area > 0.0 else None
        )


def _stoich_coefficient(config: Model3Configuration) -> float:
    names = _species_lookup(config)
    weights = dict(zip(config.species_names, config.molecular_weights))
    return 18.5 * weights[names["O2"]] / weights[names["NC12H26"]]


def _species_lookup(config: Model3Configuration) -> dict[str, str]:
    return {
        requested: next(name for name in config.species_names if name.casefold() == requested.casefold())
        for requested in set(INTEGRATED_SPECIES) | set(MAX_SPECIES) | {"O2"}
    }


def _provenance(
    root, report, config, context, strategy, indices, selection, stride,
    source_files, source, metadata, rows, heat_release,
):
    measure = {1: "kg/m2 for integrals and m for measures", 2: "kg/m for integrals and m2 for areas", 3: "kg for integrals and m3 for volumes"}.get(metadata.dimensions)
    return {
        "schema_version": "mfc-post.analyze/v1", "case": str(root),
        "source_family": report["family"], "source_path": report["path"],
        "source": {
            "family": report["family"], "path": report["path"], "layout": report["layout"],
            "files": source_files,
            "warnings": report.get("warnings", []) + report["timeline"].get("warnings", []),
        },
        "timeline": {
            "time_basis": report["timeline"]["time_basis"],
            "t_save": metadata.parameters.get("t_save"),
            "records": [
                {
                    "saved_index": row["saved_index"],
                    "simulation_step": row["simulation_step"],
                    "physical_time": row["physical_time_s"],
                }
                for row in rows
            ],
        },
        "selection": selection,
        "execution": {
            "mode": context.mode, "mpi_size": context.size, "partitioning": strategy,
            "partition_plan": _partition_plan(source, indices, context, strategy),
            "ordering": "rank 0 sorts scalar rows by physical time then saved index and is the sole writer",
        },
        "mechanism": {"path": config.mechanism_path, "phase": config.mechanism_phase},
        "species_mapping": [
            {"name": name, "equation_index": config.equation_indices[f"species_density[{name}]"], "molecular_weight_kg_per_kmol": weight}
            for name, weight in zip(config.species_names, config.molecular_weights)
        ],
        "eos_inputs": [asdict(item) for item in config.eos],
        "mask_thresholds": asdict(config.thresholds),
        "transformations": _transformation_provenance(config),
        **({
            "heat_release": {
                "backend": "cantera",
                "enabled": True,
                "state": "gas.TDY = reconstructed chemistry-clipped T, chemistry gas density, full normalized Y vector",
                "mask": "mask.chemistry_valid AND alpha_liq <= 0.5",
                "definition": "gas.heat_release_rate [W/m^3]",
                "verification": "-dot(gas.net_production_rates, gas.partial_molar_enthalpies)",
                "verification_rtol": CHECK_RTOL,
                "verification_atol_W_m3": CHECK_ATOL_W_M3,
                "integration": "2D cell area times volumetric rate, assuming unit depth; W/m",
                "species_order": list(heat_release.species_names),
            }
        } if heat_release is not None else {}),
        "diagnostics": {
            "integrated_rhoY": f"raw conservative rhoY integrated over cells with finite rhoY and positive finite cell measure; units {measure}",
            "valid_gas": "mask.chemistry_valid AND mask.gas_dominated",
            "mean_temperature": "cell-measure-weighted chemistry-clipped temperature over valid gas",
            "hot_areas": "valid-gas chemistry-clipped T > 1200 K or > 1500 K",
            "equivalence_ratio": "phi=(18.5*MW_O2/MW_NC12H26)*Y_NC12H26/Y_O2 with Y_O2>1e-12",
            "stoichiometric_coefficient": _stoich_coefficient(config),
            "combustible": "0.5 <= phi <= 2.0",
            "near_stoichiometric": "0.8 <= phi <= 1.2",
            "hot_overlap": "T > 1200 K intersected with combustible or near-stoichiometric mask",
            "liquid_area": "cell measure where reconstructed alpha_liq > 0.5",
            "liquid_area_ratio": "A/A0 where A0 is the first chronological analyzed row",
            "gas_mass_weighted_species": (
                "integral(rho_g * bounded normalized Y_k dV) / integral(rho_g dV) "
                "over chemistry-valid AND gas-dominated cells"
            ),
            "dodecane_inventories": (
                "liquid=raw conservative partial density of evap_liquid_fluid_id; "
                "vapor=raw conservative rhoY_NC12H26; total=liquid+vapor; "
                "fuel_species_id is verified as NC12H26"
            ),
        },
        "reductions": {
            "spatial": [
                "MPI SUM: integrals, weighted temperature, areas, and counts",
                "MPI MAX: temperature, species fractions, and closure residuals",
            ],
            "state": (
                "each state is owned by one MPI worker and accumulated over its source "
                "partitions/chunks without a cross-state reduction"
            ),
        },
        "stride": stride,
    }


def _write_csv(
    path: Path, rows: list[dict[str, Any]], include_heat_release: bool = False,
) -> None:
    columns = [
        "saved_index", "simulation_step", "physical_time_s", "time_us",
        *(f"integrated_rhoY_{name}" for name in INTEGRATED_SPECIES),
        "liquid_NC12H26_inventory", "vapor_NC12H26_inventory",
        "total_NC12H26_inventory",
        *(f"gas_mass_weighted_Y_{name}" for name in PRESENTATION_SPECIES),
        "max_valid_gas_temperature_K", "mean_valid_gas_temperature_K",
        *(f"max_Y_{name}" for name in MAX_SPECIES),
        "hot_gas_area_above_1200K", "hot_gas_area_above_1500K",
        "combustible_area", "near_stoichiometric_area",
        "hot_combustible_overlap_area", "hot_near_stoich_overlap_area",
        "liquid_area_alpha_gt_0p5", "liquid_area_ratio_A_A0",
        "integrated_rhoY_unit", "spatial_measure_unit",
    ]
    if include_heat_release:
        columns.extend([
            "max_heat_release_rate_W_m3",
            "x_max_heat_release_rate_m", "y_max_heat_release_rate_m",
            "integrated_heat_release_rate_net_W_per_m",
            "integrated_heat_release_rate_positive_W_per_m",
            "integrated_heat_release_rate_negative_W_per_m",
            "area_positive_heat_release_rate",
        ])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _error(context, family, saved_index, exc):
    return (
        f"worker_rank={context.rank} source={family} saved_index={saved_index}: "
        f"{type(exc).__name__}: {exc}"
    )


def _progress(context, message):
    if context.rank == 0:
        print(f"mfc-post analyze: {message}", flush=True)


def _save_progress(
    context, ordinal, total, saved_index, state_parallel, diagnostic=None,
):
    if state_parallel or context.rank == 0:
        print(
            f"mfc-post analyze: progress: save {ordinal}/{total}: "
            f"saved_index={saved_index}, worker_rank={context.rank}",
            flush=True,
        )
        if diagnostic == "cantera":
            print(
                f"mfc-post analyze: progress: Cantera HRR saved_index={saved_index}, "
                f"worker_rank={context.rank}",
                flush=True,
            )
