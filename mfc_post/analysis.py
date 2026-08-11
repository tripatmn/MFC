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
from .inspect import inspect_case
from .models import RunMetadata, State
from .process import _load_local, _partition_plan, _transformation_provenance
from .reconstruction import Model3Configuration, reconstruct_model3
from .sources import PAllSource


INTEGRATED_SPECIES = ("NC12H26", "OH", "HO2", "H2O2", "CO", "CO2", "H2O")
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


def analyze_case(
    case_path: str | Path,
    selected_times_us: Iterable[float] | None = None,
    time_range_us: tuple[float, float] | None = None,
    stride: int = 1,
    out_dir: str | Path | None = None,
    execution: str = "auto",
    mechanism: str | Path | None = None,
    phase: str | None = None,
) -> dict[str, Any] | None:
    context = ExecutionContext.create(execution)
    root = Path(case_path).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve() if out_dir else root / "mfc_post_analysis"
    if stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if selected_times_us is not None and time_range_us is not None:
        raise ValueError("--selected-times-us and --time-range-us are mutually exclusive")

    _progress(context, f"discovery: inspecting {root}")
    inspection = inspect_case(root, mechanism=mechanism, phase=phase)
    metadata = _metadata(inspection)
    report = next((item for item in inspection["sources"] if item["family"] == "p_all"), None)
    if report is None or not report["timeline"]["saved_indices"]:
        raise ValueError(f"scalar analysis requires raw p_all saves; none were found in {root}")
    config = Model3Configuration.from_metadata(metadata)
    _validate_species(config)
    source = PAllSource(root, metadata)
    indices, selection = _select_indices(
        report["timeline"], selected_times_us, time_range_us, stride
    )
    _progress(context, f"selection: {len(indices)} save(s): {indices}")
    _prepare_output(context, destination)

    strategy = context.strategy(len(indices))
    raw_fields = tuple(str(item["name"]) for item in metadata.equation_layout)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if strategy == "state":
        state_comm = context.mpi.COMM_SELF
        for saved_index in indices[context.rank::context.size]:
            try:
                _progress(context, f"derivation: saved index {saved_index}")
                states = _load_local(source, saved_index, raw_fields, "serial", context)
                partial = _accumulate(states, config)
                reduced = _reduce(partial, state_comm, context.mpi, metadata.dimensions or 1)
                rows.append(_row(saved_index, report["timeline"], reduced))
            except Exception as exc:
                errors.append(_error(context, saved_index, exc))
        all_errors = [item for group in context.comm.allgather(errors) for item in group]
        if all_errors:
            raise RuntimeError("analysis failed; no output written:\n" + "\n".join(sorted(all_errors)))
        gathered = context.comm.gather(rows, root=0)
        if context.rank == 0:
            rows = [row for group in gathered for row in group]
    else:
        for saved_index in indices:
            _progress(context, f"derivation: saved index {saved_index}")
            local_error = None
            partial = None
            try:
                states = _load_local(source, saved_index, raw_fields, strategy, context)
                partial = _accumulate(states, config)
            except Exception as exc:
                local_error = _error(context, saved_index, exc)
            state_errors = [item for item in context.comm.allgather(local_error) if item]
            if state_errors:
                raise RuntimeError("analysis failed; no output written:\n" + "\n".join(sorted(state_errors)))
            reduced = _reduce(partial, context.comm, context.mpi, metadata.dimensions or 1)
            if context.rank == 0:
                rows.append(_row(saved_index, report["timeline"], reduced))

    if context.rank != 0:
        return None
    rows.sort(key=lambda row: (row["physical_time_s"], row["saved_index"]))
    source_files = sorted(set().union(*(set(row.pop("source_files")) for row in rows)))
    quality = [
        {"saved_index": row["saved_index"], "time_us": row["time_us"], **row.pop("quality")}
        for row in rows
    ]
    provenance = _provenance(
        root, report, config, context, strategy, indices, selection, stride,
        source_files, source, metadata,
    )
    _write_csv(destination / "scalar_timeseries.csv", rows)
    _atomic_json(destination / "quality.json", {
        "schema_version": "mfc-post.quality/v1",
        "source_family": "p_all",
        "quality": quality,
    })
    _atomic_json(destination / "provenance.json", provenance)
    result = {
        "output_directory": str(destination), "rows": rows,
        "quality": quality, "provenance": provenance,
    }
    _progress(context, f"completion: wrote {len(rows)} scalar record(s) to {destination}")
    return result


def _metadata(inspection: dict[str, Any]) -> RunMetadata:
    raw = inspection["run_metadata"]
    return RunMetadata(
        case_path=raw["case_path"], parameters=raw["parameters"], dimensions=raw["dimensions"],
        grid_shape=tuple(raw["grid_shape"]) if raw["grid_shape"] else None,
        equation_layout=tuple(raw["equation_layout"]), species_names=tuple(raw["species_names"]),
        missing=tuple(raw["missing"]), warnings=tuple(raw["warnings"]),
    )


def _validate_species(config: Model3Configuration) -> None:
    required = set(INTEGRATED_SPECIES) | set(MAX_SPECIES) | {"O2"}
    by_casefold: dict[str, list[str]] = {}
    for name in config.species_names:
        by_casefold.setdefault(name.casefold(), []).append(name)
    invalid = [name for name in sorted(required) if len(by_casefold.get(name.casefold(), [])) != 1]
    if invalid:
        raise ValueError(f"required analysis species are absent or ambiguous: {invalid}")


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


def _prepare_output(context: ExecutionContext, destination: Path) -> None:
    error = None
    if context.rank == 0:
        try:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f"output directory is not empty: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [item for item in context.comm.allgather(error) if item]
    if errors:
        raise RuntimeError(errors[0])


def _accumulate(states: Iterable[State], config: Model3Configuration) -> ScalarPartial:
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
        for species in INTEGRATED_SPECIES:
            values = fields[f"rhoY[{species_names[species]}]"]
            mask = finite_weights & np.isfinite(values)
            _add(result.sums, f"integrated_rhoY_{species}", np.sum(values[mask] * weights[mask], dtype=np.float64))
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


def _reduce(partial: ScalarPartial, comm, mpi, dimensions: int) -> dict[str, Any] | None:
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
    if comm.rank != 0:
        return None
    spatial_unit = {1: "m", 2: "m2", 3: "m3"}.get(dimensions, "case_measure")
    integrated_unit = {1: "kg/m2", 2: "kg/m", 3: "kg"}.get(dimensions, "case_mass_measure")
    record = {
        **{name: float(value) for name, value in sums.items() if name not in {"valid_gas_temperature_weighted_sum", "valid_gas_area"}},
        **{name: (float(value) if np.isfinite(value) else None) for name, value in maxima.items()},
        "mean_valid_gas_temperature_K": (
            float(sums["valid_gas_temperature_weighted_sum"] / sums["valid_gas_area"])
            if sums.get("valid_gas_area", 0.0) > 0.0 else None
        ),
        "spatial_measure_unit": spatial_unit,
        "integrated_rhoY_unit": integrated_unit,
        "quality": {**{name: int(value) for name, value in counts.items()}, **{name: float(value) for name, value in quality.items()}},
        "source_files": files,
    }
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


def _stoich_coefficient(config: Model3Configuration) -> float:
    names = _species_lookup(config)
    weights = dict(zip(config.species_names, config.molecular_weights))
    return 18.5 * weights[names["O2"]] / weights[names["NC12H26"]]


def _species_lookup(config: Model3Configuration) -> dict[str, str]:
    return {
        requested: next(name for name in config.species_names if name.casefold() == requested.casefold())
        for requested in set(INTEGRATED_SPECIES) | set(MAX_SPECIES) | {"O2"}
    }


def _provenance(root, report, config, context, strategy, indices, selection, stride, source_files, source, metadata):
    measure = {1: "kg/m2 for integrals and m for measures", 2: "kg/m for integrals and m2 for areas", 3: "kg for integrals and m3 for volumes"}.get(metadata.dimensions)
    return {
        "schema_version": "mfc-post.analyze/v1", "case": str(root),
        "source": {
            "family": "p_all", "path": report["path"], "layout": report["layout"],
            "files": source_files,
            "warnings": report.get("warnings", []) + report["timeline"].get("warnings", []),
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
        },
        "reductions": {
            "spatial": [
                "MPI SUM: integrals, weighted temperature, areas, and counts",
                "MPI MAX: temperature, species fractions, and closure residuals",
            ],
            "state": "each state is owned by one MPI worker and accumulated over its stored p_all partitions without a cross-state reduction",
        },
        "stride": stride,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "saved_index", "simulation_step", "physical_time_s", "time_us",
        *(f"integrated_rhoY_{name}" for name in INTEGRATED_SPECIES),
        "max_valid_gas_temperature_K", "mean_valid_gas_temperature_K",
        *(f"max_Y_{name}" for name in MAX_SPECIES),
        "hot_gas_area_above_1200K", "hot_gas_area_above_1500K",
        "combustible_area", "near_stoichiometric_area",
        "hot_combustible_overlap_area", "hot_near_stoich_overlap_area",
        "integrated_rhoY_unit", "spatial_measure_unit",
    ]
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


def _error(context, saved_index, exc):
    return f"worker_rank={context.rank} source=p_all saved_index={saved_index}: {type(exc).__name__}: {exc}"


def _progress(context, message):
    if context.rank == 0:
        print(f"mfc-post analyze: {message}", flush=True)
