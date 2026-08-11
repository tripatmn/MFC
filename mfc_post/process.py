"""Deterministic serial/MPI processing orchestration for the initial metric slice."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import int_param
from .execution import ExecutionContext, split_range
from .inspect import inspect_case
from .metrics import accumulate, reduce_partial, required_fields
from .models import RunMetadata
from .reconstruction import Model3Configuration
from .sources import LustrePerProcessSource, LustreSharedSource, PAllSource


SOURCE_TYPES = {
    "p_all": PAllSource,
    "lustre_shared": LustreSharedSource,
    "lustre_per_process": LustrePerProcessSource,
}


def process_case(
    case_path: str | Path, execution: str = "auto", source_family: str | None = None,
    index_start: int | None = None, index_stop: int | None = None,
    output: str | Path | None = None,
    mechanism: str | Path | None = None, phase: str | None = None,
) -> dict[str, Any] | None:
    context = ExecutionContext.create(execution)
    root = Path(case_path).expanduser().resolve()
    inspection = inspect_case(root, mechanism=mechanism, phase=phase)
    metadata = _metadata(inspection)
    reconstruction = Model3Configuration.from_metadata(metadata)
    family = source_family or (inspection.get("recommendation") or {}).get("source")
    if family not in SOURCE_TYPES:
        available = sorted(report["family"] for report in inspection["sources"] if report["raw_fields"])
        raise ValueError(
            f"no processable raw source selected; requested={family!r}, available={available}"
        )
    report = next((item for item in inspection["sources"] if item["family"] == family), None)
    if report is None:
        raise ValueError(f"requested source {family!r} is not available")
    indices = [
        int(value) for value in report["timeline"]["saved_indices"]
        if (index_start is None or value >= index_start) and (index_stop is None or value < index_stop)
    ]
    if not indices:
        raise ValueError(
            f"index selection [{index_start if index_start is not None else '-inf'}, "
            f"{index_stop if index_stop is not None else 'inf'}) contains no available states"
        )
    source = SOURCE_TYPES[family](root, metadata)
    fields = required_fields(metadata.equation_layout)
    liquid_id = int_param(dict(metadata.parameters), "evap_liquid_fluid_id", 1)
    gas_ids = _gas_ids(dict(metadata.parameters))
    strategy = context.strategy(len(indices))
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if strategy == "state":
        state_comm = context.mpi.COMM_SELF
        for index in indices[context.rank::context.size]:
            try:
                states = _load_local(source, index, fields, "serial", context)
                partial = accumulate(states, liquid_id, gas_ids, reconstruction)
                metrics = reduce_partial(partial, state_comm, context.mpi, metadata.dimensions or 1)
                if metrics is not None:
                    records.append(_record(index, report["timeline"], metrics))
            except Exception as exc:  # synchronized below; preserve context from each worker
                errors.append(_error(context, family, index, exc))
        all_errors = [item for group in context.comm.allgather(errors) for item in group]
        if all_errors:
            raise RuntimeError("processing failed; no output written:\n" + "\n".join(sorted(all_errors)))
        gathered = context.comm.gather(records, root=0)
        if context.rank == 0:
            records = [item for group in gathered for item in group]
    else:
        for index in indices:
            local_error = None
            partial = None
            try:
                states = _load_local(source, index, fields, strategy, context)
                partial = accumulate(states, liquid_id, gas_ids, reconstruction)
            except Exception as exc:
                local_error = _error(context, family, index, exc)
            all_errors = [item for item in context.comm.allgather(local_error) if item]
            if all_errors:
                raise RuntimeError("processing failed; no output written:\n" + "\n".join(sorted(all_errors)))
            metrics = reduce_partial(partial, context.comm, context.mpi, metadata.dimensions or 1)
            if context.rank == 0 and metrics is not None:
                records.append(_record(index, report["timeline"], metrics))
    if context.rank != 0:
        return None
    records.sort(key=lambda item: (item["saved_index"], item["physical_time"] is None, item["physical_time"] or 0.0))
    result = _result(
        inspection, report, records, context, strategy, liquid_id, gas_ids,
        _partition_plan(source, indices, context, strategy),
        index_start, index_stop, reconstruction,
    )
    destination = Path(output).expanduser().resolve() if output else root / "mfc_post_output"
    _write_outputs(destination, result)
    result["output_directory"] = str(destination)
    return result


def _load_local(source, index, fields, strategy, context):
    if isinstance(source, PAllSource):
        partitions = source.partition_ids(index)
        if not partitions:
            raise FileNotFoundError(f"p_all saved index {index}: no stored rank directories")
        if strategy == "spatial":
            assignments: list[tuple[int, int, int]] = []
            total_workers = context.size
            # Subdivide every stored rank record, so extra analysis ranks still perform disjoint reads.
            for partition in partitions:
                count = source.partition_cell_count(index, partition)
                start, stop = split_range(count, total_workers, context.rank)
                if start < stop:
                    assignments.append((partition, start, stop))
            return [source.read_partition_range(index, part, start, stop, fields) for part, start, stop in assignments]
        return [source.read_partition(index, partition, fields) for partition in partitions]
    if isinstance(source, LustreSharedSource):
        grid = source.read_grid(index)
        cells = int(np.prod(grid.shape, dtype=np.int64))
        if strategy == "spatial":
            start, stop = split_range(cells, context.size, context.rank)
            return [source.read_chunk(index, start, stop, fields, mpi_comm=context.comm if context.mpi else None)]
        return [source.read_chunk(index, 0, cells, fields)]
    if isinstance(source, LustrePerProcessSource):
        partitions = source.partition_ids(index)
        assigned = partitions[context.rank::context.size] if strategy == "spatial" else partitions
        return [source.read_partition(index, partition, fields) for partition in assigned]
    raise ValueError(f"unsupported processing source {source.family}")


def _metadata(inspection: dict[str, Any]) -> RunMetadata:
    raw = inspection["run_metadata"]
    return RunMetadata(
        case_path=raw["case_path"], parameters=raw["parameters"],
        dimensions=raw["dimensions"],
        grid_shape=tuple(raw["grid_shape"]) if raw["grid_shape"] else None,
        equation_layout=tuple(raw["equation_layout"]),
        species_names=tuple(raw["species_names"]), missing=tuple(raw["missing"]),
        warnings=tuple(raw["warnings"]),
    )


def _gas_ids(params: dict[str, Any]) -> tuple[int, ...]:
    count = int_param(params, "chem_gas_num_fluids", 0)
    if count:
        ids = tuple(int_param(params, f"chem_gas_fluid_ids({index})", 0) for index in range(1, count + 1))
        if all(ids):
            return ids
    return (int_param(params, "chem_gas_fluid_id", 1),)


def _record(index: int, timeline: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    position = timeline["saved_indices"].index(index)
    return {
        "saved_index": index,
        "simulation_step": timeline["simulation_steps"][position] if timeline["simulation_steps"] else None,
        "physical_time": timeline["physical_times"][position],
        **metrics,
    }


def _result(inspection, report, records, context, strategy, liquid_id, gas_ids, partition_plan, start, stop, reconstruction):
    source_files = sorted(set().union(*(set(record.pop("source_files")) for record in records)))
    return {
        "schema_version": "mfc-post.process/v2",
        "case": inspection["case"],
        "source": {"family": report["family"], "path": report["path"], "layout": report["layout"]},
        "selection": {"index_start_inclusive": start, "index_stop_exclusive": stop, "saved_indices": [r["saved_index"] for r in records]},
        "timeline": {
            "time_basis": report["timeline"]["time_basis"],
            "records": [{key: record[key] for key in ("saved_index", "simulation_step", "physical_time")} for record in records],
            "warnings": report["timeline"]["warnings"],
        },
        "records": records,
        "quality": [
            {
                "saved_index": record["saved_index"], "closure": record["closure"],
                "invalid_cells": record["invalid_cells"],
                "physical_mask_counts": record["physical_state"]["mask_counts"],
            }
            for record in records
        ],
        "provenance": {
            "execution": {
                "requested": context.requested, "mode": context.mode, "mpi_size": context.size,
                "partitioning": strategy, "partition_plan": partition_plan,
                "deterministic_order": "rank 0 sorts scalar records by saved index; rank 0 is sole writer",
            },
            "source_files": source_files,
            "fields": (
                "raw conservative fields are preserved; physical fields are derived partition-locally "
                "with the branch Model-3 limiter, mixture EOS, and chemistry conversion"
            ),
            "liquid_fluid_id": liquid_id, "gas_fluid_ids": list(gas_ids),
            "equation_mapping": {
                name: int(index) for name, index in sorted(
                    reconstruction.equation_indices.items(), key=lambda item: item[1]
                )
            },
            "species_mapping": [
                {
                    "name": name,
                    "equation_index": reconstruction.equation_indices[f"species_density[{name}]"],
                    "molecular_weight_kg_per_kmol": weight,
                }
                for name, weight in zip(reconstruction.species_names, reconstruction.molecular_weights)
            ],
            "eos_inputs": [asdict(item) for item in reconstruction.eos],
            "mask_thresholds": asdict(reconstruction.thresholds),
            "mechanism": {
                "path": reconstruction.mechanism_path,
                "phase": reconstruction.mechanism_phase,
                "gas_constant_J_per_kmol_K": reconstruction.gas_constant,
            },
            "transformations": _transformation_provenance(reconstruction),
            "reductions": [
                "SUM: integrals, invalid/mask counts, and finite-field counts",
                "MAX: closure and physical-field maxima", "MIN: physical-field minima",
            ],
            "cell_measure": "derived from actual nonuniform cell-boundary coordinates",
        },
        "warnings": inspection["warnings"] + report["warnings"],
    }


def _transformation_provenance(reconstruction: Model3Configuration) -> list[str]:
    transformations = [
        "when mpp_lim: alpha_rho_i=max(0,alpha_rho_i), alpha_i=clip(alpha_i,0,1), alpha_i/=max(sum(alpha),1e-16)",
        "rho=sum(alpha_rho_i); u_j=momentum_j/rho; kinetic=0.5*sum(momentum_j*u_j)",
        "Gamma=sum(alpha_i*gamma_i); Pi_inf=sum(alpha_i*pi_inf_i); qv=sum(alpha_rho_i*qv_i)",
        "p_raw=(E-kinetic-Pi_inf-qv)/Gamma; p=max(p_raw,100 Pa)",
        "rho_g=sum(raw alpha_rho_i for configured gas fluids)",
        "Y_k=clip(rhoY_k/max(rho_g,1e-16),0,1), normalized only when bounded sum exceeds one",
        "W_mix=1/sum(Y_k/W_k)",
    ]
    if reconstruction.fixed_temperature_enabled:
        transformations.append("T_raw=chem_fixed_T; chemistry temperature=clip(T_raw,chem_T_min,chem_T_max)")
    elif reconstruction.chemistry_reaction_heat_enabled:
        transformations.append(
            "T_raw=p*W_mix/(R_u*rho_g); chemistry temperature=clip(T_raw,chem_T_min,chem_T_max)"
        )
    else:
        transformations.append("temperature unavailable: neither fixed-T nor reaction-heat reconstruction is enabled")
    return transformations


def _write_outputs(directory: Path, result: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / "results.json", result)
    _atomic_json(directory / "timeline.json", result["timeline"])
    _atomic_json(directory / "quality.json", {"quality": result["quality"]})
    _atomic_json(directory / "provenance.json", result["provenance"])
    rows = [_flatten_record(record) for record in result["records"]]
    columns = sorted(set().union(*(set(row) for row in rows)))
    temporary = directory / ".metrics.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, directory / "metrics.csv")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "saved_index": record["saved_index"], "simulation_step": record["simulation_step"],
        "physical_time": record["physical_time"], "cell_count": record["cell_count"],
        "conservative_liquid_mass": record["conservative_liquid_mass"],
        "dense_liquid_measure": record["dense_liquid"]["measure"],
        "dense_liquid_equivalent_diameter": record["dense_liquid"]["equivalent_diameter"],
        **{f"invalid_{key}": value for key, value in record["invalid_cells"].items()},
    }
    row.update({f"species_inventory_{name}": value for name, value in record["raw_species_inventories"].items()})
    row.update({
        name.replace(".", "_"): value
        for name, value in record["physical_state"]["mask_counts"].items()
    })
    for name, summary in record["physical_state"]["ranges"].items():
        safe = name.replace("[", "_").replace("]", "").replace(".", "_")
        row[f"{safe}_min"] = summary["minimum"]
        row[f"{safe}_max"] = summary["maximum"]
        row[f"{safe}_finite_cells"] = summary["finite_cell_count"]
    return row


def _error(context, family: str, index: int, exc: Exception) -> str:
    return f"worker_rank={context.rank} source={family} saved_index={index}: {type(exc).__name__}: {exc}"


def _partition_plan(source, indices: list[int], context: ExecutionContext, strategy: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices):
        if isinstance(source, PAllSource):
            stored = source.partition_ids(index)
            if strategy == "spatial":
                workers = []
                for rank in range(context.size):
                    pieces = []
                    for partition in stored:
                        start, stop = split_range(source.partition_cell_count(index, partition), context.size, rank)
                        if start < stop:
                            pieces.append({"stored_rank": partition, "flat_range": [start, stop]})
                    workers.append({"worker_rank": rank, "pieces": pieces})
                plan.append({"saved_index": index, "workers": workers})
            else:
                plan.append({"saved_index": index, "worker_rank": ordinal % context.size, "stored_ranks": list(stored)})
        elif isinstance(source, LustreSharedSource):
            cells = int(np.prod(source.read_grid(index).shape, dtype=np.int64))
            if strategy == "spatial":
                plan.append({
                    "saved_index": index,
                    "workers": [
                        {"worker_rank": rank, "global_flat_range": list(split_range(cells, context.size, rank))}
                        for rank in range(context.size)
                    ],
                })
            else:
                plan.append({"saved_index": index, "worker_rank": ordinal % context.size, "global_flat_range": [0, cells]})
        else:
            stored = source.partition_ids(index)
            plan.append({
                "saved_index": index,
                "workers": [
                    {"worker_rank": rank, "stored_ranks": list(stored[rank::context.size])}
                    for rank in range(context.size)
                ],
            })
    return plan
