"""Initial conservative inventory and closure checks on local State partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .models import State
from .reconstruction import Model3Configuration, reconstruct_model3


@dataclass
class PartialMetrics:
    cells: int = 0
    liquid_mass: float = 0.0
    dense_measure: float = 0.0
    species_inventory: dict[str, float] = field(default_factory=dict)
    species_residual_l1: float = 0.0
    species_residual_max: float = 0.0
    species_relative_max: float = 0.0
    volume_residual_l1: float = 0.0
    volume_residual_max: float = 0.0
    total_density_min: float = float("inf")
    total_density_max: float = float("-inf")
    invalid_any: int = 0
    nonfinite: int = 0
    nonpositive_density: int = 0
    negative_partial_density: int = 0
    negative_species_density: int = 0
    volume_fraction_out_of_bounds: int = 0
    physical_min: dict[str, float] = field(default_factory=dict)
    physical_max: dict[str, float] = field(default_factory=dict)
    physical_finite_count: dict[str, int] = field(default_factory=dict)
    mask_counts: dict[str, int] = field(default_factory=dict)
    source_files: set[str] = field(default_factory=set)


def required_fields(layout: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item["name"]) for item in layout)


def accumulate(
    states: Iterable[State], liquid_id: int, gas_ids: tuple[int, ...],
    reconstruction: Model3Configuration,
) -> PartialMetrics:
    result = PartialMetrics()
    for state in states:
        if state.grid is None or state.grid.cell_measures is None:
            raise ValueError(f"saved index {state.saved_index}: cell measures are unavailable")
        weights = np.asarray(state.grid.cell_measures, dtype=np.float64)
        arrays = {name: np.asarray(field.values, dtype=np.float64).reshape(-1) for name, field in state.fields.items()}
        if any(values.size != weights.size for values in arrays.values()):
            raise ValueError(f"saved index {state.saved_index}: field/grid partition size mismatch")
        partial = [arrays[name] for name in sorted(arrays) if name.startswith("partial_density[")]
        alphas = [arrays[name] for name in sorted(arrays) if name.startswith("volume_fraction[")]
        species_items = [(name, arrays[name]) for name in sorted(arrays) if name.startswith("species_density[")]
        liquid_name = f"partial_density[{liquid_id}]"
        alpha_name = f"volume_fraction[{liquid_id}]"
        if liquid_name not in arrays or alpha_name not in arrays:
            raise ValueError(f"saved index {state.saved_index}: liquid fields {liquid_name}/{alpha_name} are absent")
        if not partial or not alphas:
            raise ValueError(f"saved index {state.saved_index}: closure fields are absent")
        all_values = list(arrays.values())
        finite_masks = [np.isfinite(values) for values in all_values]
        finite_all = np.logical_and.reduce(finite_masks) if finite_masks else np.ones(weights.size, dtype=bool)
        nonfinite = ~finite_all
        total_density = np.sum(partial, axis=0)
        alpha_sum = np.sum(alphas, axis=0)
        species_sum = np.sum([values for _, values in species_items], axis=0) if species_items else np.zeros_like(total_density)
        gas_density = np.zeros_like(total_density)
        for fluid_id in gas_ids:
            name = f"partial_density[{fluid_id}]"
            if name not in arrays:
                raise ValueError(f"saved index {state.saved_index}: configured gas field {name} is absent")
            gas_density += arrays[name]
        species_residual = species_sum - gas_density
        volume_residual = alpha_sum - 1.0
        valid = finite_all & np.isfinite(weights)
        result.cells += weights.size
        result.liquid_mass += float(np.sum(arrays[liquid_name][valid] * weights[valid], dtype=np.float64))
        result.dense_measure += float(np.sum(weights[valid & (arrays[alpha_name] > 0.5)], dtype=np.float64))
        for name, values in species_items:
            label = name[len("species_density["):-1]
            result.species_inventory[label] = result.species_inventory.get(label, 0.0) + float(
                np.sum(values[valid] * weights[valid], dtype=np.float64)
            )
        if np.any(valid):
            abs_species = np.abs(species_residual[valid])
            abs_volume = np.abs(volume_residual[valid])
            denom = np.maximum(np.abs(gas_density[valid]), np.finfo(np.float64).tiny)
            result.species_residual_l1 += float(np.sum(abs_species * weights[valid], dtype=np.float64))
            result.species_residual_max = max(result.species_residual_max, float(np.max(abs_species)))
            result.species_relative_max = max(result.species_relative_max, float(np.max(abs_species / denom)))
            result.volume_residual_l1 += float(np.sum(abs_volume * weights[valid], dtype=np.float64))
            result.volume_residual_max = max(result.volume_residual_max, float(np.max(abs_volume)))
            result.total_density_min = min(result.total_density_min, float(np.min(total_density[valid])))
            result.total_density_max = max(result.total_density_max, float(np.max(total_density[valid])))
        neg_partial = np.logical_or.reduce([values < 0.0 for values in partial])
        neg_species = np.logical_or.reduce([values < 0.0 for _, values in species_items]) if species_items else np.zeros(weights.size, dtype=bool)
        alpha_bad = np.logical_or.reduce([(values < 0.0) | (values > 1.0) for values in alphas])
        rho_bad = total_density <= 0.0
        invalid_any = nonfinite | neg_partial | neg_species | alpha_bad | rho_bad
        result.nonfinite += int(np.count_nonzero(nonfinite))
        result.nonpositive_density += int(np.count_nonzero(rho_bad))
        result.negative_partial_density += int(np.count_nonzero(neg_partial))
        result.negative_species_density += int(np.count_nonzero(neg_species))
        result.volume_fraction_out_of_bounds += int(np.count_nonzero(alpha_bad))
        result.invalid_any += int(np.count_nonzero(invalid_any))
        result.source_files.update(
            field.provenance.source_path for field in state.fields.values() if field.provenance
        )
        physical = reconstruct_model3(state, reconstruction)
        summary_names = [
            "pressure", "temperature", "temperature.raw", "gas_density", "molecular_weight",
            *(f"velocity[{axis}]" for axis in ("x", "y", "z")[:reconstruction.dimensions]),
            *(f"Y[{name}]" for name in reconstruction.species_names),
        ]
        for name in summary_names:
            values = np.asarray(physical.fields[name].values, dtype=np.float64).reshape(-1)
            finite = values[np.isfinite(values)]
            result.physical_finite_count[name] = result.physical_finite_count.get(name, 0) + finite.size
            if finite.size:
                result.physical_min[name] = min(result.physical_min.get(name, float("inf")), float(np.min(finite)))
                result.physical_max[name] = max(result.physical_max.get(name, float("-inf")), float(np.max(finite)))
        for name, field_value in physical.masks.items():
            result.mask_counts[name] = result.mask_counts.get(name, 0) + int(
                np.count_nonzero(np.asarray(field_value.values, dtype=bool))
            )
    return result


def reduce_partial(partial: PartialMetrics, comm, mpi, dimensions: int) -> dict[str, Any] | None:
    sum_op = mpi.SUM if mpi else None
    max_op = mpi.MAX if mpi else None
    min_op = mpi.MIN if mpi else None
    rank = int(comm.rank)

    def red(value, op):
        return comm.reduce(value, op=op, root=0)

    summed_names = (
        "cells", "liquid_mass", "dense_measure", "species_residual_l1",
        "volume_residual_l1", "invalid_any", "nonfinite", "nonpositive_density",
        "negative_partial_density", "negative_species_density", "volume_fraction_out_of_bounds",
    )
    values = {name: red(getattr(partial, name), sum_op) for name in summed_names}
    values["species_residual_max"] = red(partial.species_residual_max, max_op)
    values["species_relative_max"] = red(partial.species_relative_max, max_op)
    values["volume_residual_max"] = red(partial.volume_residual_max, max_op)
    values["total_density_min"] = red(partial.total_density_min, min_op)
    values["total_density_max"] = red(partial.total_density_max, max_op)
    names = sorted(set().union(*comm.allgather(set(partial.species_inventory))))
    inventories = {name: red(partial.species_inventory.get(name, 0.0), sum_op) for name in names}
    physical_names = sorted(set().union(*comm.allgather(set(partial.physical_finite_count))))
    physical_counts = {
        name: red(partial.physical_finite_count.get(name, 0), sum_op) for name in physical_names
    }
    physical_minima = {
        name: red(partial.physical_min.get(name, float("inf")), min_op) for name in physical_names
    }
    physical_maxima = {
        name: red(partial.physical_max.get(name, float("-inf")), max_op) for name in physical_names
    }
    mask_names = sorted(set().union(*comm.allgather(set(partial.mask_counts))))
    mask_counts = {name: red(partial.mask_counts.get(name, 0), sum_op) for name in mask_names}
    files = sorted(set().union(*comm.allgather(partial.source_files)))
    if rank != 0:
        return None
    measure = float(values["dense_measure"])
    equivalent = None
    if dimensions == 2:
        equivalent = 2.0 * np.sqrt(measure / np.pi)
    elif dimensions == 3:
        equivalent = (6.0 * measure / np.pi) ** (1.0 / 3.0)
    return {
        "cell_count": int(values["cells"]),
        "conservative_liquid_mass": float(values["liquid_mass"]),
        "dense_liquid": {
            "threshold": "alpha_liq > 0.5", "measure": measure,
            "measure_kind": "area" if dimensions == 2 else "volume" if dimensions == 3 else "length",
            "equivalent_diameter": float(equivalent) if equivalent is not None else None,
        },
        "raw_species_inventories": {name: float(inventories[name]) for name in names},
        "closure": {
            "density": {
                "independent_check_available": False,
                "reason": "model-3 raw output stores partial densities but no independent total-density field",
                "constructed_total_min": float(values["total_density_min"]),
                "constructed_total_max": float(values["total_density_max"]),
            },
            "species": {
                "basis": "sum(raw species densities) - sum(configured gas-fluid partial densities)",
                "integrated_absolute_residual": float(values["species_residual_l1"]),
                "max_absolute_residual": float(values["species_residual_max"]),
                "max_relative_residual": float(values["species_relative_max"]),
            },
            "volume_fraction": {
                "basis": "sum(raw volume fractions) - 1",
                "integrated_absolute_residual": float(values["volume_residual_l1"]),
                "max_absolute_residual": float(values["volume_residual_max"]),
            },
        },
        "invalid_cells": {
            "any": int(values["invalid_any"]), "nonfinite": int(values["nonfinite"]),
            "nonpositive_total_density": int(values["nonpositive_density"]),
            "negative_partial_density": int(values["negative_partial_density"]),
            "negative_species_density": int(values["negative_species_density"]),
            "volume_fraction_out_of_bounds": int(values["volume_fraction_out_of_bounds"]),
        },
        "physical_state": {
            "ranges": {
                name: {
                    "minimum": float(physical_minima[name]) if physical_counts[name] else None,
                    "maximum": float(physical_maxima[name]) if physical_counts[name] else None,
                    "finite_cell_count": int(physical_counts[name]),
                }
                for name in physical_names
            },
            "mask_counts": {name: int(mask_counts[name]) for name in mask_names},
        },
        "source_files": files,
    }
