"""Canonical, deliberately small interfaces shared by input backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Provenance:
    source_family: str
    source_path: str
    source_field: str | None = None
    transformations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    execution: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Field:
    name: str
    representation: str
    centering: str = "cell"
    components: tuple[str, ...] = ()
    units: str | None = None
    values: Any = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class Grid:
    dimensions: int
    shape: tuple[int, ...]
    bounds: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    centers: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    widths: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    cell_measures: tuple[float, ...] | None = None
    nonuniform_axes: tuple[str, ...] = ()
    provenance: Provenance | None = None


@dataclass(frozen=True)
class Timeline:
    saved_indices: tuple[int, ...]
    physical_times: tuple[float | None, ...]
    time_basis: str
    simulation_steps: tuple[int | None, ...] = ()
    cadence: float | None = None
    segments: tuple[Mapping[str, Any], ...] = ()
    duplicates: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunMetadata:
    case_path: str
    parameters: Mapping[str, Any]
    dimensions: int | None
    grid_shape: tuple[int, ...] | None
    equation_layout: tuple[Mapping[str, Any], ...] = ()
    species_names: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class State:
    saved_index: int
    simulation_step: int | None
    physical_time: float | None
    grid: Grid | None
    fields: Mapping[str, Field]
    provenance: Provenance


@dataclass(frozen=True)
class PhysicalState:
    """Partition-local physical reconstruction layered over an immutable raw State."""

    raw_state: State
    fields: Mapping[str, Field]
    masks: Mapping[str, Field]
    provenance: Provenance


@dataclass(frozen=True)
class SourceReport:
    family: str
    path: str
    layout: str
    timeline: Timeline
    grid: Grid | None
    precision: str | None
    downsampled: bool | None
    fields: tuple[str, ...]
    raw_fields: bool
    restart_suitability: str
    postprocess_suitability: str
    missing_metadata: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.grid is not None:
            result["grid"] = _grid_summary(self.grid)
        return _jsonable(result)


class DataSource(ABC):
    """A source family is inspected independently and never implicitly merged."""

    family: str

    def __init__(self, case_path: Path, metadata: RunMetadata):
        self.case_path = case_path
        self.metadata = metadata

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def inspect(self) -> SourceReport:
        raise NotImplementedError

    def read_grid(self, saved_index: int) -> Grid:
        raise NotImplementedError(f"{self.family} grid reader is not implemented in Phase 1")

    def read_state(self, saved_index: int, fields: Sequence[str] | None = None) -> State:
        raise NotImplementedError(f"{self.family} state reader is not implemented in Phase 1")

    def partition_ids(self, saved_index: int) -> tuple[int, ...]:
        """Return stable source-partition identifiers for one saved state."""
        return (0,)

    def read_partition(
        self, saved_index: int, partition: int, fields: Sequence[str] | None = None
    ) -> State:
        """Load one spatial partition without depending on the execution mode."""
        if partition != 0:
            raise ValueError(f"{self.family} has no partition {partition}")
        return self.read_state(saved_index, fields)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def to_dict(value: Any) -> Any:
    return _jsonable(asdict(value))


def _grid_summary(grid: Grid) -> dict[str, Any]:
    axis_ranges = {
        axis: [values[0], values[-1]]
        for axis, values in grid.bounds.items() if values
    }
    width_ranges = {
        axis: [min(values), max(values)]
        for axis, values in grid.widths.items() if values
    }
    measures = None
    if grid.cell_measures:
        measures = {
            "count": len(grid.cell_measures),
            "min": min(grid.cell_measures),
            "max": max(grid.cell_measures),
            "sum": sum(grid.cell_measures),
        }
    return {
        "dimensions": grid.dimensions,
        "shape": grid.shape,
        "axis_ranges": axis_ranges,
        "width_ranges": width_ranges,
        "cell_measures": measures,
        "nonuniform_axes": grid.nonuniform_axes,
        "provenance": asdict(grid.provenance) if grid.provenance else None,
    }
