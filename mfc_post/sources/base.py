"""Helpers used by source inspectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..models import Grid, Provenance


def precision_label(params: dict[str, Any], raw: bool = False) -> str | None:
    if raw:
        return "working precision (native wp; normally float64), independent of formatted-output precision"
    value = params.get("precision")
    if value == 1:
        return "requested single precision"
    if value == 2:
        return "requested double precision"
    return None


def grid_from_bounds(family: str, path: Path, axes: dict[str, Iterable[float]], notes: tuple[str, ...] = ()) -> Grid:
    clean: dict[str, tuple[float, ...]] = {}
    centers: dict[str, tuple[float, ...]] = {}
    widths: dict[str, tuple[float, ...]] = {}
    nonuniform: list[str] = []
    shape: list[int] = []
    for axis in ("x", "y", "z"):
        if axis not in axes:
            continue
        values = _coalesce_boundaries(axes[axis])
        if len(values) < 2:
            continue
        clean[axis] = values
        axis_widths = tuple(b - a for a, b in zip(values, values[1:]))
        widths[axis] = axis_widths
        centers[axis] = tuple((a + b) * 0.5 for a, b in zip(values, values[1:]))
        shape.append(len(axis_widths))
        if axis_widths and max(axis_widths) - min(axis_widths) > max(abs(max(axis_widths)), 1.0) * 1e-12:
            nonuniform.append(axis)
    measures: tuple[float, ...] | None = None
    total_cells = 1
    for size in shape:
        total_cells *= size
    if total_cells <= 1_000_000 and shape:
        products = [1.0]
        for axis in clean:
            products = [old * width for old in products for width in widths[axis]]
        measures = tuple(products)
    return Grid(
        dimensions=len(shape), shape=tuple(shape), bounds=clean, centers=centers,
        widths=widths, cell_measures=measures, nonuniform_axes=tuple(nonuniform),
        provenance=Provenance(family, str(path), transformations=("cell centers and measures derived from actual boundaries",), notes=notes),
    )


def _coalesce_boundaries(raw_values: Iterable[float]) -> tuple[float, ...]:
    values = sorted(float(value) for value in raw_values)
    if not values:
        return ()
    span = values[-1] - values[0]
    tolerance = max(abs(span), abs(values[0]), abs(values[-1]), 1e-300) * 1e-12
    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(sum(group) / len(group) for group in groups)
