"""Deterministic selection of raw sources supported by analyze and render."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import RunMetadata
from .sources import LustreSharedSource, PAllSource


SOURCE_TYPES = {"p_all": PAllSource, "lustre_shared": LustreSharedSource}


def select_raw_source(
    root: Path, metadata: RunMetadata, inspection: dict, requested: str = "auto",
):
    if requested not in {"auto", *SOURCE_TYPES}:
        raise ValueError(f"unsupported source selection {requested!r}")
    reports = {item["family"]: item for item in inspection["sources"]}
    reasons = {
        family: _support_reason(root, metadata, reports.get(family), family)
        for family in SOURCE_TYPES
    }
    candidates = ("p_all", "lustre_shared") if requested == "auto" else (requested,)
    for family in candidates:
        if reasons[family] is None:
            return SOURCE_TYPES[family](root, metadata), reports[family]
    detail = ". ".join(f"Tried {family}: {reasons[family]}" for family in candidates)
    raise ValueError(f"No supported analysis/render source found. {detail}.")


def _support_reason(root, metadata, report, family):
    if report is None:
        return "none found"
    indices = tuple(int(value) for value in report["timeline"]["saved_indices"])
    if not indices:
        return "no saved states found"
    source = SOURCE_TYPES[family](root, metadata)
    if family == "p_all":
        if any("incomplete rank/state coverage" in item for item in report.get("warnings", [])):
            return "rank/state coverage is incomplete"
        observed = report.get("details", {}).get("raw_field_count")
        if observed is not None and observed != len(metadata.equation_layout):
            return (
                f"representative save has {observed} raw fields; "
                f"expected {len(metadata.equation_layout)}"
            )
        for index in indices:
            if not source.partition_ids(index):
                return f"saved index {index} has no stored rank partitions"
        return None
    try:
        grid = source.read_grid(indices[0])
    except (OSError, ValueError) as exc:
        return f"grid is incomplete or invalid: {exc}"
    cells = int(np.prod(grid.shape, dtype=np.int64))
    expected = cells * len(metadata.equation_layout) * 8
    if not metadata.equation_layout:
        return "equation layout is unavailable"
    for index in indices:
        try:
            path = source.state_file(index)
            size = path.stat().st_size
        except OSError as exc:
            return f"saved index {index} is unavailable: {exc}"
        if size != expected:
            return f"saved index {index} has {size} bytes; expected {expected}"
    return None
