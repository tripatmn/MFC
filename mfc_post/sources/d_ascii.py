"""Inspector for simulation-written ASCII D output."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..models import DataSource, SourceReport
from ..timeline import make_timeline
from .base import precision_label


_FIELD = re.compile(r"^(cons|prim)\.(\d+)\.(\d+)\.(\d+)\.dat$")


class DAsciiSource(DataSource):
    family = "D"

    @property
    def path(self) -> Path:
        return self.case_path / "D"

    def available(self) -> bool:
        return self.path.is_dir()

    def observed_field_count(self) -> int | None:
        maximum = 0
        if not self.available():
            return None
        for path in self.path.iterdir():
            match = _FIELD.match(path.name)
            if match and match.group(1) == "cons":
                maximum = max(maximum, int(match.group(2)))
        return maximum or None

    def inspect(self) -> SourceReport:
        states: set[int] = set()
        fields: set[str] = set()
        state_ranks: dict[int, set[int]] = defaultdict(set)
        state_field_counts: dict[int, int] = defaultdict(int)
        warnings: list[str] = [
            "ASCII coordinates are written from cell-boundary arrays at cell indices, not canonical cell centers",
            "prim.<index> semantics vary by equation family and, for chemistry, by dimensionality",
        ]
        for path in self.path.iterdir():
            match = _FIELD.match(path.name)
            if not match:
                continue
            kind, index, rank, saved = match.groups()
            saved_i = int(saved)
            states.add(saved_i)
            fields.add(f"{kind}.{int(index)}")
            state_ranks[saved_i].add(int(rank))
            state_field_counts[saved_i] += 1
        if states:
            rank_counts = {len(value) for value in state_ranks.values()}
            if len(rank_counts) > 1:
                warnings.append("rank coverage differs between saved indices")
            if len(set(state_field_counts.values())) > 1:
                warnings.append("field-file coverage differs between saved indices; output may be incomplete")
        params = dict(self.metadata.parameters)
        grid_shape = self.metadata.grid_shape
        grid = None
        if grid_shape:
            from ..models import Grid, Provenance
            grid = Grid(
                dimensions=len(grid_shape), shape=grid_shape,
                provenance=Provenance(self.family, str(self.path), notes=("shape from case metadata; D coordinates were not promoted to cell centers",)),
            )
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="D/{cons,prim}.<equation_index>.<rank:02>.<saved_index:06>.dat; ASCII coordinate/value rows",
            timeline=make_timeline(states, params),
            grid=grid,
            precision=precision_label(params),
            downsampled=False,
            fields=tuple(sorted(fields, key=_field_sort)),
            raw_fields=True,
            restart_suitability="not an MFC restart/checkpoint format",
            postprocess_suitability="compatibility/diagnostic input only; official post_process does not read D",
            missing_metadata=("cell-center coordinates", "units", "field semantics beyond numeric equation index"),
            warnings=tuple(warnings),
            details={"raw_field_count": self.observed_field_count(), "rank_counts_by_state": sorted({len(v) for v in state_ranks.values()})},
        )


def _field_sort(value: str) -> tuple[int, int]:
    kind, index = value.split(".")
    return (0 if kind == "cons" else 1, int(index))
