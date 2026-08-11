"""Inspector for official Silo-HDF5 visualization output."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import DataSource, Grid, Provenance, SourceReport
from ..timeline import make_timeline
from .base import precision_label
from .binary import _selected_postprocess_fields


_LOCAL = re.compile(r"^(\d+)\.silo$")
_ROOT = re.compile(r"^collection_(\d+)\.silo$")


class SiloSource(DataSource):
    family = "silo_hdf5"

    @property
    def path(self) -> Path:
        return self.case_path / "silo_hdf5"

    def available(self) -> bool:
        return self.path.is_dir()

    def inspect(self) -> SourceReport:
        rank_dirs = sorted(p for p in self.path.glob("p*") if p.is_dir())
        local_states: set[int] = set()
        local_coverage: dict[int, int] = {}
        suspect: list[str] = []
        for rank in rank_dirs:
            for file_path in rank.iterdir():
                match = _LOCAL.match(file_path.name)
                if not match:
                    continue
                saved = int(match.group(1))
                local_states.add(saved)
                local_coverage[saved] = local_coverage.get(saved, 0) + 1
                if file_path.stat().st_size < 4096:
                    suspect.append(str(file_path))
        root_states = set()
        root = self.path / "root"
        if root.is_dir():
            root_states = {int(m.group(1)) for p in root.iterdir() if p.is_file() and (m := _ROOT.match(p.name))}
        warnings: list[str] = []
        if local_states != root_states:
            warnings.append(f"local and collection saved indices differ: local_only={sorted(local_states-root_states)}, root_only={sorted(root_states-local_states)}")
        if local_coverage and len(set(local_coverage.values())) > 1:
            warnings.append("rank-file coverage differs between saved indices")
        if suspect:
            warnings.append(f"{len(suspect)} very small Silo files may be incomplete (first: {suspect[0]})")
        params = dict(self.metadata.parameters)
        grid = None
        if self.metadata.grid_shape:
            grid = Grid(
                dimensions=len(self.metadata.grid_shape), shape=self.metadata.grid_shape,
                provenance=Provenance(self.family, str(self.path), notes=("physical shape from case metadata; Silo local files may include ghost zones",)),
            )
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="silo_hdf5/p<rank>/<saved_index>.silo plus root/collection_<saved_index>.silo; rectilinear mesh and zone-centered variables",
            timeline=make_timeline(sorted(local_states | root_states), params),
            grid=grid,
            precision=precision_label(params),
            downsampled=None,
            fields=_selected_postprocess_fields(params, self.metadata.equation_layout),
            raw_fields=False,
            restart_suitability="not suitable; selected/derived visualization output",
            postprocess_suitability="already post-processed and visualization-ready",
            missing_metadata=("physical time unless supplied externally", "Phase 1 does not decode the Silo object table"),
            warnings=tuple(warnings),
            details={"rank_count": len(rank_dirs), "collection_state_count": len(root_states), "suspect_small_files": len(suspect)},
        )
