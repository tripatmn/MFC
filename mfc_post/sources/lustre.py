"""Inspect shared-file and file-per-process MPI checkpoint layouts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import bool_param
from ..models import DataSource, Field, Grid, Provenance, SourceReport, State
from .base import grid_from_bounds
from ..timeline import make_timeline
from .base import precision_label


_SHARED = re.compile(r"^lustre_(\d+)\.dat$")
_DIR = re.compile(r"^lustre_(\d+)$")


class LustreSharedSource(DataSource):
    family = "lustre_shared"

    @property
    def path(self) -> Path:
        return self.case_path / "restart_data"

    def _files(self) -> list[tuple[int, Path]]:
        if not self.path.is_dir():
            return []
        return sorted((int(m.group(1)), p) for p in self.path.iterdir() if p.is_file() and (m := _SHARED.match(p.name)))

    def available(self) -> bool:
        return bool(self._files())

    def inspect(self) -> SourceReport:
        files = self._files()
        params = dict(self.metadata.parameters)
        warnings: list[str] = []
        expected_bytes = None
        if self.metadata.grid_shape and self.metadata.equation_layout:
            cells = 1
            for size in self.metadata.grid_shape:
                cells *= size
            expected_bytes = cells * len(self.metadata.equation_layout) * 8
            mismatches = [(index, p.stat().st_size) for index, p in files if p.stat().st_size != expected_bytes]
            if mismatches:
                warnings.append(f"checkpoint sizes differ from float64 full-resolution expectation {expected_bytes}: {mismatches[:5]}")
        if not bool_param(params, "parallel_io"):
            warnings.append("Lustre files exist but current case metadata says parallel_io=F; likely stale or copied output")
        try:
            grid = self.read_grid(files[0][0]) if files else None
        except (OSError, ValueError) as exc:
            grid = _metadata_grid(self)
            warnings.append(f"could not load actual Lustre grid; using metadata shape: {exc}")
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="restart_data/lustre_<saved_index>.dat plus markerless lustre_{x,y,z}_cb.dat; global variable-major MPI-IO",
            timeline=make_timeline((index for index, _ in files), params, saved_output=True),
            grid=grid,
            precision=precision_label(params, raw=True),
            downsampled=bool_param(params, "down_sample"),
            fields=tuple(item["name"] for item in self.metadata.equation_layout),
            raw_fields=True,
            restart_suitability="authoritative shared-file MFC restart format",
            postprocess_suitability="authoritative parallel input used by MFC post_process; process count may differ",
            missing_metadata=("physical time is not stored in the filename", "dtype/endian are native and not self-described"),
            warnings=tuple(warnings),
            details={"expected_full_resolution_bytes": expected_bytes, "file_sizes": sorted({p.stat().st_size for _, p in files})},
        )

    def read_grid(self, saved_index: int) -> Grid:
        del saved_index
        axes: dict[str, np.ndarray] = {}
        for axis in ("x", "y", "z"):
            path = self.path / f"lustre_{axis}_cb.dat"
            if path.is_file():
                values = np.fromfile(path, dtype=np.float64)
                if values.size < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                    raise ValueError(f"invalid markerless Lustre {axis} boundaries: {path}")
                axes[axis] = values
        if not axes:
            raise FileNotFoundError(f"no markerless Lustre grid files under {self.path}")
        return grid_from_bounds(
            self.family, self.path, axes,
            ("native markerless float64 inferred from the working-branch MPI writer",),
        )

    def state_file(self, saved_index: int) -> Path:
        path = self.path / f"lustre_{saved_index}.dat"
        if not path.is_file():
            raise FileNotFoundError(
                f"shared Lustre saved index {saved_index}: missing {path}"
            )
        return path

    def read_chunk(
        self, saved_index: int, start: int, stop: int,
        fields: tuple[str, ...] | list[str] | None = None, mpi_comm=None,
    ) -> State:
        if bool_param(dict(self.metadata.parameters), "down_sample"):
            raise ValueError("shared Lustre payload loading does not yet support down_sample=T")
        grid = self.read_grid(saved_index)
        cell_count = int(np.prod(grid.shape, dtype=np.int64))
        if start < 0 or stop < start or stop > cell_count:
            raise ValueError(f"invalid shared Lustre flat range [{start}, {stop}) for {cell_count} cells")
        path = self.state_file(saved_index)
        selected = set(fields) if fields is not None else None
        items = [item for item in self.metadata.equation_layout if selected is None or item["name"] in selected]
        expected_bytes = cell_count * len(self.metadata.equation_layout) * 8
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"shared Lustre saved index {saved_index}: {path} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes} for {len(self.metadata.equation_layout)} float64 fields"
            )
        loaded: dict[str, Field] = {}
        count = stop - start
        mpi_file = None
        if mpi_comm is not None:
            from mpi4py import MPI
            mpi_file = MPI.File.Open(mpi_comm, str(path), MPI.MODE_RDONLY)
        try:
            for item in items:
                index, name = int(item["index"]), str(item["name"])
                offset = ((index - 1) * cell_count + start) * 8
                values = np.empty(count, dtype=np.float64)
                if mpi_file is None:
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        actual = stream.readinto(values)
                    if actual != count * 8:
                        raise ValueError(
                            f"shared Lustre saved index {saved_index}, field {name}, range [{start},{stop}): short read"
                        )
                else:
                    mpi_file.Read_at_all(offset, values)
                loaded[name] = Field(
                    name=name, representation=str(item.get("representation", "conservative")),
                    values=values,
                    provenance=Provenance(
                        self.family, str(path), source_field=f"variable block {index}",
                        notes=(f"global flat cell range [{start}, {stop})",),
                    ),
                )
        finally:
            if mpi_file is not None:
                mpi_file.Close()
        measures = _global_cell_measures(grid)[start:stop]
        local_grid = Grid(
            dimensions=grid.dimensions, shape=(count,),
            cell_measures=tuple(measures.tolist()),
            provenance=Provenance(
                self.family, str(self.path),
                transformations=("cell measures derived from actual Lustre boundaries",),
                notes=(f"flattened global C-order range [{start}, {stop})", f"global shape {grid.shape}"),
            ),
        )
        clock = make_timeline(
            (saved_index,), dict(self.metadata.parameters), saved_output=True,
        )
        return State(
            saved_index=saved_index, simulation_step=clock.simulation_steps[0], physical_time=clock.physical_times[0],
            grid=local_grid, fields=loaded,
            provenance=Provenance(
                self.family, str(path), notes=(f"global flat range [{start}, {stop})",)
            ),
        )

    def read_state(self, saved_index: int, fields=None) -> State:
        grid = self.read_grid(saved_index)
        return self.read_chunk(saved_index, 0, int(np.prod(grid.shape)), fields)


class LustrePerProcessSource(DataSource):
    family = "lustre_per_process"

    @property
    def path(self) -> Path:
        return self.case_path / "restart_data"

    def _dirs(self) -> list[tuple[int, Path]]:
        if not self.path.is_dir():
            return []
        return sorted((int(m.group(1)), p) for p in self.path.iterdir() if p.is_dir() and (m := _DIR.match(p.name)))

    def available(self) -> bool:
        return bool(self._dirs())

    def inspect(self) -> SourceReport:
        dirs = self._dirs()
        params = dict(self.metadata.parameters)
        warnings: list[str] = []
        rank_counts = []
        for index, path in dirs:
            files = list(path.glob(f"{index}_*.dat"))
            rank_counts.append(len(files))
            ranks = sorted(_rank_from_name(p.name) for p in files)
            if ranks and ranks != list(range(ranks[-1] + 1)):
                warnings.append(f"saved index {index} has non-contiguous rank files")
        if rank_counts and len(set(rank_counts)) > 1:
            warnings.append("per-process rank count changes between saved indices")
        if not bool_param(params, "file_per_process"):
            warnings.append("per-process Lustre directories exist but current metadata says file_per_process=F")
        try:
            grid = LustreSharedSource(self.case_path, self.metadata).read_grid(dirs[0][0]) if dirs else None
        except (OSError, ValueError):
            grid = _metadata_grid(self)
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="restart_data/lustre_<saved_index>/<saved_index>_<rank:07>.dat; markerless rank-local variable sequence",
            timeline=make_timeline((index for index, _ in dirs), params, saved_output=True),
            grid=grid,
            precision=precision_label(params, raw=True),
            downsampled=bool_param(params, "down_sample"),
            fields=tuple(item["name"] for item in self.metadata.equation_layout),
            raw_fields=True,
            restart_suitability="MFC restart format only with the original process decomposition",
            postprocess_suitability="officially supported only with the same number/decomposition of ranks",
            missing_metadata=("rank decomposition", "native dtype/endian", "physical time"),
            warnings=tuple(warnings),
            details={"rank_counts": rank_counts},
        )

    def partition_ids(self, saved_index: int) -> tuple[int, ...]:
        directory = self.path / f"lustre_{saved_index}"
        return tuple(sorted(
            _rank_from_name(path.name) for path in directory.glob(f"{saved_index}_*.dat")
        )) if directory.is_dir() else ()

    def read_partition(self, saved_index: int, partition: int, fields=None) -> State:
        partitions = self.partition_ids(saved_index)
        if len(partitions) != 1 or partitions != (partition,):
            raise ValueError(
                "per-process Lustre payload geometry is not self-described; processing requires "
                "an unambiguous original MFC Cartesian decomposition, which could not be established"
            )
        global_grid = LustreSharedSource(self.case_path, self.metadata).read_grid(saved_index)
        path = self.path / f"lustre_{saved_index}" / f"{saved_index}_{partition:07d}.dat"
        cell_count = int(np.prod(global_grid.shape, dtype=np.int64))
        expected_bytes = cell_count * len(self.metadata.equation_layout) * 8
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"single-rank per-process Lustre file has {path.stat().st_size} bytes; expected {expected_bytes}")
        selected = set(fields) if fields is not None else None
        raw = np.memmap(path, dtype=np.float64, mode="r", shape=(len(self.metadata.equation_layout), cell_count))
        loaded = {
            str(item["name"]): Field(
                str(item["name"]), str(item.get("representation", "conservative")),
                values=np.asarray(raw[int(item["index"]) - 1]).copy(),
                provenance=Provenance(self.family, str(path), source_field=f"variable block {item['index']}")
            )
            for item in self.metadata.equation_layout
            if selected is None or item["name"] in selected
        }
        clock = make_timeline(
            (saved_index,), dict(self.metadata.parameters), saved_output=True,
        )
        return State(
            saved_index, clock.simulation_steps[0], clock.physical_times[0], global_grid, loaded,
            Provenance(self.family, str(path), notes=(f"stored MFC rank {partition}",)),
        )


def _metadata_grid(source: DataSource) -> Grid | None:
    if not source.metadata.grid_shape:
        return None
    return Grid(
        dimensions=len(source.metadata.grid_shape), shape=source.metadata.grid_shape,
        provenance=Provenance(source.family, str(source.case_path / "restart_data"), notes=("shape from case metadata; Phase 1 does not load markerless Lustre coordinates",)),
    )


def _rank_from_name(name: str) -> int:
    match = re.search(r"_(\d+)\.dat$", name)
    return int(match.group(1)) if match else -1


def _global_cell_measures(grid: Grid) -> np.ndarray:
    ordered = [np.asarray(grid.widths[axis]) for axis in ("z", "y", "x") if axis in grid.widths]
    result = ordered[0]
    for values in ordered[1:]:
        result = np.multiply.outer(result, values)
    return np.asarray(result).ravel(order="C")
