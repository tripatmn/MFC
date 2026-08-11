"""Inspector and first concrete grid reader for serial MFC raw output."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from ..models import DataSource, Field, Grid, Provenance, SourceReport, State
from ..timeline import make_timeline
from .base import grid_from_bounds, precision_label


_Q_FILE = re.compile(r"q_cons_vf(\d+)\.dat$")


class PAllSource(DataSource):
    family = "p_all"

    @property
    def path(self) -> Path:
        return self.case_path / "p_all"

    def available(self) -> bool:
        return self.path.is_dir()

    def _ranks(self) -> list[Path]:
        return sorted((p for p in self.path.glob("p*") if p.is_dir()), key=lambda p: _suffix_int(p.name))

    def _states_by_rank(self) -> dict[int, set[int]]:
        result: dict[int, set[int]] = {}
        for rank_path in self._ranks():
            rank = _suffix_int(rank_path.name)
            result[rank] = {int(p.name) for p in rank_path.iterdir() if p.is_dir() and p.name.isdigit()}
        return result

    def observed_field_count(self) -> int | None:
        ranks = self._ranks()
        if not ranks:
            return None
        states = sorted(p for p in ranks[0].iterdir() if p.is_dir() and p.name.isdigit())
        if not states:
            return None
        indices = [int(match.group(1)) for p in states[0].iterdir() if (match := _Q_FILE.match(p.name))]
        return max(indices) if indices else None

    def partition_ids(self, saved_index: int) -> tuple[int, ...]:
        return tuple(
            _suffix_int(rank.name) for rank in self._ranks()
            if (rank / str(saved_index)).is_dir()
        )

    def inspect(self) -> SourceReport:
        by_rank = self._states_by_rank()
        all_states = sorted(set().union(*by_rank.values())) if by_rank else []
        warnings: list[str] = []
        missing: list[str] = []
        if not by_rank:
            warnings.append("p_all exists but contains no rank directories")
        else:
            expected = set(all_states)
            incomplete = {rank: sorted(expected - states) for rank, states in by_rank.items() if states != expected}
            if incomplete:
                warnings.append(f"incomplete rank/state coverage: {incomplete}")
        grid = None
        if all_states:
            try:
                grid = self.read_grid(all_states[0])
            except (OSError, ValueError, struct.error) as exc:
                warnings.append(f"could not read representative grid: {exc}")
        precision = precision_label(dict(self.metadata.parameters), raw=True)
        if grid and grid.provenance:
            inferred = next((note for note in grid.provenance.notes if "-bit reals" in note), None)
            if inferred:
                precision = f"inferred from representative grid record: {inferred}"
        count = self.observed_field_count()
        fields = tuple(item["name"] for item in self.metadata.equation_layout)
        if count is None:
            missing.append("raw equation count")
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="p_all/p<rank>/<saved_index>/{x,y,z}_cb.dat and q_cons_vf<N>.dat; one native sequential-unformatted record per file",
            timeline=make_timeline(all_states, dict(self.metadata.parameters)),
            grid=grid,
            precision=precision,
            downsampled=False,
            fields=fields,
            raw_fields=True,
            restart_suitability="suitable for serial MFC restart/pre-process input when configuration and decomposition match",
            postprocess_suitability="authoritative serial input used by MFC post_process",
            missing_metadata=tuple(missing + ["record-marker ABI and endian are inferred from representative files, not stored"]),
            warnings=tuple(warnings),
            details={"rank_count": len(by_rank), "raw_field_count": count, "record_format": "native Fortran sequential-unformatted"},
        )

    def read_grid(self, saved_index: int) -> Grid:
        axes: dict[str, list[float]] = {}
        formats: set[str] = set()
        found = False
        for rank_path in self._ranks():
            state_path = rank_path / str(saved_index)
            if not state_path.is_dir():
                continue
            for axis in ("x", "y", "z"):
                file_path = state_path / f"{axis}_cb.dat"
                if not file_path.is_file():
                    continue
                values, description = _read_fortran_real_record(file_path)
                axes.setdefault(axis, []).extend(values)
                formats.add(description)
                found = True
        if not found:
            raise ValueError(f"no p_all grid files found for saved index {saved_index}")
        return grid_from_bounds(self.family, self.path, axes, tuple(sorted(formats)))

    def read_partition(
        self, saved_index: int, partition: int, fields: tuple[str, ...] | list[str] | None = None
    ) -> State:
        return self._read_partition_range(saved_index, partition, fields, None)

    def read_partition_range(
        self, saved_index: int, partition: int, start: int, stop: int,
        fields: tuple[str, ...] | list[str] | None = None,
    ) -> State:
        return self._read_partition_range(saved_index, partition, fields, (start, stop))

    def partition_cell_count(self, saved_index: int, partition: int) -> int:
        path = self.path / f"p{partition}" / str(saved_index) / "q_cons_vf1.dat"
        count, _, _, _ = _fortran_record_layout(path)
        return count

    def _read_partition_range(self, saved_index: int, partition: int, fields, flat_range) -> State:
        state_path = self.path / f"p{partition}" / str(saved_index)
        if not state_path.is_dir():
            raise FileNotFoundError(
                f"p_all saved index {saved_index}, stored rank {partition}: missing {state_path}"
            )
        axes: dict[str, np.ndarray] = {}
        format_notes: list[str] = []
        for axis in ("x", "y", "z"):
            path = state_path / f"{axis}_cb.dat"
            if path.is_file():
                values, description = _read_fortran_real_record_numpy(path)
                axes[axis] = values
                format_notes.append(description)
        if not axes:
            raise ValueError(
                f"p_all saved index {saved_index}, stored rank {partition}: no grid records"
            )
        shape_xyz = tuple(len(axes[axis]) - 1 for axis in ("x", "y", "z") if axis in axes)
        canonical_shape = tuple(reversed(shape_xyz))
        widths = {axis: np.diff(values) for axis, values in axes.items()}
        measure = _cell_measures(widths)
        full_measure = measure.ravel(order="C")
        range_note = None
        if flat_range is not None:
            start, stop = flat_range
            if start < 0 or stop < start or stop > full_measure.size:
                raise ValueError(
                    f"p_all saved index {saved_index}, stored rank {partition}: invalid flat range [{start},{stop})"
                )
            local_measure = full_measure[start:stop]
            local_shape = (stop - start,)
            range_note = f"rank-local flat C-order range [{start}, {stop})"
        else:
            start, stop = 0, full_measure.size
            local_measure = full_measure
            local_shape = canonical_shape
        grid = Grid(
            dimensions=len(shape_xyz),
            shape=local_shape,
            bounds={} if flat_range is not None else {axis: tuple(values.tolist()) for axis, values in axes.items()},
            centers={} if flat_range is not None else {axis: tuple(((values[:-1] + values[1:]) * 0.5).tolist()) for axis, values in axes.items()},
            widths={} if flat_range is not None else {axis: tuple(values.tolist()) for axis, values in widths.items()},
            cell_measures=tuple(local_measure.tolist()),
            nonuniform_axes=tuple(
                axis for axis, values in widths.items()
                if values.size and np.ptp(values) > max(float(np.max(np.abs(values))), 1.0) * 1e-12
            ),
            provenance=Provenance(
                self.family, str(state_path),
                transformations=("cell centers and measures derived from rank-local boundaries",),
                notes=tuple(sorted(set(format_notes))) + ((range_note,) if range_note else ()),
            ),
        )
        selected = set(fields) if fields is not None else None
        loaded: dict[str, Field] = {}
        for item in self.metadata.equation_layout:
            name, index = str(item["name"]), int(item["index"])
            if selected is not None and name not in selected:
                continue
            path = state_path / f"q_cons_vf{index}.dat"
            if not path.is_file():
                raise FileNotFoundError(
                    f"p_all saved index {saved_index}, stored rank {partition}, field {name}: missing {path}"
                )
            expected = int(np.prod(canonical_shape, dtype=np.int64))
            count, offset, dtype, description = _fortran_record_layout(path)
            if count != expected:
                raise ValueError(
                    f"p_all saved index {saved_index}, stored rank {partition}, field {name}: "
                    f"{count} values, expected {expected}"
                )
            values = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))[start:stop]
            values = np.asarray(values, dtype=np.float64).copy()
            loaded[name] = Field(
                name=name, representation=str(item.get("representation", "conservative")),
                values=values.reshape(local_shape, order="C"),
                provenance=Provenance(
                    self.family, str(path), source_field=f"q_cons_vf{index}", notes=(description,)
                ),
            )
        clock = make_timeline((saved_index,), dict(self.metadata.parameters))
        return State(
            saved_index=saved_index, simulation_step=clock.simulation_steps[0], physical_time=clock.physical_times[0],
            grid=grid, fields=loaded,
            provenance=Provenance(
                self.family, str(state_path),
                notes=(f"stored MFC rank {partition}", "raw conservative partition"),
            ),
        )

    def read_state(self, saved_index: int, fields=None) -> State:
        partitions = self.partition_ids(saved_index)
        if partitions != (0,):
            raise ValueError(
                "read_state would combine p_all partitions; use read_partition explicitly"
            )
        return self.read_partition(saved_index, 0, fields)


def _read_fortran_real_record(path: Path) -> tuple[tuple[float, ...], str]:
    data = path.read_bytes()
    for endian, endian_name in (("<", "little-endian"), (">", "big-endian")):
        for marker_size, marker_code in ((4, "I"), (8, "Q")):
            if len(data) < marker_size * 2:
                continue
            payload_size = struct.unpack_from(endian + marker_code, data, 0)[0]
            if payload_size + 2 * marker_size != len(data):
                continue
            trailing = struct.unpack_from(endian + marker_code, data, marker_size + payload_size)[0]
            if trailing != payload_size:
                continue
            for real_size, real_code in ((8, "d"), (4, "f")):
                if payload_size % real_size:
                    continue
                count = payload_size // real_size
                values = struct.unpack_from(endian + str(count) + real_code, data, marker_size)
                if all(value == value and abs(value) != float("inf") for value in values):
                    return values, f"{marker_size}-byte markers, {real_size * 8}-bit reals, {endian_name}"
    raise ValueError(f"unrecognized Fortran sequential record: {path}")


def _read_fortran_real_record_numpy(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    for endian, endian_name in (("<", "little-endian"), (">", "big-endian")):
        for marker_size, marker_code in ((4, "I"), (8, "Q")):
            if len(data) < marker_size * 2:
                continue
            payload_size = struct.unpack_from(endian + marker_code, data, 0)[0]
            if payload_size + 2 * marker_size != len(data):
                continue
            trailing = struct.unpack_from(endian + marker_code, data, marker_size + payload_size)[0]
            if trailing != payload_size:
                continue
            for real_size, code in ((8, "f8"), (4, "f4")):
                if payload_size % real_size:
                    continue
                values = np.frombuffer(data, dtype=np.dtype(endian + code), count=payload_size // real_size, offset=marker_size)
                if np.all(np.isfinite(values)):
                    return values.astype(np.float64, copy=False), f"{marker_size}-byte markers, {real_size * 8}-bit reals, {endian_name}"
    raise ValueError(f"unrecognized Fortran sequential record: {path}")


def _fortran_record_layout(path: Path) -> tuple[int, int, np.dtype, str]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(8)
    for endian, endian_name in (("<", "little-endian"), (">", "big-endian")):
        for marker_size, marker_code in ((4, "I"), (8, "Q")):
            if len(head) < marker_size:
                continue
            payload_size = struct.unpack(endian + marker_code, head[:marker_size])[0]
            if payload_size + 2 * marker_size != size:
                continue
            with path.open("rb") as stream:
                stream.seek(marker_size + payload_size)
                tail = stream.read(marker_size)
            if len(tail) != marker_size or struct.unpack(endian + marker_code, tail)[0] != payload_size:
                continue
            for real_size, code in ((8, "f8"), (4, "f4")):
                if payload_size % real_size == 0:
                    dtype = np.dtype(endian + code)
                    return payload_size // real_size, marker_size, dtype, f"{marker_size}-byte markers, {real_size * 8}-bit reals, {endian_name}"
    raise ValueError(f"unrecognized Fortran sequential record: {path}")


def _cell_measures(widths: dict[str, np.ndarray]) -> np.ndarray:
    ordered = [widths[axis] for axis in ("z", "y", "x") if axis in widths]
    result = ordered[0]
    for values in ordered[1:]:
        result = np.multiply.outer(result, values)
    return np.asarray(result)


def _suffix_int(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else -1
