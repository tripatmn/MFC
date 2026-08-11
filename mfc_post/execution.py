"""Optional MPI execution without leaking MPI objects into canonical data models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class SerialComm:
    rank = 0
    size = 1

    def reduce(self, value, op=None, root=0):
        del op, root
        return value

    def gather(self, value, root=0):
        del root
        return [value]

    def allgather(self, value):
        return [value]


@dataclass(frozen=True)
class ExecutionContext:
    requested: str
    mode: str
    comm: Any
    mpi: Any = None

    @property
    def rank(self) -> int:
        return int(self.comm.rank)

    @property
    def size(self) -> int:
        return int(self.comm.size)

    @classmethod
    def create(cls, requested: str) -> "ExecutionContext":
        if requested == "serial":
            launched = _launcher_size()
            if launched > 1:
                raise RuntimeError(
                    f"--execution serial was launched with {launched} processes; run it without mpiexec"
                )
            return cls(requested, "serial", SerialComm())
        try:
            from mpi4py import MPI
        except ImportError as exc:
            if requested == "mpi" or _launcher_size() > 1:
                raise RuntimeError(
                    "MPI execution requires mpi4py; Open MPI alone is insufficient. "
                    "Install mpi4py in the mfc-post Python environment."
                ) from exc
            return cls(requested, "serial", SerialComm())
        comm = MPI.COMM_WORLD
        if requested == "mpi" or comm.size > 1:
            return cls(requested, "mpi", comm, MPI)
        return cls(requested, "serial", SerialComm())

    def strategy(self, state_count: int) -> str:
        if self.size == 1:
            return "serial"
        return "state" if state_count >= self.size else "spatial"


def split_range(length: int, size: int, rank: int) -> tuple[int, int]:
    base, extra = divmod(length, size)
    start = rank * base + min(rank, extra)
    return start, start + base + int(rank < extra)


def _launcher_size() -> int:
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        try:
            return int(os.environ.get(name, "1"))
        except ValueError:
            pass
    return 1
