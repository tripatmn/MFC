"""Inspector for official post_process format=2 output."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import DataSource, Grid, Provenance, SourceReport
from ..timeline import make_timeline
from .base import precision_label


_STATE = re.compile(r"^(\d+)\.dat$")


class BinarySource(DataSource):
    family = "binary"

    @property
    def path(self) -> Path:
        return self.case_path / "binary"

    def available(self) -> bool:
        return self.path.is_dir()

    def inspect(self) -> SourceReport:
        rank_dirs = sorted(p for p in self.path.glob("p*") if p.is_dir())
        states: set[int] = set()
        coverage: dict[int, int] = {}
        for rank in rank_dirs:
            rank_states = {int(m.group(1)) for p in rank.iterdir() if p.is_file() and (m := _STATE.match(p.name))}
            for saved in rank_states:
                coverage[saved] = coverage.get(saved, 0) + 1
            states.update(rank_states)
        warnings: list[str] = []
        if coverage and len(set(coverage.values())) > 1:
            warnings.append("rank-file coverage differs between saved indices")
        params = dict(self.metadata.parameters)
        grid = None
        if self.metadata.grid_shape:
            grid = Grid(
                dimensions=len(self.metadata.grid_shape), shape=self.metadata.grid_shape,
                provenance=Provenance(self.family, str(self.path), notes=("shape from case metadata; binary records were not fully decoded",)),
            )
        return SourceReport(
            family=self.family,
            path=str(self.path),
            layout="binary/p<rank>/<saved_index>.dat; sequential-unformatted header, grid record, then fixed-name field records (1D also binary/root)",
            timeline=make_timeline(states, params),
            grid=grid,
            precision=precision_label(params),
            downsampled=None,
            fields=_selected_postprocess_fields(params, self.metadata.equation_layout),
            raw_fields=False,
            restart_suitability="not suitable; selected visualization database, not a checkpoint",
            postprocess_suitability="already post-processed; useful only for selected fields",
            missing_metadata=("native Fortran record ABI/endian", "field precision must be validated from record lengths", "physical time"),
            warnings=tuple(warnings),
            details={"rank_count": len(rank_dirs), "root_files": len(list((self.path / "root").glob("*.dat"))) if (self.path / "root").is_dir() else 0},
        )


def _selected_postprocess_fields(params: dict, equation_layout: tuple[dict, ...]) -> tuple[str, ...]:
    names: set[str] = set()
    if params.get("cons_vars_wrt") is True:
        names.update(
            item["name"] for item in equation_layout
            if item.get("representation") == "conservative" and not item["name"].startswith("species_density[")
        )
    if params.get("prim_vars_wrt") is True:
        names.update({"pressure", "velocity"})
        names.update(
            item["name"].replace("species_density[", "species_mass_fraction[", 1)
            for item in equation_layout if item["name"].startswith("species_density[")
        )
    direct = {
        "rho_wrt": "rho", "pres_wrt": "pres", "e_wrt": "E", "gamma_wrt": "gamma",
        "heat_ratio_wrt": "heat_ratio", "pi_inf_wrt": "pi_inf", "pres_inf_wrt": "pres_inf",
        "c_wrt": "c", "schlieren_wrt": "schlieren", "qm_wrt": "qm",
        "liutex_wrt": "liutex", "chem_wrt_t": "T",
    }
    names.update(label for key, label in direct.items() if params.get(key) is True)
    for key, value in params.items():
        if value is not True:
            continue
        match = re.match(r"(alpha_rho|alpha|mom|vel|omega|chem_wrt_y)\((\d+)\)$", key)
        if match:
            if match.group(1) == "chem_wrt_y":
                names.add(f"species_mass_fraction[{match.group(2)}]")
            else:
                names.add(f"{match.group(1)}{match.group(2)}")
    return tuple(sorted(names))
