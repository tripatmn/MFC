"""Optional cell-local Cantera heat-release evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import PhysicalState, State
from .reconstruction import Model3Configuration


CHECK_RTOL = 1.0e-8
CHECK_ATOL_W_M3 = 1.0e-6


@dataclass(frozen=True)
class HeatReleaseResult:
    net: float
    positive: float
    negative: float
    positive_area: float
    maximum: float | None
    x_max: float | None
    y_max: float | None
    evaluated_cells: int
    check_mismatch_cells: int
    check_sign_mismatch_cells: int
    check_max_abs_error_W_m3: float
    check_max_relative_error: float


class CanteraHeatRelease:
    def __init__(self, config: Model3Configuration):
        try:
            import cantera as ct
        except ImportError as exc:
            raise ValueError(
                "--compute-heat-release cantera requires the Cantera Python package"
            ) from exc
        try:
            self.gas = ct.Solution(config.mechanism_path, config.mechanism_phase)
        except Exception as exc:
            raise ValueError(
                f"failed to load Cantera phase {config.mechanism_phase!r} from "
                f"{config.mechanism_path}: {exc}"
            ) from exc
        cantera_names = tuple(self.gas.species_names)
        missing = [name for name in cantera_names if name not in config.species_names]
        extra = [name for name in config.species_names if name not in cantera_names]
        if missing or extra:
            raise ValueError(
                "full HRR species mapping does not match the Cantera phase; "
                f"missing from reconstructed MFC state={missing}, extra in MFC state={extra}"
            )
        self.species_names = cantera_names
        self.config = config

    def evaluate(self, state: State, physical: PhysicalState) -> HeatReleaseResult:
        if state.grid is None or state.grid.dimensions != 2:
            raise ValueError("Cantera heat-release integration currently requires a 2D grid")
        weights = np.asarray(state.grid.cell_measures, dtype=np.float64).reshape(-1)
        fields = {
            name: np.asarray(field.values, dtype=np.float64).reshape(-1)
            for name, field in physical.fields.items()
        }
        masks = {
            name: np.asarray(field.values, dtype=bool).reshape(-1)
            for name, field in physical.masks.items()
        }
        missing = [
            name for name in self.species_names if f"Y[{name}]" not in fields
        ]
        if missing:
            raise ValueError(
                "cannot compute Cantera HRR because the reconstructed full species vector is "
                f"incomplete; missing species={missing}"
            )
        temperature = fields["temperature"]
        gas_density = fields["gas_density"]
        alpha_liq = fields[f"alpha[{self.config.liquid_fluid_id}]"]
        species = np.stack([fields[f"Y[{name}]"] for name in self.species_names], axis=1)
        valid = (
            masks["mask.chemistry_valid"]
            & (alpha_liq <= self.config.thresholds.liquid_dominated_liquid_alpha_min)
            & np.isfinite(temperature) & np.isfinite(gas_density) & (gas_density > 0.0)
            & np.all(np.isfinite(species), axis=1)
            & np.isfinite(weights) & (weights > 0.0)
        )
        x, y = _cell_coordinates(state)
        qdot = np.full(weights.shape, np.nan)
        mismatch = sign_mismatch = 0
        max_abs_error = max_relative_error = 0.0
        for cell in np.flatnonzero(valid):
            try:
                self.gas.TDY = temperature[cell], gas_density[cell], species[cell]
                value = float(self.gas.heat_release_rate)
                check = float(-np.dot(
                    np.asarray(self.gas.net_production_rates, dtype=np.float64),
                    np.asarray(self.gas.partial_molar_enthalpies, dtype=np.float64),
                ))
            except Exception as exc:
                raise ValueError(
                    f"Cantera HRR failed for saved index {state.saved_index}, local cell {cell}: {exc}"
                ) from exc
            if not np.isfinite(value) or not np.isfinite(check):
                raise ValueError(
                    f"Cantera HRR returned a non-finite value for saved index "
                    f"{state.saved_index}, local cell {cell}"
                )
            qdot[cell] = value
            error = abs(value - check)
            scale = max(abs(value), abs(check), CHECK_ATOL_W_M3)
            relative = error / scale
            max_abs_error = max(max_abs_error, error)
            max_relative_error = max(max_relative_error, relative)
            if error > CHECK_ATOL_W_M3 + CHECK_RTOL * max(abs(value), abs(check)):
                mismatch += 1
            if value * check < 0.0 and max(abs(value), abs(check)) > CHECK_ATOL_W_M3:
                sign_mismatch += 1
        finite = np.isfinite(qdot)
        positive = finite & (qdot > 0.0)
        negative = finite & (qdot < 0.0)
        if np.any(finite):
            local_index = int(np.flatnonzero(finite)[np.argmax(qdot[finite])])
            maximum = float(qdot[local_index])
            x_max, y_max = float(x[local_index]), float(y[local_index])
        else:
            maximum = x_max = y_max = None
        return HeatReleaseResult(
            net=float(np.sum(qdot[finite] * weights[finite], dtype=np.float64)),
            positive=float(np.sum(qdot[positive] * weights[positive], dtype=np.float64)),
            negative=float(np.sum(qdot[negative] * weights[negative], dtype=np.float64)),
            positive_area=float(np.sum(weights[positive], dtype=np.float64)),
            maximum=maximum, x_max=x_max, y_max=y_max,
            evaluated_cells=int(np.count_nonzero(finite)),
            check_mismatch_cells=mismatch,
            check_sign_mismatch_cells=sign_mismatch,
            check_max_abs_error_W_m3=max_abs_error,
            check_max_relative_error=max_relative_error,
        )


def _cell_coordinates(state: State) -> tuple[np.ndarray, np.ndarray]:
    grid = state.grid
    if "cell_x" in grid.centers and "cell_y" in grid.centers:
        return (
            np.asarray(grid.centers["cell_x"], dtype=np.float64),
            np.asarray(grid.centers["cell_y"], dtype=np.float64),
        )
    if "x" not in grid.centers or "y" not in grid.centers:
        raise ValueError(
            f"saved index {state.saved_index}: x/y cell centers are unavailable for HRR max location"
        )
    x = np.asarray(grid.centers["x"], dtype=np.float64)
    y = np.asarray(grid.centers["y"], dtype=np.float64)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="xy")
    return mesh_x.ravel(order="C"), mesh_y.ravel(order="C")
