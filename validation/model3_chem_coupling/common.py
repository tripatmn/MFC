#!/usr/bin/env python3
"""Shared helpers for lightweight Model-3 chemistry-coupling validation cases."""

from __future__ import annotations

from pathlib import Path

SK54_CTFILE = "/home/tripatmn/mfc_work/MFC_alpha_diag/examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
SK54_CTPHASE = "yao_sk54"
SK54_FUEL = "NC12H26"
SK54_VASU_1422_MIXTURE = "NC12H26:0.001,O2:0.037,AR:0.962"
SK54_TRACE_FUEL_AIR = "NC12H26:1e-30,O2:0.21,N2:0.79"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_solution(ctfile: str, phase: str | None = None):
    import cantera as ct

    candidates = [
        Path(ctfile),
        Path(__file__).resolve().parent / ctfile,
        repo_root() / ctfile,
        repo_root() / "examples" / "2D_reactive_shock_bubble" / ctfile,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ct.Solution(str(candidate), phase) if phase else ct.Solution(str(candidate))

    return ct.Solution(ctfile, phase) if phase else ct.Solution(ctfile)


def species_index(sol, name: str) -> int:
    for candidate in (name, name.lower(), name.upper()):
        try:
            return sol.species_index(candidate)
        except ValueError:
            pass
    raise ValueError(f"Species '{name}' was not found in {sol.source}")


def set_species(case: dict, sol, patch_ids: list[int], key_species: list[str | int] | tuple[str | int, ...] = ()) -> None:
    key_indices = set()
    for species in key_species:
        key_indices.add(species if isinstance(species, int) else species_index(sol, species) + 1)
    for idx, value in enumerate(sol.Y, start=1):
        for patch_id in patch_ids:
            case[f"patch_icpp({patch_id})%Y({idx})"] = float(value)
        if not key_indices or idx in key_indices:
            case[f"chem_wrt_Y({idx})"] = "T"


def selected_species_ids(sol, names: list[str] | tuple[str, ...]) -> list[int]:
    return [species_index(sol, name) + 1 for name in names if _has_species(sol, name)]


def _has_species(sol, name: str) -> bool:
    try:
        species_index(sol, name)
    except ValueError:
        return False
    return True


def model3_output_flags(case: dict, num_fluids: int = 3) -> None:
    case.update(
        {
            "prim_vars_wrt": "T",
            "cons_vars_wrt": "T",
            "rho_wrt": "T",
            "pres_wrt": "T",
            "E_wrt": "T",
            "chem_wrt_T": "T",
        }
    )
    for fluid_id in range(1, num_fluids + 1):
        case[f"alpha_rho_wrt({fluid_id})"] = "T"
        case[f"alpha_wrt({fluid_id})"] = "T"
        case[f"alpha_rho_e_wrt({fluid_id})"] = "T"


def transformed_gamma(gamma: float) -> float:
    return 1.0 / (gamma - 1.0)


def transformed_pi_inf(gamma: float, pi_inf: float) -> float:
    return gamma * pi_inf / (gamma - 1.0)


def add_dodecane_layout_fluids(case: dict) -> None:
    """Fluid 1 = liquid dodecane, fluid 2 = dodecane vapor, fluid 3 = carrier gas."""
    liquid_gamma = 2.35
    vapor_gamma = 1.025
    carrier_gamma = 1.4
    case.update(
        {
            "fluid_pp(1)%gamma": transformed_gamma(liquid_gamma),
            "fluid_pp(1)%pi_inf": transformed_pi_inf(liquid_gamma, 4.8e8),
            "fluid_pp(1)%cv": 1077.0,
            "fluid_pp(1)%qv": -758060.0,
            "fluid_pp(1)%qvp": 0.0,
            "fluid_pp(2)%gamma": transformed_gamma(vapor_gamma),
            "fluid_pp(2)%pi_inf": 0.0,
            "fluid_pp(2)%cv": 1956.0,
            "fluid_pp(2)%qv": -234490.0,
            "fluid_pp(2)%qvp": -24625.0,
            "fluid_pp(3)%gamma": transformed_gamma(carrier_gamma),
            "fluid_pp(3)%pi_inf": 0.0,
            "fluid_pp(3)%cv": 739.0,
            "fluid_pp(3)%qv": 0.0,
            "fluid_pp(3)%qvp": 0.0,
        }
    )


def split_model3_gas_state(sol, fuel_species: str, alpha_liquid: float = 1.0e-10) -> dict[str, float]:
    fuel_idx = species_index(sol, fuel_species)
    y_fuel = max(0.0, min(1.0, float(sol.Y[fuel_idx])))
    alpha_g = max(1.0e-12, 1.0 - alpha_liquid)
    rho_g_stored = alpha_g * float(sol.density)
    alpha_vapor = max(1.0e-12, alpha_g * y_fuel)
    alpha_carrier = max(1.0e-12, alpha_g - alpha_vapor)
    return {
        "alpha_liquid": alpha_liquid,
        "alpha_vapor": alpha_vapor,
        "alpha_carrier": alpha_carrier,
        "alpha_rho_liquid": alpha_liquid * 740.0,
        "alpha_rho_vapor": rho_g_stored * y_fuel,
        "alpha_rho_carrier": rho_g_stored * (1.0 - y_fuel),
        "rho_g_stored": rho_g_stored,
    }


def add_model3_coupling_flags(
    case: dict,
    fuel_species_id: int,
    reaction_substeps: int,
    adaptive: bool = False,
    reaction_substeps_max: int | None = None,
) -> None:
    case.update(
        {
            "chemistry": "T",
            "model3_chemistry_coupling": "T",
            "fuel_species_id": fuel_species_id,
            "chem_params%diffusion": "F",
            "chem_params%reactions": "T",
            "chem_params%reaction_substeps": reaction_substeps,
            "chem_params%adap_substeps": "T" if adaptive else "F",
            "chem_params%reaction_substeps_max": reaction_substeps_max or reaction_substeps,
        }
    )
