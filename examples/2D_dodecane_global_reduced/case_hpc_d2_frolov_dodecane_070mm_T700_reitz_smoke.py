#!/usr/bin/env python3
"""Frolov-aligned 0.7 mm n-dodecane Reitz chemistry smoke test.

This is a short stability and chemistry-activation smoke test, not a
validation-length run. It keeps the Frolov bridge geometry and thermodynamic
conditions, but uses a 128 x 128 smoke grid to reduce the first Reitz
multiphase build/run cost.
"""

import argparse
import json
import pathlib
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="2D_dodecane_frolov_070mm_T700_reitz_smoke")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (
                (case_dir/"case_hpc_d2_frolov_dodecane_070mm_T700_evap_pilot.py").is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with Frolov bridge pilot inputs.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"
CTFILE = "nDodecane_Reitz.yaml"
CTPHASE = "nDodecane_IG"
KEY_SPECIES = ("c12h26", "o2", "n2", "co2", "h2o", "oh", "ho2", "h2o2")

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import case_hpc_d2_quiescent_evap_025mm_pilot as base


sol = ct.Solution(CTFILE, CTPHASE)
sol.TPX = 700.0, 1.0e5, "o2:0.21,n2:0.79"
air_Y = sol.Y
sol.TPX = 700.0, 1.0e5, "c12h26:1.0"
fuel_vapor_Y = sol.Y
species_indices = {
    species: sol.species_index(species) + 1
    for species in KEY_SPECIES
    if species in sol.species_names
}

base.D0 = 0.7e-3
base.r_drop = 0.5*base.D0
base.domain = 8.0*base.D0
base.cells_per_diameter = 16
base.nx = 128
base.ny = 128
base.dx = base.domain/base.nx
base.dy = base.dx

base.p0 = 1.0e5
base.t_hot = 700.0
base.rho_l = 750.0
base.rho_v_hot = base.p0/((base.gamma_v - 1.0)*base.cv_v*base.t_hot)
base.rho_air_hot = base.p0/((base.gamma_air - 1.0)*base.cv_air*base.t_hot)
base.c_liq = (base.gamma_l*(base.p0 + base.pi_inf_l)/base.rho_l)**0.5
base.cfl_target = 0.10
base.dt_initial = 0.75*base.cfl_target*base.dx/base.c_liq
base.t_stop = 1.0e-6
base.t_save = 1.0e-7


def add_reitz_species_ic(case):
    # Patch 1 is the ambient gas. Patch 2 is the liquid droplet override; its
    # trace gas/vapor content is initialized as dodecane vapor so the short
    # Reitz smoke does not depend on resolving appreciable evaporation first.
    for i, value in enumerate(air_Y, start=1):
        case[f"patch_icpp(1)%Y({i})"] = float(value)
    for i, value in enumerate(fuel_vapor_Y, start=1):
        case[f"patch_icpp(2)%Y({i})"] = float(value)

    for species, index in species_indices.items():
        case[f"chem_wrt_Y({index})"] = "T"


case = base.build_case()
case.update({
    "chemistry": "T",
    "chem_params%diffusion": "F",
    "chem_params%reactions": "T",
    "cantera_file": CTFILE,
    "cantera_phase": CTPHASE,
    "chem_gas_fluid_id": 2,
    "chem_gas_num_fluids": 2,
    "chem_gas_fluid_ids(1)": 2,
    "chem_gas_fluid_ids(2)": 3,
    "fuel_species_id": species_indices["c12h26"],
    "evap_species_source": "T",
    "evap_species_src": 0.0,
    "evap_liquid_fluid_id": 1,
    "evap_alpha_thresh": 0.01,
    "chem_reaction_heat_enable": "T",
    "chem_reaction_heat_limit_frac": 0.05,
    "chem_reaction_heat_diag": "F",
    "t_step_print": 100,
})
add_reitz_species_ic(case)

if __name__ == "__main__":
    print(
        "Frolov Reitz smoke: "
        f"species={sol.n_species} reactions={sol.n_reactions} "
        f"indices={species_indices}",
        file=sys.stderr,
    )
    print(json.dumps(case))
