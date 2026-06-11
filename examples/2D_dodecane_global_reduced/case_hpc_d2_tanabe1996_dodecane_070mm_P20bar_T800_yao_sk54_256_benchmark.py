#!/usr/bin/env python3
"""Tanabe-1996-style 0.7 mm n-dodecane 256^2 speed/stability benchmark.

This uses the same physics as the 128^2 scout case but stops at 2 ms. It is a
runtime and stability benchmark only, not a full induction-time validation run.
"""

import argparse
import json
import pathlib
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="2D_dodecane_tanabe1996_070mm_P20bar_T800_yao_sk54_256_benchmark")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (
                (case_dir/"case_hpc_d2_quiescent_evap_025mm_pilot.py").is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with quiescent dodecane inputs.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"
CTFILE = "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
CTPHASE = "yao_sk54"
FUEL = "NC12H26"
KEY_SPECIES = (FUEL, "O2", "N2", "CO2", "H2O", "OH", "HO2", "H2O2")

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import case_hpc_d2_quiescent_evap_025mm_pilot as base


sol = ct.Solution(str(REPO/CTFILE), CTPHASE)
sol.TPX = 800.0, 2.0e6, "O2:0.21,N2:0.79"
air_Y = sol.Y
sol.TPX = 800.0, 2.0e6, f"{FUEL}:0.01,O2:0.205,N2:0.785"
fuel_vapor_seed_Y = sol.Y
species_indices = {
    species: sol.species_index(species) + 1
    for species in KEY_SPECIES
    if species in sol.species_names
}

base.D0 = 0.7e-3
base.r_drop = 0.5*base.D0
base.domain = 6.0*base.D0
base.cells_per_diameter = 256/6.0
base.nx = 256
base.ny = 256
base.dx = base.domain/base.nx
base.dy = base.dx

base.p0 = 2.0e6
base.t_hot = 800.0
base.rho_l = 750.0
base.rho_v_hot = base.p0/((base.gamma_v - 1.0)*base.cv_v*base.t_hot)
base.rho_air_hot = base.p0/((base.gamma_air - 1.0)*base.cv_air*base.t_hot)
base.c_liq = (base.gamma_l*(base.p0 + base.pi_inf_l)/base.rho_l)**0.5
base.cfl_target = 0.10
base.dt_initial = 0.75*base.cfl_target*base.dx/base.c_liq
base.t_stop = 2.0e-3
base.t_save = 5.0e-4


def add_yao_species_ic(case):
    for i, value in enumerate(air_Y, start=1):
        case[f"patch_icpp(1)%Y({i})"] = float(value)
    for i, value in enumerate(fuel_vapor_seed_Y, start=1):
        case[f"patch_icpp(2)%Y({i})"] = float(value)

    for index in species_indices.values():
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
    "fuel_species_id": species_indices[FUEL],
    "evap_species_source": "T",
    "evap_species_src": 0.0,
    "evap_liquid_fluid_id": 1,
    "evap_alpha_thresh": 0.01,
    "chem_reaction_heat_enable": "T",
    "chem_reaction_heat_limit_frac": 0.05,
    "chem_reaction_heat_diag": "F",
    "chem_wrt_T": "T",
    "t_step_print": 1000,
})
add_yao_species_ic(case)

if __name__ == "__main__":
    print(
        "Tanabe 1996 Yao/SK54 256 benchmark: "
        f"D0={base.D0} p0={base.p0} T_gas={base.t_hot} "
        f"domain={base.domain} grid={base.nx}x{base.ny} "
        f"species={sol.n_species} reactions={sol.n_reactions} "
        f"indices={species_indices}",
        file=sys.stderr,
    )
    print(json.dumps(case))
