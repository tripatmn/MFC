#!/usr/bin/env python3
"""Tiny gas-only nDodecane_Reitz chemistry sanity case.

This case is intended to test whether MFC can load, code-generate, build, and
run the 100-species Reitz n-dodecane mechanism before using it in droplet cases.
It is deliberately gas-only: no liquid, no phase change, no multifluid reaction
heat flag.
"""

import argparse
import json
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="1D_dodecane_reitz_smoke")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

CTFILE = "nDodecane_Reitz.yaml"
CTPHASE = "nDodecane_IG"
KEY_SPECIES = ("c12h26", "o2", "n2", "co2", "h2o", "h2o2", "oh", "ho2")

sol = ct.Solution(CTFILE, CTPHASE)
sol.TPX = 800.0, 100000.0, "c12h26:0.01,o2:0.205,n2:0.785"

length = 1.0e-3
nx = 8
dt = 1.0e-11
nstop = 100
nsave = 10

case = {
    "run_time_info": "T",
    "x_domain%beg": 0.0,
    "x_domain%end": length,
    "m": nx,
    "n": 0,
    "p": 0,
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": nstop,
    "t_step_save": nsave,
    "t_step_print": nsave,
    "parallel_io": "F",
    "model_eqns": 2,
    "num_fluids": 1,
    "num_patches": 1,
    "mpp_lim": "F",
    "mixture_err": "F",
    "time_stepper": 3,
    "weno_order": 1,
    "weno_eps": 1.0e-16,
    "riemann_solver": 1,
    "wave_speeds": 1,
    "avg_state": 2,
    "bc_x%beg": -1,
    "bc_x%end": -1,
    "chemistry": "T",
    "chem_params%diffusion": "F",
    "chem_params%reactions": "T",
    "cantera_file": CTFILE,
    "cantera_phase": CTPHASE,
    "chem_T_min": 250.0,
    "chem_T_max": 3000.0,
    "fuel_species_id": sol.species_index("c12h26") + 1,
    "format": 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "chem_wrt_T": "T",
    "patch_icpp(1)%geometry": 1,
    "patch_icpp(1)%x_centroid": length/2.0,
    "patch_icpp(1)%length_x": length,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%pres": sol.P,
    "patch_icpp(1)%alpha(1)": 1.0,
    "patch_icpp(1)%alpha_rho(1)": sol.density,
    "fluid_pp(1)%gamma": 1.0/(4.4 - 1.0),
    "fluid_pp(1)%pi_inf": 0.0,
}

for i, mass_fraction in enumerate(sol.Y, start=1):
    case[f"patch_icpp(1)%Y({i})"] = mass_fraction

for species in KEY_SPECIES:
    try:
        case[f"chem_wrt_Y({sol.species_index(species) + 1})"] = "T"
    except ValueError:
        pass

if __name__ == "__main__":
    species_indices = {
        species: sol.species_index(species) + 1
        for species in KEY_SPECIES
        if species in sol.species_names
    }
    print(
        "nDodecane_Reitz sanity: "
        f"species={sol.n_species} reactions={sol.n_reactions} "
        f"indices={species_indices}",
        file=sys.stderr,
    )
    print(json.dumps(case))
