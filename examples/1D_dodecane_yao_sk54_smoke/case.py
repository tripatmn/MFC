#!/usr/bin/env python3
"""Tiny gas-only Yao SK54 n-dodecane chemistry sanity case.

This case checks whether the converted 54-species Yao/SK54 mechanism can be
loaded by Cantera and emitted through the MFC chemistry case path before using
it in multiphase droplet smoke tests. It is deliberately gas-only: no liquid
phase and no phase change.
"""

import argparse
import json
import pathlib
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="1D_dodecane_yao_sk54_smoke")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            if (path/"toolchain"/"main.py").is_file():
                return path
    raise RuntimeError("Could not find MFC repo root.")


REPO = find_repo_root()
CTFILE = "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
CTPHASE = "yao_sk54"
FUEL = "NC12H26"
KEY_SPECIES = (FUEL, "O2", "N2", "CO2", "H2O", "OH", "HO2", "H2O2")

sol = ct.Solution(str(REPO/CTFILE), CTPHASE)
sol.TPX = 900.0, 100000.0, f"{FUEL}:0.01,O2:0.205,N2:0.785"

length = 1.0e-3
nx = 8
dt = 1.0e-10
nstop = 100
nsave = 10

species_indices = {
    species: sol.species_index(species) + 1
    for species in KEY_SPECIES
    if species in sol.species_names
}

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
    "fuel_species_id": species_indices[FUEL],
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
    case[f"patch_icpp(1)%Y({i})"] = float(mass_fraction)

for index in species_indices.values():
    case[f"chem_wrt_Y({index})"] = "T"

if __name__ == "__main__":
    print(
        "Yao SK54 gas smoke: "
        f"phase={CTPHASE} species={sol.n_species} reactions={sol.n_reactions} "
        f"indices={species_indices}",
        file=sys.stderr,
    )
    print(json.dumps(case))
