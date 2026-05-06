#!/usr/bin/env python3
import argparse
import json

import cantera as ct

parser = argparse.ArgumentParser(prog="1D_dodecane_reactive_smoke")
parser.add_argument(
    "--mfc",
    type=json.loads,
    default="{}",
    metavar="DICT",
    help="MFC's toolchain's internal state.",
)
args = parser.parse_args()

ctfile = "nDodecane_Reitz.yaml"
ctphase = "nDodecane_IG"
sol = ct.Solution(ctfile, ctphase)
sol.TPX = 1500.0, 20.0*ct.one_atm, "c12h26:0.01,o2:0.185,n2:0.695"

length = 1.0e-3
nx = 8
dt = 1.0e-10
nstop = 1000

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
    "t_step_save": nstop,
    "t_step_print": 10,
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
    "cantera_file": ctfile,
    "cantera_phase": ctphase,
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

for species in ["c12h26", "o2", "n2", "co2", "h2o"]:
    case[f"chem_wrt_Y({sol.species_index(species) + 1})"] = "T"

for i, mass_fraction in enumerate(sol.Y, start=1):
    case[f"patch_icpp(1)%Y({i})"] = mass_fraction

if __name__ == "__main__":
    print(json.dumps(case))
