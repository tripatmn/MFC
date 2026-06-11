#!/usr/bin/env python3
"""Coarse air-only reacting shock/dodecane-droplet SK54 sanity case.

This case combines the shock/droplet layout from the uploaded
``case_4comp_fix.py`` with the Yao/SK54 evaporation-to-fuel and reaction-heat
plumbing used by the Frolov-aligned SK54 droplet smoke case. It is intentionally
small and short: a first sanity check, not a production shock-droplet run.
"""

import argparse
import json
import math
import pathlib
import sys

import cantera as ct

parser = argparse.ArgumentParser(prog="2D_shockdroplet_air_sk54_sanity")
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
species_indices = {
    species: sol.species_index(species) + 1
    for species in KEY_SPECIES
    if species in sol.species_names
}

# Shock/droplet geometry from case_4comp_fix.py, coarsened for sanity testing.
D = 100.0e-6
r_drop = 0.5*D
Lx = 15.0*D
Ly = 6.0*D
x_drop = 5.0*D
y_drop = 3.0*D
nx = 160
ny = 64
dx = Lx/nx

# Air-only normal-shock closure. This replaces the old multi-gas closure in
# case_4comp_fix.py.
mach = 3.0
p01 = 101325.0
T1 = 478.0
gamma_air = 1.4
cv_air = 739.0
R_air = (gamma_air - 1.0)*cv_air
rho_air_1 = p01/(R_air*T1)
a1 = math.sqrt(gamma_air*R_air*T1)
p2_p1 = 1.0 + 2.0*gamma_air/(gamma_air + 1.0)*(mach*mach - 1.0)
rho2_rho1 = ((gamma_air + 1.0)*mach*mach)/((gamma_air - 1.0)*mach*mach + 2.0)
ps = p01*p2_p1
rho_air_2 = rho_air_1*rho2_rho1
T2 = T1*(p2_p1/rho2_rho1)
u2 = 2.0*a1/(gamma_air + 1.0)*(mach*mach - 1.0)/mach
a2 = math.sqrt(gamma_air*R_air*T2)

# Dodecane liquid/vapor properties consistent with the Frolov SK54 smoke setup.
rho_l = 750.0
gamma_l = 2.35
pi_inf_l = 4.8e8
cv_l = 1077.0
qv_l = -758060.0

gamma_v = 1.025
cv_v = 1956.0
qv_v = -234490.0
qvp_v = -24625.0
R_vapor = (gamma_v - 1.0)*cv_v
rho_v_1 = p01/(R_vapor*T1)
rho_v_2 = ps/(R_vapor*T2)

eps = 3.0e-5
alpha_l_bg = eps/2.0
alpha_v_bg = eps/2.0
alpha_air_bg = 1.0 - eps

alpha_l_drop = 1.0 - eps
alpha_v_drop = eps/2.0
alpha_air_drop = eps/2.0

c_liq = math.sqrt(gamma_l*(p01 + pi_inf_l)/rho_l)
cfl_target = 0.05
dt_initial = 0.5*cfl_target*dx/max(c_liq, abs(u2) + a2, a1)

t_stop = 2.0e-7
t_save = 5.0e-8

sol.TPX = T1, p01, "O2:0.21,N2:0.79"
air_Y = sol.Y
sol.TPX = T1, p01, f"{FUEL}:0.01,O2:0.205,N2:0.785"
dilute_fuel_air_Y = sol.Y

case = {
    "run_time_info": "T",
    "x_domain%beg": 0.0,
    "x_domain%end": Lx,
    "y_domain%beg": 0.0,
    "y_domain%end": Ly,
    "m": nx,
    "n": ny,
    "p": 0,
    "dt": dt_initial,
    "cfl_adap_dt": "T",
    "cfl_target": cfl_target,
    "n_start": 0,
    "t_stop": t_stop,
    "t_save": t_save,
    "t_step_print": 100,
    "num_patches": 3,
    "model_eqns": 3,
    "alt_soundspeed": "F",
    "num_fluids": 3,
    "mpp_lim": "T",
    "mixture_err": "T",
    "relax": "T",
    "relax_model": 6,
    "palpha_eps": eps,
    "ptgalpha_eps": eps,
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
    "time_stepper": 3,
    "weno_order": 3,
    "weno_eps": 1.0e-16,
    "weno_Re_flux": "F",
    "weno_avg": "F",
    "mapped_weno": "T",
    "null_weights": "F",
    "mp_weno": "F",
    "riemann_solver": 2,
    "wave_speeds": 1,
    "avg_state": 2,
    "bc_x%beg": -6,
    "bc_x%end": -6,
    "bc_y%beg": -6,
    "bc_y%end": -6,
    "format": 1,
    "precision": 2,
    "parallel_io": "F",
    "prim_vars_wrt": "T",
    "cons_vars_wrt": "T",
    "pres_wrt": "T",
    "chem_wrt_T": "T",
    "fluid_pp(1)%gamma": 1.0/(gamma_l - 1.0),
    "fluid_pp(1)%pi_inf": gamma_l*pi_inf_l/(gamma_l - 1.0),
    "fluid_pp(1)%cv": cv_l,
    "fluid_pp(1)%qv": qv_l,
    "fluid_pp(1)%qvp": 0.0,
    "fluid_pp(2)%gamma": 1.0/(gamma_v - 1.0),
    "fluid_pp(2)%pi_inf": 0.0,
    "fluid_pp(2)%cv": cv_v,
    "fluid_pp(2)%qv": qv_v,
    "fluid_pp(2)%qvp": qvp_v,
    "fluid_pp(3)%gamma": 1.0/(gamma_air - 1.0),
    "fluid_pp(3)%pi_inf": 0.0,
    "fluid_pp(3)%cv": cv_air,
    "fluid_pp(3)%qv": 0.0,
    "fluid_pp(3)%qvp": 0.0,
    # Patch 1: pre-shock air to the right of the initial shock.
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.5*(15.0*D + 4.0*D),
    "patch_icpp(1)%y_centroid": y_drop,
    "patch_icpp(1)%length_x": 11.0*D,
    "patch_icpp(1)%length_y": Ly,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": p01,
    "patch_icpp(1)%alpha(1)": alpha_l_bg,
    "patch_icpp(1)%alpha(2)": alpha_v_bg,
    "patch_icpp(1)%alpha(3)": alpha_air_bg,
    "patch_icpp(1)%alpha_rho(1)": alpha_l_bg*rho_l,
    "patch_icpp(1)%alpha_rho(2)": alpha_v_bg*rho_v_1,
    "patch_icpp(1)%alpha_rho(3)": alpha_air_bg*rho_air_1,
    # Patch 2: post-shock air to the left; shock travels left-to-right.
    "patch_icpp(2)%geometry": 3,
    "patch_icpp(2)%alter_patch(1)": "T",
    "patch_icpp(2)%x_centroid": 2.0*D,
    "patch_icpp(2)%y_centroid": y_drop,
    "patch_icpp(2)%length_x": 4.0*D,
    "patch_icpp(2)%length_y": Ly,
    "patch_icpp(2)%vel(1)": u2,
    "patch_icpp(2)%vel(2)": 0.0,
    "patch_icpp(2)%pres": ps,
    "patch_icpp(2)%alpha(1)": alpha_l_bg,
    "patch_icpp(2)%alpha(2)": alpha_v_bg,
    "patch_icpp(2)%alpha(3)": alpha_air_bg,
    "patch_icpp(2)%alpha_rho(1)": alpha_l_bg*rho_l,
    "patch_icpp(2)%alpha_rho(2)": alpha_v_bg*rho_v_2,
    "patch_icpp(2)%alpha_rho(3)": alpha_air_bg*rho_air_2,
    # Patch 3: liquid dodecane droplet.
    "patch_icpp(3)%geometry": 2,
    "patch_icpp(3)%alter_patch(1)": "T",
    "patch_icpp(3)%x_centroid": x_drop,
    "patch_icpp(3)%y_centroid": y_drop,
    "patch_icpp(3)%radius": r_drop,
    "patch_icpp(3)%vel(1)": 0.0,
    "patch_icpp(3)%vel(2)": 0.0,
    "patch_icpp(3)%pres": p01,
    "patch_icpp(3)%alpha(1)": alpha_l_drop,
    "patch_icpp(3)%alpha(2)": alpha_v_drop,
    "patch_icpp(3)%alpha(3)": alpha_air_drop,
    "patch_icpp(3)%alpha_rho(1)": alpha_l_drop*rho_l,
    "patch_icpp(3)%alpha_rho(2)": alpha_v_drop*rho_v_1,
    "patch_icpp(3)%alpha_rho(3)": alpha_air_drop*rho_air_1,
}

for i, value in enumerate(air_Y, start=1):
    case[f"patch_icpp(1)%Y({i})"] = float(value)
    case[f"patch_icpp(2)%Y({i})"] = float(value)
for i, value in enumerate(dilute_fuel_air_Y, start=1):
    case[f"patch_icpp(3)%Y({i})"] = float(value)

for fluid_id in range(1, 4):
    case[f"alpha_rho_wrt({fluid_id})"] = "T"
    case[f"alpha_wrt({fluid_id})"] = "T"

for index in species_indices.values():
    case[f"chem_wrt_Y({index})"] = "T"

if __name__ == "__main__":
    print(
        "Shock-droplet air SK54 sanity: "
        f"D={D} nx={nx} ny={ny} p1={p01:.6g} T1={T1:.6g} "
        f"p2={ps:.6g} T2={T2:.6g} u2={u2:.6g} "
        f"species={sol.n_species} reactions={sol.n_reactions} indices={species_indices}",
        file=sys.stderr,
    )
    print(json.dumps(case))
