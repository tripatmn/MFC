#!/usr/bin/env python3
"""Clean quiescent Tanabe-style n-dodecane/SK54 droplet smoke case.

This is intentionally a fresh, standalone JSON-emitting quiescent case:
no imposed wave, no restart continuation, no wrappers around older case files.
"""

import argparse
import json
import math


parser = argparse.ArgumentParser(prog="tanabe_quiescent_clean_128")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
parser.parse_args()


# Geometry and target condition.
D = 0.7e-3
r_drop = 0.5*D
domain = 6.0*D
nx = 128
ny = 128
dx = domain/nx

p0 = 2.0e6
T_inf = 800.0

# Short smoke run by default. Change this one value for longer scouting.
t_stop_smoke = 1.0e-4
t_stop = t_stop_smoke
t_save = 1.0e-5
n_start = 0

# Three broad fluids: liquid dodecane, dodecane vapor, air.
num_patches = 2
num_fluids = 3

# Dodecane liquid/vapor and air properties used by the current dodecane cases.
rho_l = 750.0
gamma_l = 2.35
pi_inf_l = 4.8e8
cv_l = 1077.0
qv_l = -758060.0

gamma_v = 1.025
cv_v = 1956.0
qv_v = -234490.0
qvp_v = -24625.0

gamma_air = 1.4
cv_air = 739.0

# Ideal-gas densities at 2 MPa / 800 K. These are approximate case constants.
rho_v = p0/((gamma_v - 1.0)*cv_v*T_inf)
rho_air = p0/((gamma_air - 1.0)*cv_air*T_inf)

eps = 1.0e-6
alpha_l_bg = eps
alpha_v_bg = eps
alpha_air_bg = 1.0 - alpha_l_bg - alpha_v_bg

alpha_l_drop = 1.0 - 2.0*eps
alpha_v_drop = eps
alpha_air_drop = eps

c_liq = math.sqrt(gamma_l*(p0 + pi_inf_l)/rho_l)
cfl_target = 0.10
dt_initial = 0.75*cfl_target*dx/c_liq

# Yao/SK54 mechanism. Important 1-based species indices:
# N2=1, OH=5, HO2=6, H2O=8, H2O2=9, O2=10, CO2=19, NC12H26=46.
CTFILE = "examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml"
CTPHASE = "yao_sk54"
FUEL_SPECIES_ID = 46
NUM_SPECIES = 54

MW_N2 = 28.0134
MW_O2 = 31.998
MW_NC12H26 = 170.33484


def mole_to_mass_fractions(composition):
    denom = sum(moles*mw for _name, (moles, mw) in composition.items())
    return {name: moles*mw/denom for name, (moles, mw) in composition.items()}


air_Y = mole_to_mass_fractions({
    "N2": (0.79, MW_N2),
    "O2": (0.21, MW_O2),
})

dilute_fuel_air_Y = mole_to_mass_fractions({
    "NC12H26": (0.01, MW_NC12H26),
    "O2": (0.205, MW_O2),
    "N2": (0.785, MW_N2),
})

species_name_to_index = {
    "N2": 1,
    "OH": 5,
    "HO2": 6,
    "H2O": 8,
    "H2O2": 9,
    "O2": 10,
    "CO2": 19,
    "NC12H26": 46,
}

case = {
    "run_time_info": "T",
    "x_domain%beg": -0.5*domain,
    "x_domain%end": 0.5*domain,
    "y_domain%beg": -0.5*domain,
    "y_domain%end": 0.5*domain,
    "m": nx,
    "n": ny,
    "p": 0,
    "dt": dt_initial,
    "cfl_adap_dt": "T",
    "cfl_target": cfl_target,
    "n_start": n_start,
    "t_stop": t_stop,
    "t_save": t_save,
    "t_step_print": 1000,
    "num_patches": num_patches,
    "model_eqns": 3,
    "alt_soundspeed": "F",
    "num_fluids": num_fluids,
    "mpp_lim": "T",
    "mixture_err": "T",
    "relax": "T",
    "relax_ic": "F",
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
    "fuel_species_id": FUEL_SPECIES_ID,
    "evap_species_source": "T",
    "evap_species_src": 0.0,
    "evap_liquid_fluid_id": 1,
    "evap_alpha_thresh": 0.01,
    "chem_reaction_heat_enable": "T",
    "chem_reaction_heat_limit_frac": 0.05,
    "chem_reaction_heat_diag": "F",
    "time_stepper": 3,
    "recon_type": 2,
    "muscl_order": 2,
    "muscl_lim": 4,
    "int_comp": "T",
    "null_weights": "F",
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
    "cons_vars_wrt": "T",
    "prim_vars_wrt": "T",
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
    # Patch 1: uniform quiescent hot air with trace liquid/vapor fluids.
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.0,
    "patch_icpp(1)%y_centroid": 0.0,
    "patch_icpp(1)%length_x": domain,
    "patch_icpp(1)%length_y": domain,
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": p0,
    "patch_icpp(1)%alpha(1)": alpha_l_bg,
    "patch_icpp(1)%alpha(2)": alpha_v_bg,
    "patch_icpp(1)%alpha(3)": alpha_air_bg,
    "patch_icpp(1)%alpha_rho(1)": alpha_l_bg*rho_l,
    "patch_icpp(1)%alpha_rho(2)": alpha_v_bg*rho_v,
    "patch_icpp(1)%alpha_rho(3)": alpha_air_bg*rho_air,
    # Patch 2: centered liquid n-dodecane droplet altering patch 1.
    "patch_icpp(2)%geometry": 2,
    "patch_icpp(2)%alter_patch(1)": "T",
    "patch_icpp(2)%x_centroid": 0.0,
    "patch_icpp(2)%y_centroid": 0.0,
    "patch_icpp(2)%radius": r_drop,
    "patch_icpp(2)%vel(1)": 0.0,
    "patch_icpp(2)%vel(2)": 0.0,
    "patch_icpp(2)%pres": p0,
    "patch_icpp(2)%alpha(1)": alpha_l_drop,
    "patch_icpp(2)%alpha(2)": alpha_v_drop,
    "patch_icpp(2)%alpha(3)": alpha_air_drop,
    "patch_icpp(2)%alpha_rho(1)": alpha_l_drop*rho_l,
    "patch_icpp(2)%alpha_rho(2)": alpha_v_drop*rho_v,
    "patch_icpp(2)%alpha_rho(3)": alpha_air_drop*rho_air,
}

for i in range(1, num_fluids + 1):
    case[f"alpha_rho_wrt({i})"] = "T"
    case[f"alpha_wrt({i})"] = "T"

for species_id in species_name_to_index.values():
    case[f"chem_wrt_Y({species_id})"] = "T"

for patch_id, composition in ((1, air_Y), (2, dilute_fuel_air_Y)):
    for species_id in range(1, NUM_SPECIES + 1):
        case[f"patch_icpp({patch_id})%Y({species_id})"] = 0.0
    for name, value in composition.items():
        case[f"patch_icpp({patch_id})%Y({species_name_to_index[name]})"] = float(value)

velocity_keys = [key for key in case if "patch_icpp(" in key and "%vel(" in key]
assert all(case[key] == 0.0 for key in velocity_keys)
assert case["n_start"] == 0
assert case["num_patches"] == 2

print(json.dumps(case))

# Audit command block:
# python3 examples/2D_dodecane_global_reduced/case_tanabe_quiescent_clean_128.py > /tmp/tanabe_clean.json
# python3 - <<'PY'
# import json
# d = json.load(open('/tmp/tanabe_clean.json'))
# for k in sorted(d):
#     if 'vel(' in k or k in ['n_start', 'num_patches', 'num_fluids', 't_stop', 't_save']:
#         print(k, d[k])
# PY
