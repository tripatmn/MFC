#!/usr/bin/env python3
"""Zhang-2021-style 1 mm nonreacting n-dodecane evaporation validation case.

This is an experimental/literature validation target setup, not a completed
paper match. The exact Zhang et al. D2(t) trace, regression window, and fitted
K still need to be digitized/confirmed from the paper.

Primary validation metric: mass-equivalent D2_norm from liquid alpha_rho.
Secondary metric: 2D threshold/equivalent-circle D2. The latter is a 2D
cross-section metric and should not be interpreted as a true 3D droplet D2.
"""

import argparse
import json

parser = argparse.ArgumentParser(prog="2D_dodecane_zhang2021_evap_1mm")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


# Target conditions, Zhang et al. 2021 style.
D0 = 1.0e-3
r_drop = 0.5*D0
p0 = 1.0e5
t_amb = 793.15

# Practical 2D setup: 4 droplet diameters across with 128 cells per D0.
cells_per_diameter = 128
domain = 4.0*D0
nx = int(domain/D0*cells_per_diameter)
ny = nx
dx = domain/nx
dy = domain/ny

# Dodecane liquid/vapor and air-like gas properties follow the existing clean
# evaporation framework. Air is represented by fluid 3's gamma/cv pair.
rho_l = 669.43
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

rho_v_amb = p0/((gamma_v - 1.0)*cv_v*t_amb)
rho_air_amb = p0/((gamma_air - 1.0)*cv_air*t_amb)

eps = 1.0e-6
c_liq = (gamma_l*(p0 + pi_inf_l)/rho_l)**0.5
dt = 0.75*0.25*dx/c_liq


def build_case():
    alpha_l_bg = eps
    alpha_v_bg = eps
    alpha_a_bg = 1.0 - alpha_l_bg - alpha_v_bg

    alpha_l_drop = 1.0 - 2.0*eps
    alpha_v_drop = eps
    alpha_a_drop = eps

    case = {
        "run_time_info": "T",
        "x_domain%beg": -0.5*domain,
        "x_domain%end": 0.5*domain,
        "y_domain%beg": -0.5*domain,
        "y_domain%end": 0.5*domain,
        "m": nx,
        "n": ny,
        "p": 0,
        "dt": dt,
        "t_step_start": 0,
        "t_step_stop": 120000,
        "t_step_save": 2000,
        "t_step_print": 1000,
        "num_patches": 2,
        "model_eqns": 3,
        "num_fluids": 3,
        "mpp_lim": "T",
        "mixture_err": "T",
        "relax": "T",
        "relax_ic": "F",
        "relax_model": 6,
        "palpha_eps": eps,
        "ptgalpha_eps": eps,
        "chemistry": "F",
        "chem_params%reactions": "F",
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
        "patch_icpp(1)%geometry": 3,
        "patch_icpp(1)%x_centroid": 0.0,
        "patch_icpp(1)%y_centroid": 0.0,
        "patch_icpp(1)%length_x": domain,
        "patch_icpp(1)%length_y": domain,
        "patch_icpp(1)%vel(1)": 0.0,
        "patch_icpp(1)%vel(2)": 0.0,
        "patch_icpp(1)%pres": p0,
        "patch_icpp(1)%alpha_rho(1)": alpha_l_bg*rho_l,
        "patch_icpp(1)%alpha_rho(2)": alpha_v_bg*rho_v_amb,
        "patch_icpp(1)%alpha_rho(3)": alpha_a_bg*rho_air_amb,
        "patch_icpp(1)%alpha(1)": alpha_l_bg,
        "patch_icpp(1)%alpha(2)": alpha_v_bg,
        "patch_icpp(1)%alpha(3)": alpha_a_bg,
        "patch_icpp(2)%geometry": 2,
        "patch_icpp(2)%alter_patch(1)": "T",
        "patch_icpp(2)%x_centroid": 0.0,
        "patch_icpp(2)%y_centroid": 0.0,
        "patch_icpp(2)%radius": r_drop,
        "patch_icpp(2)%vel(1)": 0.0,
        "patch_icpp(2)%vel(2)": 0.0,
        "patch_icpp(2)%pres": p0,
        "patch_icpp(2)%alpha_rho(1)": alpha_l_drop*rho_l,
        "patch_icpp(2)%alpha_rho(2)": alpha_v_drop*rho_v_amb,
        "patch_icpp(2)%alpha_rho(3)": alpha_a_drop*rho_air_amb,
        "patch_icpp(2)%alpha(1)": alpha_l_drop,
        "patch_icpp(2)%alpha(2)": alpha_v_drop,
        "patch_icpp(2)%alpha(3)": alpha_a_drop,
    }

    for i in range(1, 4):
        case[f"alpha_rho_wrt({i})"] = "T"
        case[f"alpha_wrt({i})"] = "T"

    return case


case = build_case()

if __name__ == "__main__":
    print(json.dumps(case))
