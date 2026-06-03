#!/usr/bin/env python3
"""0.25 mm quiescent n-dodecane nonreacting evaporation pilot.

This is the first literature-scaled bridge pilot from
docs/dev/literature_scaled_quiescent_burning_pilot_plan.md. It keeps the clean
quiescent phase-change setup, uses a 2D equivalent droplet, and runs with
CFL-adaptive time stepping for an early-time D2-law window.
"""

import argparse
import json

parser = argparse.ArgumentParser(prog="2D_dodecane_quiescent_evap_025mm_pilot")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


D0 = 0.25e-3
r_drop = 0.5*D0
domain = 8.0*D0
cells_per_diameter = 32
nx = int(round(domain/D0*cells_per_diameter))
ny = nx
dx = domain/nx
dy = dx

p0 = 2.0e5
t_hot = 900.0

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

rho_v_hot = p0/((gamma_v - 1.0)*cv_v*t_hot)
rho_air_hot = p0/((gamma_air - 1.0)*cv_air*t_hot)

eps = 1.0e-6
c_liq = (gamma_l*(p0 + pi_inf_l)/rho_l)**0.5

cfl_target = 0.10
dt_initial = 0.75*cfl_target*dx/c_liq
t_stop = 5.0e-4
t_save = 1.0e-5


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
        "dt": dt_initial,
        "cfl_adap_dt": "T",
        "cfl_target": cfl_target,
        "n_start": 0,
        "t_stop": t_stop,
        "t_save": t_save,
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
        "chem_reaction_heat_enable": "F",
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
        "patch_icpp(1)%alpha_rho(2)": alpha_v_bg*rho_v_hot,
        "patch_icpp(1)%alpha_rho(3)": alpha_a_bg*rho_air_hot,
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
        "patch_icpp(2)%alpha_rho(2)": alpha_v_drop*rho_v_hot,
        "patch_icpp(2)%alpha_rho(3)": alpha_a_drop*rho_air_hot,
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
