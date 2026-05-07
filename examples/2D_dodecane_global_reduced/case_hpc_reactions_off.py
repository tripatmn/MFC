#!/usr/bin/env python3
import argparse
import json
import pathlib

parser = argparse.ArgumentParser(prog="2D_dodecane_global_reduced_off")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            mechanism = path/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml"
            if mechanism.is_file() and (path/"toolchain"/"main.py").is_file():
                return path
    raise RuntimeError("Could not find MFC repo root with dodecane_global_1step.yaml.")


REPO = find_repo_root()
CTFILE = str(REPO/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml")
CTPHASE = "gas"

D = 75.0e-6
T1 = 478.0
p01 = 101325.0
ps = 1.047e6
rho_w = 669.43
rho_v = 4.35
rho_post_v = 35.631
rho_a = 0.740
rho_post_a = 2.854
eps = 1.0e-4

u_post = 972.9121577324589
dt = 4.460963578547797e-10


def build_case(reactions):
    case = {
        "run_time_info": "T",
        "x_domain%beg": -2.0*D,
        "x_domain%end": 3.0*D,
        "y_domain%beg": -1.5*D,
        "y_domain%end": 1.5*D,
        "m": 240,
        "n": 160,
        "p": 0,
        "dt": dt,
        "t_step_start": 0,
        "t_step_stop": 50,
        "t_step_save": 5,
        "t_step_print": 5,
        "num_patches": 3,
        "model_eqns": 3,
        "num_fluids": 3,
        "mpp_lim": "T",
        "mixture_err": "T",
        "relax": "T",
        "relax_ic": "F",
        "relax_model": 6,
        "palpha_eps": eps,
        "ptgalpha_eps": eps,
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
        "bc_y%beg": -2,
        "bc_y%end": -3,
        "chemistry": "T",
        "chem_params%diffusion": "F",
        "chem_params%reactions": "T" if reactions else "F",
        "cantera_file": CTFILE,
        "cantera_phase": CTPHASE,
        "chem_gas_fluid_id": 2,
        "chem_gas_num_fluids": 2,
        "chem_gas_fluid_ids(1)": 2,
        "chem_gas_fluid_ids(2)": 3,
        "fuel_species_id": 1,
        "evap_species_source": "T",
        "evap_species_src": 0.0,
        "evap_liquid_fluid_id": 1,
        "evap_alpha_thresh": 0.01,
        "format": 1,
        "precision": 2,
        "parallel_io": "F",
        "cons_vars_wrt": "T",
        "prim_vars_wrt": "T",
        "pres_wrt": "T",
        "fluid_pp(1)%gamma": 1.0/(2.35 - 1.0),
        "fluid_pp(1)%pi_inf": 2.35*4.8e8/(2.35 - 1.0),
        "fluid_pp(1)%cv": 1077.0,
        "fluid_pp(1)%qv": -758060.0,
        "fluid_pp(1)%qvp": 0.0,
        "fluid_pp(2)%gamma": 1.0/(1.025 - 1.0),
        "fluid_pp(2)%pi_inf": 0.0,
        "fluid_pp(2)%cv": 1956.0,
        "fluid_pp(2)%qv": -234490.0,
        "fluid_pp(2)%qvp": -24625.0,
        "fluid_pp(3)%gamma": 1.0/(1.4 - 1.0),
        "fluid_pp(3)%pi_inf": 0.0,
        "fluid_pp(3)%cv": 739.0,
        "fluid_pp(3)%qv": 0.0,
        "fluid_pp(3)%qvp": 0.0,
    }

    for i in range(1, 4):
        case[f"alpha_rho_wrt({i})"] = "T"
        case[f"alpha_wrt({i})"] = "T"

    for i in range(1, 6):
        case[f"chem_wrt_Y({i})"] = "T"

    air_y = [0.0, 0.233, 0.767, 0.0, 0.0]
    for pid in (1, 2, 3):
        for i, value in enumerate(air_y, start=1):
            case[f"patch_icpp({pid})%Y({i})"] = value

    case.update({
        "patch_icpp(1)%geometry": 3,
        "patch_icpp(1)%x_centroid": 0.5*D,
        "patch_icpp(1)%y_centroid": 0.0,
        "patch_icpp(1)%length_x": 5.0*D,
        "patch_icpp(1)%length_y": 3.0*D,
        "patch_icpp(1)%vel(1)": 0.0,
        "patch_icpp(1)%vel(2)": 0.0,
        "patch_icpp(1)%pres": p01,
        "patch_icpp(1)%alpha_rho(1)": (eps/2.0)*rho_w,
        "patch_icpp(1)%alpha_rho(2)": (eps/2.0)*rho_v,
        "patch_icpp(1)%alpha_rho(3)": (1.0 - eps)*rho_a,
        "patch_icpp(1)%alpha(1)": eps/2.0,
        "patch_icpp(1)%alpha(2)": eps/2.0,
        "patch_icpp(1)%alpha(3)": 1.0 - eps,
        "patch_icpp(2)%geometry": 3,
        "patch_icpp(2)%alter_patch(1)": "T",
        "patch_icpp(2)%x_centroid": -1.25*D,
        "patch_icpp(2)%y_centroid": 0.0,
        "patch_icpp(2)%length_x": 1.5*D,
        "patch_icpp(2)%length_y": 3.0*D,
        "patch_icpp(2)%vel(1)": u_post,
        "patch_icpp(2)%vel(2)": 0.0,
        "patch_icpp(2)%pres": ps,
        "patch_icpp(2)%alpha_rho(1)": (eps/2.0)*rho_w,
        "patch_icpp(2)%alpha_rho(2)": (eps/2.0)*rho_post_v,
        "patch_icpp(2)%alpha_rho(3)": (1.0 - eps)*rho_post_a,
        "patch_icpp(2)%alpha(1)": eps/2.0,
        "patch_icpp(2)%alpha(2)": eps/2.0,
        "patch_icpp(2)%alpha(3)": 1.0 - eps,
        "patch_icpp(3)%geometry": 2,
        "patch_icpp(3)%alter_patch(1)": "T",
        "patch_icpp(3)%x_centroid": 0.5*D,
        "patch_icpp(3)%y_centroid": 0.0,
        "patch_icpp(3)%radius": 0.5*D,
        "patch_icpp(3)%vel(1)": 0.0,
        "patch_icpp(3)%vel(2)": 0.0,
        "patch_icpp(3)%pres": p01,
        "patch_icpp(3)%alpha_rho(1)": 0.95*rho_w,
        "patch_icpp(3)%alpha_rho(2)": 0.01*rho_v,
        "patch_icpp(3)%alpha_rho(3)": 0.04*rho_a,
        "patch_icpp(3)%alpha(1)": 0.95,
        "patch_icpp(3)%alpha(2)": 0.01,
        "patch_icpp(3)%alpha(3)": 0.04,
    })

    return case


case = build_case(reactions=False)

if __name__ == "__main__":
    print(json.dumps(case))
