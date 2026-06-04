#!/usr/bin/env python3
"""Frolov-inspired 0.7 mm n-dodecane nonreacting evaporation bridge pilot.

This is a practical 2D equivalent-droplet bridge toward the Frolov and Basevich
2023 dodecane combustion conditions. It uses the self-ignition-size droplet
scale, but remains a short nonreacting evaporation pilot rather than a full
Frolov reproduction.
"""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_frolov_070mm_T700_evap_pilot")
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
    raise RuntimeError("Could not find MFC repo root with quiescent dodecane pilot inputs.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import case_hpc_d2_quiescent_evap_025mm_pilot as base


base.D0 = 0.7e-3
base.r_drop = 0.5*base.D0
base.domain = 8.0*base.D0
base.cells_per_diameter = 32
base.nx = 256
base.ny = 256
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
base.t_stop = 1.0e-4
base.t_save = 1.0e-5

case = base.build_case()

if __name__ == "__main__":
    print(json.dumps(case))
