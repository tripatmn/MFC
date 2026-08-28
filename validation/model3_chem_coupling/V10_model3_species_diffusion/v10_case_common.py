#!/usr/bin/env python3
"""Shared V10 case builder for paired Model-3 species-diffusion checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    SK54_CTPHASE,
    SK54_CTFILE,
    SK54_FUEL,
    add_dodecane_layout_fluids,
    load_solution,
    model3_output_flags,
    selected_species_ids,
    set_species,
    species_index,
    split_model3_gas_state,
)

LEFT_COMPOSITION = "{fuel}:0.02,O2:0.20,N2:0.78"
RIGHT_COMPOSITION = "{fuel}:0.001,O2:0.40,N2:0.599"
LENGTH = 1.0e-3
CELLS = 64
DT = 1.0e-8
T_STEP_STOP = 2000
T_STEP_SAVE = 200


def parse_case_args(prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
    parser.add_argument("--ctfile", default=SK54_CTFILE)
    parser.add_argument("--ctphase", default=SK54_CTPHASE)
    parser.add_argument("--fuel-species", default=SK54_FUEL)
    return parser.parse_args()


def build_case(*, diffusion: bool, args: argparse.Namespace) -> dict:
    left = load_solution(args.ctfile, args.ctphase)
    right = load_solution(args.ctfile, args.ctphase)
    left.TPX = 300.0, 101325.0, LEFT_COMPOSITION.format(fuel=args.fuel_species)
    right.TPX = 300.0, 101325.0, RIGHT_COMPOSITION.format(fuel=args.fuel_species)

    fuel_species_id = species_index(left, args.fuel_species) + 1
    left_state = split_model3_gas_state(left, args.fuel_species, alpha_liquid=0.10)
    right_state = split_model3_gas_state(right, args.fuel_species, alpha_liquid=0.45)

    case = {
        "run_time_info": "T",
        "x_domain%beg": 0.0,
        "x_domain%end": LENGTH,
        "m": CELLS,
        "n": 0,
        "p": 0,
        "dt": DT,
        "t_step_start": 0,
        "t_step_stop": T_STEP_STOP,
        "t_step_save": T_STEP_SAVE,
        "t_step_print": 200,
        "num_probes": 1,
        "probe(1)%x": 0.0,
        "model_eqns": 3,
        "num_fluids": 3,
        "num_patches": 2,
        "mpp_lim": "F",
        "mixture_err": "T",
        "relax": "F",
        "time_stepper": "rk3",
        "weno_order": 1,
        "weno_eps": 1.0e-16,
        "mapped_weno": "F",
        "mp_weno": "F",
        "riemann_solver": "hllc",
        "wave_speeds": "direct",
        "avg_state": "arithmetic",
        "bc_x%beg": -1,
        "bc_x%end": -1,
        "format": "silo",
        "precision": "double",
        "parallel_io": "F",
        "viscous": "F",
        "chemistry": "T",
        "model3_chemistry_coupling": "T",
        "fuel_species_id": fuel_species_id,
        "chem_params%diffusion": "T" if diffusion else "F",
        "chem_params%reactions": "F",
        "chem_params%reaction_substeps": 0,
        "chem_params%adap_substeps": "F",
        "chem_params%reaction_substeps_max": 0,
        "chem_params%transport_model": 1,
        "cantera_file": args.ctfile,
        "flux_wrt(1)": "T",
        "patch_icpp(1)%geometry": 1,
        "patch_icpp(1)%x_centroid": 0.25 * LENGTH,
        "patch_icpp(1)%length_x": 0.5 * LENGTH,
        "patch_icpp(1)%vel(1)": 0.0,
        "patch_icpp(1)%pres": left.P,
        "patch_icpp(1)%alpha(1)": left_state["alpha_liquid"],
        "patch_icpp(1)%alpha(2)": left_state["alpha_vapor"],
        "patch_icpp(1)%alpha(3)": left_state["alpha_carrier"],
        "patch_icpp(1)%alpha_rho(1)": left_state["alpha_rho_liquid"],
        "patch_icpp(1)%alpha_rho(2)": left_state["alpha_rho_vapor"],
        "patch_icpp(1)%alpha_rho(3)": left_state["alpha_rho_carrier"],
        "patch_icpp(2)%geometry": 1,
        "patch_icpp(2)%alter_patch(1)": "T",
        "patch_icpp(2)%x_centroid": 0.75 * LENGTH,
        "patch_icpp(2)%length_x": 0.5 * LENGTH,
        "patch_icpp(2)%vel(1)": 0.0,
        "patch_icpp(2)%pres": right.P,
        "patch_icpp(2)%alpha(1)": right_state["alpha_liquid"],
        "patch_icpp(2)%alpha(2)": right_state["alpha_vapor"],
        "patch_icpp(2)%alpha(3)": right_state["alpha_carrier"],
        "patch_icpp(2)%alpha_rho(1)": right_state["alpha_rho_liquid"],
        "patch_icpp(2)%alpha_rho(2)": right_state["alpha_rho_vapor"],
        "patch_icpp(2)%alpha_rho(3)": right_state["alpha_rho_carrier"],
    }

    add_dodecane_layout_fluids(case)
    model3_output_flags(case)
    set_species(case, left, [1], selected_species_ids(left, [args.fuel_species, "O2", "N2", "CO2", "H2O"]))
    set_species(case, right, [2], selected_species_ids(right, [args.fuel_species, "O2", "N2", "CO2", "H2O"]))
    return case
