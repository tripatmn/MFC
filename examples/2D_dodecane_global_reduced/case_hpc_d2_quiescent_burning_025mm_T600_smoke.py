#!/usr/bin/env python3
"""0.25 mm quiescent n-dodecane burning smoke at 600 K.

This matches the stable T600 nonreacting evaporation smoke condition and adds
baseline one-step gas chemistry with reaction heat. It intentionally does not
use the rate1000 stress mechanism.
"""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_quiescent_burning_025mm_T600_smoke")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            mechanism = path/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml"
            if (
                (case_dir/"case_hpc_d2_quiescent_evap_025mm_T600_smoke.py").is_file()
                and mechanism.is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with T600 0.25 mm pilot inputs.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"
CTFILE = str(REPO/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml")

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_quiescent_evap_025mm_T600_smoke import case as evap_case


def add_species_ic(case):
    air_y = [0.0, 0.233, 0.767, 0.0, 0.0]
    for pid in (1, 2):
        for i, value in enumerate(air_y, start=1):
            case[f"patch_icpp({pid})%Y({i})"] = value

    for i in range(1, 6):
        case[f"chem_wrt_Y({i})"] = "T"


case = dict(evap_case)
case.update({
    "chemistry": "T",
    "chem_params%diffusion": "F",
    "chem_params%reactions": "T",
    "cantera_file": CTFILE,
    "cantera_phase": "gas",
    "chem_gas_fluid_id": 2,
    "chem_gas_num_fluids": 2,
    "chem_gas_fluid_ids(1)": 2,
    "chem_gas_fluid_ids(2)": 3,
    "fuel_species_id": 1,
    "evap_species_source": "T",
    "evap_species_src": 0.0,
    "evap_liquid_fluid_id": 1,
    "evap_alpha_thresh": 0.01,
    "chem_reaction_heat_enable": "T",
    "chem_reaction_heat_limit_frac": 0.05,
    "chem_reaction_heat_diag": "F",
})
add_species_ic(case)

if __name__ == "__main__":
    print(json.dumps(case))
