#!/usr/bin/env python3
"""HPC Reacting shock-droplet SK54 combined stable case (t_stop = 190ns)."""

import argparse
import json
import pathlib
import subprocess
import sys


parser = argparse.ArgumentParser(prog="2D_shockdroplet_air_sk54_combined_stable_t190ns")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

base = pathlib.Path(__file__).with_name("case_hpc_shockdroplet_air_sk54_sanity.py")
raw = subprocess.check_output(
    [sys.executable, str(base), "--mfc", json.dumps(args.mfc)],
    text=True,
)
case = json.loads(raw)

# General settings
case["t_stop"] = 1.9e-7
case["t_save"] = 1.0e-8
case["parallel_io"] = "F"

# Boundary conditions
case["bc_y%beg"] = -3
case["bc_y%end"] = -3

# Chemistry and Evaporation
case["chemistry"] = "T"
case["chem_params%reactions"] = "T"
case["chem_reaction_heat_enable"] = "T"
case["evap_species_source"] = "T"
case["chem_reaction_heat_diag"] = "T"

# Limiter settings
case["chem_species_nonneg_limiter"] = "T"
case["chem_alpha_gas_min"] = 1.0e-4
case["chem_rho_gas_min"] = 1.0e-6

print(json.dumps(case))
