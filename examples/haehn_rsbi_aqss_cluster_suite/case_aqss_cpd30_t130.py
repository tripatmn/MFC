#!/usr/bin/env python3
"""Haehn/RSBI fixed-AQSS coarse reference diagnostic, 30 cells per D0."""

import json
import pathlib
import runpy
import sys


PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
BASE_CASE = PACKAGE_DIR / "case_base.py"
MECHANISM = PACKAGE_DIR / "h2o2_xe.yaml"

HAEHN_ARGS = [
    "--haehn-xe",
    "--d0", "0.04",
    "--cells-per-d0", "30",
    "--x-minus-d0", "2.0",
    "--x-plus-d0", "7.0",
    "--y-half-d0", "3.2",
    "--shock-offset-d0", "1.0",
    "--mach", "2.83",
    "--ctfile", str(MECHANISM),
    "--xbub", "H2:0.29,O2:0.15,XE:0.56",
    "--pamb", "101325",
    "--tamb", "295",
    "--tend", "2.0e-4",
    "--react", "T",
]

sys.argv = [str(BASE_CASE), *HAEHN_ARGS]
case = runpy.run_path(str(BASE_CASE))["case"]

case["t_step_stop"] = 130
case["t_step_save"] = 10
case["t_step_print"] = 1
case["mpp_lim"] = "F"
case["chem_params%reactions"] = "T"
case["chem_params%reaction_substeps"] = 1000
case["chem_params%adap_substeps"] = "F"
case["cons_vars_wrt"] = "T"
case["pres_wrt"] = "T"
case["chem_wrt_T"] = "T"
case["chem_reaction_heat_diag"] = "T"

if __name__ == "__main__":
    print(json.dumps(case))
