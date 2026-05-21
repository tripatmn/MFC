#!/usr/bin/env python3
"""Lean-output high-grid clean baseline dodecane burning validation case."""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_clean_burning_validation_high_lean")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            mechanism = path/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml"
            if (
                (case_dir/"case_hpc_d2_clean_validation_common.py").is_file()
                and mechanism.is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with clean validation cases.")


def apply_lean_output(case):
    # Keep pressure available for analyze_grid_validation.py while reducing
    # high-grid D/ and p_all output cadence.
    case["t_step_save"] = 200
    case["prim_vars_wrt"] = "T"
    case["pres_wrt"] = "T"
    return case


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"
CTFILE = str(REPO/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml")

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_clean_validation_common import add_baseline_burning, build_evap_validation_case


case = apply_lean_output(add_baseline_burning(build_evap_validation_case(256), CTFILE))

if __name__ == "__main__":
    print(json.dumps(case))
