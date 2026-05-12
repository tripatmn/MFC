#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_clean_evap_pilot_v2")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_case_dir():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (case_dir/"case_hpc_d2_clean_evap_pilot.py").is_file() and (path/"toolchain"/"main.py").is_file():
                return case_dir
    raise RuntimeError("Could not find case_hpc_d2_clean_evap_pilot.py.")


CASE_DIR = find_case_dir()
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_clean_evap_pilot import build_case


case = build_case()
case["dt"] *= 0.75
case["t_step_stop"] = 3400
case["t_step_save"] = 50
case["t_step_print"] = 100

if __name__ == "__main__":
    print(json.dumps(case))
