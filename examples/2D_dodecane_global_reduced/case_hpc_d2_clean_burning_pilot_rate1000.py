#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_clean_burning_pilot_rate1000")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_case_dir():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (
                (case_dir/"case_hpc_d2_clean_burning_pilot.py").is_file()
                and (case_dir/"dodecane_global_1step_rate1000.yaml").is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return case_dir
    raise RuntimeError("Could not find clean burning pilot and rate1000 mechanism.")


CASE_DIR = find_case_dir()
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_clean_burning_pilot import case as burning_case


case = dict(burning_case)
case["cantera_file"] = str(CASE_DIR/"dodecane_global_1step_rate1000.yaml")

if __name__ == "__main__":
    print(json.dumps(case))
