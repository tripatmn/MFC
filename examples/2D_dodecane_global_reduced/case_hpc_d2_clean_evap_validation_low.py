#!/usr/bin/env python3
"""Low-grid clean nonreacting dodecane evaporation validation case."""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_clean_evap_validation_low")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_case_dir():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (case_dir/"case_hpc_d2_clean_validation_common.py").is_file() and (path/"toolchain"/"main.py").is_file():
                return case_dir
    raise RuntimeError("Could not find clean validation case helper.")


CASE_DIR = find_case_dir()
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_clean_validation_common import build_evap_validation_case


case = build_evap_validation_case(128)

if __name__ == "__main__":
    print(json.dumps(case))
