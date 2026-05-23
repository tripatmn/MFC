#!/usr/bin/env python3
"""256x256 Zhang-style 1 mm nonreacting n-dodecane evaporation pilot.

This is the intermediate stability step after the successful 128x128 pilot and
before retrying the full 512x512 Zhang-style validation case. It keeps the same
physical target as case_hpc_d2_zhang2021_evap_1mm.py:
n-dodecane, D0 = 1.0 mm, p = 100000 Pa, T = 793.15 K encoded through gas
density, quiescent/no shock, chemistry off, and phase change enabled.
"""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_zhang2021_evap_1mm_pilot_256")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (
                (case_dir/"case_hpc_d2_zhang2021_evap_1mm.py").is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with Zhang-style evaporation case.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import case_hpc_d2_zhang2021_evap_1mm as zhang


# Intermediate grid: 64 cells per 1 mm droplet diameter in the same 4 mm box.
zhang.cells_per_diameter = 64
zhang.nx = 256
zhang.ny = 256
zhang.dx = zhang.domain/zhang.nx
zhang.dy = zhang.domain/zhang.ny

# Conservative dt recommended after the 128x128 stability pilot. This equals
# the full 512x512 dt, i.e. half of the nominal 256x256 acoustic-scaled dt.
zhang.dt = 1.1283507029868327e-9

case = zhang.build_case()
case["t_step_stop"] = 12000
case["t_step_save"] = 1000
case["t_step_print"] = 1000

if __name__ == "__main__":
    print(json.dumps(case))
