#!/usr/bin/env python3
"""0.25 mm quiescent n-dodecane burning at 600 K, longer comparison run.

This matches the passed T600 burning smoke condition and extends only the
physical stop/save times. It uses the baseline one-step mechanism, not rate1000.
"""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_quiescent_burning_025mm_T600_long")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            mechanism = path/"examples"/"1D_dodecane_global_smoke"/"dodecane_global_1step.yaml"
            if (
                (case_dir/"case_hpc_d2_quiescent_burning_025mm_T600_smoke.py").is_file()
                and mechanism.is_file()
                and (path/"toolchain"/"main.py").is_file()
            ):
                return path
    raise RuntimeError("Could not find MFC repo root with T600 0.25 mm burning smoke inputs.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from case_hpc_d2_quiescent_burning_025mm_T600_smoke import case as smoke_case


case = dict(smoke_case)
case.update({
    "t_stop": 1.0e-4,
    "t_save": 1.0e-5,
})

if __name__ == "__main__":
    print(json.dumps(case))
