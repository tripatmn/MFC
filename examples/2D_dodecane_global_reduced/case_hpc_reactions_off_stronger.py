#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_global_reduced_off_stronger")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            if (case_dir/"case_hpc_reactions_off.py").is_file() and (path/"toolchain"/"main.py").is_file():
                return path
    raise RuntimeError("Could not find MFC repo root with 2D_dodecane_global_reduced cases.")


REPO = find_repo_root()
sys.path.insert(0, str(REPO/"examples"/"2D_dodecane_global_reduced"))
from case_hpc_reactions_off import build_case

case = build_case(reactions=False)
case["cantera_file"] = str(
    REPO/"examples"/"2D_dodecane_global_reduced"/"dodecane_global_1step_rate1000.yaml"
)

if __name__ == "__main__":
    print(json.dumps(case))
