#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(prog="2D_dodecane_global_reduced_on")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from case_hpc_reactions_off import build_case

case = build_case(reactions=True)

if __name__ == "__main__":
    print(json.dumps(case))
