#!/usr/bin/env python3
"""Short complete-run variant of the coarse air-only SK54 shock-droplet sanity case."""

import argparse
import json
import pathlib
import subprocess
import sys


parser = argparse.ArgumentParser(prog="2D_shockdroplet_air_sk54_sanity_short")
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
args = parser.parse_args()

base = pathlib.Path(__file__).with_name("case_hpc_shockdroplet_air_sk54_sanity.py")
raw = subprocess.check_output(
    [sys.executable, str(base), "--mfc", json.dumps(args.mfc)],
    text=True,
)
case = json.loads(raw)
case["t_stop"] = 1.5e-7

print(json.dumps(case))
