#!/usr/bin/env python3
"""V10 experiment: V5 Model-3 species gradient with chemistry diffusion enabled."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v10_case_common import build_case, parse_case_args  # noqa: E402

args = parse_case_args("V10_model3_species_diffusion_on")
case = build_case(diffusion=True, args=args)

if __name__ == "__main__":
    print(json.dumps(case))
