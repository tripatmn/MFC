#!/usr/bin/env python3
"""V8: reacting version of the validated V7 shock/droplet setup.

The physical setup is loaded directly from V7 and only the requested validation
deltas are applied here: reactions on, adaptive AQSS enabled, a 825 x 90 grid,
a 2 us smoke duration, and a moderate chemistry-output cadence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve()
V7_CASE = HERE.parents[1] / "V7_single_droplet_phase_change" / "case.py"


def load_v7_case() -> dict:
    spec = importlib.util.spec_from_file_location("v7_single_droplet_case", V7_CASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load V7 case from {V7_CASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.case)


case = load_v7_case()

# Adaptive AQSS in the current solver uses chem_params%reaction_substeps as the
# rank-wide minimum/floor and chem_params%reaction_substeps_max as the cap.
case.update(
    {
        "m": 825,
        "n": 90,
        "t_stop": 2.0e-6,
        "t_save": 2.0e-7,
        "t_step_print": 100,
        "chem_params%reactions": "T",
        "chem_params%reaction_substeps": 2,
        "chem_params%adap_substeps": "T",
        "chem_params%reaction_substeps_max": 50,
        "chem_params%diffusion": "T",
    }
)


if __name__ == "__main__":
    print(json.dumps(case))
