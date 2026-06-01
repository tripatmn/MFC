#!/usr/bin/env python3
"""256x256 Zhang-style evaporation stability sweep: fixed dt 0.25x.

Physics is identical to case_hpc_d2_zhang2021_evap_1mm_pilot_256.py. This is
the conservative fixed-step diagnostic if the 1.0x and 0.5x variants stall or
develop NaNs.
"""

import json

import case_hpc_d2_zhang2021_evap_1mm_pilot_256 as base


base.zhang.dt = 0.25*1.1283507029868327e-9

case = base.zhang.build_case()
case["t_step_stop"] = 12000
case["t_step_save"] = 2000
case["t_step_print"] = 1000

if __name__ == "__main__":
    print(json.dumps(case))
