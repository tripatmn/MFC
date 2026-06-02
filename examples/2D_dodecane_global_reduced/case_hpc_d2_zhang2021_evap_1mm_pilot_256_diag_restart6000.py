#!/usr/bin/env python3
"""Restart wrapper for the 256x256 Zhang evaporation hang diagnostic.

This starts from existing p_all output at t_step = 6000 and keeps the same
physics/timestep as the diagnostic case. It is only for reaching the
TEMP_ZHANG_EVAP_HANG_DIAG window faster during local diagnosis.
"""

import json

import case_hpc_d2_zhang2021_evap_1mm_pilot_256_diag as base


case = dict(base.case)
case["old_ic"] = "T"
case["old_grid"] = "T"
case["t_step_old"] = 0
case["t_step_start"] = 6000
case["t_step_stop"] = 9300
case["t_step_save"] = 3000
case["t_step_print"] = 100

if __name__ == "__main__":
    print(json.dumps(case))
