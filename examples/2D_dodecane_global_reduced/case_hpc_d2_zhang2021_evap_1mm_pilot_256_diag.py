#!/usr/bin/env python3
"""256x256 Zhang-style evaporation hang diagnostic case.

This uses the same physics and timestep as the fixed_dt_1p0 pilot but stops
shortly after the observed hang window. Run with TEMP_ZHANG_EVAP_HANG_DIAG=1
to activate temporary source-level trace markers for t_step 8900..9200.
"""

import json

import case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_1p0 as base


case = dict(base.case)
case["t_step_stop"] = 9300
case["t_step_save"] = 3000
case["t_step_print"] = 100

if __name__ == "__main__":
    print(json.dumps(case))
