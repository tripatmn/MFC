#!/usr/bin/env python3
"""256x256 Zhang-style evaporation stability sweep: conservative CFL-adaptive.

Physics is identical to case_hpc_d2_zhang2021_evap_1mm_pilot_256.py. This
variant switches to MFC's CFL-adaptive timestep mode with a conservative target
and the same physical end time as the fixed 1.0x 12000-step run.
"""

import json

import case_hpc_d2_zhang2021_evap_1mm_pilot_256 as base


BASE_DT = 1.1283507029868327e-9
T_STOP = 12000*BASE_DT

base.zhang.dt = 0.5*BASE_DT

case = base.zhang.build_case()
case.pop("t_step_start", None)
case.pop("t_step_stop", None)
case.pop("t_step_save", None)
case["cfl_adap_dt"] = "T"
case["cfl_target"] = 0.10
case["n_start"] = 0
case["t_stop"] = T_STOP
case["t_save"] = T_STOP/6.0
case["t_step_print"] = 1000

if __name__ == "__main__":
    print(json.dumps(case))
