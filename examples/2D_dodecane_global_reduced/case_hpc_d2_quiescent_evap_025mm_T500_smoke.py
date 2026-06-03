#!/usr/bin/env python3
"""0.25 mm quiescent nonreacting evaporation smoke at 500 K.

This controlled phase-change sweep case keeps the 0.25 mm geometry/grid and
turns off chemistry. It lowers the hot ambient condition from the aggressive
900 K, 2 bar pilot to 500 K, 1 bar and uses rho_l = 750 kg/m3.
"""

import json

import case_hpc_d2_quiescent_evap_025mm_pilot as base


base.p0 = 1.0e5
base.t_hot = 500.0
base.rho_l = 750.0
base.rho_v_hot = base.p0/((base.gamma_v - 1.0)*base.cv_v*base.t_hot)
base.rho_air_hot = base.p0/((base.gamma_air - 1.0)*base.cv_air*base.t_hot)
base.c_liq = (base.gamma_l*(base.p0 + base.pi_inf_l)/base.rho_l)**0.5
base.dt_initial = 0.75*base.cfl_target*base.dx/base.c_liq
base.t_stop = 2.0e-5
base.t_save = 2.0e-6

case = base.build_case()

if __name__ == "__main__":
    print(json.dumps(case))
