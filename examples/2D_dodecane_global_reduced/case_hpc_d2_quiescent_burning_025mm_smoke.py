#!/usr/bin/env python3
"""Short 0.25 mm quiescent n-dodecane burning smoke test.

This keeps the matched 0.25 mm burning pilot physics unchanged and shortens
only the physical runtime/output cadence to check whether burning survives past
the nonreacting NaN time scale.
"""

import json

from case_hpc_d2_quiescent_burning_025mm_pilot import case


case = dict(case)
case["t_stop"] = 2.0e-5
case["t_save"] = 1.0e-6

if __name__ == "__main__":
    print(json.dumps(case))
