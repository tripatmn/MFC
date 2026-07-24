#!/usr/bin/env python3
"""C1: SK54 Mach-3 shock/droplet, diffusion ON, reactions OFF, 0.2 us."""

import json
import pathlib
import runpy
import contextlib
import io

root = pathlib.Path(__file__).resolve().parents[1]
source = root / "full_2us_evap_only_chemistry_OFF" / "case.py"
fallback_source = root / "full_1us_muscl_minmod" / "case.py"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path(str(source), run_name="mfc_source_case")
except FileNotFoundError:
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path(str(fallback_source), run_name="mfc_source_case")
case = namespace["case"]

# Fresh non-restart run.
case["old_ic"] = "F"
case["old_grid"] = "F"
case["n_start"] = 0
case["t_step_old"] = 0
case["t_step_start"] = 0

# Short diffusion test interval; keep the established saved-state cadence.
case["t_stop"] = 2.0e-7
case["t_save"] = 5.0e-8
case["t_step_print"] = 50

# Chemistry species machinery remains active; isolate diffusion from reactions.
case["chemistry"] = "T"
case["chem_params%diffusion"] = "T"
case["chem_params%transport_model"] = 1  # mixture-average transport
case["chem_params%reactions"] = "F"
case["chem_reaction_heat_enable"] = "F"
case["chem_reaction_heat_diag"] = "F"

# Keep serial p_all/restart-compatible output cadence with no parallel I/O.
case["parallel_io"] = "F"
case["prim_vars_wrt"] = "T"
case["cons_vars_wrt"] = "T"
case["pres_wrt"] = "T"
case["chem_wrt_T"] = "T"

print(json.dumps(case))
