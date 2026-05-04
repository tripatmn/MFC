#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import cantera as ct

base = pathlib.Path(__file__).with_name('case.py')
raw = subprocess.check_output([sys.executable, str(base), *sys.argv[1:]], text=True)
case = json.loads(raw)

case['chemistry'] = 'T'
case['cantera_file'] = 'h2o2.yaml'
case['chem_params%diffusion'] = 'F'
case['chem_params%reactions'] = 'F'
case['chem_gas_fluid_id'] = 2
case['fuel_species_id'] = 1
case['evap_species_source'] = 'F'
case['evap_species_src'] = 1.0e-1
case['evap_liquid_fluid_id'] = 1
case['evap_alpha_thresh'] = 0.01

sol = ct.Solution(case['cantera_file'])
for s in range(sol.n_species):
    case[f'chem_wrt_Y({s + 1})'] = 'T'
    yval = 1.0 if s == 0 else 0.0
    for pidx in (1, 2, 3):
        case[f'patch_icpp({pidx})%Y({s + 1})'] = yval

print(json.dumps(case))
