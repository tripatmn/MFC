#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys

base = pathlib.Path(__file__).with_name('case.py')
raw = subprocess.check_output([sys.executable, str(base), *sys.argv[1:]], text=True)
case = json.loads(raw)
case['chemistry'] = 'T'
case['chem_params%reactions'] = 'F'
case['user_species_source'] = 'T'
case['user_species_id'] = 1
case['user_species_src'] = 1.0e-3
print(json.dumps(case))
