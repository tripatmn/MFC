#!/usr/bin/env python3
import cantera as ct
import mfc.viz
import numpy as np

from case import dt

sol = ct.Solution("nDodecane_Reitz.yaml", "nDodecane_IG")
case = mfc.viz.Case(".", dt)

fields = {
    "c12h26": f"prim.{5 + sol.species_index('c12h26')}",
    "o2": f"prim.{5 + sol.species_index('o2')}",
    "co2": f"prim.{5 + sol.species_index('co2')}",
    "h2o": f"prim.{5 + sol.species_index('h2o')}",
}

for name, field in fields.items():
    case.load_variable(name, field)

steps = sorted(case.get_timesteps())
if len(steps) < 2:
    raise SystemExit(f"Need at least two saved steps; found {steps}")

results = {}
all_finite = True
for name in fields:
    initial = np.asarray(case.get_data()[steps[0]][name], dtype=float)
    final = np.asarray(case.get_data()[steps[-1]][name], dtype=float)
    finite = bool(np.isfinite(initial).all() and np.isfinite(final).all())
    all_finite = all_finite and finite
    results[name] = (float(initial.mean()), float(final.mean()), float(final.mean() - initial.mean()), finite)

print(f"steps {steps}")
for name, (initial, final, delta, finite) in results.items():
    print(f"{name} initial_avg={initial:.16e} final_avg={final:.16e} delta_avg={delta:.16e} finite={finite}")

checks = [
    all_finite,
    results["c12h26"][2] < 0.0,
    results["o2"][2] < 0.0,
    results["co2"][2] > 0.0,
    results["h2o"][2] > 0.0,
]

if not all(checks):
    raise SystemExit("Dodecane chemistry smoke checks failed")
