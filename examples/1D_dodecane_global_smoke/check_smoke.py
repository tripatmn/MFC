#!/usr/bin/env python3
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
NX = 8
DX = 1.0e-3/NX

SPECIES = {
    "c12h26": 5,
    "o2": 6,
    "n2": 7,
    "co2": 8,
    "h2o": 9,
}


def read_field(var, step):
    path = ROOT/"D"/f"cons.{var}.00.{step:06d}.dat"
    data = np.loadtxt(path)
    return data[:, -1]


def integrate(values):
    return float(np.sum(values)*DX)


steps = sorted(
    int(path.name.rsplit(".", 2)[1])
    for path in (ROOT/"D").glob("cons.5.00.*.dat")
)
if len(steps) < 2:
    raise SystemExit(f"Need at least two primitive output steps; found {steps}")

initial_step = steps[0]
final_step = steps[-1]
results = {}
all_finite = True

for name, var in SPECIES.items():
    initial = read_field(var, initial_step)
    final = read_field(var, final_step)
    finite = bool(np.isfinite(initial).all() and np.isfinite(final).all())
    all_finite = all_finite and finite
    results[name] = {
        "initial": integrate(initial),
        "final": integrate(final),
        "change": integrate(final) - integrate(initial),
        "finite": finite,
    }

print(f"steps {steps}")
for name, stats in results.items():
    print(
        f"{name} initial={stats['initial']:.16e} "
        f"final={stats['final']:.16e} "
        f"change={stats['change']:.16e} finite={stats['finite']}"
    )

n2_ref = max(abs(results["n2"]["initial"]), 1.0e-300)
checks = [
    all_finite,
    results["c12h26"]["change"] < 0.0,
    results["o2"]["change"] < 0.0,
    results["co2"]["change"] > 0.0,
    results["h2o"]["change"] > 0.0,
    abs(results["n2"]["change"])/n2_ref < 1.0e-10,
]

if not all(checks):
    print("Global one-step dodecane smoke checks failed", file=sys.stderr)
    raise SystemExit(1)

print("PASS")
