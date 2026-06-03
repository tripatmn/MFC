# Quiescent 0.25 mm Dodecane Pilot Runs

These runs are the first literature-scaled bridge pilots from
`docs/dev/literature_scaled_quiescent_burning_pilot_plan.md`.

- Droplet: n-dodecane, `D0 = 0.25 mm`
- Domain: `2 mm x 2 mm`
- Grid: `256 x 256`, `32 cells/D0`
- Time control: CFL-adaptive, `cfl_target = 0.10`
- Physical stop time: `0.5 ms`
- Save interval: `1.0e-5 s`, about 50 saved samples plus the initial state
- Primary metric: mass-equivalent D2 from liquid `alpha_rho`
- Secondary metric: threshold equivalent-area D2 from `alpha_liq >= 0.5`

The expected literature-scale D2 change is small: about `0.4%` at `0.5 ms` if
`K = 0.5 mm2/s`. Run the nonreacting case first. Submit burning only after the
nonreacting pilot is finite and the liquid/vapor budget is sensible.

## Sync And Build

```bash
git fetch
git checkout phasechange_local_dev
git pull --ff-only
bash ./mfc.sh build -j 8
```

## Run Folder Setup

```bash
RUN_ROOT="$PWD/runs/quiescent_025mm_pilot"
mkdir -p "$RUN_ROOT/evap" "$RUN_ROOT/burning" "$RUN_ROOT/analysis"

cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_evap_025mm_pilot.py \
  "$RUN_ROOT/evap/case.py"

cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_burning_025mm_pilot.py \
  "$RUN_ROOT/burning/case.py"
```

## Run Nonreacting First

Foreground or interactive allocation:

```bash
./mfc.sh run "$RUN_ROOT/evap/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch:

```bash
./mfc.sh run "$RUN_ROOT/evap/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

## Analyze Nonreacting

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/analyze_quiescent_025mm_pilot.py \
  --evap "$RUN_ROOT/evap" \
  --out-dir "$RUN_ROOT/analysis"
```

Check:

- all fields finite;
- liquid `alpha_rho` decreases smoothly;
- vapor `alpha_rho` increase is consistent with liquid loss;
- mass-equivalent D2 is smoother than threshold D2.

## Run Burning After Nonreacting Passes

The burning case uses the baseline one-step mechanism, reactions on, and
reaction heat with `chem_reaction_heat_limit_frac = 0.05`. It does not use the
rate1000 stress mechanism.

If the nonreacting pilot fails early, run the short burning smoke before trying
the full `0.5 ms` burning pilot. This keeps the same physics and stops at
`2.0e-5 s`, with saves every `1.0e-6 s`.

```bash
mkdir -p "$RUN_ROOT/burning_smoke"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_burning_025mm_smoke.py \
  "$RUN_ROOT/burning_smoke/case.py"

./mfc.sh run "$RUN_ROOT/burning_smoke/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch smoke:

```bash
./mfc.sh run "$RUN_ROOT/burning_smoke/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

Smoke PASS means the case reaches `t_stop = 2.0e-5 s` with finite outputs.
Smoke FAIL means NaNs, pressure blowup, or a stall before `t_stop`; record the
first failing time/step and compare it with the nonreacting failure near
`t = 5.46e-6 s`.

If the smoke run stalls or is manually cancelled, extract a compact in-place
diagnostic bundle on HPC instead of copying the full raw output:

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/extract_quiescent_burning_smoke_diagnostics.py \
  --run-dir "$RUN_ROOT/burning_smoke" \
  --out-dir "$RUN_ROOT/burning_smoke_diagnostics"
```

The extractor reads logs plus the first, previous-to-last, and last saved raw
states from `D/` or `p_all/`, then writes JSON/CSV/text summaries and small PNG
plots when plotting support is available.

Foreground or interactive allocation:

```bash
./mfc.sh run "$RUN_ROOT/burning/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch:

```bash
./mfc.sh run "$RUN_ROOT/burning/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

## Analyze Both

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/analyze_quiescent_025mm_pilot.py \
  --evap "$RUN_ROOT/evap" \
  --burning "$RUN_ROOT/burning" \
  --out-dir "$RUN_ROOT/analysis"
```

The analyzer writes:

```text
$RUN_ROOT/analysis/quiescent_025mm_pilot_summary.csv
$RUN_ROOT/analysis/quiescent_025mm_pilot_timeseries.csv
$RUN_ROOT/analysis/quiescent_025mm_pilot_K_table.csv
$RUN_ROOT/analysis/quiescent_025mm_pilot_comparison.csv
$RUN_ROOT/analysis/quiescent_025mm_pilot_fit_summary.txt
$RUN_ROOT/analysis/*.png
```
