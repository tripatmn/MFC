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

## Nonreacting Temperature Sweep

Before running more burning cases, use the nonreacting smoke sweep to find a
stable, non-explosive phase-change regime. These cases keep `D0 = 0.25 mm`,
`2 mm x 2 mm`, `256 x 256`, CFL-adaptive stepping with `cfl_target = 0.10`,
phase change on, and chemistry/reactions off. They use `p0 = 1 bar`,
`rho_l = 750 kg/m3`, `t_stop = 2.0e-5 s`, and `t_save = 2.0e-6 s`.

Current diagnostic branch note: `m_phase_change.fpp` restores the older
shock-vaporization mixture-cell threshold `mixM = 1.0e-5` for this sweep. This
is intended to test whether avoiding pTg activation in trace-mixture cells
reduces early over-vaporization and timestep collapse.

Run order: `T500` first, then `T600`, then `T700` only if the lower
temperatures are finite and do not show runaway vaporization or pressure
growth.

```bash
for TEMP in T500 T600 T700; do
  mkdir -p "$RUN_ROOT/evap_${TEMP}_smoke"
  cp "examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_evap_025mm_${TEMP}_smoke.py" \
    "$RUN_ROOT/evap_${TEMP}_smoke/case.py"
done
```

Run `T500`:

```bash
./mfc.sh run "$RUN_ROOT/evap_T500_smoke/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

After each completed or cancelled smoke, extract compact diagnostics:

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/extract_quiescent_burning_smoke_diagnostics.py \
  --run-dir "$RUN_ROOT/evap_T500_smoke" \
  --out-dir "$RUN_ROOT/evap_T500_smoke_diagnostics"
```

Repeat the same command pattern for `evap_T600_smoke` and `evap_T700_smoke`
only when the previous temperature is acceptable. Compare liquid `alpha_rho`
loss, vapor `alpha_rho` gain, pressure extrema, gas density minimum, finite
status, and `D2_mass_norm`.

## T600 Burning Smoke

After the `mixM = 1.0e-5` T600 nonreacting smoke is finite with bounded
pressure and a reasonable evaporation constant, run the matched burning smoke
with one MPI rank first.

```bash
RUN_ROOT="$PWD/runs/quiescent_025mm_condition_sweep_mixM1e5"
mkdir -p "$RUN_ROOT/T600_burning_n1"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_burning_025mm_T600_smoke.py \
  "$RUN_ROOT/T600_burning_n1/case.py"
```

The copied case locates its imported T600 evaporation base case from the repo,
so no additional case dependencies need to be copied when running under this
repository checkout.

JSON emission check:

```bash
build/venv/bin/python "$RUN_ROOT/T600_burning_n1/case.py" --mfc '{}' \
  > "$RUN_ROOT/T600_burning_n1/case.json"
```

Run `-n 1` first:

```bash
./mfc.sh run "$RUN_ROOT/T600_burning_n1/case.py" \
  -t pre_process simulation \
  --gpu acc -n 1 -j 8 --clean -b mpirun
```

Extract diagnostics:

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/extract_quiescent_burning_smoke_diagnostics.py \
  --run-dir "$RUN_ROOT/T600_burning_n1" \
  --out-dir "$RUN_ROOT/T600_burning_n1_diagnostics"
```

Compare the burning and nonreacting `mass_d2_estimate.K_mass_mm2_s` values
from their `smoke_diagnostics_summary.json` files to get the burning/nonreacting
K ratio. Species signs should be fuel/O2 negative and CO2/H2O positive.

## T600 Long Evaporation/Burning Comparison

After the T600 nonreacting and burning smoke cases both pass, run the matched
long comparison cases. These keep the same `D0 = 0.25 mm`, `2 mm x 2 mm`,
`256 x 256`, `p0 = 1 bar`, `T_hot = 600 K`, `rho_l = 750 kg/m3`, adaptive
`cfl_target = 0.10`, and `mixM = 1.0e-5` diagnostic phase-change threshold.
They extend only the runtime to `t_stop = 1.0e-4 s` with `t_save = 1.0e-5 s`.

```bash
RUN_ROOT="$PWD/runs/quiescent_025mm_condition_sweep_mixM1e5"
mkdir -p \
  "$RUN_ROOT/T600_evap_long" \
  "$RUN_ROOT/T600_burning_long" \
  "$RUN_ROOT/T600_long_comparison_diagnostics"

cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_evap_025mm_T600_long.py \
  "$RUN_ROOT/T600_evap_long/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_burning_025mm_T600_long.py \
  "$RUN_ROOT/T600_burning_long/case.py"
```

JSON emission checks:

```bash
build/venv/bin/python "$RUN_ROOT/T600_evap_long/case.py" --mfc '{}' \
  > "$RUN_ROOT/T600_evap_long/case.json"
build/venv/bin/python "$RUN_ROOT/T600_burning_long/case.py" --mfc '{}' \
  > "$RUN_ROOT/T600_burning_long/case.json"
```

Run nonreacting first with two ranks:

```bash
./mfc.sh run "$RUN_ROOT/T600_evap_long/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Run burning only after the nonreacting long run is finite:

```bash
./mfc.sh run "$RUN_ROOT/T600_burning_long/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

If the two-rank run stalls or diverges, fall back to the same case with
`-n 1` to separate local stiffness from MPI/rank effects:

```bash
./mfc.sh run "$RUN_ROOT/T600_burning_long/case.py" \
  -t pre_process simulation \
  --gpu acc -n 1 -j 8 --clean -b mpirun
```

After both long runs finish, extract the compact comparison diagnostics:

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/extract_quiescent_burning_smoke_diagnostics.py \
  --evap-run-dir "$RUN_ROOT/T600_evap_long" \
  --burning-run-dir "$RUN_ROOT/T600_burning_long" \
  --out-dir "$RUN_ROOT/T600_long_comparison_diagnostics"
```

The comparison extractor writes:

```text
$RUN_ROOT/T600_long_comparison_diagnostics/comparison_summary.json
$RUN_ROOT/T600_long_comparison_diagnostics/comparison_timeseries.csv
$RUN_ROOT/T600_long_comparison_diagnostics/D2_mass_mm2_vs_time_comparison.png
$RUN_ROOT/T600_long_comparison_diagnostics/D2_mass_norm_vs_time_comparison.png
$RUN_ROOT/T600_long_comparison_diagnostics/*alpha_liq*.png
$RUN_ROOT/T600_long_comparison_diagnostics/*pressure_final.png
$RUN_ROOT/T600_long_comparison_diagnostics/*vapor_alpha_rho_final.png
$RUN_ROOT/T600_long_comparison_diagnostics/burning_final_rhoY_*.png
```

Long-comparison PASS means both runs reach `t_stop = 1.0e-4 s` with finite
outputs, bounded pressure, stable adaptive `dt`, sensible liquid/vapor budgets,
and burning species signs of C12H26/O2 decreasing with CO2/H2O increasing.
FAIL means NaNs, pressure blowup, stall before `t_stop`, or species/energy
pathologies; preserve the run folder and extract diagnostics before retrying.

## Frolov 0.7 mm T700 Bridge Pilot

These cases are Frolov-inspired bridge pilots, not full Frolov and Basevich
2.8-4.3 mm reproductions. They use the smaller reported self-ignition droplet
scale, `D0 = 0.7 mm`, with `XO2 = 0.21`, `XN2 = 0.79`, `p0 = 1 bar`, and
`T_hot = 700 K`. The target literature burning constant scale is roughly
`0.38-0.50 mm2/s`; this short bridge run is intended to check stable coupled
behavior and early D2-law order, not a full droplet lifetime.

Case settings:

- `D0 = 0.7 mm`
- Domain `5.6 mm x 5.6 mm` (`8D`)
- Grid `256 x 256` (`32 cells/D`)
- `rho_l = 750 kg/m3`
- CFL-adaptive stepping, `cfl_target = 0.10`
- `t_stop = 1.0e-4 s`, `t_save = 1.0e-5 s`
- Burning uses the baseline one-step dodecane mechanism, not rate1000

Setup:

```bash
RUN_ROOT="$PWD/runs/frolov_dodecane_validation"
mkdir -p \
  "$RUN_ROOT/070mm_T700_evap_n1" \
  "$RUN_ROOT/070mm_T700_burning_n1" \
  "$RUN_ROOT/070mm_T700_comparison_diagnostics"

cp examples/2D_dodecane_global_reduced/case_hpc_d2_frolov_dodecane_070mm_T700_evap_pilot.py \
  "$RUN_ROOT/070mm_T700_evap_n1/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_frolov_dodecane_070mm_T700_burning_pilot.py \
  "$RUN_ROOT/070mm_T700_burning_n1/case.py"
```

JSON emission checks:

```bash
build/venv/bin/python "$RUN_ROOT/070mm_T700_evap_n1/case.py" --mfc '{}' \
  > "$RUN_ROOT/070mm_T700_evap_n1/case.json"
build/venv/bin/python "$RUN_ROOT/070mm_T700_burning_n1/case.py" --mfc '{}' \
  > "$RUN_ROOT/070mm_T700_burning_n1/case.json"
```

Run the nonreacting bridge first with one rank:

```bash
./mfc.sh run "$RUN_ROOT/070mm_T700_evap_n1/case.py" \
  -t pre_process simulation \
  --gpu acc -n 1 -j 8 --clean -b mpirun
```

Run the burning bridge only after nonreacting is finite:

```bash
./mfc.sh run "$RUN_ROOT/070mm_T700_burning_n1/case.py" \
  -t pre_process simulation \
  --gpu acc -n 1 -j 8 --clean -b mpirun
```

After both `-n 1` runs pass, repeat with `-n 2` in separate folders:

```bash
mkdir -p "$RUN_ROOT/070mm_T700_evap_n2" "$RUN_ROOT/070mm_T700_burning_n2"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_frolov_dodecane_070mm_T700_evap_pilot.py \
  "$RUN_ROOT/070mm_T700_evap_n2/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_frolov_dodecane_070mm_T700_burning_pilot.py \
  "$RUN_ROOT/070mm_T700_burning_n2/case.py"

./mfc.sh run "$RUN_ROOT/070mm_T700_evap_n2/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
./mfc.sh run "$RUN_ROOT/070mm_T700_burning_n2/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Analyze the one-rank bridge comparison:

```bash
build/venv/bin/python examples/2D_dodecane_global_reduced/extract_quiescent_burning_smoke_diagnostics.py \
  --evap-run-dir "$RUN_ROOT/070mm_T700_evap_n1" \
  --burning-run-dir "$RUN_ROOT/070mm_T700_burning_n1" \
  --out-dir "$RUN_ROOT/070mm_T700_comparison_diagnostics"
```

PASS means both runs reach `t_stop = 1.0e-4 s` with finite fields, bounded
pressure, stable adaptive `dt`, sensible liquid/vapor budgets, and burning
species signs of C12H26/O2 decreasing with CO2/H2O increasing. FAIL means NaNs,
pressure blowup, severe timestep collapse, stall, or species/energy
pathologies; extract diagnostics before changing conditions.

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
