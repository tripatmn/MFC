# Zhang 2021-Style 1 mm Dodecane Validation Runs

These cases target a Zhang et al. 2021-style quiescent n-dodecane droplet
validation condition:

- Initial droplet diameter: 1.0 mm
- Ambient pressure: 100000 Pa
- Ambient gas temperature: 793.15 K, encoded through the initial gas density
- Geometry: 2D Cartesian equivalent droplet, no shock
- Primary metric: mass-equivalent `D2_norm` from liquid `alpha_rho`
- Secondary metric: threshold `D2_norm` from `alpha_liq >= 0.5`

The exact Zhang `D2(t)` curve, evaporation constant `K`, and regression window
still need to be digitized from the paper before making an absolute K claim.

## Sync Branch On HPC

From the MFC repository on HPC:

```bash
git fetch
git checkout phasechange_local_dev
git pull --ff-only
```

Build the GPU/OpenACC target before submitting runs if the executable is not
already current:

```bash
bash ./mfc.sh build -j 8
```

## Run Folder Setup

Use a dedicated run root under `runs/zhang2021_validation/`:

```bash
RUN_ROOT="$PWD/runs/zhang2021_validation"
mkdir -p "$RUN_ROOT/evap_1mm_pilot_256" "$RUN_ROOT/evap_1mm" "$RUN_ROOT/burning_1mm"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256.py "$RUN_ROOT/evap_1mm_pilot_256/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm.py "$RUN_ROOT/evap_1mm/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_burning_1mm.py "$RUN_ROOT/burning_1mm/case.py"
```

## Submit 256 x 256 Evaporation Pilot

Run this intermediate nonreacting stability pilot before retrying the full
512 x 512 evaporation case. It uses the same physical target, 64 cells per
initial droplet diameter, `dt = 1.1283507029868327e-9`, and
`t_step_stop = 12000`.

Interactive allocation or foreground launch:

```bash
./mfc.sh run "$RUN_ROOT/evap_1mm_pilot_256/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch submission through MFC's built-in batch template:

```bash
./mfc.sh run "$RUN_ROOT/evap_1mm_pilot_256/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

## Submit Evaporation

Run pre_process + simulation only. Do not run post_process. Wait to retry this
full 512 x 512 evaporation case until the 256 x 256 pilot passes.

Interactive allocation or foreground launch:

```bash
./mfc.sh run "$RUN_ROOT/evap_1mm/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch submission through MFC's built-in batch template:

```bash
./mfc.sh run "$RUN_ROOT/evap_1mm/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

Adjust the wall time/account flags for the allocation policy on the target
Nautilus queue.

## Submit Burning

The burning case uses the baseline
`examples/1D_dodecane_global_smoke/dodecane_global_1step.yaml` mechanism,
`chem_reaction_heat_enable = T`, and `chem_reaction_heat_limit_frac = 0.05`.
It does not use the rate1000 stress mechanism.

```bash
./mfc.sh run "$RUN_ROOT/burning_1mm/case.py" \
  -t pre_process simulation \
  --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch submission:

```bash
./mfc.sh run "$RUN_ROOT/burning_1mm/case.py" \
  -t pre_process simulation \
  -e batch -c nautilus -N 1 -n 2 -j 8 \
  --gpu acc --clean
```

Submit evaporation and burning as separate jobs, or make burning depend on
successful evaporation completion if using a job scheduler wrapper.

## Analyze Completed Runs

After both simulations finish:

```bash
python examples/2D_dodecane_global_reduced/analyze_zhang2021_validation.py \
  --evap "$RUN_ROOT/evap_1mm" \
  --burning "$RUN_ROOT/burning_1mm" \
  --out "$RUN_ROOT/zhang2021_validation_summary.csv" \
  --plots-dir "$RUN_ROOT/plots"
```

The analyzer writes:

```text
$RUN_ROOT/zhang2021_validation_summary.csv
$RUN_ROOT/zhang2021_validation_timeseries.csv
$RUN_ROOT/zhang2021_validation_fit_summary.txt
$RUN_ROOT/zhang2021_validation_comparison.csv
$RUN_ROOT/plots/*.png
```

Use the mass-equivalent normalized D2 curve as the primary MFC validation curve.
Use the threshold curve as a secondary shape diagnostic. The comparison remains
a 2D equivalent-diameter comparison until the Zhang data and mapping assumptions
are finalized.
