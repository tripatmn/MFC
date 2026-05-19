# Quiescent Dodecane Grid-Validation Runs

These cases are for quiescent n-dodecane evaporation/burning grid validation.
They use the baseline one-step mechanism only. Do not use the rate1000 stress
mechanism for this grid-validation package.

## Sync Branch On HPC

From the MFC repository on HPC:

```bash
git fetch
git checkout phasechange_local_dev
git pull --ff-only
```

## Recommended Scratch Layout

Use a dedicated scratch root, for example:

```bash
SCRATCH=/N/lustre/scratch/tripatmn/MFC/dodecane_grid_validation
mkdir -p "$SCRATCH"
```

Recommended run folders:

```text
$SCRATCH/low_nonreacting
$SCRATCH/mid_nonreacting
$SCRATCH/high_nonreacting
$SCRATCH/low_burning
$SCRATCH/mid_burning
$SCRATCH/high_burning
```

## Run Commands

Run pre_process + simulation only. Do not run post_process.

```bash
mkdir -p "$SCRATCH/low_nonreacting"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_evap_validation_low.py "$SCRATCH/low_nonreacting/case.py"
./mfc.sh run "$SCRATCH/low_nonreacting/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun

mkdir -p "$SCRATCH/mid_nonreacting"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_evap_validation_mid.py "$SCRATCH/mid_nonreacting/case.py"
./mfc.sh run "$SCRATCH/mid_nonreacting/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun

mkdir -p "$SCRATCH/high_nonreacting"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_evap_validation_high.py "$SCRATCH/high_nonreacting/case.py"
./mfc.sh run "$SCRATCH/high_nonreacting/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun

mkdir -p "$SCRATCH/low_burning"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_burning_validation_low.py "$SCRATCH/low_burning/case.py"
./mfc.sh run "$SCRATCH/low_burning/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun

mkdir -p "$SCRATCH/mid_burning"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_burning_validation_mid.py "$SCRATCH/mid_burning/case.py"
./mfc.sh run "$SCRATCH/mid_burning/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun

mkdir -p "$SCRATCH/high_burning"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_clean_burning_validation_high.py "$SCRATCH/high_burning/case.py"
./mfc.sh run "$SCRATCH/high_burning/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun
```

## Analyze Completed Runs

After all six simulations finish:

```bash
python examples/2D_dodecane_global_reduced/analyze_grid_validation.py \
  --case low,nonreacting,"$SCRATCH/low_nonreacting" \
  --case mid,nonreacting,"$SCRATCH/mid_nonreacting" \
  --case high,nonreacting,"$SCRATCH/high_nonreacting" \
  --case low,burning,"$SCRATCH/low_burning" \
  --case mid,burning,"$SCRATCH/mid_burning" \
  --case high,burning,"$SCRATCH/high_burning" \
  --out "$SCRATCH/grid_validation_summary.csv"
```

The analyzer also writes:

```text
$SCRATCH/grid_validation_summary_comparison.csv
$SCRATCH/grid_validation_summary_timeseries.csv
$SCRATCH/grid_validation_plots/
```

Primary metric: mass-equivalent `D2_norm` from liquid alpha_rho.
Secondary metric: threshold equivalent-area `D2_norm` from `alpha_liq >= 0.5`.

The plot folder includes grid overlays, burning/nonreacting overlays, liquid and
vapor budget plots, pressure diagnostics, burning species-delta plots, a primary
mass-equivalent `D2_norm` summary, and early/mid/late field snapshots from the
low/current burning case when the required raw fields are available.

To choose a different plot folder:

```bash
python examples/2D_dodecane_global_reduced/analyze_grid_validation.py \
  --case low,nonreacting,"$SCRATCH/low_nonreacting" \
  --case mid,nonreacting,"$SCRATCH/mid_nonreacting" \
  --case high,nonreacting,"$SCRATCH/high_nonreacting" \
  --case low,burning,"$SCRATCH/low_burning" \
  --case mid,burning,"$SCRATCH/mid_burning" \
  --case high,burning,"$SCRATCH/high_burning" \
  --out "$SCRATCH/grid_validation_summary.csv" \
  --plots-dir "$SCRATCH/plots_grid_validation"
```
