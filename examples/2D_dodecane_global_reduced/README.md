# Reduced 2D Dodecane Global-One-Step Validation

This example is an HPC-ready refinement of the reduced 2D poly-style
shock-interface/droplet validation. It uses three fluids:

- fluid 1: liquid dodecane
- fluid 2: dodecane vapor
- fluid 3: air

The chemistry mechanism is the validation-only one-step global dodecane
mechanism in:

```text
examples/1D_dodecane_global_smoke/dodecane_global_1step.yaml
```

Species order:

```text
1 c12h26
2 o2
3 n2
4 co2
5 h2o
```

`fuel_species_id = 1`, so vaporization mass is mirrored into `c12h26`.

## What This Validates

- Phase relaxation produces positive vaporization in the reduced 2D geometry.
- Liquid dodecane mass decreases and dodecane vapor mass increases.
- Vaporization is mirrored into the global `c12h26` species.
- With reactions off, `c12h26` accumulation is an evaporation/species bookkeeping check.
- With reactions on, the artificial one-step mechanism gives sign-correct coupling:
  `c12h26` and `O2` decrease relative to reactions off, while `CO2` and `H2O`
  increase.

## What This Does Not Validate

- It is not predictive dodecane combustion kinetics.
- It is not a full shock-droplet combustion validation.
- It is not transport-controlled burning; chemistry diffusion is off.
- It is not a D2-law validation.
- It does not validate a detailed dodecane mechanism such as Reitz chemistry.

## Case Files

```text
case_hpc_reactions_off.py
case_hpc_reactions_on.py
analyze_hpc.py
```

The refined grid is `m = 240`, `n = 160`, `p = 0`. The short run uses:

```text
t_step_stop  = 50
t_step_save  = 5
t_step_print = 5
```

The timestep is the stable fixed timestep from the local 64x40 passing case.

## Recommended Run Layout

Run OFF and ON in separate directories so the `D/`, `p_all/`, logs, and timing
files do not overwrite each other.

From the repo root:

```bash
mkdir -p runs/dodecane_global_reduced_off runs/dodecane_global_reduced_on

cp examples/2D_dodecane_global_reduced/case_hpc_reactions_off.py \
  runs/dodecane_global_reduced_off/case.py

cp examples/2D_dodecane_global_reduced/case_hpc_reactions_off.py \
  runs/dodecane_global_reduced_on/case_hpc_reactions_off.py

cp examples/2D_dodecane_global_reduced/case_hpc_reactions_on.py \
  runs/dodecane_global_reduced_on/case.py
```

The case files search upward for the MFC repo root, so this copied layout still
finds the one-step mechanism under `examples/1D_dodecane_global_smoke/`.

## Local CPU Commands

Use these for a CPU check on a sufficiently capable machine:

```bash
source build/venv/bin/activate

python3 toolchain/main.py run runs/dodecane_global_reduced_off/case.py \
  -t pre_process simulation --no-gpu -n 1 -j 8 --clean \
  2>&1 | tee runs/dodecane_global_reduced_off/run_stdout.log

python3 toolchain/main.py run runs/dodecane_global_reduced_on/case.py \
  -t pre_process simulation --no-gpu -n 1 -j 8 --clean \
  2>&1 | tee runs/dodecane_global_reduced_on/run_stdout.log
```

## HPC/GPU Commands

For NVIDIA/OpenACC builds, set the compute capability for the target GPU. For a
GTX 1660-class GPU this is `75`; on HPC, use the value for the allocated device.

```bash
source build/venv/bin/activate

MFC_CUDA_CC=75 python3 toolchain/main.py build --gpu acc -j 8

python3 toolchain/main.py run runs/dodecane_global_reduced_off/case.py \
  -t pre_process simulation --gpu acc -n 1 -j 8 --clean \
  2>&1 | tee runs/dodecane_global_reduced_off/run_stdout.log

python3 toolchain/main.py run runs/dodecane_global_reduced_on/case.py \
  -t pre_process simulation --gpu acc -n 1 -j 8 --clean \
  2>&1 | tee runs/dodecane_global_reduced_on/run_stdout.log
```

If the cluster uses a scheduler, place the same commands inside the batch script
after the site-specific module loads and allocation directives.

## Expected Output Folders

Each run directory should retain:

```text
D/
p_all/
indices.dat
run_stdout.log
run_time.inf
time_data.dat
io_time_data.dat
pre_time_data.dat
pre_process.inp
simulation.inp
syscheck.inp
```

## Analyzer Usage

After both runs finish:

```bash
source build/venv/bin/activate

python3 examples/2D_dodecane_global_reduced/analyze_hpc.py \
  --off runs/dodecane_global_reduced_off \
  --on runs/dodecane_global_reduced_on \
  --out runs/dodecane_global_reduced_analysis
```

The analyzer writes:

```text
runs/dodecane_global_reduced_analysis/analyzer_summary.txt
runs/dodecane_global_reduced_analysis/analyzer_summary.json
runs/dodecane_global_reduced_analysis/mass_time_series.png
runs/dodecane_global_reduced_analysis/final_ON_minus_OFF_c12h26.png
runs/dodecane_global_reduced_analysis/final_ON_minus_OFF_o2.png
runs/dodecane_global_reduced_analysis/final_ON_minus_OFF_co2.png
runs/dodecane_global_reduced_analysis/final_ON_minus_OFF_h2o.png
runs/dodecane_global_reduced_analysis/final_ON_pressure.png
runs/dodecane_global_reduced_analysis/final_ON_alpha_rho1.png
runs/dodecane_global_reduced_analysis/final_ON_alpha_rho2.png
runs/dodecane_global_reduced_analysis/final_ON_c12h26.png
runs/dodecane_global_reduced_analysis/final_ON_co2.png
runs/dodecane_global_reduced_analysis/final_ON_h2o.png
```

PASS criteria:

- OFF: all saved fields finite, liquid mass decreases, vapor mass increases,
  and `c12h26` mass increases.
- ON: all saved fields finite, vaporization remains active, `c12h26_ON <
  c12h26_OFF`, `O2_ON < O2_OFF`, `CO2_ON > CO2_OFF`, and `H2O_ON > H2O_OFF`.
