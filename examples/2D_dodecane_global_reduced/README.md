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
- The multi-rank GPU path works after broadcasting the evaporation/species
  coupling controls to all ranks.
- The accelerated validation pair amplifies the ON/OFF chemistry signal while
  preserving the same reduced-flow setup.
- Raw `D/` output can be analyzed directly with `analyze_hpc.py`; `post_process`
  is not required for this Branch-A validation.

## What This Does Not Validate

- It is not predictive dodecane combustion kinetics.
- It is not a full shock-droplet combustion validation.
- It is not transport-controlled burning; chemistry diffusion is off.
- It is not a D2-law validation.
- It does not validate a detailed dodecane mechanism such as Reitz chemistry.
- The accelerated global one-step rate is validation-only.
- It does not establish a physical burning-rate constant.
- It is not a grid-convergence study.

## Case Files

```text
case_hpc_sanity_64x40_off.py
case_hpc_sanity_120x80_off.py
case_hpc_reactions_off.py
case_hpc_reactions_on.py
case_hpc_reactions_off_stronger.py
case_hpc_reactions_on_stronger.py
dodecane_global_1step_rate1000.yaml
analyze_hpc.py
```

The sanity and refined OFF grids are:

```text
case_hpc_sanity_64x40_off.py   m = 64,  n = 40,  t_step_stop = 5
case_hpc_sanity_120x80_off.py  m = 120, n = 80,  t_step_stop = 10
case_hpc_reactions_off.py      m = 240, n = 160, t_step_stop = 50
```

The timestep is the stable fixed timestep from the local 64x40 passing case.

## Branch-A Closeout

Branch A is a software/coupling validation, not a physical burning-droplet
study. The confirmed path is:

1. Phase-change relaxation produces `m_dot_evap`.
2. Positive vaporization is added to the global fuel species selected by
   `fuel_species_id = 1`, namely `c12h26`.
3. With reactions enabled, the validation mechanism consumes `c12h26` and `O2`.
4. With reactions enabled, the validation mechanism produces `CO2` and `H2O`.
5. The same path runs on multi-rank GPU after broadcasting the new
   evaporation/species coupling parameters.

The baseline pair uses:

```text
case_hpc_reactions_off.py
case_hpc_reactions_on.py
examples/1D_dodecane_global_smoke/dodecane_global_1step.yaml
```

The stronger Branch-A pair uses:

```text
case_hpc_reactions_off_stronger.py
case_hpc_reactions_on_stronger.py
dodecane_global_1step_rate1000.yaml
```

The stronger mechanism is identical in species and stoichiometry to the
baseline global one-step mechanism, but its Arrhenius prefactor is increased
1000x. This is only to make the ON/OFF signal easier to measure in a cheap
software validation. It should not be interpreted as physical dodecane
kinetics.

Known limitation: `post_process` has failed with exit 139 in this workflow, but
the raw `D/` analyzer is sufficient for the current Branch-A checks.

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

Recommended HPC validation order:

1. Run `case_hpc_sanity_64x40_off.py`.
2. If it passes, run `case_hpc_sanity_120x80_off.py`.
3. Only then retry `case_hpc_reactions_off.py` at 240x160.
4. Only after OFF passes, run `case_hpc_reactions_on.py`.

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

## Branch-B B1a Closeout

The reduced B1 scaffold is accepted only as a nonreacting phase-change
mass-transfer budget validation. It should not be called a D2-law validation.

Observed B1-family results:

- B1 and B1_v2 ran with reactions off, phase relaxation active, and finite raw
  fields. The thresholded droplet area and threshold `D^2` did not change.
- B1_v2 produced nearly one-for-one liquid-to-vapor mass transfer:
  `cons.1` liquid alpha_rho changed by `-7.2678e-09`, while `cons.2` vapor
  alpha_rho changed by `+7.3026e-09`.
- B1_v4 reduced the initial droplet radius on the same grid, but all
  `alpha_liq` threshold areas at 0.1, 0.5, and 0.9 remained exactly constant,
  and threshold `D^2` remained exactly constant.
- B1_v4 also produced nearly one-for-one liquid-to-vapor mass transfer:
  `cons.1` liquid alpha_rho changed by `-2.4310e-09`, while `cons.2` vapor
  alpha_rho changed by `+2.4567e-09`.
- In B1_v4, liquid alpha_rho decreased while liquid alpha slightly increased,
  so this reduced shock/dodecane scaffold does not behave like a clean
  geometrically receding droplet.

Conclusion: B1a passes as a mass-transfer budget check because liquid alpha_rho
decreases and vapor alpha_rho increases nearly one-for-one. Geometric D2-law
validation was not achieved. Do not continue tuning this reduced scaffold for a
D2-law signal.

## Branch-B Plan

Branch B should move from sign-correct software coupling toward a physical
burning-droplet validation. The target observable is the droplet-area law:

```text
D(t)^2 = D(0)^2 - K t
```

where `K` is the burning-rate constant estimated from a linear regression over
the quasi-steady burning interval.

Required case features:

- A single isolated liquid dodecane droplet in an oxidizing gas environment.
- A controlled ambient pressure, temperature, and oxidizer composition.
- Vaporization active, fuel species coupling active, and reactions active.
- Output of liquid volume fraction or liquid mass fields at enough time samples
  to reconstruct droplet size.
- Output of fuel, oxidizer, product species and temperature or pressure fields.
- Diffusion/transport likely enabled before claiming physical burning behavior;
  Branch-A chemistry diffusion is off and does not validate transport-limited
  burning.

Recommended staged path:

### Stage B0: Observable and Raw Analyzer

Define the raw-output analyzer before adding new cases. It should read `D/`
fields directly, compute finite-field status, integrate liquid mass/area, infer
an effective droplet diameter, and fit `D^2(t)` over a selectable time window.
For a 2D cross-section, a practical diameter estimate is:

```text
A_liq(t) = integral H(alpha_liq - alpha_cutoff) dA
D_eff(t) = 2 sqrt(A_liq(t) / pi)
```

An alternate smoother estimate can use `integral alpha_liq dA` instead of a
thresholded area, but the cutoff and method must be reported.

The analyzer should write:

- finite-field status and bad file list
- `A_liq(t)`, `D_eff(t)`, and `D_eff(t)^2`
- selected regression window
- fitted slope and `K = -d(D^2)/dt`
- residuals or `R^2`
- species and temperature sanity summaries

### Stage B1: Nonreacting/Evaporating Droplet Sanity

Run a minimal nonreacting case with vaporization and fuel-species coupling on.
Check that liquid decreases, vapor/fuel species increase, fields remain finite,
and the diameter estimator is smooth enough to regress later.

Clean evaporation pilot status:

- `case_hpc_d2_clean_evap_pilot_v2.py` is the first clean nonreacting
  single-droplet D2 evaporation pilot.
- Setup: no shock, chemistry disabled, `model_eqns = 3`, `relax = T`,
  `relax_model = 6`, `relax_ic = F`, 128x128 grid, GPU run with `-n 2`.
- Result: simulation passed with finite fields in `69/69` saved samples.
  Threshold `D^2` decreased from `6.4324113826307024e-09` to
  `5.1710006242829025e-09`, and threshold liquid cells decreased from `821` to
  `660`.
- Liquid alpha_rho decreased while vapor alpha_rho increased. The raw D2 fit
  gave `K = 2.8369259394735546e-03` and `R^2 = 0.7980995461141833`.
- Interpretation: this is the first stable Branch-B baseline with actual
  alpha-boundary recession.
- Limitation: this is still a pilot only. It is not grid-converged, not a
  reactive or burning validation, and the fit quality still needs window
  assessment.

### Stage B2: Reacting Single-Droplet Case

Turn reactions on with a global one-step mechanism first. This keeps the
chemistry cheap and controllable while validating the full vaporization to fuel
species to heat-release path. Do not start with a detailed mechanism until the
diameter metric and raw analyzer are stable.

Clean burning pilot status:

- `case_hpc_d2_clean_burning_pilot.py` is the first clean reacting
  single-droplet pilot. It keeps the clean evaporation pilot geometry, grid,
  no-shock setup, phase-change setup, raw outputs, and GPU `-n 2` path, while
  enabling the baseline global one-step dodecane mechanism.
- The baseline burning pilot passed with finite fields and sign-correct weak
  chemistry. `CO2` and `H2O` increased, `O2` decreased, and the `D^2`
  regression remained nearly unchanged relative to the nonreacting
  `case_hpc_d2_clean_evap_pilot_v2.py` baseline.
- `case_hpc_d2_clean_burning_pilot_rate1000.py` uses the same case setup but
  switches to `dodecane_global_1step_rate1000.yaml`. After forcing a real
  thermochemistry rebuild, the rate1000 pilot produced about `2.5x` stronger
  `CO2`, `H2O`, and `O2`-consumption signals than the baseline burning pilot.
- The clean nonreacting pilot gave `K = 2.8369259394735546e-03` and
  `R^2 = 0.7980995461141833` over the `D2_norm = 0.90..0.80` fit window. The
  baseline and rebuilt rate1000 burning pilots both gave
  `K = 2.8053606995350347e-03` and `R^2 = 0.7920795337031122` in the same
  window, so stronger chemistry did not materially change the measured `D^2`
  regression.
- Spatial diagnostics showed that products form near the droplet, not simply
  far away from the interface. Pressure and temperature-proxy maxima increased
  locally in the rebuilt rate1000 output, but global and near-droplet mean
  temperature-proxy values barely changed.
- Interpretation: these pilots are chemistry-active, but not
  heat-feedback-dominated. The current `D^2` signal appears dominated by phase
  relaxation/evaporation rather than reaction heat feedback, consistent with
  reacting-droplet studies where products and flames can form while Stefan or
  convective cooling and near-interface transport keep droplet heating weakly
  dependent on reactivity.
- Caution: direct temperature was not output in these runs. The thermal
  comparison used pressure and a gas-temperature proxy, so it should be treated
  as diagnostic rather than definitive.
- Next step: add direct temperature, internal-energy, and reaction-source
  diagnostics. After that, consider either a 1D liquid-gas interface test or a
  shock/flow case closer to the reacting-droplet literature before claiming a
  physical burning-rate validation.

#### Finalized Multifluid Reaction Heat Coupling

The multifluid reaction-energy coupling is now controlled by formal case
parameters:

```text
chem_reaction_heat_enable     = F
chem_reaction_heat_limit_frac = 0.0
chem_reaction_heat_diag       = F
```

These defaults preserve existing behavior. For the clean validation pilots, use:

```text
chem_reaction_heat_enable     = T
chem_reaction_heat_limit_frac = 0.05
chem_reaction_heat_diag       = F
```

The baseline one-step dodecane mechanism is the primary clean validation path.
The `rate1000` mechanism is a stress/sensitivity case only; it is useful for
amplifying stiffness and source-limiter behavior, but it should not be treated
as the primary validation case.

Validation summary for the clean quiescent pilots:

- nonreacting final threshold `D2_norm` was about `0.8039`
- baseline heat plus limiter final threshold `D2_norm` was about `0.7978`
- rate1000 heat plus limiter final threshold `D2_norm` was about `0.8027`
- baseline species deltas were `O2 ~= -1.224e2`, `CO2 ~= +1.092e2`, and
  `H2O ~= +4.842e1`
- rate1000 species deltas were `O2 ~= -3.538e2`, `CO2 ~= +3.157e2`, and
  `H2O ~= +1.400e2`

For heat-coupled runs, use the mass-equivalent `D2_norm` from liquid
`alpha_rho` as the primary droplet-loss metric. The threshold `D2` from
`alpha_liq >= 0.5` remains useful as a secondary geometric metric, but it does
not track liquid mass loss as cleanly once reaction heat changes the interface.

Current limitation: this is a clean quiescent single-droplet pilot, not yet a
shock-induced burning-droplet validation. The next direction is to use this
formalized heat-coupled path as the baseline for cleaner quiescent validation,
then extend toward shock and post-shock burning-droplet cases.

### Stage B3: Burning-Rate Comparison

Compute `D^2(t)` and fit a burning-rate constant over the quasi-steady interval.
Compare against an expected order of magnitude or a reference setup only after
the case shows stable finite fields, monotonic liquid loss, and sensible
temperature/species trends.

### Stage B4: Grid Refinement

Refine the grid only after the raw analyzer, droplet-size metric, and regression
window are stable on the minimal physical case. The refinement study should
track `K`, finite-field status, and regression quality, not just final mass.

### Stage B5: Mechanism and Transport Upgrade

After the global one-step case produces a stable physical metric, upgrade toward
more realistic chemistry and transport. Candidate steps are:

- enable species/thermal diffusion needed for transport-controlled burning
- replace the validation one-step mechanism with a reduced or full Cantera
  dodecane mechanism
- compare sensitivity of `K` to mechanism and transport choices

The first Branch-B case should be the smallest single-droplet configuration that
can produce a stable `D^2(t)` signal. It should precede shock-droplet burning and
any grid-convergence campaign.

## Branch-B Next Step

Design a dedicated single-droplet evaporation/D2 validation case separate from
the reduced shock/dodecane scaffold. The target observable is a thresholded or
connected-component `D^2(t)` that actually decreases over saved raw outputs.

Pass criteria for the clean case:

- all saved fields are finite
- liquid mass decreases monotonically or near-monotonically
- thresholded or connected-component `D^2(t)` decreases measurably
- at least 4 fit points are available in the selected regression window
- fitted `K = -d(D^2)/dt` is positive

Do not refine the grid until this clean case has a measurable `D^2(t)` signal.
