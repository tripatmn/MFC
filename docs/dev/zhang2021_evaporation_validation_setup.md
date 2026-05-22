# Zhang 2021-Style Dodecane Evaporation Validation Setup

This note records the first matched nonreacting evaporation target for the
experimental/literature validation pivot. The setup is intentionally limited to
case definition and lightweight checks; no full MFC run has been performed.

## Target Conditions

- Fuel: n-dodecane
- Droplet initial diameter: 1.0 mm
- Ambient pressure: 1 bar, represented as 100000 Pa
- Ambient gas temperature: 793.15 K
- Ambient gas: air/N2-like inert gas using the existing air-like fluid-3 EOS
- Flow: quiescent, no shock
- Chemistry: off
- Reactions: off
- Phase change: on through the existing `relax = T`, `relax_model = 6` path

## Case File

`examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm.py`

Matching burning case:

`examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_burning_1mm.py`

The case uses the current clean dodecane liquid/vapor property framework:
liquid dodecane is fluid 1, dodecane vapor is fluid 2, and air-like ambient gas
is fluid 3. The liquid and vapor EOS constants are inherited from the existing
clean evaporation validation cases.

The burning companion keeps the same geometry, grid, timestep, stop time, and
save cadence, then enables the baseline
`examples/1D_dodecane_global_smoke/dodecane_global_1step.yaml` mechanism with
reaction heat enabled and a 0.05 heat limiter. It intentionally does not use
the rate1000 stress mechanism.

## Practical Grid And Time Setup

- 2D square domain width: 4 droplet diameters = 4.0 mm
- Grid: 512 x 512
- Resolution: 128 cells per initial droplet diameter
- Cell size: dx = dy = 7.8125e-6 m
- Time step: acoustic CFL-style value from the existing clean framework
- Stop step: 120000
- Save cadence: every 2000 steps
- Print cadence: every 1000 steps

This is a practical case definition for validation setup, not a promise that a
full run is cheap. A shorter pilot run should be used before committing compute
time to the full window.

## Expected Metrics

- Primary: mass-equivalent normalized D2 from liquid `alpha_rho`
  - `D2_norm = liquid_alpha_rho_integral(t) / liquid_alpha_rho_integral(0)`
- Secondary: 2D equivalent-circle threshold D2 from liquid volume fraction
- Report normalized `(D/D0)^2` or `D2/D2_0` for comparison to digitized data

The threshold/equivalent-circle metric is a 2D cross-section measure. It should
not be interpreted as a true 3D spherical droplet diameter without a separate
mapping assumption.

## Analyzer And HPC Plan

Focused analyzer:

`examples/2D_dodecane_global_reduced/analyze_zhang2021_validation.py`

HPC run instructions:

`examples/2D_dodecane_global_reduced/HPC_ZHANG2021_VALIDATION_RUNS.md`

Recommended run root:

`runs/zhang2021_validation/`

Expected run folders:

- `runs/zhang2021_validation/evap_1mm`
- `runs/zhang2021_validation/burning_1mm`

Analyzer outputs:

- `zhang2021_validation_summary.csv`
- `zhang2021_validation_timeseries.csv`
- `zhang2021_validation_fit_summary.txt`
- `zhang2021_validation_comparison.csv` when both evaporation and burning runs
  are provided
- PNG plots in the requested plot directory

## Assumptions And Caveats

- The target values are Zhang et al. 2021-style values from the current
  literature review; exact experimental conditions still need confirmation.
- The initial droplet temperature is not independently specified by this case;
  the setup follows the existing stiffened-gas multi-fluid initialization path.
- The ambient gas is represented by the existing air-like EOS, not detailed
  gas composition chemistry.
- No experimental uncertainty bands or digitized data are included yet.
- Absolute comparison of the fitted evaporation constant `K` awaits digitized
  Zhang data and the confirmed regression window.

## Still Needed From The Paper

- Digitized `D2(t)` or normalized `(D/D0)^2(t)` curve
- Reported or fitted evaporation constant `K`
- Regression window used for the D2-law slope
- Confirmed ambient composition, droplet initial temperature, and pressure
  convention if the paper uses atm rather than exact bar
