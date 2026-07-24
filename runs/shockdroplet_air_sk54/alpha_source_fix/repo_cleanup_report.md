# Alpha-source / species-consistency solver stack cleanup report

Repository: `/home/tripatmn/mfc_work/MFC_alpha_diag`  
Branch: `alpha-source-flux-sum-test`  
Base HEAD: `6e199cb2 Add diagnostic-only alpha sum origin tracing`

## Inventory and classification

### Confirmed solver corrections

- `src/simulation/m_rhs.fpp`
  - `TEMP_ALPHA_SOURCE_FLUX_SUM_FIX`: model-3 alpha nonconservative source uses the sum of alpha numerical fluxes for volume-fraction equations only.
  - `TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT`: common conservative chemistry-source scaling for species and coupled heat release.
- `src/simulation/m_riemann_solvers.fpp`
  - `TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX`: model-3 HLLC now assembles chemistry species convective fluxes using the configured chemistry-gas carrier flux.
  - `TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE`: common face-composition normalization for model-3 HLLC species fluxes.
- `src/pre_process/m_start_up.fpp`
  - `TEMP_INIT_SPECIES_RHOY_RESCALE`: initialization-only rescale of chemistry rhoY after initial relaxation to match relaxed chemistry gas mass while preserving composition ratios.
- `src/common/m_phase_change.fpp` and `src/simulation/m_time_steppers.fpp`
  - `TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX`: direct coupling of committed phase-change vapor mass increment to the fuel species.
- `src/common/m_chemistry.fpp`
  - `TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX`: replaces the old model-3 diffusion density basis with configured chemistry-gas density.
  - `TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX`: model-3 diffusion evaluates Cantera transport on intrinsic gas state and alpha-weights active face fluxes.

### Validation-only model choice

- `src/common/m_phase_change.fpp`
  - `TEMP_PHASE_CHANGE_EVAP_ONLY`: validation-only one-way evaporation mode. Negative phase-change proposals are rejected before committing the relaxed mass/thermodynamic state.
  - The diffusion activation thresholds under `TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX` are also provisional validation choices: `alpha_g >= 1.0e-4` and `alpha_liq <= 0.99` on both reconstructed face sides.

### Essential runtime guards / summaries retained

- HLLC model-3 face-composition guard and first-failure report.
- Phase-change fuel coupling summaries and insufficient-fuel abort guard.
- Evaporation-only rejected-condensation summary.
- Chemistry coupled-source limiter summary/guards.
- Model-3 diffusion intrinsic-alpha activation summaries and invalid-property abort guards.
- Existing `TEMP_DT_COLLAPSE_STATE_DEBUG` diagnostics.

### Temporary diagnostic instrumentation removed or deactivated

- Hard-coded RK-stage cell dumps.
- Rank/cell/face-specific HLLC reconstruction traces.
- Alpha-origin and alpha-RHS decomposition traces.
- Species-origin and species-transport face dumps.
- Zhang evaporation hang traces in the modified simulation path.
- Repeated detailed chemistry-diffusion face-property prints tied to specific coordinates.
- Phase-relaxation internal substep snapshots tied to specific target cells.

### Analysis / test infrastructure

- `examples/2D_dodecane_global_reduced/analyze_shockdroplet_air_sk54_species_bounds.py`
  - Raw-output species admissibility analyzer.
- `runs/shockdroplet_air_sk54/alpha_source_fix/analyze_diffusion_case.py`
  - Lightweight diffusion-run analyzer; now fails clearly when `D/`, saves, pressure, or species fields are missing.
- `runs/shockdroplet_air_sk54/alpha_source_fix/C{1,2,3}_*/analyze_after_run.py`
  - Wrappers now pass each case directory explicitly as `--run-dir` and write to that case's `analysis_diffusion_lightweight/`.
- Run/case/movie/comparison/analysis files under `runs/shockdroplet_air_sk54/alpha_source_fix/`
  - Analysis and validation infrastructure or generated output.
- `runs/shockdroplet_air_sk54/case_backups_20260715/`
  - Stale/unrelated backup artifact; preserved untouched.

## Code removed / simplified

- Removed the time-stepper and startup calls to temporary Zhang, alpha-origin, species-origin, phase-change target, and RK-boundary traces.
- Removed dormant RHS helper bodies and call sites for alpha-origin, alpha-RHS decomposition, species-origin, species-transport face diagnostics, RK-boundary dumps, and HLLC face traces.
- Removed the HLLC model-3 coordinate-targeted face trace gate while preserving face-normalization validity guards.
- Removed detailed per-face chemistry-diffusion tracing while retaining compact summary/abort guards.
- Removed phase-relaxation internal target snapshots while retaining budget/fuel-coupling/evaporation-only summaries.

## Retained environment gates

Confirmed correction gates:

- `TEMP_ALPHA_SOURCE_FLUX_SUM_FIX`
- `TEMP_INIT_SPECIES_RHOY_RESCALE`
- `TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX`
- `TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE`
- `TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX`
- `TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT`
- `TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX`
- `TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX`

Validation-only gates:

- `TEMP_PHASE_CHANGE_EVAP_ONLY`
- `TEMP_PHASE_CHANGE_BUDGET_DEBUG`

Existing diagnostic:

- `TEMP_DT_COLLAPSE_STATE_DEBUG`

## Known limitations

- All correction gates remain opt-in; default solver behavior is intended to remain unchanged.
- `TEMP_PHASE_CHANGE_EVAP_ONLY` is a validation-only one-way evaporation configuration, not a bidirectional phase-change model.
- Diffusion face deactivation thresholds are provisional and should be revisited before production use.
- Chemistry limiter severity is not yet recorded with full aggregate heat-release metrics.
- Generated run outputs remain untracked and were not deleted or moved.

## Recommended commit breakdown

1. Model-3 alpha source/flux consistency.
2. Initialization species-rhoY rescale after initial relaxation.
3. Model-3 HLLC chemistry species flux plus common face normalization.
4. Phase-change fuel mass coupling and validation-only evaporation-only mode.
5. Coupled conservative chemistry source limiter.
6. Model-3 chemistry-diffusion gas-density and intrinsic-alpha handling.
7. Species-bounds and diffusion lightweight analysis infrastructure.
8. Cleanup report and run-case documentation.

## Exact short C1/C2 regression commands

C1, diffusion ON / reactions OFF:

```bash
cd /home/tripatmn/mfc_work/MFC_alpha_diag
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 \
    TEMP_INIT_SPECIES_RHOY_RESCALE=1 \
    TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 \
    TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 \
    TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 \
    TEMP_PHASE_CHANGE_EVAP_ONLY=1 \
    TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 \
    TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 \
    bash ./mfc.sh run runs/shockdroplet_air_sk54/alpha_source_fix/C1_diffusion_ON_reactions_OFF_0p2us/case.py -t pre_process simulation -j 8 -n 4
python3 runs/shockdroplet_air_sk54/alpha_source_fix/C1_diffusion_ON_reactions_OFF_0p2us/analyze_after_run.py
```

C2, diffusion ON / reactions ON:

```bash
cd /home/tripatmn/mfc_work/MFC_alpha_diag
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 \
    TEMP_INIT_SPECIES_RHOY_RESCALE=1 \
    TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 \
    TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 \
    TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 \
    TEMP_PHASE_CHANGE_EVAP_ONLY=1 \
    TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 \
    TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 \
    TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 \
    bash ./mfc.sh run runs/shockdroplet_air_sk54/alpha_source_fix/C2_diffusion_ON_reactions_ON_0p2us/case.py -t pre_process simulation -j 8 -n 4
python3 runs/shockdroplet_air_sk54/alpha_source_fix/C2_diffusion_ON_reactions_ON_0p2us/analyze_after_run.py
```
