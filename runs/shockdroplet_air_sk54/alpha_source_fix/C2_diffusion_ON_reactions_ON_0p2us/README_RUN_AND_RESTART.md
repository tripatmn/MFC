# C2 diffusion + reactions test

Purpose: test SK54 species diffusion with reactions and chemical heat release ON through 0.2 µs.

This is a fresh run (`old_ic=F`, `old_grid=F`, `n_start=0`) based on the Mach-3 evaporation-only shock/droplet framework.

Runtime gates to use:

```bash
TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1
TEMP_INIT_SPECIES_RHOY_RESCALE=1
TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1
TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1
TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1
TEMP_PHASE_CHANGE_EVAP_ONLY=1
TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1
```

Output/restart notes:

- `parallel_io=F`, `prim_vars_wrt=T`, and `cons_vars_wrt=T` preserve serial `D/` and `p_all/` saved-state output at `t_save=5.0e-8`.
- Do not rerun in this same directory after outputs exist unless you have intentionally archived/moved them; MFC pre-processing can delete existing per-rank output folders for a fresh run.
- To continue from the latest complete save `N`, create a new sibling run directory, symlink or copy this case's `p_all/`, and set `old_ic=T`, `old_grid=T`, `n_start=N`, `t_step_old=0`, `num_patches=0`, and the desired later `t_stop`.

Post-run analysis:

```bash
python3 analyze_after_run.py
```
