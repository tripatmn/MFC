# Phase Change + Combustion Coupling — Changelog and Implementation Notes

This document tracks **all code and case changes** made to couple phase change (evaporation/condensation) with combustion in MFC for the 2D burning droplet case. Use it to review what was modified and where.

For the **physical coupling** (who updates what, equations, process), see [PHASE_CHANGE_AND_COMBUSTION_PROCESS.md](PHASE_CHANGE_AND_COMBUSTION_PROCESS.md).

---

## 1. Source code changes

### 1.1 `src/common/m_chemistry.fpp`

**Purpose:** Chemistry (reaction + diffusion) restricted to the gas phase and consistent with multiphase flow.

| Change | Location / subroutine | Description |
|--------|------------------------|-------------|
| **Multiphase reaction** | `s_compute_chemistry_reaction_flux` | When `chem_params%multiphase`: (1) Skip cells with `alpha_gas < chem_params%gas_phase_threshold`. (2) Use **gas-phase density** `rho_gas` (sum of partial densities of non-liquid phases) instead of total density for `get_net_production_rates`. (3) Scale reaction source by `alpha_gas`: `omega_m = omega_m * alpha_gas` before adding to `rhs_vf(chemxb:chemxe)`. |
| **Multiphase diffusion** | `s_compute_chemistry_diffusion_flux` | When `chem_params%multiphase`: Skip faces where either cell has liquid volume fraction above `(1 - gas_phase_threshold)`. Species/energy diffusion fluxes use gas-phase state; effectively diffusion is active only in gas regions. |
| **Temperature / gas-phase density** | `s_compute_q_T_sf`, reaction/diffusion routines | In multiphase, temperature and density used for kinetics are based on gas volume fraction and gas-phase partial densities; liquid-dominated cells are skipped or given default temperature. |

**Parameters (from case):** `chem_params%multiphase`, `chem_params%liquid_phase_idx`, `chem_params%fuel_species_idx`, `chem_params%gas_phase_threshold`.

---

### 1.2 `src/common/m_phase_change.fpp`

**Purpose:** After pT/pTg relaxation, transfer mass between liquid and vapor and **couple to chemistry** by updating the fuel species.

| Change | Location / subroutine | Description |
|--------|------------------------|-------------|
| **Evaporation → fuel species** | `s_infinite_relaxation_k` | When `chemistry .and. chem_params%multiphase`: (1) Before equilibrium, store liquid partial mass: `m1_old = q_cons_vf(lp + contxb - 1)%sf(j,k,l)`. (2) After equilibrium, compute `dm_evap = m1_old - q_cons_vf(lp + contxb - 1)%sf(j,k,l)`. (3) If `dm_evap > 0` (net evaporation), add `dm_evap` to the fuel species: `q_cons_vf(chemxb + chem_params%fuel_species_idx - 1)%sf(...) += dm_evap`. |
| **Condensation (design note)** | Same block | For strict mass consistency in both evaporation and condensation, the same `dm_evap` (positive or negative) can be applied to the fuel species: when `dm_evap < 0` (net condensation), subtracting it from the fuel species keeps total mass consistent. Current code may only apply when `dm_evap > 0`; extending to both signs is a possible follow-up. |

**Parameters:** `chem_params%fuel_species_idx`, `chem_params%multiphase` (from chemistry module).

---

### 1.3 `src/simulation/m_riemann_solvers.fpp`

**Purpose:** Species fluxes at faces must be consistent with “species live in the gas phase” in multiphase flows.

| Change | Location | Description |
|--------|----------|-------------|
| **Gas-weighted species flux** | HLL and HLLC branches (all three directions `x,y,z`) | When `chemistry` is enabled, after computing the standard Riemann flux for species, multiply the species flux by `alpha_gas_face`: (1) `alpha_gas_face = 1` (single-phase). (2) If `chem_params%multiphase`: `alpha_gas_face = 0.5*((1 - alpha_L(liquid_phase_idx)) + (1 - alpha_R(liquid_phase_idx)))`. (3) `flux_rs*_vf(..., i) = flux_rs*_vf(..., i) * alpha_gas_face` for `i in [chemxb, chemxe]`. |
| **Variable declaration** | HLL and HLLC solver local variable blocks | Added `real(wp) :: alpha_gas_face` in the declaration section of both the HLL and HLLC Riemann solver subroutines so the symbol has an explicit type (fixes “Symbol 'alpha_gas_face' has no IMPLICIT type” at chemistry blocks). |
| **GPU private list** | Same HLL/HLLC `GPU_PARALLEL_LOOP` directives | `alpha_gas_face` is included in the `private=` list for the parallel regions that contain the chemistry flux scaling. |

**Parameters:** `chem_params%multiphase`, `chem_params%liquid_phase_idx`.

---

## 2. Test case and run configuration

### 2.1 `examples/2D_burning_droplet/test_phase2_validation.py`

**Purpose:** Run phase-change + chemistry in 1D (stable) or 2D thin slice (for ParaView), with MUSCL and controlled output.

| Change | Description |
|--------|-------------|
| **1D vs 2D** | Added `--2d` flag. **Default:** 1D (`n=0`, line-segment patches, no y-domain/vel(2)/bc_y). With `--2d`: 2D thin slice (`n=14`, rectangular patches, `y_domain`, `vel(2)`, `bc_y` periodic). |
| **Reconstruction** | Switched from WENO to **MUSCL**: `recon_type=2`, `weno_order=0`, `weno_eps=1e-6` (for validator), `muscl_order=2`, `muscl_lim=3` (Van Albada). |
| **Domain** | 1D: `Nx=199`, `Lx=1e-3`, `Ny=0`, `Ly=0`. 2D: `Ny=14`, `Ly=0.2e-3` (WENO/MUSCL stencil requirement: `n+1 >= num_stcls_min*order`). |
| **Patches** | 1D: geometry 1 (line segment), no `y_centroid`/`length_y`/`vel(2)`. 2D: geometry 3 (rectangle), with `y_centroid`, `length_y`, `vel(2)` set. |
| **Boundary conditions** | 1D: only `bc_x%beg`/`bc_x%end` (-3 reflective). 2D: add `bc_y%beg`/`bc_y%end` (-1 periodic). |
| **Output** | `t_step_save=1` so many restarts (e.g. 20+ by 20% of run). |
| **Chemistry** | `chemistry=T`, `chem_params%multiphase=T`, `chem_params%liquid_phase_idx=1`, `chem_params%fuel_species_idx=1` (H2), `chem_params%gas_phase_threshold` (default 0.01). |
| **Phase change** | `relax=T`, `relax_model=6` (pTg), patches and fluid properties set for liquid/vapor/oxidizer. |

**Run examples:**

- 1D (default, stable):  
  `./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4 -- --steps 102`
- 2D (thin slice, may hit NaNs):  
  Same command with `--2d` appended.

---

## 3. Visualization and diagnostics

### 3.1 `examples/2D_burning_droplet/visualize_phase2_simple.py`

| Change | Description |
|--------|-------------|
| **Restart path** | Reads **parallel_io** restart layout: `restart_data/lustre_<step>/<step>_<rank>.dat` (not a single `lustre_<step>.dat` file). |
| **Case directory** | Tries, in order: `--case-dir`, script directory, current working directory, `cwd/examples/2D_burning_droplet`. Added `--case-dir` for runs that write restarts elsewhere. |
| **Grid** | 2D: `total_cells = Nx*Ny` (200×15 for m=199,n=14). 1D slice for plotting: first row of 2D data. `read_grid` uses `restart_data/x_cb.dat` when present. |
| **D/ fallback** | When using `D/` format, uses `used_case_dir` (resolved case dir) instead of a hardcoded path. |
| **Output dir** | Figures written to `used_case_dir/figures/phase2/`; `_create_plots` takes an `output_dir` argument. |

### 3.2 `examples/2D_burning_droplet/visualize_pre_crash.py`

| Change | Description |
|--------|-------------|
| **Purpose** | Diagnostic script: list restart timesteps, print per-variable min/max/NaN/Inf, and plot key fields with NaN/Inf marked. |
| **Case directory** | Same multi-candidate search and `--case-dir` as in `visualize_phase2_simple.py`. |
| **Restart read** | Reads `restart_data/lustre_<step>/<step>_0.dat`, infers `sys_size=21` and cell count from file size. |
| **Output** | Figures in `case_dir/figures/pre_crash/`. |

---

## 4. File list (quick reference)

| File | Role |
|------|------|
| `src/common/m_chemistry.fpp` | Reaction/diffusion in gas phase; gas-phase density and alpha scaling. |
| `src/common/m_phase_change.fpp` | pT/pTg relaxation; evaporation mass → fuel species. |
| `src/simulation/m_riemann_solvers.fpp` | `alpha_gas_face` scaling of species fluxes; declaration fix. |
| `examples/2D_burning_droplet/test_phase2_validation.py` | 1D/2D case, MUSCL, chemistry + phase change options. |
| `examples/2D_burning_droplet/visualize_phase2_simple.py` | Phase 2 validation plots from restart_data. |
| `examples/2D_burning_droplet/visualize_pre_crash.py` | NaN/Inf diagnostics and pre-crash plots. |
| `examples/2D_burning_droplet/PHASE_CHANGE_AND_COMBUSTION_PROCESS.md` | Coupling design and equations. |
| `examples/2D_burning_droplet/PHASE_CHANGE_COMBUSTION_CHANGELOG.md` | This changelog. |

---

## 5. Design summary (coupling)

- **Phase change → chemistry:** In `m_phase_change.fpp`, after relaxation, evaporated mass is added to the fuel species conservative variable. No other chemistry variables are modified by phase change.
- **Chemistry → phase change:** Chemistry does not change volume fractions or partial densities. It only adds reaction and diffusion to species (and diffusion to energy). Temperature/energy feedback is through the same conservative state that phase change reads (e.g. for equilibrium).
- **Riemann (inviscid) species flux:** In `m_riemann_solvers.fpp`, species fluxes are scaled by `alpha_gas_face` so that advection of species is consistent with “species only in the gas phase” at interfaces.

For full equations and process order, see [PHASE_CHANGE_AND_COMBUSTION_PROCESS.md](PHASE_CHANGE_AND_COMBUSTION_PROCESS.md).
