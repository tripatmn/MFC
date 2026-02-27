# Phase Change and Combustion: Explained Separately

This document describes **phase change** and **chemistry/combustion** in MFC as two **independent** mechanisms. Each section is self-contained: equations, variables, transport terms, source terms, and process.

---

# Part I — Phase Change (Independently)

## 1. What phase change does

Phase change in MFC is an **algebraic relaxation** that drives liquid and vapor in each cell toward **thermodynamic equilibrium** (common pressure and temperature, and optionally Gibbs free energy equality). It does **not** solve a separate PDE for evaporation; it **updates** the existing conservative variables (partial densities and volume fractions) from equilibrium, and optionally adds evaporated mass to the fuel species when multiphase chemistry is enabled.

**Module:** `src/common/m_phase_change.fpp`  
**Entry point:** `s_infinite_relaxation_k(q_cons_vf)`  
**When it runs:** After each Runge–Kutta time step, when `relax = T` (called from `src/simulation/m_start_up.fpp`).

---

## 2. Variables phase change acts on

| Variable | Index | Meaning | Role in phase change |
|----------|--------|---------|----------------------|
| Partial density (liquid) | `q_cons_vf(lp + contxb - 1)` | \(\alpha_1\rho_1\) | Updated by equilibrium solver |
| Partial density (vapor) | `q_cons_vf(vp + contxb - 1)` | \(\alpha_2\rho_2\) | Updated by equilibrium solver |
| Volume fractions | `q_cons_vf(advxb + i - 1)` | \(\alpha_i\) | **Recomputed** from \(\alpha_i\rho_i/\rho_i(p,T)\) after equilibrium |
| Total reacting mass | \(r_M = m_1 + m_2\) | Sum of liquid and vapor partial masses | **Invariant** during phase change |

Here \(lp = 1\), \(vp = 2\) (liquid and vapor phase indices). Phase change does **not** modify continuity/momentum/energy indices except indirectly (through the updated \(\alpha_i\) and \(\alpha_i\rho_i\) that the rest of the solver uses).

**Optional (multiphase chemistry only):**  
When `chemistry .and. chem_params%multiphase`, phase change also **adds** the evaporated mass to the fuel species:  
`q_cons_vf(chemxb + chem_params%fuel_species_idx - 1) += dm_evap`.  
That is the only coupling from phase change to chemistry.

---

## 3. Equations (phase change only)

Phase change does **not** add terms to the PDEs. It **replaces** certain conservative variables with values from an algebraic equilibrium.

- **Invariant:**  
  \[
  r_M = m_1 + m_2 = \mathrm{constant}.
  \]

- **Equilibrium:**  
  Common \(p\), \(T\) (and optionally Gibbs equality between liquid and vapor).  
  Densities from EOS:  
  \[
  \rho_i(p,T) = \frac{p + p_{\infty,i}}{(\gamma_i - 1)\,c_{v,i}\,T}.
  \]

- **Updates applied after equilibrium:**  
  - Partial densities: `q_cons_vf(contxb:contxe)` overwritten with the new \(m_1\), \(m_2\) (and any other phases).  
  - Volume fractions:  
    \[
    \alpha_i = \frac{(\alpha_i\rho_i)}{\rho_i(p,T)}.
    \]

So the “equation” for phase change is: **recompute** \(\alpha_i\rho_i\) and \(\alpha_i\) from pT (and optionally pTg) equilibrium; **no** extra transport or source terms in the PDEs.

---

## 4. Transport and source terms (phase change)

- **Transport:** None. Phase change does not add advection, diffusion, or any flux.  
- **Source in PDE:** None. It does not add a \(\dot{S}\) to any continuity or volume-fraction equation.  
- **What it does instead:** After the time step, it **overwrites** partial densities and volume fractions with equilibrium values. So the **effective** “source” is the algebraic correction: new \(\alpha_i\rho_i\) and \(\alpha_i\) from the relaxation solver.

---

## 5. Process (phase change, step by step)

1. **Per cell:**
   - Compute total reacting mass \(r_M = m_1 + m_2\) (from current partial densities). Correct if needed (`s_correct_partial_densities`).
   - Store \(m_1\) as \(m_{1,\mathrm{old}}\) (for evaporation coupling if chemistry is on).
2. **pT-equilibrium:**  
   Call `s_infinite_pt_relaxation_k`: find common \(p\), \(T\) conserving mixture internal energy; get updated partial densities.
3. **Optional pTg-equilibrium** (`relax_model == 6`):  
   If both liquid and vapor present and below critical, compare with subcooled liquid and overheated vapor states; if neither, call `s_infinite_ptg_relaxation_k` to enforce Gibbs equality and update \(m_1\), \(m_2\), \(p\).
4. **Update state from equilibrium:**
   - Densities: \(\rho_i = (p + p_{\infty,i})/\bigl((\gamma_i - 1)\,c_{v,i}\,T\bigr)\).
   - **Overwrite** partial densities in `q_cons_vf(contxb:contxe)`.
   - **Recompute** volume fractions: \(\alpha_i = (\alpha_i\rho_i)/\rho_i\) and **overwrite** `q_cons_vf(advxb:advxe)`.
   - If model has internal energies per phase, update those from equilibrium as well.
5. **Optional multiphase chemistry:**  
   If `chemistry .and. chem_params%multiphase`:  
   \(\Delta m_{\mathrm{evap}} = m_{1,\mathrm{old}} - m_1\); if \(\Delta m_{\mathrm{evap}} > 0\), add it to the fuel species conservative variable.

**Summary:** Phase change is a **post-step algebraic** procedure: no new PDEs, no new transport or source terms in the PDE; it only **replaces** \(\alpha_i\rho_i\) and \(\alpha_i\) (and optionally adds evaporated mass to fuel species).

---

# Part II — Chemistry / Combustion (Independently)

## 1. What chemistry does

Chemistry in MFC advances **species** (mass fractions \(Y_k\)) by **reaction** (source term) and optionally **diffusion** (transport). It does **not** handle phase change or evaporation; when multiphase chemistry is used, the **evaporated mass** added to the fuel species comes from the **phase change** module, not from the chemistry PDEs.

**Module:** `src/common/m_chemistry.fpp`  
**RHS contributions:**  
- `s_compute_chemistry_reaction_flux` — reaction source.  
- `s_compute_chemistry_diffusion_flux` — species (and energy) diffusion (when `chem_params%diffusion` is on).

Both are called during RHS evaluation in `src/simulation/m_rhs.fpp`, before the time integrator advances the solution.

---

## 2. Variables chemistry acts on

| Variable | Index | Meaning | Role in chemistry |
|----------|--------|---------|-------------------|
| Species conservative | `q_cons_vf(chemxb:chemxe)` | Single-phase: \(\rho Y_k\); multiphase: \(\alpha_g\rho_g Y_k\) | Advected by Riemann; **reaction** and **diffusion** add to their RHS |
| Primitive species | \(Y_k\) | Mass fractions | Used to compute reaction rates and diffusion fluxes |
| Gas volume fraction | \(\alpha_g = 1 - \alpha_{\mathrm{liquid}}\) | From `advxb:advxe` | Used in multiphase to scale reaction and diffusion by \(\alpha_g\) |
| Gas-phase density | \(\rho_g\) | Sum of vapor partial densities | Used in multiphase for kinetics and diffusion |

Chemistry **only** reads/writes species-related indices and the energy index (for diffusion). It does **not** modify volume fractions or partial densities; evaporation into the fuel species is done by the **phase change** module.

---

## 3. Equations (chemistry only)

**Species equation (strong form):**

\[
\frac{\partial (\rho Y_k)}{\partial t} + \nabla\cdot(\mathbf{u}\,\rho Y_k) = \nabla\cdot(\rho D_k\nabla Y_k) + \dot{\omega}_k M_k.
\]

**Multiphase formulation** (Phase 2): conserved quantity is \(\alpha_g\rho_g Y_k\), and chemistry is restricted to the gas phase:

\[
\frac{\partial (\alpha_g\rho_g Y_k)}{\partial t} + \nabla\cdot\big(\mathbf{u}\,(\alpha_g\rho_g Y_k)\big) = \nabla\cdot(\alpha_g\rho_g D_k\nabla Y_k) + \alpha_g\,\dot{\omega}_k M_k.
\]

- **Reaction:** \(\dot{\omega}_k\) = molar production rate (from chemical kinetics), \(M_k\) = molecular weight; in multiphase the source is scaled by \(\alpha_g\).
- **Diffusion:** mixture-average diffusivities \(D_k\); in multiphase the flux is scaled by \(\alpha_g\) (and \(\rho_g\)).

There is **no** \(\dot{S}_{\mathrm{evap},k}\) in the chemistry module; evaporation is handled in **phase change**.

---

## 4. Transport and source terms (chemistry)

**Transport:**

- **Advection:** \(-\nabla\cdot(\mathbf{u}\,\rho Y_k)\) (or \(\alpha_g\rho_g Y_k\) in multiphase).  
  Not computed inside the chemistry module; it comes from the **Riemann flux** and `s_compute_advection_source_term` (same as all other conservative variables).
- **Diffusion:** \(\nabla\cdot(\rho D_k\nabla Y_k)\) (or \(\alpha_g\rho_g D_k\nabla Y_k\) in multiphase).  
  **Computed in chemistry:** `s_compute_chemistry_diffusion_flux` builds the diffusion flux (and thermal diffusion for energy); the flux divergence is then applied to the RHS in `s_compute_additional_physics_rhs`.  
  Flux form: \(\mathbf{F}_{\mathrm{diff},k} = \alpha_g\rho_g D_k\nabla Y_k\) (with \(\alpha_g = 1\) in single-phase).

**Source terms:**

- **Reaction:** \(\dot{\omega}_k M_k\) (single-phase) or \(\alpha_g\,\dot{\omega}_k M_k\) (multiphase).  
  **Computed in chemistry:** `s_compute_chemistry_reaction_flux` evaluates \(\dot{\omega}_k\) from \(\rho\) (or \(\rho_g\)), \(T\), and \(Y_k\), then adds the source directly to `rhs_vf(chemxb:chemxe)`.  
  In multiphase, cells with \(\alpha_g < \mathrm{gas\_phase\_threshold}\) are skipped.

**Summary:** Chemistry adds **one transport** (diffusion) and **one source** (reaction) to the species equations. Advection is generic; evaporation is not part of chemistry.

---

## 5. Process (chemistry, step by step)

**During RHS evaluation (each RK stage):**

1. **Reaction** (`s_compute_chemistry_reaction_flux`):
   - For each cell (and in multiphase, only if \(\alpha_g \ge \mathrm{gas\_phase\_threshold}\)):
     - Set \(\rho = \rho_g\) (multiphase) or total density (single-phase).
     - Get \(Y_k\) from primitives, \(T\) from temperature field.
     - Call `get_net_production_rates(rho, T, Ys, omega)` → \(\dot{\omega}_k\).
     - \(\omega_m = M_k\,\omega_k\); in multiphase multiply by \(\alpha_g\).
     - Add \(\omega_m\) to `rhs_vf(eqn)` for each species index.
   - No transport; only a **source** term.

2. **Diffusion** (`s_compute_chemistry_diffusion_flux`), per direction:
   - For each face (and in multiphase, skip if either cell is liquid-dominated):
     - Compute mixture-average diffusivities, thermal conductivity, species enthalpies.
     - Build species diffusion flux (and thermal diffusion); in multiphase scale by \(\alpha_g\).
     - Write flux into `flux_src_vf` (species and energy).
   - Later, `s_compute_additional_physics_rhs` turns these fluxes into \(-\nabla\cdot\mathbf{F}\) and adds to RHS.

**Order in the full RHS:**  
Riemann flux divergence → advection source → **chemistry diffusion** (flux → RHS) → other physics → **chemistry reaction** (add to `rhs_vf`).

**Summary:** Chemistry is a **RHS-only** contribution: **reaction** as a direct source, **diffusion** as a flux that becomes a divergence term. No time integration inside chemistry; no phase change; no evaporation term in the species PDE.

---

# Brief coupling note

- **Phase change → chemistry:** When both are enabled (`chemistry .and. chem_params%multiphase`), phase change adds \(\Delta m_{\mathrm{evap}}\) to the fuel species after each step. That is the **only** link from phase change to species.
- **Chemistry → phase change:** None. Chemistry does not modify \(\alpha_i\) or \(\alpha_i\rho_i\).

So: **phase change** = algebraic relaxation of phases and optional fuel-species increment. **Chemistry** = species reaction and diffusion only; each is described above on its own.
