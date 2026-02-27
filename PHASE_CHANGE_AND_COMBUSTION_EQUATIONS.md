# Phase Change and Combustion: Equation Summary and Comparison

This document summarizes the **phase change** and **combustion** mechanisms in MFC and compares the **tracing** (volume-fraction/phase) and **VOF/species** update equations used in the 2D burning droplet (multiphase chemistry) setup.

---

## 1. Phase Change Mechanism

**Module:** `src/common/m_phase_change.fpp`  
**Entry point:** `s_infinite_relaxation_k(q_cons_vf)` (called after RK time stepping when `relax = T`).

### 1.1 What it does

- **pT-equilibrium:** For each cell, finds a common pressure \(p\) and temperature \(T\) such that the mixture internal energy is conserved.
- **pTg-equilibrium (optional, `relax_model == 6`):** When both liquid and vapor are present and below critical point, enforces **Gibbs free energy equality** between liquid and vapor and **energy conservation**, and iteratively updates the **partial masses** \(m_1\) (liquid), \(m_2\) (vapor) and pressure.
- **Volume fractions** are then updated from the new partial densities:
  \[
  \alpha_i = \frac{(\alpha_i\rho_i)}{\rho_i(p,T)},\qquad \rho_i = \frac{p + p_{\infty,i}}{(\gamma_i-1)\,c_{v,i}\,T}.
  \]

### 1.2 Key variables (per cell)

| Quantity | Conservative variable | Role in phase change |
|----------|----------------------|----------------------|
| Liquid partial mass | `q_cons_vf(lp + contxb - 1)` = \(\alpha_1\rho_1\) | Updated by pT/pTg solver |
| Vapor partial mass | `q_cons_vf(vp + contxb - 1)` = \(\alpha_2\rho_2\) | Updated by pT/pTg solver |
| Volume fractions | `q_cons_vf(advxb + i - 1)` = \(\alpha_i\) | Recomputed from \(\alpha_i\rho_i/\rho_i(p,T)\) after equilibrium |
| Total reacting mass | \(r_M = m_1 + m_2\) | **Invariant** during phase change |

### 1.3 Coupling to species (multiphase chemistry)

When `chemistry .and. chem_params%multiphase`:

- **Evaporated mass** in the relaxation step:
  \[
  \Delta m_{\mathrm{evap}} = m_{1,\mathrm{old}} - m_1.
  \]
- This mass is **added to the fuel species** conservative variable:
  \[
  (\alpha_g\rho_g Y_{\mathrm{fuel}}) \;\leftarrow\; (\alpha_g\rho_g Y_{\mathrm{fuel}}) + \Delta m_{\mathrm{evap}}.
  \]
- So **phase change** does not appear as a source in the **PDE** for species; it is applied as an **algebraic source** in the **same time step**, after the hyperbolic + diffusion + reaction RHS have been applied.

**Summary:** Phase change updates **partial densities** and **volume fractions** via pT/pTg equilibrium; it then **increments** the fuel-species conservative variable by the evaporated mass. No separate “tracing” PDE for evaporation—it’s an algebraic coupling after relaxation.

---

## 2. Combustion Mechanism

**Module:** `src/common/m_chemistry.fpp`  
**RHS contributions:** reaction flux and (optionally) diffusion flux.

### 2.1 Species conservative variables

- **Single-phase:** \(\rho Y_k\) (density × mass fraction).
- **Multiphase (Phase 2):** Species are interpreted as **gas-phase** quantities. Conservative variables are stored as **\(\alpha_g\rho_g Y_k\)** in practice (same array indices `chemxb:chemxe`), with:
  - \(\alpha_g = \sum_{i \neq \mathrm{liquid}} \alpha_i\),
  - \(\rho_g = \sum_{i \neq \mathrm{liquid}} \alpha_i\rho_i\) (gas-phase density).

### 2.2 Reaction (source) term

**Subroutine:** `s_compute_chemistry_reaction_flux`.

- **Single-phase:**  
  \[
  \frac{\partial (\rho Y_k)}{\partial t}\Big|_{\mathrm{react}} = \dot{\omega}_k M_k.
  \]
- **Multiphase:**  
  - Reaction is applied only where \(\alpha_g \ge\) `gas_phase_threshold`.  
  - Gas-phase density used for kinetics: \(\rho = \rho_g\).  
  - Source scaled by gas volume fraction:
  \[
  \frac{\partial (\alpha_g\rho_g Y_k)}{\partial t}\Big|_{\mathrm{react}} = \alpha_g\,\dot{\omega}_k M_k.
  \]

So the **combustion** source in the code is \(\omega_m = M_k\,\omega_k\) (and multiplied by \(\alpha_g\) in multiphase).

### 2.3 Diffusion term

**Subroutine:** `s_compute_chemistry_diffusion_flux`.

- Flux for species \(k\) (e.g. mixture-average):  
  \[
  \mathbf{F}_{\mathrm{diff},k} = \alpha_g\,\rho_g D_k \nabla Y_k
  \]
  (and similarly thermal diffusion scaled by \(\alpha_g\) in Phase 2).
- In the RHS this appears as \(-\nabla\cdot \mathbf{F}_{\mathrm{diff},k}\) added to the species equation.

**Summary:** Combustion updates species via **reaction source** \(\dot{\omega}_k M_k\) (and \(\alpha_g\) in multiphase) and **diffusion** \(-\nabla\cdot(\alpha_g \rho_g D_k \nabla Y_k)\). Both are contributions to the **same** species conservative equation (tracing of **species** \(Y_k\)).

---

## 3. Tracing vs VOF/Species Update — Equation Comparison

Here “**tracing**” means **which quantity is carried by the PDE** (volume fraction vs species). “**VOF/species update**” means how **volume fractions** \(\alpha_i\) and **species** \(Y_k\) are advanced.

### 3.1 Volume fraction (VOF) / phase tracing

**Conservative variables:** \(\alpha_i\) for \(i=1,\ldots,\mathrm{num\_fluids}\).

- **Flux (from Riemann):**  
  \[
  \mathbf{F}_{\alpha_i} = \mathbf{u}\,\alpha_i
  \]
  (or pressure-dependent form in some solvers; see `s_convert_primitive_to_flux_variables`: `FK_vf(..., advxb:advxe) = vel_K * alpha_K`).
- **Update (RHS):**  
  \[
  \frac{\partial \alpha_i}{\partial t} + \nabla\cdot(\mathbf{u}\,\alpha_i) = \mathrm{src},
  \]
  with possible non-conservative or surface-tension terms. So **phase/interface** is “traced” by **advecting** \(\alpha_i\).
- **After phase change:** \(\alpha_i\) are **overwritten** from the equilibrium state:
  \[
  \alpha_i = \frac{(\alpha_i\rho_i)}{\rho_i(p,T)}.
  \]
  So in cells where relaxation runs, the **VOF is not only advected** but **recomputed** from pT/pTg.

**Data flow:**  
`q_cons_vf(contxb:contxe)` (partial densities) and `q_cons_vf(advxb:advxe)` (volume fractions) → Riemann → flux → RHS for continuity and advection; then `s_infinite_relaxation_k` updates both partial densities and volume fractions in one go.

### 3.2 Species (VOF/species) update

**Conservative variables:** For multiphase, effectively \(\alpha_g\rho_g Y_k\) (index `chemxb:chemxe`).

- **Advection flux (Riemann):**  
  \[
  \mathbf{F}_{\rho Y_k} = \mathbf{u}\,(\rho Y_k)
  \]
  (code: `FK_vf(..., chemxb:chemxe) = vel_K * (rho_K * Y_K)`). In multiphase, \(\rho\) here is the **mixture** density at the face; the conserved quantity in the PDE is still the same array.
- **PDE (strong form):**
  \[
  \frac{\partial (\alpha_g\rho_g Y_k)}{\partial t} + \nabla\cdot\big(\mathbf{u}\,(\alpha_g\rho_g Y_k)\big) = \nabla\cdot\big(\alpha_g\rho_g D_k\nabla Y_k\big) + \alpha_g\,\dot{\omega}_k M_k + \dot{S}_{\mathrm{evap},k}.
  \]
  - \(\dot{S}_{\mathrm{evap},k}\): only for \(k = \mathrm{fuel}\); applied **algebraically** in the phase-change step as \(+\Delta m_{\mathrm{evap}}\), not as a continuous source in the RHS.
- **Update (RHS):**  
  - Hyperbolic: \(-\nabla\cdot(\mathbf{u}\,\rho Y_k)\) from Riemann fluxes.  
  - Diffusion: \(+\nabla\cdot(\alpha_g\rho_g D_k\nabla Y_k)\) from `s_compute_chemistry_diffusion_flux`.  
  - Reaction: \(+\alpha_g\,\dot{\omega}_k M_k\) from `s_compute_chemistry_reaction_flux`.

So **species** are “traced” by **advection + diffusion + reaction**; in multiphase, **gas-phase** density and volume fraction are used for reaction and diffusion, and evaporation is added as a discrete source in the phase-change module.

### 3.3 Side-by-side comparison

| Aspect | Tracing (VOF / phase) | VOF/Species update (species \(Y_k\)) |
|--------|----------------------|--------------------------------------|
| **Quantity** | \(\alpha_i\) (and \(\alpha_i\rho_i\)) | \(\alpha_g\rho_g Y_k\) |
| **Advection** | \(\partial_t \alpha_i + \nabla\cdot(\mathbf{u}\alpha_i)\) | \(\partial_t(\alpha_g\rho_g Y_k) + \nabla\cdot(\mathbf{u}\,\alpha_g\rho_g Y_k)\) |
| **Diffusion** | None for \(\alpha_i\) | \(\nabla\cdot(\alpha_g\rho_g D_k\nabla Y_k)\) |
| **Source** | None in PDE; \(\alpha_i\) recomputed in relaxation | Reaction \(\alpha_g\,\dot{\omega}_k M_k\); evaporation as algebraic \(+ \Delta m_{\mathrm{evap}}\) for fuel |
| **Where updated** | Riemann RHS + `s_infinite_relaxation_k` | Riemann RHS + chemistry diffusion flux + chemistry reaction flux + phase change (fuel only) |
| **After phase change** | \(\alpha_i\) set from \(\alpha_i\rho_i/\rho_i(p,T)\) | Fuel species increased by \(\Delta m_{\mathrm{evap}}\) |

---

## 4. Summary

- **Phase change:** pT/pTg equilibrium updates **partial densities** \(\alpha_i\rho_i\) and **volume fractions** \(\alpha_i\); in multiphase chemistry it **adds** evaporated mass to the **fuel species** conservative variable.
- **Combustion:** **Reaction** \(\alpha_g\,\dot{\omega}_k M_k\) and **diffusion** \(\nabla\cdot(\alpha_g\rho_g D_k\nabla Y_k)\) update the **species** equation; gas-phase density and \(\alpha_g\) are used in multiphase.
- **Tracing:** **Phases** are traced by **advecting** \(\alpha_i\) (and \(\alpha_i\rho_i\)) and then **recomputing** \(\alpha_i\) in the relaxation step. **Species** are traced by **advection + diffusion + reaction**, with evaporation added algebraically in the phase-change step. So tracing is **VOF (phase)** vs **species (mass fractions)** in the same system; both use the same velocity field and are coupled through phase change (evaporation → fuel species) and through gas-phase density/volume fraction in chemistry.
