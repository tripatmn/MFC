# Liquid → Vaporize → Mix with Ambient → Combust

This note describes how MFC supports the intended chain for a burning droplet: **liquid vaporizes, mixes with ambient gas, and combusts**.

## Intended physics

1. **Liquid vaporizes** – Phase change (pT/pTg relaxation) moves mass from liquid to vapor; evaporated mass is added to the **fuel species** conservative variable in the same cell.
2. **Mix with ambient gas** – Fuel vapor and oxidizer mix by **advection** (Riemann flux) and **diffusion** (chemistry diffusion flux). Species exist only in the **gas phase** (vapor + air); liquid cells do not carry species.
3. **Combust** – In gas cells, **reaction** source \( \alpha_g \dot{\omega}_k M_k \) and **diffusion** update species; chemistry is skipped when \( \alpha_{\mathrm{gas}} < \) `gas_phase_threshold`.

## Why it “couldn’t do it with liquids” before

- **Combustion** (reaction + diffusion) was already restricted to gas (\( \alpha_{\mathrm{gas}} \ge \) threshold) and used gas-phase density.
- **Phase change** already added evaporated mass to the fuel species.
- The missing piece was **advection at liquid–gas interfaces**: the Riemann solver used **mixture** density and mass fractions on both sides, so species flux was not restricted to the gas phase. That could:
  - Advect species through liquid (unphysical), and/or
  - Produce wrong fluxes at liquid–gas interfaces so that vapor did not mix correctly with ambient and combust.

## What was changed (Riemann solver)

In **`src/simulation/m_riemann_solvers.fpp`**, for all chemistry branches (HLL, HLLC, exact):

- When **`chem_params%multiphase`** is true, the **species flux** is multiplied by a face gas volume fraction:
  - `alpha_gas_L = 1 - alpha_L(liquid_phase_idx)`, same for R.
  - `alpha_gas_face = 0.5*(alpha_gas_L + alpha_gas_R)`.
  - `flux_species := flux_species * alpha_gas_face`.
- Effect:
  - **Liquid–liquid face:** flux ≈ 0 (no species crossing).
  - **Gas–gas face:** flux unchanged (full mixing).
  - **Liquid–gas face:** flux scaled by gas side, so species only move with the gas and vapor can mix with ambient and combust.

Together with the existing phase-change → fuel species source and gas-only chemistry, this completes the chain: **liquid vaporizes → vapor mixes with ambient (advection + diffusion) → mixture combusts**.

## Enabling the full chain in a case

Use:

- **Phase change:** `relax = T`, `relax_model = 6` (pTg) if you want two-phase equilibrium.
- **Chemistry:** `chemistry = T`, `chem_params%multiphase = T`.
- **Indices:** `chem_params%liquid_phase_idx`, `chem_params%fuel_species_idx`, `chem_params%gas_phase_threshold` set for your fluids and mechanism.
- **Initial conditions:** Liquid patch with negligible gas species; ambient patch with oxidizer (and optional diluent). Evaporation will add fuel species to the gas.

Then the sequence is: liquid evaporates (phase change adds fuel species) → species advection (gas-weighted flux) and diffusion mix fuel with ambient → reaction in gas cells gives combustion.
