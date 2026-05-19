# Multifluid Reaction Heat Progress

## Starting Point

Vanilla MFC did not have validated multifluid reaction heat feedback for the
liquid-dodecane droplet workflow. Phase change was already multi-fluid oriented,
while chemistry used a global species block. The development path kept liquid as
a separate fluid and bridged the gas chemistry path with minimal architecture
change.

## Coupling Path

- Added and validated gas-fluid chemistry bookkeeping so chemistry can use a
  designated gas density while species remain global.
- Built phase-change/species coupling so liquid evaporation contributes to the
  fuel vapor species source.
- B1/B1a reduced scaffolds validated mass transfer and fuel-species accounting,
  but they were not yet D2-law validation cases.
- Built a clean nonreacting quiescent D2 evaporation pilot to isolate phase
  relaxation and liquid regression.
- Built a clean baseline burning pilot using the one-step dodecane mechanism,
  not the rate1000 stress mechanism.

## Formal Reaction Heat Coupling

The temporary heat path was formalized with case parameters:

```text
chem_reaction_heat_enable
chem_reaction_heat_limit_frac
chem_reaction_heat_diag
```

For multifluid chemistry with `model_eqns = 3`, reaction heat is applied to
total energy and distributed into the gas partial internal energies for the
designated chemistry gas fluids. The `q_T_sf` chemistry temperature is refreshed
during conservative-to-primitive conversion for the heat-coupled multifluid
path.

A common-scale limiter was added so species rates and reaction heat are reduced
by the same local factor when heat release would exceed a chosen fraction of
local gas internal energy over the timestep. The clean validation setting is:

```text
chem_reaction_heat_enable     = T
chem_reaction_heat_limit_frac = 0.05
chem_reaction_heat_diag       = F
```

## Validation Results

Clean nonreacting pilot:

- Final threshold `D2_norm ~= 0.8039`.

Clean baseline burning pilot:

- Final threshold `D2_norm ~= 0.7978`.
- Final mass-equivalent `D2_norm ~= 0.9556`.
- Burning increased liquid mass regression by about `5.1%`.
- Over the primary manual window, mass-equivalent K increased by about `4.57%`.
- Mass-equivalent fit quality was high, with `R2 ~= 0.994`.

Refined plots show that threshold equivalent-area D2 is useful as a geometric
secondary diagnostic, but it is stair-stepped and grid-threshold sensitive. The
mass-equivalent D2 metric from liquid alpha_rho is smoother, more conservative,
and better suited as the primary quiescent validation metric.

## Current Claim

The current branch has working quiescent phase-change + combustion coupling for
the multifluid dodecane workflow, with trend validation against D2-law behavior.
Burning increases liquid regression and gives a clean mass-equivalent D2-law
trend relative to the nonreacting evaporation pilot.

## Current Limitations

- Not yet an absolute 3D literature burning-constant validation.
- Not yet an autoignition validation.
- Not yet shock-induced burning or post-shock droplet validation.
- The one-step mechanism is the baseline D2 validation mechanism, not detailed
  ignition chemistry.
- The rate1000 mechanism remains a stress/sensitivity case, not the primary
  validation path.

## Next Step

Run HPC grid-sensitivity validation with low/mid/high nonreacting and burning
pairs. The primary comparison is mass-equivalent `D2_norm` from liquid
alpha_rho. Threshold equivalent-area `D2_norm` remains the secondary geometric
metric.
