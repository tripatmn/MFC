# Quiescent Dodecane Validation Summary

This note summarizes the completed quiescent n-dodecane evaporation and burning
bridge cases. The primary regression metric is the mass-equivalent droplet
diameter from the liquid `alpha_rho` integral. All runs here cover an early-time
window of `1.0e-4 s = 0.10 ms`, not a full droplet lifetime.

## Results Table

| Case group | D0 | T | p0 | chemistry | K_cumulative [mm2/s] | K_fit [mm2/s] | R2 | key interpretation |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| T600 long evaporation | 0.25 mm | 600 K | 1 atm | off | 0.0911 | 0.0416 | 0.253 | Stable matched evaporation baseline; useful for coupling comparison, but weak D2-law fit over this very short window. |
| T600 long burning | 0.25 mm | 600 K | 1 atm | on | 0.0963 | 0.0468 | 0.298 | Stable coupling check. Phase change, fuel-species transport, reaction, and heat feedback are active; burning increases cumulative K by about 5.7%. |
| Frolov bridge evaporation | 0.7 mm | 700 K | 1 atm | off | 0.460 | 0.298 | 0.457 | Literature-scale evaporation baseline. Much of the early liquid regression comes from thermal/pTg evaporation before adding chemistry. |
| Frolov bridge burning | 0.7 mm | 700 K | 1 atm | on | 0.613 | 0.401 | 0.597 | Stronger validation case. Burning enhances cumulative regression by about 33% over the matched evaporation baseline and gives literature-scale K. |

## Interpretation

The T600 pair demonstrates that the coupled phase-change, species transport,
reaction, and reaction-heat feedback path is active and stable. The burning run
has the expected species-budget signs: fuel vapor accumulates, O2 decreases, and
CO2/H2O increase. Because the fitted D2-law R2 values are low over the short
window, this pair is best framed as a stable coupling check rather than a strong
dimensional D2-law validation.

The Frolov-inspired `0.7 mm`, `700 K`, `1 atm` pair is the stronger validation
case. It gives a literature-scale burning constant and a matched nonreacting
baseline, showing that burning adds a measurable `~33%` cumulative regression
enhancement over evaporation alone. The burning cumulative K is about
`0.613 mm2/s`, compared with the Frolov computed upper range of roughly
`0.50 mm2/s`, so this bridge case overpredicts that upper computed value by
about `20-25%`.

The Frolov burning species budgets also have the expected signs:

| species | initial sum | final sum | delta |
| --- | ---: | ---: | ---: |
| C12H26 | 0 | 74.97 | +74.97 |
| O2 | 7346.76 | 7344.94 | -1.82 |
| CO2 | 0 | 1.62 | +1.62 |
| H2O | 0 | 0.72 | +0.72 |

The C12H26 increase means evaporation supplies fuel vapor faster than reaction
consumes it over this early-time window. O2 decreasing with CO2/H2O increasing
is the gas-phase combustion signature.

## Recommended Figures

Use these advisor-update figures from `runs/advisor_update_plots`:

- `t600_D2_mass_mm2_comparison.png`
- `frolov_D2_mass_mm2_vs_time.png` or `frolov_D_over_D0_squared_comparison.png`
- `frolov_normalized_D2_loss_comparison.png`
- `frolov_K_mass_comparison.png`
- `fields/frolov_burning_alpha_liq_final.png` if available
- `fields/frolov_burning_vapor_alpha_rho_final.png` if available
- `fields/frolov_burning_pressure_final.png` if available
- `fields/frolov_burning_rhoY_C12H26_final.png` if available
- `fields/frolov_burning_rhoY_O2_final.png` if available
- `fields/frolov_burning_rhoY_CO2_final.png` if available
- `fields/frolov_burning_rhoY_H2O_final.png` if available

If the exact `frolov_burning_*` field names are absent, use the corresponding
copied diagnostics images under `runs/advisor_update_plots/fields/`.

## Conservative Claim

These cases support a conservative validation claim: the current quiescent
dodecane branch produces stable coupled evaporation/combustion behavior, correct
gas-species trends, and a literature-scale early-time mass-equivalent D2
regression rate for a Frolov-aligned bridge case. This should be presented as
bridge validation, not a
full Frolov reproduction, because the setup is a 2D Cartesian-equivalent
droplet, uses one-step chemistry, lacks detailed low-temperature/cool-flame
chemistry, omits soot/radiation, and does not simulate the full `2.8-4.3 mm`
Frolov droplet configurations or full droplet lifetime.
