# C1 vs C2 comparison, 0.05-0.20 microseconds

C1 has diffusion ON with reactions and chemical heat release OFF. C2 has diffusion ON with reactions and chemical heat release ON.
Both use the same raw-output masks and definitions; save 0 is excluded because primitive pressure is unavailable.

## Key result

Both completed short regressions remain species-admissible in saved outputs, with sum(Y) and integrated gas/species mass agreeing to roundoff.
The interval is too short to support any ignition or meaningful combustion claim.

## Runtime caveat

`run_time.inf` stores `dt` and `Time` as fixed six-decimal values, so all sub-microsecond entries are printed as `0.000000`.
The comparison therefore uses `save * t_save` for physical time and records per-save timestep/dt as unavailable.

## Final selected metrics

| metric | C1 | C2 | C2-C1 | relative | class |
|---|---:|---:|---:|---:|---|
| `valid_gas_Tmax` | 1.91481863e+03 | 1.91478065e+03 | -3.79750976e-02 | -1.983e-05 | small |
| `valid_gas_Tmean` | 8.20558018e+02 | 8.20559331e+02 | 1.31319055e-03 | 1.600e-06 | small |
| `rhoY_NC12H26_integral` | 1.32767957e-08 | 1.29939783e-08 | -2.82817419e-10 | -2.130e-02 | meaningful |
| `rhoY_O2_integral` | 3.42185807e-07 | 3.42184875e-07 | -9.32297830e-13 | -2.725e-06 | roundoff-level |
| `rhoY_OH_integral` | 0.00000000e+00 | 4.33572222e-13 | 4.33572222e-13 | nan | roundoff-level |
| `rhoY_HO2_integral` | 0.00000000e+00 | 1.70519472e-12 | 1.70519472e-12 | nan | small-to-moderate |
| `rhoY_H2O2_integral` | 0.00000000e+00 | 3.82511683e-14 | 3.82511683e-14 | nan | roundoff-level |
| `rhoY_H2O_integral` | 0.00000000e+00 | 1.26193801e-12 | 1.26193801e-12 | nan | small-to-moderate |
| `rhoY_CO2_integral` | 0.00000000e+00 | 3.70033701e-14 | 3.70033701e-14 | nan | roundoff-level |
| `vaporized_fuel_mass_proxy_vapor_alpha_rho_integral` | 2.07336155e-08 | 2.06752180e-08 | -5.83974954e-11 | -2.817e-03 | small-to-moderate |
| `liquid_alpha_area` | 7.86785482e-09 | 7.86785482e-09 | 0.00000000e+00 | 0.000e+00 | roundoff-level |
| `D2_m2` | 1.00176639e-08 | 1.00176639e-08 | 0.00000000e+00 | 0.000e+00 | roundoff-level |
| `max_abs_sumY_minus_1` | 4.70734562e-14 | 7.39408534e-14 | 2.68673972e-14 | 5.708e-01 | roundoff-level |
| `gas_species_mass_relative_error` | -1.43063394e-15 | -2.14601473e-15 | -7.15380789e-16 | 5.000e-01 | roundoff-level |

## Interpretation bins

- Roundoff-level: differences at numerical noise scale.
- Small: chemistry-induced or runtime-level differences that are measurable but not physically decisive in this short window.
- Meaningful: relative difference above the comparison threshold; still not automatically a physical combustion conclusion at 0.2 microseconds.

## Limitations

- Chemistry limiter aggregate activity was not recorded in available runtime files.
- CO is reported only if the SK54 mechanism species and corresponding raw field are available.
- Total vaporized fuel mass is reported as the integrated vapor-fluid partial-density proxy; NC12H26 species mass is reported separately.
