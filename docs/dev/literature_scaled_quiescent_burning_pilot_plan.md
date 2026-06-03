# Literature-scaled quiescent dodecane burning pilot plan

## Purpose

The archived `80 um` quiescent burning case remains a coupling verification
case. It demonstrates stable vaporization-to-fuel-species plumbing, chemistry,
reaction heat feedback, and D2-law-like regression, but its dimensional burning
constant is not literature-scaled. The primary fitted value,
`K_mass ~= 965 mm2/s`, is not a unit error; the nonreacting case is already
`K_mass ~= 923 mm2/s`, so the fast regression is dominated by the aggressive
small-droplet/hot-evaporation setup.

A literature-scaled validation case is needed before comparing against isolated
dodecane burning data, where expected `K` is `O(0.1-1) mm2/s`. A full droplet
lifetime burn is likely impractical with the current explicit workflow. The
near-term target should be early-time D2-law slope and order-of-magnitude
validation, using mass-equivalent D2 from liquid `alpha_rho` as the primary
metric and threshold equivalent-area D2 as a secondary geometric metric.

## Assumptions for estimates

- Target comparison value: `K = 0.5 mm2/s = 5.0e-7 m2/s`.
- Quiescent n-dodecane droplet, no shock.
- Baseline one-step dodecane mechanism, not rate1000.
- Reaction heat enabled with `chem_reaction_heat_limit_frac = 0.05`.
- Adaptive CFL with a conservative target of about `0.10`.
- Time step estimate uses the stiffened-liquid acoustic scale from the current
  clean pilot:
  `dt ~= CFL * dx / c_liq`, with `c_liq ~= 1.7e3 m/s`.
- Domain estimate uses an `8D x 8D` square unless otherwise noted.
- Expected normalized D2 regression is `Delta(D2)/D0^2 ~= K t / D0^2`.

These are planning estimates only. The actual adaptive time step may become
smaller if pressure/phase-change stiffness increases.

## Candidate pilots

| Candidate | D0 | Domain | Grid at 32 cells/D | Grid at 64 cells/D | dx at 32/D | dx at 64/D | CFL target | dt estimate at 32/D | Suggested early window | Steps at 32/D | Expected D2 regression at K=0.5 mm2/s | Measurable? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A. Conservative literature-scaled | `0.5 mm` | `4 mm x 4 mm` | `256 x 256` | `512 x 512` | `15.625 um` | `7.8125 um` | `0.10` | `~9.1e-10 s` | `2 ms` | `~2.2e6` | `~0.4%` | Marginal primary; threshold likely noisy |
| B. Literature-like coarse | `1.0 mm` | `8 mm x 8 mm` | `256 x 256` | `512 x 512` | `31.25 um` | `15.625 um` | `0.10` | `~1.8e-9 s` | `5 ms` | `~2.8e6` | `~0.25%` | Weak; useful mostly for stability/scaling |
| C. Smaller bridge | `0.25 mm` | `2 mm x 2 mm` | `256 x 256` | `512 x 512` | `7.8125 um` | `3.90625 um` | `0.10` | `~4.6e-10 s` | `1 ms` | `~2.2e6` | `~0.8%` | Best near-term signal per cost |

Longer windows improve the measurable regression but quickly become expensive:

| Candidate | Window for `~1%` D2 regression | Steps at 32/D | Window for `~2%` D2 regression | Steps at 32/D |
| --- | ---: | ---: | ---: | ---: |
| `0.25 mm` | `1.25 ms` | `~2.7e6` | `2.5 ms` | `~5.4e6` |
| `0.5 mm` | `5 ms` | `~5.5e6` | `10 ms` | `~1.1e7` |
| `1.0 mm` | `20 ms` | `~1.1e7` | `40 ms` | `~2.2e7` |

For a full lifetime estimate, `tau ~= D0^2 / K`:

| D0 | Lifetime at `K = 0.5 mm2/s` |
| ---: | ---: |
| `0.25 mm` | `0.125 s` |
| `0.5 mm` | `0.5 s` |
| `1.0 mm` | `2.0 s` |

At explicit acoustic-limited time steps, these lifetimes are not practical
validation runs.

## Recommended first pilot

Start with Candidate C, the `0.25 mm` bridge case at `32 cells/D`.

Rationale:

- It is still much closer to literature scale than the archived `80 um`
  coupling-verification case.
- For the same `8D` domain and `32 cells/D`, all candidates use `256 x 256`
  cells, but the smaller droplet gives more fractional D2 regression per
  wall-clock cost.
- A `1 ms` window gives an expected `~0.8%` D2 change for `K = 0.5 mm2/s`,
  which should be measurable with the smooth mass-equivalent metric if the
  phase-change path is stable and conservative.
- Threshold D2 will probably remain stair-stepped at `32 cells/D`; it should be
  reported as secondary only.

Recommended first-pilot settings:

- `D0 = 0.25 mm`, `r_drop = 0.125 mm`.
- Domain: `8D x 8D = 2.0 mm x 2.0 mm`.
- Grid: `256 x 256` for `32 cells/D`.
- `dx = dy = 7.8125e-6 m`.
- Adaptive CFL: enabled, target `0.10`.
- Initial `dt`: about `4.5e-10 s`.
- Physical stop time: start with `5.0e-4 s` for a stability pilot, then extend
  to `1.0e-3 s` if stable.
- Expected steps:
  - `5.0e-4 s`: `~1.1e6` steps.
  - `1.0e-3 s`: `~2.2e6` steps.
- Save cadence:
  - Prefer physical output spacing of `1.0e-5 s` to `2.0e-5 s`.
  - This gives roughly `50-100` saved samples over the `1 ms` pilot.
  - With fixed-step fallback, this is roughly every `2e4-4e4` steps.
- Chemistry:
  - baseline `dodecane_global_1step.yaml`;
  - `chemistry = T`;
  - `chem_params%reactions = T`;
  - `chem_reaction_heat_enable = T`;
  - `chem_reaction_heat_limit_frac = 0.05`;
  - `chem_reaction_heat_diag = F`.
- Primary metric: mass-equivalent D2 from liquid `alpha_rho`.
- Secondary metric: threshold equivalent-area D2 from `alpha_liq >= 0.5`.

Selected case paths:

- Nonreacting evaporation:
  `examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_evap_025mm_pilot.py`
- Burning:
  `examples/2D_dodecane_global_reduced/case_hpc_d2_quiescent_burning_025mm_pilot.py`
- Analyzer:
  `examples/2D_dodecane_global_reduced/analyze_quiescent_025mm_pilot.py`
- HPC instructions:
  `examples/2D_dodecane_global_reduced/HPC_QUIESCENT_025MM_PILOT_RUNS.md`

The selected first pilot uses `t_stop = 5.0e-4 s` and `t_save = 1.0e-5 s`,
so it should write about 50 saved samples plus the initial state. At
`K = 0.5 mm2/s`, the expected D2 regression over this first window is only
about `0.4%`. The mass-equivalent D2 metric is therefore the primary metric.
At `32 cells/D`, the threshold D2 curve may be too stair-stepped for a reliable
small-slope fit and should be treated as secondary.

## Escalation path

1. Run Candidate C at `32 cells/D` for `0.5 ms`.
   - Goal: finite run, conservative liquid/vapor budget, plausible species
     signs, no pressure blowup.
   - Expected D2 regression at `K = 0.5 mm2/s`: `~0.4%`.
2. Extend Candidate C to `1.0 ms` if stable.
   - Goal: first measurable mass-equivalent early-time slope.
   - Expected D2 regression: `~0.8%`.
3. If Candidate C is stable and the primary metric is smooth, repeat at
   `64 cells/D` only for the shorter `0.5 ms` window.
   - This is `512 x 512` and likely millions of steps, so it should be treated
     as an HPC-only grid-sensitivity check.
4. Move to Candidate A, `0.5 mm` at `32 cells/D`, only after Candidate C shows
   stable behavior and a measurable early-time slope.
5. Treat Candidate B, `1.0 mm`, as a coarse literature-like stability/scaling
   case. It is not expected to show a strong D2 regression over short windows.

## Claim boundary

The archived `80 um` baseline validates coupling behavior. The literature-scaled
pilot validates early-time order-of-magnitude D2-law behavior only. Full lifetime
burning and direct comparison to a 3D isolated-droplet experimental K remain
outside the practical near-term runtime envelope.
