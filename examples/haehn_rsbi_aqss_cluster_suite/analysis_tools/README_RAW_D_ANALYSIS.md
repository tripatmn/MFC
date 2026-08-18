# Raw D Field Analysis

`analyze_haehn_raw_d.py` reads MFC raw post-process output tables under `D/`.

This is different from `analyze_haehn_rsbi_log.py`:

- `analyze_haehn_rsbi_log.py` parses stdout/stderr logs and AQSS diagnostic print lines.
- `analyze_haehn_raw_d.py` reads physical fields from `D/*.dat` tables.

The raw `D/*.dat` files are ASCII tables. The analyzer uses:

- column 0 as `x`
- column 1 as `y`
- the last column as the field value

It does not assume MPI rank tiling or reshape by rank layout. Plots are reconstructed from physical coordinates.

## Haehn Variable Map

For this single-fluid reactive-gas case:

- `prim.1`: density `rho`
- `prim.2`, `prim.3`: velocity components
- `prim.4`: pressure
- `prim.5`: alpha / constant 1-like field, not pressure
- `prim.6` to `prim.16`: species mass fractions

Species order:

`H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2, XE`

Pressure is reported in atm. If pressure values are O(1-100), they are treated as atm-like and converted to Pa only for temperature reconstruction.

Temperature is reconstructed as:

```text
T = p / (rho * Rmix)
Rmix = Ru * sum_k(Yk / Wk)
```

## Full t650 Analysis Command

```bash
python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/analyze_haehn_raw_d.py \
  --run-dir examples/haehn_rsbi_cpd120_t650_nsub10_rerun24h \
  --output-dir examples/haehn_rsbi_cpd120_t650_nsub10_rerun24h/raw_d_analysis \
  --steps 50,100,150,200,250,300,350,400,450,500,550,600,650 \
  --plot-steps 300,400,500,600,650
```

## Cropped Bubble-Region Example

```bash
python3 examples/haehn_rsbi_aqss_cluster_suite/analysis_tools/analyze_haehn_raw_d.py \
  --run-dir examples/haehn_rsbi_cpd120_t650_nsub10_rerun24h \
  --output-dir examples/haehn_rsbi_cpd120_t650_nsub10_rerun24h/raw_d_analysis_crop \
  --steps 300,400,500,600,650 \
  --plot-steps 300,400,500,600,650 \
  --xlim 0.04,0.12 \
  --ylim 0.08,0.18
```

## Outputs

- `haehn_raw_D_summary_by_step.csv`
- `summary.md`
- `inventory.json`
- PNG plots for requested plot steps

Plot families:

- `T_K` full scale and robust scale
- `P_atm` full scale and robust scale
- `YH2`, `YO2`
- `log10(YOH)`, `log10(YH2O)`, `log10(YHO2)`
- linear `YOH`, `YH2O`, `YHO2`
