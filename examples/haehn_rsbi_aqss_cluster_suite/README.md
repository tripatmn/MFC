# Haehn/RSBI Fixed-AQSS Cluster Diagnostics

## Purpose

This suite tests whether the pre-chemistry N2 undershoot seen in the coarse local Haehn-style H2/O2/Xe/N2 shock-bubble fixed-AQSS case is a coarse-grid species-advection/interface artifact.

The first cluster tests should run only to `t_step_stop = 130`. Do not run the longer follow-up cases until the `t130` cases pass without `AQSS_PRE_ABORT` or `AQSS_POST_ABORT`.

## Source

Copied from:

`/home/tripatmn/mfc_work/MFC_aqss_proto_preserved/examples/haehn_rsbi_aqss_case_package_coarse`

Files copied into this suite:

- `case_base.py`
- `h2o2_xe.yaml`

The mechanism file is local to this suite; these cases do not depend on any file in `/tmp`.

## Physics Preserved

- Haehn-style compact RSBI geometry
- Mach `2.83` shock in N2
- H2/O2/Xe bubble mixture: `H2:0.29,O2:0.15,XE:0.56`
- Ambient: `N2`, `101325 Pa`, `295 K`
- Mechanism: `h2o2_xe.yaml`
- `model_eqns = 2`
- `num_fluids = 1`
- fixed AQSS: `reaction_substeps = 1000`, `adap_substeps = F`

`mpp_lim` must remain `"F"` because this validation case uses `num_fluids = 1`, and MFC validation rejects `mpp_lim` for that configuration.

## Cases

| Case | cells_per_d0 | Expected Grid | Stop |
| --- | ---: | ---: | ---: |
| `case_aqss_cpd30_t130.py` | 30 | about `270 x 192` | `t_step_stop = 130` |
| `case_aqss_cpd60_t130.py` | 60 | about `540 x 384` | `t_step_stop = 130` |
| `case_aqss_cpd120_t130.py` | 120 | about `1080 x 768` | `t_step_stop = 130` |
| `case_aqss_cpd60_full.py` | 60 | about `540 x 384` | full original `tend = 2.0e-4` |
| `case_aqss_cpd120_full.py` | 120 | about `1080 x 768` | full original `tend = 2.0e-4` |

## Diagnostics

The AQSS prototype should emit:

- `AQSS_PRE_REPAIR`
- `AQSS_PRE_ABORT`
- `AQSS_POST_ABORT`
- `AQSS_ABORT_DETAIL`
- `AQSS_DIAG` with `heat_pos_sum`, `heat_neg_sum`, `heat_abs_sum`, `sumY`, and `minY` information when enabled

Interpretation:

- `AQSS_PRE_REPAIR`: small pre-existing transport/interface drift was repaired only for the temporary chemistry vector.
- `AQSS_PRE_ABORT`: pre-existing species drift exceeded the allowed repair threshold.
- `AQSS_POST_ABORT`: AQSS or the AQSS delta writeback introduced a new invalid state.

## Recommended Order

Run `case_aqss_cpd60_t130.py` first. If it passes beyond `t_step = 125` and reaches `t_step_stop = 130`, run `case_aqss_cpd120_t130.py`.

Run `case_aqss_cpd30_t130.py` only as a coarse reference if needed.

Do not run `case_aqss_cpd60_full.py` or `case_aqss_cpd120_full.py` until the corresponding `t130` diagnostic passes.

## Environment Flags

Use the standard prototype flags:

```bash
export TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1
export TEMP_INIT_SPECIES_RHOY_RESCALE=1
export TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1
export TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1
export TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1
export TEMP_PHASE_CHANGE_EVAP_ONLY=1
export TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1
export TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1
export TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1
```

## Exact Run Commands

Run from the prototype repository root:

```bash
cd /home/tripatmn/mfc_work/MFC_aqss_proto_preserved
```

Medium t130 first:

```bash
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 TEMP_INIT_SPECIES_RHOY_RESCALE=1 TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 TEMP_PHASE_CHANGE_EVAP_ONLY=1 TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 bash ./mfc.sh run /home/tripatmn/mfc_work/haehn_rsbi_aqss_cluster_suite/case_aqss_cpd60_t130.py -t pre_process simulation -j 8 -n 32 --no-build
```

Full-resolution t130 second:

```bash
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 TEMP_INIT_SPECIES_RHOY_RESCALE=1 TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 TEMP_PHASE_CHANGE_EVAP_ONLY=1 TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 bash ./mfc.sh run /home/tripatmn/mfc_work/haehn_rsbi_aqss_cluster_suite/case_aqss_cpd120_t130.py -t pre_process simulation -j 8 -n 64 --no-build
```

Coarse reference only if needed:

```bash
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 TEMP_INIT_SPECIES_RHOY_RESCALE=1 TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 TEMP_PHASE_CHANGE_EVAP_ONLY=1 TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 bash ./mfc.sh run /home/tripatmn/mfc_work/haehn_rsbi_aqss_cluster_suite/case_aqss_cpd30_t130.py -t pre_process simulation -j 8 -n 16 --no-build
```

Longer follow-ups, only after t130 passes:

```bash
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 TEMP_INIT_SPECIES_RHOY_RESCALE=1 TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 TEMP_PHASE_CHANGE_EVAP_ONLY=1 TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 bash ./mfc.sh run /home/tripatmn/mfc_work/haehn_rsbi_aqss_cluster_suite/case_aqss_cpd60_full.py -t pre_process simulation -j 8 -n 32 --no-build
env TEMP_ALPHA_SOURCE_FLUX_SUM_FIX=1 TEMP_INIT_SPECIES_RHOY_RESCALE=1 TEMP_HLLC_MODEL3_SPECIES_FLUX_FIX=1 TEMP_HLLC_MODEL3_SPECIES_FACE_NORMALIZE=1 TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX=1 TEMP_PHASE_CHANGE_EVAP_ONLY=1 TEMP_CHEM_SPECIES_COUPLED_SOURCE_LIMIT=1 TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX=1 TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX=1 bash ./mfc.sh run /home/tripatmn/mfc_work/haehn_rsbi_aqss_cluster_suite/case_aqss_cpd120_full.py -t pre_process simulation -j 8 -n 64 --no-build
```
