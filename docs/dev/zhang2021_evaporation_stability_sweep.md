# Zhang 2021 Evaporation Stability Sweep

This sweep is a diagnostic step for the hot nonreacting Zhang-style
n-dodecane evaporation setup before any burning validation is attempted.

## Target Physics

- Fuel: n-dodecane
- Initial droplet diameter: 1.0 mm
- Domain: 4 mm x 4 mm
- Grid: 256 x 256, 64 cells per initial droplet diameter
- Ambient pressure: 100000 Pa
- Ambient gas temperature: 793.15 K, encoded through gas density
- Chemistry/reactions: off
- Phase change/relaxation: on, `relax_model = 6`
- Flow: quiescent, no shock
- Primary metric: mass-equivalent `D2_norm` from liquid `alpha_rho`

## Case Variants

All cases preserve the same physics and grid. Only timestep control changes.

| Variant | Case file | Timestep control | End condition | Output cadence |
| --- | --- | --- | --- | --- |
| `256_fixed_dt_1p0` | `examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_1p0.py` | `dt = 1.1283507029868327e-9` | `t_step_stop = 12000` | save every 2000 steps |
| `256_fixed_dt_0p5` | `examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_0p5.py` | `dt = 5.641753514934164e-10` | `t_step_stop = 12000` | save every 2000 steps |
| `256_fixed_dt_0p25` | `examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_0p25.py` | `dt = 2.820876757467082e-10` | `t_step_stop = 12000` | save every 2000 steps |
| `256_adaptive_cfl_conservative` | `examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_adaptive_cfl_conservative.py` | `cfl_adap_dt = T`, `cfl_target = 0.10` | `t_stop = 1.3540208435841992e-5 s` | save every `2.2567014059736653e-6 s` |

The fixed 1.0x case uses the same timestep as the current 256 pilot and half
the nominal 256 acoustic-scaled timestep. The 0.5x and 0.25x cases test whether
the apparent stall near step 9000 is timestep sensitive. The CFL-adaptive case
tests whether local wave-speed growth should control `dt` instead of a fixed
step count.

## Recommended HPC Run Order

Run the cases sequentially and stop once a stable, informative setting is found:

1. `256_fixed_dt_1p0`
2. `256_fixed_dt_0p5`
3. `256_adaptive_cfl_conservative`
4. `256_fixed_dt_0p25`

The adaptive case is placed before the most conservative fixed case because it
keeps the same physical end time as the 1.0x run while allowing `dt` to shrink
if the phase-change dynamics stiffen.

## HPC Commands

From the MFC repository on HPC:

```bash
RUN_ROOT="$PWD/runs/zhang2021_stability_sweep_256"
mkdir -p "$RUN_ROOT/256_fixed_dt_1p0" \
         "$RUN_ROOT/256_fixed_dt_0p5" \
         "$RUN_ROOT/256_fixed_dt_0p25" \
         "$RUN_ROOT/256_adaptive_cfl_conservative"

cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_1p0.py \
  "$RUN_ROOT/256_fixed_dt_1p0/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_0p5.py \
  "$RUN_ROOT/256_fixed_dt_0p5/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_fixed_dt_0p25.py \
  "$RUN_ROOT/256_fixed_dt_0p25/case.py"
cp examples/2D_dodecane_global_reduced/case_hpc_d2_zhang2021_evap_1mm_pilot_256_adaptive_cfl_conservative.py \
  "$RUN_ROOT/256_adaptive_cfl_conservative/case.py"
```

Foreground or interactive allocation:

```bash
./mfc.sh run "$RUN_ROOT/256_fixed_dt_1p0/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun
./mfc.sh run "$RUN_ROOT/256_fixed_dt_0p5/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun
./mfc.sh run "$RUN_ROOT/256_adaptive_cfl_conservative/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun
./mfc.sh run "$RUN_ROOT/256_fixed_dt_0p25/case.py" -t pre_process simulation --gpu acc -n 2 -j 8 --clean -b mpirun
```

Nautilus batch submission:

```bash
./mfc.sh run "$RUN_ROOT/256_fixed_dt_1p0/case.py" -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --clean
./mfc.sh run "$RUN_ROOT/256_fixed_dt_0p5/case.py" -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --clean
./mfc.sh run "$RUN_ROOT/256_adaptive_cfl_conservative/case.py" -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --clean
./mfc.sh run "$RUN_ROOT/256_fixed_dt_0p25/case.py" -t pre_process simulation -e batch -c nautilus -N 1 -n 2 -j 8 --gpu acc --clean
```

Do not run the burning Zhang case until a stable evaporation setting is
identified.
