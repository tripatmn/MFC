# Testing Phase 2 Implementation

## Quick Test

To test the Phase 2 implementation, follow these steps:

### 1. Build MFC

```bash
cd g:\Projects\MFC
./mfc.sh build -t pre_process simulation -j $(nproc)
```

Or on Windows PowerShell:
```powershell
cd g:\Projects\MFC
./mfc.sh build -t pre_process simulation -j 4
```

### 2. Run Phase 2 Validation Test

The test case `test_phase2_validation.py` validates Phase 2 improvements:

```bash
# Basic test (no reactions, diffusion only)
./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j $(nproc) -- --no-reactions --steps 50

# Full test (with reactions and diffusion)
./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j $(nproc) -- --steps 50

# Diffusion only (no reactions)
./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j $(nproc) -- --no-reactions --steps 50

# Reactions only (no diffusion)
./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j $(nproc) -- --no-diffusion --steps 50
```

### 3. Compare with Phase 1

To see the improvements, you can also run the Phase 1 test:

```bash
./mfc.sh run examples/2D_burning_droplet/test_phase1_validation.py -t pre_process simulation -j $(nproc) -- --steps 50
```

## What to Check

### Phase 2 Improvements

1. **Gas-Phase Density for Reactions**
   - Reactions should use gas-phase density (excluding liquid)
   - Check that reaction rates are reasonable in gas cells
   - Verify no chemistry in liquid-dominated cells

2. **Diffusion Flux Scaling**
   - Diffusion should scale by gas volume fraction
   - Check species transport at liquid-gas interfaces
   - Verify smooth species profiles

3. **Mass Conservation**
   - Total mass should be conserved
   - Species mass should be conserved (except reactions)
   - Evaporated mass should appear in fuel species

### Expected Output

The simulation should:
- ✅ Compile without errors
- ✅ Run without crashes
- ✅ Produce output files in `silo_hdf5/` directory
- ✅ Show proper gas-phase chemistry behavior
- ✅ Maintain mass conservation

### Troubleshooting

If you encounter issues:

1. **Compilation errors:**
   - Check that all Phase 2 code changes are present
   - Verify `chem_params` is accessible in all modules
   - Check for syntax errors in modified files

2. **Runtime errors:**
   - Check initial conditions (volume fractions sum to 1.0)
   - Verify species mass fractions are properly initialized
   - Check that `gas_phase_threshold` is appropriate

3. **Numerical issues:**
   - Reduce time step if needed
   - Increase `gas_phase_threshold` for stability
   - Use lower-order numerics (weno_order=3)

## Validation Checklist

- [ ] Code compiles successfully
- [ ] Test case runs without errors
- [ ] Output files are generated
- [ ] No NaN or Inf values in output
- [ ] Mass conservation is maintained
- [ ] Chemistry only in gas phase
- [ ] Diffusion properly scaled

## Next Steps

After successful testing:
1. Run longer simulations to check stability
2. Compare results with Phase 1 implementation
3. Validate against analytical solutions (if available)
4. Proceed with remaining Phase 2 tasks (heat coupling, Riemann solvers)
