# Visual Confirmation Guide for Phase 2

## Overview

This guide helps you visually confirm that Phase 2 implementation is working correctly by generating and examining plots.

## Step-by-Step Process

### 1. Run the Test

```powershell
cd g:\Projects\MFC
./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4 -- --no-reactions --steps 50
```

### 2. Generate Images

After the test completes, run the visualization script:

```powershell
# Option 1: Simple script (recommended)
python examples/2D_burning_droplet/visualize_phase2_simple.py

# Option 2: Full script with more features
python examples/2D_burning_droplet/visualize_phase2.py

# Option 3: Automated (runs test + visualization)
.\examples\2D_burning_droplet\generate_images.ps1
```

### 3. View Images

Images will be saved to:
```
examples/2D_burning_droplet/figures/phase2/
```

Key images to check:
- `phase2_validation_t*.png` - Comprehensive validation plot
- `gas_phase_density_t*.png` - Shows Phase 2 improvement

## What the Images Show

### Image 1: Phase 2 Validation (4-panel)

**Top Left: Volume Fractions**
- Blue line: Liquid volume fraction (α₁)
- Red dashed: Vapor volume fraction (α₂)  
- Green dotted: Oxidizer volume fraction (α₃)
- Black dash-dot: Sum (should be ≈ 1.0)

**Top Right: Gas-Phase Density (Phase 2 Key Feature)**
- Black line: Total density (ρ_total = ρ₁ + ρ₂ + ρ₃)
- Red dashed: Gas-phase density (ρ_gas = ρ₂ + ρ₃) ← **Phase 2 uses this**
- Blue shaded: Liquid contribution (difference)
- **What to check:** In liquid region, large difference. In gas region, small difference.

**Bottom Left: Chemistry Active Region**
- Green line: Gas volume fraction (α_gas = 1 - α₁)
- Red dashed line: Threshold (0.01)
- Red shaded: Region where chemistry is disabled (liquid-dominated)
- **What to check:** Chemistry should only be active where α_gas > 0.01

**Bottom Right: Partial Densities**
- Blue: Liquid partial density (α₁ρ₁)
- Red dashed: Vapor partial density (α₂ρ₂)
- Green dotted: Oxidizer partial density (α₃ρ₃)
- **What to check:** Smooth transitions, no spikes or discontinuities

### Image 2: Gas-Phase Density Comparison

This detailed plot shows:
- **Total density** (black) - includes all fluids
- **Gas-phase density** (red) - Phase 2 uses this for reactions
- **Liquid contribution** (blue shaded) - excluded in Phase 2

**Key Validation:**
- The difference between black and red lines shows the liquid contribution
- Phase 2 reactions use the red line (gas-phase density)
- This is more accurate than using total density

## Expected Results

### ✅ Phase 2 Working Correctly

1. **Gas-phase density plot shows clear difference:**
   - In liquid region (left): Large gap between total and gas density
   - In gas region (right): Small gap (mostly gas)
   - This confirms Phase 2 is computing and using gas-phase density

2. **Volume fractions are physical:**
   - Sum ≈ 1.0 everywhere
   - Smooth transitions at interface
   - No negative values

3. **Chemistry threshold working:**
   - Chemistry disabled in liquid region (red shaded)
   - Chemistry enabled in gas region (green line above threshold)

4. **Partial densities smooth:**
   - No numerical artifacts
   - Proper conservation

### ❌ Issues to Watch For

1. **Flat/zero values everywhere:**
   - Test may not have run correctly
   - Check output files exist

2. **Discontinuities or spikes:**
   - Numerical instability
   - May need smaller time step

3. **Volume fractions don't sum to 1:**
   - Initial condition problem
   - Check case file

4. **No difference between total and gas density:**
   - Phase 2 code may not be active
   - Check compilation

## Troubleshooting

### Images not generated

1. **Check Python is installed:**
   ```powershell
   python --version
   ```

2. **Check matplotlib is installed:**
   ```powershell
   python -c "import matplotlib; print('OK')"
   ```

3. **Check output files exist:**
   ```powershell
   ls examples/2D_burning_droplet/D/cons.*.dat
   ```

### Wrong data in images

1. **Check variable indices:**
   - Volume fractions: cons.6, cons.7, cons.8
   - Partial densities: cons.1, cons.2, cons.3
   - May need to adjust if your case has different structure

2. **Check timestep:**
   - Script uses first and last timestep
   - Verify timesteps exist: `ls D/p_all/p*/`

## Next Steps

After visual confirmation:

1. ✅ Phase 2 improvements are working
2. Run longer simulations to check stability
3. Compare with Phase 1 results
4. Proceed with remaining Phase 2 tasks

## Files Created

- `visualize_phase2_simple.py` - Simple visualization script
- `visualize_phase2.py` - Full-featured visualization script  
- `generate_images.ps1` - Automated test + visualization
- `README_VISUALIZATION.md` - Detailed visualization guide
- `VISUAL_CONFIRMATION.md` - This file
