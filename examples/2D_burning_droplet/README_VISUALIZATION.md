# Generating Visualization Images for Phase 2

## Quick Start

After running the test, generate images with:

```powershell
# Windows PowerShell
python examples/2D_burning_droplet/visualize_phase2_simple.py
```

Or use the automated script:

```powershell
.\examples\2D_burning_droplet\generate_images.ps1
```

## What Images Are Generated

The visualization script creates images in `figures/phase2/`:

1. **`phase2_validation_t*.png`** - Comprehensive 4-panel plot showing:
   - Volume fractions (liquid, vapor, oxidizer)
   - Gas-phase density vs total density (Phase 2 key feature)
   - Gas volume fraction with chemistry threshold
   - Partial densities

2. **`gas_phase_density_t*.png`** - Detailed comparison showing:
   - Total density (all fluids)
   - Gas-phase density (Phase 2: excludes liquid)
   - Liquid contribution (highlighted)

## What to Look For

### Phase 2 Improvements (Visual Confirmation)

1. **Gas-Phase Density Plot**
   - Should show clear difference between `ρ_total` and `ρ_gas`
   - In liquid region: `ρ_total >> ρ_gas` (large difference)
   - In gas region: `ρ_total ≈ ρ_gas` (small difference)
   - This confirms Phase 2 is using correct density for reactions

2. **Volume Fractions**
   - Liquid (α₁) should be high on left, low on right
   - Vapor (α₂) and Oxidizer (α₃) should be low on left, high on right
   - Sum should be ≈ 1.0 everywhere

3. **Chemistry Active Region**
   - Gas volume fraction should be > 0.01 in gas regions
   - Red shaded area shows where chemistry is disabled (liquid-dominated)

4. **Partial Densities**
   - Should show smooth transitions at liquid-gas interface
   - Liquid density should be much higher than gas densities

## Manual Visualization

If the automatic script doesn't work, you can:

1. **Check output files exist:**
   ```powershell
   ls examples/2D_burning_droplet/D/cons.*.dat
   ```

2. **Run visualization manually:**
   ```powershell
   cd examples/2D_burning_droplet
   python visualize_phase2_simple.py
   ```

3. **View images:**
   ```powershell
   # Images will be in:
   examples/2D_burning_droplet/figures/phase2/
   ```

## Troubleshooting

### No images generated
- Check that test ran successfully (output files in `D/` directory)
- Verify Python and matplotlib are installed
- Check file permissions

### Images show flat/zero values
- Verify test actually ran (check timestep directories)
- Check that correct variable indices are being read
- May need to adjust indices in visualization script

### Missing data
- Ensure test ran with `--steps` > 0
- Check that `t_step_save` in test case is appropriate
- Verify output format is correct (format=1 for text files)
