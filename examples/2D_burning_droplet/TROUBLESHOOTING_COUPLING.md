# Troubleshooting Chemistry + Phase Change Coupling

## Common Issues and Solutions

### Issue 1: Case Validation Errors

**Error:** `"Multiphase chemistry requires relax = T for phase change"`

**Solution:** Make sure you have both:
```python
"relax": "T",
"chemistry": "T",
"chem_params%multiphase": "T",
```

**Error:** `"Multiphase chemistry requires num_fluids >= 2"`

**Solution:** You need at least 2 fluids (liquid + vapor). For burning droplet, typically 3:
```python
"num_fluids": 3,  # liquid, vapor, oxidizer
"model_eqns": 3,  # 6-equation model for phase change
```

**Error:** `"chem_params%liquid_phase_idx must be in range [1, num_fluids]"`

**Solution:** Check that your liquid phase index matches your fluid setup:
```python
"chem_params%liquid_phase_idx": 1,  # If liquid is fluid 1
```

**Error:** `"chem_params%fuel_species_idx must be in range [1, num_species]"`

**Solution:** Verify the fuel species index matches your Cantera mechanism:
```python
# For h2o2.yaml: H2=1, H=2, O=3, O2=4, OH=5, H2O=6, HO2=7, H2O2=8, AR=9, N2=10
"chem_params%fuel_species_idx": 1,  # H2 is species 1
```

---

### Issue 2: Species Not Initialized Correctly

**Problem:** Species mass fractions not set for gas phase

**Solution:** Initialize species for gas-phase patches:
```python
# For background (oxidizer gas)
for i in range(1, num_species + 1):
    case[f"patch_icpp(1)%Y({i})"] = 0.0
case["patch_icpp(1)%Y(4)"] = 0.233   # O2
case["patch_icpp(1)%Y(10)"] = 0.767  # N2

# For liquid droplet (no species initially, will be added by evaporation)
for i in range(1, num_species + 1):
    case[f"patch_icpp(2)%Y({i})"] = 0.0
```

---

### Issue 3: Chemistry Not Running in Gas Phase

**Problem:** Reactions not occurring even in gas regions

**Check:**
1. Verify `gas_phase_threshold` is appropriate:
   ```python
   "chem_params%gas_phase_threshold": 0.01,  # Too high = chemistry disabled
   ```

2. Check that gas volume fraction is above threshold:
   ```python
   alpha_gas = 1.0 - alpha_liquid
   # Chemistry runs if alpha_gas > gas_phase_threshold
   ```

3. Ensure temperature is high enough for reactions:
   ```python
   # Initial temperature should be > ignition temperature
   T0 = 1200.0  # K (for H2-O2, typically 800-1000 K)
   ```

---

### Issue 4: Evaporated Mass Not Appearing in Species

**Problem:** Liquid evaporates but fuel species doesn't increase

**Check:**
1. Verify phase change is actually occurring:
   ```python
   "relax": "T",
   "relax_model": 6,  # pTg-equilibrium for proper phase change
   ```

2. Check that `fuel_species_idx` is correct:
   ```python
   # Must match the species index in your mechanism
   "chem_params%fuel_species_idx": 1,  # For H2 in h2o2.yaml
   ```

3. Monitor the coupling in output:
   - Check `alpha_rho(1)` (liquid) decreases
   - Check `alpha_rho(2)` (vapor) increases  
   - Check `chemxb + fuel_species_idx - 1` (fuel species) increases

---

### Issue 5: Numerical Instabilities

**Problem:** NaN, Inf, or crashes during simulation

**Solutions:**
1. **Reduce time step:**
   ```python
   cfl = 0.1  # Lower CFL for stability
   dt = cfl * min(dx, dy) / c_sound
   ```

2. **Increase gas phase threshold:**
   ```python
   "chem_params%gas_phase_threshold": 0.05,  # More conservative
   ```

3. **Use lower-order numerics:**
   ```python
   "weno_order": 3,  # Instead of 5
   "time_stepper": 2,  # Instead of 3
   ```

4. **Check initial conditions:**
   - Ensure all volume fractions sum to 1.0
   - Ensure all mass fractions sum to 1.0 (in gas phase)
   - Avoid pure liquid or pure vapor at boundaries

---

### Issue 6: Mass Conservation Violations

**Problem:** Total mass not conserved

**Check:**
1. Verify species mass fractions sum to 1.0 in gas phase:
   ```python
   sum_Y = sum(Y[i] for i in range(1, num_species + 1))
   # Should be ≈ 1.0 in gas cells
   ```

2. Monitor total mass:
   ```python
   total_mass = sum(alpha_rho[i] for i in range(1, num_fluids + 1))
   # Should be constant (closed system)
   ```

3. Check that evaporated mass transfer is correct:
   ```python
   # In phase change: dm_evap = m1_old - m1_new
   # In species: dY_fuel ≈ dm_evap / rho_gas
   ```

---

### Issue 7: Wrong Fluid Indices

**Problem:** Liquid/vapor indices don't match expectations

**Important:** The phase change module uses hardcoded indices:
```fortran
integer, parameter :: lp = 1  ! liquid phase index
integer, parameter :: vp = 2  ! vapor phase index
```

**Solution:** Your fluid setup must match:
```python
# Fluid 1: Liquid (must be index 1)
# Fluid 2: Vapor (must be index 2)
# Fluid 3+: Other fluids (oxidizer, etc.)

"chem_params%liquid_phase_idx": 1,  # Must be 1
```

---

## Diagnostic Checklist

Before running your case, verify:

- [ ] `relax = "T"` (phase change enabled)
- [ ] `chemistry = "T"` (chemistry enabled)
- [ ] `chem_params%multiphase = "T"` (coupling enabled)
- [ ] `num_fluids >= 2` (at least liquid + vapor)
- [ ] `model_eqns = 3` (6-equation model for phase change)
- [ ] `relax_model = 6` (pTg-equilibrium)
- [ ] `chem_params%liquid_phase_idx = 1` (liquid is fluid 1)
- [ ] `chem_params%fuel_species_idx` matches mechanism
- [ ] `chem_params%gas_phase_threshold` in [0, 1]
- [ ] Species initialized for gas patches
- [ ] Volume fractions sum to 1.0
- [ ] Mass fractions sum to 1.0 (in gas)
- [ ] Initial temperature > ignition temperature

---

## Example Working Configuration

```python
{
    # Domain
    "m": 199,
    "n": 0,
    "p": 0,
    
    # Model
    "model_eqns": 3,
    "num_fluids": 3,
    "num_patches": 2,
    
    # Phase change
    "relax": "T",
    "relax_model": 6,
    "palpha_eps": 1e-2,
    "ptgalpha_eps": 1e-2,
    
    # Chemistry
    "chemistry": "T",
    "cantera_file": "h2o2.yaml",
    "chem_params%diffusion": "T",
    "chem_params%reactions": "T",
    "chem_params%transport_model": 2,
    
    # Multiphase coupling
    "chem_params%multiphase": "T",
    "chem_params%liquid_phase_idx": 1,
    "chem_params%fuel_species_idx": 1,  # H2
    "chem_params%gas_phase_threshold": 0.01,
    
    # Numerics
    "time_stepper": 3,
    "weno_order": 3,
    "riemann_solver": 2,
    
    # Fluid properties (3 fluids: liquid, vapor, air)
    "fluid_pp(1)%gamma": ...,
    "fluid_pp(2)%gamma": ...,
    "fluid_pp(3)%gamma": ...,
    
    # Patches with proper volume fractions and species
    "patch_icpp(1)%alpha(1)": ...,
    "patch_icpp(1)%alpha(2)": ...,
    "patch_icpp(1)%alpha(3)": ...,
    # ... species mass fractions for gas phase
}
```

---

## Getting Help

If you're still experiencing issues:

1. **Share your case file** - We can check the configuration
2. **Share error messages** - Exact error text helps diagnose
3. **Check validation output** - Run `test_phase1_validation.py`
4. **Monitor key variables** - Track alpha, species, temperature
5. **Start simple** - Try 1D case first before 2D/3D
