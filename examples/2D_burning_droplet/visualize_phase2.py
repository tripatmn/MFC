#!/usr/bin/env python3
"""
Visualization script for Phase 2 validation results.
Reads MFC output data and creates plots showing Phase 2 improvements:
1. Volume fractions (liquid, vapor, gas)
2. Species mass fractions (fuel, oxidizer)
3. Gas-phase density vs total density (Phase 2 improvement)
4. Temperature profiles
5. Phase 2 specific validations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CASE_DIR = Path(__file__).parent
OUTPUT_DIR = CASE_DIR / "figures" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain parameters (must match test_phase2_validation.py)
Nx = 200  # m+1 cells for m=199
Lx = 1.0e-3  # 1 mm

# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================

def find_output_dirs(case_dir):
    """Find MFC output directories (D* format)"""
    dirs = []
    for item in case_dir.iterdir():
        if item.is_dir() and item.name.startswith('D'):
            dirs.append(item)
    return sorted(dirs)


def find_timestep_dirs(output_dir):
    """Find timestep directories (p_all/p*/t*)"""
    timesteps = []
    p_all_dir = output_dir / "p_all"
    if not p_all_dir.exists():
        return timesteps
    
    for proc_dir in p_all_dir.iterdir():
        if proc_dir.is_dir() and proc_dir.name.startswith('p'):
            for ts_dir in proc_dir.iterdir():
                if ts_dir.is_dir():
                    try:
                        ts = int(ts_dir.name)
                        if ts not in timesteps:
                            timesteps.append(ts)
                    except ValueError:
                        continue
    
    return sorted(timesteps)


def read_1d_data(filepath, num_cells, precision=2):
    """Read 1D binary data file"""
    if not filepath.exists():
        return None
    
    dtype = np.float64 if precision == 2 else np.float32
    data = np.fromfile(str(filepath), dtype=dtype)
    
    if len(data) == num_cells:
        return data
    elif len(data) == num_cells + 1:
        # Cell boundaries, convert to cell centers
        return 0.5 * (data[:-1] + data[1:])
    else:
        print(f"Warning: Unexpected data size {len(data)} for {filepath}")
        return data[:num_cells] if len(data) > num_cells else data


def read_cons_data(output_dir, timestep, var_idx, num_cells):
    """Read conservative variable data"""
    # Find any processor directory
    p_all_dir = output_dir / "p_all"
    if not p_all_dir.exists():
        return None
    
    proc_dirs = [d for d in p_all_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    if not proc_dirs:
        return None
    
    # Try first processor
    proc_dir = proc_dirs[0]
    ts_dir = proc_dir / str(timestep)
    if not ts_dir.exists():
        return None
    
    # Read conservative variable file
    cons_file = ts_dir / f"cons.{var_idx}.{proc_dir.name[-2:]}.{timestep:06d}.dat"
    if not cons_file.exists():
        return None
    
    # Read as text file (MFC 1D format)
    try:
        data = []
        with open(cons_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        data.append(value)
                    except ValueError:
                        continue
        
        if len(data) == num_cells:
            return np.array(data)
        elif len(data) > num_cells:
            return np.array(data[:num_cells])
        else:
            # Pad with zeros if needed
            padded = np.zeros(num_cells)
            padded[:len(data)] = data
            return padded
    except Exception as e:
        print(f"Error reading {cons_file}: {e}")
        return None


def read_prim_data(output_dir, timestep, var_idx, num_cells):
    """Read primitive variable data"""
    p_all_dir = output_dir / "p_all"
    if not p_all_dir.exists():
        return None
    
    proc_dirs = [d for d in p_all_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    if not proc_dirs:
        return None
    
    proc_dir = proc_dirs[0]
    ts_dir = proc_dir / str(timestep)
    if not ts_dir.exists():
        return None
    
    prim_file = ts_dir / f"prim.{var_idx}.{proc_dir.name[-2:]}.{timestep:06d}.dat"
    if not prim_file.exists():
        return None
    
    try:
        data = []
        with open(prim_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        data.append(value)
                    except ValueError:
                        continue
        
        if len(data) == num_cells:
            return np.array(data)
        elif len(data) > num_cells:
            return np.array(data[:num_cells])
        else:
            padded = np.zeros(num_cells)
            padded[:len(data)] = data
            return padded
    except Exception as e:
        print(f"Error reading {prim_file}: {e}")
        return None


def read_grid(output_dir):
    """Read grid coordinates"""
    p_all_dir = output_dir / "p_all"
    if not p_all_dir.exists():
        # Generate uniform grid
        return np.linspace(0, Lx, Nx)
    
    proc_dirs = [d for d in p_all_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    if not proc_dirs:
        return np.linspace(0, Lx, Nx)
    
    proc_dir = proc_dirs[0]
    ts_dir = proc_dir / "0"  # First timestep
    if not ts_dir.exists():
        return np.linspace(0, Lx, Nx)
    
    # Read x_cb.dat (cell boundaries)
    x_cb_file = ts_dir / "x_cb.dat"
    if x_cb_file.exists():
        x_cb = np.fromfile(str(x_cb_file), dtype=np.float64)
        if len(x_cb) == Nx + 1:
            # Convert to cell centers
            x_cc = 0.5 * (x_cb[:-1] + x_cb[1:])
            return x_cc
        elif len(x_cb) == Nx:
            return x_cb
    
    # Fallback: uniform grid
    return np.linspace(0, Lx, Nx)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_volume_fractions(output_dir, timestep, x, num_cells):
    """Plot volume fractions for all fluids"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Read volume fractions (alpha_1, alpha_2, alpha_3)
    # For 6-eqn model with 3 fluids, alpha are at indices 6, 7, 8 (0-indexed: 5, 6, 7)
    alpha_1 = read_cons_data(output_dir, timestep, 6, num_cells)  # advxb + 0
    alpha_2 = read_cons_data(output_dir, timestep, 7, num_cells)  # advxb + 1
    alpha_3 = read_cons_data(output_dir, timestep, 8, num_cells)  # advxb + 2
    
    if alpha_1 is not None:
        ax.plot(x * 1000, alpha_1, 'b-', label=r'$\alpha_1$ (Liquid)', linewidth=2)
    if alpha_2 is not None:
        ax.plot(x * 1000, alpha_2, 'r--', label=r'$\alpha_2$ (Vapor)', linewidth=2)
    if alpha_3 is not None:
        ax.plot(x * 1000, alpha_3, 'g:', label=r'$\alpha_3$ (Oxidizer)', linewidth=2)
    
    if alpha_1 is not None and alpha_2 is not None and alpha_3 is not None:
        alpha_sum = alpha_1 + alpha_2 + alpha_3
        ax.plot(x * 1000, alpha_sum, 'k-', label='Sum', linewidth=1, alpha=0.5, linestyle='-.')
    
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Volume Fraction', fontsize=12)
    ax.set_title(f'Volume Fractions at t = {timestep} steps (Phase 2)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'volume_fractions_t{timestep:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: volume_fractions_t{timestep:04d}.png")


def plot_partial_densities(output_dir, timestep, x, num_cells):
    """Plot partial densities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Read partial densities (alpha_rho_1, alpha_rho_2, alpha_rho_3)
    rho_1 = read_cons_data(output_dir, timestep, 1, num_cells)  # contxb + 0
    rho_2 = read_cons_data(output_dir, timestep, 2, num_cells)  # contxb + 1
    rho_3 = read_cons_data(output_dir, timestep, 3, num_cells)  # contxb + 2
    
    if rho_1 is not None:
        ax.semilogy(x * 1000, rho_1 + 1e-10, 'b-', label=r'$\alpha_1 \rho_1$ (Liquid)', linewidth=2)
    if rho_2 is not None:
        ax.semilogy(x * 1000, rho_2 + 1e-10, 'r--', label=r'$\alpha_2 \rho_2$ (Vapor)', linewidth=2)
    if rho_3 is not None:
        ax.semilogy(x * 1000, rho_3 + 1e-10, 'g:', label=r'$\alpha_3 \rho_3$ (Oxidizer)', linewidth=2)
    
    if rho_1 is not None and rho_2 is not None and rho_3 is not None:
        rho_total = rho_1 + rho_2 + rho_3
        ax.semilogy(x * 1000, rho_total, 'k-', label=r'$\rho_{total}$', linewidth=1)
    
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Partial Density (kg/m³)', fontsize=12)
    ax.set_title(f'Partial Densities at t = {timestep} steps (Phase 2)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'partial_densities_t{timestep:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: partial_densities_t{timestep:04d}.png")


def plot_gas_phase_density(output_dir, timestep, x, num_cells):
    """Plot gas-phase density vs total density (Phase 2 improvement)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Read partial densities
    rho_1 = read_cons_data(output_dir, timestep, 1, num_cells)  # Liquid
    rho_2 = read_cons_data(output_dir, timestep, 2, num_cells)  # Vapor
    rho_3 = read_cons_data(output_dir, timestep, 3, num_cells)  # Oxidizer
    
    if rho_1 is not None and rho_2 is not None and rho_3 is not None:
        # Total density
        rho_total = rho_1 + rho_2 + rho_3
        
        # Gas-phase density (Phase 2: exclude liquid)
        rho_gas = rho_2 + rho_3
        
        ax.plot(x * 1000, rho_total, 'k-', label=r'$\rho_{total}$ (All fluids)', linewidth=2)
        ax.plot(x * 1000, rho_gas, 'r--', label=r'$\rho_{gas}$ (Vapor + Oxidizer)', linewidth=2)
        ax.plot(x * 1000, rho_1, 'b:', label=r'$\alpha_1 \rho_1$ (Liquid)', linewidth=2)
        
        # Show difference
        diff = rho_total - rho_gas
        ax.fill_between(x * 1000, rho_gas, rho_total, alpha=0.3, 
                        label='Difference (Liquid contribution)', color='blue')
    
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Density (kg/m³)', fontsize=12)
    ax.set_title(f'Gas-Phase vs Total Density at t = {timestep} steps\n(Phase 2: Reactions use ρ_gas)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'gas_phase_density_t{timestep:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gas_phase_density_t{timestep:04d}.png")


def plot_species_mass_fractions(output_dir, timestep, x, num_cells):
    """Plot species mass fractions (if chemistry enabled)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Species indices depend on sys_size
    # For 3-fluid 6-eqn with chemistry: chemxb starts after advxe
    # Try to read species (typically indices 11+ for 3-fluid 6-eqn)
    # H2 is typically first species (chemxb + 0)
    
    # Try reading species (indices may vary)
    species_read = False
    
    for sp_idx in range(11, 21):  # Try indices 11-20
        Y = read_prim_data(output_dir, timestep, sp_idx, num_cells)
        if Y is not None and np.max(Y) > 1e-10:
            if sp_idx == 11:
                ax.plot(x * 1000, Y, 'b-', label='Y_H2 (Fuel)', linewidth=2)
                species_read = True
            elif sp_idx == 14:
                ax.plot(x * 1000, Y, 'g--', label='Y_O2 (Oxidizer)', linewidth=2)
            elif sp_idx == 16:
                ax.plot(x * 1000, Y, 'r:', label='Y_H2O (Product)', linewidth=2)
    
    if not species_read:
        # Try reading from conservative variables
        for sp_idx in range(11, 21):
            Y_cons = read_cons_data(output_dir, timestep, sp_idx, num_cells)
            if Y_cons is not None:
                # Convert to mass fraction (divide by gas-phase density)
                rho_2 = read_cons_data(output_dir, timestep, 2, num_cells)
                rho_3 = read_cons_data(output_dir, timestep, 3, num_cells)
                if rho_2 is not None and rho_3 is not None:
                    rho_gas = rho_2 + rho_3
                    Y = Y_cons / np.maximum(rho_gas, 1e-10)
                    if sp_idx == 11:
                        ax.plot(x * 1000, Y, 'b-', label='Y_H2 (Fuel)', linewidth=2)
                        species_read = True
                    elif sp_idx == 14:
                        ax.plot(x * 1000, Y, 'g--', label='Y_O2 (Oxidizer)', linewidth=2)
    
    if species_read:
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('Mass Fraction', fontsize=12)
        ax.set_title(f'Species Mass Fractions at t = {timestep} steps (Phase 2)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'species_mass_fractions_t{timestep:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: species_mass_fractions_t{timestep:04d}.png")
    else:
        print(f"  Warning: Could not read species data for timestep {timestep}")


def plot_phase2_comparison(output_dir, timestep, x, num_cells):
    """Create comprehensive Phase 2 validation plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Read data
    alpha_1 = read_cons_data(output_dir, timestep, 6, num_cells)
    alpha_2 = read_cons_data(output_dir, timestep, 7, num_cells)
    alpha_3 = read_cons_data(output_dir, timestep, 8, num_cells)
    rho_1 = read_cons_data(output_dir, timestep, 1, num_cells)
    rho_2 = read_cons_data(output_dir, timestep, 2, num_cells)
    rho_3 = read_cons_data(output_dir, timestep, 3, num_cells)
    
    # Plot 1: Volume fractions
    ax = axes[0, 0]
    if alpha_1 is not None:
        ax.plot(x * 1000, alpha_1, 'b-', label='Liquid', linewidth=2)
    if alpha_2 is not None:
        ax.plot(x * 1000, alpha_2, 'r--', label='Vapor', linewidth=2)
    if alpha_3 is not None:
        ax.plot(x * 1000, alpha_3, 'g:', label='Oxidizer', linewidth=2)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Volume Fraction')
    ax.set_title('Volume Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 2: Gas-phase density (Phase 2 key feature)
    ax = axes[0, 1]
    if rho_1 is not None and rho_2 is not None and rho_3 is not None:
        rho_total = rho_1 + rho_2 + rho_3
        rho_gas = rho_2 + rho_3
        ax.plot(x * 1000, rho_total, 'k-', label='Total', linewidth=2)
        ax.plot(x * 1000, rho_gas, 'r--', label='Gas-phase (Phase 2)', linewidth=2)
        ax.fill_between(x * 1000, rho_gas, rho_total, alpha=0.3, color='blue', label='Liquid')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Density (kg/m³)')
    ax.set_title('Phase 2: Gas-Phase Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gas volume fraction
    ax = axes[1, 0]
    if alpha_1 is not None:
        alpha_gas = 1.0 - alpha_1
        ax.plot(x * 1000, alpha_gas, 'g-', label=r'$\alpha_{gas}$', linewidth=2)
        ax.axhline(y=0.01, color='r', linestyle='--', label='Threshold (0.01)', linewidth=1)
        ax.fill_between(x * 1000, 0, 0.01, alpha=0.2, color='red', label='No Chemistry')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Gas Volume Fraction')
    ax.set_title('Gas Phase Fraction (Chemistry Active if > 0.01)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 4: Partial densities
    ax = axes[1, 1]
    if rho_1 is not None:
        ax.semilogy(x * 1000, rho_1 + 1e-10, 'b-', label='Liquid', linewidth=2)
    if rho_2 is not None:
        ax.semilogy(x * 1000, rho_2 + 1e-10, 'r--', label='Vapor', linewidth=2)
    if rho_3 is not None:
        ax.semilogy(x * 1000, rho_3 + 1e-10, 'g:', label='Oxidizer', linewidth=2)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Partial Density (kg/m³)')
    ax.set_title('Partial Densities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase 2 Validation: t = {timestep} steps', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'phase2_validation_t{timestep:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phase2_validation_t{timestep:04d}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 2 Validation Visualization")
    print("=" * 60)
    
    # Find output directory
    output_dirs = find_output_dirs(CASE_DIR)
    if not output_dirs:
        print(f"Error: No output directories found in {CASE_DIR}")
        print("Run the simulation first:")
        print("  ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4")
        return
    
    output_dir = output_dirs[0]  # Use first output directory
    print(f"Using output directory: {output_dir.name}")
    
    # Find timesteps
    timesteps = find_timestep_dirs(output_dir)
    if not timesteps:
        print(f"Warning: No timestep directories found in {output_dir}")
        print("Trying to read from p_all structure...")
        timesteps = [0]  # Try initial timestep
    
    print(f"Found timesteps: {timesteps[:5]}..." if len(timesteps) > 5 else f"Found timesteps: {timesteps}")
    
    # Read grid
    x = read_grid(output_dir)
    num_cells = len(x)
    print(f"Grid: {num_cells} cells, x = [{x[0]*1000:.4f}, {x[-1]*1000:.4f}] mm")
    
    # Generate plots for key timesteps
    print("\nGenerating Phase 2 validation plots...")
    
    plot_timesteps = [timesteps[0], timesteps[-1]] if len(timesteps) > 1 else timesteps[:1]
    if len(timesteps) > 2:
        mid_idx = len(timesteps) // 2
        plot_timesteps.append(timesteps[mid_idx])
    
    for ts in plot_timesteps:
        print(f"\nTimestep {ts}:")
        plot_volume_fractions(output_dir, ts, x, num_cells)
        plot_partial_densities(output_dir, ts, x, num_cells)
        plot_gas_phase_density(output_dir, ts, x, num_cells)
        plot_species_mass_fractions(output_dir, ts, x, num_cells)
        plot_phase2_comparison(output_dir, ts, x, num_cells)
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print("\nKey images to check:")
    print("  - phase2_validation_t*.png: Comprehensive Phase 2 validation")
    print("  - gas_phase_density_t*.png: Shows Phase 2 improvement (gas vs total density)")
    print("  - volume_fractions_t*.png: Liquid-vapor-gas distribution")
    print("  - species_mass_fractions_t*.png: Species transport (if chemistry enabled)")


if __name__ == "__main__":
    main()
