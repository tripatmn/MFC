#!/usr/bin/env python3
"""
Simple visualization script for Phase 2 validation.
Uses MFC's built-in visualization tools if available, otherwise reads data directly.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to use MFC's visualization tools
try:
    import mfc.viz
    USE_MFC_VIZ = True
except ImportError:
    USE_MFC_VIZ = False
    print("Note: mfc.viz not available, using direct file reading")

# =============================================================================
# CONFIGURATION
# =============================================================================
CASE_DIR = Path(__file__).parent
OUTPUT_DIR = CASE_DIR / "figures" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain parameters
Nx = 200
Lx = 1.0e-3

# =============================================================================
# DATA READING
# =============================================================================

def read_1d_dat_file(filepath, num_cells):
    """Read 1D .dat file (space-separated: x value)"""
    if not filepath.exists():
        return None, None
    
    try:
        data = np.loadtxt(str(filepath))
        if data.ndim == 1:
            # Single column (values only)
            x = np.linspace(0, Lx, len(data))
            return x, data
        elif data.ndim == 2 and data.shape[1] == 2:
            # Two columns (x, value)
            return data[:, 0], data[:, 1]
        else:
            print(f"Warning: Unexpected shape {data.shape} in {filepath}")
            return None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def read_restart_data(restart_dir, timestep, num_cells):
    """Read restart data for a given timestep (binary format)"""
    filename = restart_dir / f"lustre_{timestep}.dat"
    
    if not filename.exists():
        return None
    
    # Read binary data
    data = np.fromfile(str(filename), dtype=np.float64)
    
    # Reshape based on number of variables
    num_vars = len(data) // num_cells
    if len(data) % num_cells != 0:
        print(f"Warning: Data size {len(data)} not divisible by {num_cells}")
        return None
    
    # Reshape to (num_vars, num_cells)
    data = data.reshape((num_vars, num_cells))
    
    return data


def read_grid(restart_dir, num_cells):
    """Read grid coordinates"""
    grid_file = restart_dir / "lustre_x_cb.dat"
    
    if grid_file.exists():
        x = np.fromfile(str(grid_file), dtype=np.float64)
        # Cell-center coordinates
        if len(x) == num_cells + 1:
            x_cc = 0.5 * (x[:-1] + x[1:])
        else:
            x_cc = x[:num_cells]
        return x_cc
    else:
        # Generate uniform grid
        return np.linspace(0, Lx, num_cells)


def find_data_files():
    """Find MFC output files - check both D/ and restart_data/"""
    # First try restart_data (parallel_io format)
    restart_dir = CASE_DIR / "restart_data"
    if restart_dir.exists():
        # Find timestep files
        timesteps = set()
        for f in restart_dir.glob("lustre_*.dat"):
            if "x_cb" not in f.name:
                try:
                    # Extract timestep from filename like "lustre_0.dat"
                    ts_str = f.stem.replace("lustre_", "")
                    ts = int(ts_str)
                    timesteps.add(ts)
                except ValueError:
                    continue
        if timesteps:
            return sorted(timesteps), "restart_data"
    
    # Fallback to D/ directory (serial format)
    D_dir = CASE_DIR / "D"
    if D_dir.exists():
        cons_files = list(D_dir.glob("cons.1.*.*.dat"))
        if cons_files:
            timesteps = set()
            for f in cons_files:
                parts = f.stem.split('.')
                if len(parts) >= 3:
                    try:
                        ts = int(parts[-1])
                        timesteps.add(ts)
                    except ValueError:
                        continue
            if timesteps:
                return sorted(timesteps), "D"
    
    return None, None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_phase2_validation():
    """Create comprehensive Phase 2 validation plots"""
    
    timesteps, format_type = find_data_files()
    if not timesteps:
        print("No data files found. Run the simulation first:")
        print("  ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4")
        return
    
    print(f"Found timesteps: {timesteps} (format: {format_type})")
    
    # Use first and last timestep
    plot_ts = [timesteps[0]]
    if len(timesteps) > 1:
        plot_ts.append(timesteps[-1])
    
    if format_type == "restart_data":
        # Use restart_data format (binary)
        restart_dir = CASE_DIR / "restart_data"
        x = read_grid(restart_dir, Nx)
        
        for ts in plot_ts:
            print(f"\nGenerating plots for timestep {ts}...")
            
            data = read_restart_data(restart_dir, ts, Nx)
            if data is None:
                print(f"  Warning: Could not read data for timestep {ts}")
                continue
            
            num_vars = data.shape[0]
            print(f"  Read {num_vars} variables, {data.shape[1]} cells")
            
            # For 3-fluid 6-eqn model (model_eqns=3):
            # Variables: alpha_rho_1, alpha_rho_2, alpha_rho_3, rho*u, E, alpha_1, alpha_2, alpha_3, int_e_1, int_e_2, int_e_3
            # So alpha starts at index 5 (0-indexed)
            if num_vars >= 8:
                rho_1 = data[0, :]  # alpha_rho_1
                rho_2 = data[1, :]  # alpha_rho_2
                rho_3 = data[2, :]  # alpha_rho_3
                alpha_1 = data[5, :]  # alpha_1
                alpha_2 = data[6, :]  # alpha_2
                alpha_3 = data[7, :]  # alpha_3
            else:
                print(f"  Warning: Unexpected number of variables ({num_vars})")
                continue
            
            # Create plots for this timestep
            _create_plots(x, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts)
    else:
        # Use D/ format (text files)
        D_dir = CASE_DIR / "D"
        x = np.linspace(0, Lx, Nx)
        
        for ts in plot_ts:
            print(f"\nGenerating plots for timestep {ts}...")
            
            # Read volume fractions
            alpha_1_file = D_dir / f"cons.6.00.{ts:06d}.dat"  # advxb + 0
            alpha_2_file = D_dir / f"cons.7.00.{ts:06d}.dat"  # advxb + 1
            alpha_3_file = D_dir / f"cons.8.00.{ts:06d}.dat"  # advxb + 2
            
            # Read partial densities
            rho_1_file = D_dir / f"cons.1.00.{ts:06d}.dat"  # contxb + 0
            rho_2_file = D_dir / f"cons.2.00.{ts:06d}.dat"  # contxb + 1
            rho_3_file = D_dir / f"cons.3.00.{ts:06d}.dat"  # contxb + 2
            
            # Read data
            x_a1, alpha_1 = read_1d_dat_file(alpha_1_file, Nx)
            x_a2, alpha_2 = read_1d_dat_file(alpha_2_file, Nx)
            x_a3, alpha_3 = read_1d_dat_file(alpha_3_file, Nx)
            x_r1, rho_1 = read_1d_dat_file(rho_1_file, Nx)
            x_r2, rho_2 = read_1d_dat_file(rho_2_file, Nx)
            x_r3, rho_3 = read_1d_dat_file(rho_3_file, Nx)
            
            # Use first valid x
            if x_a1 is not None:
                x = x_a1
            elif x_r1 is not None:
                x = x_r1
            
            if alpha_1 is None or alpha_2 is None or alpha_3 is None or rho_1 is None or rho_2 is None or rho_3 is None:
                print(f"  Warning: Missing data for timestep {ts}")
                continue
        
        # Create plots for this timestep
        _create_plots(x, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts)


def _create_plots(x, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts):
    """Create comprehensive Phase 2 validation plots"""
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
    # Plot 1: Volume fractions
    ax = axes[0, 0]
    if alpha_1 is not None:
        ax.plot(x * 1000, alpha_1, 'b-', label=r'$\alpha_1$ (Liquid)', linewidth=2)
    if alpha_2 is not None:
        ax.plot(x * 1000, alpha_2, 'r--', label=r'$\alpha_2$ (Vapor)', linewidth=2)
    if alpha_3 is not None:
        ax.plot(x * 1000, alpha_3, 'g:', label=r'$\alpha_3$ (Oxidizer)', linewidth=2)
    if alpha_1 is not None and alpha_2 is not None and alpha_3 is not None:
        ax.plot(x * 1000, alpha_1 + alpha_2 + alpha_3, 'k-.', label='Sum', linewidth=1, alpha=0.5)
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('Volume Fraction', fontsize=11)
    ax.set_title('Volume Fractions', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gas-phase density (Phase 2 key feature)
    ax = axes[0, 1]
    if rho_1 is not None and rho_2 is not None and rho_3 is not None:
        rho_total = rho_1 + rho_2 + rho_3
        rho_gas = rho_2 + rho_3
        ax.plot(x * 1000, rho_total, 'k-', label=r'$\rho_{total}$', linewidth=2)
        ax.plot(x * 1000, rho_gas, 'r--', label=r'$\rho_{gas}$ (Phase 2)', linewidth=2)
        ax.fill_between(x * 1000, rho_gas, rho_total, alpha=0.3, color='blue', label='Liquid')
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('Density (kg/m³)', fontsize=11)
    ax.set_title('Phase 2: Gas-Phase Density', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gas volume fraction with threshold
    ax = axes[1, 0]
    if alpha_1 is not None:
        alpha_gas = 1.0 - alpha_1
        ax.plot(x * 1000, alpha_gas, 'g-', label=r'$\alpha_{gas}$', linewidth=2)
        ax.axhline(y=0.01, color='r', linestyle='--', label='Threshold', linewidth=1)
        ax.fill_between(x * 1000, 0, 0.01, alpha=0.2, color='red', label='No Chemistry')
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('Gas Volume Fraction', fontsize=11)
    ax.set_title('Chemistry Active Region', fontsize=12)
    ax.legend(fontsize=9)
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
    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('Partial Density (kg/m³)', fontsize=11)
    ax.set_title('Partial Densities', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase 2 Validation: t = {ts} steps', fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f'phase2_validation_t{ts:04d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_file}")
    
    # Create gas-phase density comparison plot
    if rho_1 is not None and rho_2 is not None and rho_3 is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rho_total = rho_1 + rho_2 + rho_3
        rho_gas = rho_2 + rho_3
        
        ax.plot(x * 1000, rho_total, 'k-', label=r'$\rho_{total}$ (All fluids)', linewidth=2.5)
        ax.plot(x * 1000, rho_gas, 'r--', label=r'$\rho_{gas}$ (Vapor + Oxidizer)', linewidth=2.5)
        ax.plot(x * 1000, rho_1, 'b:', label=r'$\alpha_1 \rho_1$ (Liquid)', linewidth=2)
        ax.fill_between(x * 1000, rho_gas, rho_total, alpha=0.3, color='blue', 
                      label='Liquid contribution (excluded in Phase 2)')
        
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('Density (kg/m³)', fontsize=12)
        ax.set_title(f'Phase 2 Improvement: Gas-Phase Density at t = {ts} steps\n' +
                    'Reactions use ρ_gas instead of ρ_total', fontsize=13)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f'gas_phase_density_t{ts:04d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Validation Visualization")
    print("=" * 60)
    print()
    
    plot_phase2_validation()
    
    print()
    print("=" * 60)
    print(f"Images saved to: {OUTPUT_DIR}")
    print("=" * 60)
