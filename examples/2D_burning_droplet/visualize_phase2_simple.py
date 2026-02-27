#!/usr/bin/env python3
"""
Simple visualization script for Phase 2 validation.
Uses MFC's built-in visualization tools if available, otherwise reads data directly.

Usage:
  python3 examples/2D_burning_droplet/visualize_phase2_simple.py
  python3 examples/2D_burning_droplet/visualize_phase2_simple.py --case-dir /path/to/case
  # From repo root, or from the case dir (examples/2D_burning_droplet) after running simulation
"""

import argparse
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

# Default: directory containing this script
DEFAULT_CASE_DIR = Path(__file__).resolve().parent

# Domain parameters (2D case: m=199, n=14 → (m+1)*(n+1) = 200*15 cells per variable in restart)
Nx = 200
Ny = 15
Lx = 1.0e-3
Ly = 0.2e-3

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
    """Read restart data for a given timestep (parallel_io: lustre_<step>/<step>_0.dat)."""
    step_dir = restart_dir / f"lustre_{timestep}"
    if not step_dir.exists():
        return None
    rank_files = sorted(step_dir.glob(f"{timestep}_*.dat"))
    if not rank_files:
        return None
    # Single-rank: read first file; multi-rank would require concatenating by rank
    filename = rank_files[0]
    data = np.fromfile(str(filename), dtype=np.float64)
    num_vars = len(data) // num_cells
    if len(data) % num_cells != 0:
        print(f"Warning: Data size {len(data)} not divisible by {num_cells}")
        return None
    data = data.reshape((num_vars, num_cells))
    return data


def read_grid(restart_dir, num_cells):
    """Read grid coordinates (x_cb.dat in restart_data)."""
    grid_file = restart_dir / "x_cb.dat"
    if grid_file.exists():
        x = np.fromfile(str(grid_file), dtype=np.float64)
        if len(x) == num_cells + 1:
            x_cc = 0.5 * (x[:-1] + x[1:])
        elif len(x) >= num_cells:
            x_cc = x[:num_cells]
        else:
            x_cc = np.linspace(0, Lx, num_cells)
        return x_cc
    return np.linspace(0, Lx, num_cells)


def find_data_files(case_dir):
    """Find MFC output files - check restart_data (parallel_io: lustre_<step>/<step>_<rank>.dat) and D/."""
    # restart_data: parallel_io format uses directory lustre_<step> with files <step>_<rank>.dat
    restart_dir = case_dir / "restart_data"
    if restart_dir.exists():
        timesteps = []
        for d in sorted(restart_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("lustre_"):
                continue
            try:
                ts = int(d.name.replace("lustre_", ""))
            except ValueError:
                continue
            if list(d.glob(f"{ts}_*.dat")):
                timesteps.append(ts)
        if timesteps:
            return sorted(timesteps), "restart_data", case_dir

    # Fallback to D/ directory (serial format)
    D_dir = case_dir / "D"
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
                return sorted(timesteps), "D", case_dir

    return None, None, None


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_case_dir_candidates(args_case_dir):
    """Build list of case directories to try (script dir, cwd, cwd/examples/2D_burning_droplet)."""
    candidates = []
    if args_case_dir is not None:
        candidates.append(Path(args_case_dir).resolve())
    candidates.append(DEFAULT_CASE_DIR)
    cwd = Path.cwd()
    if cwd not in candidates:
        candidates.append(cwd)
    # If running from repo root, case dir might be examples/2D_burning_droplet
    repo_case = cwd / "examples" / "2D_burning_droplet"
    if repo_case.exists() and repo_case not in candidates:
        candidates.append(repo_case)
    return candidates


def plot_phase2_validation(case_dir=None):
    """Create comprehensive Phase 2 validation plots"""
    candidates = get_case_dir_candidates(case_dir)
    timesteps, format_type, used_case_dir = None, None, None
    for cdir in candidates:
        timesteps, format_type, used_case_dir = find_data_files(cdir)
        if timesteps:
            break
    if not timesteps:
        print("No data files found in any of these locations:")
        for c in candidates:
            print(f"  {c}")
        print("Run the simulation first, then run this script from repo root or from the case directory:")
        print("  ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4")
        print("  python3 examples/2D_burning_droplet/visualize_phase2_simple.py")
        return None

    OUTPUT_DIR = used_case_dir / "figures" / "phase2"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using case dir: {used_case_dir}")
    print(f"Found timesteps: {timesteps} (format: {format_type})")

    # Use first and last timestep
    plot_ts = [timesteps[0]]
    if len(timesteps) > 1:
        plot_ts.append(timesteps[-1])

    if format_type == "restart_data":
        # Use restart_data format (binary); 2D grid has (m+1)*(n+1) = Nx*Ny cells
        restart_dir = used_case_dir / "restart_data"
        total_cells = Nx * Ny
        x_grid = read_grid(restart_dir, total_cells)
        if x_grid is not None and len(x_grid) >= Nx:
            # 2D flattened: first row along x is indices 0:Nx (x fastest in MFC)
            x = x_grid[:Nx] if len(x_grid) == total_cells else np.linspace(0, Lx, Nx)
        else:
            x = np.linspace(0, Lx, Nx)

        for ts in plot_ts:
            print(f"\nGenerating plots for timestep {ts}...")

            data = read_restart_data(restart_dir, ts, total_cells)
            if data is None:
                print(f"  Warning: Could not read data for timestep {ts}")
                continue

            num_vars = data.shape[0]
            nc = data.shape[1]
            print(f"  Read {num_vars} variables, {nc} cells")
            # 1D slice: first row (indices 0:Nx) when 2D
            n_slice = min(Nx, nc)
            if num_vars >= 8:
                rho_1 = data[0, :n_slice]
                rho_2 = data[1, :n_slice]
                rho_3 = data[2, :n_slice]
                alpha_1 = data[5, :n_slice]
                alpha_2 = data[6, :n_slice]
                alpha_3 = data[7, :n_slice]
            else:
                print(f"  Warning: Unexpected number of variables ({num_vars})")
                continue
            x_slice = x[:n_slice] if len(x) >= n_slice else np.linspace(0, Lx, n_slice)

            _create_plots(x_slice, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts, OUTPUT_DIR)
    else:
        # Use D/ format (text files)
        D_dir = used_case_dir / "D"
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

            _create_plots(x, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts, OUTPUT_DIR)


def _create_plots(x, alpha_1, alpha_2, alpha_3, rho_1, rho_2, rho_3, ts, output_dir):
    """Create comprehensive Phase 2 validation plots"""
    if output_dir is None:
        output_dir = DEFAULT_CASE_DIR / "figures" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    
    output_file = output_dir / f'phase2_validation_t{ts:04d}.png'
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
        output_file = output_dir / f'gas_phase_density_t{ts:04d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")

    return OUTPUT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 validation visualization")
    parser.add_argument("--case-dir", type=str, default=None,
                        help="Case directory containing restart_data/ or D/ (default: script dir, then cwd)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 Validation Visualization")
    print("=" * 60)
    print()

    out_dir = plot_phase2_validation(case_dir=args.case_dir)

    print()
    print("=" * 60)
    if out_dir is not None:
        print(f"Images saved to: {out_dir}")
    print("=" * 60)
