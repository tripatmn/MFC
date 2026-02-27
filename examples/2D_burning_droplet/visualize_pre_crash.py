#!/usr/bin/env python3
"""
Visualize the last saved state before a crash to diagnose NaN/instability.
Reads restart_data (parallel_io format), prints per-variable diagnostics,
and plots key fields with NaN/Inf highlighted.

Usage:
  python examples/2D_burning_droplet/visualize_pre_crash.py
  python examples/2D_burning_droplet/visualize_pre_crash.py --steps 0 19   # specific steps
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_CASE_DIR = Path(__file__).resolve().parent

# Phase 2 case: 3 fluids, 10 species (from indices.dat)
NUM_FLUIDS = 3
NUM_SPECIES = 10
# Conservative layout: 1-3 alpha_rho, 4 rho*u, 5 E, 6-8 alpha, 9-11 int energy, 12-21 species
CONTXE = 3
E_IDX = 5
ADVXB = 6
ADVXE = 8
CHEMXB = 12
CHEMXE = 21
SYS_SIZE = 21

Lx = 1.0e-3  # m (from case)


def find_restart_timesteps(case_dir):
    """Find timesteps in restart_data (parallel_io: lustre_<step>/<step>_<rank>.dat)."""
    restart_base = case_dir / "restart_data"
    if not restart_base.exists():
        return []
    timesteps = []
    for d in sorted(restart_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("lustre_"):
            continue
        try:
            step = int(d.name.replace("lustre_", ""))
        except ValueError:
            continue
        # Check for at least one rank file (e.g. 0_0.dat for step 0)
        rank_files = list(d.glob(f"{step}_*.dat"))
        if rank_files:
            timesteps.append(step)
    return sorted(timesteps)


def read_restart_timestep(restart_base, step, num_cells=None):
    """
    Read one timestep from restart_data. For 1D single-rank, reads lustre_<step>/<step>_0.dat.
    Returns (x, data) where data shape is (sys_size, num_cells). Infers num_cells from file if not given.
    """
    d = restart_base / f"lustre_{step}"
    if not d.exists():
        return None, None
    rank_files = sorted(d.glob(f"{step}_*.dat"))
    if not rank_files:
        return None, None
    # Single-rank: one file
    fpath = rank_files[0]
    raw = np.fromfile(str(fpath), dtype=np.float64)
    n = len(raw)
    if n % SYS_SIZE != 0:
        print(f"  Warning: file size {n} not divisible by sys_size {SYS_SIZE}")
        return None, None
    nc = n // SYS_SIZE
    if num_cells is not None and nc != num_cells:
        print(f"  Warning: expected {num_cells} cells, got {nc}")
    data = raw.reshape((SYS_SIZE, nc))
    # Grid: assume 1D uniform
    x = np.linspace(0, Lx, nc)
    return x, data


def read_grid_1d(restart_base):
    """Read x_cb if available."""
    x_cb_path = restart_base / "x_cb.dat"
    if x_cb_path.exists():
        x = np.fromfile(str(x_cb_path), dtype=np.float64)
        if len(x) >= 2:
            x_cc = 0.5 * (x[:-1] + x[1:])
            return x_cc
    return None


def variable_diagnostics(data):
    """Return list of (idx, name, min, max, n_nan, n_inf) for each variable."""
    names = [
        "alpha_rho_1", "alpha_rho_2", "alpha_rho_3", "rho*u", "E",
        "alpha_1", "alpha_2", "alpha_3",
        "int_e_1", "int_e_2", "int_e_3",
        "Y1_rho", "Y2_rho", "Y3_rho", "Y4_rho", "Y5_rho",
        "Y6_rho", "Y7_rho", "Y8_rho", "Y9_rho", "Y10_rho",
    ]
    out = []
    for i in range(data.shape[0]):
        v = data[i, :]
        name = names[i] if i < len(names) else f"var_{i+1}"
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        v_valid = v[np.isfinite(v)]
        vmin = float(np.min(v_valid)) if len(v_valid) else np.nan
        vmax = float(np.max(v_valid)) if len(v_valid) else np.nan
        out.append((i + 1, name, vmin, vmax, n_nan, n_inf))
    return out


def plot_timestep(x, data, step, out_dir):
    """Plot key fields and mark NaNs. Save to out_dir."""
    alpha_1 = data[5, :]   # 0-indexed: 6th var
    alpha_2 = data[6, :]
    alpha_3 = data[7, :]
    alpha_rho_1 = data[0, :]
    alpha_rho_2 = data[1, :]
    alpha_rho_3 = data[2, :]
    rho_u = data[3, :]
    E = data[4, :]
    rho_total = alpha_rho_1 + alpha_rho_2 + alpha_rho_3

    x_mm = x * 1000.0  # mm

    # NaN/Inf mask (any variable)
    bad = ~np.isfinite(data)
    bad_any = np.any(bad, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Volume fractions
    ax = axes[0, 0]
    ax.plot(x_mm, alpha_1, 'b-', label=r'$\alpha_1$ (liquid)', lw=2)
    ax.plot(x_mm, alpha_2, 'r--', label=r'$\alpha_2$ (vapor)', lw=2)
    ax.plot(x_mm, alpha_3, 'g:', label=r'$\alpha_3$ (oxidizer)', lw=2)
    if np.any(bad_any):
        ax.scatter(x_mm[bad_any], np.zeros(bad_any.sum()), c='black', s=20, label='NaN/Inf', zorder=5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Volume fraction')
    ax.set_title('Volume fractions')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # 2) Densities and energy
    ax = axes[0, 1]
    ax.plot(x_mm, rho_total, 'k-', label=r'$\rho_{total}$', lw=2)
    ax.plot(x_mm, E, 'c-', label='E', lw=1.5)
    if np.any(bad_any):
        ax.scatter(x_mm[bad_any], np.zeros(bad_any.sum()), c='red', s=25, label='NaN/Inf', zorder=5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Value')
    ax.set_title('Total density & energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Partial densities (log)
    ax = axes[1, 0]
    eps = 1e-30
    ax.semilogy(x_mm, np.maximum(alpha_rho_1, eps), 'b-', label=r'$\alpha\rho_1$', lw=2)
    ax.semilogy(x_mm, np.maximum(alpha_rho_2, eps), 'r--', label=r'$\alpha\rho_2$', lw=2)
    ax.semilogy(x_mm, np.maximum(alpha_rho_3, eps), 'g:', label=r'$\alpha\rho_3$', lw=2)
    if np.any(bad_any):
        ax.scatter(x_mm[bad_any], np.full(bad_any.sum(), 1e-20), c='black', s=20, label='NaN/Inf', zorder=5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Partial density')
    ax.set_title('Partial densities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Which variable has NaNs (if any)
    ax = axes[1, 1]
    if np.any(bad):
        var_has_bad = np.any(bad, axis=1)
        ax.bar(np.where(var_has_bad)[0] + 1, np.sum(bad, axis=1)[var_has_bad], color='coral', edgecolor='darkred')
        ax.set_xlabel('Variable index')
        ax.set_ylabel('Count of NaN/Inf')
        ax.set_title('NaN/Inf per variable')
    else:
        ax.text(0.5, 0.5, 'No NaN/Inf in this timestep', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('NaN/Inf per variable')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Pre-crash diagnostic — timestep {step}', fontsize=14)
    plt.tight_layout()
    out_file = out_dir / f'pre_crash_t{step:04d}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_file}")


def get_case_dir_candidates(args_case_dir):
    """Build list of case directories to try."""
    candidates = []
    if args_case_dir is not None:
        candidates.append(Path(args_case_dir).resolve())
    candidates.append(DEFAULT_CASE_DIR)
    cwd = Path.cwd()
    if cwd not in candidates:
        candidates.append(cwd)
    repo_case = cwd / "examples" / "2D_burning_droplet"
    if repo_case.exists() and repo_case not in candidates:
        candidates.append(repo_case)
    return candidates


def main():
    ap = argparse.ArgumentParser(description='Visualize last saved state before crash')
    ap.add_argument('--steps', type=int, nargs='*', default=None,
                    help='Timesteps to plot (default: all found, then last 3)')
    ap.add_argument('--case-dir', type=str, default=None,
                    help='Case directory containing restart_data/ (default: script dir, then cwd)')
    args = ap.parse_args()

    print("=" * 60)
    print("Pre-crash diagnostic visualization")
    print("=" * 60)

    candidates = get_case_dir_candidates(args.case_dir)
    timesteps = []
    case_dir = None
    for cdir in candidates:
        timesteps = find_restart_timesteps(cdir)
        if timesteps:
            case_dir = cdir
            break
    if not timesteps:
        print("\nNo restart_data found in any of:")
        for c in candidates:
            print(f"  {c}")
        print("\nRun the simulation first (e.g. in WSL/Git Bash):")
        print("  ./mfc.sh run examples/2D_burning_droplet/test_phase2_validation.py -t pre_process simulation -j 4 -- --steps 101")
        print("\nRestart data is written under the case directory's restart_data/")
        return

    OUTPUT_DIR = case_dir / "figures" / "pre_crash"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nUsing case dir: {case_dir}")
    print(f"Found {len(timesteps)} timestep(s): {timesteps}")

    restart_base = case_dir / "restart_data"
    x_grid = read_grid_1d(restart_base)

    to_plot = args.steps if args.steps is not None else timesteps[-3:]  # last 3 by default
    to_plot = [t for t in to_plot if t in timesteps]
    if not to_plot:
        to_plot = timesteps[-1:]

    for step in to_plot:
        x, data = read_restart_timestep(restart_base, step)
        if data is None:
            print(f"  Could not read timestep {step}")
            continue
        if x_grid is not None and len(x_grid) == data.shape[1]:
            x = x_grid
        else:
            x = np.linspace(0, Lx, data.shape[1])

        print(f"\n--- Timestep {step} ---")
        print(f"  Shape: {data.shape[0]} variables x {data.shape[1]} cells")

        diag = variable_diagnostics(data)
        print("\n  Variable diagnostics (min, max, #NaN, #Inf):")
        for idx, name, vmin, vmax, n_nan, n_inf in diag:
            flag = ""
            if n_nan or n_inf:
                flag = "  <-- NaN/Inf!"
            print(f"    {idx:2d} {name:12s}  min={vmin:12.6e}  max={vmax:12.6e}  NaN={n_nan}  Inf={n_inf}{flag}")

        plot_timestep(x, data, step, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
