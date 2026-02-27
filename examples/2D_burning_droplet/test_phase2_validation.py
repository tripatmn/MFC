#!/usr/bin/env python3
"""
Phase 2 Validation Test Case: Enhanced Multiphase Chemistry Coupling

This test case validates the Phase 2 implementation improvements:
1. Proper gas-phase density averaging for reactions
2. Improved diffusion flux scaling by gas volume fraction
3. Gas-phase species conversion helper

Setup:
- 1D domain with liquid fuel on left, oxidizer gas on right
- Phase change enabled to allow evaporation at interface
- Chemistry enabled with multiphase coupling and diffusion

Expected behavior (Phase 2 improvements):
- Reactions use gas-phase density (not total density)
- Diffusion fluxes scale by gas volume fraction
- More accurate species transport at interfaces
"""

import json
import argparse

parser = argparse.ArgumentParser(
    prog="test_phase2_validation",
    description="Phase 2 validation test for enhanced multiphase chemistry coupling",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--no-chemistry", action="store_true",
                    help="Disable chemistry (phase change only)")
parser.add_argument("--no-diffusion", action="store_true",
                    help="Disable diffusion (reactions only)")
parser.add_argument("--no-reactions", action="store_true",
                    help="Disable reactions (diffusion only)")
parser.add_argument("--threshold", type=float, default=0.01,
                    help="Gas phase threshold for chemistry")
parser.add_argument("--steps", type=int, default=50,
                    help="Number of time steps to run")

# Parse known args only to allow MFC to pass additional arguments
args, _ = parser.parse_known_args()

# =============================================================================
# DOMAIN PARAMETERS
# =============================================================================
# 2D domain (thin in y) so ParaView can load the grid; x is the main flow direction
# n must be >= (num_stcls_min*weno_order - 1) = 5*3 - 1 = 14 for WENO stencil
Nx = 199
Ny = 14
Lx = 1.0e-3   # 1 mm in x
Ly = 0.2e-3   # 0.2 mm in y (thin slice for ParaView)

dx = Lx / Nx
dy = Ly / Ny

# =============================================================================
# FLUID PROPERTIES (Stiffened Gas EOS)
# =============================================================================
# Using simplified properties for water-like liquid and air-like gas

# Fluid 1: Liquid fuel (water-like for phase change compatibility)
gamma_l = 2.35
pi_inf_l = 1.0e9
cv_l = 1816.0
qv_l = -1167000.0
qvp_l = 0.0

# Fluid 2: Fuel vapor (water vapor-like)
gamma_v = 1.43
pi_inf_v = 0.0
cv_v = 1040.0
qv_v = 2030000.0
qvp_v = -23400.0

# Fluid 3: Oxidizer/Air
gamma_a = 1.4
pi_inf_a = 0.0
cv_a = 717.5
qv_a = 0.0
qvp_a = 0.0

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
# Temperature and pressure
T0 = 373.15  # K (100 C - at boiling point for water)
p0 = 1.01325e5  # Pa (1 atm)

# Compute densities from EOS: rho = (p + pi_inf) / ((gamma - 1) * cv * T)
rho_l = (p0 + pi_inf_l) / ((gamma_l - 1) * cv_l * T0)
rho_v = (p0 + pi_inf_v) / ((gamma_v - 1) * cv_v * T0)
rho_a = (p0 + pi_inf_a) / ((gamma_a - 1) * cv_a * T0)

# Velocity (initially at rest)
u0 = 0.0

# =============================================================================
# PATCH CONFIGURATION
# =============================================================================
# Patch 1: Background - oxidizer gas (entire domain initially)
# Patch 2: Liquid droplet (left portion of domain)

droplet_end = 0.3 * Lx  # Liquid occupies left 30% of domain

# =============================================================================
# CHEMISTRY CONFIGURATION
# =============================================================================
# For this validation, we use a simplified approach:
# - H2-O2 mechanism (h2o2.yaml)
# - Fuel species is H2 (index 1)
# - Oxidizer is O2 (index 4) with N2 diluent (index 10)

ctfile = "h2o2.yaml"
num_species = 10

# Species indices in h2o2.yaml:
# 1: H2, 2: H, 3: O, 4: O2, 5: OH, 6: H2O, 7: HO2, 8: H2O2, 9: AR, 10: N2
idx_H2 = 1
idx_O2 = 4
idx_N2 = 10

# Air composition (mass fractions)
Y_O2_air = 0.233
Y_N2_air = 0.767

# =============================================================================
# TIME STEPPING
# =============================================================================
dt = 1.0e-9  # 1 ns time step
t_stop = args.steps
# Save every step so that by 20% of the run we have 20 time steps written (e.g. 101 steps → 20 saves at 20%)
t_save = 1

# =============================================================================
# CASE DICTIONARY
# =============================================================================
case = {
    # -------------------------------------------------------------------------
    # Logistics
    # -------------------------------------------------------------------------
    "run_time_info": "T",
    
    # -------------------------------------------------------------------------
    # Domain (2D thin slice for ParaView)
    # -------------------------------------------------------------------------
    "m": Nx,
    "n": Ny,
    "p": 0,
    
    "x_domain%beg": 0.0,
    "x_domain%end": Lx,
    "y_domain%beg": 0.0,
    "y_domain%end": Ly,
    
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": t_stop,
    "t_step_save": t_save,
    
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    "model_eqns": 3,  # 6-equation model for phase change
    "num_fluids": 3,
    "num_patches": 2,
    "mpp_lim": "T",
    "mixture_err": "T",
    "time_stepper": 3,  # 3rd order TVD RK
    
    # -------------------------------------------------------------------------
    # Numerics (MUSCL instead of WENO for robustness; previously got to ~20% with 1D WENO)
    # -------------------------------------------------------------------------
    "recon_type": 2,       # 2 = MUSCL
    "weno_order": 0,      # not used when recon_type=2
    "weno_eps": 1.0e-6,   # required by validator when weno_order != 1; ignored for MUSCL
    "muscl_order": 2,
    "muscl_lim": 3,       # 3 = Van Albada (smooth limiter)
    "riemann_solver": 2,  # HLLC
    "wave_speeds": 1,
    "avg_state": 2,
    
    # -------------------------------------------------------------------------
    # Boundary Conditions
    # -------------------------------------------------------------------------
    "bc_x%beg": -3,  # Reflective
    "bc_x%end": -3,  # Reflective
    "bc_y%beg": -1,  # Periodic (thin slice, quasi-1D)
    "bc_y%end": -1,  # Periodic

    # -------------------------------------------------------------------------
    # Phase Change
    # -------------------------------------------------------------------------
    "relax": "T",
    "relax_model": 6,  # pTg relaxation
    "palpha_eps": 1.0e-2,
    "ptgalpha_eps": 1.0e-2,
    
    # -------------------------------------------------------------------------
    # Patch 1: Background - Oxidizer gas (entire domain)
    # -------------------------------------------------------------------------
    "patch_icpp(1)%geometry": 3,  # Rectangle (2D; line segment geometry 1 requires n=0)
    "patch_icpp(1)%x_centroid": Lx / 2,
    "patch_icpp(1)%length_x": Lx,
    "patch_icpp(1)%y_centroid": Ly / 2,
    "patch_icpp(1)%length_y": Ly,

    "patch_icpp(1)%vel(1)": u0,
    "patch_icpp(1)%vel(2)": u0,
    "patch_icpp(1)%pres": p0,
    
    # Volume fractions: pure oxidizer gas
    "patch_icpp(1)%alpha(1)": 1.0e-8,  # Trace liquid
    "patch_icpp(1)%alpha(2)": 1.0e-8,  # Trace vapor
    "patch_icpp(1)%alpha(3)": 1.0 - 2.0e-8,  # Oxidizer
    
    # Partial densities
    "patch_icpp(1)%alpha_rho(1)": 1.0e-8 * rho_l,
    "patch_icpp(1)%alpha_rho(2)": 1.0e-8 * rho_v,
    "patch_icpp(1)%alpha_rho(3)": (1.0 - 2.0e-8) * rho_a,
    
    # -------------------------------------------------------------------------
    # Patch 2: Liquid droplet (left portion)
    # -------------------------------------------------------------------------
    "patch_icpp(2)%geometry": 3,  # Rectangle (2D)
    "patch_icpp(2)%x_centroid": droplet_end / 2,
    "patch_icpp(2)%length_x": droplet_end,
    "patch_icpp(2)%y_centroid": Ly / 2,
    "patch_icpp(2)%length_y": Ly,
    "patch_icpp(2)%alter_patch(1)": "T",

    "patch_icpp(2)%vel(1)": u0,
    "patch_icpp(2)%vel(2)": u0,
    "patch_icpp(2)%pres": p0,
    
    # Volume fractions: mostly liquid
    "patch_icpp(2)%alpha(1)": 1.0 - 2.0e-8,  # Liquid
    "patch_icpp(2)%alpha(2)": 1.0e-8,  # Trace vapor
    "patch_icpp(2)%alpha(3)": 1.0e-8,  # Trace oxidizer
    
    # Partial densities
    "patch_icpp(2)%alpha_rho(1)": (1.0 - 2.0e-8) * rho_l,
    "patch_icpp(2)%alpha_rho(2)": 1.0e-8 * rho_v,
    "patch_icpp(2)%alpha_rho(3)": 1.0e-8 * rho_a,
    
    # -------------------------------------------------------------------------
    # Fluid Properties
    # -------------------------------------------------------------------------
    # Fluid 1: Liquid
    "fluid_pp(1)%gamma": 1.0 / (gamma_l - 1),
    "fluid_pp(1)%pi_inf": gamma_l * pi_inf_l / (gamma_l - 1),
    "fluid_pp(1)%cv": cv_l,
    "fluid_pp(1)%qv": qv_l,
    "fluid_pp(1)%qvp": qvp_l,
    
    # Fluid 2: Vapor
    "fluid_pp(2)%gamma": 1.0 / (gamma_v - 1),
    "fluid_pp(2)%pi_inf": gamma_v * pi_inf_v / (gamma_v - 1),
    "fluid_pp(2)%cv": cv_v,
    "fluid_pp(2)%qv": qv_v,
    "fluid_pp(2)%qvp": qvp_v,
    
    # Fluid 3: Oxidizer/Air
    "fluid_pp(3)%gamma": 1.0 / (gamma_a - 1),
    "fluid_pp(3)%pi_inf": gamma_a * pi_inf_a / (gamma_a - 1),
    "fluid_pp(3)%cv": cv_a,
    "fluid_pp(3)%qv": qv_a,
    "fluid_pp(3)%qvp": qvp_a,
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    "format": 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "parallel_io": "T",
    
    # Conservative variables
    "cons_vars_wrt": "T",
    
    # Volume fractions
    "alpha_wrt(1)": "T",
    "alpha_wrt(2)": "T", 
    "alpha_wrt(3)": "T",
}

# =============================================================================
# CHEMISTRY CONFIGURATION
# =============================================================================
if not args.no_chemistry:
    case["chemistry"] = "T"
    case["chem_params%diffusion"] = "F" if args.no_diffusion else "T"
    case["chem_params%reactions"] = "F" if args.no_reactions else "T"
    case["chem_params%transport_model"] = 2  # Unity Lewis number
    case["cantera_file"] = ctfile
    case["chem_wrt_T"] = "T"
    
    # Multiphase chemistry coupling (Phase 1 + Phase 2)
    case["chem_params%multiphase"] = "T"
    case["chem_params%liquid_phase_idx"] = 1
    case["chem_params%fuel_species_idx"] = idx_H2
    case["chem_params%gas_phase_threshold"] = args.threshold
    
    # Species mass fractions for Patch 1 (oxidizer gas)
    for i in range(1, num_species + 1):
        case[f"patch_icpp(1)%Y({i})"] = 0.0
    case[f"patch_icpp(1)%Y({idx_O2})"] = Y_O2_air
    case[f"patch_icpp(1)%Y({idx_N2})"] = Y_N2_air
    
    # Species mass fractions for Patch 2 (liquid - no species initially)
    # When liquid evaporates, mass goes to fuel species automatically
    for i in range(1, num_species + 1):
        case[f"patch_icpp(2)%Y({i})"] = 0.0

# =============================================================================
# OUTPUT
# =============================================================================
if __name__ == "__main__":
    print(json.dumps(case))
