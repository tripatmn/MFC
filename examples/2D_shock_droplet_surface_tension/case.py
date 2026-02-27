#!/usr/bin/env python3
import math, json

# ============================================================
# User knobs
# ============================================================
Ma   = 1.4          # upstream Mach number (air)
gam_a = 1.4         # air gamma (ideal gas)

# Ambient / chamber state (from XML)
p1   = 101325.0     # Pa
rho1 = 1.22         # kg/m^3

# Water (kept from your example script)
rho_w = 1000.0
gam_w = 6.12
pi_w  = 3.43e8

# Geometry from XML
x_beg, x_end = 0.0, 0.15
y_beg, y_end = 0.0, 0.06
x_shock      = 0.14          # discontinuity location
droplet_xc   = 0.06
droplet_yc   = 0.03
R            = 0.0024        # droplet radius [m] from XML
D            = 2.0 * R

# Surface tension from XML
sigma = 72.8e-3     # N/m

# Numerics
cfl      = 0.4
time_end = 786.33e-6       # s (XML totalTime)
save_dt  = 3.93165e-6      # s (XML timeFreq)
eps = 1e-9

# Uniform-grid tradeoffs to mimic AMR
grid_mode = "mid"   # "base" | "mid" | "finest_equiv"

# ============================================================
# Helpers: normal shock (ideal gas), upstream at rest
# ============================================================
def normal_shock_post(M1, gamma, p1, rho1):
    a1 = math.sqrt(gamma * p1 / rho1)
    Us = M1 * a1
    p2_p1 = 1.0 + 2.0*gamma/(gamma+1.0) * (M1*M1 - 1.0)
    rho2_rho1 = ( (gamma+1.0)*M1*M1 ) / ( (gamma-1.0)*M1*M1 + 2.0 )
    p2   = p1   * p2_p1
    rho2 = rho1 * rho2_rho1
    u2 = Us * (1.0 - rho1/rho2)
    return p2, rho2, u2, a1

p2, rho2, u2, a1 = normal_shock_post(Ma, gam_a, p1, rho1)

# ============================================================
# Grid selection
# ============================================================
if grid_mode == "base":
    Nx, Ny = 3750, 1500
elif grid_mode == "mid":
    Nx, Ny = 7500, 3000
elif grid_mode == "finest_equiv":
    Nx, Ny = 15000, 6000
else:
    raise ValueError("grid_mode must be 'base', 'mid', or 'finest_equiv'")

dx = (x_end - x_beg) / Nx
a2 = math.sqrt(gam_a * p2 / rho2)
dt = cfl * dx / max(a1, a2 + abs(u2))
Nt = int(math.ceil(time_end / dt))
t_step_save = max(1, int(round(save_dt / dt)))

# Patch layout: 1=background, 2=shocked (left), 3=droplet (circle)
# Droplet at (0.06, 0.03) lies INSIDE shocked rectangle [0, 0.14]x[0, 0.06],
# so patch 3 must be allowed to overwrite BOTH patch 1 and patch 2.
shock_xc  = 0.5 * (x_beg + x_shock)
shock_Lx  = (x_shock - x_beg)

case = {
    "run_time_info": "F",
    "x_domain%beg": x_beg,
    "x_domain%end": x_end,
    "y_domain%beg": y_beg,
    "y_domain%end": y_end,
    "m": int(Nx),
    "n": int(Ny),
    "p": 0,
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": int(Nt),
    "t_step_save": int(t_step_save),
    "num_patches": 3,
    "model_eqns": 2,
    "alt_soundspeed": "F",
    "num_fluids": 2,
    "mpp_lim": "T",
    "mixture_err": "T",
    "time_stepper": 3,
    "recon_type": 2,
    "muscl_order": 2,
    "muscl_lim": 4,
    "int_comp": "T",
    "null_weights": "F",
    "riemann_solver": 2,
    "wave_speeds": 1,
    "avg_state": 2,
    "bc_x%beg": -6,
    "bc_x%end": -6,
    "bc_y%beg": -6,
    "bc_y%end": -6,
    "format": 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "parallel_io": "T",
    "surface_tension": "T",
    "sigma": sigma,

    # Patch 1: Background chamber (air)
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.5*(x_beg+x_end),
    "patch_icpp(1)%y_centroid": 0.5*(y_beg+y_end),
    "patch_icpp(1)%length_x": (x_end-x_beg),
    "patch_icpp(1)%length_y": (y_end-y_beg),
    "patch_icpp(1)%vel(1)": 0.0,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": p1,
    "patch_icpp(1)%alpha(1)": eps,
    "patch_icpp(1)%alpha(2)": 1.0 - eps,
    "patch_icpp(1)%alpha_rho(1)": eps * rho_w,
    "patch_icpp(1)%alpha_rho(2)": (1.0 - eps) * rho1,
    "patch_icpp(1)%cf_val": 0,

    # Patch 2: Shocked region (left of x_shock)
    "patch_icpp(2)%geometry": 3,
    "patch_icpp(2)%alter_patch(1)": "T",
    "patch_icpp(2)%x_centroid": shock_xc,
    "patch_icpp(2)%y_centroid": 0.5*(y_beg+y_end),
    "patch_icpp(2)%length_x": shock_Lx,
    "patch_icpp(2)%length_y": (y_end-y_beg),
    "patch_icpp(2)%vel(1)": u2,
    "patch_icpp(2)%vel(2)": 0.0,
    "patch_icpp(2)%pres": p2,
    "patch_icpp(2)%alpha(1)": eps,
    "patch_icpp(2)%alpha(2)": 1.0 - eps,
    "patch_icpp(2)%alpha_rho(1)": eps * rho_w,
    "patch_icpp(2)%alpha_rho(2)": (1.0 - eps) * rho2,
    "patch_icpp(2)%cf_val": 0,

    # Patch 3: Droplet (circle). Must overwrite BOTH 1 and 2 because
    # droplet center is inside the shocked rectangle.
    "patch_icpp(3)%geometry": 2,
    "patch_icpp(3)%alter_patch(1)": "T",
    "patch_icpp(3)%alter_patch(2)": "T",
    "patch_icpp(3)%x_centroid": droplet_xc,
    "patch_icpp(3)%y_centroid": droplet_yc,
    "patch_icpp(3)%radius": R,
    "patch_icpp(3)%vel(1)": 0.0,
    "patch_icpp(3)%vel(2)": 0.0,
    "patch_icpp(3)%pres": p1,
    "patch_icpp(3)%alpha(1)": 1.0 - eps,
    "patch_icpp(3)%alpha(2)": eps,
    "patch_icpp(3)%alpha_rho(1)": (1.0 - eps) * rho_w,
    "patch_icpp(3)%alpha_rho(2)": eps * rho1,
    "patch_icpp(3)%cf_val": 1,

    "fluid_pp(1)%gamma": 1.0 / (gam_w - 1.0),
    "fluid_pp(1)%pi_inf": pi_w * gam_w / (gam_w - 1.0),
    "fluid_pp(2)%gamma": 1.0 / (gam_a - 1.0),
    "fluid_pp(2)%pi_inf": 0.0,
}

print(json.dumps(case, indent=2))
