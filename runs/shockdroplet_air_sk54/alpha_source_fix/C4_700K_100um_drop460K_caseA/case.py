#!/usr/bin/env python3
"""Case A: C4-derived Mach-3 SK54 shock/dodecane droplet case.

Diagnostic change relative to C4:
  - upstream air temperature is raised to 700 K;
  - the 100 um liquid droplet is initialized EOS-consistently at 460 K
    by recomputing the liquid alpha_rho from the stiffened-gas EOS;
  - upstream and post-shock gas states are recomputed from Mach 3
    normal-shock relations, not by independently prescribing p2/u2/rho2.

The C4 domain, 100 um droplet geometry, chemistry, diffusion, phase change,
boundary conditions, numerics, output cadence, and restart-style controls are
otherwise preserved.
"""

import json

MS = 3.0
P1 = 101325.0
T1 = 700.0
T_DROP_LIQ = 460.0
T_STOP = 2.0e-6

X_MIN = 0.0
X_MAX = 0.0055
Y_MIN = 0.0
Y_MAX = 0.0006000000000000001
X_SHOCK = 0.0004
X_DROP = 0.0005
Y_DROP = 0.00030000000000000003
D_DROP = 100.0e-6
R_DROP = 0.5*D_DROP
M_CELLS = 1176
N_CELLS = 128

ALPHA_TRACE = 1.5e-05
ALPHA_AIR = 0.99997
ALPHA_LIQ_DROP = 0.99997

Y_N2 = 0.7670907820415769
Y_O2 = 0.2329092179584231
MW_N2 = 0.028014
MW_O2 = 0.031998
R_UNIVERSAL = 8.31446261815324
MW_AIR = 1.0/(Y_N2/MW_N2 + Y_O2/MW_O2)
R_AIR = R_UNIVERSAL/MW_AIR

# MFC stores the stiffened-gas Gamma parameter.  The ideal-gas air gamma used
# in the normal-shock relations is 1 + 1/Gamma.
FLUID3_GAMMA_PARAM = 2.5000000000000004
GAMMA_AIR = 1.0 + 1.0/FLUID3_GAMMA_PARAM

# Preserve the C4 trace-vapor convention by inferring its gas constant from
# the original upstream trace-vapor density and inferred C4 air temperature.
C4_ALPHA_RHO_AIR_UPSTREAM = 0.7170860221179817
C4_ALPHA_RHO_VAP_UPSTREAM = 6.502361578150291e-05
C4_RHO_AIR_UPSTREAM = C4_ALPHA_RHO_AIR_UPSTREAM/ALPHA_AIR
C4_T1_BASE = P1/(C4_RHO_AIR_UPSTREAM*R_AIR)
C4_RHO_VAP_UPSTREAM = C4_ALPHA_RHO_VAP_UPSTREAM/ALPHA_TRACE
R_VAP = P1/(C4_RHO_VAP_UPSTREAM*C4_T1_BASE)

FLUID1_GAMMA_PARAM = 0.7407407407407407
FLUID1_PI_INF = 835555555.5555555
FLUID1_CV = 1077.0


def normal_shock_state(mach, gamma, p1, t1, r_air, r_vap):
    p2_p1 = 1.0 + 2.0*gamma/(gamma + 1.0)*(mach*mach - 1.0)
    rho2_rho1 = ((gamma + 1.0)*mach*mach)/((gamma - 1.0)*mach*mach + 2.0)
    t2_t1 = p2_p1/rho2_rho1
    p2 = p1*p2_p1
    t2 = t1*t2_t1
    rho_air_1 = p1/(r_air*t1)
    rho_air_2 = rho_air_1*rho2_rho1
    rho_vap_1 = p1/(r_vap*t1)
    rho_vap_2 = p2/(r_vap*t2)
    shock_speed = mach*(gamma*r_air*t1)**0.5
    u2 = shock_speed*(1.0 - 1.0/rho2_rho1)
    return p2_p1, rho2_rho1, t2_t1, p2, t2, rho_air_1, rho_air_2, rho_vap_1, rho_vap_2, u2


def stiffened_density_from_p_t(pres, temp, gamma_param, pi_inf, cv):
    gs_min = 1.0 + 1.0/gamma_param
    ps_inf = pi_inf/(1.0 + gamma_param)
    return (pres + ps_inf)/((gs_min - 1.0)*cv*temp)


p2_p1, rho2_rho1, T2_T1, P2, T2, rho_air_1, rho_air_2, rho_vap_1, rho_vap_2, u2 = normal_shock_state(
    MS, GAMMA_AIR, P1, T1, R_AIR, R_VAP
)
rho_liq_drop = stiffened_density_from_p_t(P1, T_DROP_LIQ, FLUID1_GAMMA_PARAM, FLUID1_PI_INF, FLUID1_CV)

case = {
    'chemistry': 'T',
    'cantera_file': 'examples/chemistry_mechanisms/yao_sk54/yao_sk54.yaml',
    'cantera_phase': 'yao_sk54',
    'x_domain%beg': X_MIN,
    'x_domain%end': X_MAX,
    'y_domain%beg': Y_MIN,
    'y_domain%end': Y_MAX,
    'm': M_CELLS,
    'n': N_CELLS,
    'p': 0,
    'cfl_adap_dt': 'T',
    'n_start': 0,
    'num_patches': 3,
    'model_eqns': 3,
    'num_fluids': 3,
    'mpp_lim': 'T',
    'relax': 'T',
    'relax_model': 6,
    'palpha_eps': 3e-05,
    'ptgalpha_eps': 3e-05,
    'chem_gas_fluid_id': 2,
    'chem_gas_num_fluids': 2,
    'chem_gas_fluid_ids(1)': 2,
    'chem_gas_fluid_ids(2)': 3,
    'fuel_species_id': 46,
    'evap_species_source': 'T',
    'evap_species_src': 0.0,
    'evap_liquid_fluid_id': 1,
    'evap_alpha_thresh': 0.01,
    'evap_species_alpha_min': 0.0001,
    'evap_species_mass_min': 1e-06,
    'evap_species_liq_max': 0.99,
    'chem_reaction_heat_enable': 'T',
    'chem_reaction_heat_limit_frac': 0.05,
    'chem_reaction_heat_diag': 'F',
    'recon_type': 2,
    'muscl_order': 2,
    'weno_order': 3,
    'bc_x%beg': -6,
    'bc_x%end': -6,
    'bc_y%beg': -3,
    'bc_y%end': -3,
    'precision': 2,
    'parallel_io': 'F',
    'fluid_pp(1)%gamma': 0.7407407407407407,
    'fluid_pp(1)%pi_inf': 835555555.5555555,
    'fluid_pp(1)%cv': 1077.0,
    'fluid_pp(1)%qv': -758060.0,
    'fluid_pp(1)%qvp': 0.0,
    'fluid_pp(2)%gamma': 40.00000000000014,
    'fluid_pp(2)%pi_inf': 0.0,
    'fluid_pp(2)%cv': 1956.0,
    'fluid_pp(2)%qv': -234490.0,
    'fluid_pp(2)%qvp': -24625.0,
    'fluid_pp(3)%gamma': 2.5000000000000004,
    'fluid_pp(3)%pi_inf': 0.0,
    'fluid_pp(3)%cv': 739.0,
    'fluid_pp(3)%qv': 0.0,
    'fluid_pp(3)%qvp': 0.0,
    'patch_icpp(1)%geometry': 3,
    'patch_icpp(1)%x_centroid': 0.5*(X_SHOCK + X_MAX),
    'patch_icpp(1)%y_centroid': Y_DROP,
    'patch_icpp(1)%length_x': round(X_MAX - X_SHOCK, 15),
    'patch_icpp(1)%length_y': Y_MAX - Y_MIN,
    'patch_icpp(1)%vel(1)': 0.0,
    'patch_icpp(1)%vel(2)': 0.0,
    'patch_icpp(1)%pres': P1,
    'patch_icpp(1)%alpha(1)': ALPHA_TRACE,
    'patch_icpp(1)%alpha(2)': ALPHA_TRACE,
    'patch_icpp(1)%alpha(3)': ALPHA_AIR,
    'patch_icpp(1)%alpha_rho(1)': ALPHA_TRACE*rho_liq_drop,
    'patch_icpp(1)%alpha_rho(2)': ALPHA_TRACE*rho_vap_1,
    'patch_icpp(1)%alpha_rho(3)': ALPHA_AIR*rho_air_1,
    'patch_icpp(2)%geometry': 3,
    'patch_icpp(2)%alter_patch(1)': 'T',
    'patch_icpp(2)%x_centroid': 0.5*(X_MIN + X_SHOCK),
    'patch_icpp(2)%y_centroid': Y_DROP,
    'patch_icpp(2)%length_x': round(X_SHOCK - X_MIN, 15),
    'patch_icpp(2)%length_y': Y_MAX - Y_MIN,
    'patch_icpp(2)%vel(1)': u2,
    'patch_icpp(2)%vel(2)': 0.0,
    'patch_icpp(2)%pres': P2,
    'patch_icpp(2)%alpha(1)': ALPHA_TRACE,
    'patch_icpp(2)%alpha(2)': ALPHA_TRACE,
    'patch_icpp(2)%alpha(3)': ALPHA_AIR,
    'patch_icpp(2)%alpha_rho(1)': ALPHA_TRACE*rho_liq_drop,
    'patch_icpp(2)%alpha_rho(2)': ALPHA_TRACE*rho_vap_2,
    'patch_icpp(2)%alpha_rho(3)': ALPHA_AIR*rho_air_2,
    'patch_icpp(3)%geometry': 2,
    'patch_icpp(3)%alter_patch(1)': 'T',
    'patch_icpp(3)%x_centroid': X_DROP,
    'patch_icpp(3)%y_centroid': Y_DROP,
    'patch_icpp(3)%radius': R_DROP,
    'patch_icpp(3)%vel(1)': 0.0,
    'patch_icpp(3)%vel(2)': 0.0,
    'patch_icpp(3)%pres': P1,
    'patch_icpp(3)%alpha(1)': ALPHA_LIQ_DROP,
    'patch_icpp(3)%alpha(2)': ALPHA_TRACE,
    'patch_icpp(3)%alpha(3)': ALPHA_TRACE,
    'patch_icpp(3)%alpha_rho(1)': ALPHA_LIQ_DROP*rho_liq_drop,
    'patch_icpp(3)%alpha_rho(2)': ALPHA_TRACE*rho_vap_1,
    'patch_icpp(3)%alpha_rho(3)': ALPHA_TRACE*rho_air_1,
    'patch_icpp(1)%Y(1)': 0.7670907820415769,
    'patch_icpp(2)%Y(1)': 0.7670907820415769,
    'patch_icpp(1)%Y(2)': 0.0,
    'patch_icpp(2)%Y(2)': 0.0,
    'patch_icpp(1)%Y(3)': 0.0,
    'patch_icpp(2)%Y(3)': 0.0,
    'patch_icpp(1)%Y(4)': 0.0,
    'patch_icpp(2)%Y(4)': 0.0,
    'patch_icpp(1)%Y(5)': 0.0,
    'patch_icpp(2)%Y(5)': 0.0,
    'patch_icpp(1)%Y(6)': 0.0,
    'patch_icpp(2)%Y(6)': 0.0,
    'patch_icpp(1)%Y(7)': 0.0,
    'patch_icpp(2)%Y(7)': 0.0,
    'patch_icpp(1)%Y(8)': 0.0,
    'patch_icpp(2)%Y(8)': 0.0,
    'patch_icpp(1)%Y(9)': 0.0,
    'patch_icpp(2)%Y(9)': 0.0,
    'patch_icpp(1)%Y(10)': 0.2329092179584231,
    'patch_icpp(2)%Y(10)': 0.2329092179584231,
    'patch_icpp(1)%Y(11)': 0.0,
    'patch_icpp(2)%Y(11)': 0.0,
    'patch_icpp(1)%Y(12)': 0.0,
    'patch_icpp(2)%Y(12)': 0.0,
    'patch_icpp(1)%Y(13)': 0.0,
    'patch_icpp(2)%Y(13)': 0.0,
    'patch_icpp(1)%Y(14)': 0.0,
    'patch_icpp(2)%Y(14)': 0.0,
    'patch_icpp(1)%Y(15)': 0.0,
    'patch_icpp(2)%Y(15)': 0.0,
    'patch_icpp(1)%Y(16)': 0.0,
    'patch_icpp(2)%Y(16)': 0.0,
    'patch_icpp(1)%Y(17)': 0.0,
    'patch_icpp(2)%Y(17)': 0.0,
    'patch_icpp(1)%Y(18)': 0.0,
    'patch_icpp(2)%Y(18)': 0.0,
    'patch_icpp(1)%Y(19)': 0.0,
    'patch_icpp(2)%Y(19)': 0.0,
    'patch_icpp(1)%Y(20)': 0.0,
    'patch_icpp(2)%Y(20)': 0.0,
    'patch_icpp(1)%Y(21)': 0.0,
    'patch_icpp(2)%Y(21)': 0.0,
    'patch_icpp(1)%Y(22)': 0.0,
    'patch_icpp(2)%Y(22)': 0.0,
    'patch_icpp(1)%Y(23)': 0.0,
    'patch_icpp(2)%Y(23)': 0.0,
    'patch_icpp(1)%Y(24)': 0.0,
    'patch_icpp(2)%Y(24)': 0.0,
    'patch_icpp(1)%Y(25)': 0.0,
    'patch_icpp(2)%Y(25)': 0.0,
    'patch_icpp(1)%Y(26)': 0.0,
    'patch_icpp(2)%Y(26)': 0.0,
    'patch_icpp(1)%Y(27)': 0.0,
    'patch_icpp(2)%Y(27)': 0.0,
    'patch_icpp(1)%Y(28)': 0.0,
    'patch_icpp(2)%Y(28)': 0.0,
    'patch_icpp(1)%Y(29)': 0.0,
    'patch_icpp(2)%Y(29)': 0.0,
    'patch_icpp(1)%Y(30)': 0.0,
    'patch_icpp(2)%Y(30)': 0.0,
    'patch_icpp(1)%Y(31)': 0.0,
    'patch_icpp(2)%Y(31)': 0.0,
    'patch_icpp(1)%Y(32)': 0.0,
    'patch_icpp(2)%Y(32)': 0.0,
    'patch_icpp(1)%Y(33)': 0.0,
    'patch_icpp(2)%Y(33)': 0.0,
    'patch_icpp(1)%Y(34)': 0.0,
    'patch_icpp(2)%Y(34)': 0.0,
    'patch_icpp(1)%Y(35)': 0.0,
    'patch_icpp(2)%Y(35)': 0.0,
    'patch_icpp(1)%Y(36)': 0.0,
    'patch_icpp(2)%Y(36)': 0.0,
    'patch_icpp(1)%Y(37)': 0.0,
    'patch_icpp(2)%Y(37)': 0.0,
    'patch_icpp(1)%Y(38)': 0.0,
    'patch_icpp(2)%Y(38)': 0.0,
    'patch_icpp(1)%Y(39)': 0.0,
    'patch_icpp(2)%Y(39)': 0.0,
    'patch_icpp(1)%Y(40)': 0.0,
    'patch_icpp(2)%Y(40)': 0.0,
    'patch_icpp(1)%Y(41)': 0.0,
    'patch_icpp(2)%Y(41)': 0.0,
    'patch_icpp(1)%Y(42)': 0.0,
    'patch_icpp(2)%Y(42)': 0.0,
    'patch_icpp(1)%Y(43)': 0.0,
    'patch_icpp(2)%Y(43)': 0.0,
    'patch_icpp(1)%Y(44)': 0.0,
    'patch_icpp(2)%Y(44)': 0.0,
    'patch_icpp(1)%Y(45)': 0.0,
    'patch_icpp(2)%Y(45)': 0.0,
    'patch_icpp(1)%Y(46)': 0.0,
    'patch_icpp(2)%Y(46)': 0.0,
    'patch_icpp(1)%Y(47)': 0.0,
    'patch_icpp(2)%Y(47)': 0.0,
    'patch_icpp(1)%Y(48)': 0.0,
    'patch_icpp(2)%Y(48)': 0.0,
    'patch_icpp(1)%Y(49)': 0.0,
    'patch_icpp(2)%Y(49)': 0.0,
    'patch_icpp(1)%Y(50)': 0.0,
    'patch_icpp(2)%Y(50)': 0.0,
    'patch_icpp(1)%Y(51)': 0.0,
    'patch_icpp(2)%Y(51)': 0.0,
    'patch_icpp(1)%Y(52)': 0.0,
    'patch_icpp(2)%Y(52)': 0.0,
    'patch_icpp(1)%Y(53)': 0.0,
    'patch_icpp(2)%Y(53)': 0.0,
    'patch_icpp(1)%Y(54)': 0.0,
    'patch_icpp(2)%Y(54)': 0.0,
    'patch_icpp(3)%Y(1)': 0.7268792403511868,
    'patch_icpp(3)%Y(2)': 0.0,
    'patch_icpp(3)%Y(3)': 0.0,
    'patch_icpp(3)%Y(4)': 0.0,
    'patch_icpp(3)%Y(5)': 0.0,
    'patch_icpp(3)%Y(6)': 0.0,
    'patch_icpp(3)%Y(7)': 0.0,
    'patch_icpp(3)%Y(8)': 0.0,
    'patch_icpp(3)%Y(9)': 0.0,
    'patch_icpp(3)%Y(10)': 0.2168174236910317,
    'patch_icpp(3)%Y(11)': 0.0,
    'patch_icpp(3)%Y(12)': 0.0,
    'patch_icpp(3)%Y(13)': 0.0,
    'patch_icpp(3)%Y(14)': 0.0,
    'patch_icpp(3)%Y(15)': 0.0,
    'patch_icpp(3)%Y(16)': 0.0,
    'patch_icpp(3)%Y(17)': 0.0,
    'patch_icpp(3)%Y(18)': 0.0,
    'patch_icpp(3)%Y(19)': 0.0,
    'patch_icpp(3)%Y(20)': 0.0,
    'patch_icpp(3)%Y(21)': 0.0,
    'patch_icpp(3)%Y(22)': 0.0,
    'patch_icpp(3)%Y(23)': 0.0,
    'patch_icpp(3)%Y(24)': 0.0,
    'patch_icpp(3)%Y(25)': 0.0,
    'patch_icpp(3)%Y(26)': 0.0,
    'patch_icpp(3)%Y(27)': 0.0,
    'patch_icpp(3)%Y(28)': 0.0,
    'patch_icpp(3)%Y(29)': 0.0,
    'patch_icpp(3)%Y(30)': 0.0,
    'patch_icpp(3)%Y(31)': 0.0,
    'patch_icpp(3)%Y(32)': 0.0,
    'patch_icpp(3)%Y(33)': 0.0,
    'patch_icpp(3)%Y(34)': 0.0,
    'patch_icpp(3)%Y(35)': 0.0,
    'patch_icpp(3)%Y(36)': 0.0,
    'patch_icpp(3)%Y(37)': 0.0,
    'patch_icpp(3)%Y(38)': 0.0,
    'patch_icpp(3)%Y(39)': 0.0,
    'patch_icpp(3)%Y(40)': 0.0,
    'patch_icpp(3)%Y(41)': 0.0,
    'patch_icpp(3)%Y(42)': 0.0,
    'patch_icpp(3)%Y(43)': 0.0,
    'patch_icpp(3)%Y(44)': 0.0,
    'patch_icpp(3)%Y(45)': 0.0,
    'patch_icpp(3)%Y(46)': 0.05630333595778143,
    'patch_icpp(3)%Y(47)': 0.0,
    'patch_icpp(3)%Y(48)': 0.0,
    'patch_icpp(3)%Y(49)': 0.0,
    'patch_icpp(3)%Y(50)': 0.0,
    'patch_icpp(3)%Y(51)': 0.0,
    'patch_icpp(3)%Y(52)': 0.0,
    'patch_icpp(3)%Y(53)': 0.0,
    'patch_icpp(3)%Y(54)': 0.0,
    'old_ic': 'F',
    'old_grid': 'F',
    't_step_old': 0,
    't_step_start': 0,
    'run_time_info': 'T',
    'dt': 3.413872806789289e-11,
    'cfl_target': 0.025,
    't_stop': T_STOP,
    't_save': 5e-08,
    't_step_print': 50,
    'alt_soundspeed': 'F',
    'mixture_err': 'T',
    'chem_params%diffusion': 'T',
    'chem_params%reactions': 'T',
    'time_stepper': 3,
    'muscl_lim': 1,
    'int_comp': 'F',
    'weno_eps': 1e-16,
    'weno_Re_flux': 'F',
    'weno_avg': 'F',
    'mapped_weno': 'T',
    'null_weights': 'F',
    'mp_weno': 'F',
    'riemann_solver': 2,
    'wave_speeds': 1,
    'avg_state': 2,
    'prim_vars_wrt': 'T',
    'chem_params%transport_model': 1,
    'format': 1,
    'alpha_rho_wrt(1)': 'T',
    'alpha_wrt(1)': 'T',
    'alpha_rho_wrt(2)': 'T',
    'alpha_wrt(2)': 'T',
    'alpha_rho_wrt(3)': 'T',
    'alpha_wrt(3)': 'T',
    'chem_wrt_Y(46)': 'T',
    'chem_wrt_Y(10)': 'T',
    'chem_wrt_Y(1)': 'T',
    'chem_wrt_Y(19)': 'T',
    'chem_wrt_Y(8)': 'T',
    'chem_wrt_Y(5)': 'T',
    'chem_wrt_Y(6)': 'T',
    'chem_wrt_Y(9)': 'T',
    'cons_vars_wrt': 'T',
    'pres_wrt': 'T',
    'chem_wrt_T': 'T',
}

if __name__ == "__main__":
    print(json.dumps(case))
