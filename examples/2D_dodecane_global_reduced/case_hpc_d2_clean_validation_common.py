"""Shared clean dodecane grid-validation case builders."""


def build_evap_validation_case(nx):
    ny = nx

    r_drop = 40.0e-6
    domain = 8.0*r_drop
    dx = domain/nx

    p0 = 2.0e5
    t_hot = 900.0
    rho_l = 669.43
    rho_v_hot = p0/((1.025 - 1.0)*1956.0*t_hot)
    rho_air_hot = p0/((1.4 - 1.0)*739.0*t_hot)

    eps = 1.0e-6
    c_liq = (2.35*(p0 + 4.8e8)/rho_l)**0.5
    dt = 0.75*0.25*dx/c_liq
    scale = nx/128.0

    alpha_l_bg = eps
    alpha_v_bg = eps
    alpha_a_bg = 1.0 - alpha_l_bg - alpha_v_bg

    alpha_l_drop = 1.0 - 2.0*eps
    alpha_v_drop = eps
    alpha_a_drop = eps

    case = {
        "run_time_info": "T",
        "x_domain%beg": -0.5*domain,
        "x_domain%end": 0.5*domain,
        "y_domain%beg": -0.5*domain,
        "y_domain%end": 0.5*domain,
        "m": nx,
        "n": ny,
        "p": 0,
        "dt": dt,
        "t_step_start": 0,
        "t_step_stop": int(round(3400*scale)),
        "t_step_save": int(round(50*scale)),
        "t_step_print": int(round(100*scale)),
        "num_patches": 2,
        "model_eqns": 3,
        "num_fluids": 3,
        "mpp_lim": "T",
        "mixture_err": "T",
        "relax": "T",
        "relax_ic": "F",
        "relax_model": 6,
        "palpha_eps": eps,
        "ptgalpha_eps": eps,
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
        "parallel_io": "F",
        "cons_vars_wrt": "T",
        "prim_vars_wrt": "T",
        "pres_wrt": "T",
        "fluid_pp(1)%gamma": 1.0/(2.35 - 1.0),
        "fluid_pp(1)%pi_inf": 2.35*4.8e8/(2.35 - 1.0),
        "fluid_pp(1)%cv": 1077.0,
        "fluid_pp(1)%qv": -758060.0,
        "fluid_pp(1)%qvp": 0.0,
        "fluid_pp(2)%gamma": 1.0/(1.025 - 1.0),
        "fluid_pp(2)%pi_inf": 0.0,
        "fluid_pp(2)%cv": 1956.0,
        "fluid_pp(2)%qv": -234490.0,
        "fluid_pp(2)%qvp": -24625.0,
        "fluid_pp(3)%gamma": 1.0/(1.4 - 1.0),
        "fluid_pp(3)%pi_inf": 0.0,
        "fluid_pp(3)%cv": 739.0,
        "fluid_pp(3)%qv": 0.0,
        "fluid_pp(3)%qvp": 0.0,
        "patch_icpp(1)%geometry": 3,
        "patch_icpp(1)%x_centroid": 0.0,
        "patch_icpp(1)%y_centroid": 0.0,
        "patch_icpp(1)%length_x": domain,
        "patch_icpp(1)%length_y": domain,
        "patch_icpp(1)%vel(1)": 0.0,
        "patch_icpp(1)%vel(2)": 0.0,
        "patch_icpp(1)%pres": p0,
        "patch_icpp(1)%alpha_rho(1)": alpha_l_bg*rho_l,
        "patch_icpp(1)%alpha_rho(2)": alpha_v_bg*rho_v_hot,
        "patch_icpp(1)%alpha_rho(3)": alpha_a_bg*rho_air_hot,
        "patch_icpp(1)%alpha(1)": alpha_l_bg,
        "patch_icpp(1)%alpha(2)": alpha_v_bg,
        "patch_icpp(1)%alpha(3)": alpha_a_bg,
        "patch_icpp(2)%geometry": 2,
        "patch_icpp(2)%alter_patch(1)": "T",
        "patch_icpp(2)%x_centroid": 0.0,
        "patch_icpp(2)%y_centroid": 0.0,
        "patch_icpp(2)%radius": r_drop,
        "patch_icpp(2)%vel(1)": 0.0,
        "patch_icpp(2)%vel(2)": 0.0,
        "patch_icpp(2)%pres": p0,
        "patch_icpp(2)%alpha_rho(1)": alpha_l_drop*rho_l,
        "patch_icpp(2)%alpha_rho(2)": alpha_v_drop*rho_v_hot,
        "patch_icpp(2)%alpha_rho(3)": alpha_a_drop*rho_air_hot,
        "patch_icpp(2)%alpha(1)": alpha_l_drop,
        "patch_icpp(2)%alpha(2)": alpha_v_drop,
        "patch_icpp(2)%alpha(3)": alpha_a_drop,
    }

    for i in range(1, 4):
        case[f"alpha_rho_wrt({i})"] = "T"
        case[f"alpha_wrt({i})"] = "T"

    return case


def add_baseline_burning(case, cantera_file):
    case.update({
        "chemistry": "T",
        "chem_params%diffusion": "F",
        "chem_params%reactions": "T",
        "cantera_file": str(cantera_file),
        "cantera_phase": "gas",
        "chem_gas_fluid_id": 2,
        "chem_gas_num_fluids": 2,
        "chem_gas_fluid_ids(1)": 2,
        "chem_gas_fluid_ids(2)": 3,
        "fuel_species_id": 1,
        "evap_species_source": "T",
        "evap_species_src": 0.0,
        "evap_liquid_fluid_id": 1,
        "evap_alpha_thresh": 0.01,
        "chem_reaction_heat_enable": "T",
        "chem_reaction_heat_limit_frac": 0.05,
        "chem_reaction_heat_diag": "F",
    })

    air_y = [0.0, 0.233, 0.767, 0.0, 0.0]
    for pid in (1, 2):
        for i, value in enumerate(air_y, start=1):
            case[f"patch_icpp({pid})%Y({i})"] = value
    for i in range(1, 6):
        case[f"chem_wrt_Y({i})"] = "T"

    return case
