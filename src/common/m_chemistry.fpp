!>
!! @file
!! @brief Contains module m_chemistry
!! @author Henry Le Berre <hberre3@gatech.edu>

#:include 'macros.fpp'
#:include 'case.fpp'

!> @brief Multi-species chemistry interface for thermodynamic properties, reaction rates, and transport coefficients
module m_chemistry

    use iso_fortran_env, only: output_unit

    use m_derived_types, only: scalar_field

    use m_thermochem, only: &
        num_species, molecular_weights, get_temperature, get_net_production_rates, &
        get_creation_destruction_rates, &
        gas_constant, get_mixture_molecular_weight, get_mixture_energy_mass, &
        get_mixture_specific_heat_cp_mass, get_mixture_enthalpy_mass, &
        get_species_enthalpies_rt

    #:if chemistry_transport
        use m_thermochem, only: &
            get_mole_fractions, get_species_binary_mass_diffusivities, &
            get_species_mass_diffusivities_mixavg, get_mixture_thermal_conductivity_mixavg, &
            get_mixture_viscosity_mixavg
    #:endif

    use m_global_parameters
    use m_mpi_common, only: s_mpi_abort, s_mpi_allreduce_integer_sum, s_mpi_allreduce_max
    implicit none

    #:if USING_AMD
        real(wp) :: molecular_weights_nonparameter(10) = &
                    (/2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, &
                      34.014, 39.95, 28.014/)
        $:GPU_DECLARE(create='[molecular_weights_nonparameter]')
    #:endif

    type(int_bounds_info) :: isc1, isc2, isc3
    $:GPU_DECLARE(create='[isc1, isc2, isc3]')
    integer, dimension(3) :: offsets
    $:GPU_DECLARE(create='[offsets]')
    real(wp), parameter :: chem_rho_g_min = 1.0e-14_wp
    real(wp), parameter :: aqss_sumY_norm_tol = 1.0e-10_wp
    real(wp), parameter :: aqss_sumY_abort_tol = 1.0e-5_wp
    real(wp), parameter :: aqss_negative_roundoff_tol = 1.0e-14_wp
    real(wp), parameter :: aqss_pre_sumY_temp_norm_tol = 1.0e-3_wp
    real(wp), parameter :: aqss_pre_negative_abort_tol = 1.0e-3_wp
    real(wp), parameter :: aqss_post_sumY_repair_tol = 1.0e-3_wp
    real(wp), parameter :: aqss_post_negative_repair_tol = 1.0e-12_wp

contains

    logical function s_reaction_heat_diag_active(t_step)

        integer, intent(in) :: t_step

        s_reaction_heat_diag_active = chem_reaction_heat_diag

        if (s_reaction_heat_diag_active) then
            s_reaction_heat_diag_active = (t_step <= 5) .or. (mod(t_step, 50) == 0)
        end if

    end function s_reaction_heat_diag_active

    real(wp) function s_reaction_heat_dt()
        $:GPU_ROUTINE(function_name='s_reaction_heat_dt',parallelism='[seq]', &
            & cray_inline=True)

#ifdef MFC_SIMULATION
        s_reaction_heat_dt = dt
#else
        s_reaction_heat_dt = 1._wp
#endif

    end function s_reaction_heat_dt

    real(wp) function s_reaction_heat_gamma(fluid_id)
        $:GPU_ROUTINE(function_name='s_reaction_heat_gamma',parallelism='[seq]', &
            & cray_inline=True)

        integer, intent(in) :: fluid_id

#ifdef MFC_SIMULATION
        s_reaction_heat_gamma = gammas(fluid_id)
#else
        s_reaction_heat_gamma = fluid_pp(fluid_id)%gamma
#endif

    end function s_reaction_heat_gamma

    real(wp) function s_chemistry_pi_inf(fluid_id)
        $:GPU_ROUTINE(function_name='s_chemistry_pi_inf',parallelism='[seq]', &
            & cray_inline=True)

        integer, intent(in) :: fluid_id

#ifdef MFC_SIMULATION
        s_chemistry_pi_inf = pi_infs(fluid_id)
#else
        s_chemistry_pi_inf = fluid_pp(fluid_id)%pi_inf
#endif

    end function s_chemistry_pi_inf

    real(wp) function s_chemistry_qv(fluid_id)
        $:GPU_ROUTINE(function_name='s_chemistry_qv',parallelism='[seq]', &
            & cray_inline=True)

        integer, intent(in) :: fluid_id

#ifdef MFC_SIMULATION
        s_chemistry_qv = qvs(fluid_id)
#else
        s_chemistry_qv = fluid_pp(fluid_id)%qv
#endif

    end function s_chemistry_qv

    !> @brief GPU-safe typed finite check.
    logical function s_is_finite_wp(x)
        $:GPU_ROUTINE(function_name='s_is_finite_wp',parallelism='[seq]', &
            & cray_inline=True)

        real(wp), intent(in) :: x

        s_is_finite_wp = (x == x) .and. (abs(x) <= huge(x))

    end function s_is_finite_wp

    !> @brief Computes the gas density used to convert rhoY_k into gas species mass fractions.
    subroutine s_compute_chemistry_gas_density(q_cons_vf, x, y, z, rho_g)
        $:GPU_ROUTINE(function_name='s_compute_chemistry_gas_density',parallelism='[seq]', &
            & cray_inline=True)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        integer, intent(in) :: x, y, z
        real(wp), intent(out) :: rho_g

        integer :: i, fluid_id

        rho_g = 0._wp

        if (chem_gas_num_fluids <= 0) then
            rho_g = q_cons_vf(contxb + chem_gas_fluid_id - 1)%sf(x, y, z)
        else
            do i = 1, chem_gas_num_fluids
                fluid_id = chem_gas_fluid_ids(i)
                rho_g = rho_g + q_cons_vf(contxb + fluid_id - 1)%sf(x, y, z)
            end do
        end if

    end subroutine s_compute_chemistry_gas_density

    !> @brief Computes stored and intrinsic gas density for Model-3 chemistry.
    subroutine s_compute_chemistry_gas_alpha_density(q_cons_vf, x, y, z, rho_g, alpha_g, rho_g_intrinsic)
        $:GPU_ROUTINE(function_name='s_compute_chemistry_gas_alpha_density',parallelism='[seq]', &
            & cray_inline=True)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        integer, intent(in) :: x, y, z
        real(wp), intent(out) :: rho_g, alpha_g, rho_g_intrinsic

        integer :: i, fluid_id

        rho_g = 0._wp
        alpha_g = 1._wp

        if (num_fluids > 1 .and. model_eqns == 3) then
            alpha_g = 0._wp
            if (chem_gas_num_fluids <= 0) then
                fluid_id = chem_gas_fluid_id
                rho_g = q_cons_vf(contxb + fluid_id - 1)%sf(x, y, z)
                alpha_g = q_cons_vf(advxb + fluid_id - 1)%sf(x, y, z)
            else
                do i = 1, chem_gas_num_fluids
                    fluid_id = chem_gas_fluid_ids(i)
                    rho_g = rho_g + q_cons_vf(contxb + fluid_id - 1)%sf(x, y, z)
                    alpha_g = alpha_g + q_cons_vf(advxb + fluid_id - 1)%sf(x, y, z)
                end do
            end if
            alpha_g = min(max(alpha_g, 0._wp), 1._wp)
            if (alpha_g > sgm_eps) then
                rho_g_intrinsic = rho_g/alpha_g
            else
                rho_g_intrinsic = 0._wp
            end if
        else
            rho_g = q_cons_vf(contxb)%sf(x, y, z)
            rho_g_intrinsic = rho_g
        end if

    end subroutine s_compute_chemistry_gas_alpha_density

    !> @brief Recovers the Model-3 gas pressure implied by selected gas internal-energy equations.
    subroutine s_compute_chemistry_gas_pressure(q_cons_vf, x, y, z, pres_g)
        $:GPU_ROUTINE(function_name='s_compute_chemistry_gas_pressure',parallelism='[seq]', &
            & cray_inline=True)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        integer, intent(in) :: x, y, z
        real(wp), intent(out) :: pres_g

        integer :: i, fluid_id, count_g
        real(wp) :: alpha_i, p_i, weight_i, p_weight_sum, weight_sum

        pres_g = 0._wp

        if (num_fluids > 1 .and. model_eqns == 3) then
            p_weight_sum = 0._wp
            weight_sum = 0._wp
            count_g = max(1, chem_gas_num_fluids)
            do i = 1, count_g
                if (chem_gas_num_fluids <= 0) then
                    fluid_id = chem_gas_fluid_id
                else
                    fluid_id = chem_gas_fluid_ids(i)
                end if
                alpha_i = q_cons_vf(advxb + fluid_id - 1)%sf(x, y, z)
                if (alpha_i > sgm_eps) then
                    p_i = ((q_cons_vf(intxb + fluid_id - 1)%sf(x, y, z) - &
                            q_cons_vf(contxb + fluid_id - 1)%sf(x, y, z)*s_chemistry_qv(fluid_id))/alpha_i - &
                           s_chemistry_pi_inf(fluid_id))/s_reaction_heat_gamma(fluid_id)
                    weight_i = alpha_i*s_reaction_heat_gamma(fluid_id)
                    p_weight_sum = p_weight_sum + weight_i*p_i
                    weight_sum = weight_sum + weight_i
                end if
            end do
            if (weight_sum > sgm_eps) pres_g = p_weight_sum/weight_sum
        end if

    end subroutine s_compute_chemistry_gas_pressure

    !> @brief Recovers chemistry temperature from the current Model-3 gas state.
    subroutine s_compute_chemistry_reactor_temperature(q_cons_vf, q_T_sf, x, y, z, rho_intrinsic, Ys, T)
        $:GPU_ROUTINE(function_name='s_compute_chemistry_reactor_temperature',parallelism='[seq]', &
            & cray_inline=True)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        type(scalar_field), intent(in) :: q_T_sf
        integer, intent(in) :: x, y, z
        real(wp), intent(in) :: rho_intrinsic
        real(wp), dimension(num_species), intent(in) :: Ys
        real(wp), intent(out) :: T

        real(wp) :: pres_g, mix_mol_weight, T_candidate

        T = q_T_sf%sf(x, y, z)

        if (chem_fixed_T_enable .and. num_fluids > 1) then
            T = chem_fixed_T
        elseif (num_fluids > 1 .and. model_eqns == 3 .and. rho_intrinsic > chem_rho_g_min) then
            call s_compute_chemistry_gas_pressure(q_cons_vf, x, y, z, pres_g)
            if (pres_g > 0._wp .and. s_is_finite_wp(pres_g)) then
                call get_mixture_molecular_weight(Ys, mix_mol_weight)
                T_candidate = pres_g*mix_mol_weight/(gas_constant*rho_intrinsic)
                if (T_candidate > 0._wp .and. s_is_finite_wp(T_candidate)) T = T_candidate
            end if
        end if

        T = min(max(T, chem_T_min), chem_T_max)

    end subroutine s_compute_chemistry_reactor_temperature

    !> @brief Computes mixture viscosities for left and right states and inverts them for use as reciprocal Reynolds numbers.
    subroutine compute_viscosity_and_inversion(T_L, Ys_L, T_R, Ys_R, Re_L, Re_R)

        $:GPU_ROUTINE(function_name='compute_viscosity_and_inversion',parallelism='[seq]', &
            & cray_inline=True)

        real(wp), intent(inout) :: T_L, T_R, Re_L, Re_R
        real(wp), dimension(num_species), intent(inout) :: Ys_R, Ys_L

        #:if chemistry_transport
            call get_mixture_viscosity_mixavg(T_L, Ys_L, Re_L)
            call get_mixture_viscosity_mixavg(T_R, Ys_R, Re_R)
            Re_L = 1.0_wp/Re_L
            Re_R = 1.0_wp/Re_R
        #:else
            Re_L = 0._wp
            Re_R = 0._wp
        #:endif

    end subroutine compute_viscosity_and_inversion

    !> @brief Initializes the temperature field from conservative variables by inverting the energy equation.
    subroutine s_compute_q_T_sf(q_T_sf, q_cons_vf, bounds)

        ! Initialize the temperature field at the start of the simulation to
        ! reasonable values. Temperature is computed the regular way using the
        ! conservative variables.

        type(scalar_field), intent(inout) :: q_T_sf
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds

        integer :: x, y, z, eqn
        real(wp) :: energy, T_in, rho_g
        real(wp), dimension(num_species) :: Ys

        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end
                    call s_compute_chemistry_gas_density(q_cons_vf, x, y, z, rho_g)
                    rho_g = max(rho_g, chem_rho_g_min)

                    if (chemistry .and. num_fluids > 1 .and. chem_fixed_T_enable) then
                        q_T_sf%sf(x, y, z) = min(max(chem_fixed_T, chem_T_min), chem_T_max)
                        cycle
                    end if

                    do eqn = chemxb, chemxe
                        Ys(eqn - chemxb + 1) = &
                            max(0._wp, q_cons_vf(eqn)%sf(x, y, z)/rho_g)
                    end do

                    ! e = E - 1/2*|u|^2
                    ! cons. E_idx     = \rho E
                    ! cons. contxb    = \rho         (1-fluid model)
                    ! cons. momxb + i = \rho u_i
                    energy = q_cons_vf(E_idx)%sf(x, y, z)/q_cons_vf(contxb)%sf(x, y, z)
                    do eqn = momxb, momxe
                        energy = energy - &
                                 0.5_wp*(q_cons_vf(eqn)%sf(x, y, z)/q_cons_vf(contxb)%sf(x, y, z))**2._wp
                    end do

                    T_in = real(q_T_sf%sf(x, y, z), kind=wp)
                    call get_temperature(energy, dflt_T_guess, Ys, .true., T_in)
                    q_T_sf%sf(x, y, z) = T_in

                end do
            end do
        end do

    end subroutine s_compute_q_T_sf

    !> @brief Computes the temperature field from primitive variables using the ideal gas law and mixture molecular weight.
    subroutine s_compute_T_from_primitives(q_T_sf, q_prim_vf, bounds)

        type(scalar_field), intent(inout) :: q_T_sf
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds

        integer :: x, y, z, i
        real(wp), dimension(num_species) :: Ys
        real(wp) :: mix_mol_weight

        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end
                    do i = chemxb, chemxe
                        Ys(i - chemxb + 1) = q_prim_vf(i)%sf(x, y, z)
                    end do

                    call get_mixture_molecular_weight(Ys, mix_mol_weight)
                    q_T_sf%sf(x, y, z) = q_prim_vf(E_idx)%sf(x, y, z)*mix_mol_weight/(gas_constant*q_prim_vf(1)%sf(x, y, z))
                end do
            end do
        end do

    end subroutine s_compute_T_from_primitives

    subroutine s_print_reaction_heat_cell_diag(q_cons_qp, q_T_sf, q_prim_qp, bounds, t_step, stage, heat_limit_frac)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_qp, q_prim_qp
        type(scalar_field), intent(in) :: q_T_sf
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds
        integer, intent(in) :: t_step, stage
        real(wp), intent(in) :: heat_limit_frac

        integer :: x, y, z, eqn, gas_idx, fluid_id
        integer :: valid_cells, T_clamp_count
        integer :: qdot_gt_1e12_count, qdot_gt_1e14_count, qdot_gt_1e16_count
        integer :: max_x, max_y, max_z
        real(wp), parameter :: rho_g_filter_min = 1.0e-8_wp
        real(wp), parameter :: interface_alpha_min = 1.0e-6_wp
        real(wp) :: T, T_raw
        real(wp) :: rho, rho_g, rhoYk, raw_Y, Y_sum, omega_m
        real(wp) :: h_k, qdot_h_cell, mw
        real(wp) :: alpha_liq, alpha_vap, alpha_air, alpha_g
        real(wp) :: qdot_h_sum_all, qdot_h_pos_all
        real(wp) :: qdot_h_pos_gas, qdot_h_pos_interface
        real(wp) :: qdot_h_pos_not_clamp, qdot_h_pos_at_clamp
        real(wp) :: max_qdot_h, max_alpha_liq, max_alpha_vap, max_alpha_air
        real(wp) :: gas_internal_energy_density, heat_added, max_heat, heat_scale, max_heat_scale
        real(wp) :: max_rho_g, max_pressure, max_q_T_sf, max_T_used
        real(wp) :: max_Y_fuel, max_Y_o2, max_Y_co2, max_Y_h2o
        real(wp) :: max_omega_fuel, max_omega_o2
        logical :: omega_finite, T_at_clamp, max_is_gas_filter, max_is_interface
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(10) :: Ys
            real(wp), dimension(10) :: omega
            real(wp), dimension(10) :: h_rt
        #:else
            real(wp), dimension(num_species) :: Ys
            real(wp), dimension(num_species) :: omega
            real(wp), dimension(num_species) :: h_rt
        #:endif

        valid_cells = 0
        T_clamp_count = 0
        qdot_gt_1e12_count = 0
        qdot_gt_1e14_count = 0
        qdot_gt_1e16_count = 0
        max_x = -1
        max_y = -1
        max_z = -1
        qdot_h_sum_all = 0._wp
        qdot_h_pos_all = 0._wp
        qdot_h_pos_gas = 0._wp
        qdot_h_pos_interface = 0._wp
        qdot_h_pos_not_clamp = 0._wp
        qdot_h_pos_at_clamp = 0._wp
        max_qdot_h = -huge(1._wp)
        max_alpha_liq = 0._wp
        max_alpha_vap = 0._wp
        max_alpha_air = 0._wp
        max_rho_g = 0._wp
        max_pressure = 0._wp
        max_q_T_sf = 0._wp
        max_T_used = 0._wp
        max_Y_fuel = 0._wp
        max_Y_o2 = 0._wp
        max_Y_co2 = 0._wp
        max_Y_h2o = 0._wp
        max_omega_fuel = 0._wp
        max_omega_o2 = 0._wp
        max_heat_scale = 1._wp
        max_is_gas_filter = .false.
        max_is_interface = .false.

        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end

                    if (num_fluids > 1) then
                        call s_compute_chemistry_gas_density(q_cons_qp, x, y, z, rho_g)
                        if ((.not. s_is_finite_wp(rho_g)) .or. rho_g <= chem_rho_g_min) cycle

                        Y_sum = 0._wp
                        do eqn = chemxb, chemxe
                            rhoYk = q_cons_qp(eqn)%sf(x, y, z)
                            if (.not. s_is_finite_wp(rhoYk)) then
                                Ys(eqn - chemxb + 1) = 0._wp
                            else
                                raw_Y = rhoYk/rho_g
                                if (.not. s_is_finite_wp(raw_Y)) then
                                    Ys(eqn - chemxb + 1) = 0._wp
                                else
                                    Ys(eqn - chemxb + 1) = min(max(raw_Y, 0._wp), 1._wp)
                                end if
                            end if
                            Y_sum = Y_sum + Ys(eqn - chemxb + 1)
                        end do

                        if (.not. s_is_finite_wp(Y_sum)) cycle

                        if (Y_sum > 1._wp) then
                            do eqn = 1, num_species
                                Ys(eqn) = Ys(eqn)/Y_sum
                            end do
                        end if

                        rho = rho_g
                    else
                        do eqn = chemxb, chemxe
                            Ys(eqn - chemxb + 1) = q_prim_qp(eqn)%sf(x, y, z)
                        end do

                        rho = q_cons_qp(contxe)%sf(x, y, z)
                        rho_g = rho
                    end if

                    T_raw = q_T_sf%sf(x, y, z)
                    if (.not. s_is_finite_wp(T_raw)) cycle
                    T = min(max(T_raw, chem_T_min), chem_T_max)
                    T_at_clamp = T_raw >= chem_T_max

                    call get_net_production_rates(rho, T, Ys, omega)
                    call get_species_enthalpies_rt(T, h_rt)

                    omega_finite = .true.
                    do eqn = 1, num_species
                        if (.not. s_is_finite_wp(omega(eqn))) omega_finite = .false.
                    end do
                    if (.not. omega_finite) cycle

                    qdot_h_cell = 0._wp
                    do eqn = 1, num_species
                        #:if USING_AMD
                            mw = molecular_weights_nonparameter(eqn)
                        #:else
                            mw = molecular_weights(eqn)
                        #:endif
                        omega_m = mw*omega(eqn)
                        if (s_is_finite_wp(omega_m)) then
                            h_k = h_rt(eqn)*gas_constant*T/mw
                            qdot_h_cell = qdot_h_cell - h_k*omega_m
                        end if
                    end do

                    if (.not. s_is_finite_wp(qdot_h_cell)) cycle

                    heat_scale = 1._wp
                    if (heat_limit_frac > 0._wp .and. num_fluids > 1 .and. model_eqns == 3 .and. qdot_h_cell > 0._wp) then
                        gas_internal_energy_density = 0._wp
                        if (chem_gas_num_fluids <= 0) then
                            fluid_id = chem_gas_fluid_id
                            if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                gas_internal_energy_density = gas_internal_energy_density + &
                                                              max(0._wp, q_cons_qp(intxb + fluid_id - 1)%sf(x, y, z))
                            end if
                        else
                            do gas_idx = 1, chem_gas_num_fluids
                                fluid_id = chem_gas_fluid_ids(gas_idx)
                                if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                    gas_internal_energy_density = gas_internal_energy_density + &
                                                                  max(0._wp, q_cons_qp(intxb + fluid_id - 1)%sf(x, y, z))
                                end if
                            end do
                        end if

                        if (s_is_finite_wp(gas_internal_energy_density) .and. gas_internal_energy_density > 0._wp) then
                            heat_added = qdot_h_cell*s_reaction_heat_dt()
                            max_heat = heat_limit_frac*gas_internal_energy_density
                            if (s_is_finite_wp(heat_added) .and. heat_added > 0._wp .and. s_is_finite_wp(max_heat)) then
                                heat_scale = min(1._wp, max_heat/max(heat_added, tiny(1._wp)))
                                if (.not. s_is_finite_wp(heat_scale)) heat_scale = 1._wp
                                heat_scale = max(0._wp, heat_scale)
                            end if
                        end if
                    end if

                    alpha_liq = 0._wp
                    alpha_vap = 0._wp
                    alpha_air = 0._wp
                    if (num_fluids >= 1) alpha_liq = q_prim_qp(advxb)%sf(x, y, z)
                    if (num_fluids >= 2) alpha_vap = q_prim_qp(advxb + 1)%sf(x, y, z)
                    if (num_fluids >= 3) alpha_air = q_prim_qp(advxb + 2)%sf(x, y, z)
                    alpha_g = alpha_vap + alpha_air

                    valid_cells = valid_cells + 1
                    if (T_at_clamp) T_clamp_count = T_clamp_count + 1
                    if (qdot_h_cell > 1.0e12_wp) qdot_gt_1e12_count = qdot_gt_1e12_count + 1
                    if (qdot_h_cell > 1.0e14_wp) qdot_gt_1e14_count = qdot_gt_1e14_count + 1
                    if (qdot_h_cell > 1.0e16_wp) qdot_gt_1e16_count = qdot_gt_1e16_count + 1

                    qdot_h_sum_all = qdot_h_sum_all + qdot_h_cell
                    qdot_h_pos_all = qdot_h_pos_all + max(0._wp, qdot_h_cell)
                    if (rho_g > rho_g_filter_min) qdot_h_pos_gas = qdot_h_pos_gas + max(0._wp, qdot_h_cell)
                    if (alpha_liq > interface_alpha_min .and. alpha_g > interface_alpha_min) then
                        qdot_h_pos_interface = qdot_h_pos_interface + max(0._wp, qdot_h_cell)
                    end if
                    if (T_at_clamp) then
                        qdot_h_pos_at_clamp = qdot_h_pos_at_clamp + max(0._wp, qdot_h_cell)
                    else
                        qdot_h_pos_not_clamp = qdot_h_pos_not_clamp + max(0._wp, qdot_h_cell)
                    end if

                    if (qdot_h_cell > max_qdot_h) then
                        max_qdot_h = qdot_h_cell
                        max_x = x
                        max_y = y
                        max_z = z
                        max_alpha_liq = alpha_liq
                        max_alpha_vap = alpha_vap
                        max_alpha_air = alpha_air
                        max_rho_g = rho_g
                        max_pressure = q_prim_qp(E_idx)%sf(x, y, z)
                        max_q_T_sf = T_raw
                        max_T_used = T
                        max_Y_fuel = 0._wp
                        max_Y_o2 = 0._wp
                        max_Y_co2 = 0._wp
                        max_Y_h2o = 0._wp
                        if (num_species >= 1) max_Y_fuel = Ys(1)
                        if (num_species >= 2) max_Y_o2 = Ys(2)
                        if (num_species >= 4) max_Y_co2 = Ys(4)
                        if (num_species >= 5) max_Y_h2o = Ys(5)
                        max_omega_fuel = 0._wp
                        max_omega_o2 = 0._wp
                        if (num_species >= 1) then
                            #:if USING_AMD
                                max_omega_fuel = molecular_weights_nonparameter(1)*omega(1)
                            #:else
                                max_omega_fuel = molecular_weights(1)*omega(1)
                            #:endif
                        end if
                        if (num_species >= 2) then
                            #:if USING_AMD
                                max_omega_o2 = molecular_weights_nonparameter(2)*omega(2)
                            #:else
                                max_omega_o2 = molecular_weights(2)*omega(2)
                            #:endif
                        end if
                        max_heat_scale = heat_scale
                        max_is_gas_filter = rho_g > rho_g_filter_min
                        max_is_interface = alpha_liq > interface_alpha_min .and. alpha_g > interface_alpha_min
                    end if

                end do
            end do
        end do

        if (max_qdot_h < 0._wp) max_qdot_h = 0._wp

        print '(" REACTION_HEAT_CELL_DIAG rank=", I6, " t_step=", I8, " stage=", I4, &
                &" valid_cells=", I10, " T_clamp_count=", I10, &
                &" qdot_gt_1e12=", I10, " qdot_gt_1e14=", I10, " qdot_gt_1e16=", I10, &
                &" rho_g_filter=", ES16.6, " interface_alpha_min=", ES16.6, &
                &" qdot_h_sum_all=", ES16.6, " qdot_h_pos_all=", ES16.6, &
                &" qdot_h_pos_gas=", ES16.6, " qdot_h_pos_interface=", ES16.6, &
                &" qdot_h_pos_not_clamp=", ES16.6, " qdot_h_pos_at_clamp=", ES16.6, &
                &" max_qdot_h=", ES16.6, " max_i=", I8, " max_j=", I8, " max_k=", I8, &
                &" max_alpha_liq=", ES16.6, " max_alpha_vap=", ES16.6, &
                &" max_alpha_air=", ES16.6, " max_rho_g=", ES16.6, &
                &" max_pressure=", ES16.6, " max_q_T_sf=", ES16.6, " max_T_used=", ES16.6, &
                &" max_Y_fuel=", ES16.6, " max_Y_O2=", ES16.6, &
                &" max_Y_CO2=", ES16.6, " max_Y_H2O=", ES16.6, &
                &" max_omega_fuel=", ES16.6, " max_omega_O2=", ES16.6, &
                &" max_heat_scale=", ES16.6, " limit_frac=", ES16.6, &
                &" max_is_gas_filter=", L1, " max_is_interface=", L1)', &
            proc_rank, t_step, stage, valid_cells, T_clamp_count, &
            qdot_gt_1e12_count, qdot_gt_1e14_count, qdot_gt_1e16_count, &
            rho_g_filter_min, interface_alpha_min, qdot_h_sum_all, qdot_h_pos_all, &
            qdot_h_pos_gas, qdot_h_pos_interface, qdot_h_pos_not_clamp, &
            qdot_h_pos_at_clamp, max_qdot_h, max_x, max_y, max_z, &
            max_alpha_liq, max_alpha_vap, max_alpha_air, max_rho_g, max_pressure, &
            max_q_T_sf, max_T_used, max_Y_fuel, max_Y_o2, max_Y_co2, max_Y_h2o, &
            max_omega_fuel, max_omega_o2, max_heat_scale, heat_limit_frac, max_is_gas_filter, max_is_interface
        call flush(output_unit)

    end subroutine s_print_reaction_heat_cell_diag

    !> @brief Adds chemical reaction source terms to the species transport RHS using net production rates.
    subroutine s_compute_chemistry_reaction_flux(rhs_vf, q_cons_qp, q_T_sf, q_prim_qp, bounds, t_step, stage, &
                                                 q_rk_stored, rk_a, rk_b, rk_c, rk_d, rk_dt, &
                                                 coupled_source_limit_active)

        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        type(scalar_field), intent(inout) :: q_T_sf
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_qp, q_prim_qp
        type(scalar_field), dimension(sys_size), intent(in) :: q_rk_stored
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds
        integer, intent(in) :: t_step, stage
        real(wp), intent(in) :: rk_a, rk_b, rk_c, rk_d, rk_dt
        logical, intent(in) :: coupled_source_limit_active

        integer :: x, y, z
        integer :: eqn, gas_idx, fluid_id
        real(wp) :: T, T_raw
        real(wp) :: rho, rho_g, rhoYk, raw_Y, Y_sum, omega_m, omega_m_limited
        real(wp) :: h_k, e_k, qdot_h_cell, qdot_e_cell, qdot_h_limited, qdot_e_limited
        real(wp) :: heat_weight, heat_weight_denom
        real(wp) :: heat_limit_frac, gas_internal_energy_density, heat_added, max_heat, heat_scale
        real(wp) :: chem_theta, theta_candidate, available_rhoY, consuming_source, stage_dt
        real(wp) :: chemistry_scale
        real(wp) :: diag_qdot_h_pos, diag_qdot_h_neg, diag_qdot_e_pos, diag_qdot_e_neg
        real(wp) :: diag_raw_qdot_h_pos, diag_raw_qdot_h_neg, diag_raw_qdot_e_pos, diag_raw_qdot_e_neg
        real(wp) :: diag_src_c12h26, diag_src_o2, diag_src_co2, diag_src_h2o
        real(wp) :: diag_raw_src_c12h26, diag_raw_src_o2, diag_raw_src_co2, diag_raw_src_h2o
        real(wp) :: diag_heat_applied, diag_raw_heat_applied, diag_partial_heat_applied, diag_partial_heat_skipped
        real(wp) :: diag_limited_cell_count, diag_scale_min, diag_scale_sum_limited, diag_scale_mean_limited
        real(wp) :: diag_T_clamp_count, diag_max_raw_qdot_h
        real(wp) :: diag_T_max, diag_p_max
        logical :: omega_finite
        logical :: heat_diag_active, apply_heat_active, limit_heat_active, heat_calc_active
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(10) :: Ys
            real(wp), dimension(10) :: omega
            real(wp), dimension(10) :: omega_m_species
            real(wp), dimension(10) :: h_rt
        #:else
            real(wp), dimension(num_species) :: Ys
            real(wp), dimension(num_species) :: omega
            real(wp), dimension(num_species) :: omega_m_species
            real(wp), dimension(num_species) :: h_rt
        #:endif

        heat_diag_active = s_reaction_heat_diag_active(t_step)
        apply_heat_active = chem_reaction_heat_enable .and. num_fluids > 1 .and. model_eqns == 3
        heat_limit_frac = max(0._wp, chem_reaction_heat_limit_frac)
        limit_heat_active = apply_heat_active .and. heat_limit_frac > 0._wp
        heat_calc_active = heat_diag_active .or. apply_heat_active .or. limit_heat_active
        diag_qdot_h_pos = 0._wp
        diag_qdot_h_neg = 0._wp
        diag_qdot_e_pos = 0._wp
        diag_qdot_e_neg = 0._wp
        diag_raw_qdot_h_pos = 0._wp
        diag_raw_qdot_h_neg = 0._wp
        diag_raw_qdot_e_pos = 0._wp
        diag_raw_qdot_e_neg = 0._wp
        diag_src_c12h26 = 0._wp
        diag_src_o2 = 0._wp
        diag_src_co2 = 0._wp
        diag_src_h2o = 0._wp
        diag_raw_src_c12h26 = 0._wp
        diag_raw_src_o2 = 0._wp
        diag_raw_src_co2 = 0._wp
        diag_raw_src_h2o = 0._wp
        diag_heat_applied = 0._wp
        diag_raw_heat_applied = 0._wp
        diag_partial_heat_applied = 0._wp
        diag_partial_heat_skipped = 0._wp
        diag_limited_cell_count = 0._wp
        diag_scale_min = 1._wp
        diag_scale_sum_limited = 0._wp
        diag_scale_mean_limited = 1._wp
        diag_T_clamp_count = 0._wp
        diag_max_raw_qdot_h = 0._wp
        diag_T_max = 0._wp
        diag_p_max = 0._wp

        #:set chem_reaction_private = '[Ys, omega, omega_m_species, h_rt, eqn, gas_idx, fluid_id, ' + &
            & 'T, T_raw, rho, rho_g, rhoYk, raw_Y, Y_sum, omega_m, ' + &
            & 'omega_m_limited, omega_finite, h_k, e_k, qdot_h_cell, ' + &
            & 'qdot_e_cell, qdot_h_limited, qdot_e_limited, heat_weight, ' + &
            & 'heat_weight_denom, gas_internal_energy_density, heat_added, ' + &
            & 'max_heat, heat_scale, chem_theta, theta_candidate, ' + &
            & 'available_rhoY, consuming_source, stage_dt, chemistry_scale]'
        #:set chem_reaction_reduction = '[[diag_qdot_h_pos, diag_qdot_h_neg, ' + &
            & 'diag_qdot_e_pos, diag_qdot_e_neg, diag_raw_qdot_h_pos, ' + &
            & 'diag_raw_qdot_h_neg, diag_raw_qdot_e_pos, diag_raw_qdot_e_neg, ' + &
            & 'diag_src_c12h26, diag_src_o2, diag_src_co2, diag_src_h2o, ' + &
            & 'diag_raw_src_c12h26, diag_raw_src_o2, diag_raw_src_co2, ' + &
            & 'diag_raw_src_h2o, diag_heat_applied, diag_raw_heat_applied, ' + &
            & 'diag_partial_heat_applied, diag_partial_heat_skipped, ' + &
            & 'diag_limited_cell_count, diag_scale_sum_limited, ' + &
            & 'diag_T_clamp_count], [diag_T_max, diag_p_max, ' + &
            & 'diag_max_raw_qdot_h], [diag_scale_min]]'
        $:GPU_PARALLEL_LOOP(collapse=3, &
            private=chem_reaction_private, &
            reduction=chem_reaction_reduction, &
            reductionOp='[+, MAX, MIN]', copyin='[bounds]')
        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end

                    if (num_fluids > 1) then
                        call s_compute_chemistry_gas_density(q_cons_qp, x, y, z, rho_g)
                        if ((.not. s_is_finite_wp(rho_g)) .or. rho_g <= chem_rho_g_min) then
                            ! No gas-phase chemistry if designated gas density vanishes.
                            cycle
                        end if

                        Y_sum = 0._wp
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = chemxb, chemxe
                            rhoYk = q_cons_qp(eqn)%sf(x, y, z)
                            if (.not. s_is_finite_wp(rhoYk)) then
                                Ys(eqn - chemxb + 1) = 0._wp
                            else
                                raw_Y = rhoYk/rho_g
                                if (.not. s_is_finite_wp(raw_Y)) then
                                    Ys(eqn - chemxb + 1) = 0._wp
                                else
                                    Ys(eqn - chemxb + 1) = min(max(raw_Y, 0._wp), 1._wp)
                                end if
                            end if
                            Y_sum = Y_sum + Ys(eqn - chemxb + 1)
                        end do

                        if (.not. s_is_finite_wp(Y_sum)) cycle

                        if (Y_sum > 1._wp) then
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys(eqn) = Ys(eqn)/Y_sum
                            end do
                        end if

                        rho = rho_g
                    else
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = chemxb, chemxe
                            Ys(eqn - chemxb + 1) = q_prim_qp(eqn)%sf(x, y, z)
                        end do

                        rho = q_cons_qp(contxe)%sf(x, y, z)
                    end if

                    T_raw = q_T_sf%sf(x, y, z)
                    if (.not. s_is_finite_wp(T_raw)) cycle
                    T = min(max(T_raw, chem_T_min), chem_T_max)
                    if (heat_diag_active) then
                        diag_T_max = max(diag_T_max, T)
                        diag_p_max = max(diag_p_max, q_prim_qp(E_idx)%sf(x, y, z))
                        if (T_raw >= chem_T_max) diag_T_clamp_count = diag_T_clamp_count + 1._wp
                    end if

                    call get_net_production_rates(rho, T, Ys, omega)
                    if (heat_calc_active) call get_species_enthalpies_rt(T, h_rt)
                    omega_finite = .true.
                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = 1, num_species
                        if (.not. s_is_finite_wp(omega(eqn))) omega_finite = .false.
                    end do
                    if (.not. omega_finite) cycle

                    qdot_h_cell = 0._wp
                    qdot_e_cell = 0._wp

                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = chemxb, chemxe
                        omega_m_species(eqn - chemxb + 1) = 0._wp
                        #:if USING_AMD
                            omega_m = molecular_weights_nonparameter(eqn - chemxb + 1)*omega(eqn - chemxb + 1)
                        #:else
                            omega_m = molecular_weights(eqn - chemxb + 1)*omega(eqn - chemxb + 1)
                        #:endif
                        if (s_is_finite_wp(omega_m)) then
                            omega_m_species(eqn - chemxb + 1) = omega_m
                            if (heat_calc_active) then
                                #:if USING_AMD
                                    h_k = h_rt(eqn - chemxb + 1)*gas_constant*T/molecular_weights_nonparameter(eqn - chemxb + 1)
                                    e_k = (h_rt(eqn - chemxb + 1) - 1._wp)*gas_constant*T/ &
                                          molecular_weights_nonparameter(eqn - chemxb + 1)
                                #:else
                                    h_k = h_rt(eqn - chemxb + 1)*gas_constant*T/molecular_weights(eqn - chemxb + 1)
                                    e_k = (h_rt(eqn - chemxb + 1) - 1._wp)*gas_constant*T/molecular_weights(eqn - chemxb + 1)
                                #:endif
                                qdot_h_cell = qdot_h_cell - h_k*omega_m
                                qdot_e_cell = qdot_e_cell - e_k*omega_m
                            end if
                        end if

                    end do

                    heat_scale = 1._wp
                    if (limit_heat_active .and. qdot_h_cell > 0._wp .and. s_is_finite_wp(qdot_h_cell)) then
                        gas_internal_energy_density = 0._wp
                        if (chem_gas_num_fluids <= 0) then
                            fluid_id = chem_gas_fluid_id
                            if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                gas_internal_energy_density = gas_internal_energy_density + &
                                                              max(0._wp, q_cons_qp(intxb + fluid_id - 1)%sf(x, y, z))
                            end if
                        else
                            $:GPU_LOOP(parallelism='[seq]')
                            do gas_idx = 1, chem_gas_num_fluids
                                fluid_id = chem_gas_fluid_ids(gas_idx)
                                if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                    gas_internal_energy_density = gas_internal_energy_density + &
                                                                  max(0._wp, q_cons_qp(intxb + fluid_id - 1)%sf(x, y, z))
                                end if
                            end do
                        end if

                        if (s_is_finite_wp(gas_internal_energy_density) .and. gas_internal_energy_density > 0._wp) then
                            heat_added = qdot_h_cell*s_reaction_heat_dt()
                            max_heat = heat_limit_frac*gas_internal_energy_density
                            if (s_is_finite_wp(heat_added) .and. heat_added > 0._wp .and. s_is_finite_wp(max_heat)) then
                                heat_scale = min(1._wp, max_heat/max(heat_added, tiny(1._wp)))
                                if (.not. s_is_finite_wp(heat_scale)) heat_scale = 1._wp
                                heat_scale = max(0._wp, heat_scale)
                            end if
                        end if
                    end if

                    stage_dt = 0._wp
                    if (rk_d /= 0._wp) stage_dt = rk_c*rk_dt/rk_d
                    chem_theta = 1._wp
                    if (coupled_source_limit_active .and. stage_dt > 0._wp) then
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = chemxb, chemxe
                            consuming_source = heat_scale*omega_m_species(eqn - chemxb + 1)
                            if (consuming_source < 0._wp) then
                                available_rhoY = (rk_a*q_cons_qp(eqn)%sf(x, y, z) + &
                                                  rk_b*q_rk_stored(eqn)%sf(x, y, z) + &
                                                  rk_c*rk_dt*rhs_vf(eqn)%sf(x, y, z))/rk_d
                                theta_candidate = available_rhoY/(-stage_dt*consuming_source)
                                theta_candidate = min(1._wp, max(0._wp, theta_candidate))
                                chem_theta = min(chem_theta, theta_candidate)
                            end if
                        end do
                    end if

                    chemistry_scale = heat_scale*chem_theta
                    qdot_h_limited = chemistry_scale*qdot_h_cell
                    qdot_e_limited = chemistry_scale*qdot_e_cell

                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = chemxb, chemxe
                        omega_m = omega_m_species(eqn - chemxb + 1)
                        omega_m_limited = chemistry_scale*omega_m
                        rhs_vf(eqn)%sf(x, y, z) = rhs_vf(eqn)%sf(x, y, z) + omega_m_limited

                        if (heat_diag_active) then
                            if ((eqn - chemxb + 1) == 1) then
                                diag_raw_src_c12h26 = diag_raw_src_c12h26 + omega_m
                                diag_src_c12h26 = diag_src_c12h26 + omega_m_limited
                            end if
                            if ((eqn - chemxb + 1) == 2) then
                                diag_raw_src_o2 = diag_raw_src_o2 + omega_m
                                diag_src_o2 = diag_src_o2 + omega_m_limited
                            end if
                            if ((eqn - chemxb + 1) == 4) then
                                diag_raw_src_co2 = diag_raw_src_co2 + omega_m
                                diag_src_co2 = diag_src_co2 + omega_m_limited
                            end if
                            if ((eqn - chemxb + 1) == 5) then
                                diag_raw_src_h2o = diag_raw_src_h2o + omega_m
                                diag_src_h2o = diag_src_h2o + omega_m_limited
                            end if
                        end if
                    end do

                    if (apply_heat_active .and. s_is_finite_wp(qdot_h_limited)) then
                        rhs_vf(E_idx)%sf(x, y, z) = rhs_vf(E_idx)%sf(x, y, z) + qdot_h_limited
                        diag_heat_applied = diag_heat_applied + qdot_h_limited
                        diag_raw_heat_applied = diag_raw_heat_applied + qdot_h_cell

                        heat_weight_denom = 0._wp
                        if (chem_gas_num_fluids <= 0) then
                            fluid_id = chem_gas_fluid_id
                            if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                heat_weight_denom = q_cons_qp(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                    s_reaction_heat_gamma(fluid_id)
                            end if
                        else
                            $:GPU_LOOP(parallelism='[seq]')
                            do gas_idx = 1, chem_gas_num_fluids
                                fluid_id = chem_gas_fluid_ids(gas_idx)
                                if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                    heat_weight_denom = heat_weight_denom + &
                                                        q_cons_qp(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                        s_reaction_heat_gamma(fluid_id)
                                end if
                            end do
                        end if

                        if (heat_weight_denom > chem_rho_g_min) then
                            if (chem_gas_num_fluids <= 0) then
                                fluid_id = chem_gas_fluid_id
                                if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                    heat_weight = q_cons_qp(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                  s_reaction_heat_gamma(fluid_id)/heat_weight_denom
                                    rhs_vf(intxb + fluid_id - 1)%sf(x, y, z) = &
                                        rhs_vf(intxb + fluid_id - 1)%sf(x, y, z) + qdot_h_limited*heat_weight
                                    diag_partial_heat_applied = diag_partial_heat_applied + qdot_h_limited*heat_weight
                                end if
                            else
                                $:GPU_LOOP(parallelism='[seq]')
                                do gas_idx = 1, chem_gas_num_fluids
                                    fluid_id = chem_gas_fluid_ids(gas_idx)
                                    if (fluid_id >= 1 .and. fluid_id <= num_fluids) then
                                        heat_weight = q_cons_qp(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                      s_reaction_heat_gamma(fluid_id)/heat_weight_denom
                                        rhs_vf(intxb + fluid_id - 1)%sf(x, y, z) = &
                                            rhs_vf(intxb + fluid_id - 1)%sf(x, y, z) + qdot_h_limited*heat_weight
                                        diag_partial_heat_applied = diag_partial_heat_applied + qdot_h_limited*heat_weight
                                    end if
                                end do
                            end if
                        else
                            diag_partial_heat_skipped = diag_partial_heat_skipped + 1._wp
                        end if
                    end if

                    if (heat_diag_active) then
                        diag_raw_qdot_h_pos = diag_raw_qdot_h_pos + max(0._wp, qdot_h_cell)
                        diag_raw_qdot_h_neg = diag_raw_qdot_h_neg + min(0._wp, qdot_h_cell)
                        diag_raw_qdot_e_pos = diag_raw_qdot_e_pos + max(0._wp, qdot_e_cell)
                        diag_raw_qdot_e_neg = diag_raw_qdot_e_neg + min(0._wp, qdot_e_cell)
                        diag_qdot_h_pos = diag_qdot_h_pos + max(0._wp, qdot_h_limited)
                        diag_qdot_h_neg = diag_qdot_h_neg + min(0._wp, qdot_h_limited)
                        diag_qdot_e_pos = diag_qdot_e_pos + max(0._wp, qdot_e_limited)
                        diag_qdot_e_neg = diag_qdot_e_neg + min(0._wp, qdot_e_limited)
                        diag_max_raw_qdot_h = max(diag_max_raw_qdot_h, qdot_h_cell)
                        if (heat_scale < 1._wp) then
                            diag_limited_cell_count = diag_limited_cell_count + 1._wp
                            diag_scale_min = min(diag_scale_min, heat_scale)
                            diag_scale_sum_limited = diag_scale_sum_limited + heat_scale
                        end if
                    end if

                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (heat_diag_active) then
            if (diag_limited_cell_count > 0._wp) then
                diag_scale_mean_limited = diag_scale_sum_limited/diag_limited_cell_count
            else
                diag_scale_min = 1._wp
                diag_scale_mean_limited = 1._wp
            end if
            $:GPU_UPDATE(host='[q_T_sf%sf]')
            do eqn = 1, sys_size
                $:GPU_UPDATE(host='[q_cons_qp(eqn)%sf,q_prim_qp(eqn)%sf]')
            end do
            call s_print_reaction_heat_cell_diag(q_cons_qp, q_T_sf, q_prim_qp, bounds, t_step, stage, heat_limit_frac)

            print '(" REACTION_HEAT_DIAG rank=", I6, " t_step=", I8, " stage=", I4, &
                    &" limit_frac=", ES16.6, " limited_cell_count=", ES16.6, &
                    &" scale_min=", ES16.6, " scale_mean_limited=", ES16.6, &
                    &" raw_qdot_h_pos_sum=", ES16.6, " limited_qdot_h_pos_sum=", ES16.6, &
                    &" raw_qdot_h_neg_sum=", ES16.6, " limited_qdot_h_neg_sum=", ES16.6, &
                    &" raw_qdot_e_pos_sum=", ES16.6, " limited_qdot_e_pos_sum=", ES16.6, &
                    &" raw_qdot_e_neg_sum=", ES16.6, " limited_qdot_e_neg_sum=", ES16.6, &
                    &" raw_src_c12h26_sum=", ES16.6, " limited_src_c12h26_sum=", ES16.6, &
                    &" raw_src_o2_sum=", ES16.6, " limited_src_o2_sum=", ES16.6, &
                    &" raw_src_co2_sum=", ES16.6, " limited_src_co2_sum=", ES16.6, &
                    &" raw_src_h2o_sum=", ES16.6, " limited_src_h2o_sum=", ES16.6, &
                    &" raw_heat_E_sum=", ES16.6, " limited_heat_E_sum=", ES16.6, &
                    &" heat_partial_sum=", ES16.6, " partial_skip_count=", ES16.6, &
                    &" T_clamp_count=", ES16.6, " max_raw_qdot_h=", ES16.6, &
                    &" T_max=", ES16.6, " p_max=", ES16.6, &
                    &" energy_application=", A)', &
                proc_rank, t_step, stage, heat_limit_frac, diag_limited_cell_count, &
                diag_scale_min, diag_scale_mean_limited, &
                diag_raw_qdot_h_pos, diag_qdot_h_pos, diag_raw_qdot_h_neg, diag_qdot_h_neg, &
                diag_raw_qdot_e_pos, diag_qdot_e_pos, diag_raw_qdot_e_neg, diag_qdot_e_neg, &
                diag_raw_src_c12h26, diag_src_c12h26, diag_raw_src_o2, diag_src_o2, &
                diag_raw_src_co2, diag_src_co2, diag_raw_src_h2o, diag_src_h2o, &
                diag_raw_heat_applied, diag_heat_applied, diag_partial_heat_applied, &
                diag_partial_heat_skipped, diag_T_clamp_count, diag_max_raw_qdot_h, &
                diag_T_max, diag_p_max, &
                merge("TOTAL_AND_GAS_PARTIALS", "NONE                  ", apply_heat_active)
            call flush(output_unit)
        end if

    end subroutine s_compute_chemistry_reaction_flux

    subroutine s_chemistry_reaction_substep(q_cons_vf, q_T_sf, dtime, bounds, t_step)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        type(scalar_field), intent(inout) :: q_T_sf
        real(wp), intent(in) :: dtime
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds
        integer, intent(in) :: t_step

        integer :: x, y, z, eqn, s, nsub
        integer :: abort_flag, pre_norm_flag, post_repair_flag, gas_idx, fluid_id, count_g
        integer :: local_abort_reason, local_abort_species, local_fluid_id
        integer :: diag_abort_reason, diag_abort_x, diag_abort_y, diag_abort_z
        integer :: diag_abort_species, diag_abort_fluid_id
        real(wp) :: rho_g, alpha_g, rho_react, rho_species
        real(wp) :: rho, energy, T, T_new, dt_sub, Ysum, Ysum0, Yerr0
        real(wp) :: Ysum_raw, Ysum_repaired, Ysum_write, Ywrite, closure_delta
        real(wp) :: r, r2, wr, mw, loss_i, prod_p, loss_p, Lbar, pbar
        real(wp) :: source_vol_scale
        real(wp) :: minY, minY_after, diag_pre_sumY_err, diag_post_sumY_err
        real(wp) :: diag_minY, diag_species_mass_before, diag_species_mass_after
        real(wp) :: diag_heat_pos_sum, diag_heat_neg_sum, diag_heat_abs_sum
        real(wp) :: local_abort_score, local_pre_sumY, local_post_sumY
        real(wp) :: local_pre_minY, local_post_minY
        real(wp) :: local_Y_before, local_Y_after, local_pressure
        real(wp) :: diag_abort_score, diag_abort_pre_sumY, diag_abort_post_sumY
        real(wp) :: diag_abort_pre_minY, diag_abort_post_minY
        real(wp) :: diag_abort_Y_before, diag_abort_Y_after
        real(wp) :: diag_abort_alpha_g, diag_abort_rho_stored, diag_abort_rho_intrinsic
        real(wp) :: diag_abort_pressure, diag_abort_temperature
        real(wp) :: diag_pre_norm_score, diag_pre_norm_sumY, diag_pre_norm_minY
        real(wp) :: diag_pre_norm_repaired_sumY
        real(wp) :: diag_post_repair_score, diag_post_repair_sumY_before, diag_post_repair_sumY_after
        real(wp) :: diag_post_repair_minY_before, diag_post_repair_minY_after
        real(wp) :: local_post_repair_score, local_post_repair_sumY_before, local_post_repair_minY_before
        real(wp) :: h_k, qdot_h_sub, mass_rate_k, heat_sub
        real(wp) :: heat_weight, heat_weight_denom
        integer :: diag_pre_norm_x, diag_pre_norm_y, diag_pre_norm_z, diag_pre_norm_species
        integer :: diag_post_repair_x, diag_post_repair_y, diag_post_repair_z
        real(wp), parameter :: y_floor = 1.e-16_wp
        logical :: model3_gas, local_species_finite
        character(len=32) :: diag_abort_reason_label

        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(10) :: Ys, Ys_rate, Ys_cons, Ys_initial, cdot, ddot, y0, prod0, Lloss, alp, h_rt
        #:else
            real(wp), dimension(num_species) :: Ys, Ys_rate, Ys_cons, Ys_initial, cdot, ddot, y0, prod0, Lloss, alp, h_rt
        #:endif

        if (chem_params%reaction_substeps <= 0) return

        model3_gas = num_fluids > 1 .and. model_eqns == 3
        nsub = chem_params%reaction_substeps
        if (nsub < 1) return
        dt_sub = dtime/real(nsub, wp)

        diag_pre_sumY_err = 0._wp
        diag_post_sumY_err = 0._wp
        diag_minY = huge(1._wp)
        diag_species_mass_before = 0._wp
        diag_species_mass_after = 0._wp
        diag_heat_pos_sum = 0._wp
        diag_heat_neg_sum = 0._wp
        diag_heat_abs_sum = 0._wp
        abort_flag = 0
        pre_norm_flag = 0
        diag_abort_score = -1._wp
        diag_abort_reason = 0
        diag_abort_x = -1
        diag_abort_y = -1
        diag_abort_z = -1
        diag_abort_species = -1
        diag_abort_fluid_id = -1
        diag_abort_pre_sumY = 0._wp
        diag_abort_post_sumY = 0._wp
        diag_abort_pre_minY = 0._wp
        diag_abort_post_minY = 0._wp
        diag_abort_Y_before = 0._wp
        diag_abort_Y_after = 0._wp
        diag_abort_alpha_g = 0._wp
        diag_abort_rho_stored = 0._wp
        diag_abort_rho_intrinsic = 0._wp
        diag_abort_pressure = 0._wp
        diag_abort_temperature = 0._wp
        diag_pre_norm_score = -1._wp
        diag_pre_norm_sumY = 0._wp
        diag_pre_norm_minY = 0._wp
        diag_pre_norm_repaired_sumY = 0._wp
        diag_pre_norm_x = -1
        diag_pre_norm_y = -1
        diag_pre_norm_z = -1
        diag_pre_norm_species = -1
        post_repair_flag = 0
        diag_post_repair_score = -1._wp
        diag_post_repair_sumY_before = 0._wp
        diag_post_repair_sumY_after = 0._wp
        diag_post_repair_minY_before = 0._wp
        diag_post_repair_minY_after = 0._wp
        diag_post_repair_x = -1
        diag_post_repair_y = -1
        diag_post_repair_z = -1

        $:GPU_PARALLEL_LOOP(collapse=3, &
            private='[Ys, Ys_rate, Ys_cons, Ys_initial, cdot, ddot, y0, prod0, Lloss, alp, h_rt, eqn, s, gas_idx, fluid_id, count_g, local_abort_reason, local_abort_species, local_fluid_id, rho_g, alpha_g, rho_react, rho_species, rho, energy, T, T_new, Ysum, Ysum0, Yerr0, Ysum_raw, Ysum_repaired, Ysum_write, Ywrite, closure_delta, r, r2, wr, mw, loss_i, prod_p, loss_p, Lbar, pbar, source_vol_scale, minY, minY_after, local_abort_score, local_pre_sumY, local_post_sumY, local_pre_minY, local_post_minY, local_Y_before, local_Y_after, local_pressure, local_species_finite, local_post_repair_score, local_post_repair_sumY_before, local_post_repair_minY_before, h_k, qdot_h_sub, mass_rate_k, heat_sub, heat_weight, heat_weight_denom]', &
            reduction='[[diag_pre_sumY_err, diag_post_sumY_err], [diag_species_mass_before, diag_species_mass_after, diag_heat_pos_sum, diag_heat_neg_sum, diag_heat_abs_sum], [abort_flag, pre_norm_flag, post_repair_flag], [diag_minY]]', &
            reductionOp='[MAX, +, MAX, MIN]', &
            copy='[diag_abort_score, diag_abort_reason, diag_abort_x, diag_abort_y, diag_abort_z, diag_abort_species, diag_abort_fluid_id, diag_abort_pre_sumY, diag_abort_post_sumY, diag_abort_pre_minY, diag_abort_post_minY, diag_abort_Y_before, diag_abort_Y_after, diag_abort_alpha_g, diag_abort_rho_stored, diag_abort_rho_intrinsic, diag_abort_pressure, diag_abort_temperature, diag_pre_norm_score, diag_pre_norm_sumY, diag_pre_norm_minY, diag_pre_norm_repaired_sumY, diag_pre_norm_x, diag_pre_norm_y, diag_pre_norm_z, diag_pre_norm_species, diag_post_repair_score, diag_post_repair_sumY_before, diag_post_repair_sumY_after, diag_post_repair_minY_before, diag_post_repair_minY_after, diag_post_repair_x, diag_post_repair_y, diag_post_repair_z]', &
            copyin='[bounds, dt_sub, nsub, model3_gas]')
        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end
                    local_abort_reason = 0
                    local_abort_species = -1
                    local_abort_score = -1._wp
                    local_pre_sumY = 0._wp
                    local_post_sumY = 0._wp
                    local_pre_minY = huge(1._wp)
                    local_post_minY = huge(1._wp)
                    local_Y_before = 0._wp
                    local_Y_after = 0._wp
                    local_pressure = 0._wp
                    local_fluid_id = 1
                    T = 0._wp
                    T_new = 0._wp
                    local_post_repair_score = -1._wp
                    local_post_repair_sumY_before = 0._wp
                    local_post_repair_minY_before = 0._wp

                    if (model3_gas) then
                        call s_compute_chemistry_gas_alpha_density(q_cons_vf, x, y, z, rho_g, alpha_g, rho_react)
                        rho_species = rho_g
                        source_vol_scale = alpha_g
                        if (chem_gas_num_fluids <= 0) then
                            local_fluid_id = chem_gas_fluid_id
                        else
                            local_fluid_id = chem_gas_fluid_ids(1)
                        end if
                        if (rho_g <= chem_rho_g_min .or. rho_react <= chem_rho_g_min .or. alpha_g <= sgm_eps) cycle
                    else
                        rho = q_cons_vf(contxb)%sf(x, y, z)
                        rho_species = rho
                        rho_react = rho
                        source_vol_scale = 1._wp
                        alpha_g = 1._wp
                        if (rho <= chem_rho_g_min) cycle
                    end if

                    Ysum = 0._wp
                    Ysum_raw = 0._wp
                    minY = huge(1._wp)
                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = chemxb, chemxe
                        Ys_cons(eqn - chemxb + 1) = q_cons_vf(eqn)%sf(x, y, z)/rho_species
                        Ys(eqn - chemxb + 1) = Ys_cons(eqn - chemxb + 1)
                        Ysum_raw = Ysum_raw + Ys_cons(eqn - chemxb + 1)
                        if (Ys_cons(eqn - chemxb + 1) < minY) then
                            minY = Ys_cons(eqn - chemxb + 1)
                            local_abort_species = eqn - chemxb + 1
                        end if
                        diag_species_mass_before = diag_species_mass_before + q_cons_vf(eqn)%sf(x, y, z)
                        Ys(eqn - chemxb + 1) = max(0._wp, Ys(eqn - chemxb + 1))
                        Ysum = Ysum + Ys(eqn - chemxb + 1)
                    end do
                    Ysum_repaired = Ysum

                    Yerr0 = max(abs(Ysum_raw - 1._wp), abs(Ysum_repaired - 1._wp))
                    local_pre_sumY = Ysum_raw
                    local_pre_minY = minY
                    diag_pre_sumY_err = max(diag_pre_sumY_err, Yerr0)
                    diag_minY = min(diag_minY, minY)
                    if (minY < -aqss_pre_negative_abort_tol) then
                        abort_flag = 1
                        if (abs(minY) > local_abort_score) then
                            local_abort_reason = 1
                            if (local_abort_species < 1) local_abort_species = 1
                            local_abort_score = abs(minY)
                            local_Y_before = minY
                            local_Y_after = minY
                        end if
                        if (local_abort_reason > 0 .and. local_abort_score > diag_abort_score) then
                            diag_abort_score = local_abort_score
                            diag_abort_reason = local_abort_reason
                            diag_abort_x = x
                            diag_abort_y = y
                            diag_abort_z = z
                            diag_abort_species = local_abort_species
                            diag_abort_fluid_id = local_fluid_id
                            diag_abort_pre_sumY = local_pre_sumY
                            diag_abort_post_sumY = local_post_sumY
                            diag_abort_pre_minY = local_pre_minY
                            diag_abort_post_minY = local_post_minY
                            diag_abort_Y_before = local_Y_before
                            diag_abort_Y_after = local_Y_after
                            diag_abort_alpha_g = alpha_g
                            diag_abort_rho_stored = rho_species
                            diag_abort_rho_intrinsic = rho_react
                            diag_abort_pressure = local_pressure
                            diag_abort_temperature = T
                        end if
                        cycle
                    end if
                    if ((.not. s_is_finite_wp(Ysum_raw)) .or. (.not. s_is_finite_wp(Ysum_repaired)) .or. &
                        Yerr0 > aqss_pre_sumY_temp_norm_tol .or. Ysum_repaired <= y_floor) then
                        abort_flag = 1
                        if ((.not. s_is_finite_wp(Ysum_raw)) .or. (.not. s_is_finite_wp(Ysum_repaired))) then
                            local_abort_reason = 2
                            local_abort_species = 1
                            local_abort_score = huge(1._wp)
                            local_Y_before = Ys(1)
                            local_Y_after = Ys(1)
                        elseif (max(Yerr0, 0._wp) > local_abort_score) then
                            local_abort_reason = 2
                            local_abort_species = 1
                            local_abort_score = max(Yerr0, 0._wp)
                            local_Y_before = Ys(1)
                            local_Y_after = Ys(1)
                        end if
                        if (local_abort_reason > 0 .and. local_abort_score > diag_abort_score) then
                            diag_abort_score = local_abort_score
                            diag_abort_reason = local_abort_reason
                            diag_abort_x = x
                            diag_abort_y = y
                            diag_abort_z = z
                            diag_abort_species = local_abort_species
                            diag_abort_fluid_id = local_fluid_id
                            diag_abort_pre_sumY = local_pre_sumY
                            diag_abort_post_sumY = local_post_sumY
                            diag_abort_pre_minY = local_pre_minY
                            diag_abort_post_minY = local_post_minY
                            diag_abort_Y_before = local_Y_before
                            diag_abort_Y_after = local_Y_after
                            diag_abort_alpha_g = alpha_g
                            diag_abort_rho_stored = rho_species
                            diag_abort_rho_intrinsic = rho_react
                            diag_abort_pressure = local_pressure
                            diag_abort_temperature = T
                        end if
                        cycle
                    end if

                    if (Yerr0 > aqss_sumY_norm_tol) then
                        pre_norm_flag = 1
                        if (Yerr0 > diag_pre_norm_score) then
                            diag_pre_norm_score = Yerr0
                            diag_pre_norm_sumY = Ysum_raw
                            diag_pre_norm_minY = minY
                            diag_pre_norm_repaired_sumY = 1._wp
                            diag_pre_norm_x = x
                            diag_pre_norm_y = y
                            diag_pre_norm_z = z
                            diag_pre_norm_species = local_abort_species
                        end if
                    end if
                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = 1, num_species
                        Ys(eqn) = Ys(eqn)/Ysum_repaired
                        Ys_initial(eqn) = Ys(eqn)
                    end do

                    if (model3_gas) then
                        call s_compute_chemistry_reactor_temperature(q_cons_vf, q_T_sf, x, y, z, rho_react, Ys, T)
                        call s_compute_chemistry_gas_pressure(q_cons_vf, x, y, z, local_pressure)
                    else
                        energy = q_cons_vf(E_idx)%sf(x, y, z)/rho_react
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = momxb, momxe
                            energy = energy - 0.5_wp*(q_cons_vf(eqn)%sf(x, y, z)/rho_react)**2
                        end do
                        T = q_T_sf%sf(x, y, z)
                        call get_temperature(energy, T, Ys, .true., T_new)
                        T = min(max(T_new, chem_T_min), chem_T_max)
                    end if

                    do s = 1, nsub
                        Ysum0 = 0._wp
                        local_pre_minY = huge(1._wp)
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = 1, num_species
                            y0(eqn) = Ys(eqn)
                            Ysum0 = Ysum0 + y0(eqn)
                            local_pre_minY = min(local_pre_minY, y0(eqn))
                        end do
                        local_pre_sumY = Ysum0

                        if (abs(Ysum0 - 1._wp) <= aqss_sumY_norm_tol .and. Ysum0 > y_floor) then
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = y0(eqn)/Ysum0
                            end do
                        elseif (abs(Ysum0 - 1._wp) <= aqss_sumY_abort_tol) then
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = y0(eqn)
                            end do
                        else
                            abort_flag = 1
                            if (abs(Ysum0 - 1._wp) > local_abort_score) then
                                local_abort_reason = 3
                                local_abort_species = 1
                                local_abort_score = abs(Ysum0 - 1._wp)
                                local_Y_before = y0(1)
                                local_Y_after = y0(1)
                            end if
                            cycle
                        end if

                        call get_creation_destruction_rates(rho_react, T, Ys_rate, cdot, ddot)
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = 1, num_species
                            #:if USING_AMD
                                mw = molecular_weights_nonparameter(eqn)
                            #:else
                                mw = molecular_weights(eqn)
                            #:endif
                            wr = mw/rho_react
                            prod0(eqn) = wr*cdot(eqn)
                            loss_i = wr*ddot(eqn)
                            Lloss(eqn) = loss_i/max(Ys_rate(eqn), y_floor)
                            r = dt_sub*Lloss(eqn)
                            r2 = r*r
                            alp(eqn) = (180._wp + 60._wp*r + 11._wp*r2 + r2*r)/ &
                                       (360._wp + 60._wp*r + 12._wp*r2 + r2*r)
                            Ys(eqn) = y0(eqn) + dt_sub*(prod0(eqn) - loss_i)/(1._wp + alp(eqn)*dt_sub*Lloss(eqn))
                            if (Ys(eqn) < -aqss_post_negative_repair_tol) then
                                abort_flag = 1
                                if (abs(Ys(eqn)) > local_abort_score) then
                                    local_abort_reason = 4
                                    local_abort_species = eqn
                                    local_abort_score = abs(Ys(eqn))
                                    local_Y_before = y0(eqn)
                                    local_Y_after = Ys(eqn)
                                end if
                            end if
                            Ys(eqn) = max(0._wp, Ys(eqn))
                        end do

                        Ysum = 0._wp
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = 1, num_species
                            Ysum = Ysum + Ys(eqn)
                        end do
                        if (abs(Ysum - 1._wp) <= aqss_sumY_norm_tol .and. Ysum > y_floor) then
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = Ys(eqn)/Ysum
                            end do
                        else
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = Ys(eqn)
                            end do
                        end if

                        if (model3_gas) then
                            call s_compute_chemistry_reactor_temperature(q_cons_vf, q_T_sf, x, y, z, rho_react, Ys_rate, T_new)
                        else
                            call get_temperature(energy, T, Ys_rate, .true., T_new)
                            T_new = min(max(T_new, chem_T_min), chem_T_max)
                        end if

                        call get_creation_destruction_rates(rho_react, T_new, Ys_rate, cdot, ddot)
                        qdot_h_sub = 0._wp
                        if (chem_reaction_heat_enable .and. model3_gas) call get_species_enthalpies_rt(T_new, h_rt)
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = 1, num_species
                            #:if USING_AMD
                                mw = molecular_weights_nonparameter(eqn)
                            #:else
                                mw = molecular_weights(eqn)
                            #:endif
                            wr = mw/rho_react
                            prod_p = wr*cdot(eqn)
                            loss_p = wr*ddot(eqn)
                            Lbar = 0.5_wp*(Lloss(eqn) + loss_p/max(Ys_rate(eqn), y_floor))
                            pbar = alp(eqn)*prod_p + (1._wp - alp(eqn))*prod0(eqn)
                            Ys(eqn) = y0(eqn) + dt_sub*(pbar - Lbar*y0(eqn))/(1._wp + alp(eqn)*dt_sub*Lbar)
                            if (Ys(eqn) < -aqss_post_negative_repair_tol) then
                                abort_flag = 1
                                if (abs(Ys(eqn)) > local_abort_score) then
                                    local_abort_reason = 5
                                    local_abort_species = eqn
                                    local_abort_score = abs(Ys(eqn))
                                    local_Y_before = y0(eqn)
                                    local_Y_after = Ys(eqn)
                                end if
                            end if
                            Ys(eqn) = max(0._wp, Ys(eqn))
                            if (chem_reaction_heat_enable .and. model3_gas) then
                                mass_rate_k = rho_react*(Ys(eqn) - y0(eqn))/dt_sub
                                h_k = h_rt(eqn)*gas_constant*T_new/mw
                                qdot_h_sub = qdot_h_sub - h_k*mass_rate_k
                            end if
                        end do

                        if (chem_reaction_heat_enable .and. model3_gas .and. s_is_finite_wp(qdot_h_sub)) then
                            heat_sub = source_vol_scale*qdot_h_sub*dt_sub
                            q_cons_vf(E_idx)%sf(x, y, z) = q_cons_vf(E_idx)%sf(x, y, z) + heat_sub
                            diag_heat_pos_sum = diag_heat_pos_sum + max(heat_sub, 0._wp)
                            diag_heat_neg_sum = diag_heat_neg_sum + min(heat_sub, 0._wp)
                            diag_heat_abs_sum = diag_heat_abs_sum + abs(heat_sub)

                            heat_weight_denom = 0._wp
                            count_g = max(1, chem_gas_num_fluids)
                            $:GPU_LOOP(parallelism='[seq]')
                            do gas_idx = 1, count_g
                                if (chem_gas_num_fluids <= 0) then
                                    fluid_id = chem_gas_fluid_id
                                else
                                    fluid_id = chem_gas_fluid_ids(gas_idx)
                                end if
                                heat_weight_denom = heat_weight_denom + &
                                                    q_cons_vf(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                    s_reaction_heat_gamma(fluid_id)
                            end do

                            if (heat_weight_denom > sgm_eps) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do gas_idx = 1, count_g
                                    if (chem_gas_num_fluids <= 0) then
                                        fluid_id = chem_gas_fluid_id
                                    else
                                        fluid_id = chem_gas_fluid_ids(gas_idx)
                                    end if
                                    heat_weight = q_cons_vf(advxb + fluid_id - 1)%sf(x, y, z)* &
                                                  s_reaction_heat_gamma(fluid_id)/heat_weight_denom
                                    q_cons_vf(intxb + fluid_id - 1)%sf(x, y, z) = &
                                        q_cons_vf(intxb + fluid_id - 1)%sf(x, y, z) + heat_sub*heat_weight
                                end do
                            end if
                        end if

                        Ysum = 0._wp
                        minY_after = huge(1._wp)
                        local_species_finite = .true.
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = 1, num_species
                            Ysum = Ysum + Ys(eqn)
                            minY_after = min(minY_after, Ys(eqn))
                            if (.not. s_is_finite_wp(Ys(eqn))) local_species_finite = .false.
                        end do
                        local_post_sumY = Ysum
                        local_post_minY = minY_after
                        if (local_species_finite .and. s_is_finite_wp(Ysum) .and. &
                            minY_after >= -aqss_post_negative_repair_tol .and. &
                            abs(Ysum - 1._wp) <= aqss_post_sumY_repair_tol .and. &
                            Ysum > y_floor .and. &
                            (minY_after < 0._wp .or. abs(Ysum - 1._wp) > aqss_sumY_abort_tol)) then
                            post_repair_flag = 1
                            local_post_repair_score = abs(Ysum - 1._wp)
                            local_post_repair_sumY_before = Ysum
                            local_post_repair_minY_before = minY_after
                            Ysum_repaired = 0._wp
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys(eqn) = max(0._wp, Ys(eqn))
                                Ysum_repaired = Ysum_repaired + Ys(eqn)
                            end do
                            if (Ysum_repaired > y_floor) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do eqn = 1, num_species
                                    Ys(eqn) = Ys(eqn)/Ysum_repaired
                                end do
                            end if
                            Ysum = 0._wp
                            minY_after = huge(1._wp)
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ysum = Ysum + Ys(eqn)
                                minY_after = min(minY_after, Ys(eqn))
                            end do
                            local_post_sumY = Ysum
                            local_post_minY = minY_after
                            if (local_post_repair_score > diag_post_repair_score) then
                                diag_post_repair_score = local_post_repair_score
                                diag_post_repair_sumY_before = local_post_repair_sumY_before
                                diag_post_repair_minY_before = local_post_repair_minY_before
                                diag_post_repair_sumY_after = Ysum
                                diag_post_repair_minY_after = minY_after
                                diag_post_repair_x = x
                                diag_post_repair_y = y
                                diag_post_repair_z = z
                            end if
                        end if
                        if ((.not. s_is_finite_wp(Ysum)) .or. abs(Ysum - 1._wp) > aqss_sumY_abort_tol) then
                            abort_flag = 1
                            if (max(abs(Ysum - 1._wp), 0._wp) > local_abort_score) then
                                local_abort_reason = 6
                                local_abort_species = 1
                                local_abort_score = max(abs(Ysum - 1._wp), 0._wp)
                                local_Y_before = y0(1)
                                local_Y_after = Ys(1)
                            end if
                        end if
                        diag_post_sumY_err = max(diag_post_sumY_err, abs(Ysum - 1._wp))
                        diag_minY = min(diag_minY, minY_after)

                        if (abs(Ysum - 1._wp) <= aqss_sumY_norm_tol .and. Ysum > y_floor) then
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = Ys(eqn)/Ysum
                            end do
                        else
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = 1, num_species
                                Ys_rate(eqn) = Ys(eqn)
                            end do
                        end if

                        if (model3_gas) then
                            call s_compute_chemistry_reactor_temperature(q_cons_vf, q_T_sf, x, y, z, rho_react, Ys_rate, T)
                        else
                            call get_temperature(energy, T, Ys_rate, .true., T_new)
                            T = min(max(T_new, chem_T_min), chem_T_max)
                        end if
                    end do

                    Ysum_write = 0._wp
                    local_post_minY = huge(1._wp)
                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = 1, num_species
                        Ywrite = Ys_cons(eqn) + Ys(eqn) - Ys_initial(eqn)
                        Ysum_write = Ysum_write + Ywrite
                        if (Ywrite < local_post_minY) local_post_minY = Ywrite
                        if (Ys_cons(eqn) >= -aqss_negative_roundoff_tol) then
                            if (Ywrite < -aqss_negative_roundoff_tol .and. abs(Ywrite) > local_abort_score) then
                                local_abort_reason = 7
                                local_abort_species = eqn
                                local_abort_score = abs(Ywrite)
                                local_Y_before = Ys_cons(eqn)
                                local_Y_after = Ywrite
                            end if
                        elseif (Ywrite < Ys_cons(eqn) - aqss_negative_roundoff_tol .and. &
                                abs(Ywrite - Ys_cons(eqn)) > local_abort_score) then
                            local_abort_reason = 8
                            local_abort_species = eqn
                            local_abort_score = abs(Ywrite - Ys_cons(eqn))
                            local_Y_before = Ys_cons(eqn)
                            local_Y_after = Ywrite
                        end if
                    end do
                    closure_delta = abs(Ysum_write - Ysum_raw)
                    if (closure_delta > aqss_sumY_abort_tol .and. closure_delta > local_abort_score) then
                        local_abort_reason = 9
                        local_abort_species = 1
                        local_abort_score = closure_delta
                        local_Y_before = Ys_cons(1)
                        local_Y_after = Ys_cons(1) + Ys(1) - Ys_initial(1)
                    end if
                    if (local_abort_reason == 0 .or. local_abort_reason >= 7) then
                        local_pre_sumY = Ysum_raw
                        local_post_sumY = Ysum_write
                        local_pre_minY = minY
                    end if

                    if (local_abort_reason > 0 .and. local_abort_score > diag_abort_score) then
                        diag_abort_score = local_abort_score
                        diag_abort_reason = local_abort_reason
                        diag_abort_x = x
                        diag_abort_y = y
                        diag_abort_z = z
                        diag_abort_species = local_abort_species
                        diag_abort_fluid_id = local_fluid_id
                        diag_abort_pre_sumY = local_pre_sumY
                        diag_abort_post_sumY = local_post_sumY
                        diag_abort_pre_minY = local_pre_minY
                        diag_abort_post_minY = local_post_minY
                        diag_abort_Y_before = local_Y_before
                        diag_abort_Y_after = local_Y_after
                        diag_abort_alpha_g = alpha_g
                        diag_abort_rho_stored = rho_species
                        diag_abort_rho_intrinsic = rho_react
                        diag_abort_pressure = local_pressure
                        diag_abort_temperature = T
                    end if

                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = chemxb, chemxe
                        q_cons_vf(eqn)%sf(x, y, z) = rho_species*(Ys_cons(eqn - chemxb + 1) + &
                                                                  Ys(eqn - chemxb + 1) - &
                                                                  Ys_initial(eqn - chemxb + 1))
                        diag_species_mass_after = diag_species_mass_after + q_cons_vf(eqn)%sf(x, y, z)
                    end do
                    q_T_sf%sf(x, y, z) = T
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (chem_reaction_heat_diag .and. s_reaction_heat_diag_active(t_step)) then
            print '(" AQSS_DIAG rank=", I6, " t_step=", I8, " nsub=", I8, &
                    &" max_pre_sumY_err=", ES16.6, " max_post_sumY_err=", ES16.6, &
                    &" minY=", ES16.6, " species_mass_before=", ES16.6, &
                    &" species_mass_after=", ES16.6, " heat_pos_sum=", ES16.6, &
                    &" heat_neg_sum=", ES16.6, " heat_abs_sum=", ES16.6)', &
                proc_rank, t_step, nsub, diag_pre_sumY_err, diag_post_sumY_err, &
                diag_minY, diag_species_mass_before, diag_species_mass_after, &
                diag_heat_pos_sum, diag_heat_neg_sum, diag_heat_abs_sum
            call flush (output_unit)
        end if

        if (pre_norm_flag > 0) then
            print '(" AQSS_PRE_REPAIR rank=", I6, " t_step=", I8, " i=", I8, &
                    &" j=", I8, " k=", I8, " min_species=", I4, &
                    &" minY_before=", ES16.6, " sumY_before=", ES16.6, &
                    &" repaired_sumY=", ES16.6)', &
                proc_rank, t_step, diag_pre_norm_x, diag_pre_norm_y, diag_pre_norm_z, &
                diag_pre_norm_species, diag_pre_norm_minY, diag_pre_norm_sumY, diag_pre_norm_repaired_sumY
            call flush (output_unit)
        end if

        if (post_repair_flag > 0) then
            print '(" AQSS_POST_REPAIR rank=", I6, " t_step=", I8, " i=", I8, &
                    &" j=", I8, " k=", I8, " sumY_before_repair=", ES16.6, &
                    &" sumY_after_repair=", ES16.6, " minY_before_repair=", ES16.6, &
                    &" minY_after_repair=", ES16.6)', &
                proc_rank, t_step, diag_post_repair_x, diag_post_repair_y, diag_post_repair_z, &
                diag_post_repair_sumY_before, diag_post_repair_sumY_after, &
                diag_post_repair_minY_before, diag_post_repair_minY_after
            call flush (output_unit)
        end if

        if (abort_flag > 0) then
            select case (diag_abort_reason)
                case (1)
                    diag_abort_reason_label = "pre_negative"
                case (2)
                    diag_abort_reason_label = "pre_closure"
                case (3)
                    diag_abort_reason_label = "substep_input_closure"
                case (4)
                    diag_abort_reason_label = "predictor_negative"
                case (5)
                    diag_abort_reason_label = "corrector_negative"
                case (6)
                    diag_abort_reason_label = "post_closure"
                case (7)
                    diag_abort_reason_label = "post_new_negative"
                case (8)
                    diag_abort_reason_label = "post_worse_negative"
                case (9)
                    diag_abort_reason_label = "post_closure_delta"
                case default
                    diag_abort_reason_label = "unknown"
            end select
            if (diag_abort_reason == 1 .or. diag_abort_reason == 2) then
                print '(" AQSS_PRE_ABORT rank=", I6, " t_step=", I8, &
                        &" reason=", I2, 1X, A, " i=", I8, " j=", I8, " k=", I8, &
                        &" species=", I4, " Y_before=", ES16.6, &
                        &" sumY_before=", ES16.6, " minY_before=", ES16.6, &
                        &" pre_sumY_tol=", ES16.6, " pre_minY_tol=", ES16.6)', &
                    proc_rank, t_step, diag_abort_reason, trim(diag_abort_reason_label), &
                    diag_abort_x, diag_abort_y, diag_abort_z, diag_abort_species, &
                    diag_abort_Y_before, diag_abort_pre_sumY, diag_abort_pre_minY, &
                    aqss_pre_sumY_temp_norm_tol, -aqss_pre_negative_abort_tol
                call flush (output_unit)
            end if
            if (diag_abort_reason >= 3) then
                print '(" AQSS_POST_ABORT rank=", I6, " t_step=", I8, &
                        &" reason=", I2, 1X, A, " i=", I8, " j=", I8, " k=", I8, &
                        &" species=", I4, " Y_before=", ES16.6, " Y_after=", ES16.6, &
                        &" sumY_before=", ES16.6, " sumY_after=", ES16.6, &
                        &" closure_delta=", ES16.6)', &
                    proc_rank, t_step, diag_abort_reason, trim(diag_abort_reason_label), &
                    diag_abort_x, diag_abort_y, diag_abort_z, diag_abort_species, &
                    diag_abort_Y_before, diag_abort_Y_after, diag_abort_pre_sumY, &
                    diag_abort_post_sumY, abs(diag_abort_post_sumY - diag_abort_pre_sumY)
                call flush (output_unit)
            end if
            print '(" AQSS_ABORT_DETAIL rank=", I6, " t_step=", I8, " nsub=", I8, &
                    &" reason=", I2, 1X, A, " i=", I8, " j=", I8, " k=", I8, &
                    &" fluid_id=", I4, " species=", I4, " species_name=unavailable")', &
                proc_rank, t_step, nsub, diag_abort_reason, trim(diag_abort_reason_label), &
                diag_abort_x, diag_abort_y, diag_abort_z, diag_abort_fluid_id, diag_abort_species
            print '(" AQSS_ABORT_SPECIES Y_before=", ES16.6, " Y_after=", ES16.6, &
                    &" sumY_before=", ES16.6, " sumY_after=", ES16.6, &
                    &" minY_before=", ES16.6, " minY_after=", ES16.6, &
                    &" max_abs_sumY_err=", ES16.6, " worst_score=", ES16.6)', &
                diag_abort_Y_before, diag_abort_Y_after, diag_abort_pre_sumY, diag_abort_post_sumY, &
                diag_abort_pre_minY, diag_abort_post_minY, &
                max(diag_pre_sumY_err, diag_post_sumY_err), diag_abort_score
            print '(" AQSS_ABORT_THERMO alpha_g=", ES16.6, " rho_g_stored=", ES16.6, &
                    &" rho_g_intrinsic=", ES16.6, " pressure=", ES16.6, &
                    &" temperature=", ES16.6, " heat_pos_sum=", ES16.6, &
                    &" heat_neg_sum=", ES16.6, " heat_abs_sum=", ES16.6)', &
                diag_abort_alpha_g, diag_abort_rho_stored, diag_abort_rho_intrinsic, &
                diag_abort_pressure, diag_abort_temperature, &
                diag_heat_pos_sum, diag_heat_neg_sum, diag_heat_abs_sum
            call flush (output_unit)
            if (diag_abort_reason == 1 .or. diag_abort_reason == 2) then
                call s_mpi_abort("AQSS pre-chemistry species state exceeded temporary normalization guardrail.")
            else
                call s_mpi_abort("AQSS found post-chemistry species closure or non-roundoff negative species.")
            end if
        end if

    end subroutine s_chemistry_reaction_substep

    !> @brief Computes species mass diffusion fluxes at cell interfaces using mixture-averaged diffusivities.
    subroutine s_compute_chemistry_diffusion_flux(idir, q_prim_qp, flux_src_vf, irx, iry, irz, t_step, stage)

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_qp
        type(scalar_field), dimension(sys_size), intent(inout) :: flux_src_vf
        type(int_bounds_info), intent(in) :: irx, iry, irz
        integer, intent(in), optional :: t_step, stage

        integer, intent(in) :: idir
        #:if chemistry_transport
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(10) :: Xs_L, Xs_R, Xs_cell, Ys_L, Ys_R, Ys_cell
            real(wp), dimension(10) :: mass_diffusivities_mixavg1, mass_diffusivities_mixavg2
            real(wp), dimension(10) :: mass_diffusivities_mixavg_Cell, dXk_dxi, h_l, h_r, h_k
            real(wp), dimension(10) :: Mass_Diffu_Flux, dYk_dxi
        #:else
            real(wp), dimension(num_species) :: Xs_L, Xs_R, Xs_cell, Ys_L, Ys_R, Ys_cell
            real(wp), dimension(num_species) :: mass_diffusivities_mixavg1, mass_diffusivities_mixavg2
            real(wp), dimension(num_species) :: mass_diffusivities_mixavg_Cell, dXk_dxi, h_l, h_r, h_k
            real(wp), dimension(num_species) :: Mass_Diffu_Flux, dYk_dxi
        #:endif

        real(wp) :: Mass_Diffu_Energy
        real(wp) :: MW_L, MW_R, MW_cell, Rgas_L, Rgas_R, T_L, T_R, P_L, P_R, rho_L, rho_R, rho_cell, rho_Vic
        real(wp) :: rho_old_L, rho_old_R, T_old_L, T_old_R
        real(wp) :: rho_g_intrinsic_L, rho_g_intrinsic_R, alpha_face
        real(wp) :: lambda_L, lambda_R, lambda_Cell, dT_dxi, grid_spacing
        real(wp) :: Cp_L, Cp_R
        real(wp) :: diffusivity_L, diffusivity_R, diffusivity_cell
        real(wp) :: hmix_L, hmix_R, dh_dxi

        integer :: x, y, z, i, n, eqn
        integer, dimension(3) :: offsets
        logical :: model3_gas_density_fix_enabled, model3_intrinsic_alpha_fix_enabled
        logical :: corrected_property_invalid, intrinsic_face_inactive, intrinsic_flux_invalid
        logical :: intrinsic_inactive_due_alpha_g, intrinsic_inactive_due_alpha_liq
        character(len=32) :: env_value
        integer :: env_status
        integer :: js_idx, max_abs_J_idx, max_abs_h_idx, neg_J_count
        real(wp) :: sumY_L, sumY_R, sumY_cell, alpha_g_L, alpha_g_R, rho_g_L, rho_g_R
        real(wp) :: alpha_liq_L, alpha_liq_R, rho_total_L, rho_total_R
        real(wp) :: J_sum, J_min, J_max, J_max_abs, J_O2, J_fuel, D_min, D_max
        real(wp) :: J_unweighted_min, J_unweighted_max, J_unweighted_max_abs
        real(wp) :: J_weighted_min, J_weighted_max, J_weighted_max_abs
        real(wp) :: h_min, h_max, h_max_abs, thermal_term
        real(wp) :: unweighted_energy_flux
        real(wp) :: local_max_species_flux, local_max_energy_flux, global_max_species_flux, global_max_energy_flux
        integer :: local_face_count, local_active_face_count, local_inactive_face_count
        integer :: local_inactive_alpha_g_count, local_inactive_alpha_liq_count
        integer :: global_face_count, global_active_face_count, global_inactive_face_count
        integer :: global_inactive_alpha_g_count, global_inactive_alpha_liq_count
        integer :: diff_failure_claimed, diff_failure_old_claim, diff_failure_code
        integer, dimension(3) :: diff_failure_ijk, diff_failure_int_data
        real(wp), dimension(32) :: diff_failure_data
        real(wp), parameter :: model3_diff_alpha_min = 0.5_wp
        real(wp), parameter :: model3_diff_alpha_liq_max = 0.5_wp

        isc1 = irx; isc2 = iry; isc3 = irz

        model3_gas_density_fix_enabled = .false.
        model3_intrinsic_alpha_fix_enabled = .false.
        env_value = ""
        call get_environment_variable("TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX", env_value, status=env_status)
        model3_gas_density_fix_enabled = env_status == 0 .and. trim(env_value) == "1"
        env_value = ""
        call get_environment_variable("TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX", env_value, status=env_status)
        model3_intrinsic_alpha_fix_enabled = env_status == 0 .and. trim(env_value) == "1" .and. model_eqns == 3

        $:GPU_UPDATE(device='[isc1,isc2,isc3]')

        if (chemistry .or. dummy) then

            ! Set offsets based on direction using array indexing
            offsets = 0
            offsets(idir) = 1
            ! Model 1: Mixture-Average Transport
            if (chem_params%transport_model == 1) then
                ! Note: Added 'i' and 'eqn' to private list.
                local_face_count = 0
                local_active_face_count = 0
                local_inactive_face_count = 0
                local_inactive_alpha_g_count = 0
                local_inactive_alpha_liq_count = 0
                local_max_species_flux = 0._wp
                local_max_energy_flux = 0._wp
                diff_failure_claimed = 0
                diff_failure_code = 0
                diff_failure_ijk = 0
                diff_failure_int_data = 0
                diff_failure_data = 0._wp
                #:set chem_diff_private = '[x,y,z,i,eqn,Ys_L,Ys_R,Ys_cell,Xs_L,Xs_R,' + &
                    & 'mass_diffusivities_mixavg1,mass_diffusivities_mixavg2,' + &
                    & 'mass_diffusivities_mixavg_Cell,h_l,h_r,Xs_cell,h_k,' + &
                    & 'dXk_dxi,Mass_Diffu_Flux,Mass_Diffu_Energy,MW_L,MW_R,' + &
                    & 'MW_cell,Rgas_L,Rgas_R,T_L,T_R,P_L,P_R,rho_L,rho_R,' + &
                    & 'rho_cell,rho_Vic,rho_old_L,rho_old_R,T_old_L,T_old_R,' + &
                    & 'rho_g_intrinsic_L,rho_g_intrinsic_R,alpha_face,lambda_L,' + &
                    & 'lambda_R,lambda_Cell,dT_dxi,grid_spacing,js_idx,' + &
                    & 'max_abs_J_idx,max_abs_h_idx,neg_J_count,sumY_L,sumY_R,' + &
                    & 'sumY_cell,alpha_g_L,alpha_g_R,rho_g_L,rho_g_R,' + &
                    & 'alpha_liq_L,alpha_liq_R,rho_total_L,rho_total_R,J_sum,' + &
                    & 'J_min,J_max,J_max_abs,J_O2,J_fuel,J_unweighted_min,' + &
                    & 'J_unweighted_max,J_unweighted_max_abs,J_weighted_min,' + &
                    & 'J_weighted_max,J_weighted_max_abs,D_min,D_max,h_min,' + &
                    & 'h_max,h_max_abs,thermal_term,unweighted_energy_flux,' + &
                    & 'intrinsic_face_inactive,intrinsic_inactive_due_alpha_g,' + &
                    & 'intrinsic_inactive_due_alpha_liq,intrinsic_flux_invalid,' + &
                    & 'corrected_property_invalid,diff_failure_old_claim]'
                #:set chem_diff_reduction = '[[local_face_count,local_active_face_count,' + &
                    & 'local_inactive_face_count,local_inactive_alpha_g_count,' + &
                    & 'local_inactive_alpha_liq_count], [local_max_species_flux,' + &
                    & 'local_max_energy_flux]]'
                #:set chem_diff_copy = '[local_face_count,local_active_face_count,' + &
                    & 'local_inactive_face_count,local_inactive_alpha_g_count,' + &
                    & 'local_inactive_alpha_liq_count,local_max_species_flux,' + &
                    & 'local_max_energy_flux,diff_failure_claimed,' + &
                    & 'diff_failure_code,diff_failure_ijk,diff_failure_int_data,' + &
                    & 'diff_failure_data]'
                $:GPU_PARALLEL_LOOP(collapse=3, &
                    private=chem_diff_private, reduction=chem_diff_reduction, &
                    reductionOp='[+, MAX]', copy=chem_diff_copy, copyin='[offsets]')
                do z = isc3%beg, isc3%end
                    do y = isc2%beg, isc2%end
                        do x = isc1%beg, isc1%end
	                            ! Calculate grid spacing using direction-based indexing
                            select case (idir)
                            case (1)
                                grid_spacing = x_cc(x + 1) - x_cc(x)
                            case (2)
                                grid_spacing = y_cc(y + 1) - y_cc(y)
                            case (3)
                                grid_spacing = z_cc(z + 1) - z_cc(z)
	                            end select

	                            ! Extract species mass fractions
	                            sumY_L = 0._wp
	                            sumY_R = 0._wp
	                            $:GPU_LOOP(parallelism='[seq]')
	                            do i = chemxb, chemxe
	                                Ys_L(i - chemxb + 1) = q_prim_qp(i)%sf(x, y, z)
	                                Ys_R(i - chemxb + 1) = q_prim_qp(i)%sf(x + offsets(1), y + offsets(2), z + offsets(3))
	                                Ys_cell(i - chemxb + 1) = 0.5_wp*(Ys_L(i - chemxb + 1) + Ys_R(i - chemxb + 1))
	                                sumY_L = sumY_L + Ys_L(i - chemxb + 1)
	                                sumY_R = sumY_R + Ys_R(i - chemxb + 1)
	                            end do
	                            sumY_cell = 0.5_wp*(sumY_L + sumY_R)

                            ! Calculate molecular weights and mole fractions
                            call get_mixture_molecular_weight(Ys_L, MW_L)
                            call get_mixture_molecular_weight(Ys_R, MW_R)
                            MW_cell = 0.5_wp*(MW_L + MW_R)

                            call get_mole_fractions(MW_L, Ys_L, Xs_L)
                            call get_mole_fractions(MW_R, Ys_R, Xs_R)

                            ! Calculate gas constants and thermodynamic properties
                            Rgas_L = gas_constant/MW_L
                            Rgas_R = gas_constant/MW_R

                            P_L = q_prim_qp(E_idx)%sf(x, y, z)
                            P_R = q_prim_qp(E_idx)%sf(x + offsets(1), y + offsets(2), z + offsets(3))

	                            rho_old_L = q_prim_qp(1)%sf(x, y, z)
	                            rho_old_R = q_prim_qp(1)%sf(x + offsets(1), y + offsets(2), z + offsets(3))

	                            alpha_g_L = 0._wp; alpha_g_R = 0._wp; rho_g_L = 0._wp; rho_g_R = 0._wp
	                            rho_total_L = 0._wp; rho_total_R = 0._wp
	                            do i = 1, num_fluids
	                                rho_total_L = rho_total_L + q_prim_qp(contxb + i - 1)%sf(x, y, z)
	                                rho_total_R = rho_total_R + q_prim_qp(contxb + i - 1)%sf(&
                                    x + offsets(1), y + offsets(2), z + offsets(3))
	                            end do
	                            alpha_liq_L = q_prim_qp(advxb)%sf(x, y, z)
	                            alpha_liq_R = q_prim_qp(advxb)%sf(x + offsets(1), y + offsets(2), z + offsets(3))
	                            if (chem_gas_num_fluids <= 0) then
	                                alpha_g_L = q_prim_qp(advxb + chem_gas_fluid_id - 1)%sf(x, y, z)
	                                alpha_g_R = q_prim_qp(advxb + chem_gas_fluid_id - 1)%sf(&
                                    x + offsets(1), y + offsets(2), z + offsets(3))
	                                rho_g_L = q_prim_qp(contxb + chem_gas_fluid_id - 1)%sf(x, y, z)
	                                rho_g_R = q_prim_qp(contxb + chem_gas_fluid_id - 1)%sf(&
                                    x + offsets(1), y + offsets(2), z + offsets(3))
	                            else
	                                do i = 1, chem_gas_num_fluids
	                                    alpha_g_L = alpha_g_L + q_prim_qp(advxb + chem_gas_fluid_ids(i) - 1)%sf(x, y, z)
	                                    alpha_g_R = alpha_g_R + &
                                        q_prim_qp(advxb + chem_gas_fluid_ids(i) - 1)%sf(&
                                            x + offsets(1), y + offsets(2), z + offsets(3))
	                                    rho_g_L = rho_g_L + q_prim_qp(contxb + chem_gas_fluid_ids(i) - 1)%sf(x, y, z)
	                                    rho_g_R = rho_g_R + &
                                        q_prim_qp(contxb + chem_gas_fluid_ids(i) - 1)%sf(&
                                            x + offsets(1), y + offsets(2), z + offsets(3))
	                                end do
	                            end if

	                            rho_g_intrinsic_L = 0._wp
	                            rho_g_intrinsic_R = 0._wp
	                            alpha_face = 1._wp
	                            intrinsic_face_inactive = .false.
	                            intrinsic_inactive_due_alpha_g = .false.
	                            intrinsic_inactive_due_alpha_liq = .false.
	                            if (model3_intrinsic_alpha_fix_enabled) then
	                                local_face_count = local_face_count + 1
	                                intrinsic_inactive_due_alpha_g = alpha_g_L /= alpha_g_L .or. alpha_g_R /= alpha_g_R .or. &
	                                                               alpha_g_L < model3_diff_alpha_min .or. &
                                                               alpha_g_R < model3_diff_alpha_min
	                                intrinsic_inactive_due_alpha_liq = alpha_liq_L /= alpha_liq_L .or. &
                                                                     alpha_liq_R /= alpha_liq_R .or. &
	                                                                 alpha_liq_L > model3_diff_alpha_liq_max .or. &
	                                                                 alpha_liq_R > model3_diff_alpha_liq_max
	                                intrinsic_face_inactive = intrinsic_inactive_due_alpha_g .or. intrinsic_inactive_due_alpha_liq
	                                if (alpha_g_L > 0._wp) rho_g_intrinsic_L = rho_g_L/alpha_g_L
	                                if (alpha_g_R > 0._wp) rho_g_intrinsic_R = rho_g_R/alpha_g_R
	                                if (intrinsic_face_inactive) then
	                                    local_inactive_face_count = local_inactive_face_count + 1
	                                    if (intrinsic_inactive_due_alpha_g) &
                                        local_inactive_alpha_g_count = local_inactive_alpha_g_count + 1
	                                    if (intrinsic_inactive_due_alpha_liq) &
                                        local_inactive_alpha_liq_count = local_inactive_alpha_liq_count + 1
	                                    flux_src_vf(E_idx)%sf(x, y, z) = 0._wp
	                                    $:GPU_LOOP(parallelism='[seq]')
	                                    do eqn = chemxb, chemxe
	                                        flux_src_vf(eqn)%sf(x, y, z) = 0._wp
	                                    end do
	                                    cycle
	                                end if
	                                corrected_property_invalid = .false.
		                                if (rho_g_L /= rho_g_L .or. rho_g_R /= rho_g_R .or. rho_g_L <= 0._wp .or. rho_g_R <= 0._wp) &
		                                    corrected_property_invalid = .true.
		                                if (rho_g_intrinsic_L /= rho_g_intrinsic_L .or. rho_g_intrinsic_R /= rho_g_intrinsic_R .or. &
		                                    rho_g_intrinsic_L <= 0._wp .or. rho_g_intrinsic_R <= 0._wp) &
		                                    corrected_property_invalid = .true.
		                                if (corrected_property_invalid) then
			                                    $:GPU_ATOMIC(atomic='capture')
			                                    diff_failure_old_claim = diff_failure_claimed
			                                    diff_failure_claimed = 1
			                                    $:END_GPU_ATOMIC()
			                                    if (diff_failure_old_claim == 0) then
		                                        diff_failure_code = 1
		                                        diff_failure_ijk = (/x, y, z/)
		                                        diff_failure_int_data = (/idir, 0, 0/)
		                                        diff_failure_data(5) = rho_g_L
		                                        diff_failure_data(6) = rho_g_R
		                                        diff_failure_data(7) = rho_g_intrinsic_L
		                                        diff_failure_data(8) = rho_g_intrinsic_R
		                                        diff_failure_data(9) = alpha_g_L
		                                        diff_failure_data(10) = alpha_g_R
		                                    end if
		                                    cycle
		                                end if
	                                local_active_face_count = local_active_face_count + 1
	                                alpha_face = min(alpha_g_L, alpha_g_R)
	                                rho_L = rho_g_intrinsic_L
	                                rho_R = rho_g_intrinsic_R
	                            else if (model3_gas_density_fix_enabled .and. model_eqns == 3) then
	                                rho_L = rho_g_L
	                                rho_R = rho_g_R
	                            else
	                                rho_L = rho_old_L
	                                rho_R = rho_old_R
	                            end if
	                            T_old_L = P_L/rho_old_L/Rgas_L
	                            T_old_R = P_R/rho_old_R/Rgas_R
	                            T_L = P_L/rho_L/Rgas_L
	                            T_R = P_R/rho_R/Rgas_R
	                            if ((model3_gas_density_fix_enabled .and. model_eqns == 3) .or. &
                                model3_intrinsic_alpha_fix_enabled) then
	                                corrected_property_invalid = .false.
	                                if (rho_L /= rho_L .or. rho_R /= rho_R .or. rho_L <= 0._wp .or. rho_R <= 0._wp) &
	                                    corrected_property_invalid = .true.
	                                if (T_L /= T_L .or. T_R /= T_R .or. T_L <= 1.e-6_wp .or. T_R <= 1.e-6_wp .or. &
	                                    T_L > 1.e8_wp .or. T_R > 1.e8_wp) corrected_property_invalid = .true.
		                                if (model3_intrinsic_alpha_fix_enabled .and. (T_L /= T_L .or. T_R /= T_R)) &
		                                    corrected_property_invalid = .true.
		                                if (corrected_property_invalid) then
			                                    $:GPU_ATOMIC(atomic='capture')
			                                    diff_failure_old_claim = diff_failure_claimed
			                                    diff_failure_claimed = 1
			                                    $:END_GPU_ATOMIC()
			                                    if (diff_failure_old_claim == 0) then
		                                        diff_failure_code = 2
		                                        diff_failure_ijk = (/x, y, z/)
		                                        diff_failure_int_data = (/idir, 0, 0/)
		                                        diff_failure_data(1) = P_L
		                                        diff_failure_data(2) = P_R
		                                        diff_failure_data(3) = rho_old_L
		                                        diff_failure_data(4) = rho_old_R
		                                        diff_failure_data(5) = rho_g_L
		                                        diff_failure_data(6) = rho_g_R
		                                        diff_failure_data(14) = T_old_L
		                                        diff_failure_data(15) = T_old_R
		                                        diff_failure_data(16) = T_L
		                                        diff_failure_data(17) = T_R
		                                    end if
		                                    cycle
		                                end if
		                            end if

                            rho_cell = 0.5_wp*(rho_L + rho_R)
                            dT_dxi = (T_R - T_L)/grid_spacing

                            ! Get transport properties
	                            call get_species_mass_diffusivities_mixavg(P_L, T_L, Ys_L, mass_diffusivities_mixavg1)
	                            call get_species_mass_diffusivities_mixavg(P_R, T_R, Ys_R, mass_diffusivities_mixavg2)

                            call get_mixture_thermal_conductivity_mixavg(T_L, Ys_L, lambda_L)
                            call get_mixture_thermal_conductivity_mixavg(T_R, Ys_R, lambda_R)

	                            call get_species_enthalpies_rt(T_L, h_l)
	                            call get_species_enthalpies_rt(T_R, h_r)

	                            if ((model3_gas_density_fix_enabled .and. model_eqns == 3) .or. &
                                model3_intrinsic_alpha_fix_enabled) then
	                                corrected_property_invalid = .false.
	                                if (lambda_L /= lambda_L .or. lambda_R /= lambda_R .or. lambda_L < 0._wp .or. lambda_R < 0._wp) &
	                                    corrected_property_invalid = .true.
	                                $:GPU_LOOP(parallelism='[seq]')
		                                do i = 1, num_species
		                                    if (mass_diffusivities_mixavg1(i) /= mass_diffusivities_mixavg1(i) .or. &
		                                        mass_diffusivities_mixavg2(i) /= mass_diffusivities_mixavg2(i) .or. &
		                                        mass_diffusivities_mixavg1(i) < 0._wp .or. mass_diffusivities_mixavg2(i) < 0._wp .or. &
		                                        h_l(i) /= h_l(i) .or. h_r(i) /= h_r(i)) corrected_property_invalid = .true.
		                                end do
		                                if (corrected_property_invalid) then
			                                    $:GPU_ATOMIC(atomic='capture')
			                                    diff_failure_old_claim = diff_failure_claimed
			                                    diff_failure_claimed = 1
			                                    $:END_GPU_ATOMIC()
			                                    if (diff_failure_old_claim == 0) then
		                                        diff_failure_code = 3
		                                        diff_failure_ijk = (/x, y, z/)
		                                        diff_failure_int_data = (/idir, 0, 0/)
		                                        diff_failure_data(1) = P_L
		                                        diff_failure_data(2) = P_R
		                                        diff_failure_data(3) = rho_old_L
		                                        diff_failure_data(4) = rho_old_R
		                                        diff_failure_data(5) = rho_g_L
		                                        diff_failure_data(6) = rho_g_R
		                                        diff_failure_data(9) = alpha_g_L
		                                        diff_failure_data(10) = alpha_g_R
		                                        diff_failure_data(11) = alpha_liq_L
		                                        diff_failure_data(12) = alpha_liq_R
		                                        diff_failure_data(13) = rho_total_L
		                                        diff_failure_data(14) = rho_total_R
		                                        diff_failure_data(16) = T_L
		                                        diff_failure_data(17) = T_R
		                                        diff_failure_data(18) = MW_L
		                                        diff_failure_data(19) = MW_R
		                                        diff_failure_data(20) = Rgas_L
		                                        diff_failure_data(21) = Rgas_R
		                                        diff_failure_data(22) = sumY_L
		                                        diff_failure_data(23) = sumY_R
		                                        diff_failure_data(24) = lambda_L
		                                        diff_failure_data(25) = lambda_R
		                                    end if
		                                    cycle
		                                end if
		                            end if

                            ! Calculate species properties and gradients
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = chemxb, chemxe
                                #:if USING_AMD
                                    h_l(i - chemxb + 1) = h_l(i - chemxb + 1)*gas_constant*T_L/ &
                                                          molecular_weights_nonparameter(i - chemxb + 1)
                                    h_r(i - chemxb + 1) = h_r(i - chemxb + 1)*gas_constant*T_R/ &
                                                          molecular_weights_nonparameter(i - chemxb + 1)
                                #:else
                                    h_l(i - chemxb + 1) = h_l(i - chemxb + 1)*gas_constant*T_L/molecular_weights(i - chemxb + 1)
                                    h_r(i - chemxb + 1) = h_r(i - chemxb + 1)*gas_constant*T_R/molecular_weights(i - chemxb + 1)
                                #:endif
                                Xs_cell(i - chemxb + 1) = 0.5_wp*(Xs_L(i - chemxb + 1) + Xs_R(i - chemxb + 1))
                                h_k(i - chemxb + 1) = 0.5_wp*(h_l(i - chemxb + 1) + h_r(i - chemxb + 1))
                                dXk_dxi(i - chemxb + 1) = (Xs_R(i - chemxb + 1) - Xs_L(i - chemxb + 1))/grid_spacing
                            end do

	                            ! Calculate mixture-averaged diffusivities
	                            D_min = huge(1._wp)
	                            D_max = -huge(1._wp)
	                            $:GPU_LOOP(parallelism='[seq]')
	                            do i = chemxb, chemxe
	                                mass_diffusivities_mixavg_Cell(i - chemxb + 1) = &
	                                    (mass_diffusivities_mixavg2(i - chemxb + 1) + &
                                     mass_diffusivities_mixavg1(i - chemxb + 1))/2.0_wp
	                                D_min = min(D_min, mass_diffusivities_mixavg_Cell(i - chemxb + 1))
	                                D_max = max(D_max, mass_diffusivities_mixavg_Cell(i - chemxb + 1))
	                            end do

                            lambda_Cell = 0.5_wp*(lambda_R + lambda_L)

                            ! Calculate mass diffusion fluxes
	                            rho_Vic = 0.0_wp
	                            Mass_Diffu_Energy = 0.0_wp
	                            J_sum = 0._wp; J_min = huge(1._wp); J_max = -huge(1._wp); J_max_abs = -1._wp
	                            J_O2 = 0._wp; J_fuel = 0._wp; max_abs_J_idx = -1; neg_J_count = 0

	                            $:GPU_LOOP(parallelism='[seq]')
	                            do eqn = chemxb, chemxe
                                #:if USING_AMD
                                    Mass_Diffu_Flux(eqn - chemxb + 1) = &
                                        rho_cell*mass_diffusivities_mixavg_Cell(eqn - chemxb + 1)* &
                                        molecular_weights_nonparameter(eqn - chemxb + 1)/MW_cell* &
                                        dXk_dxi(eqn - chemxb + 1)
                                #:else
                                    Mass_Diffu_Flux(eqn - chemxb + 1) = &
                                        rho_cell*mass_diffusivities_mixavg_Cell(eqn - chemxb + 1)* &
                                        molecular_weights(eqn - chemxb + 1)/MW_cell* &
                                        dXk_dxi(eqn - chemxb + 1)
	                                #:endif
	                                rho_Vic = rho_Vic + Mass_Diffu_Flux(eqn - chemxb + 1)
	                                Mass_Diffu_Energy = Mass_Diffu_Energy + h_k(eqn - chemxb + 1)*Mass_Diffu_Flux(eqn - chemxb + 1)
	                                J_sum = J_sum + Mass_Diffu_Flux(eqn - chemxb + 1)
	                                J_min = min(J_min, Mass_Diffu_Flux(eqn - chemxb + 1))
	                                J_max = max(J_max, Mass_Diffu_Flux(eqn - chemxb + 1))
	                                if (abs(Mass_Diffu_Flux(eqn - chemxb + 1)) > J_max_abs) then
	                                    J_max_abs = abs(Mass_Diffu_Flux(eqn - chemxb + 1))
	                                    max_abs_J_idx = eqn - chemxb + 1
	                                end if
	                                if (Mass_Diffu_Flux(eqn - chemxb + 1) < 0._wp) neg_J_count = neg_J_count + 1
	                                if (eqn - chemxb + 1 == 10) J_O2 = Mass_Diffu_Flux(eqn - chemxb + 1)
	                                if (eqn - chemxb + 1 == fuel_species_id) J_fuel = Mass_Diffu_Flux(eqn - chemxb + 1)
	                            end do

	                            ! Apply corrections for mass conservation
	                            $:GPU_LOOP(parallelism='[seq]')
	                            do eqn = chemxb, chemxe
	                                Mass_Diffu_Energy = Mass_Diffu_Energy - h_k(eqn - chemxb + 1)*Ys_cell(eqn - chemxb + 1)*rho_Vic
	                                Mass_Diffu_Flux(eqn - chemxb + 1) = &
                                    Mass_Diffu_Flux(eqn - chemxb + 1) - &
                                    rho_Vic*Ys_cell(eqn - chemxb + 1)
	                            end do
	                            J_sum = 0._wp; J_min = huge(1._wp); J_max = -huge(1._wp); J_max_abs = -1._wp
	                            J_O2 = 0._wp; J_fuel = 0._wp; max_abs_J_idx = -1; neg_J_count = 0
	                            h_min = huge(1._wp); h_max = -huge(1._wp); h_max_abs = -1._wp; max_abs_h_idx = -1
	                            $:GPU_LOOP(parallelism='[seq]')
	                            do eqn = chemxb, chemxe
	                                js_idx = eqn - chemxb + 1
	                                J_sum = J_sum + Mass_Diffu_Flux(js_idx)
	                                J_min = min(J_min, Mass_Diffu_Flux(js_idx))
	                                J_max = max(J_max, Mass_Diffu_Flux(js_idx))
	                                if (abs(Mass_Diffu_Flux(js_idx)) > J_max_abs) then
	                                    J_max_abs = abs(Mass_Diffu_Flux(js_idx))
	                                    max_abs_J_idx = js_idx
	                                end if
	                                if (Mass_Diffu_Flux(js_idx) < 0._wp) neg_J_count = neg_J_count + 1
	                                if (js_idx == 10) J_O2 = Mass_Diffu_Flux(js_idx)
	                                if (js_idx == fuel_species_id) J_fuel = Mass_Diffu_Flux(js_idx)
	                                h_min = min(h_min, h_k(js_idx))
	                                h_max = max(h_max, h_k(js_idx))
	                                if (abs(h_k(js_idx)) > h_max_abs) then
	                                    h_max_abs = abs(h_k(js_idx))
	                                    max_abs_h_idx = js_idx
	                                end if
	                            end do

	                            J_unweighted_min = J_min
	                            J_unweighted_max = J_max
	                            J_unweighted_max_abs = J_max_abs

	                            ! Add thermal conduction contribution
	                            thermal_term = lambda_Cell*dT_dxi
	                            Mass_Diffu_Energy = lambda_Cell*dT_dxi + Mass_Diffu_Energy
	                            unweighted_energy_flux = Mass_Diffu_Energy

	                            if (model3_intrinsic_alpha_fix_enabled) then
	                                $:GPU_LOOP(parallelism='[seq]')
	                                do eqn = chemxb, chemxe
	                                    Mass_Diffu_Flux(eqn - chemxb + 1) = alpha_face*Mass_Diffu_Flux(eqn - chemxb + 1)
	                                end do
	                                Mass_Diffu_Energy = alpha_face*Mass_Diffu_Energy
	                                thermal_term = alpha_face*thermal_term
	                                J_sum = 0._wp; J_min = huge(1._wp); J_max = -huge(1._wp); J_max_abs = -1._wp
	                                J_O2 = 0._wp; J_fuel = 0._wp; max_abs_J_idx = -1; neg_J_count = 0
	                                intrinsic_flux_invalid = .false.
	                                $:GPU_LOOP(parallelism='[seq]')
	                                do eqn = chemxb, chemxe
	                                    js_idx = eqn - chemxb + 1
	                                    J_sum = J_sum + Mass_Diffu_Flux(js_idx)
	                                    J_min = min(J_min, Mass_Diffu_Flux(js_idx))
	                                    J_max = max(J_max, Mass_Diffu_Flux(js_idx))
	                                    if (abs(Mass_Diffu_Flux(js_idx)) > J_max_abs) then
	                                        J_max_abs = abs(Mass_Diffu_Flux(js_idx))
	                                        max_abs_J_idx = js_idx
	                                    end if
	                                    if (Mass_Diffu_Flux(js_idx) < 0._wp) neg_J_count = neg_J_count + 1
	                                    if (js_idx == 10) J_O2 = Mass_Diffu_Flux(js_idx)
	                                    if (js_idx == fuel_species_id) J_fuel = Mass_Diffu_Flux(js_idx)
	                                    if (Mass_Diffu_Flux(js_idx) /= Mass_Diffu_Flux(js_idx)) intrinsic_flux_invalid = .true.
		                                end do
		                                J_weighted_min = J_min
		                                J_weighted_max = J_max
		                                J_weighted_max_abs = J_max_abs
		                                if (Mass_Diffu_Energy /= Mass_Diffu_Energy) intrinsic_flux_invalid = .true.
		                                if (intrinsic_flux_invalid) then
			                                    $:GPU_ATOMIC(atomic='capture')
			                                    diff_failure_old_claim = diff_failure_claimed
			                                    diff_failure_claimed = 1
			                                    $:END_GPU_ATOMIC()
			                                    if (diff_failure_old_claim == 0) then
		                                        diff_failure_code = 4
		                                        diff_failure_ijk = (/x, y, z/)
		                                        diff_failure_int_data = (/idir, max_abs_J_idx, max_abs_h_idx/)
		                                        diff_failure_data(26) = J_unweighted_min
		                                        diff_failure_data(27) = J_unweighted_max
		                                        diff_failure_data(28) = J_weighted_min
		                                        diff_failure_data(29) = J_weighted_max
		                                        diff_failure_data(30) = unweighted_energy_flux
		                                        diff_failure_data(31) = Mass_Diffu_Energy
		                                        diff_failure_data(32) = alpha_face
		                                    end if
		                                    cycle
		                                end if
	                                local_max_species_flux = max(local_max_species_flux, J_weighted_max_abs)
	                                local_max_energy_flux = max(local_max_energy_flux, abs(Mass_Diffu_Energy))
	                            else
	                                J_weighted_min = J_min
	                                J_weighted_max = J_max
	                                J_weighted_max_abs = J_max_abs
	                            end if

	                            ! Update flux arrays
	                            flux_src_vf(E_idx)%sf(x, y, z) = flux_src_vf(E_idx)%sf(x, y, z) - Mass_Diffu_Energy

	                            $:GPU_LOOP(parallelism='[seq]')
	                            do eqn = chemxb, chemxe
	                                flux_src_vf(eqn)%sf(x, y, z) = flux_src_vf(eqn)%sf(x, y, z) - Mass_Diffu_Flux(eqn - chemxb + 1)
	                            end do
	                        end do
	                    end do
	                end do
	                $:END_GPU_PARALLEL_LOOP()

                if (diff_failure_claimed /= 0) then
                    select case (diff_failure_code)
                    case (1)
                        write (output_unit, '(&
                            &"TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX_INVALID rank=",I0,&
                            &" direction=",I0," face_xyz=",3(I0,1X),&
                            &" reason=density alpha_g_L=",ES16.8," alpha_g_R=",ES16.8,&
                            &" rho_g_stored_L=",ES16.8," rho_g_stored_R=",ES16.8,&
                            &" rho_g_intrinsic_L=",ES16.8," rho_g_intrinsic_R=",ES16.8)') &
                            proc_rank, diff_failure_int_data(1), diff_failure_ijk, &
                            diff_failure_data(9), diff_failure_data(10), &
                            diff_failure_data(5), diff_failure_data(6), &
                            diff_failure_data(7), diff_failure_data(8)
                        call flush(output_unit)
                        call s_mpi_abort("TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX invalid gas density")
                    case (2)
                        write (output_unit, '(&
                            &"TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_INVALID rank=",I0,&
                            &" direction=",I0," face_xyz=",3(I0,1X),&
                            &" rho_old_L=",ES16.8," rho_old_R=",ES16.8,&
                            &" rho_g_L=",ES16.8," rho_g_R=",ES16.8,&
                            &" P_L=",ES16.8," P_R=",ES16.8,&
                            &" T_old_L=",ES16.8," T_old_R=",ES16.8,&
                            &" T_corr_L=",ES16.8," T_corr_R=",ES16.8)') &
                            proc_rank, diff_failure_int_data(1), diff_failure_ijk, &
                            diff_failure_data(3), diff_failure_data(4), &
                            diff_failure_data(5), diff_failure_data(6), &
                            diff_failure_data(1), diff_failure_data(2), &
                            diff_failure_data(14), diff_failure_data(15), &
                            diff_failure_data(16), diff_failure_data(17)
                        call flush(output_unit)
                        call s_mpi_abort("TEMP_CHEM_DIFFUSION_MODEL3_GAS_DENSITY_FIX "// &
                                         "invalid corrected density/temperature")
                    case (3)
                        write (output_unit, '(&
                            &"TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_INVALID_PROPERTY rank=",I0,&
                            &" direction=",I0," face_xyz=",3(I0,1X),&
                            &" P_L=",ES16.8," P_R=",ES16.8,&
                            &" rho_old_L=",ES16.8," rho_old_R=",ES16.8,&
                            &" rho_g_L=",ES16.8," rho_g_R=",ES16.8,&
                            &" rho_total_L=",ES16.8," rho_total_R=",ES16.8,&
                            &" alpha_g_L=",ES16.8," alpha_g_R=",ES16.8,&
                            &" alpha_liq_L=",ES16.8," alpha_liq_R=",ES16.8,&
                            &" MW_L=",ES16.8," MW_R=",ES16.8,&
                            &" Rgas_L=",ES16.8," Rgas_R=",ES16.8,&
                            &" T_corr_L=",ES16.8," T_corr_R=",ES16.8,&
                            &" sumY_L=",ES16.8," sumY_R=",ES16.8,&
                            &" lambda_L=",ES16.8," lambda_R=",ES16.8)') &
                            proc_rank, diff_failure_int_data(1), diff_failure_ijk, &
                            diff_failure_data(1), diff_failure_data(2), &
                            diff_failure_data(3), diff_failure_data(4), &
                            diff_failure_data(5), diff_failure_data(6), &
                            diff_failure_data(13), diff_failure_data(14), &
                            diff_failure_data(9), diff_failure_data(10), &
                            diff_failure_data(11), diff_failure_data(12), &
                            diff_failure_data(18), diff_failure_data(19), &
                            diff_failure_data(20), diff_failure_data(21), &
                            diff_failure_data(16), diff_failure_data(17), &
                            diff_failure_data(22), diff_failure_data(23), &
                            diff_failure_data(24), diff_failure_data(25)
                        call flush(output_unit)
                        call s_mpi_abort("TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX "// &
                                         "invalid corrected transport/enthalpy property")
                    case (4)
                        write (output_unit, '(&
                            &"TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_INVALID rank=",I0,&
                            &" direction=",I0," face_xyz=",3(I0,1X),&
                            &" reason=flux max_abs_J_idx=",I0," max_abs_h_idx=",I0,&
                            &" alpha_face=",ES16.8,&
                            &" J_unweighted_min=",ES16.8," J_unweighted_max=",ES16.8,&
                            &" J_weighted_min=",ES16.8," J_weighted_max=",ES16.8,&
                            &" energy_unweighted=",ES16.8," energy_weighted=",ES16.8)') &
                            proc_rank, diff_failure_int_data(1), diff_failure_ijk, &
                            diff_failure_int_data(2), diff_failure_int_data(3), &
                            diff_failure_data(32), diff_failure_data(26), &
                            diff_failure_data(27), diff_failure_data(28), &
                            diff_failure_data(29), diff_failure_data(30), &
                            diff_failure_data(31)
                        call flush(output_unit)
                        call s_mpi_abort("TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX invalid final flux")
                    case default
                        call s_mpi_abort("TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_FIX invalid unknown state")
                    end select
                end if

		                if (model3_intrinsic_alpha_fix_enabled) then
		                    if (num_procs > 1) then
		                        call s_mpi_allreduce_integer_sum(local_face_count, global_face_count)
	                        call s_mpi_allreduce_integer_sum(local_active_face_count, global_active_face_count)
	                        call s_mpi_allreduce_integer_sum(local_inactive_face_count, global_inactive_face_count)
	                        call s_mpi_allreduce_integer_sum(local_inactive_alpha_g_count, global_inactive_alpha_g_count)
	                        call s_mpi_allreduce_integer_sum(local_inactive_alpha_liq_count, global_inactive_alpha_liq_count)
	                        call s_mpi_allreduce_max(local_max_species_flux, global_max_species_flux)
	                        call s_mpi_allreduce_max(local_max_energy_flux, global_max_energy_flux)
	                    else
	                        global_face_count = local_face_count
	                        global_active_face_count = local_active_face_count
	                        global_inactive_face_count = local_inactive_face_count
	                        global_inactive_alpha_g_count = local_inactive_alpha_g_count
	                        global_inactive_alpha_liq_count = local_inactive_alpha_liq_count
	                        global_max_species_flux = local_max_species_flux
	                        global_max_energy_flux = local_max_energy_flux
	                    end if
                    if (proc_rank == 0 .and. present(t_step) .and. present(stage)) then
                        if (t_step <= 5 .or. mod(t_step, 100) == 0) then
                            write (output_unit, '(&
	                                &"TEMP_CHEM_DIFFUSION_MODEL3_INTRINSIC_ALPHA_SUMMARY t_step=",I0,&
	                                &" stage=",I0," direction=",I0,&
	                                &" global_faces=",I0," active_faces=",I0," inactive_faces=",I0,&
	                                &" inactive_alpha_g_faces=",I0," inactive_alpha_liq_faces=",I0,&
	                                &" inactive_fraction=",ES16.8,&
	                                &" max_weighted_species_flux=",ES16.8,&
	                                &" max_weighted_energy_flux=",ES16.8)') &
	                                t_step, stage, idir, global_face_count, global_active_face_count, global_inactive_face_count, &
	                                global_inactive_alpha_g_count, global_inactive_alpha_liq_count, &
	                                real(global_inactive_face_count, wp)/max(real(global_face_count, wp), 1._wp), &
	                                global_max_species_flux, global_max_energy_flux
                            call flush(output_unit)
                        end if
                    end if
                end if

                ! Model 2: Unity Lewis Number
            else if (chem_params%transport_model == 2) then
                ! Note: Added ALL scalars and 'i'/'eqn' to private list to prevent race conditions.
                #:set unity_lewis_private = '[x,y,z,i,eqn,Ys_L, Ys_R, Ys_cell, dYk_dxi, Mass_Diffu_Flux, ' + &
                    & 'grid_spacing, MW_L, MW_R, MW_cell, Rgas_L, Rgas_R, P_L, P_R, rho_L, rho_R, ' + &
                    & 'rho_cell, T_L, T_R, Cp_L, Cp_R, hmix_L, hmix_R, dh_dxi, lambda_L, lambda_R, ' + &
                    & 'lambda_Cell, diffusivity_L, diffusivity_R, diffusivity_cell, Mass_Diffu_Energy]'
                $:GPU_PARALLEL_LOOP(collapse=3, private=unity_lewis_private, copyin='[offsets]')
                do z = isc3%beg, isc3%end
                    do y = isc2%beg, isc2%end
                        do x = isc1%beg, isc1%end
                            ! Calculate grid spacing using direction-based indexing
                            select case (idir)
                            case (1)
                                grid_spacing = x_cc(x + 1) - x_cc(x)
                            case (2)
                                grid_spacing = y_cc(y + 1) - y_cc(y)
                            case (3)
                                grid_spacing = z_cc(z + 1) - z_cc(z)
                            end select

                            ! Extract species mass fractions
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = chemxb, chemxe
                                Ys_L(i - chemxb + 1) = q_prim_qp(i)%sf(x, y, z)
                                Ys_R(i - chemxb + 1) = q_prim_qp(i)%sf(x + offsets(1), y + offsets(2), z + offsets(3))
                                Ys_cell(i - chemxb + 1) = 0.5_wp*(Ys_L(i - chemxb + 1) + Ys_R(i - chemxb + 1))
                            end do

                            ! Calculate molecular weights and mole fractions
                            call get_mixture_molecular_weight(Ys_L, MW_L)
                            call get_mixture_molecular_weight(Ys_R, MW_R)
                            MW_cell = 0.5_wp*(MW_L + MW_R)

                            ! Calculate gas constants and thermodynamic properties
                            Rgas_L = gas_constant/MW_L
                            Rgas_R = gas_constant/MW_R

                            P_L = q_prim_qp(E_idx)%sf(x, y, z)
                            P_R = q_prim_qp(E_idx)%sf(x + offsets(1), y + offsets(2), z + offsets(3))

                            rho_L = q_prim_qp(1)%sf(x, y, z)
                            rho_R = q_prim_qp(1)%sf(x + offsets(1), y + offsets(2), z + offsets(3))

                            T_L = P_L/rho_L/Rgas_L
                            T_R = P_R/rho_R/Rgas_R

                            rho_cell = 0.5_wp*(rho_L + rho_R)

                            call get_mixture_specific_heat_cp_mass(T_L, Ys_L, Cp_L)
                            call get_mixture_specific_heat_cp_mass(T_R, Ys_R, Cp_R)
                            call get_mixture_enthalpy_mass(T_L, Ys_L, hmix_L)
                            call get_mixture_enthalpy_mass(T_R, Ys_R, hmix_R)
                            dh_dxi = (hmix_R - hmix_L)/grid_spacing

                            ! Get transport properties
                            call get_mixture_thermal_conductivity_mixavg(T_L, Ys_L, lambda_L)
                            call get_mixture_thermal_conductivity_mixavg(T_R, Ys_R, lambda_R)

                            ! Calculate species properties and gradients
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = chemxb, chemxe
                                dYk_dxi(i - chemxb + 1) = (Ys_R(i - chemxb + 1) - &
                                                           Ys_L(i - chemxb + 1))/grid_spacing
                            end do

                            ! Calculate mixture-averaged diffusivities
                            diffusivity_L = lambda_L/rho_L/Cp_L
                            diffusivity_R = lambda_R/rho_R/Cp_R

                            lambda_Cell = 0.5_wp*(lambda_R + lambda_L)
                            diffusivity_cell = 0.5_wp*(diffusivity_R + diffusivity_L)

                            ! Calculate mass diffusion fluxes
                            Mass_Diffu_Energy = 0.0_wp

                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = chemxb, chemxe
                                Mass_Diffu_Flux(eqn - chemxb + 1) = rho_cell* &
                                                                    diffusivity_cell* &
                                                                    dYk_dxi(eqn - chemxb + 1)
                            end do
                            Mass_Diffu_Energy = rho_cell*diffusivity_cell*dh_dxi

                            ! Update flux arrays
                            flux_src_vf(E_idx)%sf(x, y, z) = flux_src_vf(E_idx)%sf(x, y, z) - Mass_Diffu_Energy

                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = chemxb, chemxe
                                flux_src_vf(eqn)%sf(x, y, z) = flux_src_vf(eqn)%sf(x, y, z) - Mass_Diffu_Flux(eqn - chemxb + 1)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

        #:endif

    end subroutine s_compute_chemistry_diffusion_flux

end module m_chemistry
