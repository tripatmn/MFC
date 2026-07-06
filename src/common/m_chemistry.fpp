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
    subroutine s_compute_chemistry_reaction_flux(rhs_vf, q_cons_qp, q_T_sf, q_prim_qp, bounds, t_step, stage)

        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        type(scalar_field), intent(inout) :: q_T_sf
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_qp, q_prim_qp
        type(int_bounds_info), dimension(1:3), intent(in) :: bounds
        integer, intent(in) :: t_step, stage

        integer :: x, y, z
        integer :: eqn, gas_idx, fluid_id
        real(wp) :: T, T_raw
        real(wp) :: rho, rho_g, rhoYk, raw_Y, Y_sum, omega_m, omega_m_limited
        real(wp) :: h_k, e_k, qdot_h_cell, qdot_e_cell, qdot_h_limited, qdot_e_limited
        real(wp) :: heat_weight, heat_weight_denom
        real(wp) :: heat_limit_frac, gas_internal_energy_density, heat_added, max_heat, heat_scale
        real(wp) :: alpha_g
        real(wp) :: chem_limiter_factor
        real(wp) :: v_alpha
        integer :: v_id
        real(wp) :: v_skip
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
        v_skip = 0._wp
        diag_T_max = 0._wp
        diag_p_max = 0._wp

        $:GPU_PARALLEL_LOOP(collapse=3, &
            private='[Ys, omega, omega_m_species, h_rt, eqn, gas_idx, fluid_id, T, T_raw, rho, rho_g, rhoYk, raw_Y, Y_sum, omega_m, omega_m_limited, omega_finite, h_k, e_k, qdot_h_cell, qdot_e_cell, qdot_h_limited, qdot_e_limited, heat_weight, heat_weight_denom, gas_internal_energy_density, heat_added, max_heat, heat_scale, alpha_g, chem_limiter_factor, v_alpha, v_id]', &
            reduction='[[diag_qdot_h_pos, diag_qdot_h_neg, diag_qdot_e_pos, diag_qdot_e_neg, diag_raw_qdot_h_pos, diag_raw_qdot_h_neg, diag_raw_qdot_e_pos, diag_raw_qdot_e_neg, diag_src_c12h26, diag_src_o2, diag_src_co2, diag_src_h2o, diag_raw_src_c12h26, diag_raw_src_o2, diag_raw_src_co2, diag_raw_src_h2o, diag_heat_applied, diag_raw_heat_applied, diag_partial_heat_applied, diag_partial_heat_skipped, diag_limited_cell_count, diag_scale_sum_limited, diag_T_clamp_count, v_skip], [diag_T_max, diag_p_max, diag_max_raw_qdot_h], [diag_scale_min]]', &
            reductionOp='[+, MAX, MIN]', copyin='[bounds]')
        do z = bounds(3)%beg, bounds(3)%end
            do y = bounds(2)%beg, bounds(2)%end
                do x = bounds(1)%beg, bounds(1)%end

                    if (num_fluids > 1) then
                        call s_compute_chemistry_gas_density(q_cons_qp, x, y, z, rho_g)

                        alpha_g = 0._wp
                        if (chem_gas_num_fluids <= 0) then
                            alpha_g = q_cons_qp(advxb + chem_gas_fluid_id - 1)%sf(x, y, z)
                        else
                            $:GPU_LOOP(parallelism='[seq]')
                            do gas_idx = 1, chem_gas_num_fluids
                                fluid_id = chem_gas_fluid_ids(gas_idx)
                                alpha_g = alpha_g + q_cons_qp(advxb + fluid_id - 1)%sf(x, y, z)
                            end do
                        end if

                        if (chem_species_nonneg_limiter) then
                            if ((.not. s_is_finite_wp(rho_g)) .or. rho_g <= chem_rho_gas_min .or. &
                                alpha_g < chem_alpha_gas_min) then
                                cycle
                            end if

                            if (chem_reactive_vapor_alpha_min > 0.0_wp) then
                                if (chem_gas_num_fluids <= 0) then
                                    v_id = chem_gas_fluid_id
                                else
                                    v_id = chem_gas_fluid_ids(1)
                                end if

                                if (v_id < 1 .or. v_id > num_fluids) then
                                    cycle
                                end if

                                v_alpha = q_cons_qp(advxb + v_id - 1)%sf(x, y, z)

                                if (v_alpha < chem_reactive_vapor_alpha_min) then
                                    if (heat_diag_active) then
                                        v_skip = v_skip + 1._wp
                                    end if
                                    cycle
                                end if
                            end if
                        else
                            if ((.not. s_is_finite_wp(rho_g)) .or. rho_g <= chem_rho_g_min) then
                                ! No gas-phase chemistry if designated gas density vanishes.
                                cycle
                            end if
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
                    if (chem_species_nonneg_limiter) then
                        if (T_raw > chem_T_max) cycle
                    end if
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
                                    e_k = (h_rt(eqn - chemxb + 1) - 1._wp)*gas_constant*T/molecular_weights_nonparameter(eqn - chemxb + 1)
                                #:else
                                    h_k = h_rt(eqn - chemxb + 1)*gas_constant*T/molecular_weights(eqn - chemxb + 1)
                                    e_k = (h_rt(eqn - chemxb + 1) - 1._wp)*gas_constant*T/molecular_weights(eqn - chemxb + 1)
                                #:endif
                                qdot_h_cell = qdot_h_cell - h_k*omega_m
                                qdot_e_cell = qdot_e_cell - e_k*omega_m
                            end if
                        end if

                    end do

                    chem_limiter_factor = 1.0_wp
                    if (chem_species_nonneg_limiter) then
                        $:GPU_LOOP(parallelism='[seq]')
                        do eqn = chemxb, chemxe
                            omega_m = omega_m_species(eqn - chemxb + 1)
                            if (omega_m < 0._wp) then
                                chem_limiter_factor = min(chem_limiter_factor, &
                                    q_cons_qp(eqn)%sf(x, y, z) / max(tiny(1._wp), -omega_m * s_reaction_heat_dt()))
                            end if
                        end do
                        chem_limiter_factor = max(0._wp, chem_limiter_factor)
                    end if

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

                    if (chem_species_nonneg_limiter) then
                        heat_scale = heat_scale * chem_limiter_factor
                    end if

                    qdot_h_limited = heat_scale*qdot_h_cell
                    qdot_e_limited = heat_scale*qdot_e_cell

                    $:GPU_LOOP(parallelism='[seq]')
                    do eqn = chemxb, chemxe
                        omega_m = omega_m_species(eqn - chemxb + 1)
                        omega_m_limited = heat_scale*omega_m
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
                    &" T_clamp_count=", ES16.6, " vap_alpha_skip_count=", ES16.6, " max_raw_qdot_h=", ES16.6, &
                    &" T_max=", ES16.6, " p_max=", ES16.6, &
                    &" energy_application=", A)', &
                proc_rank, t_step, stage, heat_limit_frac, diag_limited_cell_count, &
                diag_scale_min, diag_scale_mean_limited, &
                diag_raw_qdot_h_pos, diag_qdot_h_pos, diag_raw_qdot_h_neg, diag_qdot_h_neg, &
                diag_raw_qdot_e_pos, diag_qdot_e_pos, diag_raw_qdot_e_neg, diag_qdot_e_neg, &
                diag_raw_src_c12h26, diag_src_c12h26, diag_raw_src_o2, diag_src_o2, &
                diag_raw_src_co2, diag_src_co2, diag_raw_src_h2o, diag_src_h2o, &
                diag_raw_heat_applied, diag_heat_applied, diag_partial_heat_applied, &
                diag_partial_heat_skipped, diag_T_clamp_count, v_skip, diag_max_raw_qdot_h, &
                diag_T_max, diag_p_max, &
                merge("TOTAL_AND_GAS_PARTIALS", "NONE                  ", apply_heat_active)
            call flush(output_unit)
        end if

    end subroutine s_compute_chemistry_reaction_flux

    !> @brief Computes species mass diffusion fluxes at cell interfaces using mixture-averaged diffusivities.
    subroutine s_compute_chemistry_diffusion_flux(idir, q_prim_qp, flux_src_vf, irx, iry, irz)

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_qp
        type(scalar_field), dimension(sys_size), intent(inout) :: flux_src_vf
        type(int_bounds_info), intent(in) :: irx, iry, irz

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
        real(wp) :: lambda_L, lambda_R, lambda_Cell, dT_dxi, grid_spacing
        real(wp) :: Cp_L, Cp_R
        real(wp) :: diffusivity_L, diffusivity_R, diffusivity_cell
        real(wp) :: hmix_L, hmix_R, dh_dxi

        integer :: x, y, z, i, n, eqn
        integer, dimension(3) :: offsets

        isc1 = irx; isc2 = iry; isc3 = irz

        $:GPU_UPDATE(device='[isc1,isc2,isc3]')

        if (chemistry .or. dummy) then

            ! Set offsets based on direction using array indexing
            offsets = 0
            offsets(idir) = 1
            ! Model 1: Mixture-Average Transport
            if (chem_params%transport_model == 1) then
                ! Note: Added 'i' and 'eqn' to private list.
                $:GPU_PARALLEL_LOOP(collapse=3,  private='[x,y,z,i,eqn,Ys_L, Ys_R, Ys_cell, Xs_L, Xs_R, mass_diffusivities_mixavg1, mass_diffusivities_mixavg2, mass_diffusivities_mixavg_Cell, h_l, h_r, Xs_cell, h_k, dXk_dxi,Mass_Diffu_Flux, Mass_Diffu_Energy, MW_L, MW_R, MW_cell, Rgas_L, Rgas_R, T_L, T_R, P_L, P_R, rho_L, rho_R, rho_cell, rho_Vic, lambda_L, lambda_R, lambda_Cell, dT_dxi, grid_spacing]', copyin='[offsets]')
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

                            call get_mole_fractions(MW_L, Ys_L, Xs_L)
                            call get_mole_fractions(MW_R, Ys_R, Xs_R)

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
                            dT_dxi = (T_R - T_L)/grid_spacing

                            ! Get transport properties
                            call get_species_mass_diffusivities_mixavg(P_L, T_L, Ys_L, mass_diffusivities_mixavg1)
                            call get_species_mass_diffusivities_mixavg(P_R, T_R, Ys_R, mass_diffusivities_mixavg2)

                            call get_mixture_thermal_conductivity_mixavg(T_L, Ys_L, lambda_L)
                            call get_mixture_thermal_conductivity_mixavg(T_R, Ys_R, lambda_R)

                            call get_species_enthalpies_rt(T_L, h_l)
                            call get_species_enthalpies_rt(T_R, h_r)

                            ! Calculate species properties and gradients
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = chemxb, chemxe
                                #:if USING_AMD
                                    h_l(i - chemxb + 1) = h_l(i - chemxb + 1)*gas_constant*T_L/molecular_weights_nonparameter(i - chemxb + 1)
                                    h_r(i - chemxb + 1) = h_r(i - chemxb + 1)*gas_constant*T_R/molecular_weights_nonparameter(i - chemxb + 1)
                                #:else
                                    h_l(i - chemxb + 1) = h_l(i - chemxb + 1)*gas_constant*T_L/molecular_weights(i - chemxb + 1)
                                    h_r(i - chemxb + 1) = h_r(i - chemxb + 1)*gas_constant*T_R/molecular_weights(i - chemxb + 1)
                                #:endif
                                Xs_cell(i - chemxb + 1) = 0.5_wp*(Xs_L(i - chemxb + 1) + Xs_R(i - chemxb + 1))
                                h_k(i - chemxb + 1) = 0.5_wp*(h_l(i - chemxb + 1) + h_r(i - chemxb + 1))
                                dXk_dxi(i - chemxb + 1) = (Xs_R(i - chemxb + 1) - Xs_L(i - chemxb + 1))/grid_spacing
                            end do

                            ! Calculate mixture-averaged diffusivities
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = chemxb, chemxe
                                mass_diffusivities_mixavg_Cell(i - chemxb + 1) = &
                                    (mass_diffusivities_mixavg2(i - chemxb + 1) + mass_diffusivities_mixavg1(i - chemxb + 1))/2.0_wp
                            end do

                            lambda_Cell = 0.5_wp*(lambda_R + lambda_L)

                            ! Calculate mass diffusion fluxes
                            rho_Vic = 0.0_wp
                            Mass_Diffu_Energy = 0.0_wp

                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = chemxb, chemxe
                                #:if USING_AMD
                                    Mass_Diffu_Flux(eqn - chemxb + 1) = rho_cell*mass_diffusivities_mixavg_Cell(eqn - chemxb + 1)* &
                                                                        molecular_weights_nonparameter(eqn - chemxb + 1)/MW_cell*dXk_dxi(eqn - chemxb + 1)
                                #:else
                                    Mass_Diffu_Flux(eqn - chemxb + 1) = rho_cell*mass_diffusivities_mixavg_Cell(eqn - chemxb + 1)* &
                                                                        molecular_weights(eqn - chemxb + 1)/MW_cell*dXk_dxi(eqn - chemxb + 1)
                                #:endif
                                rho_Vic = rho_Vic + Mass_Diffu_Flux(eqn - chemxb + 1)
                                Mass_Diffu_Energy = Mass_Diffu_Energy + h_k(eqn - chemxb + 1)*Mass_Diffu_Flux(eqn - chemxb + 1)
                            end do

                            ! Apply corrections for mass conservation
                            $:GPU_LOOP(parallelism='[seq]')
                            do eqn = chemxb, chemxe
                                Mass_Diffu_Energy = Mass_Diffu_Energy - h_k(eqn - chemxb + 1)*Ys_cell(eqn - chemxb + 1)*rho_Vic
                                Mass_Diffu_Flux(eqn - chemxb + 1) = Mass_Diffu_Flux(eqn - chemxb + 1) - rho_Vic*Ys_cell(eqn - chemxb + 1)
                            end do

                            ! Add thermal conduction contribution
                            Mass_Diffu_Energy = lambda_Cell*dT_dxi + Mass_Diffu_Energy

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

                ! Model 2: Unity Lewis Number
            else if (chem_params%transport_model == 2) then
                ! Note: Added ALL scalars and 'i'/'eqn' to private list to prevent race conditions.
                $:GPU_PARALLEL_LOOP(collapse=3, private='[x,y,z,i,eqn,Ys_L, Ys_R, Ys_cell, dYk_dxi, Mass_Diffu_Flux, grid_spacing, MW_L, MW_R, MW_cell, Rgas_L, Rgas_R, P_L, P_R, rho_L, rho_R, rho_cell, T_L, T_R, Cp_L, Cp_R, hmix_L, hmix_R, dh_dxi, lambda_L, lambda_R, lambda_Cell, diffusivity_L, diffusivity_R, diffusivity_cell, Mass_Diffu_Energy]', copyin='[offsets]')
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
