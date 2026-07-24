!>
!! @file
!! @brief Contains module m_phase_change

#:include 'case.fpp'
#:include 'macros.fpp'

!> @brief Phase transition relaxation solvers for liquid-vapor flows with cavitation and boiling
module m_phase_change

#ifndef MFC_POST_PROCESS

    use iso_fortran_env, only: output_unit

    use m_derived_types        !< Definitions of the derived types

    use m_global_parameters    !< Definitions of the global parameters

    use m_mpi_proxy            !< Message passing interface (MPI) module proxy

#ifdef MFC_SIMULATION
    use m_mpi_common, only: s_mpi_allreduce_integer_sum, s_mpi_allreduce_sum, &
                            s_mpi_allreduce_max, s_mpi_allreduce_min, s_mpi_abort
#endif

    use m_variables_conversion !< State variables type conversion procedures

    use m_helper_basic         !< Functions to compare floating point numbers

    use m_constants, only: pi

    implicit none

    private; 
    public :: s_initialize_phasechange_module, &
              s_relaxation_solver, &
              s_infinite_relaxation_k, &
              s_finalize_relaxation_solver_module

    !> @name Parameters for the first order transition phase change
    !> @{
    integer, parameter :: max_iter = 1e8_wp        !< max # of iterations
    real(wp), parameter :: pCr = 1.817e6_wp   !< Critical pressure
    real(wp), parameter :: TCr = 658.2_wp  !< Critical temperature
    real(wp), parameter :: mixM = 1.0e-5_wp !< threshold for 'mixture cell'. If Y < mixM, phase change does not happen
    integer, parameter :: lp = 1    !< index for the liquid phase of the reacting fluid
    integer, parameter :: vp = 2    !< index for the vapor phase of the reacting fluid
    !> @}

    !> @name Gibbs free energy phase change parameters
    !> @{
    real(wp) :: A, B, C, D
    !> @}
    logical :: phase_change_fuel_mass_coupling_fix_enabled = .false.
    logical :: phase_change_evap_only_enabled = .false.
    real(wp), parameter :: phase_change_fuel_mass_tolerance = 1.e-12_wp
    integer :: phase_change_signed_condensation_occurrences = 0
    integer :: phase_change_signed_condensation_insufficient = 0
    integer :: phase_change_signed_condensation_slight_negative = 0
    logical :: phase_change_budget_debug_enabled = .false.
    integer :: phase_change_budget_target_step_min = 0
    integer :: phase_change_budget_target_step_max = huge(0)
    integer :: phase_change_budget_cum_evap_cells = 0
    integer :: phase_change_budget_cum_cond_cells = 0
    integer :: phase_change_budget_cum_insufficient_cells = 0
    real(wp) :: phase_change_budget_cum_evap_mass = 0._wp
    real(wp) :: phase_change_budget_cum_cond_mass = 0._wp
    real(wp) :: phase_change_budget_cum_net_mass = 0._wp
    real(wp) :: phase_change_budget_cum_insufficient_mass = 0._wp
    real(wp) :: phase_change_budget_max_evap_delta = 0._wp
    real(wp) :: phase_change_budget_max_cond_delta = 0._wp
    integer :: phase_change_evap_only_rejections = 0
    real(wp) :: phase_change_evap_only_suppressed_mass = 0._wp
    real(wp) :: phase_change_evap_only_max_delta = 0._wp

#ifdef MFC_SIMULATION
    integer :: phase_change_signed_local_condensation = 0
    integer :: phase_change_signed_local_insufficient = 0
    integer :: phase_change_signed_local_slight_negative = 0
    integer :: phase_change_signed_failure_claimed = 0
    integer, dimension(3) :: phase_change_signed_failure_ijk = 0
    real(wp), dimension(12) :: phase_change_signed_failure_data = 0._wp
#endif

    $:GPU_DECLARE(create='[A,B,C,D,phase_change_fuel_mass_coupling_fix_enabled,phase_change_evap_only_enabled,phase_change_budget_debug_enabled]')
#ifdef MFC_SIMULATION
    $:GPU_DECLARE(create='[phase_change_signed_local_condensation,phase_change_signed_local_insufficient,phase_change_signed_local_slight_negative,phase_change_signed_failure_claimed,phase_change_signed_failure_ijk,phase_change_signed_failure_data]')
#endif

contains

#ifdef MFC_SIMULATION
    subroutine s_phase_relax_read_target_integer(name, value)
        character(len=*), intent(in) :: name
        integer, intent(inout) :: value

        character(len=32) :: env_value
        integer :: env_status, read_status, parsed_value

        env_value = ""
        call get_environment_variable(name, env_value, status=env_status)
        if (env_status /= 0 .or. len_trim(env_value) == 0) return
        read (env_value, *, iostat=read_status) parsed_value
        if (read_status == 0) value = parsed_value
    end subroutine s_phase_relax_read_target_integer
#endif

    !> @brief GPU-safe typed finite check for phase-change device routines.
    logical function s_is_finite_wp(x)
        $:GPU_ROUTINE(function_name='s_is_finite_wp',parallelism='[seq]', &
            & cray_inline=True)

        real(wp), intent(in) :: x

        s_is_finite_wp = (x == x) .and. (abs(x) <= huge(x))

    end function s_is_finite_wp

    !> This subroutine should dispatch to the correct relaxation solver based
        !!      some parameter. It replaces the procedure pointer, which CCE
        !!      is breaking on.
    impure subroutine s_relaxation_solver(q_cons_vf)
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        ! This is empty because in current master the procedure pointer
        ! was never assigned
        @:ASSERT(.false., "s_relaxation_solver called but it currently does nothing")
    end subroutine s_relaxation_solver

    !>  The purpose of this subroutine is to initialize the phase change module
        !!      by setting the parameters needed for phase change and
        !!      selecting the phase change module that will be used
        !!      (pT- or pTg-equilibrium)
    impure subroutine s_initialize_phasechange_module
        character(len=16) :: env_value
        integer :: env_status

        ! variables used in the calculation of the saturation curves for fluids 1 and 2
        A = (gs_min(lp)*cvs(lp) - gs_min(vp)*cvs(vp) &
             + qvps(vp) - qvps(lp))/((gs_min(vp) - 1.0_wp)*cvs(vp))

        B = (qvs(lp) - qvs(vp))/((gs_min(vp) - 1.0_wp)*cvs(vp))

        C = (gs_min(vp)*cvs(vp) - gs_min(lp)*cvs(lp)) &
            /((gs_min(vp) - 1.0_wp)*cvs(vp))

        D = ((gs_min(lp) - 1.0_wp)*cvs(lp)) &
            /((gs_min(vp) - 1.0_wp)*cvs(vp))

#ifdef MFC_SIMULATION
        env_value = ""
        call get_environment_variable("TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX", env_value, status=env_status)
        phase_change_fuel_mass_coupling_fix_enabled = env_status == 0 .and. trim(env_value) == "1"
        $:GPU_UPDATE(device='[phase_change_fuel_mass_coupling_fix_enabled]')
        if (phase_change_fuel_mass_coupling_fix_enabled .and. proc_rank == 0) then
            print '(A)', "TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_FIX enabled"
            call flush(output_unit)
        end if

        env_value = ""
        call get_environment_variable("TEMP_PHASE_CHANGE_EVAP_ONLY", env_value, status=env_status)
        phase_change_evap_only_enabled = env_status == 0 .and. trim(env_value) == "1"
        phase_change_evap_only_rejections = 0
        phase_change_evap_only_suppressed_mass = 0._wp
        phase_change_evap_only_max_delta = 0._wp
        $:GPU_UPDATE(device='[phase_change_evap_only_enabled]')
        if (phase_change_evap_only_enabled .and. proc_rank == 0) then
            print '(A)', "TEMP_PHASE_CHANGE_EVAP_ONLY enabled validation_only=T"
            call flush(output_unit)
        end if

        phase_change_signed_condensation_occurrences = 0
        phase_change_signed_condensation_insufficient = 0
        phase_change_signed_condensation_slight_negative = 0

        env_value = ""
        call get_environment_variable("TEMP_PHASE_CHANGE_BUDGET_DEBUG", env_value, status=env_status)
        phase_change_budget_debug_enabled = env_status == 0 .and. trim(env_value) == "1"
        call s_phase_relax_read_target_integer("TEMP_PHASE_CHANGE_BUDGET_STEP_MIN", &
                                               phase_change_budget_target_step_min)
        call s_phase_relax_read_target_integer("TEMP_PHASE_CHANGE_BUDGET_STEP_MAX", &
                                               phase_change_budget_target_step_max)
        phase_change_budget_cum_evap_cells = 0
        phase_change_budget_cum_cond_cells = 0
        phase_change_budget_cum_insufficient_cells = 0
        phase_change_budget_cum_evap_mass = 0._wp
        phase_change_budget_cum_cond_mass = 0._wp
        phase_change_budget_cum_net_mass = 0._wp
        phase_change_budget_cum_insufficient_mass = 0._wp
        phase_change_budget_max_evap_delta = 0._wp
        phase_change_budget_max_cond_delta = 0._wp
        $:GPU_UPDATE(device='[phase_change_budget_debug_enabled]')
        if (phase_change_budget_debug_enabled .and. proc_rank == 0) then
            write (output_unit, '("TEMP_PHASE_CHANGE_BUDGET_DEBUG enabled step_min=",I0," step_max=",I0)') &
                phase_change_budget_target_step_min, phase_change_budget_target_step_max
            call flush(output_unit)
        end if

#endif

    end subroutine s_initialize_phasechange_module

    !>  This subroutine is created to activate either the pT- (N fluids) or the
        !!      pTg-equilibrium (2 fluids for g-equilibrium)
        !!      model, also considering mass depletion, depending on the incoming
        !!      state conditions.
        !!  @param q_cons_vf Cell-average conservative variables
#ifdef MFC_SIMULATION
    subroutine s_infinite_relaxation_k(q_cons_vf, m_dot_evap, relax_dt, t_step_diag, stage_diag)
#else
    subroutine s_infinite_relaxation_k(q_cons_vf)
#endif

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
#ifdef MFC_SIMULATION
        type(scalar_field), intent(inout) :: m_dot_evap
        real(wp), intent(in) :: relax_dt
        integer, optional, intent(in) :: t_step_diag, stage_diag
#endif
        real(wp) :: pS, pSOV, pSSL !< equilibrium pressure for mixture, overheated vapor, and subcooled liquid
        real(wp) :: TS, TSOV, TSSL, TSatOV, TSatSL !< equilibrium temperature for mixture, overheated vapor, and subcooled liquid. Saturation Temperatures at overheated vapor and subcooled liquid
        real(wp) :: no_transfer_pS, no_transfer_TS
        real(wp) :: rhoe, dynE, rhos !< total internal energy, kinetic energy, and total entropy
        real(wp) :: rho, rM, m1, m2, m2_after, delta_m_vapor, MCT !< total density, total reacting mass, individual reacting masses
        real(wp) :: TvF !< total volume fraction
        real(wp) :: fuel_before_coupling
        real(wp) :: fuel_candidate, failure_rho_g, failure_sum_rhoY
        real(wp) :: failure_x, failure_y, failure_z
        logical :: pt_state_ok, ptg_state_ok

        ! $:GPU_DECLARE(create='[pS,pSOV,pSSL,TS,TSOV,TSSL,TSatOV,TSatSL]')
        ! $:GPU_DECLARE(create='[rhoe,dynE,rhos,rho,rM,m1,m2,MCT,TvF]')
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: p_infOV, p_infpT, p_infSL, sk, hk, gk, ek, rhok
        #:else
            real(wp), dimension(num_fluids) :: p_infOV, p_infpT, p_infSL, sk, hk, gk, ek, rhok
        #:endif
        ! $:GPU_DECLARE(create='[p_infOV,p_infpT,p_infSL,sk,hk,gk,ek,rhok]')

        !< Generic loop iterators
        integer :: i, j, k, l
#ifdef MFC_SIMULATION
        integer :: report_step, report_stage
        integer :: signed_failure_old_claim
        integer :: signed_gas_idx, signed_fluid_id, signed_species_eqn
        integer :: local_budget_evap_cells, global_budget_evap_cells
        integer :: local_budget_cond_cells, global_budget_cond_cells
        integer :: local_budget_insufficient_cells, global_budget_insufficient_cells
        integer :: local_budget_interface_cond_cells, global_budget_interface_cond_cells
        integer :: local_budget_liquid_cond_cells, global_budget_liquid_cond_cells
        integer :: local_budget_gas_cond_cells, global_budget_gas_cond_cells
        integer :: local_evap_only_rejections, global_evap_only_rejections
        real(wp) :: local_budget_evap_mass, global_budget_evap_mass
        real(wp) :: local_budget_cond_mass, global_budget_cond_mass
        real(wp) :: local_budget_net_mass, global_budget_net_mass
        real(wp) :: local_budget_insufficient_mass, global_budget_insufficient_mass
        real(wp) :: local_budget_max_evap_delta, global_budget_max_evap_delta
        real(wp) :: local_budget_max_cond_delta, global_budget_max_cond_delta
        real(wp) :: local_budget_max_insufficient_ratio, global_budget_max_insufficient_ratio
        real(wp) :: local_budget_min_alpha_liq, global_budget_min_alpha_liq
        real(wp) :: local_budget_max_alpha_liq, global_budget_max_alpha_liq
        real(wp) :: local_budget_min_alpha_vap, global_budget_min_alpha_vap
        real(wp) :: local_budget_max_alpha_vap, global_budget_max_alpha_vap
        real(wp) :: local_budget_min_alpha_air, global_budget_min_alpha_air
        real(wp) :: local_budget_max_alpha_air, global_budget_max_alpha_air
        real(wp) :: local_budget_min_pressure, global_budget_min_pressure
        real(wp) :: local_budget_max_pressure, global_budget_max_pressure
        real(wp) :: local_budget_min_temperature, global_budget_min_temperature
        real(wp) :: local_budget_max_temperature, global_budget_max_temperature
        real(wp) :: local_budget_min_vapor_arho, global_budget_min_vapor_arho
        real(wp) :: local_budget_max_vapor_arho, global_budget_max_vapor_arho
        real(wp) :: local_budget_min_fuel, global_budget_min_fuel
        real(wp) :: local_budget_max_fuel, global_budget_max_fuel
        real(wp) :: local_budget_min_request_ratio, global_budget_min_request_ratio
        real(wp) :: local_budget_max_request_ratio, global_budget_max_request_ratio
        real(wp) :: local_evap_only_suppressed_mass, global_evap_only_suppressed_mass
        real(wp) :: local_evap_only_max_delta, global_evap_only_max_delta
        real(wp) :: local_evap_only_min_alpha_liq, global_evap_only_min_alpha_liq
        real(wp) :: local_evap_only_max_alpha_liq, global_evap_only_max_alpha_liq
        real(wp) :: local_evap_only_min_pressure, global_evap_only_min_pressure
        real(wp) :: local_evap_only_max_pressure, global_evap_only_max_pressure
        real(wp) :: local_evap_only_min_temperature, global_evap_only_min_temperature
        real(wp) :: local_evap_only_max_temperature, global_evap_only_max_temperature
        real(wp) :: budget_cell_volume, budget_delta_m_vapor, budget_abs_delta
        real(wp) :: budget_cond_requested, budget_fuel_available, budget_ratio
        real(wp) :: budget_alpha_liq, budget_alpha_vap, budget_alpha_air
        real(wp) :: budget_evap_ratio
        real(wp) :: evap_only_proposed_delta, evap_only_suppressed_delta
#endif

#ifdef MFC_SIMULATION
        report_step = -1
        report_stage = -1
        if (present(t_step_diag)) report_step = t_step_diag
        if (present(stage_diag)) report_stage = stage_diag

        if (phase_change_fuel_mass_coupling_fix_enabled) then
            phase_change_signed_local_condensation = 0
            phase_change_signed_local_insufficient = 0
            phase_change_signed_local_slight_negative = 0
            phase_change_signed_failure_claimed = 0
            phase_change_signed_failure_ijk = 0
            phase_change_signed_failure_data = 0._wp
            $:GPU_UPDATE(device='[phase_change_signed_local_condensation,phase_change_signed_local_insufficient,phase_change_signed_local_slight_negative,phase_change_signed_failure_claimed,phase_change_signed_failure_ijk,phase_change_signed_failure_data]')
        end if

        local_budget_evap_cells = 0
        local_budget_cond_cells = 0
        local_budget_insufficient_cells = 0
        local_budget_interface_cond_cells = 0
        local_budget_liquid_cond_cells = 0
        local_budget_gas_cond_cells = 0
        local_evap_only_rejections = 0
        local_budget_evap_mass = 0._wp
        local_budget_cond_mass = 0._wp
        local_budget_net_mass = 0._wp
        local_budget_insufficient_mass = 0._wp
        local_budget_max_evap_delta = 0._wp
        local_budget_max_cond_delta = 0._wp
        local_budget_max_insufficient_ratio = 0._wp
        local_budget_min_alpha_liq = huge(1._wp)
        local_budget_max_alpha_liq = -huge(1._wp)
        local_budget_min_alpha_vap = huge(1._wp)
        local_budget_max_alpha_vap = -huge(1._wp)
        local_budget_min_alpha_air = huge(1._wp)
        local_budget_max_alpha_air = -huge(1._wp)
        local_budget_min_pressure = huge(1._wp)
        local_budget_max_pressure = -huge(1._wp)
        local_budget_min_temperature = huge(1._wp)
        local_budget_max_temperature = -huge(1._wp)
        local_budget_min_vapor_arho = huge(1._wp)
        local_budget_max_vapor_arho = -huge(1._wp)
        local_budget_min_fuel = huge(1._wp)
        local_budget_max_fuel = -huge(1._wp)
        local_budget_min_request_ratio = huge(1._wp)
        local_budget_max_request_ratio = -huge(1._wp)
        local_evap_only_suppressed_mass = 0._wp
        local_evap_only_max_delta = 0._wp
        local_evap_only_min_alpha_liq = huge(1._wp)
        local_evap_only_max_alpha_liq = -huge(1._wp)
        local_evap_only_min_pressure = huge(1._wp)
        local_evap_only_max_pressure = -huge(1._wp)
        local_evap_only_min_temperature = huge(1._wp)
        local_evap_only_max_temperature = -huge(1._wp)

#endif

        ! starting equilibrium solver
            $:GPU_PARALLEL_LOOP(collapse=3, private='[i,j,k,l,p_infOV, p_infpT, p_infSL, sk, hk, gk, ek, rhok,pS, pSOV, pSSL, TS, TSOV, TSatOV, TSatSL, TSSL, no_transfer_pS, no_transfer_TS, rhoe, dynE, rhos, rho, rM, m1, m2, m2_after, delta_m_vapor, MCT, TvF, pt_state_ok, ptg_state_ok, fuel_before_coupling,fuel_candidate,failure_rho_g,failure_sum_rhoY,signed_failure_old_claim,signed_gas_idx,signed_fluid_id,signed_species_eqn,budget_cell_volume,budget_delta_m_vapor,budget_abs_delta,budget_cond_requested,budget_fuel_available,budget_ratio,budget_alpha_liq,budget_alpha_vap,budget_alpha_air,evap_only_proposed_delta,evap_only_suppressed_delta]', &
            reduction='[[local_budget_evap_cells,local_budget_cond_cells,local_budget_insufficient_cells,local_budget_interface_cond_cells,local_budget_liquid_cond_cells,local_budget_gas_cond_cells,local_evap_only_rejections,local_budget_evap_mass,local_budget_cond_mass,local_budget_net_mass,local_budget_insufficient_mass,local_evap_only_suppressed_mass],[local_budget_max_evap_delta,local_budget_max_cond_delta,local_budget_max_insufficient_ratio,local_budget_max_alpha_liq,local_budget_max_alpha_vap,local_budget_max_alpha_air,local_budget_max_pressure,local_budget_max_temperature,local_budget_max_vapor_arho,local_budget_max_fuel,local_budget_max_request_ratio,local_evap_only_max_delta,local_evap_only_max_alpha_liq,local_evap_only_max_pressure,local_evap_only_max_temperature],[local_budget_min_alpha_liq,local_budget_min_alpha_vap,local_budget_min_alpha_air,local_budget_min_pressure,local_budget_min_temperature,local_budget_min_vapor_arho,local_budget_min_fuel,local_budget_min_request_ratio,local_evap_only_min_alpha_liq,local_evap_only_min_pressure,local_evap_only_min_temperature]]', &
            reductionOp='[+,MAX,MIN]')
        do j = 0, m
            do k = 0, n
                do l = 0, p

                    rho = 0.0_wp; TvF = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids

                        ! Mixture density
                        rho = rho + q_cons_vf(i + contxb - 1)%sf(j, k, l)

                        ! Total Volume Fraction
                        TvF = TvF + q_cons_vf(i + advxb - 1)%sf(j, k, l)

                    end do

                    ! calculating the total reacting mass for the phase change process. By hypothesis, this should not change
                    ! throughout the phase-change process.
                    rM = q_cons_vf(lp + contxb - 1)%sf(j, k, l) + q_cons_vf(vp + contxb - 1)%sf(j, k, l)

                    ! correcting negative (reacting) mass fraction values in case they happen
                    call s_correct_partial_densities(MCT, q_cons_vf, rM, j, k, l)

                    ! fixing m1 and m2 AFTER correcting the partial densities. Note that these values must be stored for the phase
                    ! change process that will happen a posteriori
                    m1 = q_cons_vf(lp + contxb - 1)%sf(j, k, l)

                    m2 = q_cons_vf(vp + contxb - 1)%sf(j, k, l)
                    ! kinetic energy as an auxiliary variable to the calculation of the total internal energy
                    dynE = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = momxb, momxe

                        dynE = dynE + 5.0e-1_wp*q_cons_vf(i)%sf(j, k, l)**2/rho

                    end do

                    ! calculating the total energy that MUST be preserved throughout the pT- and pTg-relaxation procedures
                    ! at each of the cells. The internal energy is calculated as the total energy minus the kinetic
                    ! energy to preserved its value at sharp interfaces
                    rhoe = q_cons_vf(E_idx)%sf(j, k, l) - dynE

                    ! Calling pT-equilibrium for either finishing phase-change module, or as an IC for the pTg-equilibrium
                    ! for this case, MFL cannot be either 0 or 1, so I chose it to be 2
                    call s_infinite_pt_relaxation_k(j, k, l, 2, pS, p_infpT, q_cons_vf, rhoe, TS)
                    no_transfer_pS = pS
                    no_transfer_TS = TS
                    pt_state_ok = s_is_finite_wp(pS) .and. s_is_finite_wp(TS) .and. &
                                  pS > 0._wp .and. TS > 0._wp

                    if (.not. pt_state_ok) then
                        q_cons_vf(lp + contxb - 1)%sf(j, k, l) = m1
                        q_cons_vf(vp + contxb - 1)%sf(j, k, l) = m2
#ifdef MFC_SIMULATION
                        m_dot_evap%sf(j, k, l) = 0._wp
#endif
                        cycle
                    end if

                    ! check if pTg-equilibrium is required
                    ! NOTE that NOTHING else needs to be updated OTHER than the individual partial densities
                    ! given the outputs from the pT- and pTg-equilibrium solvers are just p and one of the partial masses
                    ! (pTg- case)
                    if ((relax_model == 6) .and. ((q_cons_vf(lp + contxb - 1)%sf(j, k, l) > mixM*rM) &
                                                  .and. (q_cons_vf(vp + contxb - 1)%sf(j, k, l) > mixM*rM)) &
                        .and. (pS < pCr) .and. (TS < TCr)) then

                        ! Checking if phase change is needed, by checking whether the final solution is either subcoooled
                        ! liquid or overheated vapor.

                        ! overheated vapor case
                        ! depleting the mass of liquid
                        q_cons_vf(lp + contxb - 1)%sf(j, k, l) = mixM*rM

                        ! transferring the total mass to vapor
                        q_cons_vf(vp + contxb - 1)%sf(j, k, l) = (1.0_wp - mixM)*rM

                        ! calling pT-equilibrium for overheated vapor, which is MFL = 0
                        call s_infinite_pt_relaxation_k(j, k, l, 0, pSOV, p_infOV, q_cons_vf, rhoe, TSOV)
                        ! calculating Saturation temperature
                        call s_TSat(pSOV, TSatOV, TSOV)

                        ! subcooled liquid case
                        ! transferring the total mass to liquid
                        q_cons_vf(lp + contxb - 1)%sf(j, k, l) = (1.0_wp - mixM)*rM

                        ! depleting the mass of vapor
                        q_cons_vf(vp + contxb - 1)%sf(j, k, l) = mixM*rM

                        ! calling pT-equilibrium for subcooled liquid, which is MFL = 1
                        call s_infinite_pt_relaxation_k(j, k, l, 1, pSSL, p_infSL, q_cons_vf, rhoe, TSSL)
                        ! calculating Saturation temperature
                        call s_TSat(pSSL, TSatSL, TSSL)

                        ! checking the conditions for overheated vapor and subcooled liquide
                        if (TSOV > TSatOV) then

                            ! Assigning pressure
                            pS = pSOV

                            ! Assigning Temperature
                            TS = TSOV

                            ! correcting the liquid partial density
                            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = mixM*rM

                            ! correcting the vapor partial density
                            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = (1.0_wp - mixM)*rM

                        elseif (TSSL < TSatSL) then

                            ! Assigning pressure
                            pS = pSSL

                            ! Assigning Temperature
                            TS = TSSL

                            ! correcting the liquid partial density
                            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = (1.0_wp - mixM)*rM

                            ! correcting the vapor partial density
                            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = mixM*rM

                        else

                            ! returning partial pressures to what they were from the homogeneous solver
                            ! liquid
                            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = m1

                            ! vapor
                            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = m2

                            ! calling the pTg-equilibrium solver
                            if (s_is_finite_wp(pS) .and. s_is_finite_wp(TS) .and. &
                                pS > 0._wp .and. TS > 0._wp) then
                                call s_infinite_ptg_relaxation_k(j, k, l, pS, p_infpT, rhoe, q_cons_vf, TS)
                            end if

                            ptg_state_ok = s_is_finite_wp(pS) .and. s_is_finite_wp(TS) .and. &
                                           s_is_finite_wp(q_cons_vf(lp + contxb - 1)%sf(j, k, l)) .and. &
                                           s_is_finite_wp(q_cons_vf(vp + contxb - 1)%sf(j, k, l)) .and. &
                                           pS > 0._wp .and. TS > 0._wp

                            if (.not. ptg_state_ok) then
                                q_cons_vf(lp + contxb - 1)%sf(j, k, l) = m1
                                q_cons_vf(vp + contxb - 1)%sf(j, k, l) = m2
#ifdef MFC_SIMULATION
                                m_dot_evap%sf(j, k, l) = 0._wp
#endif
                                cycle
                            end if

                        end if

                    end if

#ifdef MFC_SIMULATION
                    if (phase_change_evap_only_enabled) then
                        evap_only_proposed_delta = q_cons_vf(vp + contxb - 1)%sf(j, k, l) - m2
                        if (evap_only_proposed_delta < 0._wp) then
                            evap_only_suppressed_delta = -evap_only_proposed_delta
                            budget_cell_volume = dx(j)
                            if (n > 0) budget_cell_volume = budget_cell_volume*dy(k)
                            if (p > 0) budget_cell_volume = budget_cell_volume*dz(l)
                            if (cyl_coord .and. n > 0 .and. p == 0) &
                                budget_cell_volume = budget_cell_volume*2._wp*pi*y_cc(k)
                            local_evap_only_rejections = local_evap_only_rejections + 1
                            local_evap_only_suppressed_mass = local_evap_only_suppressed_mass + &
                                                              evap_only_suppressed_delta*budget_cell_volume
                            local_evap_only_max_delta = max(local_evap_only_max_delta, &
                                                            evap_only_suppressed_delta)
                            local_evap_only_min_alpha_liq = min(local_evap_only_min_alpha_liq, &
                                                                q_cons_vf(advxb)%sf(j, k, l))
                            local_evap_only_max_alpha_liq = max(local_evap_only_max_alpha_liq, &
                                                                q_cons_vf(advxb)%sf(j, k, l))
                            local_evap_only_min_pressure = min(local_evap_only_min_pressure, pS)
                            local_evap_only_max_pressure = max(local_evap_only_max_pressure, pS)
                            local_evap_only_min_temperature = min(local_evap_only_min_temperature, TS)
                            local_evap_only_max_temperature = max(local_evap_only_max_temperature, TS)

                            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = m1
                            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = m2
                            pS = no_transfer_pS
                            TS = no_transfer_TS
                        end if
                    end if
#endif

                    ! Calculations AFTER equilibrium

                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        ! entropy
                        sk(i) = cvs(i)*log((TS**gs_min(i)) &
                                           /((pS + ps_inf(i))**(gs_min(i) - 1.0_wp))) + qvps(i)

                        ! enthalpy
                        hk(i) = gs_min(i)*cvs(i)*TS &
                                + qvs(i)

                        ! Gibbs-free energy
                        gk(i) = hk(i) - TS*sk(i)

                        ! densities
                        rhok(i) = (pS + ps_inf(i)) &
                                  /((gs_min(i) - 1)*cvs(i)*TS)

                        ! internal energy
                        ek(i) = (pS + gs_min(i) &
                                 *ps_inf(i))/(pS + ps_inf(i)) &
                                *cvs(i)*TS + qvs(i)
                    end do

                    ! calculating volume fractions, internal energies, and total entropy
                    rhos = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids

                        ! volume fractions
                        q_cons_vf(i + advxb - 1)%sf(j, k, l) = q_cons_vf(i + contxb - 1)%sf(j, k, l)/rhok(i)

                        ! alpha*rho*e
                        if (model_eqns == 3) then
                            q_cons_vf(i + intxb - 1)%sf(j, k, l) = q_cons_vf(i + contxb - 1)%sf(j, k, l)*ek(i)
                        end if

                        ! Total entropy
                        rhos = rhos + q_cons_vf(i + contxb - 1)%sf(j, k, l)*sk(i)

                    end do

#ifdef MFC_SIMULATION
                    m2_after = q_cons_vf(vp + contxb - 1)%sf(j, k, l)
                    delta_m_vapor = m2_after - m2
                    m_dot_evap%sf(j, k, l) = delta_m_vapor/relax_dt
                    fuel_before_coupling = 0._wp
                    if (chemistry .and. fuel_species_id >= 1 .and. &
                        fuel_species_id <= (chemxe - chemxb + 1)) then
                        fuel_before_coupling = q_cons_vf(chemxb + fuel_species_id - 1)%sf(j, k, l)
                    end if
                    if (phase_change_budget_debug_enabled .and. &
                        report_step >= phase_change_budget_target_step_min .and. &
                        report_step <= phase_change_budget_target_step_max) then
                        budget_cell_volume = dx(j)
                        if (n > 0) budget_cell_volume = budget_cell_volume*dy(k)
                        if (p > 0) budget_cell_volume = budget_cell_volume*dz(l)
                        if (cyl_coord .and. n > 0 .and. p == 0) &
                            budget_cell_volume = budget_cell_volume*2._wp*pi*y_cc(k)
                        budget_delta_m_vapor = delta_m_vapor
                        budget_abs_delta = abs(budget_delta_m_vapor)
                        local_budget_net_mass = local_budget_net_mass + &
                                                budget_delta_m_vapor*budget_cell_volume
                        if (budget_delta_m_vapor > 0._wp) then
                            local_budget_evap_cells = local_budget_evap_cells + 1
                            local_budget_evap_mass = local_budget_evap_mass + &
                                                     budget_delta_m_vapor*budget_cell_volume
                            local_budget_max_evap_delta = max(local_budget_max_evap_delta, &
                                                               budget_delta_m_vapor)
                        elseif (budget_delta_m_vapor < 0._wp) then
                            budget_alpha_liq = q_cons_vf(advxb)%sf(j, k, l)
                            budget_alpha_vap = q_cons_vf(advxb + 1)%sf(j, k, l)
                            budget_alpha_air = 0._wp
                            if (num_fluids >= 3) budget_alpha_air = q_cons_vf(advxb + 2)%sf(j, k, l)
                            budget_cond_requested = -budget_delta_m_vapor
                            budget_fuel_available = fuel_before_coupling
                            budget_ratio = budget_cond_requested/max(budget_fuel_available, &
                                                                      phase_change_fuel_mass_tolerance)
                            local_budget_cond_cells = local_budget_cond_cells + 1
                            local_budget_cond_mass = local_budget_cond_mass + &
                                                     budget_cond_requested*budget_cell_volume
                            local_budget_max_cond_delta = max(local_budget_max_cond_delta, &
                                                               budget_cond_requested)
                            local_budget_min_alpha_liq = min(local_budget_min_alpha_liq, budget_alpha_liq)
                            local_budget_max_alpha_liq = max(local_budget_max_alpha_liq, budget_alpha_liq)
                            local_budget_min_alpha_vap = min(local_budget_min_alpha_vap, budget_alpha_vap)
                            local_budget_max_alpha_vap = max(local_budget_max_alpha_vap, budget_alpha_vap)
                            local_budget_min_alpha_air = min(local_budget_min_alpha_air, budget_alpha_air)
                            local_budget_max_alpha_air = max(local_budget_max_alpha_air, budget_alpha_air)
                            local_budget_min_pressure = min(local_budget_min_pressure, pS)
                            local_budget_max_pressure = max(local_budget_max_pressure, pS)
                            local_budget_min_temperature = min(local_budget_min_temperature, TS)
                            local_budget_max_temperature = max(local_budget_max_temperature, TS)
                            local_budget_min_vapor_arho = min(local_budget_min_vapor_arho, m2_after)
                            local_budget_max_vapor_arho = max(local_budget_max_vapor_arho, m2_after)
                            local_budget_min_fuel = min(local_budget_min_fuel, budget_fuel_available)
                            local_budget_max_fuel = max(local_budget_max_fuel, budget_fuel_available)
                            local_budget_min_request_ratio = min(local_budget_min_request_ratio, budget_ratio)
                            local_budget_max_request_ratio = max(local_budget_max_request_ratio, budget_ratio)
                            if (budget_cond_requested > budget_fuel_available + &
                                phase_change_fuel_mass_tolerance) then
                                local_budget_insufficient_cells = local_budget_insufficient_cells + 1
                                local_budget_insufficient_mass = local_budget_insufficient_mass + &
                                    (budget_cond_requested - budget_fuel_available)*budget_cell_volume
                                local_budget_max_insufficient_ratio = max(local_budget_max_insufficient_ratio, &
                                                                           budget_ratio)
                            end if
                            if (budget_alpha_liq > 0.9_wp) then
                                local_budget_liquid_cond_cells = local_budget_liquid_cond_cells + 1
                            elseif (budget_alpha_liq < 0.1_wp .and. budget_alpha_vap + budget_alpha_air > 0.9_wp) then
                                local_budget_gas_cond_cells = local_budget_gas_cond_cells + 1
                            else
                                local_budget_interface_cond_cells = local_budget_interface_cond_cells + 1
                            end if
                        end if
                    end if
                    if (chemistry .and. evap_species_source .and. phase_change_fuel_mass_coupling_fix_enabled .and. &
                        fuel_species_id >= 1 .and. fuel_species_id <= (chemxe - chemxb + 1)) then
                        fuel_candidate = fuel_before_coupling + delta_m_vapor
                        if (delta_m_vapor < 0._wp) then
                            $:GPU_ATOMIC(atomic='update')
                            phase_change_signed_local_condensation = phase_change_signed_local_condensation + 1
                        end if
                        if (delta_m_vapor < 0._wp .and. fuel_candidate < -phase_change_fuel_mass_tolerance) then
                            $:GPU_ATOMIC(atomic='update')
                            phase_change_signed_local_insufficient = phase_change_signed_local_insufficient + 1
#ifdef MFC_OpenACC
                            !$acc atomic capture
#elif defined(MFC_OpenMP)
                            !$omp atomic capture
#endif
                            signed_failure_old_claim = phase_change_signed_failure_claimed
                            phase_change_signed_failure_claimed = 1
#ifdef MFC_OpenACC
                            !$acc end atomic
#elif defined(MFC_OpenMP)
                            !$omp end atomic
#endif
                            if (signed_failure_old_claim == 0) then
                                failure_rho_g = 0._wp
                                if (num_fluids == 1) then
                                    failure_rho_g = q_cons_vf(contxe)%sf(j, k, l)
                                elseif (chem_gas_num_fluids <= 0) then
                                    signed_fluid_id = chem_gas_fluid_id
                                    if (signed_fluid_id >= 1 .and. signed_fluid_id <= num_fluids) &
                                        failure_rho_g = q_cons_vf(contxb + signed_fluid_id - 1)%sf(j, k, l)
                                else
                                    $:GPU_LOOP(parallelism='[seq]')
                                    do signed_gas_idx = 1, chem_gas_num_fluids
                                        signed_fluid_id = chem_gas_fluid_ids(signed_gas_idx)
                                        if (signed_fluid_id >= 1 .and. signed_fluid_id <= num_fluids) &
                                            failure_rho_g = failure_rho_g + &
                                                q_cons_vf(contxb + signed_fluid_id - 1)%sf(j, k, l)
                                    end do
                                end if
                                failure_sum_rhoY = 0._wp
                                $:GPU_LOOP(parallelism='[seq]')
                                do signed_species_eqn = chemxb, chemxe
                                    failure_sum_rhoY = failure_sum_rhoY + &
                                        q_cons_vf(signed_species_eqn)%sf(j, k, l)
                                end do
                                phase_change_signed_failure_ijk = (/j, k, l/)
                                phase_change_signed_failure_data(1) = fuel_before_coupling
                                phase_change_signed_failure_data(2) = delta_m_vapor
                                phase_change_signed_failure_data(3) = fuel_candidate
                                phase_change_signed_failure_data(4) = failure_rho_g
                                phase_change_signed_failure_data(5) = failure_sum_rhoY
                                phase_change_signed_failure_data(6) = q_cons_vf(advxb)%sf(j, k, l)
                                phase_change_signed_failure_data(7) = q_cons_vf(advxb + 1)%sf(j, k, l)
                                phase_change_signed_failure_data(8) = q_cons_vf(advxb + 2)%sf(j, k, l)
                                phase_change_signed_failure_data(9) = pS
                                phase_change_signed_failure_data(10) = TS
                                phase_change_signed_failure_data(11) = m2
                                phase_change_signed_failure_data(12) = m2_after
                            end if
                        else
                            q_cons_vf(chemxb + fuel_species_id - 1)%sf(j, k, l) = fuel_candidate
                            if (fuel_candidate < 0._wp) then
                                $:GPU_ATOMIC(atomic='update')
                                phase_change_signed_local_slight_negative = &
                                    phase_change_signed_local_slight_negative + 1
                            end if
                        end if
                    end if
#endif
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

#ifdef MFC_SIMULATION
        if (phase_change_evap_only_enabled) then
            call s_mpi_allreduce_integer_sum(local_evap_only_rejections, &
                                             global_evap_only_rejections)
            call s_mpi_allreduce_sum(local_evap_only_suppressed_mass, &
                                     global_evap_only_suppressed_mass)
            call s_mpi_allreduce_max(local_evap_only_max_delta, &
                                     global_evap_only_max_delta)
            call s_mpi_allreduce_min(local_evap_only_min_alpha_liq, &
                                     global_evap_only_min_alpha_liq)
            call s_mpi_allreduce_max(local_evap_only_max_alpha_liq, &
                                     global_evap_only_max_alpha_liq)
            call s_mpi_allreduce_min(local_evap_only_min_pressure, &
                                     global_evap_only_min_pressure)
            call s_mpi_allreduce_max(local_evap_only_max_pressure, &
                                     global_evap_only_max_pressure)
            call s_mpi_allreduce_min(local_evap_only_min_temperature, &
                                     global_evap_only_min_temperature)
            call s_mpi_allreduce_max(local_evap_only_max_temperature, &
                                     global_evap_only_max_temperature)
            phase_change_evap_only_rejections = phase_change_evap_only_rejections + &
                                                global_evap_only_rejections
            phase_change_evap_only_suppressed_mass = phase_change_evap_only_suppressed_mass + &
                                                     global_evap_only_suppressed_mass
            phase_change_evap_only_max_delta = max(phase_change_evap_only_max_delta, &
                                                   global_evap_only_max_delta)
            if (proc_rank == 0 .and. global_evap_only_rejections > 0) then
                write (output_unit, '(&
                    &"TEMP_PHASE_CHANGE_EVAP_ONLY t_step=",I0," stage=",I0,&
                    &" rejected_count=",I0," cumulative_rejected_count=",I0,&
                    &" suppressed_mass=",ES16.8,&
                    &" cumulative_suppressed_mass=",ES16.8,&
                    &" max_proposed_cond_delta_rho=",ES16.8,&
                    &" cumulative_max_proposed_cond_delta_rho=",ES16.8,&
                    &" alpha_liq_min=",ES16.8," alpha_liq_max=",ES16.8,&
                    &" pressure_min=",ES16.8," pressure_max=",ES16.8,&
                    &" temperature_min=",ES16.8," temperature_max=",ES16.8,&
                    &" no_transfer_committed=T validation_only=T")') &
                    report_step, report_stage, global_evap_only_rejections, &
                    phase_change_evap_only_rejections, global_evap_only_suppressed_mass, &
                    phase_change_evap_only_suppressed_mass, global_evap_only_max_delta, &
                    phase_change_evap_only_max_delta, &
                    merge(global_evap_only_min_alpha_liq, 0._wp, global_evap_only_rejections > 0), &
                    merge(global_evap_only_max_alpha_liq, 0._wp, global_evap_only_rejections > 0), &
                    merge(global_evap_only_min_pressure, 0._wp, global_evap_only_rejections > 0), &
                    merge(global_evap_only_max_pressure, 0._wp, global_evap_only_rejections > 0), &
                    merge(global_evap_only_min_temperature, 0._wp, global_evap_only_rejections > 0), &
                    merge(global_evap_only_max_temperature, 0._wp, global_evap_only_rejections > 0)
                call flush(output_unit)
            end if
        end if

        if (phase_change_budget_debug_enabled .and. &
            report_step >= phase_change_budget_target_step_min .and. &
            report_step <= phase_change_budget_target_step_max) then
            call s_mpi_allreduce_integer_sum(local_budget_evap_cells, global_budget_evap_cells)
            call s_mpi_allreduce_integer_sum(local_budget_cond_cells, global_budget_cond_cells)
            call s_mpi_allreduce_integer_sum(local_budget_insufficient_cells, &
                                             global_budget_insufficient_cells)
            call s_mpi_allreduce_integer_sum(local_budget_interface_cond_cells, &
                                             global_budget_interface_cond_cells)
            call s_mpi_allreduce_integer_sum(local_budget_liquid_cond_cells, &
                                             global_budget_liquid_cond_cells)
            call s_mpi_allreduce_integer_sum(local_budget_gas_cond_cells, &
                                             global_budget_gas_cond_cells)
            call s_mpi_allreduce_sum(local_budget_evap_mass, global_budget_evap_mass)
            call s_mpi_allreduce_sum(local_budget_cond_mass, global_budget_cond_mass)
            call s_mpi_allreduce_sum(local_budget_net_mass, global_budget_net_mass)
            call s_mpi_allreduce_sum(local_budget_insufficient_mass, &
                                     global_budget_insufficient_mass)
            call s_mpi_allreduce_max(local_budget_max_evap_delta, global_budget_max_evap_delta)
            call s_mpi_allreduce_max(local_budget_max_cond_delta, global_budget_max_cond_delta)
            call s_mpi_allreduce_max(local_budget_max_insufficient_ratio, &
                                     global_budget_max_insufficient_ratio)
            call s_mpi_allreduce_min(local_budget_min_alpha_liq, global_budget_min_alpha_liq)
            call s_mpi_allreduce_max(local_budget_max_alpha_liq, global_budget_max_alpha_liq)
            call s_mpi_allreduce_min(local_budget_min_alpha_vap, global_budget_min_alpha_vap)
            call s_mpi_allreduce_max(local_budget_max_alpha_vap, global_budget_max_alpha_vap)
            call s_mpi_allreduce_min(local_budget_min_alpha_air, global_budget_min_alpha_air)
            call s_mpi_allreduce_max(local_budget_max_alpha_air, global_budget_max_alpha_air)
            call s_mpi_allreduce_min(local_budget_min_pressure, global_budget_min_pressure)
            call s_mpi_allreduce_max(local_budget_max_pressure, global_budget_max_pressure)
            call s_mpi_allreduce_min(local_budget_min_temperature, global_budget_min_temperature)
            call s_mpi_allreduce_max(local_budget_max_temperature, global_budget_max_temperature)
            call s_mpi_allreduce_min(local_budget_min_vapor_arho, global_budget_min_vapor_arho)
            call s_mpi_allreduce_max(local_budget_max_vapor_arho, global_budget_max_vapor_arho)
            call s_mpi_allreduce_min(local_budget_min_fuel, global_budget_min_fuel)
            call s_mpi_allreduce_max(local_budget_max_fuel, global_budget_max_fuel)
            call s_mpi_allreduce_min(local_budget_min_request_ratio, global_budget_min_request_ratio)
            call s_mpi_allreduce_max(local_budget_max_request_ratio, global_budget_max_request_ratio)

            phase_change_budget_cum_evap_cells = phase_change_budget_cum_evap_cells + &
                                                 global_budget_evap_cells
            phase_change_budget_cum_cond_cells = phase_change_budget_cum_cond_cells + &
                                                 global_budget_cond_cells
            phase_change_budget_cum_insufficient_cells = &
                phase_change_budget_cum_insufficient_cells + global_budget_insufficient_cells
            phase_change_budget_cum_evap_mass = phase_change_budget_cum_evap_mass + &
                                                global_budget_evap_mass
            phase_change_budget_cum_cond_mass = phase_change_budget_cum_cond_mass + &
                                                global_budget_cond_mass
            phase_change_budget_cum_net_mass = phase_change_budget_cum_net_mass + &
                                               global_budget_net_mass
            phase_change_budget_cum_insufficient_mass = &
                phase_change_budget_cum_insufficient_mass + global_budget_insufficient_mass
            phase_change_budget_max_evap_delta = max(phase_change_budget_max_evap_delta, &
                                                     global_budget_max_evap_delta)
            phase_change_budget_max_cond_delta = max(phase_change_budget_max_cond_delta, &
                                                     global_budget_max_cond_delta)

            budget_evap_ratio = 0._wp
            if (global_budget_evap_mass > 0._wp) &
                budget_evap_ratio = global_budget_cond_mass/global_budget_evap_mass
            if (proc_rank == 0) then
                write (output_unit, '(&
                    &"TEMP_PHASE_CHANGE_BUDGET t_step=",I0," stage=",I0,&
                    &" evap_cells=",I0," cond_cells=",I0,&
                    &" insufficient_cells=",I0,&
                    &" cond_liquid_cells=",I0," cond_interface_cells=",I0,&
                    &" cond_gas_cells=",I0,&
                    &" evap_mass=",ES16.8," cond_mass=",ES16.8,&
                    &" net_mass=",ES16.8," cond_evap_ratio=",ES16.8,&
                    &" cum_evap_mass=",ES16.8," cum_cond_mass=",ES16.8,&
                    &" cum_net_mass=",ES16.8," cum_cond_evap_ratio=",ES16.8,&
                    &" max_evap_delta_rho=",ES16.8,&
                    &" max_cond_delta_rho=",ES16.8,&
                    &" insufficient_excess_mass=",ES16.8,&
                    &" cum_insufficient_excess_mass=",ES16.8,&
                    &" max_insufficient_ratio=",ES16.8,&
                    &" alpha_liq_min=",ES16.8," alpha_liq_max=",ES16.8,&
                    &" alpha_vap_min=",ES16.8," alpha_vap_max=",ES16.8,&
                    &" alpha_air_min=",ES16.8," alpha_air_max=",ES16.8,&
                    &" pressure_min=",ES16.8," pressure_max=",ES16.8,&
                    &" temperature_min=",ES16.8," temperature_max=",ES16.8,&
                    &" vapor_arho_min=",ES16.8," vapor_arho_max=",ES16.8,&
                    &" fuel_rhoY_min=",ES16.8," fuel_rhoY_max=",ES16.8,&
                    &" requested_over_fuel_min=",ES16.8,&
                    &" requested_over_fuel_max=",ES16.8)') &
                    report_step, report_stage, global_budget_evap_cells, global_budget_cond_cells, &
                    global_budget_insufficient_cells, global_budget_liquid_cond_cells, &
                    global_budget_interface_cond_cells, global_budget_gas_cond_cells, &
                    global_budget_evap_mass, global_budget_cond_mass, global_budget_net_mass, &
                    budget_evap_ratio, phase_change_budget_cum_evap_mass, &
                    phase_change_budget_cum_cond_mass, phase_change_budget_cum_net_mass, &
                    merge(phase_change_budget_cum_cond_mass/phase_change_budget_cum_evap_mass, &
                          0._wp, phase_change_budget_cum_evap_mass > 0._wp), &
                    global_budget_max_evap_delta, global_budget_max_cond_delta, &
                    global_budget_insufficient_mass, phase_change_budget_cum_insufficient_mass, &
                    global_budget_max_insufficient_ratio, &
                    merge(global_budget_min_alpha_liq, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_alpha_liq, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_alpha_vap, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_alpha_vap, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_alpha_air, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_alpha_air, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_pressure, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_pressure, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_temperature, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_temperature, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_vapor_arho, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_vapor_arho, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_fuel, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_fuel, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_min_request_ratio, 0._wp, global_budget_cond_cells > 0), &
                    merge(global_budget_max_request_ratio, 0._wp, global_budget_cond_cells > 0)
                call flush(output_unit)
            end if
        end if

        if (phase_change_fuel_mass_coupling_fix_enabled) then
            $:GPU_UPDATE(host='[phase_change_signed_local_condensation,phase_change_signed_local_insufficient,phase_change_signed_local_slight_negative,phase_change_signed_failure_claimed,phase_change_signed_failure_ijk,phase_change_signed_failure_data]')
            if (phase_change_signed_local_slight_negative > 0 .and. &
                phase_change_signed_condensation_slight_negative == 0) then
                write (output_unit, '(&
                    &"TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING slight_negative_count=",I0,&
                    &" tolerance=",ES16.8," t_step=",I0," stage=",I0," rank=",I0)') &
                    phase_change_signed_local_slight_negative, phase_change_fuel_mass_tolerance, &
                    report_step, report_stage, proc_rank
                call flush(output_unit)
            end if
            phase_change_signed_condensation_occurrences = phase_change_signed_condensation_occurrences + &
                                                            phase_change_signed_local_condensation
            phase_change_signed_condensation_insufficient = phase_change_signed_condensation_insufficient + &
                                                            phase_change_signed_local_insufficient
            phase_change_signed_condensation_slight_negative = &
                phase_change_signed_condensation_slight_negative + &
                phase_change_signed_local_slight_negative
            if (phase_change_signed_local_insufficient > 0) then
                if (phase_change_signed_failure_claimed /= 0) then
                    j = phase_change_signed_failure_ijk(1)
                    k = phase_change_signed_failure_ijk(2)
                    l = phase_change_signed_failure_ijk(3)
                    failure_x = x_cc(j)
                    failure_y = 0._wp
                    failure_z = 0._wp
                    if (n > 0) failure_y = y_cc(k)
                    if (p > 0) failure_z = z_cc(l)
                    write (output_unit, '(&
                        &"TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING insufficient_fuel=T",&
                        &" tolerance=",ES16.8," t_step=",I0," stage=",I0," rank=",I0,&
                        &" local_ijk=",3(I0,1X)," xyz=",3(ES16.8,1X),&
                        &" available_fuel=",ES16.8," delta_m_vapor=",ES16.8,&
                        &" requested_condensation_removal=",ES16.8," candidate_fuel=",ES16.8,&
                        &" rho_g=",ES16.8," sum_rhoY=",ES16.8," alpha=",3(ES16.8,1X),&
                        &" pressure=",ES16.8," temperature=",ES16.8,&
                        &" m2_before=",ES16.8," m2_after=",ES16.8)') &
                        phase_change_fuel_mass_tolerance, report_step, report_stage, proc_rank, j, k, l, &
                        failure_x, failure_y, failure_z, phase_change_signed_failure_data(1), &
                        phase_change_signed_failure_data(2), -phase_change_signed_failure_data(2), &
                        phase_change_signed_failure_data(3:12)
                    call flush(output_unit)
                end if
                call s_mpi_abort("TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING insufficient NC12H26 for signed condensation removal")
            end if
        end if

#endif

    end subroutine s_infinite_relaxation_k

    !>  This auxiliary subroutine is created to activate the pT-equilibrium for N fluids
        !!  @param j generic loop iterator for x direction
        !!  @param k generic loop iterator for y direction
        !!  @param l generic loop iterator for z direction
        !!  @param MFL flag that tells whether the fluid is gas (0), liquid (1), or a mixture (2)
        !!  @param pS equilibrium pressure at the interface
        !!  @param p_infpT stiffness for the participating fluids under pT-equilibrium
        !!  @param q_cons_vf Cell-average conservative variables
        !!  @param rhoe mixture energy
        !!  @param TS equilibrium temperature at the interface
    subroutine s_infinite_pt_relaxation_k(j, k, l, MFL, pS, p_infpT, q_cons_vf, rhoe, TS)
        $:GPU_ROUTINE(function_name='s_infinite_pt_relaxation_k', &
            & parallelism='[seq]', cray_inline=True)

        ! initializing variables
        integer, intent(in) :: j, k, l, MFL
        real(wp), intent(out) :: pS
        real(wp), dimension(1:), intent(out) :: p_infpT
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        real(wp), intent(in) :: rhoe
        real(wp), intent(out) :: TS
        real(wp) :: gp, gpp, hp, pO, mCP, mQ !< variables for the Newton Solver
        real(wp) :: p_infpT_sum

        integer :: i, ns !< generic loop iterators

        ! auxiliary variables for the pT-equilibrium solver
        mCP = 0.0_wp; mQ = 0.0_wp; p_infpT_sum = 0._wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            p_infpT(i) = ps_inf(i)
            p_infpT_sum = p_infpT_sum + abs(p_infpT(i))
        end do
        ! Performing tests before initializing the pT-equilibrium
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids

            ! sum of the total alpha*rho*cp of the system
            mCP = mCP + q_cons_vf(i + contxb - 1)%sf(j, k, l)*cvs(i)*gs_min(i)

            ! sum of the total alpha*rho*q of the system
            mQ = mQ + q_cons_vf(i + contxb - 1)%sf(j, k, l)*qvs(i)

        end do

        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            if (num_fluids < 3) then
                $:GPU_LOOP(parallelism='[seq]')
                do i = num_fluids + 1, 3
                    p_infpT(i) = p_infpT_sum
                end do
            end if
        #:endif

        ! Checking energy constraint
        if ((rhoe - mQ - minval(p_infpT)) < 0.0_wp) then

            if ((MFL == 0) .or. (MFL == 1)) then

                ! Assigning zero values for mass depletion cases
                ! pressure
                pS = 0.0_wp

                ! temperature
                TS = 0.0_wp

                return
            end if

        end if

        ! calculating initial estimate for pressure in the pT-relaxation procedure. I will also use this variable to
        ! iterate over the Newton's solver
        pO = 0.0_wp

        ! Maybe improve this condition afterwards. As long as the initial guess is in between -min(ps_inf)
        ! and infinity, a solution should be able to be found.
        pS = 1.0e4_wp

        ! Newton Solver for the pT-equilibrium
        ns = 0
        ! change this relative error metric. 1.e4_wp is just arbitrary
        do while ((abs(pS - pO) > palpha_eps) .and. (abs((pS - pO)/pO) > palpha_eps/1.e4_wp) .or. (ns == 0))

            ! increasing counter
            ns = ns + 1

            ! updating old pressure
            pO = pS

            ! updating functions used in the Newton's solver
            gpp = 0.0_wp; gp = 0.0_wp; hp = 0.0_wp
            $:GPU_LOOP(parallelism='[seq]')
            do i = 1, num_fluids

                gp = gp + (gs_min(i) - 1.0_wp)*q_cons_vf(i + contxb - 1)%sf(j, k, l)*cvs(i) &
                     *(rhoe + pS - mQ)/(mCP*(pS + p_infpT(i)))

                gpp = gpp + (gs_min(i) - 1.0_wp)*q_cons_vf(i + contxb - 1)%sf(j, k, l)*cvs(i) &
                      *(p_infpT(i) - rhoe + mQ)/(mCP*(pS + p_infpT(i))**2)

            end do

            hp = 1.0_wp/(rhoe + pS - mQ) + 1.0_wp/(pS + minval(p_infpT))

            ! updating common pressure for the newton solver
            pS = pO + ((1.0_wp - gp)/gpp)/(1.0_wp - (1.0_wp - gp + abs(1.0_wp - gp)) &
                                           /(2.0_wp*gpp)*hp)
        end do

        ! common temperature
        TS = (rhoe + pS - mQ)/mCP

    end subroutine s_infinite_pt_relaxation_k

    !>  This auxiliary subroutine is created to activate the pTg-equilibrium for N fluids under pT
        !!      and 2 fluids under pTg-equilibrium. There is a final common p and T during relaxation
        !!  @param j generic loop iterator for x direction
        !!  @param k generic loop iterator for y direction
        !!  @param l generic loop iterator for z direction
        !!  @param pS equilibrium pressure at the interface
        !!  @param p_infpT stiffness for the participating fluids under pT-equilibrium
        !!  @param rhoe mixture energy
        !!  @param q_cons_vf Cell-average conservative variables
        !!  @param TS equilibrium temperature at the interface
    subroutine s_infinite_ptg_relaxation_k(j, k, l, pS, p_infpT, rhoe, q_cons_vf, TS)
        $:GPU_ROUTINE(function_name='s_infinite_ptg_relaxation_k', &
            & parallelism='[seq]', cray_inline=True)

        integer, intent(in) :: j, k, l
        real(wp), intent(inout) :: pS
        real(wp), dimension(1:), intent(in) :: p_infpT
        real(wp), intent(in) :: rhoe
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(inout) :: TS
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: p_infpTg !< stiffness for the participating fluids for pTg-equilibrium
        #:else
            real(wp), dimension(num_fluids) :: p_infpTg !< stiffness for the participating fluids for pTg-equilibrium
        #:endif
        real(wp), dimension(2, 2) :: Jac, InvJac, TJac !< matrices for the Newton Solver
        real(wp), dimension(2) :: R2D, DeltamP !< residual and correction array
        real(wp) :: Om ! underrelaxation factor
        real(wp) :: mCP, mCPD, mCVGP, mCVGP2, mQ, mQD ! auxiliary variables for the pTg-solver
        real(wp) :: ml, mT, dFdT, dTdm, dTdp

        !< Generic loop iterators
        integer :: i, ns
        ! pTg-equilibrium solution procedure
        ! Newton Solver parameters
        ! counter
        ns = 0

        ! Relaxation factor
        Om = 1.0e-3_wp

        p_infpTg = p_infpT

        if (((pS < 0.0_wp) .and. ((q_cons_vf(lp + contxb - 1)%sf(j, k, l) &
                                   + q_cons_vf(vp + contxb - 1)%sf(j, k, l)) > ((rhoe &
                                                                                 - gs_min(lp)*ps_inf(lp)/(gs_min(lp) - 1))/qvs(lp)))) .or. &
            ((pS >= 0.0_wp) .and. (pS < 1.0e-1_wp))) then

            ! improve this initial condition
            pS = 1.0e4_wp

        end if

        ! Loop until the solution for F(X) is satisfied
        ! Check whether I need to use both absolute and relative values
        ! for the residual, and how to do it adequately.
        ! Dummy guess to start the pTg-equilibrium problem.
        ! improve this initial condition
        R2D(1) = 0.0_wp; R2D(2) = 0.0_wp
        DeltamP(1) = 0.0_wp; DeltamP(2) = 0.0_wp
        do while (((sqrt(R2D(1)**2 + R2D(2)**2) > ptgalpha_eps) &
                   .and. ((sqrt(R2D(1)**2 + R2D(2)**2)/rhoe) > (ptgalpha_eps/1.e6_wp))) &
                  .or. (ns == 0))

            ! Updating counter for the iterative procedure
            ns = ns + 1

            ! Auxiliary variables to help in the calculation of the residue
            mCP = 0.0_wp; mCPD = 0.0_wp; mCVGP = 0.0_wp; mCVGP2 = 0.0_wp; mQ = 0.0_wp; mQD = 0.0_wp
            ! Those must be updated through the iterations, as they either depend on
            ! the partial masses for all fluids, or on the equilibrium pressure
            $:GPU_LOOP(parallelism='[seq]')
            do i = 1, num_fluids

                ! sum of the total alpha*rho*cp of the system
                mCP = mCP + q_cons_vf(i + contxb - 1)%sf(j, k, l) &
                      *cvs(i)*gs_min(i)

                ! sum of the total alpha*rho*q of the system
                mQ = mQ + q_cons_vf(i + contxb - 1)%sf(j, k, l)*qvs(i)

                ! These auxiliary variables now need to be updated, as the partial densities now
                ! vary at every iteration
                if ((i /= lp) .and. (i /= vp)) then

                    mCVGP = mCVGP + q_cons_vf(i + contxb - 1)%sf(j, k, l) &
                            *cvs(i)*(gs_min(i) - 1)/(pS + ps_inf(i))

                    mCVGP2 = mCVGP2 + q_cons_vf(i + contxb - 1)%sf(j, k, l) &
                             *cvs(i)*(gs_min(i) - 1)/((pS + ps_inf(i))**2)

                    mQD = mQD + q_cons_vf(i + contxb - 1)%sf(j, k, l)*qvs(i)

                    ! sum of the total alpha*rho*cp of the system
                    mCPD = mCPD + q_cons_vf(i + contxb - 1)%sf(j, k, l)*cvs(i) &
                           *gs_min(i)

                end if

            end do

            ! calculating the (2D) Jacobian Matrix used in the solution of the pTg-quilibrium model

            ! mass of the reacting liquid
            ml = q_cons_vf(lp + contxb - 1)%sf(j, k, l)

            ! mass of the two participating fluids
            mT = q_cons_vf(lp + contxb - 1)%sf(j, k, l) &
                 + q_cons_vf(vp + contxb - 1)%sf(j, k, l)

            TS = 1/(mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) &
                    + ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                          - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                    + mCVGP)

            dFdT = &
                -(cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp))*log(TS) &
                - (qvps(lp) - qvps(vp)) &
                + cvs(lp)*(gs_min(lp) - 1)*log(pS + ps_inf(lp)) &
                - cvs(vp)*(gs_min(vp) - 1)*log(pS + ps_inf(vp))

            dTdm = -(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                     - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)))*TS**2

            dTdp = (mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2 &
                    + ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp))**2 &
                          - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2) &
                    + mCVGP2)*TS**2

            ! F = (F1,F2) is the function whose roots we are looking for
            ! x = (m1, p) are the independent variables. m1 = mass of the first participant fluid, p = pressure
            ! F1 = 0 is the Gibbs free energy quality
            ! F2 = 0 is the enforcement of the thermodynamic (total - kinectic) energy
            ! dF1dm
            Jac(1, 1) = dFdT*dTdm

            ! dF1dp
            Jac(1, 2) = dFdT*dTdp + TS &
                        *(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                          - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)))

            ! dF2dm
            Jac(2, 1) = (qvs(vp) - qvs(lp) &
                         + (cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp)) &
                         /(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                               - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                           + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP) &
                         - (ml*(cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp)) &
                            - mT*cvs(vp)*gs_min(vp) - mCPD) &
                         *(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                           - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                         /((ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                                - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                            + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP)**2))/1
            ! dF2dp
            Jac(2, 2) = (1 + (ml*(cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp)) &
                              - mT*cvs(vp)*gs_min(vp) - mCPD) &
                         *(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp))**2 &
                               - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2) &
                           + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2 + mCVGP2) &
                         /(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                               - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                           + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP)**2)/1

            ! intermediate elements of J^{-1}
            InvJac(1, 1) = Jac(2, 2)
            InvJac(1, 2) = -1.0_wp*Jac(1, 2)
            InvJac(2, 1) = -1.0_wp*Jac(2, 1)
            InvJac(2, 2) = Jac(1, 1)

            ! elements of J^{T}
            TJac(1, 1) = Jac(1, 1)
            TJac(1, 2) = Jac(2, 1)
            TJac(2, 1) = Jac(1, 2)
            TJac(2, 2) = Jac(2, 2)

            ! dividing by det(J)
            InvJac = InvJac/(Jac(1, 1)*Jac(2, 2) - Jac(1, 2)*Jac(2, 1))

            ! calculating correction array for Newton's method
            DeltamP = -1.0_wp*(matmul(InvJac, R2D))

            ! updating two reacting 'masses'. Recall that inert 'masses' do not change during the phase change
            ! liquid
            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = q_cons_vf(lp + contxb - 1)%sf(j, k, l) + Om*DeltamP(1)

            ! gas
            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = q_cons_vf(vp + contxb - 1)%sf(j, k, l) - Om*DeltamP(1)

            ! updating pressure
            pS = pS + Om*DeltamP(2)

            ! calculating residuals, which are (i) the difference between the Gibbs Free energy of the gas and the liquid
            ! and (ii) the energy before and after the phase-change process.

            ! mass of the reacting liquid
            ml = q_cons_vf(lp + contxb - 1)%sf(j, k, l)

            ! mass of the two participating fluids
            mT = q_cons_vf(lp + contxb - 1)%sf(j, k, l) &
                 + q_cons_vf(vp + contxb - 1)%sf(j, k, l)

            TS = 1/(mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) &
                    + ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                          - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                    + mCVGP)

            ! Gibbs Free Energy Equality condition (DG)
            R2D(1) = TS*((cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp)) &
                         *(1 - log(TS)) - (qvps(lp) - qvps(vp)) &
                         + cvs(lp)*(gs_min(lp) - 1)*log(pS + ps_inf(lp)) &
                         - cvs(vp)*(gs_min(vp) - 1)*log(pS + ps_inf(vp))) &
                     + qvs(lp) - qvs(vp)

            ! Constant Energy Process condition (DE)
            R2D(2) = (rhoe + pS &
                      + ml*(qvs(vp) - qvs(lp)) - mT*qvs(vp) - mQD &
                      + (ml*(gs_min(vp)*cvs(vp) - gs_min(lp)*cvs(lp)) &
                         - mT*gs_min(vp)*cvs(vp) - mCPD) &
                      /(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) &
                            - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) &
                        + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP))/1

        end do

        ! common temperature
        TS = (rhoe + pS - mQ)/mCP
    end subroutine s_infinite_ptg_relaxation_k

    !>  This auxiliary subroutine corrects the partial densities of the REACTING fluids in case one of them is negative
        !!      but their sum is positive. Inert phases are not corrected at this moment
        !!  @param MCT partial density correction parameter
        !!  @param q_cons_vf Cell-average conservative variables
        !!  @param rM sum of the reacting masses
        !!  @param j generic loop iterator for x direction
        !!  @param k generic loop iterator for y direction
        !!  @param l generic loop iterator for z direction
    subroutine s_correct_partial_densities(MCT, q_cons_vf, rM, j, k, l)
        $:GPU_ROUTINE(function_name='s_correct_partial_densities', &
            & parallelism='[seq]', cray_inline=True)

        !> @name variables for the correction of the reacting partial densities
        !> @{
        real(wp), intent(out) :: MCT
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(inout) :: rM
        integer, intent(in) :: j, k, l
        !> @}
        if (rM < 0.0_wp) then

            if ((q_cons_vf(lp + contxb - 1)%sf(j, k, l) >= -1.0_wp*mixM) .and. &
                (q_cons_vf(vp + contxb - 1)%sf(j, k, l) >= -1.0_wp*mixM)) then

                q_cons_vf(lp + contxb - 1)%sf(j, k, l) = 0.0_wp

                q_cons_vf(vp + contxb - 1)%sf(j, k, l) = 0.0_wp

                rM = q_cons_vf(lp + contxb - 1)%sf(j, k, l) + q_cons_vf(vp + contxb - 1)%sf(j, k, l)

            end if

        end if

        ! Defining the correction in terms of an absolute value might not be the best practice.
        ! Maybe a good way to do this is to partition the partial densities, giving a small percentage of the total reacting density
        MCT = 2*mixM

        ! correcting the partial densities of the reacting fluids. What to do for the nonreacting ones?
        if (q_cons_vf(lp + contxb - 1)%sf(j, k, l) < 0.0_wp) then

            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = MCT*rM

            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = (1.0_wp - MCT)*rM

        elseif (q_cons_vf(vp + contxb - 1)%sf(j, k, l) < 0.0_wp) then

            q_cons_vf(lp + contxb - 1)%sf(j, k, l) = (1.0_wp - MCT)*rM

            q_cons_vf(vp + contxb - 1)%sf(j, k, l) = MCT*rM

        end if
    end subroutine s_correct_partial_densities

    !>  This auxiliary subroutine finds the Saturation temperature for a given
        !!      saturation pressure through a newton solver
        !!  @param pSat Saturation Pressure
        !!  @param TSat Saturation Temperature
        !!  @param TSIn equilibrium Temperature
    elemental subroutine s_TSat(pSat, TSat, TSIn)
        $:GPU_ROUTINE(function_name='s_TSat',parallelism='[seq]', &
            & cray_inline=True)

        real(wp), intent(in) :: pSat
        real(wp), intent(out) :: TSat
        real(wp), intent(in) :: TSIn

        real(wp) :: dFdT, FT, Om !< auxiliary variables

        ! Generic loop iterators
        integer :: ns

        if ((f_approx_equal(pSat, 0.0_wp)) .and. (f_approx_equal(TSIn, 0.0_wp))) then

            ! assigning Saturation temperature
            TSat = 0.0_wp

        else

            ! calculating initial estimate for temperature in the TSat procedure. I will also use this variable to
            ! iterate over the Newton's solver
            TSat = TSIn

            ! iteration counter
            ns = 0

            ! underrelaxation factor
            Om = 1.0e-3_wp
            do while ((abs(FT) > ptgalpha_eps) .or. (ns == 0))
                ! increasing counter
                ns = ns + 1

                ! calculating residual
                FT = TSat*((cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp)) &
                           *(1 - log(TSat)) - (qvps(lp) - qvps(vp)) &
                           + cvs(lp)*(gs_min(lp) - 1)*log(pSat + ps_inf(lp)) &
                           - cvs(vp)*(gs_min(vp) - 1)*log(pSat + ps_inf(vp))) &
                     + qvs(lp) - qvs(vp)

                ! calculating the jacobian
                dFdT = &
                    -(cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp))*log(TSat) &
                    - (qvps(lp) - qvps(vp)) &
                    + cvs(lp)*(gs_min(lp) - 1)*log(pSat + ps_inf(lp)) &
                    - cvs(vp)*(gs_min(vp) - 1)*log(pSat + ps_inf(vp))

                ! updating saturation temperature
                TSat = TSat - Om*FT/dFdT

            end do

        end if

    end subroutine s_TSat

    !>  This subroutine finalizes the phase change module
    impure subroutine s_finalize_relaxation_solver_module
#ifdef MFC_SIMULATION
        integer :: global_condensation_occurrences, global_condensation_insufficient
        integer :: global_condensation_slight_negative
        integer :: global_evap_only_rejections
        real(wp) :: global_evap_only_suppressed_mass, global_evap_only_max_delta

        if (phase_change_evap_only_enabled) then
            call s_mpi_allreduce_integer_sum(phase_change_evap_only_rejections, &
                                             global_evap_only_rejections)
            call s_mpi_allreduce_sum(phase_change_evap_only_suppressed_mass, &
                                     global_evap_only_suppressed_mass)
            call s_mpi_allreduce_max(phase_change_evap_only_max_delta, &
                                     global_evap_only_max_delta)
            if (proc_rank == 0) then
                write (output_unit, '(&
                    &"TEMP_PHASE_CHANGE_EVAP_ONLY_SUMMARY validation_only=T",&
                    &" rejected_count=",I0," suppressed_mass=",ES16.8,&
                    &" max_proposed_cond_delta_rho=",ES16.8)') &
                    global_evap_only_rejections, global_evap_only_suppressed_mass, &
                    global_evap_only_max_delta
                call flush(output_unit)
            end if
        end if

        if (phase_change_fuel_mass_coupling_fix_enabled) then
            call s_mpi_allreduce_integer_sum(phase_change_signed_condensation_occurrences, &
                                             global_condensation_occurrences)
            call s_mpi_allreduce_integer_sum(phase_change_signed_condensation_insufficient, &
                                             global_condensation_insufficient)
            call s_mpi_allreduce_integer_sum(phase_change_signed_condensation_slight_negative, &
                                             global_condensation_slight_negative)
            if (proc_rank == 0) then
                write (output_unit, '(&
                    &"TEMP_PHASE_CHANGE_FUEL_MASS_COUPLING_SUMMARY condensation_events=",I0,&
                    &" insufficient_fuel_events=",I0," slight_negative_events=",I0,&
                    &" tolerance=",ES16.8)') &
                    global_condensation_occurrences, global_condensation_insufficient, &
                    global_condensation_slight_negative, phase_change_fuel_mass_tolerance
                call flush(output_unit)
            end if
        end if
#endif
    end subroutine s_finalize_relaxation_solver_module

#endif

end module m_phase_change
