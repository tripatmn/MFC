!>
!! @file
!! @brief Contains module m_time_steppers

#:include 'macros.fpp'
#:include 'case.fpp'

!> @brief Total-variation-diminishing (TVD) Runge--Kutta time integrators (1st-, 2nd-, and 3rd-order SSP)
module m_time_steppers

    use iso_fortran_env, only: output_unit

    use m_derived_types        !< Definitions of the derived types

    use m_global_parameters    !< Definitions of the global parameters

    use m_rhs                  !< Right-hane-side (RHS) evaluation procedures

    use m_pressure_relaxation  !< Pressure relaxation procedures

    use m_data_output          !< Run-time info & solution data output procedures

    use m_bubbles_EE           !< Ensemble-averaged bubble dynamics routines

    use m_bubbles_EL           !< Lagrange bubble dynamics routines

    use m_ibm

    use m_hyperelastic

    use m_mpi_proxy            !< Message passing interface (MPI) module proxy

    use m_boundary_common

    use m_helper

    use m_sim_helpers

    use m_fftw

    use m_nvtx

    use m_thermochem, only: num_species, species_names

    use m_body_forces

    use m_derived_variables

    implicit none

    type(vector_field), allocatable, dimension(:) :: q_cons_ts !<
    !! Cell-average conservative variables at each time-stage (TS)

    type(scalar_field), allocatable, dimension(:) :: q_prim_vf !<
    !! Cell-average primitive variables at the current time-stage

    type(scalar_field), allocatable, dimension(:) :: rhs_vf !<
    !! Cell-average RHS variables at the current time-stage

    type(integer_field), allocatable, dimension(:, :) :: bc_type !<
    !! Boundary condition identifiers

    type(vector_field), allocatable, dimension(:) :: q_prim_ts1, q_prim_ts2 !<
    !! Cell-average primitive variables at consecutive TIMESTEPS

    real(wp), allocatable, dimension(:, :, :, :, :) :: rhs_pb

    type(scalar_field) :: q_T_sf !<
    !! Cell-average temperature variables at the current time-stage

    type(scalar_field) :: m_dot_evap !<
    !! Cell-average liquid-to-vapor mass-transfer rate from relaxation

    real(wp), allocatable, dimension(:, :, :, :, :) :: rhs_mv

    real(wp), allocatable, dimension(:, :, :) :: max_dt

    integer, private :: num_ts !<
    !! Number of time stages in the time-stepping scheme

    integer :: stor !< storage index
    real(wp), allocatable, dimension(:, :) :: rk_coef
    integer, private :: num_probe_ts

    $:GPU_DECLARE(create='[q_cons_ts,q_prim_vf,q_T_sf,m_dot_evap,rhs_vf,q_prim_ts1,q_prim_ts2,rhs_mv,rhs_pb,max_dt,rk_coef,stor,bc_type]')

!> @cond
#if defined(__NVCOMPILER_GPU_UNIFIED_MEM)
    real(stp), allocatable, dimension(:, :, :, :), pinned, target :: q_cons_ts_pool_host
#elif defined(FRONTIER_UNIFIED)
    real(stp), pointer, contiguous, dimension(:, :, :, :) :: q_cons_ts_pool_host, q_cons_ts_pool_device
    integer(kind=8) :: pool_dims(4), pool_starts(4)
    integer(kind=8) :: pool_size
    type(c_ptr) :: cptr_host, cptr_device
#endif
!> @endcond

contains

    logical function s_zhang_evap_hang_diag_active(t_step)
        integer, intent(in) :: t_step

        character(len=16) :: env_value
        integer :: env_status

        call get_environment_variable("TEMP_ZHANG_EVAP_HANG_DIAG", env_value, status=env_status)
        s_zhang_evap_hang_diag_active = env_status == 0 .and. trim(env_value) == "1" &
                                        .and. t_step >= 9100 .and. t_step <= 9120
    end function s_zhang_evap_hang_diag_active

    subroutine s_zhang_evap_hang_trace(t_step, stage, label)
        integer, intent(in) :: t_step, stage
        character(len=*), intent(in) :: label

        if (.not. s_zhang_evap_hang_diag_active(t_step)) return

        print '(" TEMP_ZHANG_EVAP_HANG_DIAG rank=", I6, " t_step=", I8, " stage=", I4, " ", A)', &
            proc_rank, t_step, stage, trim(label)
        call flush(output_unit)
    end subroutine s_zhang_evap_hang_trace

    subroutine s_zhang_evap_hang_field_diag(q_cons_vf, q_prim_vf, t_step, stage, label)
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        integer, intent(in) :: t_step, stage
        character(len=*), intent(in) :: label

        real(wp) :: pres_min, pres_max
        real(wp) :: alpha_min, alpha_max
        real(wp) :: liq_arho_min, liq_arho_max
        real(wp) :: vap_arho_min, vap_arho_max
        real(wp) :: gas_rho_min, gas_rho_max
        real(wp) :: mdot_min, mdot_max
        real(wp), allocatable, dimension(:, :, :) :: gas_rho
        integer :: i, invalid_count

        if (.not. s_zhang_evap_hang_diag_active(t_step)) return

#ifndef FRONTIER_UNIFIED
        $:GPU_UPDATE(host='[q_prim_vf(E_idx)%sf]')
        do i = contxb, contxe
            $:GPU_UPDATE(host='[q_cons_vf(i)%sf]')
        end do
        do i = advxb, advxe
            $:GPU_UPDATE(host='[q_cons_vf(i)%sf]')
        end do
        if (relax) then
            $:GPU_UPDATE(host='[m_dot_evap%sf]')
        end if
#endif

        pres_min = minval(q_prim_vf(E_idx)%sf(0:m, 0:n, 0:p))
        pres_max = maxval(q_prim_vf(E_idx)%sf(0:m, 0:n, 0:p))

        alpha_min = huge(1._wp)
        alpha_max = -huge(1._wp)
        do i = advxb, advxe
            alpha_min = min(alpha_min, minval(q_cons_vf(i)%sf(0:m, 0:n, 0:p)))
            alpha_max = max(alpha_max, maxval(q_cons_vf(i)%sf(0:m, 0:n, 0:p)))
        end do

        liq_arho_min = minval(q_cons_vf(contxb)%sf(0:m, 0:n, 0:p))
        liq_arho_max = maxval(q_cons_vf(contxb)%sf(0:m, 0:n, 0:p))
        if (num_fluids >= 2) then
            vap_arho_min = minval(q_cons_vf(contxb + 1)%sf(0:m, 0:n, 0:p))
            vap_arho_max = maxval(q_cons_vf(contxb + 1)%sf(0:m, 0:n, 0:p))
        else
            vap_arho_min = 0._wp
            vap_arho_max = 0._wp
        end if

        allocate(gas_rho(0:m, 0:n, 0:p))
        gas_rho = 0._wp
        if (num_fluids >= 2) gas_rho = gas_rho + q_cons_vf(contxb + 1)%sf(0:m, 0:n, 0:p)
        if (num_fluids >= 3) gas_rho = gas_rho + q_cons_vf(contxb + 2)%sf(0:m, 0:n, 0:p)
        gas_rho_min = minval(gas_rho)
        gas_rho_max = maxval(gas_rho)
        deallocate(gas_rho)

        if (relax) then
            mdot_min = minval(m_dot_evap%sf(0:m, 0:n, 0:p))
            mdot_max = maxval(m_dot_evap%sf(0:m, 0:n, 0:p))
        else
            mdot_min = 0._wp
            mdot_max = 0._wp
        end if

        invalid_count = 0
        do i = 1, sys_size
            invalid_count = invalid_count + count(q_cons_vf(i)%sf(0:m, 0:n, 0:p) /= q_cons_vf(i)%sf(0:m, 0:n, 0:p))
            invalid_count = invalid_count + count(abs(q_cons_vf(i)%sf(0:m, 0:n, 0:p)) > huge(1._wp))
        end do

        print '(" TEMP_ZHANG_EVAP_HANG_DIAG_FIELDS rank=", I6, " t_step=", I8, " stage=", I4, " ", A, &
            & " p_min=", ES16.8, " p_max=", ES16.8, &
            & " alpha_min=", ES16.8, " alpha_max=", ES16.8, &
            & " liq_arho_min=", ES16.8, " liq_arho_max=", ES16.8, &
            & " vap_arho_min=", ES16.8, " vap_arho_max=", ES16.8, &
            & " gas_rho_min=", ES16.8, " gas_rho_max=", ES16.8, &
            & " mdot_min=", ES16.8, " mdot_max=", ES16.8, " invalid_count=", I10)', &
            proc_rank, t_step, stage, trim(label), pres_min, pres_max, alpha_min, alpha_max, &
            liq_arho_min, liq_arho_max, vap_arho_min, vap_arho_max, gas_rho_min, gas_rho_max, &
            mdot_min, mdot_max, invalid_count
        call flush(output_unit)
    end subroutine s_zhang_evap_hang_field_diag

    logical function s_ybc_edge_state_debug_active() result(is_active)
        character(len=16) :: env_value
        integer :: env_status

        call get_environment_variable("TEMP_YBC_EDGE_STATE_DEBUG", env_value, status=env_status)
        is_active = env_status == 0 .and. trim(env_value) == "1"
    end function s_ybc_edge_state_debug_active

    logical function s_ybc_edge_bad_real(value) result(is_bad)
        real(wp), intent(in) :: value

        is_bad = value /= value .or. abs(value) > huge(1._wp)/10._wp
    end function s_ybc_edge_bad_real

    subroutine s_ybc_edge_cons_debug_cell(q_cons_vf, label, t_step, stage, cell_j, cell_k, cell_l)
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        character(len=*), intent(in) :: label
        integer, intent(in) :: t_step, stage, cell_j, cell_k, cell_l

        integer :: global_j, global_k, global_l
        logical :: triggered
        character(len=160) :: reason
        real(wp) :: alpha_liq, alpha_vap, alpha_air, alpha_sum
        real(wp) :: arho_liq, arho_vap, arho_air, gas_alpha, gas_mass
        real(wp) :: rho, pressure, vel_x, vel_y, vel_mag, mom_x, mom_y

        if (cell_j < lbound(q_cons_vf(1)%sf, 1) .or. cell_j > ubound(q_cons_vf(1)%sf, 1)) return
        if (cell_k < lbound(q_cons_vf(1)%sf, 2) .or. cell_k > ubound(q_cons_vf(1)%sf, 2)) return
        if (cell_l < lbound(q_cons_vf(1)%sf, 3) .or. cell_l > ubound(q_cons_vf(1)%sf, 3)) return

        alpha_liq = 0._wp
        alpha_vap = 0._wp
        alpha_air = 0._wp
        arho_liq = 0._wp
        arho_vap = 0._wp
        arho_air = 0._wp
        if (num_fluids >= 1) then
            alpha_liq = q_cons_vf(advxb)%sf(cell_j, cell_k, cell_l)
            arho_liq = q_cons_vf(contxb)%sf(cell_j, cell_k, cell_l)
        end if
        if (num_fluids >= 2) then
            alpha_vap = q_cons_vf(advxb + 1)%sf(cell_j, cell_k, cell_l)
            arho_vap = q_cons_vf(contxb + 1)%sf(cell_j, cell_k, cell_l)
        end if
        if (num_fluids >= 3) then
            alpha_air = q_cons_vf(advxb + 2)%sf(cell_j, cell_k, cell_l)
            arho_air = q_cons_vf(contxb + 2)%sf(cell_j, cell_k, cell_l)
        end if
        alpha_sum = alpha_liq + alpha_vap + alpha_air
        gas_alpha = alpha_vap + alpha_air
        gas_mass = arho_vap + arho_air
        rho = arho_liq + arho_vap + arho_air

        mom_x = 0._wp
        mom_y = 0._wp
        if (num_vels >= 1) mom_x = q_cons_vf(momxb)%sf(cell_j, cell_k, cell_l)
        if (num_vels >= 2) mom_y = q_cons_vf(momxb + 1)%sf(cell_j, cell_k, cell_l)
        vel_x = 0._wp
        vel_y = 0._wp
        if (abs(rho) > tiny(1._wp)) then
            vel_x = mom_x/rho
            vel_y = mom_y/rho
        end if
        vel_mag = sqrt(vel_x*vel_x + vel_y*vel_y)

        pressure = q_cons_vf(E_idx)%sf(cell_j, cell_k, cell_l)
        if (allocated(q_prim_vf)) then
            if (cell_j >= lbound(q_prim_vf(E_idx)%sf, 1) .and. cell_j <= ubound(q_prim_vf(E_idx)%sf, 1) .and. &
                cell_k >= lbound(q_prim_vf(E_idx)%sf, 2) .and. cell_k <= ubound(q_prim_vf(E_idx)%sf, 2) .and. &
                cell_l >= lbound(q_prim_vf(E_idx)%sf, 3) .and. cell_l <= ubound(q_prim_vf(E_idx)%sf, 3)) then
                pressure = q_prim_vf(E_idx)%sf(cell_j, cell_k, cell_l)
            end if
        end if

        triggered = .false.
        reason = "ok"
        if (s_ybc_edge_bad_real(alpha_liq) .or. s_ybc_edge_bad_real(alpha_vap) .or. &
            s_ybc_edge_bad_real(alpha_air) .or. s_ybc_edge_bad_real(arho_liq) .or. &
            s_ybc_edge_bad_real(arho_vap) .or. s_ybc_edge_bad_real(arho_air) .or. &
            s_ybc_edge_bad_real(rho) .or. s_ybc_edge_bad_real(pressure) .or. &
            s_ybc_edge_bad_real(mom_x) .or. s_ybc_edge_bad_real(mom_y) .or. &
            s_ybc_edge_bad_real(vel_x) .or. s_ybc_edge_bad_real(vel_y)) then
            triggered = .true.; reason = "nonfinite"
        elseif (alpha_air > 1.05_wp) then
            triggered = .true.; reason = "alpha_air_high"
        elseif (gas_alpha > 1.05_wp) then
            triggered = .true.; reason = "gas_alpha_high"
        elseif (alpha_sum < 0.95_wp .or. alpha_sum > 1.05_wp) then
            triggered = .true.; reason = "alpha_sum"
        elseif (pressure <= 1000._wp) then
            triggered = .true.; reason = "pressure_low"
        elseif (abs(vel_x) > 1.e5_wp .or. abs(vel_y) > 1.e5_wp) then
            triggered = .true.; reason = "velocity"
        end if

        if (.not. triggered) return

        global_j = cell_j
        global_k = cell_k
        global_l = cell_l
        if (allocated(start_idx)) then
            if (size(start_idx) >= 1) global_j = start_idx(1) + cell_j
            if (size(start_idx) >= 2) global_k = start_idx(2) + cell_k
            if (size(start_idx) >= 3) global_l = start_idx(3) + cell_l
        end if

        print '(" TEMP_YBC_EDGE_STATE_DEBUG label=", A, " rank=", I6, " t_step=", I10, " stage=", I4, &
            &" local_ijk=", 3(I8,1X), " global_ijk=", 3(I8,1X), " x=", ES14.6, " y=", ES14.6, &
            &" reason=", A)', &
            trim(label), proc_rank, t_step, stage, cell_j, cell_k, cell_l, global_j, global_k, global_l, &
            x_cc(max(min(cell_j, ubound(x_cc, 1)), lbound(x_cc, 1))), &
            y_cc(max(min(cell_k, ubound(y_cc, 1)), lbound(y_cc, 1))), trim(reason)
        print '(" TEMP_YBC_EDGE_STATE_DEBUG_STATE alpha_liq=", ES14.6, " alpha_vap=", ES14.6, &
            &" alpha_air=", ES14.6, " alpha_sum=", ES14.6, " arho_liq=", ES14.6, &
            &" arho_vap=", ES14.6, " arho_air=", ES14.6, " gas_alpha=", ES14.6, &
            &" gas_mass=", ES14.6)', &
            alpha_liq, alpha_vap, alpha_air, alpha_sum, arho_liq, arho_vap, arho_air, gas_alpha, gas_mass
        print '(" TEMP_YBC_EDGE_STATE_DEBUG_FLOW rho=", ES14.6, " pressure=", ES14.6, &
            &" mom_x=", ES14.6, " mom_y=", ES14.6, " u=", ES14.6, " v=", ES14.6, &
            &" vel_mag=", ES14.6)', &
            rho, pressure, mom_x, mom_y, vel_x, vel_y, vel_mag
        call flush(output_unit)
    end subroutine s_ybc_edge_cons_debug_cell

    subroutine s_ybc_edge_cons_debug_report(q_cons_vf, label, t_step, stage)
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        character(len=*), intent(in) :: label
        integer, intent(in) :: t_step, stage

        integer :: cell_j, cell_k, l0
        integer :: j_right_beg, k_top_beg

        if (.not. s_ybc_edge_state_debug_active()) return
        if (n == 0) return

        l0 = 0
        j_right_beg = max(0, m - 4)
        k_top_beg = max(0, n - 4)

        do cell_j = j_right_beg, m
            do cell_k = k_top_beg, n
                call s_ybc_edge_cons_debug_cell(q_cons_vf, label, t_step, stage, cell_j, cell_k, l0)
            end do
        end do
    end subroutine s_ybc_edge_cons_debug_report

    !> The computation of parameters, the allocation of memory,
        !!      the association of pointers and/or the execution of any
        !!      other procedures that are necessary to setup the module.
    impure subroutine s_initialize_time_steppers_module
#ifdef FRONTIER_UNIFIED
        use hipfort
        use hipfort_hipmalloc
        use hipfort_check
#if defined(MFC_OpenACC)
        use openacc
#endif
#endif
        integer :: i, j !< Generic loop iterators

        ! Setting number of time-stages for selected time-stepping scheme
        if (time_stepper == 1) then
            num_ts = 1
        elseif (any(time_stepper == (/2, 3/))) then
            num_ts = 2
        end if

        if (probe_wrt) then
            num_probe_ts = 2
        end if

        ! Allocating the cell-average conservative variables
        @:ALLOCATE(q_cons_ts(1:num_ts))
        @:PREFER_GPU(q_cons_ts)

        do i = 1, num_ts
            @:ALLOCATE(q_cons_ts(i)%vf(1:sys_size))
            @:PREFER_GPU(q_cons_ts(i)%vf)
        end do

!> @cond
#if defined(__NVCOMPILER_GPU_UNIFIED_MEM)
        if (num_ts == 2 .and. nv_uvm_out_of_core) then
            ! host allocation for q_cons_ts(2)%vf(j)%sf for all j
            allocate (q_cons_ts_pool_host(idwbuff(1)%beg:idwbuff(1)%end, &
                                          idwbuff(2)%beg:idwbuff(2)%end, &
                                          idwbuff(3)%beg:idwbuff(3)%end, &
                                          1:sys_size))
        end if

        do j = 1, sys_size
            ! q_cons_ts(1) lives on the device
            @:ALLOCATE(q_cons_ts(1)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end))
            @:PREFER_GPU(q_cons_ts(1)%vf(j)%sf)
            if (num_ts == 2) then
                if (nv_uvm_out_of_core) then
                    ! q_cons_ts(2) lives on the host
                    q_cons_ts(2)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                                          idwbuff(2)%beg:idwbuff(2)%end, &
                                          idwbuff(3)%beg:idwbuff(3)%end) => q_cons_ts_pool_host(:, :, :, j)
                else
                    @:ALLOCATE(q_cons_ts(2)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:PREFER_GPU(q_cons_ts(2)%vf(j)%sf)
                end if
            end if
        end do

        do i = 1, num_ts
            @:ACC_SETUP_VFs(q_cons_ts(i))
        end do
#elif defined(FRONTIER_UNIFIED)
        ! Allocate to memory regions using hip calls
        ! that we will attach pointers to
        do i = 1, 3
            pool_dims(i) = idwbuff(i)%end - idwbuff(i)%beg + 1
            pool_starts(i) = idwbuff(i)%beg
        end do
        pool_dims(4) = sys_size
        pool_starts(4) = 1
#ifdef MFC_MIXED_PRECISION
        pool_size = 1_8*(idwbuff(1)%end - idwbuff(1)%beg + 1)*(idwbuff(2)%end - idwbuff(2)%beg + 1)*(idwbuff(3)%end - idwbuff(3)%beg + 1)*sys_size
        call hipCheck(hipMalloc_(cptr_device, pool_size*2_8))
        call c_f_pointer(cptr_device, q_cons_ts_pool_device, shape=pool_dims)
        q_cons_ts_pool_device(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:) => q_cons_ts_pool_device

        call hipCheck(hipMallocManaged_(cptr_host, pool_size*2_8, hipMemAttachGlobal))
        call c_f_pointer(cptr_host, q_cons_ts_pool_host, shape=pool_dims)
        q_cons_ts_pool_host(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:) => q_cons_ts_pool_host
#else
        ! Doing hipMalloc then mapping should be most performant
        call hipCheck(hipMalloc(q_cons_ts_pool_device, dims8=pool_dims, lbounds8=pool_starts))
        ! Without this map CCE will still create a device copy, because it's silly like that
#if defined(MFC_OpenACC)
        call acc_map_data(q_cons_ts_pool_device, c_loc(q_cons_ts_pool_device), c_sizeof(q_cons_ts_pool_device))
#endif
        ! CCE see it can access this and will leave it on the host. It will stay on the host so long as HSA_XNACK=1
        ! NOTE: WE CANNOT DO ATOMICS INTO THIS MEMORY. We have to change a property to use atomics here
        ! Otherwise leaving this as fine-grained will actually help performance since it can't be cached in GPU L2
        if (num_ts == 2) then
            call hipCheck(hipMallocManaged(q_cons_ts_pool_host, dims8=pool_dims, lbounds8=pool_starts, flags=hipMemAttachGlobal))
#if defined(MFC_OpenMP)
            call hipCheck(hipMemAdvise(c_loc(q_cons_ts_pool_host), c_sizeof(q_cons_ts_pool_host), hipMemAdviseSetPreferredLocation, -1))
#endif
        end if
#endif

        do j = 1, sys_size
            ! q_cons_ts(1) lives on the device
            q_cons_ts(1)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                                  idwbuff(2)%beg:idwbuff(2)%end, &
                                  idwbuff(3)%beg:idwbuff(3)%end) => q_cons_ts_pool_device(:, :, :, j)
            if (num_ts == 2) then
                ! q_cons_ts(2) lives on the host
                q_cons_ts(2)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                                      idwbuff(2)%beg:idwbuff(2)%end, &
                                      idwbuff(3)%beg:idwbuff(3)%end) => q_cons_ts_pool_host(:, :, :, j)
            end if
        end do

        do i = 1, num_ts
            @:ACC_SETUP_VFs(q_cons_ts(i))
            do j = 1, sys_size
                $:GPU_UPDATE(device='[q_cons_ts(i)%vf(j)]')
            end do
        end do
#else
!> @endcond
        do i = 1, num_ts
            do j = 1, sys_size
                @:ALLOCATE(q_cons_ts(i)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
            end do
            @:ACC_SETUP_VFs(q_cons_ts(i))
        end do
!> @cond
#endif
!> @endcond

        ! Allocating the cell-average primitive ts variables
        if (probe_wrt) then
            @:ALLOCATE(q_prim_ts1(1:num_probe_ts))

            do i = 1, num_probe_ts
                @:ALLOCATE(q_prim_ts1(i)%vf(1:sys_size))
            end do

            do i = 1, num_probe_ts
                do j = 1, sys_size
                    @:ALLOCATE(q_prim_ts1(i)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                end do
                @:ACC_SETUP_VFs(q_prim_ts1(i))
            end do

            @:ALLOCATE(q_prim_ts2(1:num_probe_ts))

            do i = 1, num_probe_ts
                @:ALLOCATE(q_prim_ts2(i)%vf(1:sys_size))
            end do

            do i = 1, num_probe_ts
                do j = 1, sys_size
                    @:ALLOCATE(q_prim_ts2(i)%vf(j)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                end do
                @:ACC_SETUP_VFs(q_prim_ts2(i))
            end do
        end if

        ! Allocating the cell-average primitive variables
        @:ALLOCATE(q_prim_vf(1:sys_size))

        if (.not. igr) then
            do i = 1, adv_idx%end
                @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(q_prim_vf(i))
            end do

            if (bubbles_euler) then
                do i = bub_idx%beg, bub_idx%end
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do
                if (adv_n) then
                    @:ALLOCATE(q_prim_vf(n_idx)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(n_idx))
                end if
            end if

            if (mhd) then
                do i = B_idx%beg, B_idx%end
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do
            end if

            if (elasticity) then
                do i = stress_idx%beg, stress_idx%end
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do
            end if

            if (hyperelasticity) then
                do i = xibeg, xiend + 1
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do
            end if

            if (cont_damage) then
                @:ALLOCATE(q_prim_vf(damage_idx)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(q_prim_vf(damage_idx))
            end if

            if (hyper_cleaning) then
                @:ALLOCATE(q_prim_vf(psi_idx)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(q_prim_vf(psi_idx))
            end if

            if (model_eqns == 3) then
                do i = internalEnergies_idx%beg, internalEnergies_idx%end
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do
            end if

            if (surface_tension) then
                @:ALLOCATE(q_prim_vf(c_idx)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(q_prim_vf(c_idx))
            end if

            if (chemistry) then
                do i = chemxb, chemxe
                    @:ALLOCATE(q_prim_vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, &
                        idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(q_prim_vf(i))
                end do

                @:ALLOCATE(q_T_sf%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(q_T_sf)
            end if
        end if

        @:ALLOCATE(pb_ts(1:2))
        !Initialize bubble variables pb and mv at all quadrature nodes for all R0 bins
        if (qbmm .and. (.not. polytropic)) then
            @:ALLOCATE(pb_ts(1)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(pb_ts(1))

            @:ALLOCATE(pb_ts(2)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(pb_ts(2))

            @:ALLOCATE(rhs_pb(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))
        else if (qbmm .and. polytropic) then
            @:ALLOCATE(pb_ts(1)%sf(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(pb_ts(1))

            @:ALLOCATE(pb_ts(2)%sf(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(pb_ts(2))

            @:ALLOCATE(rhs_pb(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
        else
            @:ALLOCATE(pb_ts(1)%sf(0,0,0,0,0))
            @:ACC_SETUP_SFs(pb_ts(1))

            @:ALLOCATE(pb_ts(2)%sf(0,0,0,0,0))
            @:ACC_SETUP_SFs(pb_ts(2))

            @:ALLOCATE(rhs_pb(0,0,0,0,0))
        end if

        @:ALLOCATE(mv_ts(1:2))

        if (qbmm .and. (.not. polytropic)) then
            @:ALLOCATE(mv_ts(1)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(mv_ts(1))

            @:ALLOCATE(mv_ts(2)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(mv_ts(2))

            @:ALLOCATE(rhs_mv(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, &
                idwbuff(3)%beg:idwbuff(3)%end, 1:nnode, 1:nb))

        else if (qbmm .and. polytropic) then
            @:ALLOCATE(mv_ts(1)%sf(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(mv_ts(1))

            @:ALLOCATE(mv_ts(2)%sf(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
            @:ACC_SETUP_SFs(mv_ts(2))

            @:ALLOCATE(rhs_mv(idwbuff(1)%beg:idwbuff(1)%beg + 1, &
                idwbuff(2)%beg:idwbuff(2)%beg + 1, &
                idwbuff(3)%beg:idwbuff(3)%beg + 1, 1:nnode, 1:nb))
        else
            @:ALLOCATE(mv_ts(1)%sf(0,0,0,0,0))
            @:ACC_SETUP_SFs(mv_ts(1))

            @:ALLOCATE(mv_ts(2)%sf(0,0,0,0,0))
            @:ACC_SETUP_SFs(mv_ts(2))

            @:ALLOCATE(rhs_mv(0,0,0,0,0))
        end if

        ! Allocating the cell-average RHS variables
        @:ALLOCATE(rhs_vf(1:sys_size))
        @:PREFER_GPU(rhs_vf)

        if (igr) then
            do i = 1, sys_size
                @:ALLOCATE(rhs_vf(i)%sf(-1:m+1,-1:n+1,-1:p+1))
                @:ACC_SETUP_SFs(rhs_vf(i))
                @:PREFER_GPU(rhs_vf(i)%sf)
            end do
        else
            do i = 1, sys_size
                @:ALLOCATE(rhs_vf(i)%sf(0:m, 0:n, 0:p))
                @:ACC_SETUP_SFs(rhs_vf(i))
            end do
        end if

        ! Opening and writing the header of the run-time information file
        if (proc_rank == 0 .and. run_time_info) then
            call s_open_run_time_information_file()
        end if

        if (cfl_dt) then
            @:ALLOCATE(max_dt(0:m, 0:n, 0:p))
        end if

        if (relax) then
            @:ALLOCATE(m_dot_evap%sf(0:m, 0:n, 0:p))
            @:ACC_SETUP_SFs(m_dot_evap)
            call s_reset_m_dot_evap()
        end if

        ! Allocating arrays to store the bc types
        @:ALLOCATE(bc_type(1:num_dims,1:2))

        @:ALLOCATE(bc_type(1,1)%sf(0:0,0:n,0:p))
        @:ALLOCATE(bc_type(1,2)%sf(0:0,0:n,0:p))
        #:if not MFC_CASE_OPTIMIZATION or num_dims > 1
            if (n > 0) then
                @:ALLOCATE(bc_type(2,1)%sf(-buff_size:m+buff_size,0:0,0:p))
                @:ALLOCATE(bc_type(2,2)%sf(-buff_size:m+buff_size,0:0,0:p))
                #:if not MFC_CASE_OPTIMIZATION or num_dims > 2
                    if (p > 0) then
                        @:ALLOCATE(bc_type(3,1)%sf(-buff_size:m+buff_size,-buff_size:n+buff_size,0:0))
                        @:ALLOCATE(bc_type(3,2)%sf(-buff_size:m+buff_size,-buff_size:n+buff_size,0:0))
                    end if
                #:endif
            end if
        #:endif

        do i = 1, num_dims
            do j = 1, 2
                @:ACC_SETUP_SFs(bc_type(i,j))
            end do
        end do

        if (any(time_stepper == (/1, 2, 3/))) then
            ! temporary array index for TVD RK
            if (time_stepper == 1) then
                stor = 1
            else
                stor = 2
            end if

            ! TVD RK coefficients
            @:ALLOCATE (rk_coef(time_stepper, 4))
            if (time_stepper == 1) then
                rk_coef(1, :) = (/1._wp, 0._wp, 1._wp, 1._wp/)
            else if (time_stepper == 2) then
                rk_coef(1, :) = (/1._wp, 0._wp, 1._wp, 1._wp/)
                rk_coef(2, :) = (/1._wp, 1._wp, 1._wp, 2._wp/)
            else if (time_stepper == 3) then
                rk_coef(1, :) = (/1._wp, 0._wp, 1._wp, 1._wp/)
                rk_coef(2, :) = (/1._wp, 3._wp, 1._wp, 4._wp/)
                rk_coef(3, :) = (/2._wp, 1._wp, 2._wp, 3._wp/)
            end if
            $:GPU_UPDATE(device='[rk_coef, stor]')
        end if

    end subroutine s_initialize_time_steppers_module

    !> @brief Advances the solution one full step using a TVD Runge-Kutta time integrator.
    impure subroutine s_tvd_rk(t_step, time_avg, nstage)
#ifdef _CRAYFTN
        !DIR$ OPTIMIZE (-haggress)
#endif
        integer, intent(in) :: t_step
        real(wp), intent(inout) :: time_avg
        integer, intent(in) :: nstage

        integer :: i, j, k, l, q, s !< Generic loop iterator
        real(wp) :: start, finish
        integer :: dest

        call cpu_time(start)
        call nvtxStartRange("TIMESTEP")
        call s_zhang_evap_hang_trace(t_step, 0, "TVD_RK_BEGIN")

        ! Adaptive dt: initial stage
        if (adap_dt) call s_adaptive_dt_bubble(1)

        do s = 1, nstage
            call s_zhang_evap_hang_trace(t_step, s, "RK_STAGE_BEGIN")
            call s_zhang_evap_hang_trace(t_step, s, "RHS_CALL_BEGIN")
            call s_compute_rhs(q_cons_ts(1)%vf, q_T_sf, q_prim_vf, bc_type, rhs_vf, pb_ts(1)%sf, rhs_pb, mv_ts(1)%sf, rhs_mv, t_step, time_avg, s)
            call s_zhang_evap_hang_trace(t_step, s, "RHS_CALL_END")

            if (s == 1) then
                if (run_time_info) then
                    if (igr .or. dummy) then
                        call s_zhang_evap_hang_trace(t_step, s, "RUN_TIME_INFO_WRITE_BEGIN_CONS")
                        call s_write_run_time_information(q_cons_ts(1)%vf, t_step)
                        call s_zhang_evap_hang_trace(t_step, s, "RUN_TIME_INFO_WRITE_END_CONS")
                    end if
                    if (.not. igr .or. dummy) then
                        call s_zhang_evap_hang_trace(t_step, s, "RUN_TIME_INFO_WRITE_BEGIN_PRIM")
                        call s_write_run_time_information(q_prim_vf, t_step)
                        call s_zhang_evap_hang_trace(t_step, s, "RUN_TIME_INFO_WRITE_END_PRIM")
                    end if
                end if

                if (probe_wrt) then
                    call s_time_step_cycling(t_step)
                    call s_compute_derived_variables(t_step, q_cons_ts(1)%vf, q_prim_ts1, q_prim_ts2)
                end if

                if (cfl_dt) then
                    if (mytime >= t_stop) return
                else
                    if (t_step == t_step_stop) return
                end if
            end if

            if (bubbles_lagrange .and. .not. adap_dt) call s_update_lagrange_tdv_rk(stage=s)
            call s_zhang_evap_hang_trace(t_step, s, "RK_CONS_UPDATE_BEGIN")
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            if (s == 1 .and. nstage > 1) then
                                q_cons_ts(stor)%vf(i)%sf(j, k, l) = &
                                    q_cons_ts(1)%vf(i)%sf(j, k, l)
                            end if
                            if (igr) then
                                q_cons_ts(1)%vf(i)%sf(j, k, l) = &
                                    (rk_coef(s, 1)*q_cons_ts(1)%vf(i)%sf(j, k, l) &
                                     + rk_coef(s, 2)*q_cons_ts(stor)%vf(i)%sf(j, k, l) &
                                     + rk_coef(s, 3)*rhs_vf(i)%sf(j, k, l))/rk_coef(s, 4)
                            else
                                q_cons_ts(1)%vf(i)%sf(j, k, l) = &
                                    (rk_coef(s, 1)*q_cons_ts(1)%vf(i)%sf(j, k, l) &
                                     + rk_coef(s, 2)*q_cons_ts(stor)%vf(i)%sf(j, k, l) &
                                     + rk_coef(s, 3)*dt*rhs_vf(i)%sf(j, k, l))/rk_coef(s, 4)
                            end if
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
            call s_ybc_edge_cons_debug_report(q_cons_ts(1)%vf, "RK_CONS_UPDATE_AFTER_RAW", t_step, s)
            call s_zhang_evap_hang_trace(t_step, s, "RK_CONS_UPDATE_END")

            !Evolve pb and mv for non-polytropic qbmm
            if (qbmm .and. (.not. polytropic)) then
                $:GPU_PARALLEL_LOOP(collapse=5)
                do i = 1, nb
                    do l = 0, p
                        do k = 0, n
                            do j = 0, m
                                do q = 1, nnode
                                    if (s == 1 .and. nstage > 1) then
                                        pb_ts(stor)%sf(j, k, l, q, i) = &
                                            pb_ts(1)%sf(j, k, l, q, i)
                                        mv_ts(stor)%sf(j, k, l, q, i) = &
                                            mv_ts(1)%sf(j, k, l, q, i)
                                    end if
                                    pb_ts(1)%sf(j, k, l, q, i) = &
                                        (rk_coef(s, 1)*pb_ts(1)%sf(j, k, l, q, i) &
                                         + rk_coef(s, 2)*pb_ts(stor)%sf(j, k, l, q, i) &
                                         + rk_coef(s, 3)*dt*rhs_pb(j, k, l, q, i))/rk_coef(s, 4)
                                    mv_ts(1)%sf(j, k, l, q, i) = &
                                        (rk_coef(s, 1)*mv_ts(1)%sf(j, k, l, q, i) &
                                         + rk_coef(s, 2)*mv_ts(stor)%sf(j, k, l, q, i) &
                                         + rk_coef(s, 3)*dt*rhs_mv(j, k, l, q, i))/rk_coef(s, 4)
                                end do
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if (bodyForces) call s_apply_bodyforces(q_cons_ts(1)%vf, q_prim_vf, rhs_vf, rk_coef(s, 3)*dt/rk_coef(s, 4))

            if (grid_geometry == 3) call s_apply_fourier_filter(q_cons_ts(1)%vf)

            if (model_eqns == 3 .and. (.not. relax)) then
                call s_zhang_evap_hang_trace(t_step, s, "PRESSURE_RELAX_BEGIN")
                call s_pressure_relaxation_procedure(q_cons_ts(1)%vf)
                call s_zhang_evap_hang_trace(t_step, s, "PRESSURE_RELAX_END")
            end if

            if (adv_n) call s_comp_alpha_from_n(q_cons_ts(1)%vf)

            if (ib) then
                ! check if any IBMS are moving, and if so, update the markers, ghost points, levelsets, and levelset norms
                if (moving_immersed_boundary_flag) then
                    call s_propagate_immersed_boundaries(s)
                end if

                ! update the ghost fluid properties point values based on IB state
                if (qbmm .and. .not. polytropic) then
                    call s_ibm_correct_state(q_cons_ts(1)%vf, q_prim_vf, pb_ts(1)%sf, mv_ts(1)%sf)
                else
                    call s_ibm_correct_state(q_cons_ts(1)%vf, q_prim_vf)
                end if
            end if

            call s_zhang_evap_hang_trace(t_step, s, "RK_STAGE_END")
        end do

        ! Adaptive dt: final stage
        if (adap_dt) call s_adaptive_dt_bubble(3)

        call s_zhang_evap_hang_trace(t_step, 0, "TVD_RK_END")
        call nvtxEndRange
        call cpu_time(finish)

        wall_time = abs(finish - start)

        if (t_step >= 2) then
            wall_time_avg = (wall_time + (t_step - 2)*wall_time_avg)/(t_step - 1)
        else
            wall_time_avg = 0._wp
        end if

    end subroutine s_tvd_rk

    subroutine s_reset_m_dot_evap()
        integer :: j, k, l

        if (.not. relax) return

        $:GPU_PARALLEL_LOOP(collapse=3, private='[j,k,l]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    m_dot_evap%sf(j, k, l) = 0._wp
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()
    end subroutine s_reset_m_dot_evap

    subroutine s_apply_evap_to_fuel_species(q_cons_vf, ldt)
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(in) :: ldt
        integer :: fuel_species_eqn
        integer :: i, j, k, l
        real(wp) :: alpha_liq, gas_alpha, gas_mass

        if (.not. relax) return
        if (.not. chemistry) return
        if (.not. evap_species_source) return
        if (fuel_species_id < 1 .or. fuel_species_id > (chemxe - chemxb + 1)) return
        if (evap_liquid_fluid_id < 1 .or. evap_liquid_fluid_id > num_fluids) return

        fuel_species_eqn = chemxb + fuel_species_id - 1

        $:GPU_PARALLEL_LOOP(collapse=3, private='[i,j,k,l,alpha_liq,gas_alpha,gas_mass]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    alpha_liq = q_cons_vf(advxb + evap_liquid_fluid_id - 1)%sf(j, k, l)
                    gas_alpha = 0._wp
                    gas_mass = 0._wp
                    do i = 1, num_fluids
                        if (i /= evap_liquid_fluid_id) then
                            gas_alpha = gas_alpha + q_cons_vf(advxb + i - 1)%sf(j, k, l)
                            gas_mass = gas_mass + q_cons_vf(contxb + i - 1)%sf(j, k, l)
                        end if
                    end do
                    if (alpha_liq > evap_alpha_thresh .and. alpha_liq < evap_species_liq_max .and. gas_alpha > evap_species_alpha_min .and. &
                        gas_mass > evap_species_mass_min) then
                        q_cons_vf(fuel_species_eqn)%sf(j, k, l) = q_cons_vf(fuel_species_eqn)%sf(j, k, l) &
                                                                   + ldt*max(0._wp, m_dot_evap%sf(j, k, l))
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()
    end subroutine s_apply_evap_to_fuel_species

    !> Bubble source part in Strang operator splitting scheme
        !! @param stage Current time-stage
    impure subroutine s_adaptive_dt_bubble(stage)

        integer, intent(in) :: stage

        type(vector_field) :: gm_alpha_qp

        call s_convert_conservative_to_primitive_variables( &
            q_cons_ts(1)%vf, &
            q_T_sf, &
            q_prim_vf, &
            idwint)

        if (bubbles_euler) then

            call s_compute_bubble_EE_source(q_cons_ts(1)%vf, q_prim_vf, rhs_vf, divu)
            call s_comp_alpha_from_n(q_cons_ts(1)%vf)

        elseif (bubbles_lagrange) then

            call s_populate_variables_buffers(bc_type, q_prim_vf, pb_ts(1)%sf, mv_ts(1)%sf)
            call s_compute_bubble_EL_dynamics(q_prim_vf, stage)
            call s_transfer_data_to_tmp()
            call s_smear_voidfraction()
            if (stage == 3) then
                if (lag_params%write_bubbles_stats) call s_calculate_lag_bubble_stats()
                if (lag_params%write_bubbles) then
                    $:GPU_UPDATE(host='[gas_p,gas_mv,intfc_rad,intfc_vel]')
                    call s_write_lag_particles(mytime)
                end if
                call s_write_void_evol(mytime)
            end if

        end if

    end subroutine s_adaptive_dt_bubble

    pure function s_dt_debug_lowercase(value) result(lower_value)
        character(len=*), intent(in) :: value
        character(len=len(value)) :: lower_value
        integer :: i, c

        lower_value = value
        do i = 1, len(value)
            c = iachar(value(i:i))
            if (c >= iachar('A') .and. c <= iachar('Z')) then
                lower_value(i:i) = achar(c + iachar('a') - iachar('A'))
            end if
        end do
    end function s_dt_debug_lowercase

    integer function s_dt_debug_species_index(name) result(species_id)
        character(len=*), intent(in) :: name
        integer :: sp
        character(len=:), allocatable :: target_name, candidate_name

        species_id = 0
        target_name = trim(adjustl(s_dt_debug_lowercase(name)))
        do sp = 1, num_species
            candidate_name = trim(adjustl(s_dt_debug_lowercase(species_names(sp))))
            if (candidate_name == target_name) then
                species_id = sp
                return
            end if
        end do
    end function s_dt_debug_species_index

    subroutine s_dt_collapse_debug_report(t_step, time_value)
        integer, intent(in) :: t_step
        real(wp), intent(in) :: time_value

        integer, parameter :: dt_debug_species_count = 7
        character(len=16), dimension(dt_debug_species_count), parameter :: dt_debug_species_names = &
            [character(len=16) :: "NC12H26", "O2", "OH", "HO2", "H2O2", "CO2", "H2O"]

        character(len=16) :: env_value
        character(len=32) :: region_label
        integer :: env_status
        integer :: j, k, l, id, species_pos, species_id, species_eqn
        integer :: best_j, best_k, best_l
        integer :: global_j, global_k, global_l
        integer :: gas_idx, gas_fluid_id
        logical :: found_finite
        real(wp) :: local_min_dt, cfl_denom, x_loc, y_loc
        real(wp) :: rho, vel_sum, pres, gamma, pi_inf, qv, c, H
        real(wp) :: vel_x, vel_y, vel_z
        real(wp) :: alpha_liq, alpha_vap, alpha_air, gas_alpha, gas_mass
        real(wp), dimension(dt_debug_species_count) :: species_values
        real(wp), dimension(2) :: Re
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: vel
            real(wp), dimension(3) :: alpha
        #:else
            real(wp), dimension(num_vels) :: vel
            real(wp), dimension(num_fluids) :: alpha
        #:endif

        call get_environment_variable("TEMP_DT_COLLAPSE_DEBUG", env_value, status=env_status)
        if (env_status /= 0 .or. trim(env_value) /= "1") return

        $:GPU_UPDATE(host='[max_dt]')
        do id = 1, sys_size
            $:GPU_UPDATE(host='[q_cons_ts(1)%vf(id)%sf,q_prim_vf(id)%sf]')
        end do

        found_finite = .false.
        local_min_dt = huge(1._wp)
        best_j = 0
        best_k = 0
        best_l = 0
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    if (max_dt(j, k, l) == max_dt(j, k, l)) then
                        if (.not. found_finite .or. max_dt(j, k, l) < local_min_dt) then
                            found_finite = .true.
                            local_min_dt = max_dt(j, k, l)
                            best_j = j
                            best_k = k
                            best_l = l
                        end if
                    end if
                end do
            end do
        end do

        if (.not. found_finite) then
            local_min_dt = max_dt(0, 0, 0)
        end if

        if (local_min_dt > tiny(local_min_dt) .and. local_min_dt == local_min_dt) then
            cfl_denom = cfl_target/local_min_dt
        else
            cfl_denom = huge(1._wp)
        end if

        if (igr) then
            call s_compute_enthalpy(q_cons_ts(1)%vf, pres, rho, gamma, pi_inf, Re, H, alpha, vel, vel_sum, qv, &
                                    best_j, best_k, best_l)
        else
            call s_compute_enthalpy(q_prim_vf, pres, rho, gamma, pi_inf, Re, H, alpha, vel, vel_sum, qv, &
                                    best_j, best_k, best_l)
        end if
        call s_compute_speed_of_sound(pres, rho, gamma, pi_inf, H, alpha, vel_sum, 0._wp, c, qv)

        vel_x = 0._wp
        vel_y = 0._wp
        vel_z = 0._wp
        if (num_vels >= 1) vel_x = vel(1)
        if (num_vels >= 2) vel_y = vel(2)
        if (num_vels >= 3) vel_z = vel(3)

        alpha_liq = 0._wp
        alpha_vap = 0._wp
        alpha_air = 0._wp
        if (num_fluids >= 1) alpha_liq = q_cons_ts(1)%vf(advxb)%sf(best_j, best_k, best_l)
        if (num_fluids >= 2) alpha_vap = q_cons_ts(1)%vf(advxb + 1)%sf(best_j, best_k, best_l)
        if (num_fluids >= 3) alpha_air = q_cons_ts(1)%vf(advxb + 2)%sf(best_j, best_k, best_l)

        gas_alpha = 0._wp
        gas_mass = 0._wp
        do gas_idx = 1, chem_gas_num_fluids
            gas_fluid_id = chem_gas_fluid_ids(gas_idx)
            if (gas_fluid_id >= 1 .and. gas_fluid_id <= num_fluids) then
                gas_alpha = gas_alpha + q_cons_ts(1)%vf(advxb + gas_fluid_id - 1)%sf(best_j, best_k, best_l)
                gas_mass = gas_mass + q_cons_ts(1)%vf(contxb + gas_fluid_id - 1)%sf(best_j, best_k, best_l)
            end if
        end do

        species_values = 0._wp
        if (chemistry) then
            do species_pos = 1, dt_debug_species_count
                species_id = s_dt_debug_species_index(dt_debug_species_names(species_pos))
                if (species_id > 0) then
                    species_eqn = chemxb + species_id - 1
                    if (species_eqn >= chemxb .and. species_eqn <= chemxe) then
                        species_values(species_pos) = q_cons_ts(1)%vf(species_eqn)%sf(best_j, best_k, best_l)
                    end if
                end if
            end do
        end if

        global_j = best_j
        global_k = best_k
        global_l = best_l
        if (allocated(start_idx)) then
            if (size(start_idx) >= 1) global_j = start_idx(1) + best_j
            if (size(start_idx) >= 2) global_k = start_idx(2) + best_k
            if (size(start_idx) >= 3) global_l = start_idx(3) + best_l
        end if

        x_loc = 0._wp
        y_loc = 0._wp
        if (allocated(x_cc)) x_loc = x_cc(best_j)
        if (allocated(y_cc)) y_loc = y_cc(best_k)

        if (alpha_liq > 0.99_wp) then
            region_label = "liquid_dominated"
        elseif (alpha_liq > 0.01_wp .and. alpha_liq < 0.99_wp) then
            region_label = "interface"
        elseif (gas_alpha > 0.9_wp .and. alpha_liq < 0.1_wp) then
            region_label = "gas_dominant"
        else
            region_label = "mixed_or_other"
        end if

        print '(" TEMP_DT_COLLAPSE_DEBUG rank=", I6, " t_step=", I10, " time=", ES16.8, " global_dt=", ES16.8, " local_min_dt=", ES16.8)', &
            proc_rank, t_step, time_value, dt, local_min_dt
        print '(" TEMP_DT_COLLAPSE_DEBUG_CELL rank=", I6, " local_ijk=", 3(I8,1X), " global_ijk=", 3(I8,1X), " x=", ES16.8, " y=", ES16.8)', &
            proc_rank, best_j, best_k, best_l, global_j, global_k, global_l, x_loc, y_loc
        print '(" TEMP_DT_COLLAPSE_DEBUG_CFL rank=", I6, " cfl_denom=", ES16.8, " u=", ES16.8, " v=", ES16.8, " w=", ES16.8, " c=", ES16.8)', &
            proc_rank, cfl_denom, vel_x, vel_y, vel_z, c
        print '(" TEMP_DT_COLLAPSE_DEBUG_STATE rank=", I6, " pressure=", ES16.8, " rho=", ES16.8, " alpha_liq=", ES16.8, " alpha_vap=", ES16.8, " alpha_air=", ES16.8, " gas_alpha=", ES16.8, " gas_mass=", ES16.8, " region=", A)', &
            proc_rank, pres, rho, alpha_liq, alpha_vap, alpha_air, gas_alpha, gas_mass, trim(region_label)
        print '(" TEMP_DT_COLLAPSE_DEBUG_SPECIES rank=", I6, " NC12H26=", ES16.8, " O2=", ES16.8, " OH=", ES16.8, " HO2=", ES16.8, " H2O2=", ES16.8, " CO2=", ES16.8, " H2O=", ES16.8)', &
            proc_rank, species_values(1), species_values(2), species_values(3), species_values(4), &
            species_values(5), species_values(6), species_values(7)
        call flush(output_unit)
    end subroutine s_dt_collapse_debug_report

    !> @brief Computes the global time step size from CFL stability constraints across all cells.
    impure subroutine s_compute_dt(t_step_debug)
        integer, intent(in), optional :: t_step_debug

        real(wp) :: rho        !< Cell-avg. density
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: vel        !< Cell-avg. velocity
            real(wp), dimension(3) :: alpha      !< Cell-avg. volume fraction
        #:else
            real(wp), dimension(num_vels) :: vel        !< Cell-avg. velocity
            real(wp), dimension(num_fluids) :: alpha      !< Cell-avg. volume fraction
        #:endif
        real(wp) :: vel_sum    !< Cell-avg. velocity sum
        real(wp) :: pres       !< Cell-avg. pressure
        real(wp) :: gamma      !< Cell-avg. sp. heat ratio
        real(wp) :: pi_inf     !< Cell-avg. liquid stiffness function
        real(wp) :: qv         !< Cell-avg. fluid reference energy
        real(wp) :: c          !< Cell-avg. sound speed
        real(wp) :: H          !< Cell-avg. enthalpy
        real(wp), dimension(2) :: Re         !< Cell-avg. Reynolds numbers
        type(vector_field) :: gm_alpha_qp

        real(wp) :: dt_local
        integer :: j, k, l !< Generic loop iterators
        integer :: ybc_debug_t_step

        ybc_debug_t_step = -1
        if (present(t_step_debug)) ybc_debug_t_step = t_step_debug
        call s_ybc_edge_cons_debug_report(q_cons_ts(1)%vf, "COMPUTE_DT_BEFORE_CONS_TO_PRIM", ybc_debug_t_step, 0)

        if (.not. igr .or. dummy) then
            call s_convert_conservative_to_primitive_variables( &
                q_cons_ts(1)%vf, &
                q_T_sf, &
                q_prim_vf, &
                idwint)
        end if

        $:GPU_PARALLEL_LOOP(collapse=3, private='[vel, alpha, Re, rho, vel_sum, pres, gamma, pi_inf, c, H, qv]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    if (igr) then
                        call s_compute_enthalpy(q_cons_ts(1)%vf, pres, rho, gamma, pi_inf, Re, H, alpha, vel, vel_sum, qv, j, k, l)
                    else
                        call s_compute_enthalpy(q_prim_vf, pres, rho, gamma, pi_inf, Re, H, alpha, vel, vel_sum, qv, j, k, l)
                    end if

                    ! Compute mixture sound speed
                    call s_compute_speed_of_sound(pres, rho, gamma, pi_inf, H, alpha, vel_sum, 0._wp, c, qv)

                    call s_compute_dt_from_cfl(vel, c, max_dt, rho, Re, j, k, l)
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        #:call GPU_PARALLEL(copyout='[dt_local]', copyin='[max_dt]')
            dt_local = minval(max_dt)
        #:endcall GPU_PARALLEL

        if (num_procs == 1) then
            dt = dt_local
        else
            call s_mpi_allreduce_min(dt_local, dt)
        end if

        $:GPU_UPDATE(device='[dt]')

    end subroutine s_compute_dt

    !> This subroutine applies the body forces source term at each
        !! Runge-Kutta stage
        !! @param q_cons_vf Conservative variables
        !! @param q_prim_vf_in Primitive variables
        !! @param rhs_vf_in Right-hand side variables
    subroutine s_apply_bodyforces(q_cons_vf, q_prim_vf_in, rhs_vf_in, ldt)

        type(scalar_field), dimension(1:sys_size), intent(inout) :: q_cons_vf
        type(scalar_field), dimension(1:sys_size), intent(in) :: q_prim_vf_in
        type(scalar_field), dimension(1:sys_size), intent(inout) :: rhs_vf_in

        real(wp), intent(in) :: ldt !< local dt

        integer :: i, j, k, l

        call nvtxStartRange("RHS-BODYFORCES")
        call s_compute_body_forces_rhs(q_prim_vf_in, q_cons_vf, rhs_vf_in)

        $:GPU_PARALLEL_LOOP(collapse=4)
        do i = momxb, E_idx
            do l = 0, p
                do k = 0, n
                    do j = 0, m
                        q_cons_vf(i)%sf(j, k, l) = q_cons_vf(i)%sf(j, k, l) + &
                                                   ldt*rhs_vf_in(i)%sf(j, k, l)
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call nvtxEndRange

    end subroutine s_apply_bodyforces

    !> @brief Updates immersed boundary positions and velocities at the current Runge-Kutta stage.
    subroutine s_propagate_immersed_boundaries(s)

        integer, intent(in) :: s
        integer :: i
        logical :: forces_computed

        forces_computed = .false.

        do i = 1, num_ibs
            if (s == 1) then
                patch_ib(i)%step_vel = patch_ib(i)%vel
                patch_ib(i)%step_angular_vel = patch_ib(i)%angular_vel
                patch_ib(i)%step_angles = patch_ib(i)%angles
                patch_ib(i)%step_x_centroid = patch_ib(i)%x_centroid
                patch_ib(i)%step_y_centroid = patch_ib(i)%y_centroid
                patch_ib(i)%step_z_centroid = patch_ib(i)%z_centroid
            end if

            if (patch_ib(i)%moving_ibm > 0) then
                patch_ib(i)%vel = (rk_coef(s, 1)*patch_ib(i)%step_vel + rk_coef(s, 2)*patch_ib(i)%vel)/rk_coef(s, 4)
                patch_ib(i)%angular_vel = (rk_coef(s, 1)*patch_ib(i)%step_angular_vel + rk_coef(s, 2)*patch_ib(i)%angular_vel)/rk_coef(s, 4)

                if (patch_ib(i)%moving_ibm == 1) then
                    ! plug in analytic velocities for 1-way coupling, if it exists
                    @:mib_analytical()
                else if (patch_ib(i)%moving_ibm == 2) then ! if we are using two-way coupling, apply force and torque
                    ! compute the force and torque on the IB from the fluid
                    if (.not. forces_computed) then
                        call s_compute_ib_forces(q_prim_vf, fluid_pp)
                        forces_computed = .true.
                    end if

                    ! update the velocity from the force value
                    patch_ib(i)%vel = patch_ib(i)%vel + rk_coef(s, 3)*dt*(patch_ib(i)%force/patch_ib(i)%mass)/rk_coef(s, 4)

                    ! update the angular velocity with the torque value
                    patch_ib(i)%angular_vel = (patch_ib(i)%angular_vel*patch_ib(i)%moment) + (rk_coef(s, 3)*dt*patch_ib(i)%torque/rk_coef(s, 4)) ! add the torque to the angular momentum
                    call s_compute_moment_of_inertia(i, patch_ib(i)%angular_vel) ! update the moment of inertia to be based on the direction of the angular momentum
                    patch_ib(i)%angular_vel = patch_ib(i)%angular_vel/patch_ib(i)%moment ! convert back to angular velocity with the new moment of inertia
                end if

                ! Update the angle of the IB
                patch_ib(i)%angles = (rk_coef(s, 1)*patch_ib(i)%step_angles + rk_coef(s, 2)*patch_ib(i)%angles + rk_coef(s, 3)*patch_ib(i)%angular_vel*dt)/rk_coef(s, 4)

                ! Update the position of the IB
                patch_ib(i)%x_centroid = (rk_coef(s, 1)*patch_ib(i)%step_x_centroid + rk_coef(s, 2)*patch_ib(i)%x_centroid + rk_coef(s, 3)*patch_ib(i)%vel(1)*dt)/rk_coef(s, 4)
                patch_ib(i)%y_centroid = (rk_coef(s, 1)*patch_ib(i)%step_y_centroid + rk_coef(s, 2)*patch_ib(i)%y_centroid + rk_coef(s, 3)*patch_ib(i)%vel(2)*dt)/rk_coef(s, 4)
                patch_ib(i)%z_centroid = (rk_coef(s, 1)*patch_ib(i)%step_z_centroid + rk_coef(s, 2)*patch_ib(i)%z_centroid + rk_coef(s, 3)*patch_ib(i)%vel(3)*dt)/rk_coef(s, 4)
            end if
        end do

        call s_update_mib(num_ibs)

    end subroutine s_propagate_immersed_boundaries

    !> This subroutine saves the temporary q_prim_vf vector
        !!      into the q_prim_ts vector that is then used in p_main
        !! @param t_step current time-step
    subroutine s_time_step_cycling(t_step)

        integer, intent(in) :: t_step

        integer :: i, j, k, l !< Generic loop iterator

        if (t_step == t_step_start) then
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            q_prim_ts2(2)%vf(i)%sf(j, k, l) = q_prim_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        elseif (t_step == t_step_start + 1) then
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            q_prim_ts2(1)%vf(i)%sf(j, k, l) = q_prim_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        elseif (t_step == t_step_start + 2) then
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            q_prim_ts1(2)%vf(i)%sf(j, k, l) = q_prim_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        elseif (t_step == t_step_start + 3) then
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            q_prim_ts1(1)%vf(i)%sf(j, k, l) = q_prim_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        else ! All other timesteps
            $:GPU_PARALLEL_LOOP(collapse=4)
            do i = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            q_prim_ts2(2)%vf(i)%sf(j, k, l) = q_prim_ts2(1)%vf(i)%sf(j, k, l)
                            q_prim_ts2(1)%vf(i)%sf(j, k, l) = q_prim_ts1(2)%vf(i)%sf(j, k, l)
                            q_prim_ts1(2)%vf(i)%sf(j, k, l) = q_prim_ts1(1)%vf(i)%sf(j, k, l)
                            q_prim_ts1(1)%vf(i)%sf(j, k, l) = q_prim_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

    end subroutine s_time_step_cycling

    !> Module deallocation and/or disassociation procedures
    impure subroutine s_finalize_time_steppers_module
#ifdef FRONTIER_UNIFIED
        use hipfort
        use hipfort_hipmalloc
        use hipfort_check
#endif
        integer :: i, j !< Generic loop iterators

        ! Deallocating the cell-average conservative variables
#if defined(__NVCOMPILER_GPU_UNIFIED_MEM)
        do j = 1, sys_size
            @:DEALLOCATE(q_cons_ts(1)%vf(j)%sf)
            if (num_ts == 2) then
                if (nv_uvm_out_of_core) then
                    nullify (q_cons_ts(2)%vf(j)%sf)
                else
                    @:DEALLOCATE(q_cons_ts(2)%vf(j)%sf)
                end if
            end if
        end do
        if (num_ts == 2 .and. nv_uvm_out_of_core) then
            deallocate (q_cons_ts_pool_host)
        end if
#elif defined(FRONTIER_UNIFIED)
        do i = 1, num_ts
            do j = 1, sys_size
                nullify (q_cons_ts(i)%vf(j)%sf)
            end do
        end do
#ifdef MFC_MIXED_PRECISION
        call hipCheck(hipHostFree_(c_loc(q_cons_ts_pool_host)))
        nullify (q_cons_ts_pool_host)
        call hipCheck(hipFree_(c_loc(q_cons_ts_pool_device)))
        nullify (q_cons_ts_pool_device)
#else
        call hipCheck(hipHostFree(q_cons_ts_pool_host))
        call hipCheck(hipFree(q_cons_ts_pool_device))
#endif
#else
        do i = 1, num_ts
            do j = 1, sys_size
                @:DEALLOCATE(q_cons_ts(i)%vf(j)%sf)
            end do
        end do
#endif
        do i = 1, num_ts
            @:DEALLOCATE(q_cons_ts(i)%vf)
        end do

        @:DEALLOCATE(q_cons_ts)

        ! Deallocating the cell-average primitive ts variables
        if (probe_wrt) then
            do i = 1, num_probe_ts
                do j = 1, sys_size
                    @:DEALLOCATE(q_prim_ts1(i)%vf(j)%sf,q_prim_ts2(i)%vf(j)%sf )
                end do
                @:DEALLOCATE(q_prim_ts1(i)%vf, q_prim_ts2(i)%vf)
            end do
            @:DEALLOCATE(q_prim_ts1, q_prim_ts2)
        end if

        if (.not. igr) then
            ! Deallocating the cell-average primitive variables
            do i = 1, adv_idx%end
                @:DEALLOCATE(q_prim_vf(i)%sf)
            end do

            if (mhd) then
                do i = B_idx%beg, B_idx%end
                    @:DEALLOCATE(q_prim_vf(i)%sf)
                end do
            end if

            if (elasticity) then
                do i = stress_idx%beg, stress_idx%end
                    @:DEALLOCATE(q_prim_vf(i)%sf)
                end do
            end if

            if (hyperelasticity) then
                do i = xibeg, xiend + 1
                    @:DEALLOCATE(q_prim_vf(i)%sf)
                end do
            end if

            if (cont_damage) then
                @:DEALLOCATE(q_prim_vf(damage_idx)%sf)
            end if

            if (hyper_cleaning) then
                @:DEALLOCATE(q_prim_vf(psi_idx)%sf)
            end if

            if (bubbles_euler) then
                do i = bub_idx%beg, bub_idx%end
                    @:DEALLOCATE(q_prim_vf(i)%sf)
                end do
            end if

            if (model_eqns == 3) then
                do i = internalEnergies_idx%beg, internalEnergies_idx%end
                    @:DEALLOCATE(q_prim_vf(i)%sf)
                end do
            end if
        end if

        @:DEALLOCATE(q_prim_vf)

        ! Deallocating the cell-average RHS variables
        do i = 1, sys_size
            @:DEALLOCATE(rhs_vf(i)%sf)
        end do

        @:DEALLOCATE(rhs_vf)

        if (relax) then
            @:DEALLOCATE(m_dot_evap%sf)
        end if

        ! Writing the footer of and closing the run-time information file
        if (proc_rank == 0 .and. run_time_info) then
            call s_close_run_time_information_file()
        end if

    end subroutine s_finalize_time_steppers_module

end module m_time_steppers
