!>
!! @file
!! @brief Contains module m_rhs

#:include 'case.fpp'
#:include 'macros.fpp'

!> @brief Assembles the right-hand side of the governing equations using finite-volume flux differencing, Riemann solvers, and physical source terms
module m_rhs

    use iso_fortran_env, only: output_unit

    use m_derived_types        !< Definitions of the derived types

    use m_global_parameters    !< Definitions of the global parameters

    use m_mpi_proxy            !< Message passing interface (MPI) module proxy

    use m_mpi_common, only: s_mpi_allreduce_max_common => s_mpi_allreduce_max

    use m_variables_conversion !< State variables type conversion procedures

    use m_weno                 !< Weighted and essentially non-oscillatory (WENO)
                               !! schemes for spatial reconstruction of variables

    use m_muscl                !< Monotonic Upstream-centered (MUSCL)
                               !! schemes for conservation laws

    use m_riemann_solvers      !< Exact and approximate Riemann problem solvers

    use m_cbc                  !< Characteristic boundary conditions (CBC)

    use m_bubbles_EE           !< Ensemble-averaged bubble dynamics routines

    use m_bubbles_EL

    use m_qbmm                 !< Moment inversion

    use m_hypoelastic

    use m_hyperelastic

    use m_acoustic_src

    use m_viscous

    use m_ibm

    use m_nvtx

    use m_boundary_common

    use m_helper

    use m_surface_tension

    use m_body_forces

    use m_chemistry

    use m_igr

    use m_pressure_relaxation

    implicit none

    private; public :: s_initialize_rhs_module, &
 s_compute_rhs, &
 s_alpha_origin_checkpoint, &
 s_alpha_origin_capture_rk_cause, &
 s_alpha_origin_print_rk_cause, &
 s_finalize_rhs_module

    !! This variable contains the WENO-reconstructed values of the cell-average
    !! conservative variables, which are located in q_cons_vf, at cell-interior
    !! Gaussian quadrature points (QP).
    type(vector_field) :: q_cons_qp !<
    $:GPU_DECLARE(create='[q_cons_qp]')

    !! The primitive variables at cell-interior Gaussian quadrature points. These
    !! are calculated from the conservative variables and gradient magnitude (GM)
    !! of the volume fractions, q_cons_qp and gm_alpha_qp, respectively.
    type(vector_field) :: q_prim_qp !<
    $:GPU_DECLARE(create='[q_prim_qp]')

    !> @name The first-order spatial derivatives of the primitive variables at cell-
    !! interior Gaussian quadrature points. These are WENO-reconstructed from
    !! their respective cell-average values, obtained through the application
    !! of the divergence theorem on the integral-average cell-boundary values
    !! of the primitive variables, located in qK_prim_n, where K = L or R.
    !> @{
    type(vector_field), allocatable, dimension(:) :: dq_prim_dx_qp, dq_prim_dy_qp, dq_prim_dz_qp
    $:GPU_DECLARE(create='[dq_prim_dx_qp,dq_prim_dy_qp,dq_prim_dz_qp]')
    !> @}

    !> @name The left and right WENO-reconstructed cell-boundary values of the cell-
    !! average first-order spatial derivatives of the primitive variables. The
    !! cell-average of the first-order spatial derivatives may be found in the
    !! variables dq_prim_ds_qp, where s = x, y or z.
    !> @{
    type(vector_field), allocatable, dimension(:) :: dqL_prim_dx_n, dqL_prim_dy_n, dqL_prim_dz_n
    type(vector_field), allocatable, dimension(:) :: dqR_prim_dx_n, dqR_prim_dy_n, dqR_prim_dz_n
#if defined(MFC_OpenACC)
    $:GPU_DECLARE(create='[dqL_prim_dx_n,dqL_prim_dy_n,dqL_prim_dz_n]')
    $:GPU_DECLARE(create='[dqR_prim_dx_n,dqR_prim_dy_n,dqR_prim_dz_n]')
#endif
    !> @}

    type(scalar_field), allocatable, dimension(:) :: tau_Re_vf
    $:GPU_DECLARE(create='[tau_Re_vf]')

    type(vector_field) :: gm_alpha_qp  !<
    !! The gradient magnitude of the volume fractions at cell-interior Gaussian
    !! quadrature points. gm_alpha_qp is calculated from individual first-order
    !! spatial derivatives located in dq_prim_ds_qp.

    $:GPU_DECLARE(create='[gm_alpha_qp]')

    !> @name The left and right WENO-reconstructed cell-boundary values of the cell-
    !! average gradient magnitude of volume fractions, located in gm_alpha_qp.
    !> @{
    type(vector_field), allocatable, dimension(:) :: gm_alphaL_n
    type(vector_field), allocatable, dimension(:) :: gm_alphaR_n
#if defined(MFC_OpenACC)
    $:GPU_DECLARE(create='[gm_alphaL_n,gm_alphaR_n]')
#endif
    !> @}

    !> @name The cell-boundary values of the fluxes (src - source, gsrc - geometrical
    !! source). These are computed by applying the chosen Riemann problem solver
    !! .on the left and right cell-boundary values of the primitive variables
    !> @{
    type(vector_field), allocatable, dimension(:) :: flux_n
    type(vector_field), allocatable, dimension(:) :: flux_src_n
    type(vector_field), allocatable, dimension(:) :: flux_gsrc_n

#if defined(MFC_OpenACC)
    $:GPU_DECLARE(create='[flux_n,flux_src_n,flux_gsrc_n]')
#endif

    !> @}

    type(vector_field), allocatable, dimension(:) :: qL_prim, qR_prim
#if defined(MFC_OpenACC)
    $:GPU_DECLARE(create='[qL_prim,qR_prim]')
#endif

    type(int_bounds_info) :: iv !< Vector field indical bounds
    $:GPU_DECLARE(create='[iv]')

    !> @name Indical bounds in the x-, y- and z-directions
    !> @{
    type(int_bounds_info) :: irx, iry, irz
    $:GPU_DECLARE(create='[irx,iry,irz]')

    type(int_bounds_info) :: is1, is2, is3
    !> @}
    $:GPU_DECLARE(create='[is1,is2,is3]')

    !> @name Saved fluxes for testing
    !> @{
    type(scalar_field) :: alf_sum
    !> @}
    $:GPU_DECLARE(create='[alf_sum]')

    real(wp), allocatable, dimension(:, :, :) :: blkmod1, blkmod2, alpha1, alpha2, Kterm
    real(wp), allocatable, dimension(:, :, :, :) :: qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, qR_rsx_vf, qR_rsy_vf, qR_rsz_vf
    real(wp), allocatable, dimension(:, :, :, :) :: dqL_rsx_vf, dqL_rsy_vf, dqL_rsz_vf, dqR_rsx_vf, dqR_rsy_vf, dqR_rsz_vf
    $:GPU_DECLARE(create='[blkmod1,blkmod2,alpha1,alpha2,Kterm]')
    $:GPU_DECLARE(create='[qL_rsx_vf,qL_rsy_vf,qL_rsz_vf,qR_rsx_vf,qR_rsy_vf,qR_rsz_vf]')
    $:GPU_DECLARE(create='[dqL_rsx_vf,dqL_rsy_vf,dqL_rsz_vf,dqR_rsx_vf,dqR_rsy_vf,dqR_rsz_vf]')

    real(wp), allocatable, dimension(:, :, :) :: nbub !< Bubble number density
    $:GPU_DECLARE(create='[nbub]')

    integer, parameter :: alpha_origin_max_records = 16
    logical :: alpha_origin_env_checked = .false.
    logical :: alpha_origin_enabled = .false.
    logical :: alpha_origin_first_reported = .false.
    integer :: alpha_origin_record_count = 0
    integer :: alpha_origin_record_t_step = -huge(0)
    integer :: alpha_origin_record_stage = -huge(0)
    character(len=40), dimension(alpha_origin_max_records) :: alpha_origin_record_checkpoint
    character(len=8), dimension(alpha_origin_max_records) :: alpha_origin_record_scope
    integer, dimension(alpha_origin_max_records) :: alpha_origin_record_invalid
    integer, dimension(alpha_origin_max_records) :: alpha_origin_record_nonfinite
    integer, dimension(alpha_origin_max_records) :: alpha_origin_record_bounds
    integer, dimension(alpha_origin_max_records) :: alpha_origin_record_sum

    logical :: alpha_origin_rk_cause_ready = .false.
    integer :: alpha_origin_rk_owner = -1
    integer :: alpha_origin_rk_t_step = -1
    integer :: alpha_origin_rk_stage = -1
    integer, dimension(3) :: alpha_origin_rk_local_ijk = 0
    integer, dimension(3) :: alpha_origin_rk_global_ijk = 0
    real(wp), dimension(num_fluids_max) :: alpha_origin_rk_before = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_origin_rk_stored = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_origin_rk_rhs = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_origin_rk_predicted = 0._wp
    real(wp) :: alpha_origin_rk_rhs_sum = 0._wp
    real(wp) :: alpha_origin_rk_a = 0._wp
    real(wp) :: alpha_origin_rk_b = 0._wp
    real(wp) :: alpha_origin_rk_c = 0._wp
    real(wp) :: alpha_origin_rk_d = 1._wp
    real(wp) :: alpha_origin_rk_dt = 0._wp
    real(wp) :: alpha_origin_rk_rhs_factor = 0._wp
    logical :: alpha_origin_rk_igr = .false.

    integer, parameter :: alpha_rhs_target_i = 85
    integer, parameter :: alpha_rhs_target_j = 0
    integer, parameter :: alpha_rhs_target_k = 0
    real(wp), dimension(num_fluids_max) :: alpha_rhs_previous = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_rhs_cbc_flux_left = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_rhs_cbc_flux_right = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_rhs_cbc_src_left = 0._wp
    real(wp), dimension(num_fluids_max) :: alpha_rhs_cbc_src_right = 0._wp
    logical :: alpha_rhs_previous_valid = .false.
    logical :: alpha_rhs_cbc_pre_valid = .false.
    logical :: alpha_origin_invalid_region_reported = .false.

contains

    logical function f_alpha_origin_active()
        character(len=16) :: env_value
        integer :: env_status

        if (.not. alpha_origin_env_checked) then
            call get_environment_variable("TEMP_ALPHA_ORIGIN_DEBUG", env_value, status=env_status)
            alpha_origin_enabled = env_status == 0 .and. trim(env_value) == "1"
            alpha_origin_env_checked = .true.
        end if

        f_alpha_origin_active = alpha_origin_enabled .and. (.not. alpha_origin_first_reported)
    end function f_alpha_origin_active

    logical function f_alpha_origin_finite(value)
        $:GPU_ROUTINE(function_name='f_alpha_origin_finite', parallelism='[seq]', cray_inline=True)
        real(stp), intent(in) :: value

        f_alpha_origin_finite = value == value .and. abs(value) <= huge(value)
    end function f_alpha_origin_finite

    logical function f_alpha_rhs_target_owner(local_i, local_j, local_k)
        integer, intent(out) :: local_i, local_j, local_k
        integer :: i_offset, j_offset, k_offset

        i_offset = 0
        j_offset = 0
        k_offset = 0
        if (allocated(start_idx)) then
            if (size(start_idx) >= 1) i_offset = start_idx(1)
            if (size(start_idx) >= 2) j_offset = start_idx(2)
            if (size(start_idx) >= 3) k_offset = start_idx(3)
        end if

        local_i = alpha_rhs_target_i - i_offset
        local_j = alpha_rhs_target_j - j_offset
        local_k = alpha_rhs_target_k - k_offset
        f_alpha_rhs_target_owner = local_i >= 0 .and. local_i <= m .and. &
                                   local_j >= 0 .and. local_j <= n .and. &
                                   local_k >= 0 .and. local_k <= p
    end function f_alpha_rhs_target_owner

    subroutine s_alpha_rhs_term(rhs_vf, checkpoint, t_step, stage, zero_state)
        type(scalar_field), dimension(sys_size), intent(in) :: rhs_vf
        character(len=*), intent(in) :: checkpoint
        integer, intent(in) :: t_step, stage
        logical, optional, intent(in) :: zero_state

        integer :: i, local_i, local_j, local_k
        logical :: report_zero
        real(wp), dimension(num_fluids_max) :: alpha_rhs, alpha_increment

        if (.not. f_alpha_origin_active()) return
        if (igr .and. .not. dummy) return
        if (t_step /= 0 .or. stage /= 1) return
        if (.not. f_alpha_rhs_target_owner(local_i, local_j, local_k)) return

        report_zero = .false.
        if (present(zero_state)) report_zero = zero_state
        alpha_rhs = 0._wp
        if (.not. report_zero) then
            $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', copyout='[alpha_rhs]')
            do i = 1, num_fluids
                alpha_rhs(i) = real(rhs_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        if (alpha_rhs_previous_valid) then
            alpha_increment(1:num_fluids) = alpha_rhs(1:num_fluids) - alpha_rhs_previous(1:num_fluids)
        else
            alpha_increment = 0._wp
        end if

        write (output_unit, '(&
            &"TEMP_ALPHA_RHS_TERM t_step=", I0, " stage=", I0, " checkpoint=", A, &
            &" global_ijk=", 3(I0,1X), " alpha_rhs=")', advance='no') &
            t_step, stage, trim(checkpoint), alpha_rhs_target_i, alpha_rhs_target_j, alpha_rhs_target_k
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_rhs(i)
        end do
        write (output_unit, '(" rhs_alpha_sum=", ES16.8, " incremental=")', advance='no') &
            sum(alpha_rhs(1:num_fluids))
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_increment(i)
        end do
        write (output_unit, '(" incremental_sum=", ES16.8)') sum(alpha_increment(1:num_fluids))
        call flush(output_unit)

        alpha_rhs_previous(1:num_fluids) = alpha_rhs(1:num_fluids)
        alpha_rhs_previous_valid = .true.
    end subroutine s_alpha_rhs_term

    subroutine s_alpha_rhs_target_state(q_cons_vf, q_prim_vf, bc_type, t_step, stage)
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf, q_prim_vf
        type(integer_field), dimension(1:num_dims, 1:2), intent(in) :: bc_type
        integer, intent(in) :: t_step, stage

        integer :: i, offset, local_i, local_j, local_k, lower_y_bc_type
        logical :: lower_y_cbc
        real(wp) :: pressure
        real(wp), dimension(3) :: velocity
        real(wp), dimension(num_fluids_max) :: alpha_before
        real(wp), dimension(num_fluids_max, -buff_size:buff_size) :: alpha_x_stencil, alpha_y_stencil

        if (.not. f_alpha_origin_active()) return
        if (igr .and. .not. dummy) return
        if (t_step /= 0 .or. stage /= 1) return
        if (.not. f_alpha_rhs_target_owner(local_i, local_j, local_k)) return

        alpha_before = 0._wp
        alpha_x_stencil = 0._wp
        alpha_y_stencil = 0._wp
        velocity = 0._wp
        pressure = 0._wp
        lower_y_bc_type = 0

        $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', copyout='[alpha_before]')
        do i = 1, num_fluids
            alpha_before(i) = real(q_cons_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
        end do
        $:END_GPU_PARALLEL_LOOP()

        $:GPU_PARALLEL_LOOP(collapse=2, firstprivate='[local_i,local_j,local_k]', &
            private='[i,offset]', copyout='[alpha_x_stencil,alpha_y_stencil]')
        do i = 1, num_fluids
            do offset = -buff_size, buff_size
                alpha_x_stencil(i, offset) = real(q_prim_vf(advxb + i - 1)%sf(local_i + offset, local_j, local_k), kind=wp)
                alpha_y_stencil(i, offset) = real(q_prim_vf(advxb + i - 1)%sf(local_i, local_j + offset, local_k), kind=wp)
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', copyout='[velocity]')
        do i = 1, 3
            velocity(i) = 0._wp
            if (i <= num_vels) &
                velocity(i) = real(q_prim_vf(momxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
        end do
        $:END_GPU_PARALLEL_LOOP()

        $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', copyout='[pressure]')
        do i = 1, 1
            pressure = real(q_prim_vf(E_idx)%sf(local_i, local_j, local_k), kind=wp)
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (n > 0) then
            $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_k]', copyout='[lower_y_bc_type]')
            do i = 1, 1
                lower_y_bc_type = int(bc_type(2, 1)%sf(local_i, 0, local_k))
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if
        lower_y_cbc = bc_y%beg <= BC_CHAR_SLIP_WALL .and. bc_y%beg >= BC_CHAR_SUP_OUTFLOW

        write (output_unit, '(&
            &"TEMP_ALPHA_RHS_TARGET t_step=", I0, " stage=", I0, " global_ijk=", 3(I0,1X), &
            &" rank=", I0, " local_ijk=", 3(I0,1X), " alpha_before=")', advance='no') &
            t_step, stage, alpha_rhs_target_i, alpha_rhs_target_j, alpha_rhs_target_k, proc_rank, &
            local_i, local_j, local_k
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_before(i)
        end do
        write (output_unit, '(&
            &" velocity=", 3(1X,ES16.8), " pressure=", ES16.8, &
            &" lower_y_boundary_id=", I0, " lower_y_boundary_type=", I0, &
            &" target_on_lower_y=", L1, " lower_y_cbc_active=", L1, " stencil_radius=", I0)') &
            velocity, pressure, bc_y%beg, lower_y_bc_type, alpha_rhs_target_j == 0, lower_y_cbc, buff_size
        do i = 1, num_fluids
            write (output_unit, '(&
                &"TEMP_ALPHA_RHS_TARGET stencil fluid=", I0, " x_alpha=")', advance='no') i
            do offset = -buff_size, buff_size
                write (output_unit, '(1X,I0,":",ES16.8)', advance='no') offset, alpha_x_stencil(i, offset)
            end do
            write (output_unit, '(" y_alpha=")', advance='no')
            do offset = -buff_size, buff_size
                write (output_unit, '(1X,I0,":",ES16.8)', advance='no') offset, alpha_y_stencil(i, offset)
            end do
            write (output_unit, *)
        end do
        call flush(output_unit)
    end subroutine s_alpha_rhs_target_state

    subroutine s_alpha_rhs_cbc_faces(flux_vf, flux_src_vf, idir, phase, t_step, stage)
        type(scalar_field), dimension(sys_size), intent(in) :: flux_vf, flux_src_vf
        integer, intent(in) :: idir, t_step, stage
        character(len=*), intent(in) :: phase

        integer :: i, local_i, local_j, local_k
        logical :: cbc_active
        real(wp) :: inv_ds
        real(wp), dimension(num_fluids_max) :: flux_left, flux_right, src_left, src_right, rhs_effect
        character(len=1) :: dir_label

        if (.not. f_alpha_origin_active()) return
        if (igr .and. .not. dummy) return
        if (t_step /= 0 .or. stage /= 1 .or. idir > 2) return
        if (.not. f_alpha_rhs_target_owner(local_i, local_j, local_k)) return

        flux_left = 0._wp
        flux_right = 0._wp
        src_left = 0._wp
        src_right = 0._wp
        if (idir == 1) then
            dir_label = "X"
            $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', &
                copyout='[flux_left,flux_right,src_left,src_right]')
            do i = 1, num_fluids
                flux_left(i) = real(flux_vf(advxb + i - 1)%sf(local_i - 1, local_j, local_k), kind=wp)
                flux_right(i) = real(flux_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
                src_left(i) = real(flux_src_vf(advxb + i - 1)%sf(local_i - 1, local_j, local_k), kind=wp)
                src_right(i) = real(flux_src_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
            end do
            $:END_GPU_PARALLEL_LOOP()
            inv_ds = 1._wp/dx(local_i)
            cbc_active = (alpha_rhs_target_i == 0 .and. bc_x%beg <= BC_CHAR_SLIP_WALL .and. bc_x%beg >= BC_CHAR_SUP_OUTFLOW) .or. &
                         (alpha_rhs_target_i == m_glb .and. bc_x%end <= BC_CHAR_SLIP_WALL .and. bc_x%end >= BC_CHAR_SUP_OUTFLOW)
        else
            dir_label = "Y"
            $:GPU_PARALLEL_LOOP(firstprivate='[local_i,local_j,local_k]', &
                copyout='[flux_left,flux_right,src_left,src_right]')
            do i = 1, num_fluids
                flux_left(i) = real(flux_vf(advxb + i - 1)%sf(local_i, local_j - 1, local_k), kind=wp)
                flux_right(i) = real(flux_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
                src_left(i) = real(flux_src_vf(advxb + i - 1)%sf(local_i, local_j - 1, local_k), kind=wp)
                src_right(i) = real(flux_src_vf(advxb + i - 1)%sf(local_i, local_j, local_k), kind=wp)
            end do
            $:END_GPU_PARALLEL_LOOP()
            inv_ds = 1._wp/dy(local_j)
            cbc_active = (alpha_rhs_target_j == 0 .and. bc_y%beg <= BC_CHAR_SLIP_WALL .and. bc_y%beg >= BC_CHAR_SUP_OUTFLOW) .or. &
                         (alpha_rhs_target_j == n_glb .and. bc_y%end <= BC_CHAR_SLIP_WALL .and. bc_y%end >= BC_CHAR_SUP_OUTFLOW)
        end if

        if (trim(phase) == "PRE") then
            alpha_rhs_cbc_flux_left(1:num_fluids) = flux_left(1:num_fluids)
            alpha_rhs_cbc_flux_right(1:num_fluids) = flux_right(1:num_fluids)
            alpha_rhs_cbc_src_left(1:num_fluids) = src_left(1:num_fluids)
            alpha_rhs_cbc_src_right(1:num_fluids) = src_right(1:num_fluids)
            alpha_rhs_cbc_pre_valid = .true.
            return
        end if
        if (.not. alpha_rhs_cbc_pre_valid) return

        rhs_effect = 0._wp
        rhs_effect(1:num_fluids) = inv_ds*((flux_left(1:num_fluids) - alpha_rhs_cbc_flux_left(1:num_fluids)) &
                                           - (flux_right(1:num_fluids) - alpha_rhs_cbc_flux_right(1:num_fluids)))
        write (output_unit, '(&
            &"TEMP_ALPHA_RHS_TERM t_step=", I0, " stage=", I0, " checkpoint=", A, &
            &"_CBC_BOUNDARY_FLUX cbc_active=", L1, " alpha_rhs=")', advance='no') &
            t_step, stage, dir_label, cbc_active
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_rhs_previous(i)
        end do
        write (output_unit, '(" rhs_alpha_sum=", ES16.8, " incremental=")', advance='no') &
            sum(alpha_rhs_previous(1:num_fluids))
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') 0._wp
        end do
        write (output_unit, '(" incremental_sum=", ES16.8, " boundary_flux_rhs_effect=")', advance='no') 0._wp
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') rhs_effect(i)
        end do
        write (output_unit, '(" boundary_flux_rhs_effect_sum=", ES16.8)') sum(rhs_effect(1:num_fluids))

        write (output_unit, '(&
            &"TEMP_ALPHA_RHS_TARGET t_step=", I0, " stage=", I0, " face_idir=", I0, &
            &" global_ijk=", 3(I0,1X), " alpha_flux_left=")', advance='no') &
            t_step, stage, idir, alpha_rhs_target_i, alpha_rhs_target_j, alpha_rhs_target_k
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') flux_left(i)
        end do
        write (output_unit, '(" alpha_flux_right=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') flux_right(i)
        end do
        write (output_unit, '(" alpha_src_flux_left=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') src_left(i)
        end do
        write (output_unit, '(" alpha_src_flux_right=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') src_right(i)
        end do
        write (output_unit, '(" src_flux_delta_left=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') src_left(i) - alpha_rhs_cbc_src_left(i)
        end do
        write (output_unit, '(" src_flux_delta_right=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') src_right(i) - alpha_rhs_cbc_src_right(i)
        end do
        write (output_unit, *)
        call flush(output_unit)
        alpha_rhs_cbc_pre_valid = .false.
    end subroutine s_alpha_rhs_cbc_faces

    subroutine s_alpha_origin_record_count(t_step, stage, checkpoint, scope, invalid_count, nonfinite_count, bounds_count, sum_count)
        integer, intent(in) :: t_step, stage
        character(len=*), intent(in) :: checkpoint, scope
        integer, intent(in) :: invalid_count, nonfinite_count, bounds_count, sum_count

        if (t_step /= alpha_origin_record_t_step .or. stage /= alpha_origin_record_stage) then
            alpha_origin_record_count = 0
            alpha_origin_record_t_step = t_step
            alpha_origin_record_stage = stage
        end if

        if (alpha_origin_record_count >= alpha_origin_max_records) return

        alpha_origin_record_count = alpha_origin_record_count + 1
        alpha_origin_record_checkpoint(alpha_origin_record_count) = trim(checkpoint)
        alpha_origin_record_scope(alpha_origin_record_count) = trim(scope)
        alpha_origin_record_invalid(alpha_origin_record_count) = invalid_count
        alpha_origin_record_nonfinite(alpha_origin_record_count) = nonfinite_count
        alpha_origin_record_bounds(alpha_origin_record_count) = bounds_count
        alpha_origin_record_sum(alpha_origin_record_count) = sum_count
    end subroutine s_alpha_origin_record_count

    subroutine s_alpha_origin_flush_counts()
        integer :: record_id

        if (proc_rank /= 0) return

        do record_id = 1, alpha_origin_record_count
            print '(&
                &"TEMP_ALPHA_ORIGIN_COUNT t_step=", I10, " stage=", I4, &
                &" checkpoint=", A, " scope=", A, &
                &" global_invalid=", I10, " global_nonfinite=", I10, &
                &" global_bounds=", I10, " global_sum=", I10)', &
                alpha_origin_record_t_step, alpha_origin_record_stage, &
                trim(alpha_origin_record_checkpoint(record_id)), &
                trim(alpha_origin_record_scope(record_id)), &
                alpha_origin_record_invalid(record_id), &
                alpha_origin_record_nonfinite(record_id), &
                alpha_origin_record_bounds(record_id), &
                alpha_origin_record_sum(record_id)
        end do
        call flush(output_unit)
    end subroutine s_alpha_origin_flush_counts

    subroutine s_alpha_origin_checkpoint(q_vf, t_step, stage, checkpoint, scope, first_found)
        type(scalar_field), dimension(sys_size), intent(in) :: q_vf
        integer, intent(in) :: t_step, stage
        character(len=*), intent(in) :: checkpoint, scope
        logical, optional, intent(out) :: first_found

        integer :: i, j, k, l
        integer :: j_beg, j_end, k_beg, k_end, l_beg, l_end
        integer :: global_j_offset, global_k_offset, global_l_offset
        integer :: local_invalid, local_nonfinite, local_bounds, local_sum
        integer :: global_invalid, global_nonfinite, global_bounds, global_sum
        integer :: owner_rank, selected_j, selected_k, selected_l
        integer :: selected_global_j, selected_global_k, selected_global_l
        integer(kind=8) :: local_first_key, global_first_key, cell_key
        integer(kind=8) :: global_width, global_height
        logical :: scan_ghost, in_scope
        logical :: cell_nonfinite, cell_bounds, cell_sum_invalid, cell_invalid
        real(stp) :: alpha_value
        real(wp) :: alpha_sum
        real(wp), dimension(2) :: owner_pair
        real(wp), dimension(num_fluids_max) :: selected_alpha

        if (present(first_found)) first_found = .false.
        if (.not. f_alpha_origin_active()) return

        scan_ghost = trim(scope) == "ghost"
        if (scan_ghost) then
            j_beg = idwbuff(1)%beg; j_end = idwbuff(1)%end
            k_beg = idwbuff(2)%beg; k_end = idwbuff(2)%end
            l_beg = idwbuff(3)%beg; l_end = idwbuff(3)%end
        else
            j_beg = 0; j_end = m
            k_beg = 0; k_end = n
            l_beg = 0; l_end = p
        end if

        global_j_offset = 0
        global_k_offset = 0
        global_l_offset = 0
        if (allocated(start_idx)) then
            if (size(start_idx) >= 1) global_j_offset = start_idx(1)
            if (size(start_idx) >= 2) global_k_offset = start_idx(2)
            if (size(start_idx) >= 3) global_l_offset = start_idx(3)
        end if

        global_width = int(m_glb + 1 + 2*buff_size, kind=8)
        global_height = int(max(1, n_glb + 1 + 2*buff_size), kind=8)

        local_invalid = 0
        local_nonfinite = 0
        local_bounds = 0
        local_sum = 0
        local_first_key = huge(local_first_key)

        $:GPU_PARALLEL_LOOP(collapse=3, &
            private='[i,j,k,l,in_scope,cell_nonfinite,cell_bounds,cell_sum_invalid,cell_invalid,alpha_value,alpha_sum,cell_key]', &
            firstprivate='[scan_ghost,j_beg,j_end,k_beg,k_end,l_beg,l_end,global_j_offset,global_k_offset,global_l_offset,global_width,global_height]', &
            reduction='[[local_invalid,local_nonfinite,local_bounds,local_sum],[local_first_key]]', &
            reductionOp='[+,MIN]')
        do l = l_beg, l_end
            do k = k_beg, k_end
                do j = j_beg, j_end
                    in_scope = .true.
                    if (scan_ghost) then
                        in_scope = j < 0 .or. j > m .or. k < 0 .or. k > n .or. l < 0 .or. l > p
                    end if

                    if (in_scope) then
                        alpha_sum = 0._wp
                        cell_nonfinite = .false.
                        cell_bounds = .false.
                        $:GPU_LOOP(parallelism='[seq]')
                        do i = 1, num_fluids
                            alpha_value = q_vf(advxb + i - 1)%sf(j, k, l)
                            if (.not. f_alpha_origin_finite(alpha_value)) then
                                cell_nonfinite = .true.
                            else
                                alpha_sum = alpha_sum + real(alpha_value, kind=wp)
                                if (real(alpha_value, kind=wp) < -1.e-12_wp .or. &
                                    real(alpha_value, kind=wp) > 1._wp + 1.e-12_wp) &
                                    cell_bounds = .true.
                            end if
                        end do

                        cell_sum_invalid = (.not. cell_nonfinite) .and. abs(alpha_sum - 1._wp) > 1.e-6_wp
                        cell_invalid = cell_nonfinite .or. cell_bounds .or. cell_sum_invalid

                        if (cell_nonfinite) local_nonfinite = local_nonfinite + 1
                        if (cell_bounds) local_bounds = local_bounds + 1
                        if (cell_sum_invalid) local_sum = local_sum + 1
                        if (cell_invalid) then
                            local_invalid = local_invalid + 1
                            cell_key = int(j + global_j_offset + buff_size, kind=8) &
                                       + global_width*int(k + global_k_offset + buff_size, kind=8) &
                                       + global_width*global_height*int(l + global_l_offset + buff_size, kind=8)
                            local_first_key = min(local_first_key, cell_key)
                        end if
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call s_mpi_allreduce_integer_sum(local_invalid, global_invalid)
        call s_mpi_allreduce_integer_sum(local_nonfinite, global_nonfinite)
        call s_mpi_allreduce_integer_sum(local_bounds, global_bounds)
        call s_mpi_allreduce_integer_sum(local_sum, global_sum)

        call s_alpha_origin_record_count(t_step, stage, checkpoint, scope, &
                                         global_invalid, global_nonfinite, global_bounds, global_sum)

        if (trim(checkpoint) == "RHS_PRIM_PRE_HALO" .and. trim(scope) == "ghost") return

        if (global_invalid == 0) return

        owner_pair = [merge(-real(local_first_key, kind=wp), -huge(1._wp), local_invalid > 0), &
                      real(proc_rank, kind=wp)]
        call s_mpi_reduce_maxloc(owner_pair)
        global_first_key = int(-owner_pair(1), kind=8)
        owner_rank = int(owner_pair(2))

        selected_global_l = int(global_first_key/(global_width*global_height)) - buff_size
        global_first_key = global_first_key &
                           - int(selected_global_l + buff_size, kind=8)*global_width*global_height
        selected_global_k = int(global_first_key/global_width) - buff_size
        selected_global_j = int(global_first_key &
                                - int(selected_global_k + buff_size, kind=8)*global_width) - buff_size
        selected_j = selected_global_j - global_j_offset
        selected_k = selected_global_k - global_k_offset
        selected_l = selected_global_l - global_l_offset

        call s_alpha_origin_flush_counts()

        if (proc_rank == owner_rank) then
            selected_alpha = 0._wp
            $:GPU_PARALLEL_LOOP(firstprivate='[selected_j,selected_k,selected_l]', copyout='[selected_alpha]')
            do i = 1, num_fluids
                selected_alpha(i) = real(q_vf(advxb + i - 1)%sf(selected_j, selected_k, selected_l), kind=wp)
            end do
            $:END_GPU_PARALLEL_LOOP()

            alpha_sum = sum(selected_alpha(1:num_fluids))
            write (output_unit, '(&
                &"TEMP_ALPHA_ORIGIN_FIRST t_step=", I10, " stage=", I4, &
                &" checkpoint=", A, " rank=", I6, &
                &" local_ijk=", 3(I8,1X), " global_ijk=", 3(I8,1X), &
                &" alpha=")', advance='no') &
                t_step, stage, trim(checkpoint), proc_rank, &
                selected_j, selected_k, selected_l, &
                selected_global_j, selected_global_k, selected_global_l
            do i = 1, num_fluids
                write (output_unit, '(1X,ES16.8)', advance='no') selected_alpha(i)
            end do
            write (output_unit, '(" alpha_sum=", ES16.8)') alpha_sum
            call flush(output_unit)
        end if

        alpha_origin_first_reported = .true.
        if (present(first_found)) first_found = .true.
    end subroutine s_alpha_origin_checkpoint

    subroutine s_alpha_origin_capture_rk_cause(q_current, q_stored, rhs, t_step, stage, rk_a, rk_b, rk_c, rk_d, dt_value)
        type(scalar_field), dimension(sys_size), intent(in) :: q_current, q_stored, rhs
        integer, intent(in) :: t_step, stage
        real(wp), intent(in) :: rk_a, rk_b, rk_c, rk_d, dt_value

        integer :: i, j, k, l
        integer :: global_j_offset, global_k_offset, global_l_offset
        integer :: local_invalid, global_invalid, owner_rank
        integer :: local_bounds_count, global_bounds_count, local_sum_count, global_sum_count
        integer :: local_j0_count, global_j0_count, local_jmax_count, global_jmax_count
        integer :: local_x_boundary_count, global_x_boundary_count, local_interior_count, global_interior_count
        integer :: selected_j, selected_k, selected_l
        integer :: selected_global_j, selected_global_k, selected_global_l
        integer(kind=8) :: local_first_key, global_first_key, cell_key
        integer(kind=8) :: global_width, global_height
        logical :: cell_nonfinite, cell_bounds, cell_sum_invalid
        real(stp) :: alpha_before_value, alpha_stored_value, alpha_rhs_value, alpha_predicted_value
        real(wp) :: alpha_sum, rhs_factor
        real(wp) :: local_i_min, local_i_max, local_j_min, local_j_max
        real(wp) :: global_i_max, global_j_max
        real(wp), dimension(2) :: owner_pair
        real(wp), dimension(num_fluids_max) :: selected_before, selected_stored, selected_rhs, selected_predicted

        alpha_origin_rk_cause_ready = .false.
        if (.not. f_alpha_origin_active()) return

        global_j_offset = 0
        global_k_offset = 0
        global_l_offset = 0
        if (allocated(start_idx)) then
            if (size(start_idx) >= 1) global_j_offset = start_idx(1)
            if (size(start_idx) >= 2) global_k_offset = start_idx(2)
            if (size(start_idx) >= 3) global_l_offset = start_idx(3)
        end if
        global_width = int(m_glb + 1 + 2*buff_size, kind=8)
        global_height = int(max(1, n_glb + 1 + 2*buff_size), kind=8)
        rhs_factor = merge(1._wp, dt_value, igr)

        local_invalid = 0
        local_bounds_count = 0
        local_sum_count = 0
        local_j0_count = 0
        local_jmax_count = 0
        local_x_boundary_count = 0
        local_interior_count = 0
        local_i_min = huge(1._wp)
        local_i_max = -huge(1._wp)
        local_j_min = huge(1._wp)
        local_j_max = -huge(1._wp)
        local_first_key = huge(local_first_key)
        $:GPU_PARALLEL_LOOP(collapse=3, &
            private='[i,j,k,l,cell_nonfinite,cell_bounds,cell_sum_invalid,alpha_before_value,alpha_stored_value,alpha_rhs_value,alpha_predicted_value,alpha_sum,cell_key]', &
            firstprivate='[global_j_offset,global_k_offset,global_l_offset,global_width,global_height,rk_a,rk_b,rk_c,rk_d,rhs_factor]', &
            reduction='[[local_invalid,local_bounds_count,local_sum_count,local_j0_count,local_jmax_count,local_x_boundary_count,local_interior_count],[local_i_min,local_j_min],[local_i_max,local_j_max],[local_first_key]]', &
            reductionOp='[+,MIN,MAX,MIN]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    alpha_sum = 0._wp
                    cell_nonfinite = .false.
                    cell_bounds = .false.
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        alpha_before_value = q_current(advxb + i - 1)%sf(j, k, l)
                        if (rk_b == 0._wp) then
                            alpha_stored_value = alpha_before_value
                        else
                            alpha_stored_value = q_stored(advxb + i - 1)%sf(j, k, l)
                        end if
                        alpha_rhs_value = rhs(advxb + i - 1)%sf(j, k, l)
                        alpha_predicted_value = real((rk_a*alpha_before_value &
                                                      + rk_b*alpha_stored_value &
                                                      + rk_c*rhs_factor*alpha_rhs_value)/rk_d, kind=stp)
                        if (.not. f_alpha_origin_finite(alpha_predicted_value)) then
                            cell_nonfinite = .true.
                        else
                            alpha_sum = alpha_sum + real(alpha_predicted_value, kind=wp)
                            if (real(alpha_predicted_value, kind=wp) < -1.e-12_wp .or. &
                                real(alpha_predicted_value, kind=wp) > 1._wp + 1.e-12_wp) cell_bounds = .true.
                        end if
                    end do
                    cell_sum_invalid = (.not. cell_nonfinite) .and. abs(alpha_sum - 1._wp) > 1.e-6_wp
                    if (cell_nonfinite .or. cell_bounds .or. cell_sum_invalid) then
                        local_invalid = local_invalid + 1
                        if (cell_bounds) local_bounds_count = local_bounds_count + 1
                        if (cell_sum_invalid) local_sum_count = local_sum_count + 1
                        if (k + global_k_offset == 0) local_j0_count = local_j0_count + 1
                        if (k + global_k_offset == n_glb) local_jmax_count = local_jmax_count + 1
                        if (j + global_j_offset == 0 .or. j + global_j_offset == m_glb) &
                            local_x_boundary_count = local_x_boundary_count + 1
                        if (j + global_j_offset > 0 .and. j + global_j_offset < m_glb .and. &
                            k + global_k_offset > 0 .and. k + global_k_offset < n_glb) &
                            local_interior_count = local_interior_count + 1
                        local_i_min = min(local_i_min, real(j + global_j_offset, kind=wp))
                        local_i_max = max(local_i_max, real(j + global_j_offset, kind=wp))
                        local_j_min = min(local_j_min, real(k + global_k_offset, kind=wp))
                        local_j_max = max(local_j_max, real(k + global_k_offset, kind=wp))
                        cell_key = int(j + global_j_offset + buff_size, kind=8) &
                                   + global_width*int(k + global_k_offset + buff_size, kind=8) &
                                   + global_width*global_height*int(l + global_l_offset + buff_size, kind=8)
                        local_first_key = min(local_first_key, cell_key)
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call s_mpi_allreduce_integer_sum(local_invalid, global_invalid)
        if (global_invalid == 0) return

        call s_mpi_allreduce_integer_sum(local_bounds_count, global_bounds_count)
        call s_mpi_allreduce_integer_sum(local_sum_count, global_sum_count)
        call s_mpi_allreduce_integer_sum(local_j0_count, global_j0_count)
        call s_mpi_allreduce_integer_sum(local_jmax_count, global_jmax_count)
        call s_mpi_allreduce_integer_sum(local_x_boundary_count, global_x_boundary_count)
        call s_mpi_allreduce_integer_sum(local_interior_count, global_interior_count)
        call s_mpi_reduce_min(local_i_min)
        call s_mpi_reduce_min(local_j_min)
        if (num_procs > 1) then
            call s_mpi_allreduce_max_common(local_i_max, global_i_max)
            call s_mpi_allreduce_max_common(local_j_max, global_j_max)
            local_i_max = global_i_max
            local_j_max = global_j_max
        end if

        if (.not. alpha_origin_invalid_region_reported) then
            if (proc_rank == 0) then
                write (output_unit, '(&
                    &"TEMP_ALPHA_INVALID_REGION t_step=", I0, " stage=", I0, &
                    &" checkpoint=POST_RK_CONS_UPDATE_PREDICTED", &
                    &" global_i_min=", I0, " global_i_max=", I0, &
                    &" global_j_min=", I0, " global_j_max=", I0, &
                    &" count_j0=", I0, " count_jmax=", I0, &
                    &" count_x_boundaries=", I0, " count_interior=", I0, &
                    &" count_alpha_bounds=", I0, " count_alpha_sum=", I0, &
                    &" global_invalid=", I0)') &
                    t_step, stage, int(local_i_min), int(local_i_max), int(local_j_min), int(local_j_max), &
                    global_j0_count, global_jmax_count, global_x_boundary_count, global_interior_count, &
                    global_bounds_count, global_sum_count, global_invalid
                call flush(output_unit)
            end if
            alpha_origin_invalid_region_reported = .true.
        end if

        owner_pair = [merge(-real(local_first_key, kind=wp), -huge(1._wp), local_invalid > 0), &
                      real(proc_rank, kind=wp)]
        call s_mpi_reduce_maxloc(owner_pair)
        global_first_key = int(-owner_pair(1), kind=8)
        owner_rank = int(owner_pair(2))

        selected_global_l = int(global_first_key/(global_width*global_height)) - buff_size
        global_first_key = global_first_key &
                           - int(selected_global_l + buff_size, kind=8)*global_width*global_height
        selected_global_k = int(global_first_key/global_width) - buff_size
        selected_global_j = int(global_first_key &
                                - int(selected_global_k + buff_size, kind=8)*global_width) - buff_size
        selected_j = selected_global_j - global_j_offset
        selected_k = selected_global_k - global_k_offset
        selected_l = selected_global_l - global_l_offset

        if (proc_rank == owner_rank) then
            selected_before = 0._wp
            selected_stored = 0._wp
            selected_rhs = 0._wp
            selected_predicted = 0._wp
            $:GPU_PARALLEL_LOOP(firstprivate='[selected_j,selected_k,selected_l,rk_a,rk_b,rk_c,rk_d,rhs_factor]', &
                private='[alpha_before_value,alpha_stored_value,alpha_rhs_value,alpha_predicted_value]', &
                copyout='[selected_before,selected_stored,selected_rhs,selected_predicted]')
            do i = 1, num_fluids
                alpha_before_value = q_current(advxb + i - 1)%sf(selected_j, selected_k, selected_l)
                if (rk_b == 0._wp) then
                    alpha_stored_value = alpha_before_value
                else
                    alpha_stored_value = q_stored(advxb + i - 1)%sf(selected_j, selected_k, selected_l)
                end if
                alpha_rhs_value = rhs(advxb + i - 1)%sf(selected_j, selected_k, selected_l)
                alpha_predicted_value = real((rk_a*alpha_before_value &
                                              + rk_b*alpha_stored_value &
                                              + rk_c*rhs_factor*alpha_rhs_value)/rk_d, kind=stp)
                selected_before(i) = real(alpha_before_value, kind=wp)
                selected_stored(i) = real(alpha_stored_value, kind=wp)
                selected_rhs(i) = real(alpha_rhs_value, kind=wp)
                selected_predicted(i) = real(alpha_predicted_value, kind=wp)
            end do
            $:END_GPU_PARALLEL_LOOP()

            alpha_origin_rk_before(1:num_fluids) = selected_before(1:num_fluids)
            alpha_origin_rk_stored(1:num_fluids) = selected_stored(1:num_fluids)
            alpha_origin_rk_rhs(1:num_fluids) = selected_rhs(1:num_fluids)
            alpha_origin_rk_predicted(1:num_fluids) = selected_predicted(1:num_fluids)
            alpha_origin_rk_rhs_sum = sum(selected_rhs(1:num_fluids))
        end if

        alpha_origin_rk_cause_ready = .true.
        alpha_origin_rk_owner = owner_rank
        alpha_origin_rk_t_step = t_step
        alpha_origin_rk_stage = stage
        alpha_origin_rk_local_ijk = [selected_j, selected_k, selected_l]
        alpha_origin_rk_global_ijk = [selected_global_j, selected_global_k, selected_global_l]
        alpha_origin_rk_a = rk_a
        alpha_origin_rk_b = rk_b
        alpha_origin_rk_c = rk_c
        alpha_origin_rk_d = rk_d
        alpha_origin_rk_dt = dt_value
        alpha_origin_rk_rhs_factor = rhs_factor
        alpha_origin_rk_igr = igr
    end subroutine s_alpha_origin_capture_rk_cause

    subroutine s_alpha_origin_print_rk_cause(q_after, t_step, stage)
        type(scalar_field), dimension(sys_size), intent(in) :: q_after
        integer, intent(in) :: t_step, stage

        integer :: i, selected_j, selected_k, selected_l
        real(wp), dimension(num_fluids_max) :: alpha_after

        if (.not. alpha_origin_rk_cause_ready) return
        if (t_step /= alpha_origin_rk_t_step .or. stage /= alpha_origin_rk_stage) return
        if (proc_rank /= alpha_origin_rk_owner) return

        selected_j = alpha_origin_rk_local_ijk(1)
        selected_k = alpha_origin_rk_local_ijk(2)
        selected_l = alpha_origin_rk_local_ijk(3)
        alpha_after = 0._wp
        $:GPU_PARALLEL_LOOP(firstprivate='[selected_j,selected_k,selected_l]', copyout='[alpha_after]')
        do i = 1, num_fluids
            alpha_after(i) = real(q_after(advxb + i - 1)%sf(selected_j, selected_k, selected_l), kind=wp)
        end do
        $:END_GPU_PARALLEL_LOOP()

        write (output_unit, '(&
            &"TEMP_ALPHA_ORIGIN_RK_CAUSE t_step=", I10, " stage=", I4, &
            &" rank=", I6, " local_ijk=", 3(I8,1X), " global_ijk=", 3(I8,1X), &
            &" rk_a=", ES16.8, " rk_b=", ES16.8, " rk_c=", ES16.8, &
            &" rk_d=", ES16.8, " dt=", ES16.8, " rhs_factor=", ES16.8, &
            &" igr=", L1, " rhs_alpha_sum=", ES16.8, " alpha_before=")', advance='no') &
            t_step, stage, proc_rank, alpha_origin_rk_local_ijk, alpha_origin_rk_global_ijk, &
            alpha_origin_rk_a, alpha_origin_rk_b, alpha_origin_rk_c, alpha_origin_rk_d, &
            alpha_origin_rk_dt, alpha_origin_rk_rhs_factor, alpha_origin_rk_igr, &
            alpha_origin_rk_rhs_sum
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_origin_rk_before(i)
        end do
        write (output_unit, '(" alpha_stored=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_origin_rk_stored(i)
        end do
        write (output_unit, '(" alpha_rhs=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_origin_rk_rhs(i)
        end do
        write (output_unit, '(" alpha_predicted=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_origin_rk_predicted(i)
        end do
        write (output_unit, '(" alpha_after=")', advance='no')
        do i = 1, num_fluids
            write (output_unit, '(1X,ES16.8)', advance='no') alpha_after(i)
        end do
        write (output_unit, *)
        call flush(output_unit)
    end subroutine s_alpha_origin_print_rk_cause

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

    !> The computation of parameters, the allocation of memory,
        !!      the association of pointers and/or the execution of any
        !!      other procedures that are necessary to setup the module.
    impure subroutine s_initialize_rhs_module

        integer :: i, j, k, l, id !< Generic loop iterators

        $:GPU_ENTER_DATA(copyin='[idwbuff]')
        $:GPU_UPDATE(device='[idwbuff]')

        @:ALLOCATE(q_cons_qp%vf(1:sys_size))
        @:ALLOCATE(q_prim_qp%vf(1:sys_size))

        if (.not. igr) then
            do l = 1, sys_size
                @:ALLOCATE(q_cons_qp%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
            end do
            do l = mom_idx%beg, E_idx
                @:ALLOCATE(q_prim_qp%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
            end do

        end if

        if (surface_tension) then
            ! This assumes that the color function advection equation is
            ! the last equation. If this changes then this logic will
            ! need updated
            do l = adv_idx%end + 1, sys_size - 1
                @:ALLOCATE(q_prim_qp%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
            end do
        else
            do l = adv_idx%end + 1, sys_size
                @:ALLOCATE(q_prim_qp%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
            end do

        end if

        if (.not. igr) then
            @:ACC_SETUP_VFs(q_cons_qp, q_prim_qp)

            do l = 1, cont_idx%end
                if (relativity) then
                    ! Cons and Prim densities are different for relativity
                    @:ALLOCATE(q_prim_qp%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
                else
                    q_prim_qp%vf(l)%sf => q_cons_qp%vf(l)%sf
                    $:GPU_ENTER_DATA(copyin='[q_prim_qp%vf(l)%sf]')
                    $:GPU_ENTER_DATA(attach='[q_prim_qp%vf(l)%sf]')
                end if
            end do

            do l = adv_idx%beg, adv_idx%end
                q_prim_qp%vf(l)%sf => q_cons_qp%vf(l)%sf
                $:GPU_ENTER_DATA(copyin='[q_prim_qp%vf(l)%sf]')
                $:GPU_ENTER_DATA(attach='[q_prim_qp%vf(l)%sf]')
            end do
        end if

        if (surface_tension) then
            q_prim_qp%vf(c_idx)%sf => &
                q_cons_qp%vf(c_idx)%sf
            $:GPU_ENTER_DATA(copyin='[q_prim_qp%vf(c_idx)%sf]')
            $:GPU_ENTER_DATA(attach='[q_prim_qp%vf(c_idx)%sf]')
        end if

        if (hyper_cleaning) then
            q_prim_qp%vf(psi_idx)%sf => &
                q_cons_qp%vf(psi_idx)%sf
            $:GPU_ENTER_DATA(copyin='[q_prim_qp%vf(psi_idx)%sf]')
            $:GPU_ENTER_DATA(attach='[q_prim_qp%vf(psi_idx)%sf]')
        end if

        ! Allocation/Association of flux_n, flux_src_n, and flux_gsrc_n
        if (.not. igr) then
            @:ALLOCATE(flux_n(1:num_dims))
            @:ALLOCATE(flux_src_n(1:num_dims))
            @:ALLOCATE(flux_gsrc_n(1:num_dims))

            do i = 1, num_dims

                @:ALLOCATE(flux_n(i)%vf(1:sys_size))
                @:ALLOCATE(flux_src_n(i)%vf(1:sys_size))
                @:ALLOCATE(flux_gsrc_n(i)%vf(1:sys_size))

                if (i == 1) then
                    do l = 1, sys_size
                        @:ALLOCATE(flux_n(i)%vf(l)%sf( &
                                 & idwbuff(1)%beg:idwbuff(1)%end, &
                                 & idwbuff(2)%beg:idwbuff(2)%end, &
                                 & idwbuff(3)%beg:idwbuff(3)%end))
                        @:ALLOCATE(flux_gsrc_n(i)%vf(l)%sf( &
                                & idwbuff(1)%beg:idwbuff(1)%end, &
                                & idwbuff(2)%beg:idwbuff(2)%end, &
                                & idwbuff(3)%beg:idwbuff(3)%end))
                    end do

                    if (viscous .or. surface_tension) then
                        do l = mom_idx%beg, E_idx
                            @:ALLOCATE(flux_src_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                    end if

                    @:ALLOCATE(flux_src_n(i)%vf(adv_idx%beg)%sf( &
                             & idwbuff(1)%beg:idwbuff(1)%end, &
                             & idwbuff(2)%beg:idwbuff(2)%end, &
                             & idwbuff(3)%beg:idwbuff(3)%end))

                    if (riemann_solver == 1 .or. riemann_solver == 4) then
                        do l = adv_idx%beg + 1, adv_idx%end
                            @:ALLOCATE(flux_src_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                    end if

                    if (chemistry) then
                        do l = chemxb, chemxe
                            @:ALLOCATE(flux_src_n(i)%vf(l)%sf( &
                                    & idwbuff(1)%beg:idwbuff(1)%end, &
                                    & idwbuff(2)%beg:idwbuff(2)%end, &
                                    & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                        if (chem_params%diffusion .and. .not. viscous) then
                            @:ALLOCATE(flux_src_n(i)%vf(E_idx)%sf( &
                                    & idwbuff(1)%beg:idwbuff(1)%end, &
                                    & idwbuff(2)%beg:idwbuff(2)%end, &
                                    & idwbuff(3)%beg:idwbuff(3)%end))
                        end if
                    end if

                else
                    do l = 1, sys_size
                        @:ALLOCATE(flux_gsrc_n(i)%vf(l)%sf( &
                            idwbuff(1)%beg:idwbuff(1)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, &
                            idwbuff(3)%beg:idwbuff(3)%end))
                    end do
                end if

                @:ACC_SETUP_VFs(flux_n(i))
                @:ACC_SETUP_VFs(flux_src_n(i), flux_gsrc_n(i))

                if (i == 1) then
                    if (riemann_solver /= 1) then
                        do l = adv_idx%beg + 1, adv_idx%end
                            flux_src_n(i)%vf(l)%sf => flux_src_n(i)%vf(adv_idx%beg)%sf
                            $:GPU_ENTER_DATA(attach='[flux_src_n(i)%vf(l)%sf]')
                        end do
                    end if
                else
                    do l = 1, sys_size
                        flux_n(i)%vf(l)%sf => flux_n(1)%vf(l)%sf
                        $:GPU_ENTER_DATA(attach='[flux_n(i)%vf(l)%sf]')
                        flux_src_n(i)%vf(l)%sf => flux_src_n(1)%vf(l)%sf
                        $:GPU_ENTER_DATA(attach='[flux_src_n(i)%vf(l)%sf]')
                    end do
                end if

            end do
            ! END: Allocation/Association of flux_n, flux_src_n, and flux_gsrc_n
        end if

        if ((.not. igr) .or. dummy) then

            ! Allocation of dq_prim_ds_qp
            @:ALLOCATE(dq_prim_dx_qp(1:1))
            @:ALLOCATE(dq_prim_dy_qp(1:1))
            @:ALLOCATE(dq_prim_dz_qp(1:1))

            @:ALLOCATE(qL_prim(1:num_dims))
            @:ALLOCATE(qR_prim(1:num_dims))

            ! Allocation/Association of dqK_prim_ds_n
            @:ALLOCATE(dqL_prim_dx_n(1:num_dims))
            @:ALLOCATE(dqL_prim_dy_n(1:num_dims))
            @:ALLOCATE(dqL_prim_dz_n(1:num_dims))
            @:ALLOCATE(dqR_prim_dx_n(1:num_dims))
            @:ALLOCATE(dqR_prim_dy_n(1:num_dims))
            @:ALLOCATE(dqR_prim_dz_n(1:num_dims))

            do i = 1, num_dims
                @:ALLOCATE(qL_prim(i)%vf(1:sys_size))
                @:ALLOCATE(qR_prim(i)%vf(1:sys_size))
                do l = mom_idx%beg, mom_idx%end
                    @:ALLOCATE(qL_prim(i)%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
                    @:ALLOCATE(qR_prim(i)%vf(l)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
                end do
                @:ACC_SETUP_VFs(qL_prim(i), qR_prim(i))
            end do

            @:ALLOCATE(qL_rsx_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
            @:ALLOCATE(qR_rsx_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))

            if (n > 0) then

                @:ALLOCATE(qL_rsy_vf(idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(1)%beg:idwbuff(1)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
                @:ALLOCATE(qR_rsy_vf(idwbuff(2)%beg:idwbuff(2)%end, &
                    idwbuff(1)%beg:idwbuff(1)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
            else
                @:ALLOCATE(qL_rsy_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
                @:ALLOCATE(qR_rsy_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
            end if

            if (p > 0) then
                @:ALLOCATE(qL_rsz_vf(idwbuff(3)%beg:idwbuff(3)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(1)%beg:idwbuff(1)%end, 1:sys_size))
                @:ALLOCATE(qR_rsz_vf(idwbuff(3)%beg:idwbuff(3)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(1)%beg:idwbuff(1)%end, 1:sys_size))
            else
                @:ALLOCATE(qL_rsz_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))
                @:ALLOCATE(qR_rsz_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                    idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, 1:sys_size))

            end if

            if (.not. viscous) then
                do i = 1, num_dims
                    @:ALLOCATE(dqL_prim_dx_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqL_prim_dy_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqL_prim_dz_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dx_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dy_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dz_n(i)%vf(1:sys_size))

                    do l = momxb, momxe
                        @:ALLOCATE(dqL_prim_dx_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                        @:ALLOCATE(dqL_prim_dy_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                        @:ALLOCATE(dqL_prim_dz_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                        @:ALLOCATE(dqR_prim_dx_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                        @:ALLOCATE(dqR_prim_dy_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                        @:ALLOCATE(dqR_prim_dz_n(i)%vf(l)%sf(1:1, 1:1, 1:1))
                    end do
                    @:ACC_SETUP_VFs(dqL_prim_dx_n(i), dqL_prim_dy_n(i), dqL_prim_dz_n(i))
                    @:ACC_SETUP_VFs(dqR_prim_dx_n(i), dqR_prim_dy_n(i), dqR_prim_dz_n(i))
                end do
            end if

            if (viscous) then

                @:ALLOCATE(tau_Re_vf(1:sys_size))
                do i = 1, num_dims
                    @:ALLOCATE(tau_Re_vf(cont_idx%end + i)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                                                        &  idwbuff(2)%beg:idwbuff(2)%end, &
                                                        &  idwbuff(3)%beg:idwbuff(3)%end))
                    @:ACC_SETUP_SFs(tau_Re_vf(cont_idx%end + i))
                end do
                @:ALLOCATE(tau_Re_vf(E_idx)%sf(idwbuff(1)%beg:idwbuff(1)%end, &
                                            & idwbuff(2)%beg:idwbuff(2)%end, &
                                            & idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(tau_Re_vf(E_idx))

                @:ALLOCATE(dq_prim_dx_qp(1)%vf(1:sys_size))
                @:ALLOCATE(dq_prim_dy_qp(1)%vf(1:sys_size))
                @:ALLOCATE(dq_prim_dz_qp(1)%vf(1:sys_size))

                do l = mom_idx%beg, mom_idx%end
                    @:ALLOCATE(dq_prim_dx_qp(1)%vf(l)%sf( &
                              & idwbuff(1)%beg:idwbuff(1)%end, &
                              & idwbuff(2)%beg:idwbuff(2)%end, &
                              & idwbuff(3)%beg:idwbuff(3)%end))
                end do

                @:ACC_SETUP_VFs(dq_prim_dx_qp(1))

                if (n > 0) then

                    do l = mom_idx%beg, mom_idx%end
                        @:ALLOCATE(dq_prim_dy_qp(1)%vf(l)%sf( &
                                 & idwbuff(1)%beg:idwbuff(1)%end, &
                                 & idwbuff(2)%beg:idwbuff(2)%end, &
                                 & idwbuff(3)%beg:idwbuff(3)%end))
                    end do

                    @:ACC_SETUP_VFs(dq_prim_dy_qp(1))

                    if (p > 0) then

                        do l = mom_idx%beg, mom_idx%end
                            @:ALLOCATE(dq_prim_dz_qp(1)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                        @:ACC_SETUP_VFs(dq_prim_dz_qp(1))
                    end if

                end if

                do i = 1, num_dims
                    @:ALLOCATE(dqL_prim_dx_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqL_prim_dy_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqL_prim_dz_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dx_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dy_n(i)%vf(1:sys_size))
                    @:ALLOCATE(dqR_prim_dz_n(i)%vf(1:sys_size))
                end do

                do i = 1, num_dims

                    do l = mom_idx%beg, mom_idx%end
                        @:ALLOCATE(dqL_prim_dx_n(i)%vf(l)%sf( &
                                 & idwbuff(1)%beg:idwbuff(1)%end, &
                                 & idwbuff(2)%beg:idwbuff(2)%end, &
                                 & idwbuff(3)%beg:idwbuff(3)%end))
                        @:ALLOCATE(dqR_prim_dx_n(i)%vf(l)%sf( &
                                 & idwbuff(1)%beg:idwbuff(1)%end, &
                                 & idwbuff(2)%beg:idwbuff(2)%end, &
                                 & idwbuff(3)%beg:idwbuff(3)%end))
                    end do

                    if (n > 0) then
                        do l = mom_idx%beg, mom_idx%end
                            @:ALLOCATE(dqL_prim_dy_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                            @:ALLOCATE(dqR_prim_dy_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                    end if

                    if (p > 0) then
                        do l = mom_idx%beg, mom_idx%end
                            @:ALLOCATE(dqL_prim_dz_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                            @:ALLOCATE(dqR_prim_dz_n(i)%vf(l)%sf( &
                                     & idwbuff(1)%beg:idwbuff(1)%end, &
                                     & idwbuff(2)%beg:idwbuff(2)%end, &
                                     & idwbuff(3)%beg:idwbuff(3)%end))
                        end do
                    end if

                    @:ACC_SETUP_VFs(dqL_prim_dx_n(i), dqL_prim_dy_n(i), dqL_prim_dz_n(i))
                    @:ACC_SETUP_VFs(dqR_prim_dx_n(i), dqR_prim_dy_n(i), dqR_prim_dz_n(i))
                end do

                if (weno_Re_flux) then
                    @:ALLOCATE(dqL_rsx_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                    @:ALLOCATE(dqR_rsx_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                        idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))

                    if (n > 0) then
                        @:ALLOCATE(dqL_rsy_vf(idwbuff(2)%beg:idwbuff(2)%end, &
                            idwbuff(1)%beg:idwbuff(1)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                        @:ALLOCATE(dqR_rsy_vf(idwbuff(2)%beg:idwbuff(2)%end, &
                            idwbuff(1)%beg:idwbuff(1)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                    else
                        @:ALLOCATE(dqL_rsy_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                        @:ALLOCATE(dqR_rsy_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                    end if

                    if (p > 0) then
                        @:ALLOCATE(dqL_rsz_vf(idwbuff(3)%beg:idwbuff(3)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(1)%beg:idwbuff(1)%end, mom_idx%beg:mom_idx%end))
                        @:ALLOCATE(dqR_rsz_vf(idwbuff(3)%beg:idwbuff(3)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(1)%beg:idwbuff(1)%end, mom_idx%beg:mom_idx%end))
                    else
                        @:ALLOCATE(dqL_rsz_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                        @:ALLOCATE(dqR_rsz_vf(idwbuff(1)%beg:idwbuff(1)%end, &
                            idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end, mom_idx%beg:mom_idx%end))
                    end if
                end if ! end allocation for weno_Re_flux

            else
                @:ALLOCATE(dq_prim_dx_qp(1)%vf(1:sys_size))
                @:ALLOCATE(dq_prim_dy_qp(1)%vf(1:sys_size))
                @:ALLOCATE(dq_prim_dz_qp(1)%vf(1:sys_size))

                do l = momxb, momxe
                    @:ALLOCATE(dq_prim_dx_qp(1)%vf(l)%sf(0, 0, 0))
                    @:ACC_SETUP_VFs(dq_prim_dx_qp(1))
                    if (n > 0) then
                        @:ALLOCATE(dq_prim_dy_qp(1)%vf(l)%sf(0, 0, 0))
                        @:ACC_SETUP_VFs(dq_prim_dy_qp(1))
                        if (p > 0) then
                            @:ALLOCATE(dq_prim_dz_qp(1)%vf(l)%sf(0, 0, 0))
                            @:ACC_SETUP_VFs(dq_prim_dz_qp(1))
                        end if
                    end if
                end do
            end if ! end allocation of viscous variables

            $:GPU_PARALLEL_LOOP(private='[i,j,k,l,id]', collapse=4)
            do id = 1, num_dims
                do i = 1, sys_size
                    do l = idwbuff(3)%beg, idwbuff(3)%end
                        do k = idwbuff(2)%beg, idwbuff(2)%end
                            do j = idwbuff(1)%beg, idwbuff(1)%end
                                flux_gsrc_n(id)%vf(i)%sf(j, k, l) = 0._wp
                            end do
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

        end if ! end allocation for .not. igr

        if (qbmm) then
            @:ALLOCATE(mom_sp(1:nmomsp), mom_3d(0:2, 0:2, nb))

            do i = 0, 2
                do j = 0, 2
                    do k = 1, nb
                        @:ALLOCATE(mom_3d(i, j, k)%sf( &
                                      & idwbuff(1)%beg:idwbuff(1)%end, &
                                      & idwbuff(2)%beg:idwbuff(2)%end, &
                                      & idwbuff(3)%beg:idwbuff(3)%end))
                        @:ACC_SETUP_SFs(mom_3d(i, j, k))
                    end do
                end do
            end do

            do i = 1, nmomsp
                @:ALLOCATE(mom_sp(i)%sf( &
                        & idwbuff(1)%beg:idwbuff(1)%end, &
                        & idwbuff(2)%beg:idwbuff(2)%end, &
                        & idwbuff(3)%beg:idwbuff(3)%end))
                @:ACC_SETUP_SFs(mom_sp(i))
            end do
        end if

        if (mpp_lim .and. bubbles_euler) then
            @:ALLOCATE(alf_sum%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
        end if
        ! END: Allocation/Association of qK_cons_n and qK_prim_n

        ! Allocation of gm_alphaK_n
        if (.not. igr) then
            @:ALLOCATE(gm_alphaL_n(1:num_dims))
            @:ALLOCATE(gm_alphaR_n(1:num_dims))
        end if

        if (alt_soundspeed) then
            @:ALLOCATE(blkmod1(0:m, 0:n, 0:p), blkmod2(0:m, 0:n, 0:p), alpha1(0:m, 0:n, 0:p), alpha2(0:m, 0:n, 0:p), Kterm(0:m, 0:n, 0:p))
        end if

        call s_initialize_pressure_relaxation_module

        if (bubbles_euler) then
            @:ALLOCATE(nbub(0:m, 0:n, 0:p))
        end if

    end subroutine s_initialize_rhs_module

    !> @brief Computes the right-hand side of the semi-discrete governing equations for a single time stage.
    impure subroutine s_compute_rhs(q_cons_vf, q_T_sf, q_prim_vf, bc_type, rhs_vf, pb_in, rhs_pb, mv_in, rhs_mv, t_step, time_avg, stage)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        type(scalar_field), intent(inout) :: q_T_sf
        type(scalar_field), dimension(sys_size), intent(inout) :: q_prim_vf
        type(integer_field), dimension(1:num_dims, 1:2), intent(in) :: bc_type
        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        real(stp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:, 1:), intent(inout) :: pb_in
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:, 1:), intent(inout) :: rhs_pb  ! TODO :: I think these other two variables need to be stp as well, but it doesn't compile like that right now
        real(stp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:, 1:), intent(inout) :: mv_in
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:, 1:), intent(inout) :: rhs_mv
        integer, intent(in) :: t_step
        real(wp), intent(inout) :: time_avg
        integer, intent(in) :: stage

        real(wp) :: t_start, t_finish
        real(wp) :: alpha_liq, gas_alpha, gas_mass
        integer :: id
        integer :: user_species_eqn
        integer :: fuel_species_eqn
        integer :: liquid_alpha_eqn
        integer(kind=8) :: i, j, k, l, q !< Generic loop iterators

        call nvtxStartRange("COMPUTE-RHS")
        call s_zhang_evap_hang_trace(t_step, stage, "S_COMPUTE_RHS_BEGIN")

        call cpu_time(t_start)

        if (.not. igr .or. dummy) then
            ! Association/Population of Working Variables
            $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
            do i = 1, sys_size
                do l = idwbuff(3)%beg, idwbuff(3)%end
                    do k = idwbuff(2)%beg, idwbuff(2)%end
                        do j = idwbuff(1)%beg, idwbuff(1)%end
                            q_cons_qp%vf(i)%sf(j, k, l) = q_cons_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            ! Converting Conservative to Primitive Variables

            if (mpp_lim .and. bubbles_euler) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = idwbuff(3)%beg, idwbuff(3)%end
                    do k = idwbuff(2)%beg, idwbuff(2)%end
                        do j = idwbuff(1)%beg, idwbuff(1)%end
                            alf_sum%sf(j, k, l) = 0._wp
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = advxb, advxe - 1
                                alf_sum%sf(j, k, l) = alf_sum%sf(j, k, l) + q_cons_qp%vf(i)%sf(j, k, l)
                            end do
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = advxb, advxe - 1
                                q_cons_qp%vf(i)%sf(j, k, l) = q_cons_qp%vf(i)%sf(j, k, l)*(1._wp - q_cons_qp%vf(alf_idx)%sf(j, k, l)) &
                                                              /alf_sum%sf(j, k, l)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

        if (.not. igr .or. dummy) then
            call s_alpha_origin_checkpoint(q_cons_qp%vf, t_step, stage, &
                                           "RHS_WORK_CONS_PRE_C2P", "owned")
        end if

        if (igr .or. dummy) then
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_COMMUNICATION_CONS_BEGIN")
            call nvtxStartRange("RHS-COMMUNICATION")
            call s_populate_variables_buffers(bc_type, q_cons_vf, pb_in, mv_in)
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_COMMUNICATION_CONS_END")
        end if
        if (.not. igr .or. dummy) then
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_CONS_TO_PRIM_BEGIN")
            call nvtxStartRange("RHS-CONVERT")
            call s_convert_conservative_to_primitive_variables( &
                q_cons_qp%vf, &
                q_T_sf, &
                q_prim_qp%vf, &
                idwint, &
                t_step, &
                stage, &
                "RHS")
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_CONS_TO_PRIM_END")
            call s_alpha_origin_checkpoint(q_prim_qp%vf, t_step, stage, &
                                           "RHS_PRIM_PRE_HALO", "owned")
            call s_alpha_origin_checkpoint(q_prim_qp%vf, t_step, stage, &
                                           "RHS_PRIM_PRE_HALO", "ghost")

            call s_zhang_evap_hang_trace(t_step, stage, "RHS_COMMUNICATION_PRIM_BEGIN")
            call nvtxStartRange("RHS-COMMUNICATION")
            call s_populate_variables_buffers(bc_type, q_prim_qp%vf, pb_in, mv_in, &
                                              t_step, stage, "RHS_PRIM")
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_COMMUNICATION_PRIM_END")
            call s_alpha_origin_checkpoint(q_prim_qp%vf, t_step, stage, &
                                           "RHS_PRIM_POST_HALO", "owned")
            call s_alpha_origin_checkpoint(q_prim_qp%vf, t_step, stage, &
                                           "RHS_PRIM_POST_HALO", "ghost")
        end if

        call nvtxStartRange("RHS-ELASTIC")
        if (hyperelasticity) call s_hyperelastic_rmt_stress_update(q_cons_qp%vf, q_prim_qp%vf)
        call nvtxEndRange

        if (cfl_dt) then
            if (mytime >= t_stop) return
        else
            if (t_step == t_step_stop) return
        end if

        if (qbmm) call s_mom_inv(q_cons_qp%vf, q_prim_qp%vf, mom_sp, mom_3d, pb_in, rhs_pb, mv_in, rhs_mv, idwbuff(1), idwbuff(2), idwbuff(3))

        if ((viscous .and. .not. igr) .or. dummy) then
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_VISCOUS_BEGIN")
            call nvtxStartRange("RHS-VISCOUS")
            call s_get_viscous(qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                               dqL_prim_dx_n, dqL_prim_dy_n, dqL_prim_dz_n, &
                               qL_prim, &
                               qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                               dqR_prim_dx_n, dqR_prim_dy_n, dqR_prim_dz_n, &
                               qR_prim, &
                               q_prim_qp, &
                               dq_prim_dx_qp, dq_prim_dy_qp, dq_prim_dz_qp, &
                               idwbuff(1), idwbuff(2), idwbuff(3))
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_VISCOUS_END")
        end if

        if (surface_tension) then
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_SURFACE_TENSION_BEGIN")
            call nvtxStartRange("RHS-SURFACE-TENSION")
            call s_get_capillary(q_prim_qp%vf, bc_type)
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_SURFACE_TENSION_END")
        end if

        if (.not. igr .or. dummy) then
            call s_alpha_rhs_target_state(q_cons_qp%vf, q_prim_qp%vf, bc_type, t_step, stage)
            call s_alpha_rhs_term(rhs_vf, "INITIAL_ZERO", t_step, stage, .true.)
        end if

        ! Dimensional Splitting Loop
        do id = 1, num_dims
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_DIMENSION_BEGIN")

            if (igr .or. dummy) then

                if (id == 1) then
                    $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
                    do l = -1, p + 1
                        do k = -1, n + 1
                            do j = -1, m + 1
                                do i = 1, sys_size
                                    rhs_vf(i)%sf(j, k, l) = 0._stp
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                end if

                call nvtxStartRange("IGR_RIEMANN")
                call s_igr_riemann_solver(q_cons_vf, rhs_vf, id)
                call nvtxEndRange

                if (id == 1) then
                    call nvtxStartRange("IGR_Jacobi")
                    call s_igr_iterative_solve(q_cons_vf, bc_type, t_step)
                    call nvtxEndRange

                    call nvtxStartRange("IGR_SIGMA")
                    call s_igr_sigma_x(q_cons_vf, rhs_vf)
                    call nvtxEndRange
                end if
            end if
            if ((.not. igr) .or. dummy) then! Finite volume solve

                ! Reconstructing Primitive/Conservative Variables
                call nvtxStartRange("RHS-WENO")

                if (.not. surface_tension) then
                    if (all(Re_size == 0)) then
                        ! Reconstruct densitiess
                        iv%beg = 1; iv%end = sys_size
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(1:sys_size), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)
                    else
                        iv%beg = 1; iv%end = contxe
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)

                        iv%beg = E_idx; iv%end = sys_size
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)
                    end if

                else
                    if (all(Re_size == 0)) then
                        iv%beg = 1; iv%end = E_idx - 1
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)

                        iv%beg = E_idx; iv%end = E_idx
                        call s_reconstruct_cell_boundary_values_first_order( &
                            q_prim_qp%vf(E_idx), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)

                        iv%beg = E_idx + 1; iv%end = sys_size
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)
                    else
                        iv%beg = 1; iv%end = contxe
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)

                        iv%beg = E_idx; iv%end = E_idx
                        call s_reconstruct_cell_boundary_values_first_order( &
                            q_prim_qp%vf(E_idx), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)

                        iv%beg = E_idx + 1; iv%end = sys_size
                        call s_reconstruct_cell_boundary_values( &
                            q_prim_qp%vf(iv%beg:iv%end), &
                            qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                            qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                            id)
                    end if

                end if

                ! Reconstruct viscous derivatives for viscosity
                if (weno_Re_flux) then
                    iv%beg = momxb; iv%end = momxe
                    call s_reconstruct_cell_boundary_values_visc_deriv( &
                        dq_prim_dx_qp(1)%vf(iv%beg:iv%end), &
                        dqL_rsx_vf, dqL_rsy_vf, dqL_rsz_vf, &
                        dqR_rsx_vf, dqR_rsy_vf, dqR_rsz_vf, &
                        id, dqL_prim_dx_n(id)%vf(iv%beg:iv%end), dqR_prim_dx_n(id)%vf(iv%beg:iv%end), &
                        idwbuff(1), idwbuff(2), idwbuff(3))
                    if (n > 0) then
                        call s_reconstruct_cell_boundary_values_visc_deriv( &
                            dq_prim_dy_qp(1)%vf(iv%beg:iv%end), &
                            dqL_rsx_vf, dqL_rsy_vf, dqL_rsz_vf, &
                            dqR_rsx_vf, dqR_rsy_vf, dqR_rsz_vf, &
                            id, dqL_prim_dy_n(id)%vf(iv%beg:iv%end), dqR_prim_dy_n(id)%vf(iv%beg:iv%end), &
                            idwbuff(1), idwbuff(2), idwbuff(3))
                        if (p > 0) then
                            call s_reconstruct_cell_boundary_values_visc_deriv( &
                                dq_prim_dz_qp(1)%vf(iv%beg:iv%end), &
                                dqL_rsx_vf, dqL_rsy_vf, dqL_rsz_vf, &
                                dqR_rsx_vf, dqR_rsy_vf, dqR_rsz_vf, &
                                id, dqL_prim_dz_n(id)%vf(iv%beg:iv%end), dqR_prim_dz_n(id)%vf(iv%beg:iv%end), &
                                idwbuff(1), idwbuff(2), idwbuff(3))
                        end if
                    end if
                end if

                call nvtxEndRange ! WENO

                ! Configuring Coordinate Direction Indexes
                if (id == 1) then
                    irx%beg = -1; iry%beg = 0; irz%beg = 0
                elseif (id == 2) then
                    irx%beg = 0; iry%beg = -1; irz%beg = 0
                else
                    irx%beg = 0; iry%beg = 0; irz%beg = -1
                end if
                irx%end = m; iry%end = n; irz%end = p
                ! $:GPU_UPDATE(host='[qL_rsx_vf,qR_rsx_vf]')
                ! print *, "L", qL_rsx_vf(100:300, 0, 0, 1)
                ! print *, "R", qR_rsx_vf(100:300, 0, 0, 1)

                !Computing Riemann Solver Flux and Source Flux
                call nvtxStartRange("RHS-RIEMANN-SOLVER")
                call s_riemann_solver(qR_rsx_vf, qR_rsy_vf, qR_rsz_vf, &
                                      dqR_prim_dx_n(id)%vf, &
                                      dqR_prim_dy_n(id)%vf, &
                                      dqR_prim_dz_n(id)%vf, &
                                      qR_prim(id)%vf, &
                                      qL_rsx_vf, qL_rsy_vf, qL_rsz_vf, &
                                      dqL_prim_dx_n(id)%vf, &
                                      dqL_prim_dy_n(id)%vf, &
                                      dqL_prim_dz_n(id)%vf, &
                                      qL_prim(id)%vf, &
                                      q_prim_qp%vf, &
                                      flux_n(id)%vf, &
                                      flux_src_n(id)%vf, &
                                      flux_gsrc_n(id)%vf, &
                                      id, irx, iry, irz)
                call nvtxEndRange

                !$:GPU_UPDATE(host='[flux_n(1)%vf(1)%sf]')
                !print *, "FLUX", flux_n(1)%vf(1)%sf(100:300, 0, 0)

                ! Additional physics and source terms
                ! RHS addition for advection source
                call nvtxStartRange("RHS-ADVECTION-SRC")
                call s_compute_advection_source_term(id, &
                                                     rhs_vf, &
                                                     q_cons_qp, &
                                                     q_prim_qp, &
                                                     flux_src_n(id), &
                                                     t_step, stage)
                call nvtxEndRange

                ! RHS additions for hypoelasticity
                call nvtxStartRange("RHS-HYPOELASTICITY")
                if (hypoelasticity) call s_compute_hypoelastic_rhs(id, &
                                                                   q_prim_qp%vf, &
                                                                   rhs_vf)
                call nvtxEndRange

                ! RHS for diffusion
                if (chemistry .and. chem_params%diffusion) then
                    call nvtxStartRange("RHS-CHEM-DIFFUSION")
                    call s_compute_chemistry_diffusion_flux(id, q_prim_qp%vf, flux_src_n(id)%vf, irx, iry, irz)
                    call nvtxEndRange
                end if

                ! RHS additions for viscosity
                if (viscous .or. surface_tension .or. chem_params%diffusion) then
                    call nvtxStartRange("RHS-ADD-PHYSICS")
                    call s_compute_additional_physics_rhs(id, &
                                                          q_prim_qp%vf, &
                                                          rhs_vf, &
                                                          flux_src_n(id)%vf, &
                                                          dq_prim_dx_qp(1)%vf, &
                                                          dq_prim_dy_qp(1)%vf, &
                                                          dq_prim_dz_qp(1)%vf)
                    call nvtxEndRange
                end if

                ! RHS additions for sub-grid bubbles_euler
                if (bubbles_euler) then
                    call nvtxStartRange("RHS-BUBBLES-COMPUTE")
                    call s_compute_bubbles_EE_rhs(id, q_prim_qp%vf, divu)
                    call nvtxEndRange
                end if

                ! RHS additions for qbmm bubbles
                if (qbmm) then
                    call nvtxStartRange("RHS-QBMM")
                    call s_compute_qbmm_rhs(id, &
                                            q_cons_qp%vf, &
                                            q_prim_qp%vf, &
                                            rhs_vf, &
                                            flux_n(id)%vf, &
                                            pb_in, &
                                            rhs_pb)
                    call nvtxEndRange
                end if
                ! END: Additional physics and source terms

                if (hyper_cleaning) then
                    $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                    do l = 0, p
                        do k = 0, n
                            do j = 0, m
                                rhs_vf(psi_idx)%sf(j, k, l) = rhs_vf(psi_idx)%sf(j, k, l) - &
                                                              q_prim_vf(psi_idx)%sf(j, k, l)/hyper_cleaning_tau
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                end if

                if (id == 1) then
                    call s_alpha_rhs_term(rhs_vf, "X_AFTER_MULTIPHASE", t_step, stage)
                else if (id == 2) then
                    call s_alpha_rhs_term(rhs_vf, "Y_AFTER_MULTIPHASE", t_step, stage)
                else if (id == 3) then
                    call s_alpha_rhs_term(rhs_vf, "Z_AFTER_MULTIPHASE", t_step, stage)
                end if

                ! END: Additional physics and source terms
            end if
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_DIMENSION_END")
        end do
        ! END: Dimensional Splitting Loop

        if (ib) then
            $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
            do l = 0, p
                do k = 0, n
                    do j = 0, m
                        if (ib_markers%sf(j, k, l) /= 0) then
                            do i = 1, sys_size
                                rhs_vf(i)%sf(j, k, l) = 0._wp
                            end do
                        end if
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        ! Additional Physics and Source Temrs
        ! Additions for acoustic_source
        if (acoustic_source) then
            call nvtxStartRange("RHS-ACOUSTIC-SRC")
            call s_acoustic_src_calculations(q_cons_qp%vf(1:sys_size), &
                                             q_prim_qp%vf(1:sys_size), &
                                             t_step, &
                                             rhs_vf)
            call nvtxEndRange
        end if

        ! Add bubbles source term
        if (bubbles_euler .and. (.not. adap_dt) .and. (.not. qbmm)) then
            call nvtxStartRange("RHS-BUBBLES-SRC")
            call s_compute_bubble_EE_source( &
                q_cons_qp%vf(1:sys_size), &
                q_prim_qp%vf(1:sys_size), &
                rhs_vf, &
                divu)
            call nvtxEndRange
        end if

        if (bubbles_lagrange) then
            ! RHS additions for sub-grid bubbles_lagrange
            call nvtxStartRange("RHS-EL-BUBBLES-SRC")
            call s_compute_bubbles_EL_source( &
                q_cons_qp%vf(1:sys_size), &
                q_prim_qp%vf(1:sys_size), &
                rhs_vf)
            call nvtxEndRange
            ! Compute bubble dynamics
            if (.not. adap_dt) then
                call nvtxStartRange("RHS-EL-BUBBLES-DYN")
                call s_compute_bubble_EL_dynamics( &
                    q_prim_qp%vf(1:sys_size), &
                    stage)
                call nvtxEndRange
            end if
        end if

        call s_alpha_rhs_term(rhs_vf, "AFTER_OTHER_SOURCES", t_step, stage)

        if (chemistry .and. chem_params%reactions) then
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_CHEM_REACTIONS_BEGIN")
            call nvtxStartRange("RHS-CHEM-REACTIONS")
            call s_compute_chemistry_reaction_flux(rhs_vf, q_cons_qp%vf, q_T_sf, q_prim_qp%vf, idwint, t_step, stage)
            call nvtxEndRange
            call s_zhang_evap_hang_trace(t_step, stage, "RHS_CHEM_REACTIONS_END")
        end if

        if (chemistry .and. user_species_source) then
            if (user_species_id >= 1 .and. user_species_id <= (chemxe - chemxb + 1)) then
                user_species_eqn = chemxb + user_species_id - 1
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            rhs_vf(user_species_eqn)%sf(j, k, l) = rhs_vf(user_species_eqn)%sf(j, k, l) + user_species_src
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

        if (chemistry .and. evap_species_source) then
            if (fuel_species_id >= 1 .and. fuel_species_id <= (chemxe - chemxb + 1) .and. &
                evap_liquid_fluid_id >= 1 .and. evap_liquid_fluid_id <= num_fluids) then
                fuel_species_eqn = chemxb + fuel_species_id - 1
                liquid_alpha_eqn = advxb + evap_liquid_fluid_id - 1
                $:GPU_PARALLEL_LOOP(private='[i,j,k,l,alpha_liq,gas_alpha,gas_mass]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            alpha_liq = q_prim_qp%vf(liquid_alpha_eqn)%sf(j, k, l)
                            gas_alpha = 0._wp
                            gas_mass = 0._wp
                            do i = 1, num_fluids
                                if (i /= evap_liquid_fluid_id) then
                                    gas_alpha = gas_alpha + q_prim_qp%vf(advxb + i - 1)%sf(j, k, l)
                                    gas_mass = gas_mass + q_cons_qp%vf(contxb + i - 1)%sf(j, k, l)
                                end if
                            end do
                            if (alpha_liq > evap_alpha_thresh .and. alpha_liq <= evap_species_liq_max .and. &
                                gas_alpha > evap_species_alpha_min .and. gas_mass > evap_species_mass_min) then
                                rhs_vf(fuel_species_eqn)%sf(j, k, l) = rhs_vf(fuel_species_eqn)%sf(j, k, l) + evap_species_src
                            end if
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

        call s_alpha_rhs_term(rhs_vf, "AFTER_CHEMISTRY_SPECIES", t_step, stage)

        if (cont_damage) call s_compute_damage_state(q_cons_qp%vf, rhs_vf)

        call s_alpha_rhs_term(rhs_vf, "FINAL_RHS", t_step, stage)

        if (.not. igr .or. dummy) then
            call s_alpha_origin_checkpoint(q_cons_qp%vf, t_step, stage, &
                                           "RHS_POST_ALL_SOURCES", "owned")
        else
            call s_alpha_origin_checkpoint(q_cons_vf, t_step, stage, &
                                           "RHS_POST_ALL_SOURCES", "owned")
        end if

        ! END: Additional pphysics and source terms

        if (run_time_info .or. probe_wrt .or. ib .or. bubbles_lagrange) then
            if (.not. igr .or. dummy) then
                call s_zhang_evap_hang_trace(t_step, stage, "RHS_COPY_PRIM_OUT_BEGIN")
                $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
                do i = 1, sys_size
                    do l = idwbuff(3)%beg, idwbuff(3)%end
                        do k = idwbuff(2)%beg, idwbuff(2)%end
                            do j = idwbuff(1)%beg, idwbuff(1)%end
                                q_prim_vf(i)%sf(j, k, l) = q_prim_qp%vf(i)%sf(j, k, l)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
                call s_zhang_evap_hang_trace(t_step, stage, "RHS_COPY_PRIM_OUT_END")
            end if
        end if

        call cpu_time(t_finish)

        if (t_step >= 2) then
            time_avg = (abs(t_finish - t_start) + (t_step - 2)*time_avg)/(t_step - 1)
        else
            time_avg = 0._wp
        end if

        call nvtxEndRange
        call s_zhang_evap_hang_trace(t_step, stage, "S_COMPUTE_RHS_END")

    end subroutine s_compute_rhs

    !> @brief Accumulates advection source contributions from a given coordinate direction into the RHS.
    subroutine s_compute_advection_source_term(idir, rhs_vf, q_cons_vf, q_prim_vf, flux_src_n_vf, t_step, stage)

        integer, intent(in) :: idir
        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        type(vector_field), intent(inout) :: q_cons_vf
        type(vector_field), intent(inout) :: q_prim_vf
        type(vector_field), intent(inout) :: flux_src_n_vf
        integer, intent(in) :: t_step, stage

        integer :: j, k, l, q ! Loop iterators from original, meaning varies
        integer :: k_loop, l_loop, q_loop ! Standardized spatial loop iterators 0:m, 0:n, 0:p
        integer :: i_fluid_loop

        real(wp) :: inv_ds, flux_face1, flux_face2
        real(wp) :: advected_qty_val, pressure_val, velocity_val

        if (alt_soundspeed) then
            $:GPU_PARALLEL_LOOP(private='[k_loop,l_loop,q_loop]', collapse=3)
            do q_loop = 0, p
                do l_loop = 0, n
                    do k_loop = 0, m
                        blkmod1(k_loop, l_loop, q_loop) = ((gammas(1) + 1._wp)*q_prim_vf%vf(E_idx)%sf(k_loop, l_loop, q_loop) + &
                                                           pi_infs(1))/gammas(1)
                        blkmod2(k_loop, l_loop, q_loop) = ((gammas(2) + 1._wp)*q_prim_vf%vf(E_idx)%sf(k_loop, l_loop, q_loop) + &
                                                           pi_infs(2))/gammas(2)
                        alpha1(k_loop, l_loop, q_loop) = q_cons_vf%vf(advxb)%sf(k_loop, l_loop, q_loop)

                        if (bubbles_euler) then
                            alpha2(k_loop, l_loop, q_loop) = q_cons_vf%vf(alf_idx - 1)%sf(k_loop, l_loop, q_loop)
                        else
                            alpha2(k_loop, l_loop, q_loop) = q_cons_vf%vf(advxe)%sf(k_loop, l_loop, q_loop)
                        end if

                        Kterm(k_loop, l_loop, q_loop) = alpha1(k_loop, l_loop, q_loop)*alpha2(k_loop, l_loop, q_loop)* &
                                                        (blkmod2(k_loop, l_loop, q_loop) - blkmod1(k_loop, l_loop, q_loop))/ &
                                                        (alpha1(k_loop, l_loop, q_loop)*blkmod2(k_loop, l_loop, q_loop) + &
                                                         alpha2(k_loop, l_loop, q_loop)*blkmod1(k_loop, l_loop, q_loop))
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        select case (idir)
        case (1)  ! x-direction
            call s_alpha_rhs_cbc_faces(flux_n(idir)%vf, flux_src_n_vf%vf, idir, "PRE", t_step, stage)
            if (bc_x%beg <= BC_CHAR_SLIP_WALL .and. bc_x%beg >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, -1, irx, iry, irz)
            end if
            if (bc_x%end <= BC_CHAR_SLIP_WALL .and. bc_x%end >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, 1, irx, iry, irz)
            end if
            call s_alpha_rhs_cbc_faces(flux_n(idir)%vf, flux_src_n_vf%vf, idir, "POST", t_step, stage)

            $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k_loop,l_loop,q_loop,inv_ds,flux_face1,flux_face2]')
            do j = 1, sys_size
                do q_loop = 0, p
                    do l_loop = 0, n
                        do k_loop = 0, m
                            inv_ds = 1._wp/dx(k_loop)
                            flux_face1 = flux_n(1)%vf(j)%sf(k_loop - 1, l_loop, q_loop)
                            flux_face2 = flux_n(1)%vf(j)%sf(k_loop, l_loop, q_loop)
                            rhs_vf(j)%sf(k_loop, l_loop, q_loop) = inv_ds*(flux_face1 - flux_face2)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            call s_alpha_rhs_term(rhs_vf, "X_FLUX_DIVERGENCE", t_step, stage)

            if (model_eqns == 3) then
                $:GPU_PARALLEL_LOOP(collapse=4,private='[i_fluid_loop,k_loop,l_loop,q_loop,inv_ds,advected_qty_val, pressure_val,flux_face1,flux_face2]')
                do q_loop = 0, p
                    do l_loop = 0, n
                        do k_loop = 0, m
                            do i_fluid_loop = 1, num_fluids
                                inv_ds = 1._wp/dx(k_loop)
                                advected_qty_val = q_cons_vf%vf(i_fluid_loop + advxb - 1)%sf(k_loop, l_loop, q_loop)
                                pressure_val = q_prim_vf%vf(E_idx)%sf(k_loop, l_loop, q_loop)
                                flux_face1 = flux_src_n_vf%vf(advxb)%sf(k_loop, l_loop, q_loop)
                                flux_face2 = flux_src_n_vf%vf(advxb)%sf(k_loop - 1, l_loop, q_loop)
                                rhs_vf(i_fluid_loop + intxb - 1)%sf(k_loop, l_loop, q_loop) = &
                                    rhs_vf(i_fluid_loop + intxb - 1)%sf(k_loop, l_loop, q_loop) - &
                                    inv_ds*advected_qty_val*pressure_val*(flux_face1 - flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            call s_add_directional_advection_source_terms(idir, rhs_vf, q_cons_vf, q_prim_vf, flux_src_n_vf, Kterm)
            call s_alpha_rhs_term(rhs_vf, "X_ALPHA_ADVECTION", t_step, stage)

        case (2) ! y-direction
            call s_alpha_rhs_cbc_faces(flux_n(idir)%vf, flux_src_n_vf%vf, idir, "PRE", t_step, stage)
            if (bc_y%beg <= BC_CHAR_SLIP_WALL .and. bc_y%beg >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, -1, irx, iry, irz)
            end if
            if (bc_y%end <= BC_CHAR_SLIP_WALL .and. bc_y%end >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, 1, irx, iry, irz)
            end if
            call s_alpha_rhs_cbc_faces(flux_n(idir)%vf, flux_src_n_vf%vf, idir, "POST", t_step, stage)

            $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k,l,q,inv_ds,flux_face1,flux_face2]')
            do j = 1, sys_size
                do l = 0, p
                    do k = 0, n
                        do q = 0, m
                            inv_ds = 1._wp/dy(k)
                            flux_face1 = flux_n(2)%vf(j)%sf(q, k - 1, l)
                            flux_face2 = flux_n(2)%vf(j)%sf(q, k, l)
                            rhs_vf(j)%sf(q, k, l) = rhs_vf(j)%sf(q, k, l) + inv_ds*(flux_face1 - flux_face2)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            if (model_eqns == 3) then
                $:GPU_PARALLEL_LOOP(collapse=4,private='[i_fluid_loop,k,l,q,inv_ds,advected_qty_val, pressure_val,flux_face1,flux_face2]')
                do l = 0, p
                    do k = 0, n
                        do q = 0, m
                            do i_fluid_loop = 1, num_fluids
                                inv_ds = 1._wp/dy(k)
                                advected_qty_val = q_cons_vf%vf(i_fluid_loop + advxb - 1)%sf(q, k, l)
                                pressure_val = q_prim_vf%vf(E_idx)%sf(q, k, l)
                                flux_face1 = flux_src_n_vf%vf(advxb)%sf(q, k, l)
                                flux_face2 = flux_src_n_vf%vf(advxb)%sf(q, k - 1, l)
                                rhs_vf(i_fluid_loop + intxb - 1)%sf(q, k, l) = &
                                    rhs_vf(i_fluid_loop + intxb - 1)%sf(q, k, l) - &
                                    inv_ds*advected_qty_val*pressure_val*(flux_face1 - flux_face2)
                                if (cyl_coord) then
                                    rhs_vf(i_fluid_loop + intxb - 1)%sf(q, k, l) = &
                                        rhs_vf(i_fluid_loop + intxb - 1)%sf(q, k, l) - &
                                        5.e-1_wp/y_cc(k)*advected_qty_val*pressure_val*(flux_face1 + flux_face2)
                                end if
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if (cyl_coord) then
                $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k,l,q,flux_face1,flux_face2]')
                do j = 1, sys_size
                    do l = 0, p
                        do k = 0, n
                            do q = 0, m
                                flux_face1 = flux_gsrc_n(2)%vf(j)%sf(q, k - 1, l)
                                flux_face2 = flux_gsrc_n(2)%vf(j)%sf(q, k, l)
                                rhs_vf(j)%sf(q, k, l) = rhs_vf(j)%sf(q, k, l) - &
                                                        5.e-1_wp/y_cc(k)*(flux_face1 + flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            call s_alpha_rhs_term(rhs_vf, "Y_FLUX_DIVERGENCE", t_step, stage)

            call s_add_directional_advection_source_terms(idir, rhs_vf, q_cons_vf, q_prim_vf, flux_src_n_vf, Kterm)
            call s_alpha_rhs_term(rhs_vf, "Y_ALPHA_ADVECTION", t_step, stage)

        case (3) ! z-direction
            if (bc_z%beg <= BC_CHAR_SLIP_WALL .and. bc_z%beg >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, -1, irx, iry, irz)
            end if
            if (bc_z%end <= BC_CHAR_SLIP_WALL .and. bc_z%end >= BC_CHAR_SUP_OUTFLOW) then
                call s_cbc(q_prim_vf%vf, flux_n(idir)%vf, flux_src_n_vf%vf, idir, 1, irx, iry, irz)
            end if

            if (grid_geometry == 3) then ! Cylindrical Coordinates
                $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k,l,q,inv_ds,velocity_val,flux_face1,flux_face2]')
                do j = 1, sys_size
                    do k = 0, p
                        do q = 0, n
                            do l = 0, m
                                inv_ds = 1._wp/(dz(k)*y_cc(q))
                                velocity_val = q_prim_vf%vf(contxe + idir)%sf(l, q, k)
                                flux_face1 = flux_n(3)%vf(j)%sf(l, q, k - 1)
                                flux_face2 = flux_n(3)%vf(j)%sf(l, q, k)
                                rhs_vf(j)%sf(l, q, k) = rhs_vf(j)%sf(l, q, k) + &
                                                        inv_ds*velocity_val*(flux_face1 - flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
                $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k,l,q,flux_face1,flux_face2]')
                do j = 1, sys_size
                    do k = 0, p
                        do q = 0, n
                            do l = 0, m
                                flux_face1 = flux_gsrc_n(3)%vf(j)%sf(l, q, k - 1)
                                flux_face2 = flux_gsrc_n(3)%vf(j)%sf(l, q, k)
                                rhs_vf(j)%sf(l, q, k) = rhs_vf(j)%sf(l, q, k) - &
                                                        5.e-1_wp/y_cc(q)*(flux_face1 + flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            else ! Cartesian Coordinates
                $:GPU_PARALLEL_LOOP(collapse=4,private='[j,k,l,q,inv_ds,flux_face1,flux_face2]')
                do j = 1, sys_size
                    do k = 0, p
                        do q = 0, n
                            do l = 0, m
                                inv_ds = 1._wp/dz(k)
                                flux_face1 = flux_n(3)%vf(j)%sf(l, q, k - 1)
                                flux_face2 = flux_n(3)%vf(j)%sf(l, q, k)
                                rhs_vf(j)%sf(l, q, k) = rhs_vf(j)%sf(l, q, k) + inv_ds*(flux_face1 - flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            call s_alpha_rhs_term(rhs_vf, "Z_FLUX_DIVERGENCE", t_step, stage)

            if (model_eqns == 3) then
                $:GPU_PARALLEL_LOOP(collapse=4,private='[i_fluid_loop,k,l,q,inv_ds,advected_qty_val, pressure_val,flux_face1,flux_face2]')
                do k = 0, p
                    do q = 0, n
                        do l = 0, m
                            do i_fluid_loop = 1, num_fluids
                                inv_ds = 1._wp/dz(k)
                                advected_qty_val = q_cons_vf%vf(i_fluid_loop + advxb - 1)%sf(l, q, k)
                                pressure_val = q_prim_vf%vf(E_idx)%sf(l, q, k)
                                flux_face1 = flux_src_n_vf%vf(advxb)%sf(l, q, k)
                                flux_face2 = flux_src_n_vf%vf(advxb)%sf(l, q, k - 1)
                                rhs_vf(i_fluid_loop + intxb - 1)%sf(l, q, k) = &
                                    rhs_vf(i_fluid_loop + intxb - 1)%sf(l, q, k) - &
                                    inv_ds*advected_qty_val*pressure_val*(flux_face1 - flux_face2)
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            call s_add_directional_advection_source_terms(idir, rhs_vf, q_cons_vf, q_prim_vf, flux_src_n_vf, Kterm)
            call s_alpha_rhs_term(rhs_vf, "Z_ALPHA_ADVECTION", t_step, stage)

        end select

    contains

        !> @brief Adds the advection source flux-difference terms for a single coordinate direction to the RHS.
        subroutine s_add_directional_advection_source_terms(current_idir, rhs_vf_arg, q_cons_vf_arg, &
                                                            q_prim_vf_arg, flux_src_n_vf_arg, Kterm_arg)
            integer, intent(in) :: current_idir
            type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf_arg
            type(vector_field), intent(in) :: q_cons_vf_arg
            type(vector_field), intent(in) :: q_prim_vf_arg
            type(vector_field), intent(in) :: flux_src_n_vf_arg
            ! CORRECTED DECLARATION FOR Kterm_arg:
            real(wp), allocatable, dimension(:, :, :), intent(in) :: Kterm_arg

            integer :: j_adv, k_idx, l_idx, q_idx
            real(wp) :: local_inv_ds, local_term_coeff, local_flux1, local_flux2
            real(wp) :: local_q_cons_val, local_k_term_val
            logical :: use_standard_riemann

            select case (current_idir)
            case (1) ! x-direction
                use_standard_riemann = (riemann_solver == 1 .or. riemann_solver == 4)
                if (use_standard_riemann) then
                    $:GPU_PARALLEL_LOOP(collapse=4,private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                    do j_adv = advxb, advxe
                        do q_idx = 0, p ! z_extent
                            do l_idx = 0, n ! y_extent
                                do k_idx = 0, m ! x_extent
                                    local_inv_ds = 1._wp/dx(k_idx)
                                    local_term_coeff = q_prim_vf_arg%vf(contxe + current_idir)%sf(k_idx, l_idx, q_idx)
                                    local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(k_idx - 1, l_idx, q_idx)
                                    local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(k_idx, l_idx, q_idx)
                                    rhs_vf_arg(j_adv)%sf(k_idx, l_idx, q_idx) = rhs_vf_arg(j_adv)%sf(k_idx, l_idx, q_idx) + &
                                                                                local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                else ! Other Riemann solvers
                    if (alt_soundspeed) then
                        if (bubbles_euler .neqv. .true.) then
                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds, local_q_cons_val, local_k_term_val, local_term_coeff, local_flux1, local_flux2]')
                            do q_idx = 0, p; do l_idx = 0, n; do k_idx = 0, m
                                        local_inv_ds = 1._wp/dx(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxe)%sf(k_idx, l_idx, q_idx)
                                        local_k_term_val = Kterm_arg(k_idx, l_idx, q_idx) ! Access is safe due to outer alt_soundspeed check
                                        local_term_coeff = local_q_cons_val - local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxe)%sf(k_idx, l_idx, q_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxe)%sf(k_idx - 1, l_idx, q_idx)
                                        rhs_vf_arg(advxe)%sf(k_idx, l_idx, q_idx) = rhs_vf_arg(advxe)%sf(k_idx, l_idx, q_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()

                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds,local_q_cons_val, local_k_term_val,local_term_coeff, local_flux1, local_flux2]')
                            do q_idx = 0, p; do l_idx = 0, n; do k_idx = 0, m
                                        local_inv_ds = 1._wp/dx(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxb)%sf(k_idx, l_idx, q_idx)
                                        local_k_term_val = Kterm_arg(k_idx, l_idx, q_idx) ! Access is safe
                                        local_term_coeff = local_q_cons_val + local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxb)%sf(k_idx, l_idx, q_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxb)%sf(k_idx - 1, l_idx, q_idx)
                                        rhs_vf_arg(advxb)%sf(k_idx, l_idx, q_idx) = rhs_vf_arg(advxb)%sf(k_idx, l_idx, q_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()
                        end if
                    else ! NOT alt_soundspeed
                        $:GPU_PARALLEL_LOOP(collapse=4,private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                        do j_adv = advxb, advxe
                            do q_idx = 0, p; do l_idx = 0, n; do k_idx = 0, m
                                        local_inv_ds = 1._wp/dx(k_idx)
                                        local_term_coeff = q_cons_vf_arg%vf(j_adv)%sf(k_idx, l_idx, q_idx)
                                        local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(k_idx, l_idx, q_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(k_idx - 1, l_idx, q_idx)
                                        rhs_vf_arg(j_adv)%sf(k_idx, l_idx, q_idx) = rhs_vf_arg(j_adv)%sf(k_idx, l_idx, q_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                        end do
                        $:END_GPU_PARALLEL_LOOP()
                    end if
                end if

            case (2) ! y-direction: loops q_idx (x), k_idx (y), l_idx (z); sf(q_idx, k_idx, l_idx); dy(k_idx); Kterm(q_idx,k_idx,l_idx)
                use_standard_riemann = (riemann_solver == 1 .or. riemann_solver == 4)
                if (use_standard_riemann) then
                    $:GPU_PARALLEL_LOOP(collapse=4,private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                    do j_adv = advxb, advxe
                        do l_idx = 0, p ! z_extent
                            do k_idx = 0, n ! y_extent
                                do q_idx = 0, m ! x_extent
                                    local_inv_ds = 1._wp/dy(k_idx)
                                    local_term_coeff = q_prim_vf_arg%vf(contxe + current_idir)%sf(q_idx, k_idx, l_idx)
                                    local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(q_idx, k_idx - 1, l_idx)
                                    local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(q_idx, k_idx, l_idx)
                                    rhs_vf_arg(j_adv)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(j_adv)%sf(q_idx, k_idx, l_idx) + &
                                                                                local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                else ! Other Riemann solvers
                    if (alt_soundspeed) then
                        if (bubbles_euler .neqv. .true.) then
                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds, local_q_cons_val, local_k_term_val, local_term_coeff, local_flux1, local_flux2]')
                            do l_idx = 0, p; do k_idx = 0, n; do q_idx = 0, m
                                        local_inv_ds = 1._wp/dy(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxe)%sf(q_idx, k_idx, l_idx)
                                        local_k_term_val = Kterm_arg(q_idx, k_idx, l_idx) ! Access is safe
                                        local_term_coeff = local_q_cons_val - local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxe)%sf(q_idx, k_idx, l_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxe)%sf(q_idx, k_idx - 1, l_idx)
                                        rhs_vf_arg(advxe)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(advxe)%sf(q_idx, k_idx, l_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                        if (cyl_coord) then
                                            rhs_vf_arg(advxe)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(advxe)%sf(q_idx, k_idx, l_idx) - &
                                                                                        (local_k_term_val/(2._wp*y_cc(k_idx)))*(local_flux1 + local_flux2)
                                        end if
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()

                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds, local_q_cons_val, local_k_term_val,local_term_coeff, local_flux1, local_flux2]')
                            do l_idx = 0, p; do k_idx = 0, n; do q_idx = 0, m
                                        local_inv_ds = 1._wp/dy(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxb)%sf(q_idx, k_idx, l_idx)
                                        local_k_term_val = Kterm_arg(q_idx, k_idx, l_idx) ! Access is safe
                                        local_term_coeff = local_q_cons_val + local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxb)%sf(q_idx, k_idx, l_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxb)%sf(q_idx, k_idx - 1, l_idx)
                                        rhs_vf_arg(advxb)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(advxb)%sf(q_idx, k_idx, l_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                        if (cyl_coord) then
                                            rhs_vf_arg(advxb)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(advxb)%sf(q_idx, k_idx, l_idx) + &
                                                                                        (local_k_term_val/(2._wp*y_cc(k_idx)))*(local_flux1 + local_flux2)
                                        end if
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()
                        end if
                    else ! NOT alt_soundspeed
                        $:GPU_PARALLEL_LOOP(collapse=4,private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                        do j_adv = advxb, advxe
                            do l_idx = 0, p; do k_idx = 0, n; do q_idx = 0, m
                                        local_inv_ds = 1._wp/dy(k_idx)
                                        local_term_coeff = q_cons_vf_arg%vf(j_adv)%sf(q_idx, k_idx, l_idx)
                                        local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(q_idx, k_idx, l_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(q_idx, k_idx - 1, l_idx)
                                        rhs_vf_arg(j_adv)%sf(q_idx, k_idx, l_idx) = rhs_vf_arg(j_adv)%sf(q_idx, k_idx, l_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                        end do
                        $:END_GPU_PARALLEL_LOOP()
                    end if
                end if

            case (3) ! z-direction: loops l_idx (x), q_idx (y), k_idx (z); sf(l_idx, q_idx, k_idx); dz(k_idx); Kterm(l_idx,q_idx,k_idx)
                if (grid_geometry == 3) then
                    use_standard_riemann = (riemann_solver == 1)
                else
                    use_standard_riemann = (riemann_solver == 1 .or. riemann_solver == 4)
                end if

                if (use_standard_riemann) then
                    $:GPU_PARALLEL_LOOP(collapse=4,private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                    do j_adv = advxb, advxe
                        do k_idx = 0, p ! z_extent
                            do q_idx = 0, n ! y_extent
                                do l_idx = 0, m ! x_extent
                                    local_inv_ds = 1._wp/dz(k_idx)
                                    local_term_coeff = q_prim_vf_arg%vf(contxe + current_idir)%sf(l_idx, q_idx, k_idx)
                                    local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(l_idx, q_idx, k_idx - 1)
                                    local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(l_idx, q_idx, k_idx)
                                    rhs_vf_arg(j_adv)%sf(l_idx, q_idx, k_idx) = rhs_vf_arg(j_adv)%sf(l_idx, q_idx, k_idx) + &
                                                                                local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                else ! Other Riemann solvers
                    if (alt_soundspeed) then
                        if (bubbles_euler .neqv. .true.) then
                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds,local_q_cons_val, local_k_term_val, local_term_coeff, local_flux1, local_flux2]')
                            do k_idx = 0, p; do q_idx = 0, n; do l_idx = 0, m
                                        local_inv_ds = 1._wp/dz(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxe)%sf(l_idx, q_idx, k_idx)
                                        local_k_term_val = Kterm_arg(l_idx, q_idx, k_idx) ! Access is safe
                                        local_term_coeff = local_q_cons_val - local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxe)%sf(l_idx, q_idx, k_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxe)%sf(l_idx, q_idx, k_idx - 1)
                                        rhs_vf_arg(advxe)%sf(l_idx, q_idx, k_idx) = rhs_vf_arg(advxe)%sf(l_idx, q_idx, k_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()

                            $:GPU_PARALLEL_LOOP(collapse=3, private='[k_idx,l_idx,q_idx,local_inv_ds, local_q_cons_val, local_k_term_val, local_term_coeff, local_flux1, local_flux2]')
                            do k_idx = 0, p; do q_idx = 0, n; do l_idx = 0, m
                                        local_inv_ds = 1._wp/dz(k_idx)
                                        local_q_cons_val = q_cons_vf_arg%vf(advxb)%sf(l_idx, q_idx, k_idx)
                                        local_k_term_val = Kterm_arg(l_idx, q_idx, k_idx) ! Access is safe
                                        local_term_coeff = local_q_cons_val + local_k_term_val
                                        local_flux1 = flux_src_n_vf_arg%vf(advxb)%sf(l_idx, q_idx, k_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(advxb)%sf(l_idx, q_idx, k_idx - 1)
                                        rhs_vf_arg(advxb)%sf(l_idx, q_idx, k_idx) = rhs_vf_arg(advxb)%sf(l_idx, q_idx, k_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                            $:END_GPU_PARALLEL_LOOP()
                        end if
                    else ! NOT alt_soundspeed
                        $:GPU_PARALLEL_LOOP(collapse=4, private='[j_adv,k_idx,l_idx,q_idx,local_inv_ds, local_term_coeff,local_flux1,local_flux2]')
                        do j_adv = advxb, advxe
                            do k_idx = 0, p; do q_idx = 0, n; do l_idx = 0, m
                                        local_inv_ds = 1._wp/dz(k_idx)
                                        local_term_coeff = q_cons_vf_arg%vf(j_adv)%sf(l_idx, q_idx, k_idx)
                                        local_flux1 = flux_src_n_vf_arg%vf(j_adv)%sf(l_idx, q_idx, k_idx)
                                        local_flux2 = flux_src_n_vf_arg%vf(j_adv)%sf(l_idx, q_idx, k_idx - 1)
                                        rhs_vf_arg(j_adv)%sf(l_idx, q_idx, k_idx) = rhs_vf_arg(j_adv)%sf(l_idx, q_idx, k_idx) + &
                                                                                    local_inv_ds*local_term_coeff*(local_flux1 - local_flux2)
                                    end do; end do; end do
                        end do
                        $:END_GPU_PARALLEL_LOOP()
                    end if
                end if
            end select
        end subroutine s_add_directional_advection_source_terms

    end subroutine s_compute_advection_source_term

    !> @brief Adds viscous, surface-tension, and species-diffusion source flux contributions to the RHS for a given direction.
    subroutine s_compute_additional_physics_rhs(idir, q_prim_vf, rhs_vf, flux_src_n_in, &
                                                dq_prim_dx_vf, dq_prim_dy_vf, dq_prim_dz_vf)

        integer, intent(in) :: idir
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        type(scalar_field), dimension(sys_size), intent(inout) :: rhs_vf
        type(scalar_field), dimension(sys_size), intent(in) :: flux_src_n_in
        type(scalar_field), dimension(sys_size), intent(in) :: dq_prim_dx_vf, dq_prim_dy_vf, dq_prim_dz_vf

        integer :: i, j, k, l

        if (idir == 1) then ! x-direction

            if (surface_tension) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            rhs_vf(c_idx)%sf(j, k, l) = &
                                rhs_vf(c_idx)%sf(j, k, l) + 1._wp/dx(j)* &
                                q_prim_vf(c_idx)%sf(j, k, l)* &
                                (flux_src_n_in(advxb)%sf(j, k, l) - &
                                 flux_src_n_in(advxb)%sf(j - 1, k, l))
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if ((surface_tension .or. viscous) .or. chem_params%diffusion) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            if (surface_tension .or. viscous) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = momxb, E_idx
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) + 1._wp/dx(j)* &
                                        (flux_src_n_in(i)%sf(j - 1, k, l) &
                                         - flux_src_n_in(i)%sf(j, k, l))
                                end do
                            end if

                            if (chem_params%diffusion) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = chemxb, chemxe
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) + 1._wp/dx(j)* &
                                        (flux_src_n_in(i)%sf(j - 1, k, l) &
                                         - flux_src_n_in(i)%sf(j, k, l))
                                end do

                                if (.not. viscous) then
                                    rhs_vf(E_idx)%sf(j, k, l) = &
                                        rhs_vf(E_idx)%sf(j, k, l) + 1._wp/dx(j)* &
                                        (flux_src_n_in(E_idx)%sf(j - 1, k, l) &
                                         - flux_src_n_in(E_idx)%sf(j, k, l))
                                end if
                            end if
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

        elseif (idir == 2) then ! y-direction

            if (surface_tension) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            rhs_vf(c_idx)%sf(j, k, l) = &
                                rhs_vf(c_idx)%sf(j, k, l) + 1._wp/dy(k)* &
                                q_prim_vf(c_idx)%sf(j, k, l)* &
                                (flux_src_n_in(advxb)%sf(j, k, l) - &
                                 flux_src_n_in(advxb)%sf(j, k - 1, l))
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if (cyl_coord .and. ((bc_y%beg == -2) .or. (bc_y%beg == -14))) then
                if (viscous .or. dummy) then
                    if (p > 0) then
                        call s_compute_viscous_stress_cylindrical_boundary(q_prim_vf, &
                                                                           dq_prim_dx_vf(mom_idx%beg:mom_idx%end), &
                                                                           dq_prim_dy_vf(mom_idx%beg:mom_idx%end), &
                                                                           dq_prim_dz_vf(mom_idx%beg:mom_idx%end), &
                                                                           tau_Re_vf, &
                                                                           idwbuff(1), idwbuff(2), idwbuff(3))
                    else
                        call s_compute_viscous_stress_cylindrical_boundary(q_prim_vf, &
                                                                           dq_prim_dx_vf(mom_idx%beg:mom_idx%end), &
                                                                           dq_prim_dy_vf(mom_idx%beg:mom_idx%end), &
                                                                           dq_prim_dy_vf(mom_idx%beg:mom_idx%end), &
                                                                           tau_Re_vf, &
                                                                           idwbuff(1), idwbuff(2), idwbuff(3))
                    end if

                    $:GPU_PARALLEL_LOOP(private='[i,j,l]', collapse=2)
                    do l = 0, p
                        do j = 0, m
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = momxb, E_idx
                                rhs_vf(i)%sf(j, 0, l) = &
                                    rhs_vf(i)%sf(j, 0, l) + 1._wp/(y_cc(1) - y_cc(-1))* &
                                    (tau_Re_vf(i)%sf(j, -1, l) &
                                     - tau_Re_vf(i)%sf(j, 1, l))
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()

                end if

                $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
                do l = 0, p
                    do k = 1, n
                        do j = 0, m
                            $:GPU_LOOP(parallelism='[seq]')
                            do i = momxb, E_idx
                                rhs_vf(i)%sf(j, k, l) = &
                                    rhs_vf(i)%sf(j, k, l) + 1._wp/dy(k)* &
                                    (flux_src_n_in(i)%sf(j, k - 1, l) &
                                     - flux_src_n_in(i)%sf(j, k, l))
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()

            else

                if ((surface_tension .or. viscous) .or. chem_params%diffusion) then
                    $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
                    do l = 0, p
                        do k = 0, n
                            do j = 0, m
                                if (surface_tension .or. viscous) then
                                    $:GPU_LOOP(parallelism='[seq]')
                                    do i = momxb, E_idx
                                        rhs_vf(i)%sf(j, k, l) = &
                                            rhs_vf(i)%sf(j, k, l) + 1._wp/dy(k)* &
                                            (flux_src_n_in(i)%sf(j, k - 1, l) &
                                             - flux_src_n_in(i)%sf(j, k, l))
                                    end do
                                end if

                                if (chem_params%diffusion) then
                                    $:GPU_LOOP(parallelism='[seq]')
                                    do i = chemxb, chemxe
                                        rhs_vf(i)%sf(j, k, l) = &
                                            rhs_vf(i)%sf(j, k, l) + 1._wp/dy(k)* &
                                            (flux_src_n_in(i)%sf(j, k - 1, l) &
                                             - flux_src_n_in(i)%sf(j, k, l))
                                    end do
                                    if (.not. viscous) then
                                        rhs_vf(E_idx)%sf(j, k, l) = &
                                            rhs_vf(E_idx)%sf(j, k, l) + 1._wp/dy(k)* &
                                            (flux_src_n_in(E_idx)%sf(j, k - 1, l) &
                                             - flux_src_n_in(E_idx)%sf(j, k, l))
                                    end if
                                end if
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                end if
            end if

            ! Applying the geometrical viscous Riemann source fluxes calculated as average
            ! of values at cell boundaries
            if (cyl_coord) then
                if ((bc_y%beg == -2) .or. (bc_y%beg == -14)) then

                    $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
                    do l = 0, p
                        do k = 1, n
                            do j = 0, m
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = momxb, E_idx
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) - 5.e-1_wp/y_cc(k)* &
                                        (flux_src_n_in(i)%sf(j, k - 1, l) &
                                         + flux_src_n_in(i)%sf(j, k, l))
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()

                    if (viscous .or. dummy) then
                        $:GPU_PARALLEL_LOOP(private='[i,j,l]', collapse=2)
                        do l = 0, p
                            do j = 0, m
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = momxb, E_idx
                                    rhs_vf(i)%sf(j, 0, l) = &
                                        rhs_vf(i)%sf(j, 0, l) - 1._wp/y_cc(0)* &
                                        tau_Re_vf(i)%sf(j, 0, l)
                                end do
                            end do
                        end do
                        $:END_GPU_PARALLEL_LOOP()
                    end if
                else

                    $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
                    do l = 0, p
                        do k = 0, n
                            do j = 0, m
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = momxb, E_idx
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) - 5.e-1_wp/y_cc(k)* &
                                        (flux_src_n_in(i)%sf(j, k - 1, l) &
                                         + flux_src_n_in(i)%sf(j, k, l))
                                end do
                            end do
                        end do
                    end do
                    $:END_GPU_PARALLEL_LOOP()
                end if
            end if

        elseif (idir == 3) then ! z-direction

            if (surface_tension) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            rhs_vf(c_idx)%sf(j, k, l) = &
                                rhs_vf(c_idx)%sf(j, k, l) + 1._wp/dz(l)* &
                                q_prim_vf(c_idx)%sf(j, k, l)* &
                                (flux_src_n_in(advxb)%sf(j, k, l) - &
                                 flux_src_n_in(advxb)%sf(j, k, l - 1))
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if ((surface_tension .or. viscous) .or. chem_params%diffusion) then
                $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            if (surface_tension .or. viscous) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = momxb, E_idx
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) + 1._wp/dz(l)* &
                                        (flux_src_n_in(i)%sf(j, k, l - 1) &
                                         - flux_src_n_in(i)%sf(j, k, l))
                                end do
                            end if

                            if (chem_params%diffusion) then
                                $:GPU_LOOP(parallelism='[seq]')
                                do i = chemxb, chemxe
                                    rhs_vf(i)%sf(j, k, l) = &
                                        rhs_vf(i)%sf(j, k, l) + 1._wp/dz(l)* &
                                        (flux_src_n_in(i)%sf(j, k, l - 1) &
                                         - flux_src_n_in(i)%sf(j, k, l))
                                end do
                                if (.not. viscous) then
                                    rhs_vf(E_idx)%sf(j, k, l) = &
                                        rhs_vf(E_idx)%sf(j, k, l) + 1._wp/dz(l)* &
                                        (flux_src_n_in(E_idx)%sf(j, k, l - 1) &
                                         - flux_src_n_in(E_idx)%sf(j, k, l))
                                end if
                            end if
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            if (grid_geometry == 3) then
                $:GPU_PARALLEL_LOOP(private='[j,k,l]', collapse=3)
                do l = 0, p
                    do k = 0, n
                        do j = 0, m
                            rhs_vf(momxb + 1)%sf(j, k, l) = &
                                rhs_vf(momxb + 1)%sf(j, k, l) + 5.e-1_wp* &
                                (flux_src_n_in(momxe)%sf(j, k, l - 1) &
                                 + flux_src_n_in(momxe)%sf(j, k, l))

                            rhs_vf(momxe)%sf(j, k, l) = &
                                rhs_vf(momxe)%sf(j, k, l) - 5.e-1_wp* &
                                (flux_src_n_in(momxb + 1)%sf(j, k, l - 1) &
                                 + flux_src_n_in(momxb + 1)%sf(j, k, l))
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if
        end if

    end subroutine s_compute_additional_physics_rhs

    !>  The purpose of this subroutine is to WENO-reconstruct the
        !!      left and the right cell-boundary values, including values
        !!      at the Gaussian quadrature points, from the cell-averaged
        !!      variables.
        !!  @param v_vf Cell-average variables
        !!  @param vL_x Left reconstructed cell-boundary values in x
        !!  @param vL_y Left reconstructed cell-boundary values in y
        !!  @param vL_z Left reconstructed cell-boundary values in z
        !!  @param vR_x Right reconstructed cell-boundary values in x
        !!  @param vR_y Right reconstructed cell-boundary values in y
        !!  @param vR_z Right reconstructed cell-boundary values in z
        !!  @param norm_dir Splitting coordinate direction
    subroutine s_reconstruct_cell_boundary_values(v_vf, vL_x, vL_y, vL_z, vR_x, vR_y, vR_z, &
                                                  norm_dir)

        type(scalar_field), dimension(iv%beg:iv%end), intent(in) :: v_vf
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:), intent(inout) :: vL_x, vL_y, vL_z
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:), intent(inout) :: vR_x, vR_y, vR_z
        integer, intent(in) :: norm_dir

        integer :: recon_dir !< Coordinate direction of the reconstruction

        integer :: i, j, k, l

        #:for SCHEME, TYPE in [('weno','WENO_TYPE'), ('muscl','MUSCL_TYPE')]
            if (recon_type == ${TYPE}$ .or. dummy) then
                ! Reconstruction in s1-direction
                if (norm_dir == 1) then
                    is1 = idwbuff(1); is2 = idwbuff(2); is3 = idwbuff(3)
                    recon_dir = 1; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn

                elseif (norm_dir == 2) then
                    is1 = idwbuff(2); is2 = idwbuff(1); is3 = idwbuff(3)
                    recon_dir = 2; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn

                else
                    is1 = idwbuff(3); is2 = idwbuff(2); is3 = idwbuff(1)
                    recon_dir = 3; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn
                end if

                if (n > 0) then
                    if (p > 0) then
                        call s_${SCHEME}$ (v_vf(iv%beg:iv%end), &
                                           vL_x(:, :, :, iv%beg:iv%end), vL_y(:, :, :, iv%beg:iv%end), vL_z(:, :, :, iv%beg:iv%end), vR_x(:, :, :, iv%beg:iv%end), vR_y(:, :, :, iv%beg:iv%end), vR_z(:, :, :, iv%beg:iv%end), &
                                           recon_dir, &
                                           is1, is2, is3)
                    else
                        call s_${SCHEME}$ (v_vf(iv%beg:iv%end), &
                                           vL_x(:, :, :, iv%beg:iv%end), vL_y(:, :, :, iv%beg:iv%end), vL_z(:, :, :, :), vR_x(:, :, :, iv%beg:iv%end), vR_y(:, :, :, iv%beg:iv%end), vR_z(:, :, :, :), &
                                           recon_dir, &
                                           is1, is2, is3)
                    end if
                else

                    call s_${SCHEME}$ (v_vf(iv%beg:iv%end), &
                                       vL_x(:, :, :, iv%beg:iv%end), vL_y(:, :, :, :), vL_z(:, :, :, :), vR_x(:, :, :, iv%beg:iv%end), vR_y(:, :, :, :), vR_z(:, :, :, :), &
                                       recon_dir, &
                                       is1, is2, is3)
                end if
            end if
        #:endfor
    end subroutine s_reconstruct_cell_boundary_values

    !> @brief Performs first-order (piecewise constant) reconstruction of left and right cell-boundary values.
    subroutine s_reconstruct_cell_boundary_values_first_order(v_vf, vL_x, vL_y, vL_z, vR_x, vR_y, vR_z, &
                                                              norm_dir)

        type(scalar_field), dimension(iv%beg:iv%end), intent(in) :: v_vf
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:), intent(inout) :: vL_x, vL_y, vL_z
        real(wp), dimension(idwbuff(1)%beg:, idwbuff(2)%beg:, idwbuff(3)%beg:, 1:), intent(inout) :: vR_x, vR_y, vR_z
        integer, intent(in) :: norm_dir

        integer :: recon_dir !< Coordinate direction of the WENO reconstruction

        integer :: i, j, k, l
        ! Reconstruction in s1-direction

        #:for SCHEME, TYPE in [('weno','WENO_TYPE'), ('muscl', 'MUSCL_TYPE')]
            if (recon_type == ${TYPE}$ .or. dummy) then
                if (norm_dir == 1) then
                    is1 = idwbuff(1); is2 = idwbuff(2); is3 = idwbuff(3)
                    recon_dir = 1; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn

                elseif (norm_dir == 2) then
                    is1 = idwbuff(2); is2 = idwbuff(1); is3 = idwbuff(3)
                    recon_dir = 2; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn

                else
                    is1 = idwbuff(3); is2 = idwbuff(2); is3 = idwbuff(1)
                    recon_dir = 3; is1%beg = is1%beg + ${SCHEME}$_polyn
                    is1%end = is1%end - ${SCHEME}$_polyn

                end if

                $:GPU_UPDATE(device='[is1,is2,is3,iv]')
            end if
        #:endfor

        if (recon_dir == 1) then
            $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
            do i = iv%beg, iv%end
                do l = is3%beg, is3%end
                    do k = is2%beg, is2%end
                        do j = is1%beg, is1%end
                            vL_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                            vR_x(j, k, l, i) = v_vf(i)%sf(j, k, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        else if (recon_dir == 2) then
            $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
            do i = iv%beg, iv%end
                do l = is3%beg, is3%end
                    do k = is2%beg, is2%end
                        do j = is1%beg, is1%end
                            vL_y(j, k, l, i) = v_vf(i)%sf(k, j, l)
                            vR_y(j, k, l, i) = v_vf(i)%sf(k, j, l)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        else if (recon_dir == 3) then
            $:GPU_PARALLEL_LOOP(private='[i,j,k,l]', collapse=4)
            do i = iv%beg, iv%end
                do l = is3%beg, is3%end
                    do k = is2%beg, is2%end
                        do j = is1%beg, is1%end
                            vL_z(j, k, l, i) = v_vf(i)%sf(l, k, j)
                            vR_z(j, k, l, i) = v_vf(i)%sf(l, k, j)
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

    end subroutine s_reconstruct_cell_boundary_values_first_order

    !> Module deallocation and/or disassociation procedures
    impure subroutine s_finalize_rhs_module

        integer :: i, j, l

        call s_finalize_pressure_relaxation_module

        if (.not. igr) then
            do j = cont_idx%beg, cont_idx%end
                if (relativity) then
                    ! Cons and Prim densities are different for relativity
                    @:DEALLOCATE(q_cons_qp%vf(j)%sf)
                    @:DEALLOCATE(q_prim_qp%vf(j)%sf)
                else
                    nullify (q_prim_qp%vf(j)%sf)
                end if
            end do

            do j = adv_idx%beg, adv_idx%end
                nullify (q_prim_qp%vf(j)%sf)
            end do

            do j = mom_idx%beg, E_idx
                @:DEALLOCATE(q_cons_qp%vf(j)%sf)
                @:DEALLOCATE(q_prim_qp%vf(j)%sf)
            end do
        end if

        @:DEALLOCATE(q_cons_qp%vf, q_prim_qp%vf)

        if (.not. igr) then
            @:DEALLOCATE(qL_rsx_vf, qR_rsx_vf)

            if (n > 0) then
                @:DEALLOCATE(qL_rsy_vf, qR_rsy_vf)
            end if

            if (p > 0) then
                @:DEALLOCATE(qL_rsz_vf, qR_rsz_vf)
            end if

            if (viscous) then
                do l = mom_idx%beg, mom_idx%end
                    @:DEALLOCATE(dq_prim_dx_qp(1)%vf(l)%sf)
                end do

                if (n > 0) then

                    do l = mom_idx%beg, mom_idx%end
                        @:DEALLOCATE(dq_prim_dy_qp(1)%vf(l)%sf)
                    end do

                    if (p > 0) then
                        do l = mom_idx%beg, mom_idx%end
                            @:DEALLOCATE(dq_prim_dz_qp(1)%vf(l)%sf)
                        end do
                    end if

                end if

                @:DEALLOCATE(dq_prim_dx_qp(1)%vf)
                @:DEALLOCATE(dq_prim_dy_qp(1)%vf)
                @:DEALLOCATE(dq_prim_dz_qp(1)%vf)

                do i = num_dims, 1, -1

                    do l = mom_idx%beg, mom_idx%end
                        @:DEALLOCATE(dqL_prim_dx_n(i)%vf(l)%sf)
                        @:DEALLOCATE(dqR_prim_dx_n(i)%vf(l)%sf)
                    end do

                    if (n > 0) then
                        do l = mom_idx%beg, mom_idx%end
                            @:DEALLOCATE(dqL_prim_dy_n(i)%vf(l)%sf)
                            @:DEALLOCATE(dqR_prim_dy_n(i)%vf(l)%sf)
                        end do
                    end if

                    if (p > 0) then
                        do l = mom_idx%beg, mom_idx%end
                            @:DEALLOCATE(dqL_prim_dz_n(i)%vf(l)%sf)
                            @:DEALLOCATE(dqR_prim_dz_n(i)%vf(l)%sf)
                        end do
                    end if

                    @:DEALLOCATE(dqL_prim_dx_n(i)%vf)
                    @:DEALLOCATE(dqL_prim_dy_n(i)%vf)
                    @:DEALLOCATE(dqL_prim_dz_n(i)%vf)
                    @:DEALLOCATE(dqR_prim_dx_n(i)%vf)
                    @:DEALLOCATE(dqR_prim_dy_n(i)%vf)
                    @:DEALLOCATE(dqR_prim_dz_n(i)%vf)
                end do

                if (weno_Re_flux) then
                    @:DEALLOCATE(dqL_rsx_vf, dqR_rsx_vf)

                    if (n > 0) then
                        @:DEALLOCATE(dqL_rsy_vf, dqR_rsy_vf)
                    end if

                    if (p > 0) then
                        @:DEALLOCATE(dqL_rsz_vf, dqR_rsz_vf)
                    end if
                end if

                if (cyl_coord) then
                    do i = 1, num_dims
                        @:DEALLOCATE(tau_re_vf(cont_idx%end + i)%sf)
                    end do
                    @:DEALLOCATE(tau_re_vf(e_idx)%sf)
                    @:DEALLOCATE(tau_re_vf)
                end if
            end if
            @:DEALLOCATE(dqL_prim_dx_n, dqL_prim_dy_n, dqL_prim_dz_n)
            @:DEALLOCATE(dqR_prim_dx_n, dqR_prim_dy_n, dqR_prim_dz_n)
        end if

        if (mpp_lim .and. bubbles_euler) then
            $:GPU_EXIT_DATA(delete='[alf_sum%sf]')
            deallocate (alf_sum%sf)
        end if

        if (.not. igr) then
            do i = num_dims, 1, -1
                if (i /= 1) then
                    do l = 1, sys_size
                        nullify (flux_n(i)%vf(l)%sf)
                        nullify (flux_src_n(i)%vf(l)%sf)
                        @:DEALLOCATE(flux_gsrc_n(i)%vf(l)%sf)
                    end do
                else
                    do l = 1, sys_size
                        @:DEALLOCATE(flux_n(i)%vf(l)%sf)
                        @:DEALLOCATE(flux_gsrc_n(i)%vf(l)%sf)
                    end do

                    if (viscous) then
                        do l = mom_idx%beg, E_idx
                            @:DEALLOCATE(flux_src_n(i)%vf(l)%sf)
                        end do
                    end if

                    if (chem_params%diffusion .and. .not. viscous) then
                        @:DEALLOCATE(flux_src_n(i)%vf(E_idx)%sf)
                    end if

                    if (riemann_solver == 1 .or. riemann_solver == 4) then
                        do l = adv_idx%beg + 1, adv_idx%end
                            @:DEALLOCATE(flux_src_n(i)%vf(l)%sf)
                        end do
                    else
                        do l = adv_idx%beg + 1, adv_idx%end
                            nullify (flux_src_n(i)%vf(l)%sf)
                        end do
                    end if

                    @:DEALLOCATE(flux_src_n(i)%vf(adv_idx%beg)%sf)
                end if

                @:DEALLOCATE(flux_n(i)%vf, flux_src_n(i)%vf, flux_gsrc_n(i)%vf)
            end do

            @:DEALLOCATE(flux_n, flux_src_n, flux_gsrc_n)
        end if

    end subroutine s_finalize_rhs_module

end module m_rhs
