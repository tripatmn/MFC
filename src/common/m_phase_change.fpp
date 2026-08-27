!>
!! @file
!! @brief Contains module m_phase_change

#:include 'case.fpp'
#:include 'macros.fpp'

!> @brief Phase transition relaxation solvers for liquid-vapor flows with cavitation and boiling
module m_phase_change

    use m_derived_types
    use m_global_parameters
    use m_mpi_proxy
    use m_variables_conversion
    use ieee_arithmetic
    use m_helper_basic
    use m_constants, only: model_eqns_6eq

    implicit none

    private
    public :: s_initialize_phasechange_module, s_relaxation_solver, s_infinite_relaxation_k, &
        & s_apply_model3_vapor_delta_to_fuel_species, s_finalize_relaxation_solver_module, delta_m_vapor

    !> @name Parameters for the first order transition phase change
    !> @{
    integer, parameter  :: max_iter = 100000            !< max Newton iterations before accepting the last iterate
    real(wp), parameter :: pCr = 1.817e6_wp             !< Critical pressure of n-dodecane [Pa]
    real(wp), parameter :: TCr = 658.2_wp               !< Critical temperature of n-dodecane [K]
    integer, parameter  :: ptg_ls_max = 30              !< max backtracking-line-search halvings in the pTg solver
    real(wp), parameter :: mixM = 1.0e-8_wp             !< Mixture mass fraction threshold for triggering phase change
    integer, parameter  :: lp = 1                       !< liquid dodecane fluid id
    integer, parameter  :: vp = 2                       !< dodecane vapor fluid id
    !> @}

    type(scalar_field) :: delta_m_vapor  !< Cell-local change in vapor-fluid partial density from phase change
    $:GPU_DECLARE(create='[delta_m_vapor]')

contains

    !> Dispatch to the correct relaxation solver. Replaces the procedure pointer, which CCE is breaking on.
    impure subroutine s_relaxation_solver(q_cons_vf)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        ! This is empty because in current master the procedure pointer was never assigned

        @:ASSERT(.false., "s_relaxation_solver called but it currently does nothing")

    end subroutine s_relaxation_solver

    !> Initialize the phase change module.
    impure subroutine s_initialize_phasechange_module

        if (model3_chemistry_coupling) then
            @:ALLOCATE(delta_m_vapor%sf(0:m, 0:n, 0:p))
            delta_m_vapor%sf = 0._wp
            @:ACC_SETUP_SFs(delta_m_vapor)
        end if

    end subroutine s_initialize_phasechange_module

    !> Apply pT- or pTg-equilibrium relaxation with mass depletion based on the incoming state conditions.
    subroutine s_infinite_relaxation_k(q_cons_vf, phase_t_step)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        integer, optional, intent(in) :: phase_t_step
        real(wp) :: pS                    !< equilibrium pressure
        real(wp) :: TS                    !< equilibrium temperature
        real(wp) :: rhoe, dynE, rhos      !< total internal energy, kinetic energy, and total entropy
        real(wp) :: rho, rM, m1, m2, MCT  !< total density, total reacting mass, individual reacting masses
        real(wp) :: TvF                   !< total volume fraction
        real(wp) :: vapor_mass_before     !< vapor partial density before pTg mass transfer
        real(wp) :: vapor_mass_after      !< vapor partial density after candidate pTg mass transfer
        real(wp) :: no_transfer_pS        !< pT-equilibrium pressure before pTg mass transfer
        real(wp) :: no_transfer_TS        !< pT-equilibrium temperature before pTg mass transfer
        real(wp) :: pt_iter_max_loc, pt_cap_hits_loc
        real(wp) :: ptg_cells_loc, ptg_iter_max_loc, ptg_cap_hits_loc, ptg_ls_max_loc, ptg_cap_res_max_loc
        real(wp) :: pt_iter_max_glb, pt_cap_hits_glb
        real(wp) :: ptg_cells_glb, ptg_iter_max_glb, ptg_cap_hits_glb, ptg_ls_max_glb, ptg_cap_res_max_glb
        ! $:GPU_DECLARE(create='[pS,TS,rhoe,dynE,rhos,rho,rM,m1,m2,MCT,TvF]')

        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: p_infpT, sk, hk, gk, ek, rhok
        #:else
            real(wp), dimension(num_fluids) :: p_infpT, sk, hk, gk, ek, rhok
        #:endif
        ! $:GPU_DECLARE(create='[p_infpT,sk,hk,gk,ek,rhok]')

        !> Generic loop iterators
        integer :: i, j, k, l
        integer :: pt_iter, pt_cap_hit, ptg_iter, ptg_cap_hit, ptg_ls_iter
        integer :: phase_perf_step
        real(wp) :: ptg_resnorm_final
        logical :: phase_perf_sample

#ifdef _CRAYFTN
#ifdef MFC_OpenACC
        ! CCE 19 IPA workaround: prevent bring_routine_resident SIGSEGV DIR$ NOINLINE s_infinite_pt_relaxation_k DIR$ NOINLINE
        ! s_infinite_ptg_relaxation_k DIR$ NOINLINE s_correct_partial_densities
#endif
#endif

        ! starting equilibrium solver

        phase_perf_step = -1
        phase_perf_sample = .false.
        if (present(phase_t_step)) then
            phase_perf_step = phase_t_step
            if (t_step_print > 0) phase_perf_sample = mod(phase_t_step - t_step_start, t_step_print) == 0
        end if

        pt_iter_max_loc = 0._wp; pt_cap_hits_loc = 0._wp
        ptg_cells_loc = 0._wp; ptg_iter_max_loc = 0._wp; ptg_cap_hits_loc = 0._wp
        ptg_ls_max_loc = 0._wp; ptg_cap_res_max_loc = 0._wp

        $:GPU_PARALLEL_LOOP(collapse=3, private='[i, j, k, l, p_infpT, sk, hk, gk, ek, rhok, pS, TS, rhoe, dynE, rhos, rho, rM, &
                            & m1, m2, MCT, TvF, vapor_mass_before, vapor_mass_after, no_transfer_pS, no_transfer_TS, &
                            & pt_iter, pt_cap_hit, ptg_iter, ptg_cap_hit, ptg_ls_iter, ptg_resnorm_final]', &
                            & reduction='[[pt_iter_max_loc, ptg_iter_max_loc, ptg_ls_max_loc, ptg_cap_res_max_loc], &
                            & [pt_cap_hits_loc, ptg_cells_loc, ptg_cap_hits_loc]]', reductionOp='[MAX, SUM]')
        do j = 0, m
            do k = 0, n
                do l = 0, p
                    if (model3_chemistry_coupling) delta_m_vapor%sf(j, k, l) = 0._wp

                    rho = 0.0_wp; TvF = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        ! Mixture density
                        rho = rho + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)

                        ! Total Volume Fraction
                        TvF = TvF + q_cons_vf(i + eqn_idx%adv%beg - 1)%sf(j, k, l)
                    end do

                    ! calculating the total reacting mass for the phase change process. By hypothesis, this should not change
                    ! throughout the phase-change process.
                    rM = q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) + q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)

                    ! correcting negative (reacting) mass fraction values in case they happen
                    call s_correct_partial_densities(MCT, q_cons_vf, rM, j, k, l)

                    ! fixing m1 and m2 AFTER correcting the partial densities. Note that these values must be stored for the phase
                    ! change process that will happen a posteriori
                    m1 = q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l)

                    m2 = q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)
                    vapor_mass_before = m2

                    ! kinetic energy as an auxiliary variable to the calculation of the total internal energy
                    dynE = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = eqn_idx%mom%beg, eqn_idx%mom%end
                        dynE = dynE + 5.0e-1_wp*q_cons_vf(i)%sf(j, k, l)**2/rho
                    end do

                    ! calculating the total energy that MUST be preserved throughout the pT- and pTg-relaxation procedures at each
                    ! of the cells. The internal energy is calculated as the total energy minus the kinetic energy to preserved its
                    ! value at sharp interfaces
                    rhoe = q_cons_vf(eqn_idx%E)%sf(j, k, l) - dynE

                    ! Calling pT-equilibrium for either finishing phase-change module, or as an IC for the pTg-equilibrium for this
                    ! case, MFL cannot be either 0 or 1, so I chose it to be 2
                    call s_infinite_pt_relaxation_k(j, k, l, 2, pS, p_infpT, q_cons_vf, rhoe, TS, pt_iter, pt_cap_hit)
                    if (phase_perf_sample) then
                        pt_iter_max_loc = max(pt_iter_max_loc, real(pt_iter, wp))
                        if (pt_cap_hit /= 0) pt_cap_hits_loc = pt_cap_hits_loc + 1._wp
                    end if
                    no_transfer_pS = pS
                    no_transfer_TS = TS

                    ! Check if pTg-equilibrium needed; only partial densities require updating
                    if ((relax_model == 6) .and. ((q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, &
                        & l) > mixM*rM) .and. (q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, &
                        & l) > mixM*rM)) .and. (pS < pCr) .and. (TS < TCr)) then
                        ! Solve pTg-equilibrium directly on the actual reacting masses. The Newton solver projects
                        ! the liquid mass onto [0, mT], so it recovers the single-phase limits itself (ml -> 0 for
                        ! all-vapor, ml -> mT for all-liquid). The former overheated-vapor / subcooled-liquid pT
                        ! shortcuts were removed: their pT states differ O(1) from the pTg equilibrium, so the
                        ! sub-ULP shortcut/pTg branch decision flipped across backends (CPU vs GPU) near a phase
                        ! boundary and destroyed cross-backend reproducibility.
                        q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = m1
                        q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = m2
                        if (model3_chemistry_coupling) vapor_mass_before = q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)

                        call s_infinite_ptg_relaxation_k(j, k, l, pS, rhoe, q_cons_vf, TS, ptg_iter, ptg_cap_hit, &
                                                         & ptg_ls_iter, ptg_resnorm_final)
                        if (phase_perf_sample) then
                            ptg_cells_loc = ptg_cells_loc + 1._wp
                            ptg_iter_max_loc = max(ptg_iter_max_loc, real(ptg_iter, wp))
                            ptg_ls_max_loc = max(ptg_ls_max_loc, real(ptg_ls_iter, wp))
                            if (ptg_cap_hit /= 0) then
                                ptg_cap_hits_loc = ptg_cap_hits_loc + 1._wp
                                ptg_cap_res_max_loc = max(ptg_cap_res_max_loc, ptg_resnorm_final)
                            end if
                        end if

                        if (model3_chemistry_coupling) then
                            vapor_mass_after = q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)
                            if (vapor_mass_after < vapor_mass_before) then
                                q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = m1
                                q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = m2
                                pS = no_transfer_pS
                                TS = no_transfer_TS
                                delta_m_vapor%sf(j, k, l) = 0._wp
                            else
                                delta_m_vapor%sf(j, k, l) = vapor_mass_after - vapor_mass_before
                            end if
                        end if
                    end if

                    ! Calculations AFTER equilibrium

                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        ! entropy
                        sk(i) = cvs(i)*log((TS**gs_min(i))/((pS + ps_inf(i))**(gs_min(i) - 1.0_wp))) + qvps(i)

                        ! enthalpy
                        hk(i) = gs_min(i)*cvs(i)*TS + qvs(i)

                        ! Gibbs-free energy
                        gk(i) = hk(i) - TS*sk(i)

                        ! densities
                        rhok(i) = (pS + ps_inf(i))/((gs_min(i) - 1)*cvs(i)*TS)

                        ! internal energy
                        ek(i) = (pS + gs_min(i)*ps_inf(i))/(pS + ps_inf(i))*cvs(i)*TS + qvs(i)
                    end do

                    ! calculating volume fractions, internal energies, and total entropy
                    rhos = 0.0_wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        ! volume fractions
                        q_cons_vf(i + eqn_idx%adv%beg - 1)%sf(j, k, l) = q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)/rhok(i)

                        ! alpha*rho*e
                        if (model_eqns == model_eqns_6eq) then
                            q_cons_vf(i + eqn_idx%int_en%beg - 1)%sf(j, k, l) = q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, &
                                      & l)*ek(i)
                        end if

                        ! Total entropy
                        rhos = rhos + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*sk(i)
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (phase_perf_sample) then
            $:GPU_WAIT()

            if (num_procs > 1) then
                call s_mpi_allreduce_max(pt_iter_max_loc, pt_iter_max_glb)
                call s_mpi_allreduce_sum(pt_cap_hits_loc, pt_cap_hits_glb)
                call s_mpi_allreduce_sum(ptg_cells_loc, ptg_cells_glb)
                call s_mpi_allreduce_max(ptg_iter_max_loc, ptg_iter_max_glb)
                call s_mpi_allreduce_sum(ptg_cap_hits_loc, ptg_cap_hits_glb)
                call s_mpi_allreduce_max(ptg_ls_max_loc, ptg_ls_max_glb)
                call s_mpi_allreduce_max(ptg_cap_res_max_loc, ptg_cap_res_max_glb)
            else
                pt_iter_max_glb = pt_iter_max_loc
                pt_cap_hits_glb = pt_cap_hits_loc
                ptg_cells_glb = ptg_cells_loc
                ptg_iter_max_glb = ptg_iter_max_loc
                ptg_cap_hits_glb = ptg_cap_hits_loc
                ptg_ls_max_glb = ptg_ls_max_loc
                ptg_cap_res_max_glb = ptg_cap_res_max_loc
            end if

            if (proc_rank == 0) then
                print '(" PHASE PERF step=", I0, " pt_iter_max=", I0, " pt_cap_hits=", I0, &
                    & " ptg_cells=", I0, " ptg_iter_max=", I0, " ptg_cap_hits=", I0, &
                    & " ptg_ls_max=", I0, " ptg_cap_res_max=", ES12.5)', &
                    & phase_perf_step, nint(pt_iter_max_glb), nint(pt_cap_hits_glb), nint(ptg_cells_glb), &
                    & nint(ptg_iter_max_glb), nint(ptg_cap_hits_glb), nint(ptg_ls_max_glb), ptg_cap_res_max_glb
            end if
        end if

    end subroutine s_infinite_relaxation_k

    !> Add accepted vaporized fuel mass to the coupled gas species state exactly once.
    subroutine s_apply_model3_vapor_delta_to_fuel_species(q_cons_vf)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        integer :: fuel_eqn
        integer :: j, k, l

        if (.not. model3_chemistry_coupling) return
        if (.not. associated(delta_m_vapor%sf)) return

        fuel_eqn = eqn_idx%species%beg + fuel_species_id - 1
        if ((fuel_eqn < eqn_idx%species%beg) .or. (fuel_eqn > eqn_idx%species%end)) then
            call s_mpi_abort("model3_chemistry_coupling fuel_species_id is outside the chemistry species range")
        end if

        $:GPU_PARALLEL_LOOP(collapse=3, private='[j, k, l]', copyin='[fuel_eqn]')
        do j = 0, m
            do k = 0, n
                do l = 0, p
                    q_cons_vf(fuel_eqn)%sf(j, k, l) = q_cons_vf(fuel_eqn)%sf(j, k, l) + delta_m_vapor%sf(j, k, l)
                    delta_m_vapor%sf(j, k, l) = 0._wp
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_apply_model3_vapor_delta_to_fuel_species

    !> Apply pT-equilibrium relaxation for N fluids
    !! @param MFL flag: 0=gas, 1=liquid, 2=mixture
    subroutine s_infinite_pt_relaxation_k(j, k, l, MFL, pS, p_infpT, q_cons_vf, rhoe, TS, pt_iter, pt_cap_hit)

        $:GPU_ROUTINE(function_name='s_infinite_pt_relaxation_k', parallelism='[seq]', cray_noinline=True)

        ! initializing variables
        integer, intent(in)                                 :: j, k, l, MFL
        real(wp), intent(out)                               :: pS
        real(wp), dimension(1:), intent(out)                :: p_infpT
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        real(wp), intent(in)                                :: rhoe
        real(wp), intent(out)                               :: TS
        integer, intent(out)                                :: pt_iter
        integer, intent(out)                                :: pt_cap_hit
        real(wp)                                            :: gp, gpp, hp, pO, mCP, mQ  !< variables for the Newton Solver
        real(wp)                                            :: p_infpT_sum
        integer                                             :: i, ns                     !< generic loop iterators
        ! auxiliary variables for the pT-equilibrium solver
        pt_iter = 0
        pt_cap_hit = 0
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
            mCP = mCP + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*cvs(i)*gs_min(i)

            ! sum of the total alpha*rho*q of the system
            mQ = mQ + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*qvs(i)
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
                ! Assigning zero values for mass depletion cases pressure
                pS = 0.0_wp

                ! temperature
                TS = 0.0_wp

                return
            end if
        end if

        ! calculating initial estimate for pressure in the pT-relaxation procedure. I will also use this variable to iterate over
        ! the Newton's solver
        pO = 0.0_wp

        ! Maybe improve this condition afterwards. As long as the initial guess is in between -min(ps_inf) and infinity, a solution
        ! should be able to be found.
        pS = 1.0e4_wp

        ! Newton Solver for the pT-equilibrium
        ns = 0
        ! change this relative error metric. 1.e4_wp is just arbitrary
        ! Relative criterion written in multiply form to avoid dividing by pO (pO = 0 on the first pass).
        do while ((abs(pS - pO) > palpha_eps) .and. (abs(pS - pO) > (palpha_eps/1.e4_wp)*abs(pO)) .or. (ns == 0))
            ! increasing counter
            ns = ns + 1
            pt_iter = ns
            ! guard against non-convergence: accept the last iterate rather than looping forever
            if (ns >= max_iter) then
                pt_cap_hit = 1
                exit
            end if

            ! updating old pressure
            pO = pS

            ! updating functions used in the Newton's solver
            gpp = 0.0_wp; gp = 0.0_wp; hp = 0.0_wp
            $:GPU_LOOP(parallelism='[seq]')
            do i = 1, num_fluids
                gp = gp + (gs_min(i) - 1.0_wp)*q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, &
                           & l)*cvs(i)*(rhoe + pS - mQ)/(mCP*(pS + p_infpT(i)))

                gpp = gpp + (gs_min(i) - 1.0_wp)*q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, &
                             & l)*cvs(i)*(p_infpT(i) - rhoe + mQ)/(mCP*(pS + p_infpT(i))**2)
            end do

            hp = 1.0_wp/(rhoe + pS - mQ) + 1.0_wp/(pS + minval(p_infpT))

            ! updating common pressure for the newton solver
            pS = pO + ((1.0_wp - gp)/gpp)/(1.0_wp - (1.0_wp - gp + abs(1.0_wp - gp))/(2.0_wp*gpp)*hp)
        end do

        ! common temperature
        TS = (rhoe + pS - mQ)/mCP

    end subroutine s_infinite_pt_relaxation_k

    !> Evaluate the pTg-equilibrium residual R2D and temperature TS at a trial state (ml, pS) WITHOUT mutating q_cons_vf, so the
    !! Newton driver can line-search. The total reacting mass mT is conserved, so the reacting masses are (ml, mT - ml) and only the
    !! inert fluids are read from q_cons_vf. Also returns the mixture sums the Jacobian and the final temperature need.
    subroutine s_compute_ptg_residual(ml, mT, pS, j, k, l, q_cons_vf, rhoe, R2D, TS, mCP, mQ, mCVGP, mCVGP2, mCPD)

        $:GPU_ROUTINE(function_name='s_compute_ptg_residual', parallelism='[seq]')

        real(wp), intent(in)                                :: ml, mT, pS, rhoe
        integer, intent(in)                                 :: j, k, l
        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        real(wp), dimension(2), intent(out)                 :: R2D
        real(wp), intent(out)                               :: TS, mCP, mQ, mCVGP, mCVGP2, mCPD
        real(wp)                                            :: mQD
        integer                                             :: i

        ! reacting fluids contribute via (ml, mT - ml); inert fluids are summed from q_cons_vf
        mCP = ml*cvs(lp)*gs_min(lp) + (mT - ml)*cvs(vp)*gs_min(vp)
        mQ = ml*qvs(lp) + (mT - ml)*qvs(vp)
        mCVGP = 0.0_wp; mCVGP2 = 0.0_wp; mCPD = 0.0_wp; mQD = 0.0_wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_fluids
            if ((i /= lp) .and. (i /= vp)) then
                mCP = mCP + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*cvs(i)*gs_min(i)
                mQ = mQ + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*qvs(i)
                mCVGP = mCVGP + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*cvs(i)*(gs_min(i) - 1)/(pS + ps_inf(i))
                mCVGP2 = mCVGP2 + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*cvs(i)*(gs_min(i) - 1)/((pS + ps_inf(i))**2)
                mQD = mQD + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*qvs(i)
                mCPD = mCPD + q_cons_vf(i + eqn_idx%cont%beg - 1)%sf(j, k, l)*cvs(i)*gs_min(i)
            end if
        end do

        TS = 1.0_wp/(mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) - cvs(vp) &
                     & *(gs_min(vp) - 1)/(pS + ps_inf(vp))) + mCVGP)

        ! (i) Gibbs free-energy equality
        R2D(1) = TS*((cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp))*(1 - log(TS)) - (qvps(lp) - qvps(vp)) + cvs(lp)*(gs_min(lp) - 1) &
            & *log(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)*log(pS + ps_inf(vp))) + qvs(lp) - qvs(vp)

        ! (ii) constant-energy condition
        R2D(2) = rhoe + pS + ml*(qvs(vp) - qvs(lp)) - mT*qvs(vp) - mQD + (ml*(gs_min(vp)*cvs(vp) - gs_min(lp)*cvs(lp)) &
            & - mT*gs_min(vp)*cvs(vp) - mCPD)/(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS &
            & + ps_inf(vp))) + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP)

    end subroutine s_compute_ptg_residual

    !> Apply pTg-equilibrium relaxation: a damped (backtracking line search) Newton solve for the reacting liquid mass ml and
    !! pressure pS enforcing Gibbs equality and energy conservation, converging on the residual norm (absolute ptgalpha_eps, or the
    !! rhoe-relative branch). Every step is projected onto the physical bounds 0 <= ml <= mT, pS > pmin. This converges in a handful
    !! of iterations with a bounded, uniform count (no GPU warp divergence), unlike the former fixed 1e-3 underrelaxation that
    !! stalled far from the root.
    subroutine s_infinite_ptg_relaxation_k(j, k, l, pS, rhoe, q_cons_vf, TS, ptg_iter, ptg_cap_hit, ptg_ls_iter, &
                                           & ptg_resnorm_final)

        $:GPU_ROUTINE(function_name='s_infinite_ptg_relaxation_k', parallelism='[seq]', cray_noinline=True)

        integer, intent(in)                                    :: j, k, l
        real(wp), intent(inout)                                :: pS
        real(wp), intent(in)                                   :: rhoe
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(inout)                                :: TS
        integer, intent(out)                                   :: ptg_iter, ptg_cap_hit, ptg_ls_iter
        real(wp), intent(out)                                  :: ptg_resnorm_final
        real(wp), dimension(2, 2)                              :: Jac, InvJac
        real(wp), dimension(2)                                 :: R2D, R2D_try, DeltamP
        real(wp)                                               :: mCP, mCPD, mCVGP, mCVGP2, mQ
        real(wp)                                               :: ml, ml_try, mT, pS_try, pmin, lambda, resnorm, resnorm_try
        real(wp)                                               :: dFdT, dTdm, dTdp, detJ
        integer                                                :: ns, ls

        ptg_iter = 0
        ptg_cap_hit = 0
        ptg_ls_iter = 0
        ptg_resnorm_final = 0._wp

        ! total reacting mass is conserved; the liquid mass ml is the primary unknown, vapor mass = mT - ml
        mT = q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) + q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)
        ml = q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l)

        ! recover a physical pressure guess when the incoming pS is non-physical
        if (((pS < 0.0_wp) .and. ((q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) + q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, &
            & k, &
            & l)) > ((rhoe - gs_min(lp)*ps_inf(lp)/(gs_min(lp) - 1))/qvs(lp)))) .or. ((pS >= 0.0_wp) .and. (pS < 1.0e-1_wp))) then
            pS = 1.0e4_wp
        end if

        ! pressure floor (stiffened gas requires pS + ps_inf > 0 for both phases)
        pmin = -min(ps_inf(lp), ps_inf(vp)) + 1.0_wp

        call s_compute_ptg_residual(ml, mT, pS, j, k, l, q_cons_vf, rhoe, R2D, TS, mCP, mQ, mCVGP, mCVGP2, mCPD)
        resnorm = sqrt(R2D(1)**2 + R2D(2)**2)

        do ns = 1, max_iter
            ! converged on the absolute residual, or on the rhoe-relative residual (multiply form, rhoe > 0)
            if ((resnorm <= ptgalpha_eps) .or. (resnorm <= (ptgalpha_eps/1.e6_wp)*rhoe)) exit
            ptg_iter = ns

            ! 2x2 Jacobian of (Gibbs equality, energy) with respect to (ml, pS) at the current state
            dFdT = -(cvs(lp)*gs_min(lp) - cvs(vp)*gs_min(vp))*log(TS) - (qvps(lp) - qvps(vp)) + cvs(lp)*(gs_min(lp) - 1)*log(pS &
                     & + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)*log(pS + ps_inf(vp))
            dTdm = -(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)))*TS**2
            dTdp = (mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2 + ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp))**2 - cvs(vp) &
                    & *(gs_min(vp) - 1)/(pS + ps_inf(vp))**2) + mCVGP2)*TS**2

            Jac(1, 1) = dFdT*dTdm
            Jac(1, 2) = dFdT*dTdp + TS*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)))
            Jac(2, &
                & 1) = qvs(vp) - qvs(lp) + (cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp))/(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS &
                & + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) &
                & + mCVGP) - (ml*(cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp)) - mT*cvs(vp)*gs_min(vp) - mCPD)*(cvs(lp)*(gs_min(lp) &
                & - 1)/(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)))/((ml*(cvs(lp)*(gs_min(lp) - 1)/(pS &
                & + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))) + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) &
                & + mCVGP)**2)
            Jac(2, &
                & 2) = 1 + (ml*(cvs(vp)*gs_min(vp) - cvs(lp)*gs_min(lp)) - mT*cvs(vp)*gs_min(vp) - mCPD)*(ml*(cvs(lp)*(gs_min(lp) &
                & - 1)/(pS + ps_inf(lp))**2 - cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp))**2) + mT*cvs(vp)*(gs_min(vp) - 1)/(pS &
                & + ps_inf(vp))**2 + mCVGP2)/(ml*(cvs(lp)*(gs_min(lp) - 1)/(pS + ps_inf(lp)) - cvs(vp)*(gs_min(vp) - 1)/(pS &
                & + ps_inf(vp))) + mT*cvs(vp)*(gs_min(vp) - 1)/(pS + ps_inf(vp)) + mCVGP)**2

            detJ = Jac(1, 1)*Jac(2, 2) - Jac(1, 2)*Jac(2, 1)
            ! singular Jacobian: no usable Newton direction, accept the current (best) state
            if (detJ == 0.0_wp) exit

            InvJac(1, 1) = Jac(2, 2)/detJ
            InvJac(1, 2) = -Jac(1, 2)/detJ
            InvJac(2, 1) = -Jac(2, 1)/detJ
            InvJac(2, 2) = Jac(1, 1)/detJ

            DeltamP(1) = -(InvJac(1, 1)*R2D(1) + InvJac(1, 2)*R2D(2))
            DeltamP(2) = -(InvJac(2, 1)*R2D(1) + InvJac(2, 2)*R2D(2))

            ! backtracking line search: halve the step until the residual decreases, keeping the state
            ! physical (0 <= ml <= mT, pS above the stiffened-gas floor)
            lambda = 1.0_wp
            do ls = 1, ptg_ls_max
                ml_try = min(max(ml + lambda*DeltamP(1), 0.0_wp), mT)
                pS_try = max(pS + lambda*DeltamP(2), pmin)
                call s_compute_ptg_residual(ml_try, mT, pS_try, j, k, l, q_cons_vf, rhoe, R2D_try, TS, mCP, mQ, mCVGP, mCVGP2, mCPD)
                resnorm_try = sqrt(R2D_try(1)**2 + R2D_try(2)**2)
                if ((resnorm_try < resnorm) .or. (ls == ptg_ls_max)) exit
                lambda = 0.5_wp*lambda
            end do
            ptg_ls_iter = max(ptg_ls_iter, ls)

            ! accept the trial state (TS, mCP, mQ, mCVGP, mCVGP2, mCPD already set to it by the last call)
            ml = ml_try; pS = pS_try; R2D = R2D_try; resnorm = resnorm_try
            if (ns == max_iter) then
                if (.not. ((resnorm <= ptgalpha_eps) .or. (resnorm <= (ptgalpha_eps/1.e6_wp)*rhoe))) ptg_cap_hit = 1
            end if
        end do
        ptg_resnorm_final = resnorm

        ! commit the reacting masses (mT conserved) and set the common temperature
        q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = ml
        q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = mT - ml

        TS = (rhoe + pS - mQ)/mCP

    end subroutine s_infinite_ptg_relaxation_k

    !> Correct the partial densities of the reacting fluids in case one of them is negative but their sum is positive. Inert phases
    !! are not corrected at this moment
    subroutine s_correct_partial_densities(MCT, q_cons_vf, rM, j, k, l)

        $:GPU_ROUTINE(function_name='s_correct_partial_densities', parallelism='[seq]', cray_noinline=True)

        !> @name variables for the correction of the reacting partial densities
        !> @{
        real(wp), intent(out)                                  :: MCT
        type(scalar_field), dimension(sys_size), intent(inout) :: q_cons_vf
        real(wp), intent(inout)                                :: rM
        integer, intent(in)                                    :: j, k, l
        !> @}
        if (rM < 0.0_wp) then
            if ((q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, &
                & l) >= -1.0_wp*mixM) .and. (q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) >= -1.0_wp*mixM)) then
                q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = 0.0_wp

                q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = 0.0_wp

                rM = q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) + q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l)
            end if
        end if

        ! TODO: Consider partitioning partial densities instead of absolute-value correction
        MCT = 2*mixM

        ! correcting the partial densities of the reacting fluids. What to do for the nonreacting ones?
        if (q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) < 0.0_wp) then
            q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = MCT*rM

            q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = (1.0_wp - MCT)*rM
        else if (q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) < 0.0_wp) then
            q_cons_vf(lp + eqn_idx%cont%beg - 1)%sf(j, k, l) = (1.0_wp - MCT)*rM

            q_cons_vf(vp + eqn_idx%cont%beg - 1)%sf(j, k, l) = MCT*rM
        end if

    end subroutine s_correct_partial_densities

    !> Finalize the phase change module
    impure subroutine s_finalize_relaxation_solver_module

        if (associated(delta_m_vapor%sf)) then
            @:DEALLOCATE(delta_m_vapor%sf)
        end if

    end subroutine s_finalize_relaxation_solver_module

end module m_phase_change
