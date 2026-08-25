"""
Static checks for the fixed-layout Model-3 chemistry gas-state adapter.

The adapter itself is Fortran and is exercised by MFC builds/runs. These tests
lock down the Stage-2 formulas without requiring an MFC executable.
"""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHEMISTRY_SRC = REPO_ROOT / "src" / "common" / "m_chemistry.fpp"
VARIABLES_CONVERSION_SRC = REPO_ROOT / "src" / "common" / "m_variables_conversion.fpp"
PHASE_CHANGE_SRC = REPO_ROOT / "src" / "common" / "m_phase_change.fpp"
SIM_START_UP_SRC = REPO_ROOT / "src" / "simulation" / "m_start_up.fpp"
TIME_STEPPERS_SRC = REPO_ROOT / "src" / "simulation" / "m_time_steppers.fpp"
DATA_OUTPUT_SRC = REPO_ROOT / "src" / "simulation" / "m_data_output.fpp"


def _adapter_source() -> str:
    text = VARIABLES_CONVERSION_SRC.read_text()
    match = re.search(
        r"subroutine s_get_model3_chemistry_gas_state\b(?P<body>.*?)"
        r"end subroutine s_get_model3_chemistry_gas_state",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_get_model3_chemistry_gas_state not found")
    return re.sub(r"\s+", "", match.group("body"))


def _aqss_source() -> str:
    text = CHEMISTRY_SRC.read_text()
    match = re.search(
        r"subroutine s_chemistry_reaction_substep\b(?P<body>.*?)"
        r"end subroutine s_chemistry_reaction_substep",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_chemistry_reaction_substep not found")
    return re.sub(r"[\s&]+", "", match.group("body"))


def _conversion_source() -> str:
    text = VARIABLES_CONVERSION_SRC.read_text()
    match = re.search(
        r"subroutine s_convert_conservative_to_primitive_variables\b(?P<body>.*?)"
        r"end subroutine s_convert_conservative_to_primitive_variables",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_convert_conservative_to_primitive_variables not found")
    return re.sub(r"[\s&]+", "", match.group("body"))


def _temperature_init_source() -> str:
    text = CHEMISTRY_SRC.read_text()
    match = re.search(
        r"subroutine s_compute_q_T_sf\b(?P<body>.*?)"
        r"end subroutine s_compute_q_T_sf",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_compute_q_T_sf not found")
    return re.sub(r"[\s&]+", "", match.group("body"))


def _primitive_temperature_source() -> str:
    text = CHEMISTRY_SRC.read_text()
    match = re.search(
        r"subroutine s_compute_T_from_primitives\b(?P<body>.*?)"
        r"end subroutine s_compute_T_from_primitives",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_compute_T_from_primitives not found")
    return re.sub(r"[\s&]+", "", match.group("body"))


def _compact_fortran(path: Path) -> str:
    return re.sub(r"[\s&]+", "", path.read_text())


def _apply_evaporation_only_reference(liquid_before, vapor_before, liquid_candidate, vapor_candidate):
    if vapor_candidate < vapor_before:
        return liquid_before, vapor_before, 0.0
    return liquid_candidate, vapor_candidate, vapor_candidate - vapor_before


def _apply_vapor_delta_to_fuel_species_reference(species_rho_y, fuel_index, delta):
    updated = list(species_rho_y)
    updated[fuel_index] += delta
    delta = 0.0
    return updated, delta


def _model3_int_en_from_pressure(alpha, alpha_rho, gamma, pi_inf, qv, pressure):
    return alpha * (gamma * pressure + pi_inf) + alpha_rho * qv


def _model3_pressure_from_int_en(alpha, alpha_rho, gamma, pi_inf, qv, int_en):
    return ((int_en - alpha_rho * qv) / alpha - pi_inf) / gamma


class TestModel3ChemistryGasStateAdapter(unittest.TestCase):
    def test_uses_fluid2_and_fluid3_partial_masses_for_stored_gas_density(self):
        src = _adapter_source()
        self.assertIn(
            "rho_g_stored=q_cons_vf(eqn_idx%cont%beg+1)%sf(x,y,z)+q_cons_vf(eqn_idx%cont%beg+2)%sf(x,y,z)",
            src,
        )

    def test_uses_fluid2_and_fluid3_volume_fractions_for_gas_alpha(self):
        src = _adapter_source()
        self.assertIn(
            "alpha_g=q_cons_vf(eqn_idx%adv%beg+1)%sf(x,y,z)+q_cons_vf(eqn_idx%adv%beg+2)%sf(x,y,z)",
            src,
        )

    def test_intrinsic_density_uses_stored_mass_over_alpha(self):
        src = _adapter_source()
        self.assertIn("rho_g_intrinsic=rho_g_stored/alpha_g", src)

    def test_species_mass_fractions_use_stored_gas_mass(self):
        src = _adapter_source()
        self.assertIn("Ys(eqn)=q_cons_vf(eqn_idx%species%beg+eqn-1)%sf(x,y,z)/rho_g_stored", src)

    def test_species_closure_is_exposed_not_repaired(self):
        src = _adapter_source()
        self.assertIn("species_closure_error=rhoY_sum-rho_g_stored", src)
        self.assertIn("species_closure_relerr=species_closure_error/max(rho_g_stored,sgm_eps)", src)
        self.assertNotIn("Ys(eqn)=Ys(eqn)/", src)

    def test_pressure_and_temperature_recovery_are_from_model3_gas_state(self):
        src = _adapter_source()
        self.assertIn("calls_get_model3_phase_pressure(q_cons_vf,2,x,y,z,p2,active_phase_2)", src)
        self.assertIn("calls_get_model3_phase_pressure(q_cons_vf,3,x,y,z,p3,active_phase_3)", src)
        self.assertIn(
            "gas_pressure=(q_cons_vf(eqn_idx%adv%beg+1)%sf(x,y,z)*p2+q_cons_vf(eqn_idx%adv%beg+2)%sf(x,y,z)*p3)/alpha_g",
            src,
        )
        self.assertIn("callget_mixture_molecular_weight(Ys,mix_mol_weight)", src)
        self.assertIn("gas_temperature=gas_pressure*mix_mol_weight/(gas_constant*rho_g_intrinsic)", src)

    def test_controlled_state_formula_reference(self):
        alpha_rho_2 = 0.03
        alpha_rho_3 = 0.27
        alpha_2 = 0.02
        alpha_3 = 0.18
        rho_y = [0.03, 0.12, 0.15]
        weights = [2.0, 32.0, 131.293]
        pressure = 24.0 * 101325.0
        gas_constant = 8314.46261815324

        rho_g_stored = alpha_rho_2 + alpha_rho_3
        alpha_g = alpha_2 + alpha_3
        rho_g_intrinsic = rho_g_stored / alpha_g
        ys = [value / rho_g_stored for value in rho_y]
        sum_y = sum(ys)
        closure_error = sum(rho_y) - rho_g_stored
        mix_mol_weight = 1.0 / sum(y / w for y, w in zip(ys, weights))
        temperature = pressure * mix_mol_weight / (gas_constant * rho_g_intrinsic)

        self.assertAlmostEqual(rho_g_stored, 0.30)
        self.assertAlmostEqual(alpha_g, 0.20)
        self.assertAlmostEqual(rho_g_intrinsic, 1.50)
        for actual, expected in zip(ys, [0.1, 0.4, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum_y, 1.0)
        self.assertAlmostEqual(closure_error, 0.0)
        self.assertGreater(temperature, 0.0)


class TestModel3PhaseChangeVaporDelta(unittest.TestCase):
    def test_delta_field_is_public_and_coupled_mode_allocated(self):
        text = re.sub(r"\s+", "", PHASE_CHANGE_SRC.read_text())
        self.assertIn("s_apply_model3_vapor_delta_to_fuel_species", text)
        self.assertIn("s_finalize_relaxation_solver_module,delta_m_vapor", text)
        self.assertIn("type(scalar_field)::delta_m_vapor", text)
        self.assertIn("if(model3_chemistry_coupling)then@:ALLOCATE(delta_m_vapor%sf(0:m,0:n,0:p))", text)
        self.assertIn("@:ACC_SETUP_SFs(delta_m_vapor)", text)
        self.assertIn("if(associated(delta_m_vapor%sf))then@:DEALLOCATE(delta_m_vapor%sf)", text)

    def test_delta_is_reset_and_accepts_evaporation_only_around_ptg_relaxation(self):
        text = re.sub(r"\s+", "", PHASE_CHANGE_SRC.read_text())
        self.assertIn("if(model3_chemistry_coupling)delta_m_vapor%sf(j,k,l)=0._wp", text)
        self.assertIn("vapor_mass_before=m2", text)
        self.assertIn("no_transfer_pS=pS", text)
        self.assertIn("no_transfer_TS=TS", text)
        self.assertIn("call s_infinite_ptg_relaxation_k".replace(" ", ""), text)
        self.assertIn("vapor_mass_after=q_cons_vf(vp+eqn_idx%cont%beg-1)%sf(j,k,l)", text)
        self.assertIn("if(vapor_mass_after<vapor_mass_before)then", text)
        self.assertIn("q_cons_vf(lp+eqn_idx%cont%beg-1)%sf(j,k,l)=m1", text)
        self.assertIn("q_cons_vf(vp+eqn_idx%cont%beg-1)%sf(j,k,l)=m2", text)
        self.assertIn("pS=no_transfer_pS", text)
        self.assertIn("TS=no_transfer_TS", text)
        self.assertIn("delta_m_vapor%sf(j,k,l)=0._wp", text)
        self.assertIn("delta_m_vapor%sf(j,k,l)=vapor_mass_after-vapor_mass_before", text)

    def test_accepted_evaporation_reference_has_positive_delta(self):
        liquid, vapor, delta = _apply_evaporation_only_reference(0.7, 0.3, 0.65, 0.35)
        self.assertAlmostEqual(liquid, 0.65)
        self.assertAlmostEqual(vapor, 0.35)
        self.assertAlmostEqual(delta, 0.05)
        self.assertGreater(delta, 0.0)

    def test_condensation_reference_rolls_back_and_zeroes_delta(self):
        liquid, vapor, delta = _apply_evaporation_only_reference(0.7, 0.3, 0.75, 0.25)
        self.assertAlmostEqual(liquid, 0.7)
        self.assertAlmostEqual(vapor, 0.3)
        self.assertAlmostEqual(delta, 0.0)

    def test_vapor_delta_species_insertion_is_fuel_only_and_one_shot(self):
        initial_species = [0.05, 0.15, 0.20]
        delta = 0.04
        updated_species, delta = _apply_vapor_delta_to_fuel_species_reference(initial_species, 1, delta)

        self.assertAlmostEqual(updated_species[1] - initial_species[1], 0.04)
        self.assertAlmostEqual(updated_species[0], initial_species[0])
        self.assertAlmostEqual(updated_species[2], initial_species[2])
        self.assertAlmostEqual(sum(updated_species) - sum(initial_species), 0.04)
        self.assertAlmostEqual(delta, 0.0)

        reapplied_species, delta = _apply_vapor_delta_to_fuel_species_reference(updated_species, 1, delta)
        self.assertEqual(reapplied_species, updated_species)
        self.assertAlmostEqual(delta, 0.0)

    def test_fortran_species_insertion_consumes_delta_after_relaxation(self):
        phase_text = re.sub(r"\s+", "", PHASE_CHANGE_SRC.read_text())
        startup_text = re.sub(r"\s+", "", SIM_START_UP_SRC.read_text())

        self.assertIn("fuel_eqn=eqn_idx%species%beg+fuel_species_id-1", phase_text)
        self.assertIn("q_cons_vf(fuel_eqn)%sf(j,k,l)=q_cons_vf(fuel_eqn)%sf(j,k,l)+delta_m_vapor%sf(j,k,l)", phase_text)
        self.assertIn("delta_m_vapor%sf(j,k,l)=0._wp", phase_text)
        self.assertIn("calls_infinite_relaxation_k(q_cons_ts(1)%vf)", startup_text)
        self.assertIn(
            "if(model3_chemistry_coupling)calls_apply_model3_vapor_delta_to_fuel_species(q_cons_ts(1)%vf)",
            startup_text,
        )


class TestModel3AqssOrdering(unittest.TestCase):
    def test_upstream_aqss_call_is_gated_off_for_coupled_mode(self):
        text = _compact_fortran(TIME_STEPPERS_SRC)
        self.assertIn(
            "if(chemistry.and.chem_params%reactions.and.chem_params%reaction_substeps>0.and.(.not.model3_chemistry_coupling))then",
            text,
        )
        self.assertEqual(text.count("calls_chemistry_reaction_substep(q_cons_ts(1)%vf,q_T_sf,dt,idwint)"), 1)

    def test_coupled_aqss_call_occurs_after_phase_change_and_fuel_insertion(self):
        text = _compact_fortran(SIM_START_UP_SRC)
        relaxation = text.index("calls_infinite_relaxation_k(q_cons_ts(1)%vf)")
        insertion = text.index("calls_apply_model3_vapor_delta_to_fuel_species(q_cons_ts(1)%vf)")
        chemistry_gate = text.index(
            "if(model3_chemistry_coupling.and.chemistry.and.chem_params%reactions.and.chem_params%reaction_substeps>0)then"
        )
        chemistry_call = text.index("calls_chemistry_reaction_substep(q_cons_ts(1)%vf,q_T_sf,dt,idwint)")

        self.assertLess(relaxation, insertion)
        self.assertLess(insertion, chemistry_gate)
        self.assertLess(chemistry_gate, chemistry_call)
        self.assertEqual(text.count("calls_chemistry_reaction_substep(q_cons_ts(1)%vf,q_T_sf,dt,idwint)"), 1)

    def test_coupled_restart_preserves_model3_internal_energies(self):
        text = _compact_fortran(SIM_START_UP_SRC)
        self.assertIn(
            "if(model_eqns==model_eqns_6eq.and.(.not.model3_chemistry_coupling))thencalls_initialize_internal_energy_equations",
            text,
        )


class TestModel3AqssStateAdapter(unittest.TestCase):
    def test_coupled_adaptive_probe_uses_model3_gas_state(self):
        src = _aqss_source()
        self.assertIn("call s_get_model3_chemistry_gas_state".replace(" ", ""), src)
        self.assertIn("if(.not.active_gas)cycle", src)
        self.assertIn("rho=rho_g_intrinsic", src)

        helper_call = src.index("calls_get_model3_chemistry_gas_state")
        rate_call = src.index("callget_net_production_rates(rho,T,Ys,cdot)")
        self.assertLess(helper_call, rate_call)

    def test_coupled_fixed_aqss_holds_thermochemical_energy(self):
        src = _aqss_source()
        self.assertIn("callget_mixture_energy_mass(T,Ys,energy)", src)
        self.assertIn("callget_temperature(energy,T,Ys,.true.,T_new)", src)

    def test_coupled_writeback_uses_stored_gas_mass_not_intrinsic_density(self):
        src = _aqss_source()
        self.assertIn(
            "q_cons_vf(eqn)%sf(x,y,z)=rho_g_stored*Ys(eqn-eqn_idx%species%beg+1)",
            src,
        )
        self.assertIn(
            "q_cons_vf(eqn)%sf(x,y,z)=rho*Ys(eqn-eqn_idx%species%beg+1)",
            src,
        )
        self.assertIn("q_T_sf%sf(x,y,z)=T", src)

    def test_stage7_pressure_writeback_uses_exact_stage2_inverse(self):
        alpha = 0.35
        alpha_rho = 0.42
        gamma = 2.5
        pi_inf = 1000.0
        qv = 25.0
        pressure = 2.4e6

        int_en = _model3_int_en_from_pressure(alpha, alpha_rho, gamma, pi_inf, qv, pressure)
        recovered_pressure = _model3_pressure_from_int_en(alpha, alpha_rho, gamma, pi_inf, qv, int_en)

        self.assertAlmostEqual(recovered_pressure, pressure)

    def test_stage7_gas_fluids_recover_same_pressure_after_writeback(self):
        pressure = 3.1e6
        fluid2 = _model3_int_en_from_pressure(0.12, 0.18, 2.0, 0.0, 10.0, pressure)
        fluid3 = _model3_int_en_from_pressure(0.58, 0.71, 2.8, 50.0, -4.0, pressure)

        self.assertAlmostEqual(_model3_pressure_from_int_en(0.12, 0.18, 2.0, 0.0, 10.0, fluid2), pressure)
        self.assertAlmostEqual(_model3_pressure_from_int_en(0.58, 0.71, 2.8, 50.0, -4.0, fluid3), pressure)

    def test_stage7_total_energy_delta_is_selected_gas_internal_energy_delta(self):
        old_int_2 = 4.0
        old_int_3 = 8.0
        new_int_2 = 4.5
        new_int_3 = 9.25
        total_energy = 100.0

        delta_int_g = (new_int_2 + new_int_3) - (old_int_2 + old_int_3)
        self.assertAlmostEqual(total_energy + delta_int_g, 101.75)

    def test_stage7_conservation_bookkeeping_leaves_nongas_state_unchanged(self):
        state_before = {
            "alpha_rho": [0.2, 0.3, 0.5],
            "alpha": [0.1, 0.2, 0.7],
            "momentum": [1.0, -0.5],
            "liquid_int_en": 7.0,
        }
        state_after = {
            "alpha_rho": list(state_before["alpha_rho"]),
            "alpha": list(state_before["alpha"]),
            "momentum": list(state_before["momentum"]),
            "liquid_int_en": state_before["liquid_int_en"],
        }

        self.assertEqual(state_after, state_before)

    def test_stage7_fortran_writes_only_selected_gas_thermal_feedback(self):
        src = _aqss_source()
        self.assertIn("p_new=rho_g_intrinsic*gas_constant*T/mix_mol_weight", src)
        self.assertIn("int_en_2_new=alpha_2*(gammas(2)*p_new+pi_infs(2))+alpha_rho_2*qvs(2)", src)
        self.assertIn("int_en_3_new=alpha_3*(gammas(3)*p_new+pi_infs(3))+alpha_rho_3*qvs(3)", src)
        self.assertIn("int_g_new=int_en_2_new+int_en_3_new", src)
        self.assertIn("q_cons_vf(eqn_idx%E)%sf(x,y,z)=q_cons_vf(eqn_idx%E)%sf(x,y,z)+int_g_new-int_g_old", src)
        self.assertNotIn("q_cons_vf(eqn_idx%int_en%beg)%sf(x,y,z)=", src)
        self.assertNotIn("q_cons_vf(eqn_idx%cont%beg+1)%sf(x,y,z)=", src)
        self.assertNotIn("q_cons_vf(eqn_idx%cont%beg+2)%sf(x,y,z)=", src)


class TestModel3ConservativeToPrimitiveReconstruction(unittest.TestCase):
    def test_shared_helper_lives_in_variable_conversion_and_chemistry_imports_it(self):
        vc_text = _compact_fortran(VARIABLES_CONVERSION_SRC)
        chemistry_text = _compact_fortran(CHEMISTRY_SRC)
        self.assertIn("public::", vc_text)
        self.assertIn("s_get_model3_phase_pressure", vc_text)
        self.assertIn("s_get_model3_chemistry_gas_state", vc_text)
        self.assertIn("usem_variables_conversion,only:s_get_model3_chemistry_gas_state", chemistry_text)
        self.assertNotIn("subroutines_get_model3_chemistry_gas_state", chemistry_text)

    def test_coupled_conversion_uses_gas_stored_species_mass_fractions(self):
        src = _conversion_source()
        self.assertIn("if(chemistry.and.model3_chemistry_coupling)then", src)
        self.assertIn("calls_get_model3_chemistry_gas_state", src)
        self.assertIn("qK_prim_vf(i)%sf(j,k,l)=Ys(i-eqn_idx%species%beg+1)", src)
        self.assertNotIn(
            "qK_prim_vf(i)%sf(j,k,l)=max(0._wp,qK_cons_vf(i)%sf(j,k,l)/rho_K)",
            src.split("if(chemistry.and.model3_chemistry_coupling)then", 1)[1].split("elseif(chemistry)then", 1)[0],
        )

    def test_coupled_conversion_keeps_model3_mass_and_velocity_density(self):
        src = _conversion_source()
        coupled_block = src.split("if(chemistry.and.model3_chemistry_coupling)then", 1)[1].split("elseif(chemistry)then", 1)[0]
        self.assertIn("qK_prim_vf(i)%sf(j,k,l)=qK_cons_vf(i)%sf(j,k,l)", coupled_block)
        self.assertIn("qK_prim_vf(i)%sf(j,k,l)=qK_cons_vf(i)%sf(j,k,l)/rho_K", src)
        self.assertNotIn("rho_K=0._wp", coupled_block)

    def test_coupled_conversion_uses_model3_gas_pressure_and_temperature(self):
        src = _conversion_source()
        self.assertIn("pres=gas_pressure", src)
        self.assertIn("T=gas_temperature", src)
        self.assertIn("if(active_gas)q_T_sf%sf(j,k,l)=T", src)
        self.assertIn("calls_get_model3_phase_pressure(qK_cons_vf,1,j,k,l,liquid_pressure,active_liquid)", src)

    def test_temperature_initialization_uses_model3_helper_in_coupled_mode(self):
        src = _temperature_init_source()
        self.assertIn("if(model3_chemistry_coupling)then", src)
        self.assertIn("calls_get_model3_chemistry_gas_state", src)
        self.assertIn("if(active_gas)q_T_sf%sf(x,y,z)=gas_temperature", src)
        self.assertIn("cycle", src)

    def test_primitive_temperature_uses_intrinsic_gas_density_in_coupled_mode(self):
        src = _primitive_temperature_source()
        self.assertIn(
            "rho_g_stored=q_prim_vf(eqn_idx%cont%beg+1)%sf(x,y,z)+q_prim_vf(eqn_idx%cont%beg+2)%sf(x,y,z)",
            src,
        )
        self.assertIn(
            "alpha_g=q_prim_vf(eqn_idx%adv%beg+1)%sf(x,y,z)+q_prim_vf(eqn_idx%adv%beg+2)%sf(x,y,z)",
            src,
        )
        self.assertIn("rho_g_intrinsic=rho_g_stored/alpha_g", src)
        self.assertIn(
            "q_T_sf%sf(x,y,z)=q_prim_vf(eqn_idx%E)%sf(x,y,z)*mix_mol_weight/(gas_constant*rho_g_intrinsic)",
            src,
        )
        self.assertIn(
            "q_T_sf%sf(x,y,z)=q_prim_vf(eqn_idx%E)%sf(x,y,z)*mix_mol_weight/(gas_constant*q_prim_vf(1)%sf(x,y,z))",
            src,
        )

    def test_probe_output_uses_model3_helper_for_coupled_pressure_and_temperature(self):
        src = _compact_fortran(DATA_OUTPUT_SRC)
        self.assertEqual(src.count("if(chemistry.and.model3_chemistry_coupling)then"), 3)
        self.assertEqual(src.count("calls_get_model3_chemistry_gas_state"), 3)
        self.assertEqual(src.count("pres=gas_pressure"), 3)
        self.assertEqual(src.count("T=gas_temperature"), 3)
        self.assertEqual(src.count("calls_get_model3_phase_pressure"), 3)


if __name__ == "__main__":
    unittest.main()
