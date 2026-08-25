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
PHASE_CHANGE_SRC = REPO_ROOT / "src" / "common" / "m_phase_change.fpp"
SIM_START_UP_SRC = REPO_ROOT / "src" / "simulation" / "m_start_up.fpp"


def _adapter_source() -> str:
    text = CHEMISTRY_SRC.read_text()
    match = re.search(
        r"subroutine s_get_model3_chemistry_gas_state\b(?P<body>.*?)"
        r"end subroutine s_get_model3_chemistry_gas_state",
        text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError("s_get_model3_chemistry_gas_state not found")
    return re.sub(r"\s+", "", match.group("body"))


def _apply_evaporation_only_reference(liquid_before, vapor_before, liquid_candidate, vapor_candidate):
    if vapor_candidate < vapor_before:
        return liquid_before, vapor_before, 0.0
    return liquid_candidate, vapor_candidate, vapor_candidate - vapor_before


def _apply_vapor_delta_to_fuel_species_reference(species_rho_y, fuel_index, delta):
    updated = list(species_rho_y)
    updated[fuel_index] += delta
    delta = 0.0
    return updated, delta


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


if __name__ == "__main__":
    unittest.main()
