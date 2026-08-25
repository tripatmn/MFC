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


if __name__ == "__main__":
    unittest.main()
