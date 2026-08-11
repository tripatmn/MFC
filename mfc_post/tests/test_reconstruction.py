from __future__ import annotations

import unittest

import numpy as np

from mfc_post.equations import build_equation_layout
from mfc_post.models import Field, Grid, Provenance, RunMetadata, State
from mfc_post.reconstruction import FieldRegistry, Model3Configuration, reconstruct_model3


class Model3ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "m": 2, "n": 0, "p": 0, "model_eqns": 3, "num_fluids": 2,
            "chemistry": True, "num_species": 2, "species_names": ("H2", "O2"),
            "molecular_weights": (2.016, 31.998), "cantera_file": "/test/mech.yaml",
            "cantera_phase": "gas", "chem_gas_fluid_id": 2,
            "evap_liquid_fluid_id": 1, "chem_reaction_heat_enable": True,
            "mpp_lim": True,
            "fluid_pp(1)%gamma": 2.0, "fluid_pp(1)%pi_inf": 0.0,
            "fluid_pp(1)%qv": 0.0, "fluid_pp(2)%gamma": 1.0,
            "fluid_pp(2)%pi_inf": 0.0, "fluid_pp(2)%qv": 0.0,
        }
        layout = build_equation_layout(self.params)
        self.metadata = RunMetadata(
            "/test", self.params, 1, (3,), layout.fields, ("H2", "O2")
        )

    def test_solver_eos_species_temperature_masks_and_registry(self):
        values = {
            "partial_density[1]": [2.0, -1.0, 0.0],
            "partial_density[2]": [1.0, 1.0, 1.0],
            "momentum": [3.0, 1.0, 0.0],
            "total_energy_density": [2000.0, 1000.0, 500.0],
            "volume_fraction[1]": [0.6, 1.2, 0.0],
            "volume_fraction[2]": [0.4, 0.2, 1.0],
            "partial_internal_energy[1]": [1.0, 1.0, 1.0],
            "partial_internal_energy[2]": [1.0, 1.0, 1.0],
            "species_density[H2]": [0.4, 0.2, 0.0],
            "species_density[O2]": [0.8, 0.8, 1.0],
        }
        provenance = Provenance("p_all", "/test/p0/0")
        state = State(
            0, 0, 0.0, Grid(1, (3,), cell_measures=(1.0, 1.0, 1.0)),
            {name: Field(name, "conservative", values=np.asarray(data), provenance=provenance)
             for name, data in values.items()},
            provenance,
        )
        config = Model3Configuration.from_metadata(self.metadata)
        physical = reconstruct_model3(state, config)
        registry = FieldRegistry(physical, config)

        np.testing.assert_allclose(registry.resolve("rho").values, [3.0, 1.0, 1.0])
        np.testing.assert_allclose(registry.resolve("velocity[x]").values, [1.0, 1.0, 0.0])
        expected_pressure = np.maximum(
            [(2000.0 - 1.5) / 1.6, (1000.0 - 0.5) / (11.0 / 6.0), 500.0], 100.0
        )
        np.testing.assert_allclose(registry.resolve("pressure").values, expected_pressure)
        np.testing.assert_allclose(registry.resolve("gas_density").values, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(registry.resolve("Y[H2]").values, [1.0 / 3.0, 0.2, 0.0])
        np.testing.assert_allclose(registry.resolve("Y[O2]").values, [2.0 / 3.0, 0.8, 1.0])
        expected_mw = 1.0 / (
            np.asarray([[1.0 / 3.0, 0.2, 0.0]]) / 2.016
            + np.asarray([[2.0 / 3.0, 0.8, 1.0]]) / 31.998
        )[0]
        np.testing.assert_allclose(registry.resolve("molecular_weight").values, expected_mw)
        raw_temperature = expected_pressure * expected_mw / 8314.46261815324
        np.testing.assert_allclose(registry.resolve("temperature.raw").values, raw_temperature)
        np.testing.assert_allclose(
            registry.resolve("T").values, np.clip(raw_temperature, 250.0, 3000.0)
        )
        np.testing.assert_array_equal(registry.resolve("mask.valid").values, [True, True, True])
        np.testing.assert_array_equal(registry.resolve("mask.gas_dominated").values, [False, False, True])
        np.testing.assert_array_equal(registry.resolve("mask.liquid_dominated").values, [True, True, False])
        np.testing.assert_array_equal(registry.resolve("mask.chemistry_valid").values, [True, True, True])
        self.assertEqual(state.fields["partial_density[1]"].values[1], -1.0)
        self.assertEqual(registry.resolve("alpha_rho[liquid]").values[1], 0.0)
        self.assertIn("mechanism=/test/mech.yaml", physical.provenance.notes)

    def test_ambiguous_fluid_role_fails_clearly(self):
        bad = dict(self.params)
        bad.pop("evap_liquid_fluid_id")
        layout = build_equation_layout(bad)
        metadata = RunMetadata("/test", bad, 1, (3,), layout.fields, ("H2", "O2"))
        with self.assertRaisesRegex(ValueError, "liquid fluid role is ambiguous"):
            Model3Configuration.from_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
