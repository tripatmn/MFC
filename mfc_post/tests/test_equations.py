from __future__ import annotations

import unittest

from mfc_post.equations import build_equation_layout


class EquationLayoutTests(unittest.TestCase):
    def test_six_equation_species_begin_is_constructed(self):
        params = {
            "m": 7, "n": 3, "p": 0, "model_eqns": 3, "num_fluids": 3,
            "chemistry": True,
        }
        layout = build_equation_layout(params, observed_size=17)
        self.assertEqual(layout.base_size, 12)
        self.assertEqual(layout.species_begin, 13)
        self.assertEqual(layout.species_count, 5)
        self.assertEqual(layout.total_size, 17)
        self.assertEqual(layout.fields[12]["name"], "species_density[1]")

    def test_model_two_igr_omits_last_stored_volume_fraction(self):
        params = {"m": 3, "n": 0, "p": 0, "model_eqns": 2, "num_fluids": 2, "igr": True}
        layout = build_equation_layout(params)
        self.assertEqual(layout.total_size, 5)
        self.assertEqual(sum(field["name"].startswith("volume_fraction") for field in layout.fields), 1)


if __name__ == "__main__":
    unittest.main()
