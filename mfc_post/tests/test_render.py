from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from mfc_post.render import RENDER_FIELDS, _select_times, render_case


def _record(path: Path, values) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


class RenderTests(unittest.TestCase):
    def test_nearest_time_tie_chooses_earlier_save(self):
        timeline = {
            "saved_indices": [2, 4],
            "physical_times": [4.0e-6, 6.0e-6],
        }
        selected = _select_times(timeline, (5.0,))
        self.assertEqual(selected[0]["saved_index"], 2)
        self.assertEqual(selected[0]["actual_time_us"], 4.0)

    def test_selected_state_full_domain_manifest_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            species = ("N2", "NC12H26", "OH", "CO2", "H2O")
            root.joinpath("mechanism.yaml").write_text(
                "phases:\n- name: gas\n  species: [N2, NC12H26, OH, CO2, H2O]\n"
                "species:\n"
                "- name: N2\n  composition: {N: 2}\n"
                "- name: NC12H26\n  composition: {C: 12, H: 26}\n"
                "- name: OH\n  composition: {O: 1, H: 1}\n"
                "- name: CO2\n  composition: {C: 1, O: 2}\n"
                "- name: H2O\n  composition: {H: 2, O: 1}\n"
            )
            root.joinpath("simulation.inp").write_text(
                "m=1\nn=1\np=0\nmodel_eqns=3\nnum_fluids=2\nchemistry=T\n"
                "parallel_io=F\ncfl_adap_dt=T\nt_save=5e-6\nmpp_lim=T\n"
                "cantera_file='mechanism.yaml'\ncantera_phase='gas'\n"
                "chem_gas_fluid_id=2\nevap_liquid_fluid_id=1\nevap_alpha_thresh=0.01\n"
                "chem_reaction_heat_enable=T\n"
                "fluid_pp(1)%gamma=2\nfluid_pp(1)%pi_inf=0\nfluid_pp(1)%qv=0\n"
                "fluid_pp(2)%gamma=1\nfluid_pp(2)%pi_inf=0\nfluid_pp(2)%qv=0\n"
            )
            state = root / "p_all" / "p0" / "1"
            state.mkdir(parents=True)
            _record(state / "x_cb.dat", [0.0, 1.0, 2.0])
            _record(state / "y_cb.dat", [0.0, 1.0, 2.0])
            fields = (
                [0.0, 0.0, 2.0, 0.2], [1.0, 1.0, 1.0, 1.0],
                [0.0] * 4, [0.0] * 4, [50.0] * 4,
                [0.0, 0.0, 0.8, 0.2], [1.0, 1.0, 0.2, 0.8],
                [1.0] * 4, [1.0] * 4,
                [0.84] * 4, [0.10] * 4, [0.01] * 4, [0.02] * 4, [0.03] * 4,
            )
            for index, values in enumerate(fields, 1):
                _record(state / f"q_cons_vf{index}.dat", values)
            out = root / "render"
            result = render_case(
                root, (5.0,), RENDER_FIELDS, out, execution="serial",
                no_zoom=True, no_mp4=True, skip_scalars=True, skip_trends=True,
            )
            self.assertEqual(result["selections"][0]["saved_index"], 1)
            self.assertEqual(result["selections"][0]["actual_time_us"], 5.0)
            self.assertEqual(len(result["frames"]), 5)
            self.assertEqual(len(list(out.glob("*.png"))), 5)
            counts = result["frames"][0]["counts"]
            self.assertEqual(counts["gas_dominated"], 2)
            self.assertEqual(counts["interface"], 1)
            self.assertEqual(counts["liquid_dominated"], 1)
            self.assertEqual(counts["temperature_clipped"], 4)
            self.assertEqual(counts["pressure_floored"], 4)
            temperature = next(frame for frame in result["frames"] if frame["field"] == "temperature")
            self.assertLess(temperature["raw_range"]["maximum"], 250.0)
            self.assertEqual(temperature["plotted_range"], {"minimum": 250.0, "maximum": 250.0})
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["options"]["scalar_history_computed"])
            self.assertEqual(manifest["frames"][1]["species_equation_index"], 11)
            missing_out = root / "missing"
            with self.assertRaisesRegex(ValueError, "ABSENT.*absent or ambiguous"):
                render_case(root, (5.0,), ("Y[ABSENT]",), missing_out, execution="serial")
            self.assertFalse(missing_out.exists())
            with self.assertRaisesRegex(RuntimeError, "output directory is not empty"):
                render_case(root, (5.0,), ("temperature",), out, execution="serial")


if __name__ == "__main__":
    unittest.main()
