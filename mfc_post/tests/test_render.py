from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mfc_post.render import RENDER_FIELDS, _select_times, render_case


def _record(path: Path, values) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


def _case(root: Path) -> None:
    root.joinpath("mechanism.yaml").write_text(
        "phases:\n- name: gas\n  species: [N2, NC12H26, OH, O2]\n"
        "species:\n"
        "- name: N2\n  composition: {N: 2}\n"
        "- name: NC12H26\n  composition: {C: 12, H: 26}\n"
        "- name: OH\n  composition: {O: 1, H: 1}\n"
        "- name: O2\n  composition: {O: 2}\n"
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
    for saved_index in range(3):
        state = root / "p_all" / "p0" / str(saved_index)
        state.mkdir(parents=True)
        _record(state / "x_cb.dat", [0.0, 1.0e-6, 2.0e-6])
        _record(state / "y_cb.dat", [0.0, 1.0e-6, 2.0e-6])
        fuel = 0.05 + 0.01 * saved_index
        oxygen = 0.20
        oh = 0.01 * (saved_index + 1)
        nitrogen = 1.0 - fuel - oxygen - oh
        fields = (
            [0.0, 0.0, 2.0, 0.2], [1.0] * 4,
            [0.0] * 4, [0.0] * 4, [5.0e5 + saved_index * 1.0e5] * 4,
            [0.0, 0.0, 0.8, 0.2], [1.0, 1.0, 0.2, 0.8],
            [1.0] * 4, [1.0] * 4,
            [nitrogen] * 4, [fuel] * 4, [oh] * 4, [oxygen] * 4,
        )
        for index, values in enumerate(fields, 1):
            _record(state / f"q_cons_vf{index}.dat", values)


class RenderTests(unittest.TestCase):
    def test_nearest_time_tie_chooses_earlier_save(self):
        timeline = {"saved_indices": [2, 4], "physical_times": [4.0e-6, 6.0e-6]}
        selected = _select_times(timeline, (5.0,))
        self.assertEqual(selected[0]["saved_index"], 2)
        self.assertEqual(selected[0]["actual_time_us"], 4.0)

    def test_clean_batch_inventory_limits_overlay_overwrite_and_zero_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            out = root / "render_clean"
            result = render_case(
                root, fields=RENDER_FIELDS, out_dir=out, execution="serial",
                time_range_us=(0.0, 10.0), stride=2, no_mp4=True,
                overlay=("temperature", "phi"),
            )
            self.assertEqual([item["saved_index"] for item in result["selections"]], [0, 2])
            self.assertEqual(len(result["frames"]), 14)
            expected_directories = (
                set(RENDER_FIELDS) - {"temperature"}
            ) | {"temperature_strict_gas", "overlay_temperature_strict_gas_phi"}
            self.assertEqual(
                {path.name for path in out.iterdir() if path.is_dir()}, expected_directories,
            )
            expected_names = {
                "temperature_strict_gas/temperature_strict_gas_t0000p00us.png",
                "temperature_strict_gas/temperature_strict_gas_t0010p00us.png",
                "OH/OH_t0000p00us.png", "OH/OH_t0010p00us.png",
                "NC12H26/NC12H26_t0000p00us.png", "NC12H26/NC12H26_t0010p00us.png",
                "O2/O2_t0000p00us.png", "O2/O2_t0010p00us.png",
                "phi/phi_t0000p00us.png", "phi/phi_t0010p00us.png",
                "alpha_liq/alpha_liq_t0000p00us.png", "alpha_liq/alpha_liq_t0010p00us.png",
                "overlay_temperature_strict_gas_phi/temperature_strict_gas_phi_t0000p00us.png",
                "overlay_temperature_strict_gas_phi/temperature_strict_gas_phi_t0010p00us.png",
            }
            observed_names = {
                str(path.relative_to(out)) for path in out.rglob("*.png")
            }
            self.assertEqual(observed_names, expected_names)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "mfc-post.render-clean/v1")
            self.assertEqual(manifest["overlay"]["levels"], [1.0])
            self.assertTrue(all(
                frame["liquid_context"]["contour"]["level"] == 0.5
                for frame in manifest["frames"]
            ))
            self.assertTrue(all(
                frame["liquid_context"]["contour"]["drawn"]
                for frame in manifest["frames"]
            ))
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertEqual(
                provenance["render_policy"]["liquid_context"]["contour"]["level"], 0.5,
            )
            temperature_limits = {
                tuple(frame["color_limits"].values())
                for frame in manifest["frames"] if frame.get("field") == "temperature"
            }
            self.assertEqual(len(temperature_limits), 1)
            self.assertTrue((out / "provenance.json").is_file())
            self.assertFalse((out / "frames.csv").exists())
            with self.assertRaisesRegex(RuntimeError, "output directory is not empty.*--overwrite"):
                render_case(root, (5.0,), ("temperature",), out, execution="serial")
            replaced = render_case(
                root, (5.0,), ("temperature",), out, execution="serial", overwrite=True,
            )
            self.assertEqual(len(replaced["frames"]), 1)
            self.assertEqual(
                {str(path.relative_to(out)) for path in out.rglob("*.png")},
                {"temperature_strict_gas/temperature_strict_gas_t0005p00us.png"},
            )
            with self.assertRaisesRegex(ValueError, "zero saved states"):
                render_case(
                    root, fields=("temperature",), out_dir=root / "empty",
                    execution="serial", time_range_us=(20.0, 30.0),
                )
            self.assertFalse((root / "empty").exists())

    def test_nonliquid_temperature_mask_plots_more_mixed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            strict = render_case(
                root, selected_times_us=(5.0,), fields=("temperature",),
                out_dir=root / "strict", execution="serial",
            )
            nonliquid = render_case(
                root, selected_times_us=(5.0,), fields=("temperature",),
                out_dir=root / "nonliquid", execution="serial",
                temperature_mask="nonliquid", overlay=("temperature", "phi"),
            )
            strict_frame = strict["frames"][0]
            nonliquid_frame = next(
                frame for frame in nonliquid["frames"] if frame["kind"] == "field"
            )
            self.assertEqual(strict["temperature_mask"]["mode"], "strict_gas")
            self.assertEqual(nonliquid["temperature_mask"]["mode"], "nonliquid")
            self.assertGreater(
                nonliquid_frame["plot_cell_counts"]["plotted"],
                strict_frame["plot_cell_counts"]["plotted"],
            )
            self.assertEqual(
                nonliquid_frame["plot_cell_counts"]["masked"],
                nonliquid_frame["plot_cell_counts"]["total"]
                - nonliquid_frame["plot_cell_counts"]["plotted"],
            )
            self.assertTrue(
                (root / "nonliquid" / "temperature_nonliquid"
                 / "temperature_nonliquid_t0005p00us.png").is_file()
            )
            self.assertTrue(
                (root / "nonliquid" / "overlay_temperature_nonliquid_phi"
                 / "temperature_nonliquid_phi_t0005p00us.png").is_file()
            )
            provenance = json.loads((root / "nonliquid" / "provenance.json").read_text())
            self.assertEqual(
                provenance["render_policy"]["temperature_mask"]["definition"],
                "alpha_liq <= 0.5 AND isfinite(temperature)",
            )
            provenance_counts = provenance["render_policy"]["temperature_mask"]["cell_counts"]
            self.assertEqual(len(provenance_counts), 2)
            self.assertEqual(
                provenance_counts[0]["plotted"],
                nonliquid_frame["plot_cell_counts"]["plotted"],
            )

    def test_render_help(self):
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(repository / "mfc-post"), "render", "--help"],
            cwd=repository, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--selected-times-us", "--time-range-us", "--stride", "--fields",
            "--overwrite", "--execution", "--out-dir", "--no-mp4", "--overlay",
            "--temperature-mask",
        ):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
