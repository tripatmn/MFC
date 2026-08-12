from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mfc_post.cli import _field_limits
from mfc_post.inspect import inspect_case
from mfc_post.render import RENDER_FIELDS, _select_times, render_case


def _record(path: Path, values) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


def _case(root: Path) -> None:
    root.joinpath("mechanism.yaml").write_text(
        "phases:\n- name: gas\n  species: [N2, NC12H26, OH, O2, HO2, H2O2, CO2, H2O]\n"
        "species:\n"
        "- name: N2\n  composition: {N: 2}\n"
        "- name: NC12H26\n  composition: {C: 12, H: 26}\n"
        "- name: OH\n  composition: {O: 1, H: 1}\n"
        "- name: O2\n  composition: {O: 2}\n"
        "- name: HO2\n  composition: {H: 1, O: 2}\n"
        "- name: H2O2\n  composition: {H: 2, O: 2}\n"
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
    for saved_index in range(3):
        state = root / "p_all" / "p0" / str(saved_index)
        state.mkdir(parents=True)
        _record(state / "x_cb.dat", [0.0, 1.0e-6, 2.0e-6])
        _record(state / "y_cb.dat", [0.0, 1.0e-6, 2.0e-6])
        fuel = 0.05 + 0.01 * saved_index
        oxygen = 0.20
        oh = 0.01 * (saved_index + 1)
        ho2, h2o2, co2, h2o = 0.005, 0.005, 0.02, 0.02
        nitrogen = 1.0 - fuel - oxygen - oh - ho2 - h2o2 - co2 - h2o
        fields = (
            [0.0, 0.0, 2.0, 0.2], [1.0] * 4,
            [0.0] * 4, [0.0] * 4, [5.0e5 + saved_index * 1.0e5] * 4,
            [0.0, 0.0, 0.8, 0.2], [1.0, 1.0, 0.2, 0.8],
            [1.0] * 4, [1.0] * 4,
            [nitrogen] * 4, [fuel] * 4, [oh] * 4, [oxygen] * 4,
            [ho2] * 4, [h2o2] * 4, [co2] * 4, [h2o] * 4,
        )
        for index, values in enumerate(fields, 1):
            _record(state / f"q_cons_vf{index}.dat", values)


def _lustre_from_p_all(root: Path, saved_index: int) -> None:
    restart = root / "restart_data"
    restart.mkdir()
    restart.joinpath("lustre_x_cb.dat").write_bytes(struct.pack("<3d", 0.0, 1.0e-6, 2.0e-6))
    restart.joinpath("lustre_y_cb.dat").write_bytes(struct.pack("<3d", 0.0, 1.0e-6, 2.0e-6))
    state = root / "p_all" / "p0" / str(saved_index)
    fields = sorted(
        state.glob("q_cons_vf*.dat"),
        key=lambda path: int(path.stem.removeprefix("q_cons_vf")),
    )
    restart.joinpath(f"lustre_{saved_index}.dat").write_bytes(
        b"".join(path.read_bytes()[4:-4] for path in fields)
    )


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
            _lustre_from_p_all(root, 0)
            out = root / "render_clean"
            result = render_case(
                root, fields=RENDER_FIELDS, out_dir=out, execution="serial",
                time_range_us=(0.0, 10.0), stride=2, no_mp4=True,
                overlay=("temperature", "phi"), temperature_mask="strict_gas",
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
            self.assertEqual(manifest["source"]["family"], "p_all")
            self.assertEqual(manifest["field_limits"]["temperature"]["mode"], "batch")
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
                temperature_mask="strict_gas",
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

    def test_manual_temperature_limits_parse_render_and_record(self):
        self.assertEqual(
            _field_limits(["temperature:1000:2200"]),
            {"temperature": (1000.0, 2200.0)},
        )
        with self.assertRaisesRegex(ValueError, "FIELD:VMIN:VMAX"):
            _field_limits(["temperature:1000"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            out = root / "manual"
            result = render_case(
                root, selected_times_us=(5.0,), fields=("temperature",),
                temperature_mask="nonliquid", field_limits={"temperature": (1000, 2200)},
                execution="serial", out_dir=out,
            )
            expected = (
                out / "temperature_nonliquid_T1000_2200"
                / "temperature_nonliquid_T1000_2200_t0005p00us.png"
            )
            self.assertTrue(expected.is_file())
            frame = result["frames"][0]
            self.assertEqual(
                frame["color_limits"],
                {"minimum": 1000.0, "maximum": 2200.0, "mode": "manual"},
            )
            self.assertEqual(
                result["field_limits"]["temperature"],
                {"minimum": 1000.0, "maximum": 2200.0, "mode": "manual"},
            )
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertEqual(
                provenance["manual_field_limits"]["temperature"],
                {"minimum": 1000.0, "maximum": 2200.0},
            )

    def test_auto_renders_shared_lustre_temperature_without_p_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            _lustre_from_p_all(root, 1)
            shutil.rmtree(root / "p_all")
            inspection = inspect_case(root)
            self.assertEqual(
                {item["family"] for item in inspection["sources"]}, {"lustre_shared"},
            )
            result = render_case(
                root, selected_times_us=(5.0,), fields=("temperature",),
                temperature_mask="nonliquid", execution="serial",
                out_dir=root / "lustre_render",
            )
            self.assertEqual(result["source"]["family"], "lustre_shared")
            self.assertTrue(
                (root / "lustre_render" / "temperature_nonliquid"
                 / "temperature_nonliquid_t0005p00us.png").is_file()
            )
            provenance = json.loads((root / "lustre_render" / "provenance.json").read_text())
            self.assertEqual(provenance["source_family"], "lustre_shared")
            self.assertIn("saved_index * t_save", provenance["timeline"]["time_basis"])

    def test_nonliquid_temperature_mask_plots_more_mixed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            strict = render_case(
                root, selected_times_us=(5.0,), fields=("temperature",),
                out_dir=root / "strict", execution="serial", temperature_mask="strict_gas",
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
                "alpha_liq <= 0.5 AND isfinite(field)",
            )
            provenance_counts = provenance["render_policy"]["temperature_mask"]["cell_counts"]
            self.assertEqual(len(provenance_counts), 2)
            self.assertEqual(
                provenance_counts[0]["plotted"],
                nonliquid_frame["plot_cell_counts"]["plotted"],
            )

    def test_nonliquid_is_default_for_all_supported_species_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            species = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO2", "H2O")
            strict = render_case(
                root, selected_times_us=(5.0,), fields=species,
                out_dir=root / "species_strict", execution="serial",
                temperature_mask="strict_gas",
            )
            presentation = render_case(
                root, selected_times_us=(5.0,), fields=species,
                out_dir=root / "species_nonliquid", execution="serial",
            )
            self.assertEqual(presentation["render_mask"]["mode"], "nonliquid")
            strict_counts = {
                frame["field"]: frame["plot_cell_counts"]["plotted"]
                for frame in strict["frames"]
            }
            presentation_counts = {
                frame["field"]: frame["plot_cell_counts"]["plotted"]
                for frame in presentation["frames"]
            }
            self.assertEqual(set(presentation_counts), set(species))
            for species_name in species:
                self.assertGreater(
                    presentation_counts[species_name], strict_counts[species_name],
                    species_name,
                )
            self.assertTrue(all(
                frame["mask_policy"] == "alpha_liq <= 0.5 AND isfinite(field)"
                and frame["render_mask"] == "nonliquid"
                and frame["liquid_context"]["contour"]["level"] == 0.5
                for frame in presentation["frames"]
            ))
            provenance = json.loads(
                (root / "species_nonliquid" / "provenance.json").read_text()
            )
            self.assertEqual(
                provenance["render_policy"]["scalar_mask"]["mode"], "nonliquid",
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
            "--render-mask",
            "--source",
            "--field-limits",
        ):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
