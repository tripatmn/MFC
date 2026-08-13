from __future__ import annotations

import csv
import contextlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from mfc_post.analysis import _select_indices, analyze_case
from mfc_post.cli import main
from mfc_post.inspect import inspect_case
from mfc_post.plotting import plot_history


SPECIES = ("NC12H26", "O2", "OH", "HO2", "H2O2", "CO", "CO2", "H2O")


def _record(path: Path, values) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


def _case(root: Path) -> None:
    root.joinpath("mechanism.yaml").write_text(
        "phases:\n- name: gas\n  species: [NC12H26, O2, OH, HO2, H2O2, CO, CO2, H2O]\n"
        "species:\n"
        "- name: NC12H26\n  composition: {C: 12, H: 26}\n"
        "- name: O2\n  composition: {O: 2}\n"
        "- name: OH\n  composition: {O: 1, H: 1}\n"
        "- name: HO2\n  composition: {H: 1, O: 2}\n"
        "- name: H2O2\n  composition: {H: 2, O: 2}\n"
        "- name: CO\n  composition: {C: 1, O: 1}\n"
        "- name: CO2\n  composition: {C: 1, O: 2}\n"
        "- name: H2O\n  composition: {H: 2, O: 1}\n"
    )
    root.joinpath("simulation.inp").write_text(
        "m=3\nn=0\np=0\nmodel_eqns=3\nnum_fluids=2\nchemistry=T\n"
        "parallel_io=F\ncfl_adap_dt=T\nt_save=5e-6\nmpp_lim=T\n"
        "cantera_file='mechanism.yaml'\ncantera_phase='gas'\n"
        "fuel_species_id=1\n"
        "chem_gas_fluid_id=2\nevap_liquid_fluid_id=1\nevap_alpha_thresh=0.01\n"
        "chem_reaction_heat_enable=T\n"
        "fluid_pp(1)%gamma=2\nfluid_pp(1)%pi_inf=0\nfluid_pp(1)%qv=0\n"
        "fluid_pp(2)%gamma=1\nfluid_pp(2)%pi_inf=0\nfluid_pp(2)%qv=0\n"
    )
    fractions = {
        "NC12H26": 0.05, "O2": 0.20, "OH": 0.01, "HO2": 0.01,
        "H2O2": 0.01, "CO": 0.10, "CO2": 0.20, "H2O": 0.42,
    }
    for saved_index in range(3):
        state = root / "p_all" / "p0" / str(saved_index)
        state.mkdir(parents=True)
        _record(state / "x_cb.dat", [0.0, 1.0, 2.0, 4.0, 5.0])
        fields = (
            [0.0, 0.1, 1.0, 0.0], [1.0] * 4, [0.0] * 4, [1.0e6] * 4,
            [0.0, 0.2, 0.6, 0.0], [1.0, 0.8, 0.4, 1.0],
            [1.0] * 4, [1.0] * 4,
            *([fractions[name] * (saved_index + 1)] * 4 for name in SPECIES),
        )
        # Preserve species closure while changing the inventories between saves.
        species_total = float(saved_index + 1)
        for index, values in enumerate(fields, 1):
            if index > 8:
                values = [value / species_total for value in values]
            _record(state / f"q_cons_vf{index}.dat", values)


def _lustre_from_p_all(root: Path, saved_indices) -> None:
    restart = root / "restart_data"
    restart.mkdir()
    restart.joinpath("lustre_x_cb.dat").write_bytes(
        struct.pack("<5d", 0.0, 1.0, 2.0, 4.0, 5.0)
    )
    for saved_index in saved_indices:
        state = root / "p_all" / "p0" / str(saved_index)
        fields = sorted(
            state.glob("q_cons_vf*.dat"),
            key=lambda path: int(path.stem.removeprefix("q_cons_vf")),
        )
        restart.joinpath(f"lustre_{saved_index}.dat").write_bytes(
            b"".join(path.read_bytes()[4:-4] for path in fields)
        )


class AnalysisTests(unittest.TestCase):
    def test_selection_range_stride_and_nearest(self):
        timeline = {"saved_indices": [0, 1, 2], "physical_times": [0.0, 5e-6, 10e-6]}
        indices, selection = _select_indices(timeline, (4.0, 9.0), None, 1)
        self.assertEqual(indices, [1, 2])
        self.assertEqual(selection["mappings"][0]["actual_time_us"], 5.0)
        indices, _ = _select_indices(timeline, None, (0.0, 10.0), 2)
        self.assertEqual(indices, [0, 2])

    def test_three_save_analysis_and_csv_only_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            analysis = root / "analysis"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = analyze_case(root, out_dir=analysis, execution="serial")
            messages = stdout.getvalue()
            self.assertIn(f"startup: case path: {root}", messages)
            self.assertIn("startup: selected source family: p_all", messages)
            self.assertIn(f"startup: selected source path: {root / 'p_all'}", messages)
            self.assertIn("startup: saves discovered: 3", messages)
            self.assertIn("startup: saves selected after filters: 3", messages)
            self.assertIn(f"startup: output directory: {analysis}", messages)
            self.assertEqual(messages.count("mfc-post analyze: progress: save "), 3)
            self.assertIn("completion: row count: 3", messages)
            self.assertIn(f"completion: scalar_timeseries.csv: {analysis / 'scalar_timeseries.csv'}", messages)
            self.assertIn(f"completion: quality.json: {analysis / 'quality.json'}", messages)
            self.assertIn(f"completion: provenance.json: {analysis / 'provenance.json'}", messages)
            self.assertEqual(len(result["rows"]), 3)
            self.assertEqual([row["time_us"] for row in result["rows"]], [0.0, 5.0, 10.0])
            self.assertAlmostEqual(result["rows"][0]["integrated_rhoY_NC12H26"], 0.25)
            self.assertAlmostEqual(result["rows"][0]["integrated_rhoY_O2"], 1.0)
            self.assertAlmostEqual(result["rows"][0]["gas_mass_weighted_Y_NC12H26"], 0.05)
            self.assertAlmostEqual(result["rows"][0]["liquid_NC12H26_inventory"], 2.1)
            self.assertAlmostEqual(result["rows"][0]["vapor_NC12H26_inventory"], 0.25)
            self.assertAlmostEqual(result["rows"][0]["total_NC12H26_inventory"], 2.35)
            self.assertAlmostEqual(result["rows"][0]["liquid_area_alpha_gt_0p5"], 2.0)
            self.assertAlmostEqual(result["rows"][0]["liquid_area_ratio_A_A0"], 1.0)
            self.assertAlmostEqual(result["rows"][0]["combustible_area"], 2.0)
            self.assertEqual(result["rows"][0]["spatial_measure_unit"], "m")
            with (analysis / "scalar_timeseries.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 3)
            self.assertNotIn("max_heat_release_rate_W_m3", csv_rows[0])
            quality = json.loads((analysis / "quality.json").read_text())
            self.assertEqual(quality["quality"][0]["total_cells"], 4)
            self.assertEqual(quality["quality"][0]["gas_dominated_cells"], 2)
            provenance = json.loads((analysis / "provenance.json").read_text())
            self.assertEqual(provenance["source"]["family"], "p_all")
            self.assertEqual(provenance["execution"]["mpi_size"], 1)

            with self.assertRaisesRegex(RuntimeError, "output directory is not empty.*--overwrite"):
                analyze_case(root, out_dir=analysis, execution="serial")
            with contextlib.redirect_stdout(io.StringIO()):
                overwritten = analyze_case(
                    root, out_dir=analysis, execution="serial", overwrite=True
                )
            self.assertEqual(len(overwritten["rows"]), 3)

            failed_output = root / "empty-selection"
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                exit_code = main([
                    "analyze", str(root), "--time-range-us", "20,30",
                    "--out-dir", str(failed_output), "--execution", "serial",
                ])
            self.assertNotEqual(exit_code, 0)
            self.assertIn("produced no saved states", stderr.getvalue())
            self.assertFalse(failed_output.exists())

            shutil.rmtree(root / "p_all")
            plotted = plot_history(
                analysis, fields=("max_valid_gas_temperature_K", "combustible_area"),
                out_dir=analysis / "trends",
            )
            self.assertEqual(len(plotted["files"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in plotted["files"]))
            manifest = json.loads((analysis / "trends" / "plot_manifest.json").read_text())
            self.assertIn("does not inspect or read p_all", manifest["data_access"])
            clean_trends = analysis / "clean_trends"
            with contextlib.redirect_stdout(io.StringIO()):
                default_plots = plot_history(analysis, out_dir=clean_trends)
            self.assertEqual(len(default_plots["files"]), 8)
            self.assertEqual(
                {path.name for path in clean_trends.glob("*.png")},
                {
                    "valid_gas_temperature_max.png",
                    "products_CO2_H2O.png",
                    "radicals_OH_HO2_H2O2.png",
                    "hot_combustible_overlap_area.png",
                    "hot_near_stoich_overlap_area.png",
                    "near_stoichiometric_area.png",
                    "liquid_area_ratio_A_A0.png",
                    "dodecane_inventory.png",
                },
            )
            with self.assertRaisesRegex(FileExistsError, "output directory is not empty.*--overwrite"):
                plot_history(
                    analysis, fields=("max_valid_gas_temperature_K",),
                    out_dir=analysis / "trends",
                )
            with contextlib.redirect_stdout(io.StringIO()):
                replotted = plot_history(
                    analysis, fields=("max_valid_gas_temperature_K",),
                    out_dir=analysis / "trends", overwrite=True,
                )
            self.assertEqual(len(replotted["files"]), 1)

    def test_auto_reads_complete_shared_lustre_when_p_all_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            _lustre_from_p_all(root, (1,))
            shutil.rmtree(root / "p_all")
            inspection = inspect_case(root)
            self.assertEqual(
                {item["family"] for item in inspection["sources"]}, {"lustre_shared"},
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = analyze_case(
                    root, selected_times_us=(5.0,), execution="serial",
                    out_dir=root / "lustre_analysis",
                )
            self.assertIn("selected source family: lustre_shared", stdout.getvalue())
            self.assertEqual(result["provenance"]["source_family"], "lustre_shared")
            self.assertEqual(result["rows"][0]["saved_index"], 1)
            self.assertAlmostEqual(result["rows"][0]["physical_time_s"], 5.0e-6)
            self.assertIn("saved_index * t_save", result["provenance"]["timeline"]["time_basis"])

    def test_optional_cantera_heat_release_columns_full_species_and_location(self):
        class FakeGas:
            species_names = list(SPECIES)

            def __init__(self):
                self._qdot = 0.0

            @property
            def TDY(self):
                raise AssertionError("TDY is write-only in this test")

            @TDY.setter
            def TDY(self, state):
                temperature, density, mass_fractions = state
                self._qdot = float(temperature) * float(density)
                self.received_species_count = len(mass_fractions)

            @property
            def heat_release_rate(self):
                return self._qdot

            @property
            def net_production_rates(self):
                return [self._qdot, *([0.0] * (len(SPECIES) - 1))]

            @property
            def partial_molar_enthalpies(self):
                return [-1.0, *([0.0] * (len(SPECIES) - 1))]

        fake_cantera = types.SimpleNamespace(
            Solution=lambda path, phase: FakeGas(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _case(root)
            simulation = root / "simulation.inp"
            simulation.write_text(simulation.read_text().replace("m=3\nn=0", "m=1\nn=1"))
            for saved_index in range(3):
                state = root / "p_all" / "p0" / str(saved_index)
                _record(state / "x_cb.dat", [0.0, 1.0, 2.0])
                _record(state / "y_cb.dat", [0.0, 1.0, 2.0])
                fractions = {
                    "NC12H26": 0.05, "O2": 0.20, "OH": 0.01, "HO2": 0.01,
                    "H2O2": 0.01, "CO": 0.10, "CO2": 0.20, "H2O": 0.42,
                }
                fields = (
                    [0.0, 0.1, 1.0, 0.0], [1.0] * 4,
                    [0.0] * 4, [0.0] * 4, [1.0e6] * 4,
                    [0.0, 0.2, 0.6, 0.0], [1.0, 0.8, 0.4, 1.0],
                    [1.0] * 4, [1.0] * 4,
                    *([fractions[name]] * 4 for name in SPECIES),
                )
                for index, values in enumerate(fields, 1):
                    _record(state / f"q_cons_vf{index}.dat", values)
            with mock.patch.dict(sys.modules, {"cantera": fake_cantera}):
                result = analyze_case(
                    root, selected_times_us=(5.0,), execution="serial",
                    compute_heat_release="cantera", out_dir=root / "hrr",
                )
            row = result["rows"][0]
            for column in (
                "max_heat_release_rate_W_m3", "x_max_heat_release_rate_m",
                "y_max_heat_release_rate_m", "integrated_heat_release_rate_net_W_per_m",
                "integrated_heat_release_rate_positive_W_per_m",
                "integrated_heat_release_rate_negative_W_per_m",
                "area_positive_heat_release_rate",
            ):
                self.assertIn(column, row)
            self.assertGreater(row["max_heat_release_rate_W_m3"], 0.0)
            self.assertEqual(row["x_max_heat_release_rate_m"], 0.5)
            self.assertEqual(row["y_max_heat_release_rate_m"], 0.5)
            self.assertEqual(row["integrated_heat_release_rate_negative_W_per_m"], 0.0)
            self.assertEqual(row["area_positive_heat_release_rate"], 3.0)
            with (root / "hrr" / "scalar_timeseries.csv").open(newline="") as stream:
                columns = tuple(csv.DictReader(stream).fieldnames or ())
            self.assertIn("max_heat_release_rate_W_m3", columns)
            quality = result["quality"][0]
            self.assertEqual(quality["heat_release_evaluated_cells"], 3)
            self.assertEqual(quality["heat_release_check_mismatch_cells"], 0)
            provenance = result["provenance"]["heat_release"]
            self.assertEqual(provenance["species_order"], list(SPECIES))
            self.assertEqual(provenance["definition"], "gas.heat_release_rate [W/m^3]")

    def test_plot_help_and_cli_failure_are_explicit(self):
        repository = Path(__file__).resolve().parents[2]
        help_result = subprocess.run(
            [sys.executable, str(repository / "mfc-post"), "plot", "--help"],
            cwd=repository, capture_output=True, text=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--overwrite", help_result.stdout)
        analyze_help = subprocess.run(
            [sys.executable, str(repository / "mfc-post"), "analyze", "--help"],
            cwd=repository, capture_output=True, text=True, check=False,
        )
        self.assertEqual(analyze_help.returncode, 0, analyze_help.stderr)
        self.assertIn("--source", analyze_help.stdout)
        self.assertIn("--compute-heat-release", analyze_help.stdout)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(["plot", "/definitely/missing/scalar_timeseries.csv"])
        self.assertNotEqual(exit_code, 0)
        self.assertIn("does not exist", stderr.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "scalar_timeseries.csv"
            malformed.write_text(
                "saved_index,time_us,max_valid_gas_temperature_K\n0,0,not-a-number\n"
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                exit_code = main([
                    "plot", str(malformed), "--fields", "max_valid_gas_temperature_K",
                    "--out-dir", str(root / "plots"),
                ])
            self.assertNotEqual(exit_code, 0)
            self.assertIn("invalid numeric value", stderr.getvalue())
            self.assertIn("CSV row 2", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
