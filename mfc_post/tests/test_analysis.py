from __future__ import annotations

import csv
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from mfc_post.analysis import _select_indices, analyze_case
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
            result = analyze_case(root, out_dir=analysis, execution="serial")
            self.assertEqual(len(result["rows"]), 3)
            self.assertEqual([row["time_us"] for row in result["rows"]], [0.0, 5.0, 10.0])
            self.assertAlmostEqual(result["rows"][0]["integrated_rhoY_NC12H26"], 0.25)
            self.assertAlmostEqual(result["rows"][0]["combustible_area"], 2.0)
            self.assertEqual(result["rows"][0]["spatial_measure_unit"], "m")
            with (analysis / "scalar_timeseries.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 3)
            quality = json.loads((analysis / "quality.json").read_text())
            self.assertEqual(quality["quality"][0]["total_cells"], 4)
            self.assertEqual(quality["quality"][0]["gas_dominated_cells"], 2)
            provenance = json.loads((analysis / "provenance.json").read_text())
            self.assertEqual(provenance["source"]["family"], "p_all")
            self.assertEqual(provenance["execution"]["mpi_size"], 1)

            shutil.rmtree(root / "p_all")
            plotted = plot_history(
                analysis, fields=("max_valid_gas_temperature_K", "combustible_area"),
                out_dir=analysis / "trends",
            )
            self.assertEqual(len(plotted["files"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in plotted["files"]))
            manifest = json.loads((analysis / "trends" / "plot_manifest.json").read_text())
            self.assertIn("does not inspect or read p_all", manifest["data_access"])


if __name__ == "__main__":
    unittest.main()
