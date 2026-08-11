from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mfc_post.process import process_case


FIELDS = (
    [2.0, 2.0, 0.0, 0.0],  # liquid partial density
    [1.0, 1.0, 1.0, 1.0],  # gas partial density
    [0.0, 0.0, 0.0, 0.0],  # momentum
    [3.0, 3.0, 3.0, 3.0],  # total energy
    [0.6, 0.6, 0.1, 0.1],  # liquid volume fraction
    [0.4, 0.4, 0.9, 0.9],  # gas volume fraction
    [1.0, 1.0, 1.0, 1.0],  # liquid internal energy
    [1.0, 1.0, 1.0, 1.0],  # gas internal energy
    [0.4, 0.4, 0.4, 0.4],  # species 1
    [0.6, 0.6, 0.6, 0.6],  # species 2
)


def write_record(path: Path, values) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


def write_config(root: Path, parallel: bool) -> None:
    root.joinpath("mechanism.yaml").write_text(
        "phases:\n"
        "- name: test-gas\n"
        "  species: [H2, O2]\n"
        "species:\n"
        "- name: H2\n"
        "  composition: {H: 2}\n"
        "- name: O2\n"
        "  composition: {O: 2}\n"
    )
    root.joinpath("simulation.inp").write_text(
        "m = 3\nn = 0\np = 0\nmodel_eqns = 3\nnum_fluids = 2\n"
        "chemistry = T\nchem_gas_fluid_id = 2\nevap_liquid_fluid_id = 1\n"
        "cantera_file = 'mechanism.yaml'\ncantera_phase = 'test-gas'\n"
        "chem_reaction_heat_enable = T\nmpp_lim = T\n"
        "fluid_pp(1)%gamma = 1.0\nfluid_pp(1)%pi_inf = 0.0\nfluid_pp(1)%qv = 0.0\n"
        "fluid_pp(2)%gamma = 1.0\nfluid_pp(2)%pi_inf = 0.0\nfluid_pp(2)%qv = 0.0\n"
        f"parallel_io = {'T' if parallel else 'F'}\ndt = 0.25\n"
    )


class ProcessTests(unittest.TestCase):
    def test_p_all_raw_metrics_and_nonuniform_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, False)
            state = root / "p_all" / "p0" / "0"
            state.mkdir(parents=True)
            write_record(state / "x_cb.dat", [0.0, 1.0, 2.0, 4.0, 5.0])
            for index, values in enumerate(FIELDS, 1):
                write_record(state / f"q_cons_vf{index}.dat", values)
            result = process_case(root, execution="serial", output=root / "out")
            record = result["records"][0]
            self.assertEqual(record["cell_count"], 4)
            self.assertAlmostEqual(record["conservative_liquid_mass"], 4.0)
            self.assertAlmostEqual(record["dense_liquid"]["measure"], 2.0)
            self.assertAlmostEqual(record["closure"]["species"]["max_absolute_residual"], 0.0)
            self.assertAlmostEqual(record["closure"]["volume_fraction"]["max_absolute_residual"], 0.0)
            self.assertEqual(record["invalid_cells"]["any"], 0)
            self.assertEqual(result["schema_version"], "mfc-post.process/v2")
            self.assertEqual(record["physical_state"]["mask_counts"]["mask.valid"], 4)
            self.assertEqual(record["physical_state"]["ranges"]["pressure"]["minimum"], 100.0)
            self.assertEqual(record["physical_state"]["ranges"]["temperature"]["minimum"], 250.0)
            self.assertEqual(result["quality"][0]["physical_mask_counts"]["mask.valid"], 4)
            json.loads((root / "out" / "results.json").read_text())

    def test_shared_lustre_payload_matches_p_all_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, True)
            restart = root / "restart_data"
            restart.mkdir()
            np.asarray([0.0, 1.0, 2.0, 4.0, 5.0], dtype=np.float64).tofile(restart / "lustre_x_cb.dat")
            np.asarray(FIELDS, dtype=np.float64).tofile(restart / "lustre_0.dat")
            result = process_case(
                root, execution="serial", source_family="lustre_shared", output=root / "out"
            )
            record = result["records"][0]
            self.assertAlmostEqual(record["conservative_liquid_mass"], 4.0)
            self.assertAlmostEqual(record["dense_liquid"]["measure"], 2.0)
            self.assertEqual(sorted(record["raw_species_inventories"]), ["H2", "O2"])

    def test_index_stop_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, True)
            restart = root / "restart_data"
            restart.mkdir()
            np.asarray([0.0, 1.0, 2.0, 4.0, 5.0], dtype=np.float64).tofile(restart / "lustre_x_cb.dat")
            for index in (0, 1):
                np.asarray(FIELDS, dtype=np.float64).tofile(restart / f"lustre_{index}.dat")
            result = process_case(
                root, execution="serial", source_family="lustre_shared",
                index_start=0, index_stop=1, output=root / "out",
            )
            self.assertEqual(result["selection"]["saved_indices"], [0])

    def test_corrupt_state_names_source_index_rank_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, False)
            state = root / "p_all" / "p0" / "7"
            state.mkdir(parents=True)
            write_record(state / "x_cb.dat", [0.0, 1.0, 2.0, 4.0, 5.0])
            # Establish the observed layout, but leave a required conservative field absent.
            for index, values in enumerate(FIELDS, 1):
                if index != 2:
                    write_record(state / f"q_cons_vf{index}.dat", values)
            with self.assertRaisesRegex(
                RuntimeError, r"worker_rank=0 source=p_all saved_index=7.*q_cons_vf2\.dat"
            ):
                process_case(root, execution="serial", output=root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
