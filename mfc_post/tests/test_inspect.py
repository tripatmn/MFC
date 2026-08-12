from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from mfc_post.inspect import inspect_case


def write_record(path: Path, values: list[float]) -> None:
    payload = struct.pack("<" + "d" * len(values), *values)
    marker = struct.pack("<I", len(payload))
    path.write_bytes(marker + payload + marker)


class InspectTests(unittest.TestCase):
    def test_detects_source_families_without_merging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simulation.inp").write_text(
                """&user_inputs
m = 1
n = 0
p = 0
dt = 0.25
model_eqns = 2
num_fluids = 1
parallel_io = F
precision = 2
&end/
"""
            )

            state = root / "p_all" / "p0" / "0"
            state.mkdir(parents=True)
            write_record(state / "x_cb.dat", [0.0, 0.2, 1.0])
            for index in range(1, 5):
                write_record(state / f"q_cons_vf{index}.dat", [float(index), float(index)])

            d_dir = root / "D"
            d_dir.mkdir()
            for index in range(1, 5):
                (d_dir / f"cons.{index}.00.000000.dat").write_text("0.2 1.0\n1.0 2.0\n")

            restart = root / "restart_data"
            restart.mkdir()
            (restart / "lustre_0.dat").write_bytes(b"\0" * 64)
            rank_dir = restart / "lustre_1"
            rank_dir.mkdir()
            (rank_dir / "1_0000000.dat").write_bytes(b"\0" * 64)

            (root / "binary" / "p0").mkdir(parents=True)
            (root / "binary" / "p0" / "0.dat").write_bytes(b"binary")
            (root / "silo_hdf5" / "p0").mkdir(parents=True)
            (root / "silo_hdf5" / "root").mkdir()
            (root / "silo_hdf5" / "p0" / "0.silo").write_bytes(b"\x89HDF\r\n\x1a\n")
            (root / "silo_hdf5" / "root" / "collection_0.silo").write_bytes(b"\x89HDF\r\n\x1a\n")

            result = inspect_case(root)
            families = {source["family"] for source in result["sources"]}
            self.assertEqual(
                families,
                {"lustre_shared", "lustre_per_process", "p_all", "D", "binary", "silo_hdf5"},
            )
            p_all = next(source for source in result["sources"] if source["family"] == "p_all")
            self.assertEqual(p_all["grid"]["shape"], [2])
            self.assertEqual(p_all["grid"]["width_ranges"]["x"], [0.2, 0.8])
            self.assertTrue(any(item["kind"] == "alternative_writer_families_coexist" for item in result["conflicts"]))
            json.dumps(result, allow_nan=False)

    def test_adaptive_timeline_does_not_claim_simulation_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simulation.inp").write_text(
                "m = 1\nn = 0\np = 0\nmodel_eqns = 2\nnum_fluids = 1\n"
                "cfl_adap_dt = T\nt_save = 0.5\nparallel_io = T\n"
            )
            restart = root / "restart_data"
            restart.mkdir()
            (restart / "lustre_2.dat").write_bytes(b"\0" * 64)
            result = inspect_case(root)
            timeline = result["sources"][0]["timeline"]
            self.assertEqual(timeline["physical_times"], [1.0])
            self.assertEqual(timeline["simulation_steps"], [None])
            self.assertIn("saved_index", timeline["time_basis"])

    def test_empty_output_directory_is_not_recommended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simulation.inp").write_text("m = 1\nn = 0\np = 0\nmodel_eqns = 2\nnum_fluids = 1\n")
            (root / "D").mkdir()
            result = inspect_case(root)
            self.assertIsNone(result["recommendation"])

    def test_lustre_restart_continuation_warns_about_global_time_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simulation.inp").write_text(
                "m=0\nn=0\np=0\nmodel_eqns=2\nnum_fluids=1\n"
                "parallel_io=T\nt_save=1e-6\nn_start=100\n"
            )
            restart = root / "restart_data"
            restart.mkdir()
            (restart / "lustre_x_cb.dat").write_bytes(struct.pack("<2d", 0.0, 1.0))
            # Four Model-2 fields for one cell.
            (restart / "lustre_0.dat").write_bytes(struct.pack("<4d", 1.0, 0.0, 1.0, 1.0))
            result = inspect_case(root)
            timeline = result["timeline"]["lustre_shared"]
            self.assertTrue(any(
                "global saved-index/time offset is ambiguous" in warning
                for warning in timeline["warnings"]
            ))


if __name__ == "__main__":
    unittest.main()
