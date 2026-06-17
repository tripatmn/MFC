#!/usr/bin/env python3
"""Restart-capable Tanabe-1996-style n-dodecane/SK54 scout case.

This case intentionally preserves the physics of
case_hpc_d2_tanabe1996_dodecane_070mm_P20bar_T800_yao_sk54_128.py and only
switches the output path to shared-file MPI restart data. With
parallel_io = T and file_per_process = F, MFC writes restart_data/lustre_/
files that can be used by the old_ic/old_grid restart path.
"""

import json
import pathlib
import sys


def find_repo_root():
    starts = [pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()]
    for start in starts:
        for path in (start, *start.parents):
            case_dir = path/"examples"/"2D_dodecane_global_reduced"
            source_case = case_dir/"case_hpc_d2_tanabe1996_dodecane_070mm_P20bar_T800_yao_sk54_128.py"
            if source_case.is_file() and (path/"toolchain"/"main.py").is_file():
                return path
    raise RuntimeError("Could not find MFC repo root with Tanabe/SK54 source case.")


REPO = find_repo_root()
CASE_DIR = REPO/"examples"/"2D_dodecane_global_reduced"

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import case_hpc_d2_tanabe1996_dodecane_070mm_P20bar_T800_yao_sk54_128 as base


case = dict(base.case)
case.update({
    "parallel_io": "T",
    "file_per_process": "F",
})


if __name__ == "__main__":
    print(json.dumps(case))
