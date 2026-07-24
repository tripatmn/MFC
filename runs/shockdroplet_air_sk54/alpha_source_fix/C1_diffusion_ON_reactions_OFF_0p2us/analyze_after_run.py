#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

case_dir = Path(__file__).resolve().parent
analyzer = case_dir.parent / "analyze_diffusion_case.py"
sys.argv = [
    str(analyzer),
    "--run-dir",
    str(case_dir),
    "--out-dir",
    str(case_dir / "analysis_diffusion_lightweight"),
]
runpy.run_path(str(analyzer), run_name="__main__")
