"""Read metadata without executing a user's case file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_ASSIGNMENT = re.compile(r"^\s*([^!&][^=]*?)\s*=\s*(.*?)\s*,?\s*$")


def parse_value(text: str) -> Any:
    value = text.strip().rstrip(",").strip()
    upper = value.upper().strip(".'\"")
    if upper in {"T", "TRUE"}:
        return True
    if upper in {"F", "FALSE"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    numeric = value.replace("D", "e").replace("d", "e")
    try:
        return int(numeric)
    except ValueError:
        try:
            return float(numeric)
        except ValueError:
            return value


def parse_namelist(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.is_file():
        return result
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line or line.startswith("&"):
            continue
        match = _ASSIGNMENT.match(line)
        if match:
            result[match.group(1).strip().lower()] = parse_value(match.group(2))
    return result


def parse_case_literals(path: Path) -> dict[str, Any]:
    """Recover only obvious literals; never import or execute case.py."""
    if not path.is_file():
        return {}
    text = path.read_text(errors="replace")
    result: dict[str, Any] = {}
    for key in (
        "chemistry", "cantera_file", "cantera_phase", "parallel_io",
        "file_per_process", "down_sample", "model_eqns", "num_fluids",
        "m", "n", "p", "dt", "t_save", "t_stop", "n_start",
        "t_step_start", "t_step_stop", "t_step_save", "igr", "mhd",
        "bubbles_euler", "bubbles_lagrange", "qbmm", "polytropic", "adv_n", "nb", "nmom",
        "hypoelasticity", "hyperelasticity", "surface_tension", "cont_damage",
        "hyper_cleaning", "cyl_coord", "precision",
    ):
        pattern = rf'["\']{re.escape(key)}["\']\s*:\s*(["\'][^"\']*["\']|[-+0-9.eEdD]+)'
        match = re.search(pattern, text)
        if match:
            result[key] = parse_value(match.group(1))
    for name, key in (("CTFILE", "cantera_file"), ("CTPHASE", "cantera_phase")):
        match = re.search(rf'^\s*{name}\s*=\s*(["\'][^"\']+["\'])', text, re.MULTILINE)
        if match:
            result[key] = parse_value(match.group(1))
    return result


def load_parameters(case_path: Path) -> tuple[dict[str, Any], list[str]]:
    params: dict[str, Any] = {}
    used: list[str] = []
    case_literals = parse_case_literals(case_path / "case.py")
    if case_literals:
        params.update(case_literals)
        used.append("case.py (static literals only)")
    for name in ("pre_process.inp", "simulation.inp", "post_process.inp"):
        parsed = parse_namelist(case_path / name)
        if parsed:
            params.update(parsed)
            used.append(name)
    if "chemistry" not in params:
        chemistry_signals = (
            "chem_params%reactions", "chem_params%diffusion", "fuel_species_id",
            "chem_gas_fluid_id", "chem_wrt_t",
        )
        params["chemistry"] = any(key in params for key in chemistry_signals)
    return params, used


def bool_param(params: dict[str, Any], name: str, default: bool = False) -> bool:
    value = params.get(name.lower(), default)
    if isinstance(value, str):
        return value.strip().upper().strip(".'\"") in {"T", "TRUE"}
    return bool(value)


def int_param(params: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(params.get(name.lower(), default))
    except (TypeError, ValueError):
        return default
