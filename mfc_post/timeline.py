"""Timeline inference that keeps saved indices, steps, and times distinct."""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable

from .config import bool_param
from .models import Timeline


def make_timeline(
    indices: Iterable[int], params: dict[str, Any], duplicates: Iterable[int] = (),
    saved_output: bool = False,
) -> Timeline:
    saved = tuple(sorted(set(indices)))
    duplicate_tuple = tuple(sorted(set(duplicates)))
    if not saved:
        return Timeline((), (), "unavailable", warnings=("no saved indices found",))
    adaptive = bool_param(params, "cfl_adap_dt") or bool_param(params, "cfl_const_dt")
    warnings: list[str] = []
    if _number(params.get("t_save")) is not None and (saved_output or adaptive):
        scale = float(params["t_save"])
        times = tuple(index * scale for index in saved)
        steps = tuple(None for _ in saved)
        basis = (
            "inferred: saved_index * t_save (saved output index is not simulation step)"
            if saved_output
            else "inferred: saved_index * t_save (adaptive output index is not simulation step)"
        )
    elif not saved_output and _number(params.get("dt")) is not None:
        scale = float(params["dt"])
        times = tuple(index * scale for index in saved)
        steps = tuple(saved)
        basis = "inferred: simulation_step * dt"
    else:
        times = tuple(None for _ in saved)
        steps = tuple(None for _ in saved)
        basis = "unavailable"
        warnings.append(
            "physical time cannot be inferred from t_save"
            if saved_output else "physical time cannot be inferred from t_save or dt"
        )
    deltas = [b - a for a, b in zip(times, times[1:]) if a is not None and b is not None]
    cadence = median(deltas) if deltas else None
    if cadence is not None and any(abs(delta - cadence) > max(abs(cadence), 1e-300) * 1e-9 for delta in deltas):
        warnings.append("saved-time cadence is irregular or contains gaps")
    starts = []
    for key in ("n_start", "t_step_start", "t_step_old"):
        value = params.get(key)
        if _number(value) is not None and float(value) > 0:
            starts.append({"kind": "configured_restart", "parameter": key, "value": value})
    if starts:
        warnings.append("restart configuration detected; inferred time may belong to a continuation segment")
        if saved_output:
            warnings.append(
                "restart continuation detected; the global saved-index/time offset is ambiguous"
            )
    configured_stop = _number(params.get("t_stop"))
    if configured_stop is None and _number(params.get("t_step_stop")) is not None and _number(params.get("dt")) is not None:
        configured_stop = float(params["t_step_stop"]) * float(params["dt"])
    known_times = [time for time in times if time is not None]
    if configured_stop is not None and known_times:
        tolerance = max(abs(configured_stop), abs(cadence or 0.0), 1e-300) * 1e-9
        if max(known_times) < configured_stop - tolerance:
            warnings.append(f"output ends at {max(known_times):.9g}, before configured stop {configured_stop:.9g}")
    if duplicate_tuple:
        warnings.append("duplicate saved indices detected")
    return Timeline(saved, times, basis, steps, cadence, tuple(starts), duplicate_tuple, tuple(warnings))


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
