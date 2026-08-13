"""Coordinate source-family inspection without merging their state inventories."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import bool_param, int_param, load_parameters
from .equations import build_equation_layout
from .mechanism import MechanismMetadata, resolve_mechanism
from .models import RunMetadata, SourceReport, to_dict
from .sources import BinarySource, DAsciiSource, LustrePerProcessSource, LustreSharedSource, PAllSource, SiloSource


def inspect_case(
    case_path: str | Path,
    mechanism: str | Path | None = None,
    phase: str | None = None,
    resolved_mechanism: MechanismMetadata | None = None,
) -> dict[str, Any]:
    root = Path(case_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"case directory does not exist: {root}")
    params, parameter_sources = load_parameters(root)
    observed_size = _observed_raw_size(root)
    species_names, species_warnings = _recover_species(
        root, params, observed_size, mechanism, phase, resolved_mechanism,
    )
    if species_names:
        params["species_names"] = species_names
        params["num_species"] = len(species_names)
        if resolved_mechanism is not None:
            # Explicit HRR requests establish that the stored trailing equations
            # are the selected mechanism's global chemistry-species block.
            params["chemistry"] = True
    layout = build_equation_layout(params, observed_size)
    n, p = int_param(params, "n", 0), int_param(params, "p", 0)
    dimensions = 1 + int(n > 0) + int(p > 0) if "m" in params else None
    shape = None
    if dimensions is not None:
        shape = tuple(int_param(params, axis, 0) + 1 for axis in ("m", "n", "p")[:dimensions])
    missing = [key for key in ("m", "n", "p", "model_eqns", "num_fluids") if key not in params]
    metadata = RunMetadata(
        case_path=str(root), parameters=params, dimensions=dimensions, grid_shape=shape,
        equation_layout=layout.fields, species_names=tuple(species_names), missing=tuple(missing),
        warnings=tuple(layout.warnings) + tuple(species_warnings),
    )
    source_types = (LustreSharedSource, LustrePerProcessSource, PAllSource, DAsciiSource, BinarySource, SiloSource)
    reports = [source.inspect() for cls in source_types if (source := cls(root, metadata)).available()]
    conflicts = _detect_conflicts(reports, params)
    recommendation = _recommend(reports, len(layout.fields))
    warnings = list(metadata.warnings)
    if dimensions is None:
        warnings.append("dimensionality is unknown; equation indices that depend on velocity count are provisional")
    if not reports:
        warnings.append("no recognized MFC output source found")
    return {
        "schema_version": "mfc-post.inspect/v1",
        "case": str(root),
        "metadata_sources": parameter_sources,
        "run_metadata": to_dict(metadata),
        "equation_layout": {
            "base_size": layout.base_size,
            "total_size": layout.total_size,
            "species_begin": layout.species_begin,
            "species_count": layout.species_count,
            "fields": list(layout.fields),
        },
        "sources": [report.to_dict() for report in reports],
        "timeline": {report.family: to_dict(report.timeline) for report in reports},
        "conflicts": conflicts,
        "recommendation": recommendation,
        "warnings": warnings,
    }


def _observed_raw_size(root: Path) -> int | None:
    empty = RunMetadata(str(root), {}, None, None)
    for source_type in (PAllSource, DAsciiSource):
        source = source_type(root, empty)
        if source.available() and (count := source.observed_field_count()) is not None:
            return count
    restart = root / "restart_data"
    shared = sorted(restart.glob("lustre_[0-9]*.dat")) if restart.is_dir() else []
    params, _ = load_parameters(root)
    if shared and all(key in params for key in ("m", "n", "p")):
        cells = (int_param(params, "m") + 1) * max(1, int_param(params, "n") + 1) * max(1, int_param(params, "p") + 1)
        size = shared[0].stat().st_size
        if cells and size % (cells * 8) == 0:
            return size // (cells * 8)
    return None


def _recover_species(
    root: Path, params: dict[str, Any], observed_size: int | None,
    mechanism_override: str | Path | None = None, phase_override: str | None = None,
    resolved_mechanism: MechanismMetadata | None = None,
) -> tuple[list[str], list[str]]:
    if resolved_mechanism is None and not bool_param(params, "chemistry"):
        return [], []
    warnings: list[str] = []
    try:
        resolved = resolved_mechanism or resolve_mechanism(
            root, params, mechanism_override, phase_override,
        )
        if observed_size is not None:
            params["num_species"] = len(resolved.species_names)
        params["cantera_file"] = resolved.path
        params["cantera_phase"] = resolved.phase
        params["molecular_weights"] = resolved.molecular_weights
        params["gas_constant"] = resolved.gas_constant
        return list(resolved.species_names), warnings
    except ValueError as exc:
        return [], [str(exc)]


def _repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "mfc.sh").is_file() and (candidate / "src").is_dir():
            return candidate
    return path


def _detect_conflicts(reports: list[SourceReport], params: dict[str, Any]) -> list[dict[str, Any]]:
    by_family = {report.family: report for report in reports}
    conflicts: list[dict[str, Any]] = []
    has_lustre = "lustre_shared" in by_family or "lustre_per_process" in by_family
    has_serial = "p_all" in by_family or "D" in by_family
    if has_lustre and has_serial:
        conflicts.append({
            "kind": "alternative_writer_families_coexist",
            "message": "Lustre and p_all/D are exclusive writer modes in this branch; coexistence indicates stale, copied, or multi-segment output.",
        })
    expected_parallel = bool_param(params, "parallel_io")
    if expected_parallel and has_serial:
        conflicts.append({"kind": "configuration_mismatch", "message": "current metadata says parallel_io=T but serial p_all/D output exists"})
    if not expected_parallel and has_lustre:
        conflicts.append({"kind": "configuration_mismatch", "message": "current metadata says parallel_io=F but Lustre output exists"})
    for i, left in enumerate(reports):
        left_indices = set(left.timeline.saved_indices)
        for right in reports[i + 1:]:
            overlap = sorted(left_indices & set(right.timeline.saved_indices))
            if overlap:
                conflicts.append({
                    "kind": "overlapping_saved_indices", "sources": [left.family, right.family],
                    "count": len(overlap), "indices": overlap[:50],
                    "message": "overlap is reported only; sources were not combined or deduplicated",
                })
    return conflicts


def _recommend(reports: list[SourceReport], expected_fields: int) -> dict[str, Any] | None:
    reports = [report for report in reports if report.timeline.saved_indices]
    if not reports:
        return None
    spans = []
    cadences = []
    for report in reports:
        times = [time for time in report.timeline.physical_times if time is not None]
        spans.append((max(times) - min(times)) if len(times) > 1 else 0.0)
        if report.timeline.cadence:
            cadences.append(report.timeline.cadence)
    max_span = max(spans, default=0.0)
    min_cadence = min(cadences, default=None)
    ranked = []
    for report, span in zip(reports, spans):
        completeness = min(1.0, len(report.fields) / max(1, expected_fields))
        precision = 1.0 if report.precision and ("64" in report.precision or "double" in report.precision) else 0.5
        resolution = 0.4 if report.downsampled else 1.0
        coverage = span / max_span if max_span else (1.0 if report.timeline.saved_indices else 0.0)
        cadence = 1.0
        if min_cadence and report.timeline.cadence:
            cadence = min(1.0, min_cadence / report.timeline.cadence)
        raw_bonus = 0.25 if report.raw_fields else 0.0
        score = 0.30 * completeness + 0.20 * precision + 0.15 * resolution + 0.20 * coverage + 0.15 * cadence + raw_bonus
        ranked.append((score, report, {"field_completeness": completeness, "precision": precision, "resolution": resolution, "temporal_coverage": coverage, "cadence": cadence, "raw_completeness_bonus": raw_bonus}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    score, winner, factors = ranked[0]
    return {
        "source": winner.family, "path": winner.path, "score": round(score, 6), "factors": factors,
        "reason": "highest case-specific score across field completeness, precision, resolution, temporal coverage, cadence, and preservation of raw state",
        "alternatives": [{"source": report.family, "score": round(value, 6)} for value, report, _ in ranked[1:]],
    }
