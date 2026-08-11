"""Resolve authoritative chemistry phase ordering and molecular weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# These are the element weights used by Cantera/Pyrometheus for the mechanisms
# currently supported by this checkout, as visible in generated m_thermochem.
_CANTERA_ELEMENT_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ar": 39.95,
    "AR": 39.95,
}


@dataclass(frozen=True)
class MechanismMetadata:
    path: str
    phase: str
    species_names: tuple[str, ...]
    molecular_weights: tuple[float, ...]
    gas_constant: float = 8314.46261815324


def resolve_mechanism(
    case_root: Path,
    params: dict[str, Any],
    mechanism_override: str | Path | None = None,
    phase_override: str | None = None,
) -> MechanismMetadata:
    configured = mechanism_override or params.get("cantera_file")
    if not configured:
        raise ValueError(
            "chemistry mechanism path is ambiguous: it is absent from generated inputs and case metadata; "
            "provide --mechanism and --phase"
        )
    path = _resolve_path(case_root, Path(configured))
    if path is None:
        raise ValueError(f"chemistry mechanism could not be resolved: {configured}")
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to resolve chemistry mechanism metadata") from exc
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to parse chemistry mechanism {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"chemistry mechanism is not a YAML mapping: {path}")
    phases = document.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"chemistry mechanism has no phase definitions: {path}")
    requested_phase = phase_override or params.get("cantera_phase")
    if requested_phase:
        matches = [item for item in phases if isinstance(item, dict) and item.get("name") == requested_phase]
        if len(matches) != 1:
            raise ValueError(f"chemistry phase {requested_phase!r} was not uniquely found in {path}")
        phase = matches[0]
    elif len(phases) == 1 and isinstance(phases[0], dict):
        phase = phases[0]
        requested_phase = phase.get("name")
    else:
        raise ValueError(f"chemistry phase is ambiguous in {path}; provide --phase")
    if not isinstance(requested_phase, str) or not requested_phase:
        raise ValueError(f"selected chemistry phase has no name in {path}")
    ordered = phase.get("species")
    if not isinstance(ordered, list) or not ordered or not all(isinstance(name, str) for name in ordered):
        raise ValueError(
            f"phase {requested_phase!r} does not contain an explicit ordered species-name list"
        )
    definitions = document.get("species")
    if not isinstance(definitions, list):
        raise ValueError(f"chemistry mechanism has no species definitions: {path}")
    by_name = {
        item.get("name"): item for item in definitions
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    custom_weights = _custom_element_weights(document)
    weights: list[float] = []
    for name in ordered:
        definition = by_name.get(name)
        if definition is None:
            raise ValueError(f"species {name!r} from phase {requested_phase!r} has no definition")
        composition = definition.get("composition")
        if not isinstance(composition, dict) or not composition:
            raise ValueError(f"species {name!r} has no elemental composition")
        weight = 0.0
        for element, count in composition.items():
            atomic_weight = custom_weights.get(str(element), _CANTERA_ELEMENT_WEIGHTS.get(str(element)))
            if atomic_weight is None:
                raise ValueError(
                    f"atomic weight for element {element!r} is unavailable; install Cantera support or add an explicit mechanism element weight"
                )
            weight += float(count) * atomic_weight
        if weight <= 0.0:
            raise ValueError(f"species {name!r} has nonpositive molecular weight")
        weights.append(weight)
    return MechanismMetadata(
        path=str(path), phase=requested_phase,
        species_names=tuple(ordered), molecular_weights=tuple(weights),
    )


def _resolve_path(case_root: Path, configured: Path) -> Path | None:
    if configured.is_absolute() and configured.is_file():
        return configured.resolve()
    candidates = [case_root / configured]
    repo = _repo_root(case_root)
    if repo is not None:
        candidates.append(repo / configured)
    candidates.append(Path.cwd() / configured)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _repo_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / "mfc.sh").is_file() and (candidate / "src").is_dir():
            return candidate
    return None


def _custom_element_weights(document: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    elements = document.get("elements", [])
    if isinstance(elements, list):
        for item in elements:
            if isinstance(item, dict) and isinstance(item.get("symbol"), str) and "atomic-weight" in item:
                result[item["symbol"]] = float(item["atomic-weight"])
    return result
