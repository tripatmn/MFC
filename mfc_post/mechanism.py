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


@dataclass(frozen=True)
class LoadedCanteraMechanism:
    metadata: MechanismMetadata
    gas: Any


def load_cantera_mechanism(
    case_root: Path,
    params: dict[str, Any],
    mechanism_override: str | Path | None = None,
    phase_override: str | None = None,
) -> LoadedCanteraMechanism:
    """Load one authoritative Cantera phase for an explicitly requested diagnostic."""
    configured = mechanism_override or params.get("cantera_file")
    requested_phase = phase_override or params.get("cantera_phase")
    if not configured:
        raise ValueError(
            "Cantera mechanism is absent from --mechanism and case metadata; "
            "provide --mechanism and --phase"
        )
    configured_path = Path(configured).expanduser()
    candidates = _candidate_paths(case_root, configured_path)
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    attempted = ", ".join(str(candidate) for candidate in candidates)
    if path is None:
        raise ValueError(
            f"Cantera mechanism could not be resolved from {configured!s}; "
            f"attempted paths=[{attempted}], phase={requested_phase!r}"
        )
    try:
        import cantera as ct
    except ImportError as exc:
        raise ValueError(
            "--compute-heat-release cantera requires the Cantera Python package; "
            f"mechanism={path}, phase={requested_phase!r}"
        ) from exc
    try:
        gas = ct.Solution(str(path), requested_phase) if requested_phase else ct.Solution(str(path))
    except Exception as exc:
        raise ValueError(
            f"failed to load Cantera mechanism={path}, phase={requested_phase!r}: {exc}"
        ) from exc
    selected_phase = getattr(gas, "name", None) or requested_phase
    if not selected_phase:
        raise ValueError(f"Cantera loaded {path}, but the selected phase name is unavailable")
    species_names = tuple(str(name) for name in gas.species_names)
    molecular_weights = tuple(float(value) for value in gas.molecular_weights)
    if not species_names or len(species_names) != len(molecular_weights):
        raise ValueError(
            f"Cantera phase {selected_phase!r} in {path} returned inconsistent species metadata"
        )
    return LoadedCanteraMechanism(
        metadata=MechanismMetadata(
            path=str(path), phase=str(selected_phase), species_names=species_names,
            molecular_weights=molecular_weights,
            gas_constant=float(getattr(ct, "gas_constant", 8314.46261815324)),
        ),
        gas=gas,
    )


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
    return next(
        (path.resolve() for path in _candidate_paths(case_root, configured) if path.is_file()),
        None,
    )


def _candidate_paths(case_root: Path, configured: Path) -> list[Path]:
    if configured.is_absolute():
        return [configured]
    candidates = [case_root / configured]
    repo = _repo_root(case_root)
    if repo is not None:
        candidates.append(repo / configured)
    candidates.append(Path.cwd() / configured)
    # Preserve resolution precedence while avoiding repeated attempted paths.
    return list(dict.fromkeys(candidates))


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
