"""Authoritative Model-3 conservative-state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import bool_param, int_param
from .models import Field, PhysicalState, Provenance, RunMetadata, State


@dataclass(frozen=True)
class FluidEOS:
    fluid_id: int
    gamma: float
    pi_inf: float
    cv: float | None
    qv: float
    qvp: float | None


@dataclass(frozen=True)
class MaskThresholds:
    density_floor: float = 1.0e-16
    chemistry_density_floor: float = 1.0e-16
    pressure_floor: float = 1.0e2
    gas_dominated_liquid_alpha_max: float = 1.0e-2
    liquid_dominated_liquid_alpha_min: float = 0.5
    species_sum_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class Model3Configuration:
    dimensions: int
    num_fluids: int
    liquid_fluid_id: int
    gas_fluid_ids: tuple[int, ...]
    eos: tuple[FluidEOS, ...]
    species_names: tuple[str, ...]
    molecular_weights: tuple[float, ...]
    gas_constant: float
    mpp_lim: bool
    chemistry_reaction_heat_enabled: bool
    fixed_temperature_enabled: bool
    fixed_temperature: float | None
    temperature_min: float
    temperature_max: float
    thresholds: MaskThresholds
    equation_indices: Mapping[str, int]
    mechanism_path: str
    mechanism_phase: str

    @classmethod
    def from_metadata(
        cls, metadata: RunMetadata, thresholds: MaskThresholds | None = None
    ) -> "Model3Configuration":
        params = dict(metadata.parameters)
        if int_param(params, "model_eqns", -1) != 3:
            raise ValueError("physical reconstruction requires model_eqns=3")
        if metadata.dimensions not in {1, 2, 3}:
            raise ValueError("Model-3 dimensionality is ambiguous")
        unsupported = [
            name for name in ("mhd", "hypoelasticity", "hyperelasticity", "bubbles_euler")
            if bool_param(params, name)
        ]
        if unsupported:
            raise ValueError(f"Model-3 reconstruction does not support enabled physics: {unsupported}")
        num_fluids = int_param(params, "num_fluids", 0)
        if num_fluids <= 1:
            raise ValueError("this Model-3 reconstruction layer requires an unambiguous multi-fluid case")
        liquid_id = int_param(params, "evap_liquid_fluid_id", 0)
        if not 1 <= liquid_id <= num_fluids:
            raise ValueError("liquid fluid role is ambiguous: evap_liquid_fluid_id is absent or invalid")
        gas_count = int_param(params, "chem_gas_num_fluids", 0)
        if gas_count > 0:
            gas_ids = tuple(int_param(params, f"chem_gas_fluid_ids({i})", 0) for i in range(1, gas_count + 1))
        else:
            gas_ids = (int_param(params, "chem_gas_fluid_id", 0),)
        if not gas_ids or len(set(gas_ids)) != len(gas_ids) or any(not 1 <= i <= num_fluids for i in gas_ids):
            raise ValueError("chemistry gas-fluid roles are absent, duplicated, or invalid")
        if liquid_id in gas_ids:
            raise ValueError("liquid and chemistry-gas fluid roles overlap")
        eos: list[FluidEOS] = []
        for fluid_id in range(1, num_fluids + 1):
            prefix = f"fluid_pp({fluid_id})%"
            required = [prefix + name for name in ("gamma", "pi_inf", "qv")]
            missing = [name for name in required if name not in params]
            if missing:
                raise ValueError(f"EOS metadata is incomplete for fluid {fluid_id}: missing {missing}")
            eos.append(FluidEOS(
                fluid_id=fluid_id,
                gamma=float(params[prefix + "gamma"]), pi_inf=float(params[prefix + "pi_inf"]),
                cv=_optional_float(params.get(prefix + "cv")), qv=float(params[prefix + "qv"]),
                qvp=_optional_float(params.get(prefix + "qvp")),
            ))
        species = tuple(metadata.species_names)
        weights = tuple(float(value) for value in params.get("molecular_weights", ()))
        mechanism_path = params.get("cantera_file")
        mechanism_phase = params.get("cantera_phase")
        if not species or not weights or len(species) != len(weights):
            raise ValueError(
                "mechanism species names and molecular weights are required and must match; provide --mechanism/--phase"
            )
        if not mechanism_path or not mechanism_phase:
            raise ValueError("mechanism path or phase is ambiguous")
        indices = {str(item["name"]): int(item["index"]) for item in metadata.equation_layout}
        expected = _required_equation_names(num_fluids, metadata.dimensions, species)
        missing_fields = [name for name in expected if name not in indices]
        if missing_fields:
            raise ValueError(f"Model-3 equation layout is incomplete: {missing_fields[:12]}")
        core = _core_equation_names(num_fluids, metadata.dimensions)
        for offset, name in enumerate(core, 1):
            if indices[name] != offset:
                raise ValueError(
                    f"Model-3 equation layout mismatch for {name}: observed {indices[name]}, expected {offset}"
                )
        observed_species = [
            str(item["name"])[len("species_density["):-1]
            for item in sorted(metadata.equation_layout, key=lambda item: int(item["index"]))
            if str(item["name"]).startswith("species_density[")
        ]
        if observed_species != list(species):
            raise ValueError(
                "equation-layout species ordering does not match the selected mechanism phase: "
                f"layout={observed_species[:8]}, mechanism={list(species[:8])}"
            )
        species_indices = [indices[f"species_density[{name}]"] for name in species]
        if species_indices != list(range(species_indices[0], species_indices[0] + len(species))):
            raise ValueError("chemistry species equations are not one contiguous ordered block")
        threshold_values = thresholds or MaskThresholds(
            gas_dominated_liquid_alpha_max=float(params.get("evap_alpha_thresh", 1.0e-2))
        )
        if not 0.0 <= threshold_values.gas_dominated_liquid_alpha_max < threshold_values.liquid_dominated_liquid_alpha_min <= 1.0:
            raise ValueError("mask alpha thresholds must satisfy 0 <= gas < liquid <= 1")
        fixed_enabled = bool_param(params, "chem_fixed_t_enable")
        fixed_value = _optional_float(params.get("chem_fixed_t"))
        if fixed_enabled and fixed_value is None:
            raise ValueError("chem_fixed_T_enable=T but chem_fixed_T is absent")
        if fixed_enabled and fixed_value <= 0.0:
            raise ValueError("chem_fixed_T must be positive")
        temperature_min = float(params.get("chem_t_min", 250.0))
        temperature_max = float(params.get("chem_t_max", 3000.0))
        if temperature_min <= 0.0 or temperature_max < temperature_min:
            raise ValueError("chemistry temperature bounds are invalid")
        return cls(
            dimensions=metadata.dimensions, num_fluids=num_fluids,
            liquid_fluid_id=liquid_id, gas_fluid_ids=gas_ids, eos=tuple(eos),
            species_names=species, molecular_weights=weights,
            gas_constant=float(params.get("gas_constant", 8314.46261815324)),
            mpp_lim=bool_param(params, "mpp_lim"),
            chemistry_reaction_heat_enabled=bool_param(params, "chem_reaction_heat_enable"),
            fixed_temperature_enabled=fixed_enabled, fixed_temperature=fixed_value,
            temperature_min=temperature_min, temperature_max=temperature_max,
            thresholds=threshold_values, equation_indices=indices,
            mechanism_path=str(mechanism_path), mechanism_phase=str(mechanism_phase),
        )


class FieldRegistry:
    """Resolve symbolic physical fields without numeric equation offsets."""

    def __init__(self, physical_state: PhysicalState, config: Model3Configuration):
        self.physical_state = physical_state
        self.config = config

    def resolve(self, symbol: str) -> Field:
        canonical = self._canonical(symbol.strip())
        if canonical in self.physical_state.fields:
            return self.physical_state.fields[canonical]
        if canonical in self.physical_state.masks:
            return self.physical_state.masks[canonical]
        raise KeyError(f"unknown physical field symbol {symbol!r}")

    def _canonical(self, symbol: str) -> str:
        aliases = {
            "rho": "density", "p": "pressure", "T": "temperature",
            "alpha[liquid]": f"alpha[{self.config.liquid_fluid_id}]",
            "alpha_rho[liquid]": f"alpha_rho[{self.config.liquid_fluid_id}]",
        }
        if symbol in aliases:
            return aliases[symbol]
        for prefix in ("Y[", "rhoY["):
            if symbol.startswith(prefix) and symbol.endswith("]"):
                requested = symbol[len(prefix):-1]
                matches = [name for name in self.config.species_names if name.casefold() == requested.casefold()]
                if len(matches) != 1:
                    raise KeyError(f"species {requested!r} is absent or ambiguous")
                return f"{prefix}{matches[0]}]"
        return symbol


def reconstruct_model3(state: State, config: Model3Configuration) -> PhysicalState:
    arrays = {
        name: np.asarray(field.values, dtype=np.float64)
        for name, field in state.fields.items()
    }
    expected_shape = next(iter(arrays.values())).shape if arrays else ()
    required = _required_equation_names(config.num_fluids, config.dimensions, config.species_names)
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"saved index {state.saved_index}: required Model-3 fields are absent: {missing[:12]}")
    if any(value.shape != expected_shape for value in arrays.values()):
        raise ValueError(f"saved index {state.saved_index}: raw field shapes differ")

    raw_alpha_rho = np.stack([arrays[f"partial_density[{i}]"] for i in range(1, config.num_fluids + 1)])
    raw_alpha = np.stack([arrays[f"volume_fraction[{i}]"] for i in range(1, config.num_fluids + 1)])
    alpha_rho = raw_alpha_rho.copy()
    alpha = raw_alpha.copy()
    transformations: list[str] = []
    if config.mpp_lim:
        alpha_rho = np.maximum(alpha_rho, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_sum = np.sum(alpha, axis=0)
        alpha = alpha / np.maximum(alpha_sum, config.thresholds.density_floor)
        transformations.extend((
            "mpp_lim: partial densities clipped to >=0",
            "mpp_lim: volume fractions clipped to [0,1] and normalized by max(sum(alpha),1e-16)",
        ))
    else:
        alpha_sum = np.sum(alpha, axis=0)
    density = np.sum(alpha_rho, axis=0)
    momentum = np.stack([
        arrays["momentum" if config.dimensions == 1 else f"momentum[{i}]"]
        for i in range(1, config.dimensions + 1)
    ])
    velocity = np.full_like(momentum, np.nan)
    positive_density = density > config.thresholds.density_floor
    np.divide(momentum, density, out=velocity, where=positive_density)
    kinetic = 0.5 * np.sum(momentum * velocity, axis=0)
    total_energy = arrays["total_energy_density"]
    mixture_internal = total_energy - kinetic
    eos_gamma = np.sum(alpha * np.asarray([item.gamma for item in config.eos]).reshape((-1,) + (1,) * len(expected_shape)), axis=0)
    eos_pi = np.sum(alpha * np.asarray([item.pi_inf for item in config.eos]).reshape((-1,) + (1,) * len(expected_shape)), axis=0)
    eos_qv = np.sum(alpha_rho * np.asarray([item.qv for item in config.eos]).reshape((-1,) + (1,) * len(expected_shape)), axis=0)
    raw_pressure = np.full(expected_shape, np.nan, dtype=np.float64)
    valid_denominator = np.isfinite(eos_gamma) & (eos_gamma > config.thresholds.density_floor)
    np.divide(mixture_internal - eos_pi - eos_qv, eos_gamma, out=raw_pressure, where=valid_denominator)
    pressure = np.maximum(raw_pressure, config.thresholds.pressure_floor)

    gas_density = np.sum(
        [raw_alpha_rho[fluid_id - 1] for fluid_id in config.gas_fluid_ids], axis=0
    )
    gas_divisor = np.maximum(gas_density, config.thresholds.chemistry_density_floor)
    raw_species = np.stack([arrays[f"species_density[{name}]"] for name in config.species_names])
    mass_fractions = np.clip(raw_species / gas_divisor, 0.0, 1.0)
    y_sum_before = np.sum(mass_fractions, axis=0)
    overfull = y_sum_before > 1.0
    y_normalizer = np.where(overfull, y_sum_before, 1.0)
    mass_fractions = mass_fractions / y_normalizer[None, ...]
    y_sum = np.sum(mass_fractions, axis=0)
    reciprocal_mw = np.sum(
        mass_fractions / np.asarray(config.molecular_weights).reshape((-1,) + (1,) * len(expected_shape)), axis=0
    )
    molecular_weight = np.full(expected_shape, np.nan)
    np.divide(1.0, reciprocal_mw, out=molecular_weight, where=reciprocal_mw > 0.0)
    temperature_raw = np.full(expected_shape, np.nan)
    temperature = np.full(expected_shape, np.nan)
    if config.fixed_temperature_enabled:
        fixed = min(max(float(config.fixed_temperature), config.temperature_min), config.temperature_max)
        temperature_raw.fill(float(config.fixed_temperature))
        temperature.fill(fixed)
    elif config.chemistry_reaction_heat_enabled:
        temp_candidate = pressure * molecular_weight / (config.gas_constant * gas_divisor)
        temp_ok = (
            (gas_density > config.thresholds.chemistry_density_floor)
            & np.isfinite(pressure) & (pressure > 0.0)
            & np.isfinite(molecular_weight) & (molecular_weight > 0.0)
            & np.isfinite(temp_candidate) & (temp_candidate > 0.0)
        )
        temperature_raw[temp_ok] = temp_candidate[temp_ok]
        temperature[temp_ok] = np.clip(
            temp_candidate[temp_ok], config.temperature_min, config.temperature_max
        )

    finite_raw = np.logical_and.reduce([np.isfinite(arrays[name]) for name in required])
    base_valid = finite_raw & positive_density & valid_denominator & np.isfinite(pressure)
    thermo_invalid = ~base_valid | ~np.isfinite(raw_pressure) | (raw_pressure <= 0.0)
    chemistry_valid = (
        ~thermo_invalid
        & (gas_density > config.thresholds.chemistry_density_floor)
        & np.isfinite(molecular_weight) & (molecular_weight > 0.0)
        & np.isfinite(temperature_raw) & (temperature_raw > 0.0)
        & (np.abs(y_sum - 1.0) <= config.thresholds.species_sum_tolerance)
    )
    liquid_alpha = alpha[config.liquid_fluid_id - 1]
    valid = base_valid
    gas_dominated = valid & (liquid_alpha <= config.thresholds.gas_dominated_liquid_alpha_max)
    liquid_dominated = valid & (liquid_alpha > config.thresholds.liquid_dominated_liquid_alpha_min)
    interface = valid & ~gas_dominated & ~liquid_dominated

    source = state.provenance.source_path
    eos_notes = tuple(
        f"fluid {item.fluid_id}: gamma={item.gamma}, pi_inf={item.pi_inf}, cv={item.cv}, qv={item.qv}, qvp={item.qvp}"
        for item in config.eos
    )
    derived: dict[str, Field] = {}

    def add(name: str, values, units: str | None, derivation: str, extra=()):
        derived[name] = Field(
            name=name, representation="derived", units=units, values=values,
            provenance=Provenance(
                state.provenance.source_family, source, transformations=(derivation, *extra),
                notes=eos_notes if name in {"pressure", "pressure.raw", "eos.gamma", "eos.pi_inf", "eos.qv"} else (),
            ),
        )

    for i in range(config.num_fluids):
        add(f"alpha_rho[{i + 1}]", alpha_rho[i], "kg/m^3", "copied from raw partial density" if not config.mpp_lim else transformations[0])
        add(f"alpha[{i + 1}]", alpha[i], "1", "copied from raw volume fraction" if not config.mpp_lim else transformations[1])
        add(f"internal_energy[{i + 1}]", arrays[f"partial_internal_energy[{i + 1}]"], "J/m^3", "raw Model-3 phasic internal-energy density")
    add("density", density, "kg/m^3", "sum effective partial densities")
    axis_names = ("x", "y", "z")
    for i in range(config.dimensions):
        add(f"velocity[{axis_names[i]}]", velocity[i], "m/s", "momentum divided by effective mixture density")
    add("energy.total_density", total_energy, "J/m^3", "raw total mixture energy density")
    add("energy.kinetic_density", kinetic, "J/m^3", "0.5*sum(momentum*velocity)")
    add("energy.internal_density", mixture_internal, "J/m^3", "total energy density minus kinetic-energy density")
    add("eos.gamma", eos_gamma, "1", "sum(alpha_i*fluid_gamma_i)")
    add("eos.pi_inf", eos_pi, "Pa", "sum(alpha_i*fluid_pi_inf_i)")
    add("eos.qv", eos_qv, "J/m^3", "sum(alpha_rho_i*fluid_qv_i)")
    add("pressure.raw", raw_pressure, "Pa", "(E-kinetic-Pi_mix-qv_mix)/Gamma_mix")
    add("pressure", pressure, "Pa", "solver multi-fluid Model-3 EOS", ("chemistry pressure floor max(p,100 Pa)",))
    add("gas_density", gas_density, "kg/m^3", "sum raw conservative partial densities for configured chemistry gas fluids")
    for i, name in enumerate(config.species_names):
        add(f"rhoY[{name}]", raw_species[i], "kg/m^3", "raw conservative species density")
        add(f"Y[{name}]", mass_fractions[i], "1", "clip(rhoY/rho_g,0,1), normalized only when bounded sum exceeds one")
    add("Y.sum", y_sum, "1", "sum solver-bounded gas mass fractions")
    add("molecular_weight", molecular_weight, "kg/kmol", "1/sum(Y_k/W_k)")
    add("temperature.raw", temperature_raw, "K", "solver reconstructed q_T before chemistry-source bounds")
    add(
        "temperature", temperature, "K",
        "temperature consumed by chemistry: reconstructed q_T clipped to [chem_T_min,chem_T_max]",
    )
    phasic_sum = np.sum([arrays[f"partial_internal_energy[{i}]"] for i in range(1, config.num_fluids + 1)], axis=0)
    add("energy.phasic_sum_density", phasic_sum, "J/m^3", "sum raw Model-3 phasic internal-energy densities")
    add("energy.phasic_closure_residual", phasic_sum - mixture_internal, "J/m^3", "phasic internal-energy sum minus E-kinetic")

    masks = {
        name: Field(
            name=name, representation="mask", units="1", values=values,
            provenance=Provenance(
                state.provenance.source_family, source,
                transformations=(definition,),
            ),
        )
        for name, values, definition in (
            ("mask.valid", valid, "finite raw state, rho>1e-16, Gamma_mix>1e-16, finite solver pressure"),
            ("mask.gas_dominated", gas_dominated, f"valid and alpha_liquid<={config.thresholds.gas_dominated_liquid_alpha_max}"),
            ("mask.liquid_dominated", liquid_dominated, f"valid and alpha_liquid>{config.thresholds.liquid_dominated_liquid_alpha_min}"),
            ("mask.interface", interface, "valid and neither gas- nor liquid-dominated"),
            ("mask.thermodynamically_invalid", thermo_invalid, "invalid raw pressure, density, or EOS"),
            ("mask.chemistry_valid", chemistry_valid, "thermodynamically valid gas with positive rho_g, molecular weight, temperature, and mass-fraction closure"),
        )
    }
    provenance = Provenance(
        state.provenance.source_family, source,
        transformations=tuple(transformations) + (
            "Model-3 mixture EOS reconstruction",
            "solver-bounded gas mass fractions",
            "mixture molecular weight and chemistry temperature reconstruction",
        ),
        notes=(
            f"mechanism={config.mechanism_path}", f"phase={config.mechanism_phase}",
            f"species_count={len(config.species_names)}", f"liquid_fluid_id={config.liquid_fluid_id}",
            f"gas_fluid_ids={config.gas_fluid_ids}", f"mask_thresholds={config.thresholds}",
            *eos_notes,
        ),
    )
    return PhysicalState(state, derived, masks, provenance)


def _required_equation_names(num_fluids: int, dimensions: int, species: tuple[str, ...]) -> tuple[str, ...]:
    return _core_equation_names(num_fluids, dimensions) + tuple(
        [f"species_density[{name}]" for name in species]
    )


def _core_equation_names(num_fluids: int, dimensions: int) -> tuple[str, ...]:
    return tuple(
        [f"partial_density[{i}]" for i in range(1, num_fluids + 1)]
        + (["momentum"] if dimensions == 1 else [f"momentum[{i}]" for i in range(1, dimensions + 1)])
        + ["total_energy_density"]
        + [f"volume_fraction[{i}]" for i in range(1, num_fluids + 1)]
        + [f"partial_internal_energy[{i}]" for i in range(1, num_fluids + 1)]
    )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
