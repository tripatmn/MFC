"""Construct the state-vector layout from the active MFC model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import bool_param, int_param


@dataclass(frozen=True)
class EquationLayout:
    fields: tuple[dict[str, Any], ...]
    base_size: int
    total_size: int
    species_begin: int | None
    species_count: int
    warnings: tuple[str, ...] = ()


def build_equation_layout(params: dict[str, Any], observed_size: int | None = None) -> EquationLayout:
    m, n, p = (int_param(params, key, 0) for key in ("m", "n", "p"))
    dimensions = 1 + int(n > 0) + int(p > 0)
    mhd = bool_param(params, "mhd")
    num_vels = 3 if mhd else dimensions
    model = int_param(params, "model_eqns", 2)
    fluids = max(1, int_param(params, "num_fluids", 1))
    fields: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add(name: str, count: int = 1, representation: str = "conservative") -> tuple[int, int]:
        begin = len(fields) + 1
        for component in range(1, count + 1):
            canonical = name if count == 1 else f"{name}[{component}]"
            fields.append({"index": len(fields) + 1, "name": canonical, "representation": representation})
        return begin, len(fields)

    if model == 1:
        add("density")
        add("momentum", num_vels)
        add("total_energy_density")
        add("gamma_function")
        add("liquid_stiffness_function")
    elif model in {2, 3}:
        add("partial_density", fluids)
        add("momentum", num_vels)
        add("total_energy_density")
        alpha_count = fluids - 1 if model == 2 and bool_param(params, "igr") else fluids
        add("volume_fraction", max(0, alpha_count))
        if model == 3:
            add("partial_internal_energy", fluids)
    elif model == 4:
        add("density")
        add("momentum", num_vels)
        add("total_energy_density")
        add("void_fraction")
    else:
        warnings.append(f"unsupported model_eqns={model}; layout is incomplete")

    if model == 2 and bool_param(params, "bubbles_euler"):
        nb = max(0, int_param(params, "nb", 0))
        if bool_param(params, "qbmm"):
            add("bubble_moment", nb * max(0, int_param(params, "nmom", 6)))
        else:
            add("bubble_variable", nb * (2 if bool_param(params, "polytropic") else 4))
        if bool_param(params, "adv_n"):
            add("bubble_number_density")
    elif model == 4 and bool_param(params, "bubbles_euler"):
        nb = max(0, int_param(params, "nb", 0))
        add("bubble_variable", nb * (2 if bool_param(params, "polytropic") else 4))

    if model == 2 and bool_param(params, "bubbles_lagrange"):
        add("lagrangian_void_fraction")
        warnings.append(
            "Lagrangian beta ordering is ambiguous in this branch: post_process inserts beta into sys_size, while the simulation serial writer emits an extra q_cons_vf(sys_size+1) file"
        )

    if model == 2 and mhd:
        add("magnetic_field", 2 if dimensions == 1 else 3)

    if model in {2, 3}:
        if bool_param(params, "hypoelasticity") or bool_param(params, "hyperelasticity"):
            stress_count = dimensions * (dimensions + 1) // 2
            if bool_param(params, "cyl_coord"):
                stress_count += 1
            add("elastic_stress", stress_count)
        if bool_param(params, "hyperelasticity"):
            add("reference_map", dimensions)
            add("elastic_energy")
        if bool_param(params, "surface_tension"):
            add("color_function")
        if bool_param(params, "cont_damage"):
            add("damage_state")
        if bool_param(params, "hyper_cleaning"):
            add("hyperbolic_cleaning")

    base_size = len(fields)
    chemistry = bool_param(params, "chemistry")
    configured_species = int_param(params, "num_species", 0)
    if chemistry and configured_species <= 0 and observed_size is not None:
        configured_species = max(0, observed_size - base_size)
    species_begin = base_size + 1 if chemistry and configured_species > 0 else None
    species_names = params.get("species_names") or ()
    for species_id in range(configured_species):
        label = str(species_names[species_id]) if species_id < len(species_names) else str(species_id + 1)
        fields.append({
            "index": len(fields) + 1,
            "name": f"species_density[{label}]",
            "representation": "conservative",
        })
    if chemistry and configured_species == 0:
        warnings.append("chemistry is active but species count could not be determined")
    if observed_size is not None and len(fields) != observed_size:
        warnings.append(f"constructed sys_size={len(fields)} but observed {observed_size} raw fields")
    return EquationLayout(tuple(fields), base_size, len(fields), species_begin, configured_species, tuple(warnings))
