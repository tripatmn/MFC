"""Helpers for generating MFC thermochemistry modules."""


def _render_fortran_template(template_source, name, sol, opts):
    # Keep these imports local so non-chemistry command startup remains light.
    import cantera as ct  # pylint: disable=import-outside-toplevel
    import pyrometheus.chem_expr  # pylint: disable=import-outside-toplevel
    from mako.template import Template  # pylint: disable=import-outside-toplevel
    from pyrometheus.codegen import fortran  # pylint: disable=import-outside-toplevel

    falloff_rxn = [
        (i, r) for i, r in enumerate(sol.reactions())
        if r.reaction_type.startswith("falloff")
    ]
    three_body_rxn = [
        (i, r) for i, r in enumerate(sol.reactions())
        if r.reaction_type == "three-body-Arrhenius"
    ]

    if opts.directive_offload == "acc":
        gpu_routine_str = """
#ifdef _CRAYFTN
#define GPU_ROUTINE(name) !DIR$ INLINEALWAYS name
#else
#define GPU_ROUTINE(name) !$acc routine seq
#endif
"""
    elif opts.directive_offload == "mp":
        gpu_routine_str = """
#define GPU_ROUTINE(name) !$omp declare target
"""
    else:
        gpu_routine_str = """
#define GPU_ROUTINE(name) ! name
"""

    return fortran.wrap_code(Template(template_source).render(
        ct=ct,
        sol=sol,

        str_np=fortran.str_np,
        cgm=fortran.FortranExpressionMapper(),
        Variable=fortran.p.Variable,
        float_to_fortran=fortran.float_to_fortran,

        real_type=opts.scalar_type or "real(dp)",
        gpu_routine=gpu_routine_str,

        module_name=name,

        ce=pyrometheus.chem_expr,

        falloff_reactions=falloff_rxn,
        three_body_reactions=three_body_rxn
    ))


def generate_fortran_without_transport(name, sol, opts):
    """Generate a Pyrometheus Fortran module without transport routines.

    Some mechanisms are intentionally reactions-only and report Cantera
    transport model ``none``.  The installed Pyrometheus Fortran generator
    always emits transport routines, so it calls Cantera transport polynomial
    accessors during code generation even when MFC has species diffusion off.
    This helper reuses the upstream template with only the trailing transport
    routine block removed.
    """
    from pyrometheus.codegen import fortran  # pylint: disable=import-outside-toplevel

    source = fortran.module_tpl.source
    transport_beg = source.index("    subroutine get_species_viscosities")
    module_end = source.index("\nend module ${module_name}", transport_beg)
    reactions_only_source = source[:transport_beg] + source[module_end:]
    return _render_fortran_template(reactions_only_source, name, sol, opts)
