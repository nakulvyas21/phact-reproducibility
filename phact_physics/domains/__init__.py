"""
PHACT Domain Registry
====================
Generic plugin system - any physics domain can be registered here.
The agent's tool list is built dynamically from the registry.

Usage:
    from phact_physics.domains import DOMAIN_REGISTRY, get_tools_for_domain

    tools = get_tools_for_domain("drug_formulation")
    # → list of Python functions, each becomes an ADK FunctionTool
"""

from __future__ import annotations
from typing import Callable

# ── Registry structure ───────────────────────────────────────────────────────

DOMAIN_REGISTRY: dict[str, dict] = {}


def register_domain(
    name:        str,
    description: str,
    tools:       list[Callable],
    requires:    list[str] = None,
) -> None:
    """
    Register a physics domain and its validation tools.

    Args:
        name:        Domain identifier (e.g. "drug_formulation")
        description: One-line description shown to the agent
        tools:       List of Python functions - each becomes an ADK FunctionTool.
                     Docstrings are passed directly to Gemini as tool specs.
        requires:    Optional list of pip packages needed (for user guidance)
    """
    DOMAIN_REGISTRY[name] = {
        "name":        name,
        "description": description,
        "tools":       tools,
        "requires":    requires or [],
    }


def _wrap_tool(fn: Callable) -> Callable:
    """
    Wrap a physics tool so it always returns a plain dict.
    ADK FunctionTool requires dict returns - it cannot build a JSON schema
    for dataclass or TypedDict return annotations.

    Also coerces kwargs to their annotated types before calling the function.
    Weaker models (e.g. Llama 3.1 8B) occasionally pass numeric args as
    strings ("25") or single-element lists (["25"]) - this silently fixes both.
    """
    import dataclasses
    import functools
    import inspect

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Coerce keyword arguments to their annotated types
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs:
                continue
            val = kwargs[param_name]
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                continue
            # Unwrap single-element list (["25"] → "25")
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            # LLMs sometimes pass "None"/"null" as a string for optional params
            if isinstance(val, str) and val.lower() in ("none", "null", ""):
                val = None
                kwargs[param_name] = val
                continue
            # Unwrap dict passed as a scalar - LLMs sometimes wrap values in
            # {"value": x} or {"amount": x} dicts; extract the first numeric value
            if isinstance(val, dict) and ann in (float, int):
                for v in val.values():
                    if isinstance(v, (int, float)):
                        val = v
                        break
                    try:
                        val = ann(v)
                        break
                    except (TypeError, ValueError):
                        continue
            # Cast to annotated scalar type (str "25" → float 25.0)
            if ann in (float, int) and not isinstance(val, (float, int)):
                try:
                    val = ann(val)
                except (TypeError, ValueError):
                    pass
            kwargs[param_name] = val

        # Guard: if required params are missing or None, return an error dict
        # instead of crashing. LLMs sometimes call tools with empty args or
        # pass None for required numeric fields.
        all_required = [
            p for p, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        bad = [
            p for p in all_required
            if p not in kwargs or kwargs[p] is None
        ]
        if bad:
            all_params_desc = ", ".join(
                f"{p}: {sig.parameters[p].annotation.__name__ if sig.parameters[p].annotation is not inspect.Parameter.empty else 'any'}"
                for p in sig.parameters
            )
            return {
                "passed": False,
                "error": (
                    f"Tool '{fn.__name__}' called with missing or null required parameters: {bad}. "
                    f"This tool requires ALL of these exact parameter names: {all_required}. "
                    f"Full signature: {fn.__name__}({all_params_desc}). "
                    f"Call the tool again using these exact argument names."
                ),
                "violations": [f"Tool called with missing/null args: {bad}"],
                "correction_hint": (
                    f"Required parameters for {fn.__name__}: {all_required}. "
                    f"You must pass numeric values for each. "
                    f"Full parameter list with types: {all_params_desc}."
                ),
            }

        # Run the physics tool. If it raises on a malformed or degenerate input
        # (e.g. a zero separation causing division by zero, or an argument that
        # slipped through coercion as the wrong type), convert the crash into a
        # structured rejection the model can correct from - never let a tool
        # exception abort the whole PHACT run, which would otherwise be recorded
        # as a spurious loop-closure rather than a real model outcome.
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            return {
                "passed": False,
                "error": (
                    f"Tool '{fn.__name__}' raised an error on the given inputs: "
                    f"{type(e).__name__}: {e}. The inputs are likely degenerate "
                    f"(e.g. a zero or negative value where a positive physical "
                    f"quantity is required) or of the wrong type."
                ),
                "violations": [f"Invalid input to {fn.__name__}: {type(e).__name__}: {e}"],
                "correction_hint": (
                    f"Check that every argument to {fn.__name__} is a positive, "
                    f"physically sensible number, then call the tool again."
                ),
            }
        if dataclasses.is_dataclass(result):
            return dataclasses.asdict(result)
        return result

    # Strip the return annotation so ADK doesn't try to introspect it
    wrapper.__signature__ = sig.replace(return_annotation=inspect.Parameter.empty)
    return wrapper


def get_tools_for_domain(domain: str) -> list[Callable]:
    """Return the list of physics tool functions for a given domain, wrapped as plain-dict returns.

    Three certification-contract modes select which tools the proposer sees. They
    are the experimental variable in the certification-integrity ablation: the
    goal text is identical across modes, only the available tools change.

      (default)               all tools, including the contract-gated
                              verify_binary_target alongside the bare validators.
      PHACT_DISABLE_SCM_GATE=1
                              removes verify_binary_target - bare answer-submitting
                              validators only. The gameable control arm.
      PHACT_SCM_EXCLUSIVE=1   keeps ONLY verify_binary_target (drops the bare
                              answer-submitting astro validators). The deployed
                              safe arm: certification authority is the SOLE path,
                              not merely one option. This is the configuration the
                              latch-fault ablation shows is necessary, because a
                              sound contract offered alongside gameable tools can
                              be routed around.

    For non-astro domains (drone, biochem, electrical, drug) the modes are no-ops:
    those domains have no verify_binary_target tool, and their validators are
    design-only (not answer-submitting), so they are not gameable in the first place.
    """
    import os
    if domain not in DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain!r}. "
            f"Available: {list(DOMAIN_REGISTRY.keys())}"
        )
    tools = list(DOMAIN_REGISTRY[domain]["tools"])
    # Names of the bare, answer-submitting astro validators that the contract-gated
    # tool supersedes. These are the only gameable validators in the suite.
    _BARE_ASTRO = {"check_chirp_mass", "check_gw_strain",
                   "check_merger_time", "check_isco_frequency"}

    if os.environ.get("PHACT_SCM_EXCLUSIVE") == "1":
        # Keep verify_binary_target; drop the bare answer-submitting astro tools.
        tools = [fn for fn in tools if fn.__name__ not in _BARE_ASTRO]
    elif os.environ.get("PHACT_DISABLE_SCM_GATE") == "1":
        # Drop the contract-gated tool; the proposer must use the bare validators.
        tools = [fn for fn in tools if fn.__name__ != "verify_binary_target"]

    return [_wrap_tool(fn) for fn in tools]


def list_domains() -> list[dict]:
    """Return metadata for all registered domains."""
    return [
        {
            "domain":      d["name"],
            "description": d["description"],
            "tools":       [f.__name__ for f in d["tools"]],
            "requires":    d["requires"],
        }
        for d in DOMAIN_REGISTRY.values()
    ]


# ── Register built-in domains ────────────────────────────────────────────────

from phact_physics.domains.nrt import (
    check_ncross_scaling,
    check_isotropy,
    check_encoding_ratio,
    check_local_record_equilibrium,
    check_relational_entropy,
)

register_domain(
    name        = "nrt",
    description = (
        "N-Record Theory (NRT) physics engine: tests Nakul Vyas's geometric theory "
        "of relational physics. Validates the three core NRT signatures - "
        "(1) N_cross(R) ∝ R² dimensional scaling, "
        "(2) N_out/N_cross → 1/2 isotropy parity, "
        "(3) η = N_bits/N_cross ≤ 1/(2m) encoding bound - "
        "plus Local Record Equilibrium (δQ = T_rec·δS) and the relational "
        "Bekenstein entropy bound (S ≤ k_B·N_cross·ln2)."
    ),
    tools = [
        check_ncross_scaling,
        check_isotropy,
        check_encoding_ratio,
        check_local_record_equilibrium,
        check_relational_entropy,
    ],
    requires = [],
)

from phact_physics.domains.drug_formulation import (
    check_colloidal_stability,
    check_polymer_drug_compatibility,
    check_drug_permeability,
    check_diffusion_coefficient,
)

register_domain(
    name        = "drug_formulation",
    description = (
        "Nanoparticle drug delivery formulation physics: DLVO colloidal stability, "
        "Flory-Huggins polymer-drug compatibility, Lipinski oral permeability, "
        "and Stokes-Einstein diffusion at physiological temperature."
    ),
    tools = [
        check_colloidal_stability,
        check_polymer_drug_compatibility,
        check_drug_permeability,
        check_diffusion_coefficient,
    ],
    requires = ["rdkit", "numpy", "scipy"],
)

from phact_physics.domains.electrical import (
    check_pcb_trace_thermal,
    check_power_budget,
    check_rc_filter,
    check_antenna_impedance,
)

register_domain(
    name        = "electrical",
    description = (
        "Electrical engineering physics: IPC-2221B PCB trace thermal analysis, "
        "power budget and junction temperature, RC filter -3dB cutoff (f=1/2πRC), "
        "and antenna impedance matching (VSWR, return loss, L-network design)."
    ),
    tools = [
        check_pcb_trace_thermal,
        check_power_budget,
        check_rc_filter,
        check_antenna_impedance,
    ],
    requires = [],
)

from phact_physics.physics_engine import DronePhysicsValidator, DNAHydrogelValidator
from phact_physics.agent import validate_drone_design, validate_biochem_design

register_domain(
    name        = "drone",
    description = (
        "Aerodynamic drone trajectory validation: drag force, roll authority, "
        "wind shear, and canyon clearance using Newton's laws."
    ),
    tools    = [validate_drone_design],
    requires = ["numpy"],
)

register_domain(
    name        = "biochem",
    description = (
        "DNA hydrogel thermodynamic stability: nearest-neighbour melting point "
        "(SantaLucia 1998), GC-content stability thresholds."
    ),
    tools    = [validate_biochem_design],
    requires = ["numpy"],
)

from phact_physics.domains.astrophysics import (
    check_chirp_mass,
    check_gw_strain,
    check_merger_time,
    check_isco_frequency,
    verify_binary_target,
)

register_domain(
    name        = "astrophysics",
    description = (
        "Gravitational wave astrophysics: validates compact binary inspiral parameters "
        "against exact general-relativistic post-Newtonian equations - the physics "
        "LIGO/Virgo uses to detect mergers. Checks chirp mass M_c, GW strain amplitude h, "
        "Peters (1964) merger timescale, and Schwarzschild ISCO peak frequency. "
        "Catches common AI errors: wrong chirp mass exponent, using total mass instead "
        "of M_c, coordinate vs luminosity distance, f_GW vs f_orbital confusion, "
        "and wrong ISCO radius (photon sphere vs 6GM/c²). For goals that fix a binary's "
        "inputs and ask to verify a target derived property, use verify_binary_target, "
        "which routes the check through the contract so the certified quantity is derived from "
        "the fixed inputs and cannot be supplied (and thus cannot be gamed) by the model."
    ),
    tools = [
        check_chirp_mass,
        check_gw_strain,
        check_merger_time,
        check_isco_frequency,
        verify_binary_target,
    ],
    requires = [],
)
