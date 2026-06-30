#!/usr/bin/env python3
"""
PHACT certification demo
========================
A 30-second, no-API-key tour of the certification surface a language model
actually sees and is bound by. Run:

    python3 demo.py

This is the deterministic core of the paper made tangible. It does three things:

  1. Prints the exact tool specifications (names, typed parameters, docstrings)
     that the proposing model is given. In the live experiment the model emits
     a design by calling one of these tools; here we call them directly so the
     contract is visible without a model or a network.

  2. Runs a proposed design through the physics engine and shows the verdict
     plus the physics feedback the model would receive on a failure.

  3. Demonstrates the structural contract's anti-forgery property: the certified
     quantity is derived from the fixed inputs, so a proposer cannot assert a
     value it did not earn.

Everything here is the same code path exercised by `make verify`.
"""

import inspect

from phact_physics.agent import validate_drone_design, validate_biochem_design
from phact_physics.domains.astrophysics import verify_binary_target


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def show_tool_spec(fn) -> None:
    sig = inspect.signature(fn)
    params = ", ".join(
        p.name if p.annotation is inspect._empty else f"{p.name}: {p.annotation.__name__}"
        for p in sig.parameters.values()
    )
    print(f"\nTool: {fn.__name__}({params})")
    print(inspect.getdoc(fn))


def main() -> None:
    rule("1. The tool specifications the proposing model is given")
    print(
        "In the live experiment the model proposes a design by calling one of\n"
        "these tools. The typed signature and docstring below are exactly what\n"
        "the model sees - no hidden prompt does the physics for it."
    )
    show_tool_spec(validate_drone_design)
    show_tool_spec(validate_biochem_design)

    rule("2. A proposed design, certified by the physics engine")
    drone = validate_drone_design(
        path_type="banked_curve", crosswind_knots=15.0, canyon_width_m=10.0,
        speed_ms=12.0, bank_angle_deg=18.0,
        reasoning="counter-steer into the crosswind to hold the line",
    )
    print(f"\nProposal: banked-curve, 15 kt crosswind, 10 m gap, 12 m/s, 18 deg bank")
    print(f"  certified (physics_passed): {drone['passed']}")
    print(f"  metrics: {drone['metrics']}")

    # A failing DNA proposal, to show the feedback the model would act on.
    dna = validate_biochem_design(
        sequence="ATATATATATAT", target_temp_c=65.0, strand_conc_nm=250.0,
        reasoning="AT-rich sequence",
    )
    print(f"\nProposal: sequence ATATATATATAT, target Tm 65 C")
    print(f"  certified (physics_passed): {dna['passed']}")
    print(f"  calculated Tm (SantaLucia 1998): {dna['metrics'].get('calculated_tm_C')} C")
    print(f"  physics feedback returned to the model:")
    print(f"    {dna['physics_feedback'].strip().splitlines()[0]}")

    rule("3. The structural contract cannot be forged")
    print(
        "The proposer supplies only the fixed inputs (the component masses).\n"
        "The contract derives the certified quantity; the proposer cannot assert it."
    )
    forged = verify_binary_target(quantity="chirp_mass", m1_msun=36, m2_msun=29,
                                  target_value=50.0)
    honest = verify_binary_target(quantity="chirp_mass", m1_msun=36, m2_msun=29,
                                  target_value=28.1)
    print(f"\n  claim: chirp mass of (36, 29) M_sun is 50.0  -> certified: {forged.passed}")
    print(f"  claim: chirp mass of (36, 29) M_sun is 28.1  -> certified: {honest.passed}")
    print("\n  The true value is derived from the inputs, so only the true claim passes.")

    print("\nThis is the soundness mechanism of the paper, runnable with no API key.\n")


if __name__ == "__main__":
    main()
