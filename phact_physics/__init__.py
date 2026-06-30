"""
phact_physics - Deterministic physics engines for Physics-Anchored Certification (PHACT)
========================================================================================

This package contains ONLY the deterministic physics validators and the
structural contracts used as the ground-truth oracle in the PHACT
paper. There is no machine learning here, no language model, and no network
access. Every function implements a published, peer-reviewed physical equation
and returns an exact result for a given input.

These engines are the scientific core of the paper. The contract layer
(``phact_physics.contracts``) is the paper's central contribution: it derives the
certified quantity from the goal's fixed inputs, so the proposer cannot supply
the value being certified. ``verify_binary_target`` is the astrophysics
structural check used in the forgeability experiments.

Domains
-------
    drone             Aerodynamic drag, roll authority, wind shear, clearance
    biochem           DNA melting temperature (SantaLucia 1998 + Owczarzy 2004)
    astrophysics      Chirp mass, GW strain, Peters merger time, ISCO frequency
    electrical        RC filter cutoff, PCB thermal, power budget, antenna match
    drug_formulation  DLVO colloidal stability, Flory-Huggins, permeability

Quick start
-----------
    from phact_physics import get_validator

    v = get_validator("biochem")
    result = v.validate({
        "sequence": "ATCGATCGATCGATCGATCG",
        "target_temp_c": 37.0,
        "strand_conc_nm": 250.0,
    })
    print(result.passed, result.metrics["calculated_tm_C"])

    # The structural check: you supply the FIXED inputs and the target only;
    # the contract derives the true value, which you cannot override.
    from phact_physics.domains.astrophysics import verify_binary_target
    r = verify_binary_target(quantity="chirp_mass", m1_msun=36, m2_msun=29,
                             target_value=50.0)
    print(r.passed)  # False: true chirp mass is 28.1 M_sun, not 50
"""

from phact_physics.physics_engine import (
    PhysicsResult,
    DronePhysicsValidator,
    DNAHydrogelValidator,
    get_validator,
)
from phact_physics.domains.astrophysics import verify_binary_target

__all__ = [
    "PhysicsResult",
    "DronePhysicsValidator",
    "DNAHydrogelValidator",
    "get_validator",
    "verify_binary_target",
]

__version__ = "2.0.0"
