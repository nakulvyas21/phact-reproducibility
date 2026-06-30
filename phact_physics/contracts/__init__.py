"""
PHACT Structural Contracts
==========================
Directed dependency graphs for deterministic forward evaluation over
physics design spaces.

A structural contract is a directed acyclic graph of known physics equations:
designer-controlled inputs are root nodes, and the certified quantity is a
derived leaf computed from them. This is ordinary forward evaluation of those
equations -- not causal inference or do-calculus. The contract's value is
structural: because the certified quantity is derived from the inputs rather
than supplied by the proposer, it cannot be forged.

Each domain graph encodes:
  - Nodes:     physical variables (input parameters + derived outcomes)
  - Edges:     dependency links derived from governing equations
  - Functions: structural equations f_i(parents) mapping inputs to outputs
"""

from phact_physics.contracts.engine import DependencyGraph, Override, ComparisonQuery
from phact_physics.contracts.drug_formulation_contract import build_drug_formulation_contract
from phact_physics.contracts.electrical_contract import (
    build_electrical_rc_contract,
    exact_resistance_for_cutoff,
)
from phact_physics.contracts.dna_contract import (
    build_dna_dual_contract,
    find_dna_correction,
)
from phact_physics.contracts.astrophysics_contract import (
    build_chirp_mass_contract,
    build_isco_contract,
    build_strain_contract,
    build_merger_time_contract,
    ASTROPHYSICS_CONTRACTS,
)

__all__ = [
    "DependencyGraph",
    "Override",
    "ComparisonQuery",
    "build_drug_formulation_contract",
    "build_electrical_rc_contract",
    "exact_resistance_for_cutoff",
    "build_dna_dual_contract",
    "find_dna_correction",
    "build_chirp_mass_contract",
    "build_isco_contract",
    "build_strain_contract",
    "build_merger_time_contract",
    "ASTROPHYSICS_CONTRACTS",
]
