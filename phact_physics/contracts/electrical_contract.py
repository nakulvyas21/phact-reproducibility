"""
Structural contract - Electrical RC filter
================================================
A minimal, exact contract for a first-order RC filter.

The feasible region of an RC filter is a tight one-dimensional curve: to hit a
target -3 dB cutoff f_c, the resistance R and capacitance C must satisfy the
exact product relation

    f_c = 1 / (2 pi R C).

Unlike colloidal stability (a large feasible volume), there is essentially no
slack here: a naive round-number guess for R and C usually misses the target by
tens to hundreds of percent. This makes RC filter design a domain with real
"correction depth", and therefore a fair test of whether an exact minimal corrective override
beats a free-text correction hint.

Graph
-----
    resistance_ohm   (exogenous, do-able)  --.
                                              >--> cutoff_freq_hz (outcome)
    capacitance_uf   (exogenous, do-able)  --'

Structural equation (exact, first-principles):
    cutoff_freq_hz = 1 / (2 pi * R * C)

Reference:
    Horowitz & Hill, The Art of Electronics, 3rd ed. (2015), Ch. 1-2.
"""

from __future__ import annotations

import math

from phact_physics.contracts.engine import DependencyGraph, GraphNode, GraphEdge


def build_electrical_rc_contract() -> DependencyGraph:
    """
    Build and return the contract for first-order RC filter cutoff design.

    The single structural equation is exact: f_c = 1/(2 pi R C). The graph is
    used to return a minimal corrective override on resistance (the conventional free
    variable once a capacitor value is chosen) that lands the cutoff exactly on
    target.
    """
    g = DependencyGraph(
        domain      = "electrical",
        description = (
            "Structural contract for a first-order RC filter. Encodes the "
            "exact cutoff relation f_c = 1/(2 pi R C) as a DAG: resistance and "
            "capacitance jointly determine the -3 dB cutoff frequency."
        ),
    )

    # ── Exogenous (designer-controlled) nodes ────────────────────────────────
    g.add_node(GraphNode(
        name        = "resistance_ohm",
        node_type   = "exogenous",
        unit        = "ohm",
        description = (
            "Series resistance of the RC filter. Designer-controlled. "
            "Together with C it sets the cutoff: f_c = 1/(2 pi R C)."
        ),
        bounds      = (1.0, 1.0e9),
    ))

    g.add_node(GraphNode(
        name        = "capacitance_uf",
        node_type   = "exogenous",
        unit        = "uF",
        description = (
            "Filter capacitance in microfarads. Designer-controlled. "
            "Together with R it sets the cutoff: f_c = 1/(2 pi R C)."
        ),
        bounds      = (1.0e-6, 1.0e3),
    ))

    # ── Outcome node: cutoff frequency ───────────────────────────────────────
    def compute_cutoff_hz(p: dict) -> float:
        """f_c = 1 / (2 pi R C), with C converted uF -> F."""
        R = p["resistance_ohm"]
        C = p["capacitance_uf"] * 1e-6
        if R <= 0 or C <= 0:
            return 0.0
        return 1.0 / (2.0 * math.pi * R * C)

    g.add_node(GraphNode(
        name          = "cutoff_freq_hz",
        node_type     = "outcome",
        unit          = "Hz",
        description   = "The -3 dB cutoff frequency of the RC filter.",
        bounds        = (0.0, 1.0e12),
        structural_eq = compute_cutoff_hz,
        parent_names  = ["resistance_ohm", "capacitance_uf"],
    ))

    # ── Edges (with physical mechanism) ──────────────────────────────────────
    g.add_edge(GraphEdge(
        parent           = "resistance_ohm",
        child            = "cutoff_freq_hz",
        effect_direction = -1,   # higher R -> lower cutoff
        mechanism        = "f_c = 1/(2 pi R C): cutoff is inversely proportional to R",
    ))
    g.add_edge(GraphEdge(
        parent           = "capacitance_uf",
        child            = "cutoff_freq_hz",
        effect_direction = -1,   # higher C -> lower cutoff
        mechanism        = "f_c = 1/(2 pi R C): cutoff is inversely proportional to C",
    ))

    return g


def exact_resistance_for_cutoff(target_hz: float, capacitance_uf: float) -> float:
    """
    Closed-form inverse of the cutoff relation: given a target cutoff and a
    fixed capacitor, the resistance that lands the cutoff exactly on target.

        R = 1 / (2 pi f_target C)

    This is the minimal corrective override the contract recommends: override resistance_ohm = R*.
    """
    C = capacitance_uf * 1e-6
    return 1.0 / (2.0 * math.pi * target_hz * C)
