"""
Structural contract -- DNA oligonucleotide dual-constraint design
======================================================================
Two simultaneous thermodynamic constraints that must BOTH be satisfied:

    Tm   >= target_tm_c      (melting temperature, SantaLucia 1998)
    ΔG   <= target_dg_kcal   (free energy of hybridisation at 37°C, more negative = tighter)

Unlike a single-constraint Tm goal (which the model can solve by adding GC),
these constraints are COUPLED and CONFLICTING:

    Raising GC content   → Tm ↑  (good for Tm constraint)
                         → ΔG ↓  (more negative - good for ΔG constraint)
    Raising sequence length → Tm ↑  and  ΔG ↓ (both improve)
    Adding CG / GC stacks  → large ΔH contribution (both Tm and ΔG affected)

The feasible region is a SURFACE in (length, GC_fraction) space, not a curve.
Naive proposals that satisfy Tm may miss ΔG, and vice versa.
Weak feedback ("Tm is 58°C, target 65°C, and ΔG is -7.2, target < -9 kcal/mol")
gives TWO violation signals with no indication of how to jointly satisfy them.

The contract encodes this as a DAG:

    sequence_length   (exo) ──┐
    gc_fraction       (exo) ──┼──→ delta_H   (endo)
                              ├──→ delta_S   (endo)
    strand_conc_nm    (exo) ──┤
                              ├──→ melting_temp_c  (outcome 1)
                              └──→ delta_g_kcal    (outcome 2)

The minimal corrective override computes, via joint closed-form inversion, the (length,
gc_fraction) pair that satisfies BOTH constraints simultaneously with the
smallest sequence length.

Reference:
    SantaLucia J. (1998) PNAS 95:1460-1465.
    Owczarzy R. et al. (2004) Biochemistry 43:3537.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from phact_physics.contracts.engine import DependencyGraph, GraphNode, GraphEdge


# ── SantaLucia 1998 Table 2 (same as physics_engine.py) ─────────────────────
_NN: dict[str, tuple[float, float]] = {
    "AA": (-7.9,  -22.2),
    "AT": (-7.2,  -20.4),
    "TA": (-7.2,  -21.3),
    "CA": (-8.5,  -22.7),
    "GT": (-8.4,  -22.4),
    "CT": (-7.8,  -21.0),
    "GA": (-8.2,  -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8,  -24.4),
    "GG": (-8.0,  -19.9),
}
_INIT_AT = (2.3,   4.1)
_INIT_GC = (0.1,  -2.8)
_R       = 1.987e-3   # kcal/(mol·K)
_NA_M    = 0.05       # 50 mM Na+


def _synthetic_params(n: int, gc_frac: float) -> tuple[float, float]:
    """
    Estimate ΔH (kcal/mol) and ΔS (cal/mol·K) for a sequence of length n with
    GC fraction gc_frac.  Uses the per-dinucleotide average weighted by GC content:

        average NN ΔH for GC dinucleotides ≈ -9.4 kcal/mol  (mean of CG/GC/GG)
        average NN ΔH for AT dinucleotides ≈ -7.6 kcal/mol  (mean of AA/AT/TA)

    This is an approximation over the composition - the exact value depends on
    sequence order. The contract uses it to find the (n, gc) pair that is in the
    feasible region; the exact engine rejects or certifies once the model proposes
    a real sequence.
    """
    # Fraction of GC-GC, GC-AT, AT-AT steps scales with gc_frac
    dH_gc = -9.375   # mean of CG, GC, GG (kcal/mol) - high-energy GC stacks
    dH_at = -7.575   # mean of AA, AT, TA, CA, GT, CT, GA (kcal/mol)
    dS_gc = -23.83   # mean of CG, GC, GG (cal/mol·K)
    dS_at = -21.86   # mean of AT dinucleotides

    # (n-1) dinucleotide steps
    steps    = n - 1
    gc_steps = steps * gc_frac
    at_steps = steps * (1.0 - gc_frac)

    total_dH = gc_steps * dH_gc + at_steps * dH_at
    total_dS = gc_steps * dS_gc + at_steps * dS_at

    # Initiation
    n_gc_terms = round(gc_frac * n)
    n_at_terms = n - n_gc_terms
    if n_gc_terms > 0:
        total_dH += _INIT_GC[0]; total_dS += _INIT_GC[1]
    if n_at_terms > 0:
        total_dH += _INIT_AT[0]; total_dS += _INIT_AT[1]

    return total_dH, total_dS


def _calc_tm(dH: float, dS_cal: float, c_t_nm: float, gc_frac: float) -> float:
    """Tm in °C from ΔH/ΔS + Owczarzy salt correction."""
    dS = dS_cal / 1000.0
    denom = dS + _R * math.log(c_t_nm * 1e-9 / 4.0)
    if abs(denom) < 1e-12:
        return 0.0
    tm_1m = dH / denom
    ln_na = math.log(_NA_M)
    inv_tm = 1.0 / tm_1m + (4.29 * gc_frac - 3.95) * 1e-5 * ln_na + 9.40e-6 * ln_na ** 2
    return 1.0 / inv_tm - 273.15 if abs(inv_tm) > 1e-12 else 0.0


def _calc_dg_37(dH: float, dS_cal: float) -> float:
    """ΔG at 37°C in kcal/mol = ΔH - T·ΔS."""
    T = 310.15  # 37°C in K
    return dH - T * (dS_cal / 1000.0)


@dataclass
class DNACorrection:
    """Result of overriding (sequence_length=n*, gc_fraction=f*) in the contract."""
    target_length:    int
    target_gc_frac:   float
    predicted_tm_c:   float
    predicted_dg_kcal: float
    physics_feedback:  str
    correction_hint:  str
    feasible:         bool   # False if no (n,gc) pair can jointly satisfy both constraints


def find_dna_correction(
    target_tm_c:   float,
    target_dg_kcal: float,
    strand_conc_nm: float = 250.0,
    max_length:    int   = 30,
) -> DNACorrection:
    """
    Binary search: find the minimum-length (n, gc_fraction) pair that jointly
    satisfies:
        Tm   >= target_tm_c
        ΔG37 <= target_dg_kcal   (ΔG is negative; target is the upper bound)

    Searches over length 8..max_length and gc_frac 0.0..1.0 in 0.05 steps.
    Returns the smallest n first, then lowest gc_frac for that n.
    """
    best: Optional[tuple[int, float, float, float]] = None  # (n, gc, tm, dg)

    for n in range(8, max_length + 1):
        for gc_i in range(0, 21):       # 0.00, 0.05, ..., 1.00
            gc = gc_i / 20.0
            dH, dS = _synthetic_params(n, gc)
            tm  = _calc_tm(dH, dS, strand_conc_nm, gc)
            dg  = _calc_dg_37(dH, dS)
            if tm >= target_tm_c and dg <= target_dg_kcal:
                best = (n, gc, tm, dg)
                break          # minimum gc for this n
        if best is not None:
            break              # minimum n overall

    if best is None:
        return DNACorrection(
            target_length    = max_length,
            target_gc_frac   = 1.0,
            predicted_tm_c   = 0.0,
            predicted_dg_kcal = 0.0,
            physics_feedback  = (
                "CONTRACT ANALYSIS: no sequence of length ≤ {} can simultaneously "
                "satisfy Tm ≥ {:.0f}°C and ΔG ≤ {:.1f} kcal/mol under the "
                "SantaLucia 1998 model. The joint constraint set is infeasible.".format(
                    max_length, target_tm_c, target_dg_kcal)
            ),
            correction_hint  = (
                "The two constraints cannot be jointly satisfied. "
                "Relax either the Tm lower bound or the ΔG upper bound."
            ),
            feasible = False,
        )

    n, gc, tm, dg = best
    n_gc = round(gc * n)
    physics_feedback = (
        "STRUCTURAL CONTRACT ANALYSIS (SantaLucia 1998):\n"
        "Dependency DAG: sequence_length → ΔH → Tm   AND   sequence_length → ΔH → ΔG₃₇\n"
        "            gc_fraction    → ΔH → Tm   AND   gc_fraction    → ΔH → ΔG₃₇\n"
        "Joint minimal corrective override to satisfy BOTH constraints simultaneously:\n"
        f"  sequence_length \u2190 {n}\n"
        f"  gc_fraction     \u2190 {gc:.2f}  ({n_gc}/{n} GC bases)\n"
        f"Predicted outcome of the correction:\n"
        f"  Tm   = {tm:.1f}°C  (target ≥ {target_tm_c:.0f}°C)  ✓\n"
        f"  ΔG₃₇ = {dg:.2f} kcal/mol  (target ≤ {target_dg_kcal:.1f} kcal/mol)  ✓"
    )
    correction_hint = (
        f"Design a {n}-mer with {n_gc}/{n} GC bases (GC fraction = {gc:.2f}). "
        f"Place GC pairs in the interior for maximum stacking energy. "
        f"Predicted: Tm = {tm:.1f}°C, ΔG₃₇ = {dg:.2f} kcal/mol."
    )
    return DNACorrection(
        target_length    = n,
        target_gc_frac   = gc,
        predicted_tm_c   = tm,
        predicted_dg_kcal = dg,
        physics_feedback  = physics_feedback,
        correction_hint  = correction_hint,
        feasible         = True,
    )


def build_dna_dual_contract() -> DependencyGraph:
    """Build the contract for DNA dual-constraint (Tm + ΔG) design."""
    g = DependencyGraph(
        domain      = "biochem",
        description = (
            "Structural contract for oligonucleotide design satisfying two "
            "simultaneous thermodynamic constraints: melting temperature Tm and "
            "hybridisation free energy ΔG at 37°C. Both outcomes share the same "
            "parents (length, GC fraction) via the SantaLucia 1998 "
            "nearest-neighbour model, creating a coupled constraint surface."
        ),
    )

    g.add_node(GraphNode(
        name="sequence_length", node_type="exogenous", unit="nt",
        description="Number of bases in the oligonucleotide.",
        bounds=(8.0, 50.0),
    ))
    g.add_node(GraphNode(
        name="gc_fraction", node_type="exogenous", unit="fraction",
        description="Fraction of G/C bases (0 = all AT, 1 = all GC).",
        bounds=(0.0, 1.0),
    ))
    g.add_node(GraphNode(
        name="strand_conc_nm", node_type="exogenous", unit="nM",
        description="Total strand concentration in nanomolar.",
        bounds=(1.0, 10000.0),
    ))

    def compute_dH(p: dict) -> float:
        dH, _ = _synthetic_params(int(round(p["sequence_length"])), p["gc_fraction"])
        return dH

    def compute_dS(p: dict) -> float:
        _, dS = _synthetic_params(int(round(p["sequence_length"])), p["gc_fraction"])
        return dS

    g.add_node(GraphNode(
        name="delta_H_kcal", node_type="endogenous", unit="kcal/mol",
        description="Total hybridisation enthalpy (SantaLucia 1998 NN sum).",
        bounds=(-200.0, 0.0),
        structural_eq=compute_dH,
        parent_names=["sequence_length", "gc_fraction"],
    ))
    g.add_node(GraphNode(
        name="delta_S_cal", node_type="endogenous", unit="cal/mol·K",
        description="Total hybridisation entropy.",
        bounds=(-600.0, 0.0),
        structural_eq=compute_dS,
        parent_names=["sequence_length", "gc_fraction"],
    ))

    def compute_tm(p: dict) -> float:
        return _calc_tm(p["delta_H_kcal"], p["delta_S_cal"],
                        p["strand_conc_nm"], p["gc_fraction"])

    def compute_dg(p: dict) -> float:
        return _calc_dg_37(p["delta_H_kcal"], p["delta_S_cal"])

    g.add_node(GraphNode(
        name="melting_temp_c", node_type="outcome", unit="°C",
        description="Melting temperature Tm (SantaLucia 1998 + Owczarzy 2004 salt).",
        bounds=(0.0, 100.0),
        structural_eq=compute_tm,
        parent_names=["delta_H_kcal", "delta_S_cal", "strand_conc_nm", "gc_fraction"],
    ))
    g.add_node(GraphNode(
        name="delta_g_kcal", node_type="outcome", unit="kcal/mol",
        description="Hybridisation free energy at 37°C: ΔG = ΔH - T·ΔS.",
        bounds=(-100.0, 10.0),
        structural_eq=compute_dg,
        parent_names=["delta_H_kcal", "delta_S_cal"],
    ))

    for parent, child, direction, mech in [
        ("sequence_length", "delta_H_kcal", -1,
         "longer oligo → more NN stacking steps → more negative ΔH"),
        ("gc_fraction",     "delta_H_kcal", -1,
         "GC stacks (avg -9.4 kcal/mol) stronger than AT (avg -7.6 kcal/mol)"),
        ("sequence_length", "delta_S_cal",  -1,
         "longer oligo → more entropy loss on hybridisation"),
        ("gc_fraction",     "delta_S_cal",  -1,
         "GC stacks have larger ΔS magnitude than AT stacks"),
        ("delta_H_kcal",    "melting_temp_c", +1,
         "Tm = ΔH / (ΔS + R ln(C_T/4)) - more negative ΔH raises Tm"),
        ("delta_S_cal",     "melting_temp_c", -1,
         "more negative ΔS lowers denominator, raises Tm (non-linearly)"),
        ("delta_H_kcal",    "delta_g_kcal",   +1,
         "ΔG = ΔH - TΔS; more negative ΔH makes ΔG more negative"),
        ("delta_S_cal",     "delta_g_kcal",   -1,
         "-TΔS term: more negative ΔS (×-T) makes ΔG more positive at 37°C"),
        ("strand_conc_nm",  "melting_temp_c", +1,
         "higher strand concentration raises Tm via R ln(C_T/4) term"),
    ]:
        g.add_edge(GraphEdge(
            parent=parent, child=child,
            effect_direction=direction, mechanism=mech,
        ))

    return g
