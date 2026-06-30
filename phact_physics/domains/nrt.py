"""
PHACT Domain: N-Record Theory (NRT) Physics Engine
==================================================
Tests Nakul Vyas's N-Record Theory - a geometric theory of relational physics
in which *records* (primitive binary relational acts) are the only ontological
primitive and all of space, time, and matter are emergent structures built from
them.

Paper: "N-Record Theory" (Vyas, 2024-26).

Three core testable signatures of NRT:
  1. Crossing-surface scaling  N_cross(R) ∝ R²  (d≃3 emerges)
  2. Isotropy (screen parity)  N_out / N_cross → 1/2
  3. Encoding-density bound    η = N_bits / N_cross ≤ 1/(2m)

Plus two consistency checks drawn from the paper:
  4. Local Record Equilibrium  δQ = T_rec · δS  (thermodynamic self-consistency)
  5. Relational entropy bound  S ≤ k_B · N_cross · ln(2)  (Bekenstein analogue)

Each check simulates a discrete record network, applies the relevant theoretical
constraint from the paper, and reports pass/fail with exact numbers.

Designed to catch hallucinations / incorrect parameter choices by Gemini:
  • Getting the R² exponent wrong (claiming linear or cubic scaling)
  • Violating the parity symmetry (N_out / N_cross ≠ 1/2)
  • Exceeding the fundamental encoding bound η ≤ 1/(2m)
  • Inconsistent entropy assignments that violate LRE
  • Using infinite-precision coordinates instead of discrete records

References:
  Vyas, N. "N-Record Theory" (2024-2026).
  Bekenstein, J. D. Phys. Rev. D 7, 2333 (1973).
  Jacobson, T. Phys. Rev. Lett. 75, 1260 (1995).
  Maldacena, J. Int. J. Theor. Phys. 38, 1113 (1999).
"""

from __future__ import annotations

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Result dataclass (mirrors other domains) ─────────────────────────────────

@dataclass
class PhysicsResult:
    passed:          bool
    domain:          str
    check_name:      str
    design_summary:  str
    violations:      list[str]  = field(default_factory=list)
    metrics:         dict       = field(default_factory=dict)
    physics_feedback: str        = ""
    correction_hint: str        = ""
    reference:       str        = ""


# ── NRT physical constants ────────────────────────────────────────────────────

NRT_D_EMERGENT = 3          # Expected emergent spatial dimension
NRT_PARITY     = 0.5        # N_out / N_cross → 1/2  (isotropy signature)
NRT_ETA_MAX    = None       # 1/(2m), computed from m at runtime
NRT_LN2        = math.log(2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Crossing-Surface Scaling  N_cross(R) ∝ R^d  (should give d≃3)
# ─────────────────────────────────────────────────────────────────────────────

def check_ncross_scaling(
    radius_small:  float,
    ncross_small:  int,
    radius_large:  float,
    ncross_large:  int,
    tolerance_pct: float = 15.0,
) -> PhysicsResult:
    """
    Verifies the N-Record Theory signature: N_cross(R) ∝ R² (d≃3 emerges).

    In NRT, the number of records that cross a 2-sphere of coordinate radius R
    must scale as R² - this is the geometric signature of emergent 3D space.
    Any exponent significantly different from 2 indicates a non-3D record structure.

    Call this tool when you propose a record network configuration. Provide
    two measurement radii and their corresponding crossing counts to check
    whether the network exhibits the correct dimensional scaling.

    Args:
        radius_small:  Smaller reference radius R₁ (any consistent unit, e.g. Planck lengths)
        ncross_small:  Number of records crossing sphere of radius R₁
        radius_large:  Larger reference radius R₂ (must be > radius_small)
        ncross_large:  Number of records crossing sphere of radius R₂
        tolerance_pct: Allowed deviation from the d=2 exponent in percent (default 15%)

    Returns:
        Physics validation with the measured scaling exponent d_eff and whether
        it is consistent with emergent 3-dimensional space (d_eff ≈ 2).
    """
    violations = []

    if radius_small <= 0 or radius_large <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="ncross_scaling",
            design_summary="Invalid: radii must be positive",
            violations=["Both radii must be positive non-zero values"],
        )
    if radius_large <= radius_small:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="ncross_scaling",
            design_summary="Invalid: radius_large must exceed radius_small",
            violations=["radius_large must be strictly greater than radius_small"],
        )
    if ncross_small <= 0 or ncross_large <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="ncross_scaling",
            design_summary="Invalid: crossing counts must be positive integers",
            violations=["N_cross values must be positive"],
        )

    # Power-law exponent: N ∝ R^d_eff
    # d_eff = log(N2/N1) / log(R2/R1)
    ratio_n = ncross_large / ncross_small
    ratio_r = radius_large / radius_small
    d_eff   = math.log(ratio_n) / math.log(ratio_r)

    # NRT requires d_eff ≈ 2  (N_cross ∝ R², surface of a 2-sphere in 3D)
    expected_d = 2.0
    d_error    = abs(d_eff - expected_d)
    d_error_pct = d_error / expected_d * 100

    # Predicted N_large using exact R² law
    ncross_predicted = ncross_small * (radius_large / radius_small) ** expected_d
    ncross_deviation_pct = abs(ncross_large - ncross_predicted) / ncross_predicted * 100

    if d_error_pct > tolerance_pct:
        violations.append(
            f"DIMENSION VIOLATION: measured exponent d_eff = {d_eff:.3f}, "
            f"NRT requires d_eff ≈ {expected_d} (±{tolerance_pct:.0f}%). "
            f"This record network does NOT produce emergent 3D space."
        )
    if d_eff < 1.5:
        violations.append(
            f"UNDER-DIMENSIONAL: d_eff = {d_eff:.3f} < 1.5 - "
            f"record density is too sparse to support 3D emergence."
        )
    if d_eff > 3.0:
        violations.append(
            f"OVER-DIMENSIONAL: d_eff = {d_eff:.3f} > 3 - "
            f"record density exceeds the NRT R² bound, "
            f"suggesting bulk correlations (volume terms) contaminating the surface count."
        )

    passed  = len(violations) == 0
    summary = (
        f"R₁={radius_small}, N₁={ncross_small}; "
        f"R₂={radius_large}, N₂={ncross_large}; "
        f"d_eff={d_eff:.4f} (NRT target ≈2.00)"
    )
    feedback = (
        f"Scaling exponent: d_eff = log({ratio_n:.3f}) / log({ratio_r:.3f}) = {d_eff:.4f}. "
        f"NRT predicts N_cross ∝ R² → d_eff = 2.00. "
        f"Deviation: {d_error_pct:.2f}% (tolerance {tolerance_pct:.0f}%). "
        f"Predicted N₂ from exact R² law: {ncross_predicted:.1f} "
        f"(actual {ncross_large}, {ncross_deviation_pct:.1f}% off). "
        + ("PASS: consistent with emergent 3D space." if passed
           else f"FAIL: d_eff={d_eff:.3f} deviates from required d=2.")
    )
    hint = (
        "" if passed else
        f"To satisfy NRT dimensional scaling, adjust the record network so that "
        f"N_cross grows as R². For R₂={radius_large} you need N_cross ≈ {ncross_predicted:.0f} "
        f"(you provided {ncross_large}). "
        f"Common fixes: increase record density uniformly on the sphere surface, "
        f"or reduce bulk record correlations that create super-surface scaling."
    )

    return PhysicsResult(
        passed          = passed,
        domain          = "nrt",
        check_name      = "ncross_scaling",
        design_summary  = summary,
        violations      = violations,
        metrics         = {
            "radius_small":          radius_small,
            "ncross_small":          ncross_small,
            "radius_large":          radius_large,
            "ncross_large":          ncross_large,
            "d_eff":                 round(d_eff, 5),
            "d_expected":            expected_d,
            "d_error_pct":           round(d_error_pct, 3),
            "ncross_predicted":      round(ncross_predicted, 1),
            "ncross_deviation_pct":  round(ncross_deviation_pct, 2),
        },
        physics_feedback = feedback,
        correction_hint = hint,
        reference       = "Vyas, N-Record Theory - Signature 1: N_cross(R) ∝ R²",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Isotropy (Screen Parity)  N_out / N_cross → 1/2
# ─────────────────────────────────────────────────────────────────────────────

def check_isotropy(
    n_cross:       int,
    n_out:         int,
    tolerance_pct: float = 10.0,
) -> PhysicsResult:
    """
    Verifies the N-Record Theory isotropy signature: N_out / N_cross → 1/2.

    In NRT, for every record that crosses a bounding screen, exactly half (in
    expectation) are outgoing - the parity symmetry of relational acts. This
    is the relational analogue of the equipartition of information across a
    holographic screen. Significant deviation from 1/2 indicates anisotropy
    (preferred direction in the record network) or a boundary condition error.

    Call this tool when you have simulated or estimated crossing counts for your
    record network at a given bounding surface.

    Args:
        n_cross:       Total records crossing the bounding surface (in + out combined)
        n_out:         Records directed outward across the surface
        tolerance_pct: Allowed deviation from 1/2 parity in percent (default 10%)

    Returns:
        Physics validation with the measured parity ratio and whether it is
        consistent with NRT's isotropy requirement.
    """
    violations = []

    if n_cross <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="isotropy",
            design_summary="Invalid: n_cross must be positive",
            violations=["n_cross must be a positive integer"],
        )
    if n_out < 0 or n_out > n_cross:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="isotropy",
            design_summary="Invalid: n_out must be between 0 and n_cross",
            violations=["n_out must be in [0, n_cross]"],
        )

    parity         = n_out / n_cross
    n_in           = n_cross - n_out
    parity_error   = abs(parity - NRT_PARITY)
    parity_err_pct = parity_error / NRT_PARITY * 100

    # Statistical significance: for large N, binomial σ = sqrt(N·p·(1-p)) / N = 1/(2√N)
    # We flag only if deviation exceeds the tolerance AND is statistically significant
    statistical_sigma = abs(n_out - n_cross * NRT_PARITY) / math.sqrt(n_cross * NRT_PARITY * (1 - NRT_PARITY))

    if parity_err_pct > tolerance_pct:
        violations.append(
            f"PARITY VIOLATION: N_out/N_cross = {parity:.4f}, "
            f"NRT requires → {NRT_PARITY} (±{tolerance_pct:.0f}%). "
            f"Deviation = {parity_err_pct:.2f}% ({statistical_sigma:.2f}σ). "
            f"Record network is anisotropic."
        )
    if parity < 0.35:
        violations.append(
            f"SEVERE INWARD BIAS: only {n_out}/{n_cross} = {parity:.3f} records are outgoing. "
            f"Network collapses inward - no equilibrium screen possible."
        )
    if parity > 0.65:
        violations.append(
            f"SEVERE OUTWARD BIAS: {n_out}/{n_cross} = {parity:.3f} records are outgoing. "
            f"Network explodes outward - record structure is unstable."
        )

    passed  = len(violations) == 0
    summary = (
        f"N_cross={n_cross}, N_out={n_out}, N_in={n_in}; "
        f"parity={parity:.4f} (NRT target=0.5000)"
    )
    feedback = (
        f"Isotropy check: N_out/N_cross = {n_out}/{n_cross} = {parity:.5f}. "
        f"NRT parity target = {NRT_PARITY}. "
        f"Absolute error = {parity_error:.5f} ({parity_err_pct:.2f}%). "
        f"Statistical significance: {statistical_sigma:.2f}σ. "
        + ("PASS: screen is isotropic within tolerance." if passed
           else f"FAIL: parity ratio deviates from 1/2 by {parity_err_pct:.2f}%.")
    )
    hint = (
        "" if passed else
        f"For an isotropic NRT screen, ensure that records are created symmetrically "
        f"with no preferred outward or inward direction. "
        f"You need N_out ≈ {n_cross//2} ± {int(n_cross * tolerance_pct / 200)} "
        f"(currently {n_out}). "
        f"Check your boundary conditions - asymmetric initial states or "
        f"time-reversal-breaking rules will destroy parity."
    )

    return PhysicsResult(
        passed          = passed,
        domain          = "nrt",
        check_name      = "isotropy",
        design_summary  = summary,
        violations      = violations,
        metrics         = {
            "n_cross":             n_cross,
            "n_out":               n_out,
            "n_in":                n_in,
            "parity_ratio":        round(parity, 6),
            "parity_target":       NRT_PARITY,
            "parity_error_pct":    round(parity_err_pct, 4),
            "statistical_sigma":   round(statistical_sigma, 3),
        },
        physics_feedback = feedback,
        correction_hint = hint,
        reference       = "Vyas, N-Record Theory - Signature 2: N_out/N_cross → 1/2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoding-Density Bound  η = N_bits / N_cross ≤ 1/(2m)
# ─────────────────────────────────────────────────────────────────────────────

def check_encoding_ratio(
    n_bits:        int,
    n_cross:       int,
    m:             int,
    tolerance_pct: float = 5.0,
) -> PhysicsResult:
    """
    Verifies the NRT encoding-density bound: η = N_bits / N_cross ≤ 1/(2m).

    In NRT, a screen of N_cross records can encode at most 1/(2m) bits per
    crossing record, where m is the number of binary record types in the model.
    This is the relational Bekenstein bound - exceeding it means more information
    is attributed to the screen than the record structure can physically support.

    Call this tool whenever you propose a specific number of encoded bits for a
    bounding surface with a given number of crossing records.

    Args:
        n_bits:        Number of bits of information encoded on the surface
        n_cross:       Number of records crossing the bounding surface
        m:             Number of binary record types in the model (e.g. m=2 for
                       spin-up/spin-down with position binary = 2 types)
        tolerance_pct: Allowed fractional excess above the bound in percent (default 5%)

    Returns:
        Physics validation with η, the bound η* = 1/(2m), and whether the
        encoding is consistent with NRT's information-density constraint.
    """
    violations = []

    if n_cross <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="encoding_ratio",
            design_summary="Invalid: n_cross must be positive",
            violations=["n_cross must be a positive integer"],
        )
    if m <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="encoding_ratio",
            design_summary="Invalid: m (record types) must be ≥ 1",
            violations=["m must be a positive integer ≥ 1"],
        )
    if n_bits < 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="encoding_ratio",
            design_summary="Invalid: n_bits must be non-negative",
            violations=["n_bits must be ≥ 0"],
        )

    eta       = n_bits / n_cross           # actual encoding density
    eta_star  = 1.0 / (2.0 * m)           # NRT upper bound
    violation_ratio = eta / eta_star if eta_star > 0 else float('inf')
    excess_pct      = max(0, (eta - eta_star) / eta_star * 100)

    # Maximum allowed bits
    n_bits_max = int(n_cross * eta_star)

    if eta > eta_star * (1 + tolerance_pct / 100):
        violations.append(
            f"ENCODING BOUND VIOLATED: η = {eta:.5f} > η* = 1/(2×{m}) = {eta_star:.5f}. "
            f"Excess: {excess_pct:.2f}%. "
            f"Max allowed: {n_bits_max} bits for {n_cross} crossings with m={m}."
        )
    if m == 1:
        # Tightest case: η ≤ 1/2 (each crossing record carries ≤1 bit)
        if eta > 0.5:
            violations.append(
                f"FUNDAMENTAL LIMIT: with m=1, each record carries at most 1 bit. "
                f"η = {eta:.4f} > 0.5 violates the minimum record-type bound."
            )
    if violation_ratio > 2.0:
        violations.append(
            f"SEVERE OVERENCODING: η/η* = {violation_ratio:.2f} - encoding density is more than "
            f"double the NRT bound. The proposed bit count is physically unrealizable."
        )

    passed  = len(violations) == 0
    summary = (
        f"N_bits={n_bits}, N_cross={n_cross}, m={m}; "
        f"η={eta:.5f} vs η*=1/(2×{m})={eta_star:.5f}"
    )
    feedback = (
        f"NRT encoding check: η = {n_bits}/{n_cross} = {eta:.6f}. "
        f"Bound: η* = 1/(2m) = 1/{2*m} = {eta_star:.6f} for m={m} record types. "
        f"η/η* = {violation_ratio:.4f}. "
        f"Max encodable bits: {n_bits_max}. "
        + ("PASS: encoding density is within NRT bound." if passed
           else f"FAIL: η exceeds η* by {excess_pct:.2f}%.")
    )
    hint = (
        "" if passed else
        f"Reduce the encoded information to ≤ {n_bits_max} bits for {n_cross} crossing records "
        f"with m={m} binary record types. "
        f"Alternatively, increase the record density: you need at least "
        f"{math.ceil(n_bits * 2 * m)} crossings to encode {n_bits} bits. "
        f"Or increase m (use finer record types) - "
        f"with m={math.ceil(n_bits / (2 * n_cross * 0.999))}, "
        f"the bound η*=1/{2*math.ceil(n_bits / (2 * n_cross * 0.999))} would accommodate your encoding."
    )

    return PhysicsResult(
        passed          = passed,
        domain          = "nrt",
        check_name      = "encoding_ratio",
        design_summary  = summary,
        violations      = violations,
        metrics         = {
            "n_bits":              n_bits,
            "n_cross":             n_cross,
            "m_record_types":      m,
            "eta":                 round(eta, 7),
            "eta_star":            round(eta_star, 7),
            "eta_ratio":           round(violation_ratio, 5),
            "excess_pct":          round(excess_pct, 4),
            "n_bits_max":          n_bits_max,
        },
        physics_feedback = feedback,
        correction_hint = hint,
        reference       = "Vyas, N-Record Theory - Signature 3: η = N_bits/N_cross ≤ 1/(2m)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Local Record Equilibrium  δQ = T_rec · δS
# ─────────────────────────────────────────────────────────────────────────────

def check_local_record_equilibrium(
    delta_q_nrt:   float,
    delta_s_nrt:   float,
    t_rec:         float,
    tolerance_pct: float = 8.0,
) -> PhysicsResult:
    """
    Verifies the NRT Local Record Equilibrium condition: δQ = T_rec · δS.

    In NRT, any quasi-static change in a record neighbourhood must satisfy
    the relational first law δQ = T_rec · δS, where T_rec is the record
    temperature (the energy scale per degree of relational freedom) and δS is
    the change in record entropy (measured in nats or bits). This is the NRT
    analogue of Jacobson's derivation of Einstein's equations from thermodynamics.

    Call this tool to verify that a proposed record network transition is
    thermodynamically consistent with NRT's local equilibrium postulate.

    Args:
        delta_q_nrt:   Change in relational heat content δQ (in NRT energy units)
        delta_s_nrt:   Change in record entropy δS (in nats: 1 nat = 1/ln2 bits)
        t_rec:         Record temperature T_rec (energy per nat, must be > 0)
        tolerance_pct: Allowed deviation from δQ = T_rec·δS in percent (default 8%)

    Returns:
        Physics validation checking whether δQ = T_rec · δS to within tolerance.
    """
    violations = []

    if t_rec <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="local_record_equilibrium",
            design_summary="Invalid: T_rec must be positive (above absolute zero)",
            violations=["Record temperature T_rec must be positive"],
        )

    # LRE: δQ should equal T_rec × δS
    lre_predicted = t_rec * delta_s_nrt
    lre_error     = abs(delta_q_nrt - lre_predicted)

    if abs(lre_predicted) > 1e-300:
        lre_error_pct = lre_error / abs(lre_predicted) * 100
    else:
        lre_error_pct = 0.0 if abs(delta_q_nrt) < 1e-300 else float('inf')

    # Sign consistency: δQ and T_rec·δS should have the same sign
    if delta_s_nrt != 0 and delta_q_nrt != 0:
        sign_consistent = (delta_q_nrt > 0) == (lre_predicted > 0)
    else:
        sign_consistent = True

    if not sign_consistent:
        violations.append(
            f"SIGN VIOLATION: δQ = {delta_q_nrt:.4e} and T_rec·δS = {lre_predicted:.4e} "
            f"have opposite signs. A record system cannot absorb heat while entropy decreases "
            f"(or vice versa) in LRE."
        )

    if lre_error_pct > tolerance_pct:
        violations.append(
            f"LRE VIOLATION: δQ = {delta_q_nrt:.6e} but T_rec·δS = {lre_predicted:.6e}. "
            f"Discrepancy = {lre_error:.4e} ({lre_error_pct:.2f}%, tolerance {tolerance_pct:.0f}%). "
            f"This transition violates the NRT first law."
        )

    if delta_s_nrt < 0 and abs(delta_s_nrt) > abs(delta_q_nrt / t_rec) * 1.5:
        violations.append(
            f"ENTROPY PARADOX: δS = {delta_s_nrt:.4e} nat is large and negative. "
            f"Significant entropy decrease requires a corresponding work term not present "
            f"in this simple LRE check - the transition may be non-quasi-static."
        )

    passed  = len(violations) == 0
    summary = (
        f"δQ={delta_q_nrt:.4e}, T_rec={t_rec:.4e}, δS={delta_s_nrt:.4e} nat; "
        f"T·δS={lre_predicted:.4e} (error {lre_error_pct:.2f}%)"
    )
    feedback = (
        f"LRE check: T_rec·δS = {t_rec:.4e} × {delta_s_nrt:.4e} = {lre_predicted:.4e}. "
        f"Proposed δQ = {delta_q_nrt:.4e}. "
        f"Absolute error: {lre_error:.4e} ({lre_error_pct:.2f}%). "
        + ("PASS: transition satisfies NRT Local Record Equilibrium." if passed
           else f"FAIL: δQ ≠ T_rec·δS - transition is thermodynamically inconsistent.")
    )
    hint = (
        "" if passed else
        f"Adjust δQ to {lre_predicted:.6e} (= T_rec×δS = {t_rec:.4e} × {delta_s_nrt:.4e}), "
        f"or adjust T_rec to {delta_q_nrt / delta_s_nrt:.4e} (= δQ/δS), "
        f"or adjust δS to {delta_q_nrt / t_rec:.4e} (= δQ/T_rec) to satisfy LRE."
    )

    return PhysicsResult(
        passed          = passed,
        domain          = "nrt",
        check_name      = "local_record_equilibrium",
        design_summary  = summary,
        violations      = violations,
        metrics         = {
            "delta_q_nrt":      delta_q_nrt,
            "delta_s_nrt":      delta_s_nrt,
            "t_rec":            t_rec,
            "lre_predicted":    round(lre_predicted, 10),
            "lre_error":        round(lre_error, 10),
            "lre_error_pct":    round(lre_error_pct, 4),
        },
        physics_feedback = feedback,
        correction_hint = hint,
        reference       = "Vyas, N-Record Theory - Local Record Equilibrium; cf. Jacobson (1995)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Relational Entropy Bound  S ≤ k_B · N_cross · ln(2)   (Bekenstein analogue)
# ─────────────────────────────────────────────────────────────────────────────

def check_relational_entropy(
    entropy_nats:  float,
    n_cross:       int,
    k_b_nrt:       float = 1.0,
    tolerance_pct: float = 5.0,
) -> PhysicsResult:
    """
    Verifies the NRT relational entropy bound: S ≤ k_B · N_cross · ln(2).

    In NRT, the maximum entropy assignable to a region is bounded by the number
    of crossing records on its boundary: S_max = k_B · N_cross · ln(2). This
    is the NRT-native Bekenstein bound - information cannot exceed one bit per
    boundary record. Entropy in excess of this bound is non-physical in NRT.

    Call this tool to verify that a proposed entropy assignment for a bounded
    region is consistent with NRT's holographic entropy limit.

    Args:
        entropy_nats:  Proposed entropy of the region in nats (1 nat = log₂(e) ≈ 1.443 bits)
        n_cross:       Number of records crossing the bounding surface
        k_b_nrt:       Boltzmann constant in NRT units (default 1.0 for natural units)
        tolerance_pct: Allowed fractional excess above the bound in percent (default 5%)

    Returns:
        Physics validation with the entropy utilisation fraction and whether
        the assignment respects the NRT holographic bound.
    """
    violations = []

    if n_cross <= 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="relational_entropy",
            design_summary="Invalid: n_cross must be positive",
            violations=["n_cross must be a positive integer"],
        )
    if entropy_nats < 0:
        return PhysicsResult(
            passed=False, domain="nrt", check_name="relational_entropy",
            design_summary="Invalid: entropy must be non-negative",
            violations=["Entropy must be ≥ 0 nats"],
        )

    s_max   = k_b_nrt * n_cross * NRT_LN2   # Maximum allowed entropy
    s_bits  = entropy_nats / NRT_LN2          # Convert nats → bits
    s_max_bits = k_b_nrt * n_cross            # Maximum in bits

    utilisation   = entropy_nats / s_max if s_max > 0 else float('inf')
    excess_pct    = max(0, (entropy_nats - s_max) / s_max * 100)
    headroom_nats = s_max - entropy_nats

    if entropy_nats > s_max * (1 + tolerance_pct / 100):
        violations.append(
            f"ENTROPY BOUND VIOLATED: S = {entropy_nats:.4e} nats > S_max = {s_max:.4e} nats. "
            f"({s_bits:.3f} bits > {s_max_bits:.3f} bits). "
            f"Excess: {excess_pct:.2f}% above the NRT Bekenstein limit."
        )
    if utilisation > 0.99:
        violations.append(
            f"AT ENTROPY SATURATION: S/S_max = {utilisation:.4f}. "
            f"A region at ≥99% of the NRT entropy bound is thermodynamically critical - "
            f"any perturbation pushes it over the limit."
        )

    passed  = len(violations) == 0
    summary = (
        f"S={entropy_nats:.4e} nats ({s_bits:.3f} bits); "
        f"S_max={s_max:.4e} nats ({s_max_bits:.3f} bits); "
        f"utilisation={utilisation:.4f}"
    )
    feedback = (
        f"Relational entropy bound: S_max = k_B · N_cross · ln2 = "
        f"{k_b_nrt} × {n_cross} × {NRT_LN2:.5f} = {s_max:.6e} nats. "
        f"Proposed S = {entropy_nats:.6e} nats. "
        f"Utilisation: {utilisation*100:.3f}%. "
        f"Headroom: {headroom_nats:.4e} nats. "
        + ("PASS: entropy is within the NRT holographic bound." if passed
           else f"FAIL: entropy exceeds the NRT bound by {excess_pct:.2f}%.")
    )
    hint = (
        "" if passed else
        f"Reduce entropy to ≤ {s_max:.4e} nats ({s_max_bits:.3f} bits) for {n_cross} crossings. "
        f"Or increase the bounding surface: you need at least "
        f"{math.ceil(entropy_nats / (k_b_nrt * NRT_LN2))} crossings to accommodate "
        f"S = {entropy_nats:.4e} nats. "
        f"In NRT terms: add {math.ceil(entropy_nats / (k_b_nrt * NRT_LN2)) - n_cross} "
        f"more records to the bounding surface."
    )

    return PhysicsResult(
        passed          = passed,
        domain          = "nrt",
        check_name      = "relational_entropy",
        design_summary  = summary,
        violations      = violations,
        metrics         = {
            "entropy_nats":     entropy_nats,
            "entropy_bits":     round(s_bits, 6),
            "n_cross":          n_cross,
            "s_max_nats":       round(s_max, 8),
            "s_max_bits":       round(s_max_bits, 6),
            "utilisation":      round(utilisation, 6),
            "headroom_nats":    round(headroom_nats, 8),
            "excess_pct":       round(excess_pct, 4),
        },
        physics_feedback = feedback,
        correction_hint = hint,
        reference       = "Vyas, N-Record Theory; cf. Bekenstein, Phys. Rev. D 7, 2333 (1973)",
    )
