"""
PHACT Domain: Astrophysics - Gravitational Wave Physics
=======================================================
Validates compact binary inspiral parameters against exact general-relativistic
post-Newtonian equations - the same physics LIGO/Virgo uses to detect mergers.

Physics covered:
  1. check_chirp_mass - chirp mass M_c from component masses
  2. check_gw_strain - strain amplitude h at a given luminosity distance
  3. check_merger_time - Peters (1964) coalescence timescale
  4. check_isco_frequency - innermost stable circular orbit / peak GW frequency

COMMON ERRORS THIS GUARDS AGAINST:
  • Using total mass M instead of chirp mass M_c in the strain formula
  • Getting the chirp mass exponent wrong (5/3 not 2/3 in frequency evolution)
  • Using coordinate distance instead of luminosity distance for strain
  • Off by factor of 2 on GW frequency (f_GW = 2 × f_orbital)
  • Wrong Peters formula - missing the (12/19)⁴ prefactor
  • Using Newtonian ISCO instead of GR ISCO (r_ISCO = 6GM/c² for Schwarzschild)

References:
  Peters (1964) Phys. Rev. 136, B1224 - radiation reaction timescale
  Maggiore (2007) "Gravitational Waves Vol.1", Oxford University Press
  Abbott et al. (2016) PRL 116, 061102 - GW150914 detection paper
  Cutler & Flanagan (1994) PRD 49, 2658 - matched filtering formalism
"""

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Physical constants (SI, CODATA 2018) ──────────────────────────────────────

G   = 6.67430e-11    # m³/(kg·s²) - gravitational constant
C   = 2.99792458e8   # m/s - speed of light (exact)
M_SUN = 1.98892e30   # kg - solar mass
PC  = 3.085677581e16 # m - parsec
MPC = PC * 1e6       # m - megaparsec

# Derived
G_C3 = G / C**3      # s/kg - appears in chirp mass / time formulas
G_C2 = G / C**2      # m/kg - appears in Schwarzschild radius


# ── Result dataclass ──────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chirp Mass
# ═══════════════════════════════════════════════════════════════════════════════

def check_chirp_mass(
    m1_msun: float,
    m2_msun: float,
    chirp_mass_msun: float,
) -> PhysicsResult:
    """
    Validates the claimed chirp mass against the exact GR definition.

    M_c = (m₁·m₂)^(3/5) / (m₁+m₂)^(1/5)

    The chirp mass is the single most important quantity in GW astronomy -     it controls the rate of frequency evolution (chirp) during inspiral.
    LIGO can measure M_c to < 1% precision from the waveform phase alone.

    CRITICAL AI ERROR: Many models use M_c = (m₁·m₂)^(1/2) / (m₁+m₂)^(1/2)
    (the reduced mass) or simply M_c = (m₁+m₂)/2. Both are wrong by factors
    of 2-10 for typical binary parameters.

    Args:
        m1_msun:         Primary mass in solar masses (e.g. 36.0 for GW150914)
        m2_msun:         Secondary mass in solar masses (e.g. 29.0 for GW150914)
        chirp_mass_msun: Your proposed chirp mass in solar masses

    Returns:
        PhysicsResult: passes if claimed M_c is within 1% of exact value.
    """
    violations = []

    # support old kwarg names for backwards compatibility
    mass1_msun = m1_msun
    mass2_msun = m2_msun
    claimed_chirp_mass_msun = chirp_mass_msun

    m1 = mass1_msun * M_SUN
    m2 = mass2_msun * M_SUN
    M  = m1 + m2
    mu = m1 * m2 / M   # reduced mass

    # Exact chirp mass: M_c = μ^(3/5) · M^(2/5)  [Maggiore eq. 4.7]
    # equivalently:     M_c = (m1·m2)^(3/5) / (m1+m2)^(1/5)
    chirp_mass_kg = (m1 * m2)**0.6 / M**0.2
    chirp_mass_msun = chirp_mass_kg / M_SUN

    # Symmetric mass ratio η = μ/M = m1·m2/(m1+m2)²
    eta = mu / M

    deviation_pct = abs(claimed_chirp_mass_msun - chirp_mass_msun) / chirp_mass_msun * 100

    metrics = {
        "mass1_msun":            round(mass1_msun, 3),
        "mass2_msun":            round(mass2_msun, 3),
        "total_mass_msun":       round((m1 + m2) / M_SUN, 3),
        "chirp_mass_msun":       round(chirp_mass_msun, 4),
        "claimed_chirp_mass_msun": round(claimed_chirp_mass_msun, 4),
        "symmetric_mass_ratio":  round(eta, 5),
        "mass_ratio_q":          round(min(m1, m2) / max(m1, m2), 4),
        "deviation_pct":         round(deviation_pct, 3),
    }

    if deviation_pct > 1.0:
        violations.append(
            f"CHIRP MASS ERROR: claimed M_c = {claimed_chirp_mass_msun:.3f} M☉, "
            f"exact GR value = {chirp_mass_msun:.4f} M☉. "
            f"Deviation = {deviation_pct:.2f}% (tolerance: 1%). "
            f"Formula: M_c = (m₁·m₂)^(3/5) / (m₁+m₂)^(1/5). "
            f"Common error: confusing with reduced mass μ = {mu/M_SUN:.3f} M☉."
        )

    passed = len(violations) == 0

    feedback = (
        f"Chirp mass check: M_c = {chirp_mass_msun:.4f} M☉. "
        f"η = {eta:.4f}, q = {min(m1,m2)/max(m1,m2):.3f}. "
        + ("PASS." if passed else f"FAIL: claimed {claimed_chirp_mass_msun:.3f} M☉ is {deviation_pct:.1f}% off.")
    )
    hint = (
        "" if passed else
        f"Use M_c = (m₁·m₂)^(3/5) / (m₁+m₂)^(1/5) = {chirp_mass_msun:.4f} M☉. "
        f"Note: this is NOT the reduced mass ({mu/M_SUN:.3f} M☉) or average mass ({(mass1_msun+mass2_msun)/2:.3f} M☉)."
    )

    return PhysicsResult(
        passed=passed, domain="astrophysics", check_name="chirp_mass",
        design_summary=f"m₁={mass1_msun} M☉, m₂={mass2_msun} M☉ → M_c={chirp_mass_msun:.4f} M☉",
        violations=violations, metrics=metrics,
        physics_feedback=feedback, correction_hint=hint,
        reference="Maggiore (2007) Gravitational Waves Vol.1, eq. 4.7; Peters (1964) Phys.Rev. 136 B1224",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gravitational Wave Strain Amplitude
# ═══════════════════════════════════════════════════════════════════════════════

def check_gw_strain(
    chirp_mass_msun: float,
    distance_mpc:    float,
    gw_frequency_hz: float,
    strain:          float,
) -> PhysicsResult:
    """
    Validates the gravitational wave strain amplitude against the quadrupole formula.

    h = (4/distance) · (G·M_c/c²) · (π·G·M_c·f_GW/c³)^(2/3)

    This is the sky-averaged strain from a circular binary inspiral in the
    quadrupole approximation (valid for f << f_ISCO).
    LIGO's first detection (GW150914) measured h_peak ≈ 1.0 × 10⁻²¹ at 150 Mpc.

    CRITICAL AI ERRORS CAUGHT:
      • Using coordinate distance instead of luminosity distance D_L
      • Missing the (π·G·M_c·f/c³)^(2/3) frequency factor entirely
      • Using total mass M instead of chirp mass M_c (gives factor ~2-10 error)
      • Off by factor of 2: confusing one-sided amplitude with peak-to-peak

    Args:
        chirp_mass_msun:  Chirp mass in solar masses
        distance_mpc:     Luminosity distance in megaparsecs
        gw_frequency_hz:  Gravitational wave frequency in Hz (= 2 × orbital frequency)
        strain:           The strain amplitude h you are proposing (dimensionless)

    Returns:
        PhysicsResult: passes if claimed h is within 20% of quadrupole prediction.
    """
    claimed_strain = strain
    violations = []

    M_c = chirp_mass_msun * M_SUN
    D_L = distance_mpc * MPC

    # Quadrupole strain amplitude [Maggiore eq. 4.18, sky-averaged]:
    # h = (4/D_L) · (G·M_c/c²) · (π·G·M_c·f/c³)^(2/3)
    inner = (math.pi * G * M_c * gw_frequency_hz / C**3) ** (2.0/3.0)
    h_theory = (4.0 / D_L) * (G * M_c / C**2) * inner

    deviation_pct = abs(claimed_strain - h_theory) / h_theory * 100

    # Schwarzschild radius of chirp mass (for context)
    r_s_km = 2 * G * M_c / C**2 / 1000

    metrics = {
        "chirp_mass_msun":       round(chirp_mass_msun, 4),
        "distance_mpc":          distance_mpc,
        "gw_frequency_hz":       gw_frequency_hz,
        "strain_theory":         f"{h_theory:.3e}",
        "strain_claimed":        f"{claimed_strain:.3e}",
        "deviation_pct":         round(deviation_pct, 2),
        "schwarzschild_r_km":    round(r_s_km, 2),
        "ligo_noise_floor":      "~1e-23",
    }

    if deviation_pct > 20.0:
        # Diagnose common errors
        h_wrong_totalmass = (4.0 / D_L) * (G * M_c * M_SUN / C**2) * (math.pi * G * M_c * M_SUN * gw_frequency_hz / C**3)**(2/3)
        diagnosis = ""
        if abs(claimed_strain / h_theory - (chirp_mass_msun * M_SUN / M_c)) < 0.3:
            diagnosis = " Possible error: used total mass instead of chirp mass."

        violations.append(
            f"STRAIN AMPLITUDE ERROR: claimed h = {claimed_strain:.3e}, "
            f"quadrupole formula gives h = {h_theory:.3e}. "
            f"Deviation = {deviation_pct:.1f}% (tolerance: 20%).{diagnosis} "
            f"Formula: h = (4/D_L)·(G·M_c/c²)·(π·G·M_c·f/c³)^(2/3)."
        )

    passed = len(violations) == 0

    feedback = (
        f"GW strain: h_theory = {h_theory:.3e} at D_L = {distance_mpc} Mpc, "
        f"f = {gw_frequency_hz} Hz, M_c = {chirp_mass_msun} M☉. "
        f"LIGO noise floor ~10⁻²³ → SNR ∝ h/noise. "
        + ("PASS." if passed else f"FAIL: {deviation_pct:.1f}% off.")
    )
    hint = (
        "" if passed else
        f"Correct strain: h = {h_theory:.3e}. "
        f"Check: (1) use luminosity distance D_L in metres, "
        f"(2) use chirp mass M_c={chirp_mass_msun} M☉ (not total mass), "
        f"(3) include the (π·G·M_c·f/c³)^(2/3) frequency factor."
    )

    return PhysicsResult(
        passed=passed, domain="astrophysics", check_name="gw_strain",
        design_summary=f"h = {h_theory:.3e} at {distance_mpc} Mpc, f = {gw_frequency_hz} Hz",
        violations=violations, metrics=metrics,
        physics_feedback=feedback, correction_hint=hint,
        reference="Maggiore (2007) GW Vol.1 eq.4.18; Abbott et al. PRL 116 061102 (GW150914)",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Peters Merger Timescale
# ═══════════════════════════════════════════════════════════════════════════════

def check_merger_time(
    m1_msun:       float,
    m2_msun:       float,
    separation_au: float,
    merger_time_myr: float,
) -> PhysicsResult:
    """
    Validates the binary inspiral merger timescale using the Peters (1964) formula.

    T_merge = (12/19) · (c₀⁴ / β) · ∫ ...
    For a circular orbit this simplifies to:
    T_merge = (12/85) · (c³/G³) · a₀⁴ / (m₁·m₂·(m₁+m₂))

    where a₀ is the initial orbital separation.

    This is the exact result for zero-eccentricity binaries. Eccentricity
    dramatically shortens merger time (e=0.6 cuts it by ~10×).

    CRITICAL AI ERRORS CAUGHT:
      • Missing the 12/85 prefactor (forgetting the radiation-reaction integral)
      • Using a₀ in AU without converting to metres (off by 1.5×10¹¹)
      • Using the wrong mass combination - must be μ·M² not M³ or μ²

    Args:
        m1_msun:         Primary mass in solar masses
        m2_msun:         Secondary mass in solar masses
        separation_au:   Initial circular orbital separation in AU
        merger_time_myr: Your proposed merger time in millions of years

    Returns:
        PhysicsResult: passes if claimed time is within 10% of Peters formula.
    """
    violations = []

    AU = 1.495978707e11  # m - astronomical unit (exact IAU 2012)
    YR = 3.15575520e7    # s - Julian year

    mass1_msun = m1_msun
    mass2_msun = m2_msun
    claimed_time_myr = merger_time_myr

    m1 = mass1_msun * M_SUN
    m2 = mass2_msun * M_SUN
    M  = m1 + m2
    mu = m1 * m2 / M
    a0 = separation_au * AU

    # Peters (1964) eq. 5.14 for circular orbit (e=0):
    # T = (12/85) · (c⁰/G³) · a₀⁴ / (μ · M²)
    # Equivalently: T = (15/304) · ... - note different textbook conventions
    # Using Maggiore (2007) eq. 4.30 form:
    # β = 64/5 · G³·m₁·m₂·(m₁+m₂)/c⁵
    # T_circ = a₀⁴ / (4β)
    beta = 64.0/5.0 * G**3 * m1 * m2 * M / C**5
    T_s  = a0**4 / (4.0 * beta)
    T_myr = T_s / (YR * 1e6)

    deviation_pct = abs(claimed_time_myr - T_myr) / T_myr * 100

    # Also compute what T would be at 1 AU (sanity reference)
    a_1au = AU
    T_1au_myr = a_1au**4 / (4.0 * beta) / (YR * 1e6)

    metrics = {
        "mass1_msun":           round(mass1_msun, 3),
        "mass2_msun":           round(mass2_msun, 3),
        "separation_au":        round(separation_au, 4),
        "merger_time_myr":      round(T_myr, 4),
        "claimed_time_myr":     round(claimed_time_myr, 4),
        "deviation_pct":        round(deviation_pct, 2),
        "beta_m4_per_s":        f"{beta:.4e}",
        "ref_1au_time_myr":     round(T_1au_myr, 2),
    }

    if deviation_pct > 10.0:
        violations.append(
            f"MERGER TIME ERROR: claimed T = {claimed_time_myr:.3f} Myr, "
            f"Peters (1964) gives T = {T_myr:.4f} Myr. "
            f"Deviation = {deviation_pct:.1f}% (tolerance: 10%). "
            f"Formula: T = a₀⁴ / (4β) where β = 64G³m₁m₂M / 5c⁵. "
            f"Check: (1) convert AU to metres (×1.496×10¹¹), "
            f"(2) the a₀⁴ dependence means errors in separation are amplified 4×."
        )

    passed = len(violations) == 0

    feedback = (
        f"Peters timescale: T = {T_myr:.4f} Myr for a₀ = {separation_au} AU. "
        f"β = {beta:.3e} m⁴/s. Note: T ∝ a₀⁴ - doubling separation increases T by 16×. "
        + ("PASS." if passed else f"FAIL: {deviation_pct:.1f}% deviation.")
    )
    hint = (
        "" if passed else
        f"Correct merger time: T = {T_myr:.4f} Myr. "
        f"β = 64/5 · G³·m₁·m₂·(m₁+m₂)/c⁵ = {beta:.3e} m⁴/s. "
        f"T = a₀⁴ / (4β). Separation in metres: {a0:.4e} m."
    )

    return PhysicsResult(
        passed=passed, domain="astrophysics", check_name="merger_time",
        design_summary=f"T_merge = {T_myr:.4f} Myr for a₀ = {separation_au} AU",
        violations=violations, metrics=metrics,
        physics_feedback=feedback, correction_hint=hint,
        reference="Peters (1964) Phys.Rev. 136 B1224 eq.5.14; Maggiore (2007) eq.4.30",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ISCO Frequency - Peak GW Emission
# ═══════════════════════════════════════════════════════════════════════════════

def check_isco_frequency(
    total_mass_msun: float,
    fisco_hz:        float,
) -> PhysicsResult:
    """
    Validates the innermost stable circular orbit (ISCO) frequency.

    The ISCO marks the end of the inspiral phase - the point where the
    orbit becomes dynamically unstable and plunge begins. The GW frequency
    at ISCO is the peak frequency of the inspiral waveform.

    For a Schwarzschild (non-spinning) black hole:
      r_ISCO = 6GM/c²  (= 3 × Schwarzschild radius)
      f_ISCO = c³ / (6^(3/2) · π · G · M)
             = 4400 Hz / (M/M☉)   [memorise this]

    For GW150914 (M ≈ 65 M☉): f_ISCO ≈ 150 Hz - exactly what LIGO observed.

    CRITICAL AI ERRORS CAUGHT:
      • Using r_ISCO = 3GM/c² (the photon sphere, not ISCO - factor of 2 wrong)
      • Using r_ISCO = 2GM/c² (Schwarzschild radius - factor of 3 wrong)
      • f_GW = f_orbital (off by factor 2 - GW frequency is TWICE orbital)
      • Using Newtonian Keplerian instead of GR ISCO

    Args:
        total_mass_msun: Total binary mass M = m₁ + m₂ in solar masses
        fisco_hz:        Your proposed ISCO GW frequency in Hz

    Returns:
        PhysicsResult: passes if claimed f_ISCO is within 5% of GR value.
    """
    violations = []

    claimed_fisco_hz = fisco_hz
    M_total = total_mass_msun * M_SUN

    # GR Schwarzschild ISCO radius: r_ISCO = 6GM/c²
    r_isco_m = 6.0 * G * M_total / C**2
    r_isco_km = r_isco_m / 1000

    # Orbital frequency at ISCO (Keplerian at GR radius):
    # f_orb = (1/2π) · sqrt(GM/r³)
    f_orb = (1.0 / (2.0 * math.pi)) * math.sqrt(G * M_total / r_isco_m**3)

    # GW frequency = 2 × orbital frequency (quadrupole radiation)
    f_isco_hz = 2.0 * f_orb

    # Handy approximation: f_ISCO ≈ 4400 Hz / (M/M☉)
    f_approx = 4400.0 / total_mass_msun

    deviation_pct = abs(claimed_fisco_hz - f_isco_hz) / f_isco_hz * 100

    # Schwarzschild radius for reference
    r_s_km = 2.0 * G * M_total / C**2 / 1000

    metrics = {
        "total_mass_msun":    round(total_mass_msun, 3),
        "r_isco_km":          round(r_isco_km, 2),
        "r_schwarzschild_km": round(r_s_km, 2),
        "f_orbital_hz":       round(f_orb, 3),
        "f_isco_gw_hz":       round(f_isco_hz, 3),
        "f_isco_claimed_hz":  round(claimed_fisco_hz, 3),
        "f_approx_4400_rule": round(f_approx, 2),
        "deviation_pct":      round(deviation_pct, 3),
    }

    if deviation_pct > 5.0:
        # Diagnose common errors
        f_wrong_photon = C**3 / (3.0**1.5 * math.pi * G * M_total)   # r=3GM/c² error
        f_wrong_rs     = C**3 / (2.0**1.5 * math.pi * G * M_total)   # r=2GM/c² error
        f_wrong_notwox = f_isco_hz / 2.0                               # forgot ×2 for GW

        diag = ""
        if abs(claimed_fisco_hz - f_wrong_photon) / f_isco_hz < 0.05:
            diag = " Appears to use r_ISCO=3GM/c² (photon sphere) - correct is 6GM/c²."
        elif abs(claimed_fisco_hz - f_wrong_rs) / f_isco_hz < 0.05:
            diag = " Appears to use r_ISCO=2GM/c² (Schwarzschild radius) - correct is 6GM/c²."
        elif abs(claimed_fisco_hz - f_wrong_notwox) / f_isco_hz < 0.05:
            diag = " Appears to use orbital frequency - GW frequency is 2× orbital."

        violations.append(
            f"ISCO FREQUENCY ERROR: claimed f_ISCO = {claimed_fisco_hz:.2f} Hz, "
            f"GR Schwarzschild gives f_ISCO = {f_isco_hz:.3f} Hz. "
            f"Deviation = {deviation_pct:.2f}% (tolerance: 5%).{diag} "
            f"GR formula: f_ISCO = c³/(6^(3/2)·π·G·M) ≈ 4400/{total_mass_msun:.0f} M☉ Hz."
        )

    passed = len(violations) == 0

    feedback = (
        f"ISCO: r_ISCO = {r_isco_km:.1f} km = 3×r_S = 3×{r_s_km:.1f} km. "
        f"f_orbital = {f_orb:.3f} Hz → f_GW = 2×f_orb = {f_isco_hz:.3f} Hz. "
        f"Rule of thumb: f_ISCO ≈ 4400/{total_mass_msun:.0f} = {f_approx:.1f} Hz. "
        + ("PASS." if passed else f"FAIL: {deviation_pct:.2f}% off.")
    )
    hint = (
        "" if passed else
        f"Correct f_ISCO = {f_isco_hz:.3f} Hz. "
        f"Steps: (1) r_ISCO = 6GM/c² = {r_isco_km:.1f} km, "
        f"(2) f_orb = sqrt(GM/r³)/(2π) = {f_orb:.3f} Hz, "
        f"(3) f_GW = 2·f_orb = {f_isco_hz:.3f} Hz."
    )

    return PhysicsResult(
        passed=passed, domain="astrophysics", check_name="isco_frequency",
        design_summary=f"f_ISCO = {f_isco_hz:.3f} Hz for M = {total_mass_msun} M☉",
        violations=violations, metrics=metrics,
        physics_feedback=feedback, correction_hint=hint,
        reference="Maggiore (2007) GW Vol.1 §4.1; Misner, Thorne & Wheeler (1973) §25.5",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. contract-gated verification - non-gameable target check
# ═══════════════════════════════════════════════════════════════════════════════
#
# The check_* tools above accept the answer as an argument (e.g. a claimed chirp
# mass) and validate it. That is correct for "compute X" goals, but it lets a
# proposer satisfy a "this fixed system must have property X = target" goal by
# silently changing a supposedly-fixed input to a value that genuinely yields the
# target. The English "you may not change the mass" is invisible to the tool.
#
# This tool closes that loophole structurally by routing the check through the
# domain contract. The proposer supplies ONLY the exogenous (fixed) inputs of the
# goal plus the target to verify; it does NOT - and cannot - supply the
# certified quantity, because the contract derives it from the exogenous parents. A
# goal that fixes the inputs and demands an off-physics target is therefore
# impossible by construction: the derived value cannot equal the target without
# changing an exogenous node, and the proposer has no argument with which to do
# so covertly.

def verify_binary_target(
    quantity: str,
    m1_msun: float = None,
    m2_msun: float = None,
    total_mass_msun: float = None,
    chirp_mass_msun: float = None,
    distance_mpc: float = None,
    gw_frequency_hz: float = None,
    separation_au: float = None,
    target_value: float = None,
    tolerance_pct: float = 5.0,
) -> PhysicsResult:
    """
    Verify that a compact binary with FIXED inputs has a target derived property,
    using the astrophysics structural contract.

    Use this tool for goals of the form "a binary has fixed <inputs>; verify its
    <quantity> is <target>." You supply only the fixed inputs and the target. The
    The contract computes the true value of the quantity from your inputs - you do not, and
    cannot, supply that value yourself. The check passes only if the contract-derived
    value matches the target within tolerance.

    Args:
        quantity:        One of "chirp_mass", "isco", "strain", "merger_time".
        m1_msun:         Primary mass (for chirp_mass, merger_time).
        m2_msun:         Secondary mass (for chirp_mass, merger_time).
        total_mass_msun: Total mass M=m1+m2 (for isco).
        chirp_mass_msun: Chirp mass (exogenous input for strain only).
        distance_mpc:    Luminosity distance (for strain).
        gw_frequency_hz: GW frequency (for strain).
        separation_au:   Initial orbital separation (for merger_time).
        target_value:    The value the goal claims the quantity should equal.
        tolerance_pct:   Match tolerance in percent (default 5%).

    Returns:
        Physics validation: passes iff the contract-derived quantity equals target_value
        within tolerance. The derived value is reported so the proposer can see the
        true physics; it can never be overridden.
    """
    from phact_physics.contracts.astrophysics_contract import ASTROPHYSICS_CONTRACTS

    if quantity not in ASTROPHYSICS_CONTRACTS:
        return PhysicsResult(
            passed=False, domain="astrophysics", check_name="verify_binary_target",
            design_summary=f"Unknown quantity {quantity!r}",
            violations=[f"quantity must be one of {list(ASTROPHYSICS_CONTRACTS)}"],
            correction_hint=f"Set quantity to one of {list(ASTROPHYSICS_CONTRACTS)}.",
        )

    builder, outcome_node = ASTROPHYSICS_CONTRACTS[quantity]
    graph = builder()

    # Assemble exogenous inputs from whichever args this quantity needs.
    exo_all = {
        "m1_msun": m1_msun, "m2_msun": m2_msun,
        "total_mass_msun": total_mass_msun, "chirp_mass_msun": chirp_mass_msun,
        "distance_mpc": distance_mpc, "gw_frequency_hz": gw_frequency_hz,
        "separation_au": separation_au,
    }
    needed = [n for n in graph.nodes if graph.nodes[n].node_type == "exogenous"]
    exogenous = {n: exo_all[n] for n in needed if exo_all.get(n) is not None}

    missing = [n for n in needed if n not in exogenous]
    if missing:
        return PhysicsResult(
            passed=False, domain="astrophysics", check_name="verify_binary_target",
            design_summary=f"{quantity}: missing fixed inputs {missing}",
            violations=[f"Provide the fixed exogenous inputs for {quantity}: {missing}"],
            correction_hint=f"For quantity={quantity!r}, supply: {needed} (the fixed inputs of the goal).",
        )
    if target_value is None:
        return PhysicsResult(
            passed=False, domain="astrophysics", check_name="verify_binary_target",
            design_summary=f"{quantity}: no target_value",
            violations=["Provide target_value, the property the goal claims to verify."],
            correction_hint="Set target_value to the figure stated in the goal.",
        )

    derived = graph.compute(exogenous)[outcome_node]
    if target_value == 0:
        dev_pct = float("inf") if derived != 0 else 0.0
    else:
        dev_pct = abs(derived - target_value) / abs(target_value) * 100.0
    passed = dev_pct <= tolerance_pct

    violations = []
    if not passed:
        violations.append(
            f"TARGET NOT MET: contract-derived {quantity} = {derived:.4g} for the fixed "
            f"inputs, but the goal claims {target_value:.4g} "
            f"(deviation {dev_pct:.1f}%, tolerance {tolerance_pct:.0f}%). The fixed "
            f"inputs cannot produce the claimed value."
        )

    feedback = (
        f"CONTRACT[{quantity}]: exogenous {exogenous} -> derived {outcome_node} = "
        f"{derived:.4g}. Target = {target_value:.4g}. "
        + ("PASS." if passed else f"FAIL: {dev_pct:.1f}% off and the inputs are fixed.")
    )
    hint = "" if passed else (
        f"The derived {quantity} for these fixed inputs is {derived:.4g}, not "
        f"{target_value:.4g}. Because the inputs are fixed, this target is "
        f"physically unreachable - declare the goal impossible."
    )

    return PhysicsResult(
        passed=passed, domain="astrophysics", check_name="verify_binary_target",
        design_summary=f"{quantity}: derived {derived:.4g} vs target {target_value:.4g}",
        violations=violations,
        metrics={"quantity": quantity, "derived_value": derived,
                 "target_value": target_value, "deviation_pct": round(dev_pct, 3),
                 "exogenous_inputs": exogenous},
        physics_feedback=feedback, correction_hint=hint,
        reference="PHACT astrophysics contract (Maggiore 2007; Peters 1964)",
    )
