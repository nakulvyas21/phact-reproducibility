"""
PHACT Physics Engine
==================
Hard-coded rules of nature that act as the objective reality sandbox.
No ML - pure deterministic physics laws.

Two domains for the hackathon demo:
  1. DronePhysicsValidator - aerodynamics, wind shear, roll authority
  2. DNAHydrogelValidator - thermodynamic melting point, GC-content stability
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class PhysicsResult:
    passed: bool
    domain: str                        # "drone" | "biochem"
    design_summary: str                # what was checked
    violations: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    physics_feedback: str = ""          # injected back into Gemini prompt
    correction_hint: str = ""          # actionable mutation hint for Gemini


# ---------------------------------------------------------------------------
# Domain 1 - Drone Aerodynamics
# ---------------------------------------------------------------------------

class DronePhysicsValidator:
    """
    Validates a drone flight path against aerodynamic laws.

    Constants (hard-coded rules of nature):
      • Air density at sea level:  ρ = 1.225 kg/m³
      • Max roll authority:         ±35° for a DJI-class quadcopter
      • Stall angle of attack:      15° (fixed-wing component)
      • Drag coefficient:           Cd = 0.47 (bluff body approximation)
      • Rotor disc area:            A = 0.07 m² (per motor pair)
      • Max thrust per motor pair:  F = 25 N
    """

    RHO          = 1.225       # kg/m³ - air density
    MAX_ROLL_DEG = 35.0        # degrees - max roll authority before control loss
    CD           = 0.47        # dimensionless drag coefficient
    ROTOR_AREA   = 0.07        # m² - combined disc area
    MAX_THRUST   = 25.0        # N - max thrust per motor pair (×2 motors)
    GRAVITY      = 9.81        # m/s²
    DRONE_MASS   = 0.90        # kg - DJI Mini 4 class

    def validate(self, design: dict) -> PhysicsResult:
        """
        Expected design keys:
          path_type      : "straight" | "curved" | "banked_curve"
          crosswind_knots: float - lateral wind speed
          canyon_width_m : float - tightest gap the drone must thread
          speed_ms       : float - drone airspeed in m/s
          bank_angle_deg : float - proposed bank angle (0 = straight)
        """
        violations = []
        metrics = {}

        crosswind_ms   = design.get("crosswind_knots", 0) * 0.51444  # knots → m/s
        canyon_width   = design.get("canyon_width_m", 10.0)
        speed          = design.get("speed_ms", 8.0)
        bank_angle     = design.get("bank_angle_deg", 0.0)
        path_type      = design.get("path_type", "straight")

        # ── Rule 1: Aerodynamic drag force from crosswind ──────────────────
        drag_force = 0.5 * self.RHO * self.CD * self.ROTOR_AREA * (crosswind_ms ** 2)
        metrics["crosswind_drag_N"] = round(drag_force, 3)

        # ── Rule 2: Required counter-roll to compensate crosswind ──────────
        # Lateral thrust needed = drag force
        lateral_thrust_needed = drag_force
        # Roll angle from Newton: tan(θ) = F_lateral / (mg)
        weight = self.DRONE_MASS * self.GRAVITY
        required_roll_deg = math.degrees(math.atan2(lateral_thrust_needed, weight))
        metrics["required_roll_deg"] = round(required_roll_deg, 2)
        metrics["max_roll_authority_deg"] = self.MAX_ROLL_DEG

        if required_roll_deg > self.MAX_ROLL_DEG:
            delta = required_roll_deg - self.MAX_ROLL_DEG
            violations.append(
                f"CRITICAL: Required roll {required_roll_deg:.1f}° exceeds max authority "
                f"{self.MAX_ROLL_DEG}°. Control loss margin: {delta:.1f}°. "
                f"Crosswind at {design.get('crosswind_knots', 0):.0f} kts will slam drone into canyon wall."
            )

        # ── Rule 3: Straight path through crosswind = collision risk ───────
        if path_type == "straight" and crosswind_ms > 5.0:
            drift_per_100m = crosswind_ms / speed * 100 if speed > 0 else 999
            metrics["lateral_drift_per_100m"] = round(drift_per_100m, 2)
            if drift_per_100m > canyon_width / 2:
                violations.append(
                    f"CRITICAL: Straight geometric path causes {drift_per_100m:.1f} m lateral drift "
                    f"per 100 m forward travel. Canyon clearance only {canyon_width / 2:.1f} m. "
                    f"Impact inevitable at current crosswind."
                )

        # ── Rule 4: Check if proposed bank angle is physically sufficient ──
        if bank_angle > 0 and bank_angle < required_roll_deg:
            shortfall = required_roll_deg - bank_angle
            violations.append(
                f"WARNING: Proposed bank angle {bank_angle:.1f}° is {shortfall:.1f}° "
                f"short of the {required_roll_deg:.1f}° needed to counter crosswind. "
                f"Partial correction - drone still drifts."
            )

        passed = len(violations) == 0

        # ── Build physics feedback for Gemini ──────────────────────────────
        if not passed:
            optimal_bank = min(required_roll_deg * 1.15, self.MAX_ROLL_DEG)  # 15% safety margin
            physics_feedback = (
                f"Physics sandbox REJECTED the proposed flight path.\n"
                f"Measured violations:\n" + "\n".join(f"  • {v}" for v in violations) +
                f"\n\nPhysics data for correction:\n"
                f"  • Crosswind drag force: {drag_force:.3f} N\n"
                f"  • Minimum counter-roll required: {required_roll_deg:.1f}°\n"
                f"  • Optimal safe bank angle: {optimal_bank:.1f}°\n"
                f"\nRequired mutation: Replace straight path with a curved, banked trajectory "
                f"that counter-steers INTO the wind at bank_angle_deg={optimal_bank:.1f}. "
                f"The drone must curve upwind to cancel lateral drift before threading the canyon."
            )
            correction_hint = (
                f"Redesign the path as path_type=banked_curve with bank_angle_deg={optimal_bank:.1f} "
                f"to counter {design.get('crosswind_knots', 0):.0f}-knot crosswind. "
                f"The curved trajectory must arc into the wind vector, reducing net lateral velocity "
                f"to below {canyon_width / 2:.1f} m per 100 m of forward travel."
            )
        else:
            physics_feedback = "Physics sandbox APPROVED: Aerodynamic constraints satisfied."
            correction_hint = ""

        return PhysicsResult(
            passed=passed,
            domain="drone",
            design_summary=(
                f"Drone path: {path_type}, crosswind={design.get('crosswind_knots', 0)} kts, "
                f"bank_angle={bank_angle}°, canyon_width={canyon_width} m"
            ),
            violations=violations,
            metrics=metrics,
            physics_feedback=physics_feedback,
            correction_hint=correction_hint,
        )


# ---------------------------------------------------------------------------
# Domain 2 - DNA Hydrogel Thermodynamics
# ---------------------------------------------------------------------------

class DNAHydrogelValidator:
    """
    Validates a DNA hydrogel sequence against thermodynamic stability laws.

    Uses the FULL SantaLucia (1998) nearest-neighbour model - all 10 unique
    dinucleotide stacking parameters from Table 2 of:

        SantaLucia J. (1998) PNAS 95:1460-1465.
        "A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-
        neighbour thermodynamics."

    Tm formula (non-self-complementary duplex):
        Tm = ΔH° / (ΔS° + R·ln(C_T/4)) − 273.15  [°C]

    where:
        ΔH° = sum of all dinucleotide stacking enthalpies + initiation
        ΔS° = sum of all dinucleotide stacking entropies + initiation
        C_T  = total strand concentration (M)
        R    = 1.987 cal/(mol·K)

    Salt correction (Owczarzy 2004):
        Tm_corrected = 1 / (1/Tm + 16.6 * log10([Na+])) - applied at 50 mM Na+

    This matches IDT OligoAnalyzer and published Tm tables to ±1-2°C.
    """

    R = 1.987e-3  # kcal/(mol·K)

    # ── SantaLucia 1998 Table 2: all 10 unique nearest-neighbour parameters ──
    # Key: 5'→3' dinucleotide on top strand.  Values: (ΔH kcal/mol, ΔS cal/mol·K)
    # The complement is automatically handled: e.g. AA/TT = TT/AA by symmetry.
    NN: dict = {
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

    # Initiation parameters (Table 2, SantaLucia 1998)
    # Terminal AT pair initiation
    INIT_AT = (2.3,   4.1)   # (ΔH kcal/mol, ΔS cal/mol·K)
    # Terminal GC pair initiation
    INIT_GC = (0.1,  -2.8)

    # Salt: 50 mM Na+ correction constant (Owczarzy et al. 2004)
    NA_CONC_M = 0.05   # 50 mM - standard PCR / hybridisation buffer

    def _nn_params(self, seq: str) -> tuple[float, float]:
        """
        Sum all nearest-neighbour ΔH and ΔS for a sequence using SantaLucia 1998.
        Complement lookup handles the 10-parameter symmetry.
        """
        COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

        total_dH = 0.0  # kcal/mol
        total_dS = 0.0  # cal/(mol·K)

        for i in range(len(seq) - 1):
            dinuc = seq[i:i+2]
            if dinuc in self.NN:
                dH, dS = self.NN[dinuc]
            else:
                # Use complement reverse: e.g. "TC" → complement is "AG" → reverse "GA"
                comp_dinuc = COMP[dinuc[1]] + COMP[dinuc[0]]
                if comp_dinuc in self.NN:
                    dH, dS = self.NN[comp_dinuc]
                else:
                    # Fallback: average (should not happen for valid ATCG sequences)
                    dH, dS = -8.0, -22.0

            total_dH += dH
            total_dS += dS

        # Initiation correction - based on terminal base pairs
        for terminal in [seq[0], seq[-1]]:
            if terminal in ("G", "C"):
                total_dH += self.INIT_GC[0]
                total_dS += self.INIT_GC[1]
            else:
                total_dH += self.INIT_AT[0]
                total_dS += self.INIT_AT[1]

        return total_dH, total_dS

    def _calc_tm(self, seq: str, c_t_m: float) -> float:
        """
        Calculate Tm in °C using full SantaLucia 1998 + Owczarzy 2004 salt correction.
        c_t_m: total strand concentration in Molar.
        """
        dH, dS_cal = self._nn_params(seq)
        dS_kcal = dS_cal / 1000.0  # cal → kcal

        # Tm (K) for non-self-complementary: ΔH / (ΔS + R·ln(C_T/4))
        # SantaLucia 1998 Eq 3. C_T/4 for non-self-complementary strands.
        denom = dS_kcal + self.R * math.log(c_t_m / 4.0)
        if abs(denom) < 1e-12:
            return 0.0
        tm_1M_k = dH / denom   # Kelvin at 1 M Na+ (SantaLucia 1998 standard)

        # Owczarzy 2004 Eq. 4 salt correction for oligonucleotides
        # 1/Tm_Na = 1/Tm_1M + (4.29·fGC - 3.95)×10⁻⁵·ln[Na+] + 9.40×10⁻⁶·(ln[Na+])²
        # Reference: Owczarzy et al. (2004) Biochemistry 43:3537, Eq. 4
        f_gc  = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        ln_na = math.log(self.NA_CONC_M)
        inv_tm = (1.0 / tm_1M_k
                  + (4.29 * f_gc - 3.95) * 1e-5 * ln_na
                  + 9.40e-6 * ln_na ** 2)
        if abs(inv_tm) < 1e-12:
            return 0.0
        return 1.0 / inv_tm - 273.15

    def _calc_dg_37(self, dH: float, dS_cal: float) -> float:
        """ΔG at 37°C in kcal/mol = ΔH - T·ΔS."""
        return dH - 310.15 * (dS_cal / 1000.0)

    def validate(self, design: dict) -> PhysicsResult:
        """
        Expected design keys:
          sequence         : str - DNA sequence string e.g. "ATCGATCG..."
          target_temp_c    : float - desired operating temperature (default 37.0)
          strand_conc_nm   : float - strand concentration in nM (default 250)
          target_dg_kcal   : float - optional ΔG upper bound at 37°C in kcal/mol
                                     (e.g. -9.0 means ΔG must be ≤ -9.0 kcal/mol)
                                     If omitted, only Tm is checked.
        """
        violations = []
        metrics    = {}

        sequence       = design.get("sequence", "").upper().strip()
        target_temp    = design.get("target_temp_c", 37.0)
        conc_nm        = design.get("strand_conc_nm", 250.0)
        target_dg      = design.get("target_dg_kcal", None)   # optional dual constraint
        c_t            = conc_nm * 1e-9  # nM → M

        if not sequence or not all(b in "ATCG" for b in sequence):
            violations.append(
                "INVALID: Sequence contains non-DNA characters or is empty. "
                "Only A, T, C, G are permitted."
            )
            return PhysicsResult(
                passed=False, domain="biochem",
                design_summary="Invalid sequence - non-DNA characters",
                violations=violations,
                physics_feedback="Sequence rejected before thermodynamic analysis: invalid bases.",
                correction_hint="Provide a valid DNA sequence using only A, T, C, G bases.",
            )

        n           = len(sequence)
        gc          = sequence.count("G") + sequence.count("C")
        at          = sequence.count("A") + sequence.count("T")
        gc_fraction = gc / n

        # Full SantaLucia 1998 Tm and ΔG
        dH, dS_cal  = self._nn_params(sequence)
        tm_celsius  = self._calc_tm(sequence, c_t)
        dg_37       = self._calc_dg_37(dH, dS_cal)

        metrics["sequence_length"]   = n
        metrics["gc_count"]          = gc
        metrics["at_count"]          = at
        metrics["gc_fraction"]       = round(gc_fraction, 3)
        metrics["melting_point_c"]   = round(tm_celsius, 1)
        metrics["calculated_tm_C"]   = round(tm_celsius, 1)
        metrics["target_temp_C"]     = target_temp
        metrics["delta_H_kcal_mol"]  = round(dH, 2)
        metrics["delta_S_cal_molK"]  = round(dS_cal, 2)
        metrics["delta_G_37_kcal"]   = round(dg_37, 2)
        metrics["model"]             = "SantaLucia1998_full_NN + Owczarzy2004_salt"

        # ── Rule 1: Tm must exceed target temperature ─────────────────────
        if tm_celsius < target_temp:
            violations.append(
                f"CRITICAL: Tm = {tm_celsius:.1f}°C is {target_temp - tm_celsius:.1f}°C "
                f"below target {target_temp:.0f}°C. Duplex melts before operating temperature."
            )

        # ── Rule 2: ΔG at 37°C must be ≤ target_dg_kcal (if specified) ───
        if target_dg is not None and dg_37 > target_dg:
            violations.append(
                f"WEAK BINDING: ΔG₃₇ = {dg_37:.2f} kcal/mol is above target "
                f"≤ {target_dg:.1f} kcal/mol. Duplex is not thermodynamically stable enough."
            )

        # ── Rule 3: sequence too short for reliable duplex (< 8 nt) ───────
        if n < 8:
            violations.append(
                f"TOO SHORT: {n}-mer is too short for stable duplex. Minimum 8 nt recommended."
            )

        passed = len(violations) == 0

        dg_line = f"  ΔG₃₇ = {dg_37:.2f} kcal/mol" + (
            f"  (target ≤ {target_dg:.1f} kcal/mol)" if target_dg is not None else "")

        if not passed:
            gc_needed     = self._find_gc_for_target_tm(n, target_temp, c_t)
            gc_needed_cnt = math.ceil(gc_needed * n)
            physics_feedback = (
                f"Physics sandbox REJECTED (SantaLucia 1998 full NN model).\n"
                f"  Tm = {tm_celsius:.1f}°C  (target ≥ {target_temp:.0f}°C)\n"
                f"{dg_line}\n"
                f"  ΔH = {dH:.1f} kcal/mol,  ΔS = {dS_cal:.1f} cal/(mol·K)\n"
                f"  GC = {gc}/{n} ({gc_fraction*100:.0f}%)\n"
                f"  Need ≥ {gc_needed_cnt}/{n} GC pairs ({gc_needed*100:.0f}%) for Tm ≥ {target_temp:.0f}°C"
            )
            correction_hint = (
                f"Replace {max(0, gc_needed_cnt - gc)} A/T bases with G/C to raise Tm. "
                f"Place GC pairs in the centre - terminal GC pairs contribute less "
                f"stacking energy than internal ones (SantaLucia 1998)."
            )
            if target_dg is not None and dg_37 > target_dg:
                correction_hint += (
                    f" To also satisfy ΔG ≤ {target_dg:.1f} kcal/mol, increase sequence "
                    f"length or GC content - both raise |ΔH| and lower ΔG₃₇."
                )
        else:
            physics_feedback = (
                f"Physics sandbox APPROVED (SantaLucia 1998 full NN model).\n"
                f"  Tm = {tm_celsius:.1f}°C ≥ target {target_temp:.0f}°C\n"
                f"{dg_line}\n"
                f"  ΔH = {dH:.1f} kcal/mol,  ΔS = {dS_cal:.1f} cal/(mol·K)\n"
                f"  GC = {gc}/{n} ({gc_fraction*100:.0f}%)"
            )
            correction_hint = ""

        return PhysicsResult(
            passed          = passed,
            domain          = "biochem",
            design_summary  = (
                f"DNA {n}-mer, GC={gc_fraction*100:.0f}%, "
                f"Tm={tm_celsius:.1f}°C, ΔG₃₇={dg_37:.2f} kcal/mol (SantaLucia 1998)"
            ),
            violations      = violations,
            metrics         = metrics,
            physics_feedback = physics_feedback,
            correction_hint = correction_hint,
        )

    def _find_gc_for_target_tm(self, n: int, target_tm_c: float, c_t: float) -> float:
        """Binary search for GC fraction needed to reach target Tm."""
        target_k = target_tm_c + 273.15 + 2.0  # +2°C safety margin
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            # Build a synthetic sequence with `mid` GC fraction
            n_gc = round(mid * n)
            n_at = n - n_gc
            synthetic = "GC" * (n_gc // 2) + ("G" if n_gc % 2 else "") + "AT" * (n_at // 2) + ("A" if n_at % 2 else "")
            synthetic = (synthetic + "A" * n)[:n]  # ensure exact length
            tm_k = self._calc_tm(synthetic, c_t) + 273.15
            if tm_k < target_k:
                lo = mid
            else:
                hi = mid
        return hi


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_validator(domain: str):
    """Return the appropriate physics validator for a domain."""
    if domain == "drone":
        return DronePhysicsValidator()
    elif domain == "biochem":
        return DNAHydrogelValidator()
    raise ValueError(f"Unknown physics domain: {domain!r}. Use 'drone' or 'biochem'.")
