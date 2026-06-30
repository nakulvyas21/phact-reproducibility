"""
phact_physics.agent - thin pure-physics validation wrappers
===========================================================

These two functions wrap the deterministic DronePhysicsValidator and
DNAHydrogelValidator from phact_physics.physics_engine. They contain NO language
model and NO network access; they exist only to give the drone and biochem
domains a uniform tool-style entry point in the domain registry. (The full agent
stack that drives these tools with a language model is not part of this
reproducibility package; see the paper's Methods.)
"""

import logging
from phact_physics.physics_engine import DronePhysicsValidator, DNAHydrogelValidator

logger = logging.getLogger(__name__)


def validate_drone_design(
    path_type: str,
    crosswind_knots: float,
    canyon_width_m: float,
    speed_ms: float,
    bank_angle_deg: float,
    reasoning: str,
) -> dict:
    """
    Validates a drone flight trajectory against real aerodynamic physics laws.

    Call this tool EVERY TIME you propose a drone flight path design.
    The physics sandbox will tell you exactly what fails and how to fix it.
    You MUST keep calling this tool and adjusting your design until it returns passed=True.

    Args:
        path_type: Flight path shape. One of: 'straight', 'curved', 'banked_curve'
        crosswind_knots: Lateral wind speed in knots (must match the goal exactly)
        canyon_width_m: Width of the canyon gap in metres
        speed_ms: Drone airspeed in metres per second
        bank_angle_deg: Bank/roll angle in degrees (0 = straight, positive = counter-steering into wind)
        reasoning: One sentence explaining why you chose these values

    Returns:
        Physics validation result with passed status, violations, metrics, and correction hints.
    """
    validator = DronePhysicsValidator()
    result = validator.validate({
        "path_type":       str(path_type),
        "crosswind_knots": float(crosswind_knots) if not isinstance(crosswind_knots, float) else crosswind_knots,
        "canyon_width_m":  float(canyon_width_m) if not isinstance(canyon_width_m, float) else canyon_width_m,
        "speed_ms":        float(speed_ms) if not isinstance(speed_ms, float) else speed_ms,
        "bank_angle_deg":  float(bank_angle_deg) if not isinstance(bank_angle_deg, float) else bank_angle_deg,
        "reasoning":       reasoning,
    })

    return {
        "passed":           result.passed,
        "design_summary":   result.design_summary,
        "violations":       result.violations,
        "metrics":          result.metrics,
        "physics_feedback":  result.physics_feedback,
        "correction_hint":  result.correction_hint,
        "proposed_design": {
            "path_type":       path_type,
            "crosswind_knots": crosswind_knots,
            "canyon_width_m":  canyon_width_m,
            "speed_ms":        speed_ms,
            "bank_angle_deg":  bank_angle_deg,
            "reasoning":       reasoning,
        }
    }


def validate_biochem_design(
    sequence: str,
    target_temp_c: float,
    strand_conc_nm: float,
    reasoning: str,
) -> dict:
    """
    Validates a DNA hydrogel sequence against real thermodynamic physics laws.

    Call this tool EVERY TIME you propose a DNA sequence.
    The physics sandbox calculates the actual melting point using nearest-neighbour
    thermodynamics (SantaLucia 1998). You MUST keep calling this tool and mutating
    your sequence until it returns passed=True.

    Args:
        sequence: DNA sequence string using ONLY the characters A, T, C, G
        target_temp_c: Target operating temperature in Celsius (e.g. 37.0 for human body)
        strand_conc_nm: Strand concentration in nanomolar (default 250.0)
        reasoning: One sentence explaining your sequence design choice

    Returns:
        Physics validation result with passed status, calculated Tm, violations, and mutation hints.
    """
    validator = DNAHydrogelValidator()
    result = validator.validate({
        "sequence":       str(sequence),
        "target_temp_c":  float(target_temp_c) if not isinstance(target_temp_c, float) else target_temp_c,
        "strand_conc_nm": float(strand_conc_nm) if not isinstance(strand_conc_nm, float) else strand_conc_nm,
        "reasoning":      reasoning,
    })

    return {
        "passed":           result.passed,
        "design_summary":   result.design_summary,
        "violations":       result.violations,
        "metrics":          result.metrics,
        "physics_feedback":  result.physics_feedback,
        "correction_hint":  result.correction_hint,
        "proposed_design": {
            "sequence":       sequence,
            "target_temp_c":  target_temp_c,
            "strand_conc_nm": strand_conc_nm,
            "reasoning":      reasoning,
        }
    }

