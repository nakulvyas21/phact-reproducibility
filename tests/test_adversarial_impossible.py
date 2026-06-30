"""
Adversarial goals are genuinely impossible (engine-verified)
=============================================================

The paper claims PHACT correctly *refuses* physically impossible goals. That
claim only holds if the adversarial goals are, in fact, impossible. These tests
verify, deterministically and without any language model, that the
over-constrained adversarial parameter sets admit no valid design: the physics
engine rejects every one of them.

This is the ground truth behind the impossible-goal evaluation. A reviewer can
confirm that the "impossible" goals are not merely hard but actually violate
physical law.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phact_physics import get_validator
from phact_physics.domains.electrical import check_rc_filter
from phact_physics.domains.drug_formulation import check_colloidal_stability


# ---------------------------------------------------------------------------
# BIOCHEM - sequences too short to be stable at the requested temperature
# ---------------------------------------------------------------------------

# (sequence, requested stable temp C) - all far above achievable Tm
BIOCHEM_IMPOSSIBLE = [
    ("ATCGAT",   80.0),   # 6-mer asked to be stable at 80 C
    ("ATCGA",    90.0),   # 5-mer at 90 C
    ("ATCG",     75.0),   # 4-mer at 75 C
    ("ATCGATC",  85.0),   # 7-mer at 85 C
    ("ATCGT",    70.0),   # 5-mer at 70 C
]


@pytest.mark.parametrize("sequence,temp", BIOCHEM_IMPOSSIBLE)
def test_biochem_impossible_is_rejected(sequence, temp):
    """A short oligo cannot be stable at the requested high temperature."""
    v = get_validator("biochem")
    result = v.validate({
        "sequence":       sequence,
        "target_temp_c":  temp,
        "strand_conc_nm": 250.0,
    })
    assert not result.passed, (
        f"{sequence} unexpectedly certified stable at {temp} C; "
        f"engine must reject this as thermodynamically impossible"
    )


# ---------------------------------------------------------------------------
# ELECTRICAL - fixed R and C that cannot produce the requested cutoff
# f_c = 1/(2 pi R C) is fully determined once R and C are fixed.
# ---------------------------------------------------------------------------

# (R ohm, C uF, requested cutoff Hz) - the fixed components give a wildly
# different f_c, and the components may not be changed.
ELECTRICAL_IMPOSSIBLE = [
    (1.0,      1e-6,   1000.0),    # R=1, C=1pF -> ~159 GHz, asked for 1 kHz
    (1e6,      1.0,    1e6),       # R=1M, C=1F -> ~159 nHz, asked for 1 MHz
    (1.0,      1e-3,   60.0),      # R=1, C=1nF -> ~159 MHz, asked for 60 Hz
    (1e9,      1e-6,   10000.0),   # R=1G, C=1pF -> ~159 Hz, asked for 10 kHz
    (1.0,      1.0,    100.0),     # R=1, C=1uF -> ~159 kHz, asked for 100 Hz
]


@pytest.mark.parametrize("R,C_uf,fc_requested", ELECTRICAL_IMPOSSIBLE)
def test_electrical_impossible_is_rejected(R, C_uf, fc_requested):
    """Fixed R,C cannot meet the requested cutoff; engine must flag mismatch."""
    r = check_rc_filter(resistance_ohm=R, capacitance_uf=C_uf,
                        target_cutoff_hz=fc_requested)
    assert not r.passed, (
        f"R={R} C={C_uf}uF unexpectedly certified for f_c={fc_requested} Hz; "
        f"the fixed components give {r.metrics['actual_cutoff_hz']:.3e} Hz"
    )


# ---------------------------------------------------------------------------
# DRUG - over-constrained formulations with no electrostatic barrier
# ---------------------------------------------------------------------------

# (radius nm, zeta mV, salt mM, material) - near-zero zeta at high salt
DRUG_IMPOSSIBLE = [
    (250, -15, 1000, "gold"),
    (500,  -8,  500, "polystyrene"),
    (400,  -5,  800, "plga"),
    (300,  -3,  600, "silica"),
    (350,  -2,  700, "lipid"),
]


@pytest.mark.parametrize("radius,zeta,salt,material", DRUG_IMPOSSIBLE)
def test_drug_impossible_is_rejected(radius, zeta, salt, material):
    """No electrostatic barrier at high salt + near-zero zeta -> unstable."""
    r = check_colloidal_stability(
        particle_radius_nm=radius, zeta_potential_mv=zeta,
        salt_concentration_mm=salt, particle_material=material,
    )
    assert not r.passed, (
        f"{material} r={radius}nm zeta={zeta}mV at {salt}mM unexpectedly "
        f"certified stable; barrier={r.metrics.get('energy_barrier_kT')} kT"
    )


def test_determinism_same_input_same_output():
    """The engines are deterministic: identical input -> identical output."""
    v = get_validator("biochem")
    design = {"sequence": "ATCGATCGATCGATCGATCG",
              "target_temp_c": 37.0, "strand_conc_nm": 250.0}
    tm_runs = []
    for _ in range(5):
        m = v.validate(design).metrics
        tm_runs.append(m.get("calculated_tm_C") or m.get("melting_point_c"))
    assert len(set(tm_runs)) == 1, f"non-deterministic Tm: {tm_runs}"
