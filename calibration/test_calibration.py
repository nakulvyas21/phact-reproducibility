"""
Engine calibration against peer-reviewed literature
====================================================

These tests are the scientific validation of the PHACT physics engines. Each one
checks that a deterministic engine reproduces an independently published value
within a stated tolerance. They require no credentials and no network access,
and they are fully deterministic: the same input always yields the same output.

This is the basis of the paper's "Methods: engine calibration" paragraph. A
reviewer can run `pytest calibration/` and confirm that every engine matches
the literature it claims to implement.

References
----------
    SantaLucia (1998)   PNAS 95:1460 - DNA nearest-neighbour thermodynamics
    Owczarzy  (2004)    Biochemistry 43:3537 - salt correction
    Abbott et al (2016) PRL 116:061102 - GW150914 chirp mass (LIGO/Virgo)
    Abbott et al (2017) PRL 119:161101 - GW170817 chirp mass (BNS)
    Horowitz & Hill (2015) Art of Electronics 3rd ed. - RC filter f=1/(2piRC)
    Israelachvili (2011) Intermolecular & Surface Forces 3rd ed. - DLVO theory
"""

import math
import sys
from pathlib import Path

import pytest

# Make the package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phact_physics import get_validator
from phact_physics.domains.astrophysics import check_chirp_mass, check_isco_frequency
from phact_physics.domains.electrical import check_rc_filter
from phact_physics.domains.drug_formulation import check_colloidal_stability


# ---------------------------------------------------------------------------
# BIOCHEM - DNA melting temperature (SantaLucia 1998 + Owczarzy 2004)
# Reference Tm values computed from the full 10-parameter nearest-neighbour
# model at 250 nM strand concentration, 50 mM Na+.
# ---------------------------------------------------------------------------

BIOCHEM_REFERENCE = [
    # (sequence, expected Tm in C, tolerance in C)
    ("ATCGATCGATCGATCGATCG", 53.1, 2.0),   # alternating, 50% GC, 20-mer
    ("GCGCGCGCGCGCGCGCGC",   80.5, 2.0),   # 100% GC, 18-mer
    ("AATTTAATTTAATTT",      20.4, 2.5),   # 0% GC, 15-mer (low-Tm regime)
    ("GCTAGCTAGCTAGCTAGC",   50.8, 2.0),   # 56% GC, 18-mer
    ("GCGCATGCGCATGCGCAT",   63.7, 2.0),   # 67% GC, 18-mer
]


@pytest.mark.parametrize("sequence,expected_tm,tol", BIOCHEM_REFERENCE)
def test_dna_melting_temperature(sequence, expected_tm, tol):
    """DNA Tm engine reproduces SantaLucia 1998 NN-model values within tol."""
    v = get_validator("biochem")
    result = v.validate({
        "sequence":       sequence,
        "target_temp_c":  37.0,
        "strand_conc_nm": 250.0,
    })
    tm = result.metrics.get("calculated_tm_C")
    if tm is None:
        tm = result.metrics.get("melting_point_c")
    assert tm is not None, f"engine returned no Tm for {sequence}"
    assert abs(tm - expected_tm) <= tol, (
        f"{sequence}: engine Tm {tm:.1f} C deviates from "
        f"SantaLucia reference {expected_tm} C by more than {tol} C"
    )


def test_dna_tm_monotonic_in_gc():
    """Higher GC content must give higher Tm - a hard thermodynamic ordering."""
    v = get_validator("biochem")

    def tm_of(seq):
        m = v.validate({"sequence": seq, "target_temp_c": 37.0,
                        "strand_conc_nm": 250.0}).metrics
        return m.get("calculated_tm_C") or m.get("melting_point_c")

    tm_low  = tm_of("AATTTAATTTAATTT")        # 0% GC
    tm_mid  = tm_of("GCTAGCTAGCTAGCTAGC")     # 56% GC
    tm_high = tm_of("GCGCGCGCGCGCGCGCGC")     # 100% GC
    assert tm_low < tm_mid < tm_high


# ---------------------------------------------------------------------------
# ASTROPHYSICS - chirp mass and ISCO frequency (LIGO/Virgo GWTC)
# ---------------------------------------------------------------------------

# (event, m1, m2, published chirp mass Msun, tolerance Msun)
GW_EVENTS = [
    ("GW150914", 35.6, 30.6, 28.6,  1.0),   # Abbott et al 2016 PRL 116:061102
    ("GW170817", 1.46, 1.27, 1.186, 0.05),  # Abbott et al 2017 PRL 119:161101 (BNS)
    ("GW170104", 30.5, 25.0, 24.1,  1.0),   # Abbott et al 2017 PRL 118:221101
]


@pytest.mark.parametrize("event,m1,m2,mc_pub,tol", GW_EVENTS)
def test_chirp_mass_matches_catalog(event, m1, m2, mc_pub, tol):
    """Engine chirp mass reproduces the published GWTC value within tol."""
    r = check_chirp_mass(m1_msun=m1, m2_msun=m2,
                         chirp_mass_msun=mc_pub)
    mc_engine = r.metrics["chirp_mass_msun"]
    assert abs(mc_engine - mc_pub) <= tol, (
        f"{event}: engine chirp mass {mc_engine:.3f} deviates from "
        f"catalog {mc_pub} by more than {tol} Msun"
    )


def test_chirp_mass_formula_exact():
    """Chirp mass must equal (m1 m2)^{3/5} / (m1+m2)^{1/5} exactly."""
    m1, m2 = 35.6, 30.6
    expected = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    r = check_chirp_mass(m1_msun=m1, m2_msun=m2,
                         chirp_mass_msun=expected)
    assert math.isclose(r.metrics["chirp_mass_msun"], expected, rel_tol=1e-3)


def test_isco_frequency_scaling():
    """ISCO GW frequency scales as ~1/M (Schwarzschild 6GM/c^2 orbit)."""
    r1 = check_isco_frequency(total_mass_msun=66.2,  fisco_hz=66.0)
    r2 = check_isco_frequency(total_mass_msun=132.4, fisco_hz=33.0)
    f1 = r1.metrics["f_isco_gw_hz"]
    f2 = r2.metrics["f_isco_gw_hz"]
    # Doubling total mass halves the ISCO frequency.
    assert math.isclose(f1 / f2, 2.0, rel_tol=0.02)


# ---------------------------------------------------------------------------
# ELECTRICAL - RC filter cutoff f_c = 1/(2 pi R C)
# ---------------------------------------------------------------------------

# (R ohm, C uF, expected cutoff Hz, tolerance Hz)
RC_REFERENCE = [
    (1000.0,   0.159,    1000.0, 5.0),    # canonical 1 kHz
    (10000.0,  0.00159, 10000.0, 50.0),   # 10 kHz
    (1000.0,   1.59,      100.0, 1.0),    # 100 Hz (high-pass AC coupling)
]


@pytest.mark.parametrize("R,C_uf,fc_expected,tol", RC_REFERENCE)
def test_rc_cutoff_frequency(R, C_uf, fc_expected, tol):
    """RC filter engine computes f_c = 1/(2 pi R C) to within tol."""
    r = check_rc_filter(resistance_ohm=R, capacitance_uf=C_uf,
                        target_cutoff_hz=fc_expected)
    fc = r.metrics["actual_cutoff_hz"]
    assert abs(fc - fc_expected) <= tol, (
        f"R={R} C={C_uf}uF: engine f_c {fc:.2f} Hz deviates from "
        f"1/(2piRC) = {fc_expected} Hz by more than {tol} Hz"
    )


def test_rc_cutoff_closed_form():
    """Engine cutoff must equal the closed-form 1/(2 pi R C) exactly."""
    R, C_uf = 4700.0, 0.022
    C = C_uf * 1e-6
    expected = 1.0 / (2.0 * math.pi * R * C)
    r = check_rc_filter(resistance_ohm=R, capacitance_uf=C_uf,
                        target_cutoff_hz=expected)
    # Engine reports actual_cutoff_hz rounded to 2 decimals; compare at that
    # precision. The underlying computation is exact (1/(2 pi R C)).
    assert math.isclose(r.metrics["actual_cutoff_hz"], expected, abs_tol=0.01)


# ---------------------------------------------------------------------------
# DRUG FORMULATION - DLVO colloidal stability (Israelachvili)
# ---------------------------------------------------------------------------

def test_dlvo_stable_at_low_salt_high_zeta():
    """High zeta potential, low salt -> a large barrier -> stable."""
    r = check_colloidal_stability(
        particle_radius_nm=75, zeta_potential_mv=-45,
        salt_concentration_mm=10, particle_material="silica",
    )
    assert r.passed
    assert r.metrics["energy_barrier_kT"] > 15  # Israelachvili stability threshold


def test_dlvo_unstable_at_high_salt_low_zeta():
    """Near-zero zeta, high salt -> screened EDL -> no barrier -> unstable."""
    r = check_colloidal_stability(
        particle_radius_nm=250, zeta_potential_mv=-5,
        salt_concentration_mm=1000, particle_material="gold",
    )
    assert not r.passed
    assert r.metrics["energy_barrier_kT"] < 5


def test_dlvo_debye_length_scaling():
    """Debye length scales as 1/sqrt(salt) - a hard electrostatics law."""
    r_low  = check_colloidal_stability(
        particle_radius_nm=100, zeta_potential_mv=-30,
        salt_concentration_mm=10, particle_material="plga")
    r_high = check_colloidal_stability(
        particle_radius_nm=100, zeta_potential_mv=-30,
        salt_concentration_mm=1000, particle_material="plga")
    debye_low  = r_low.metrics["debye_length_nm"]
    debye_high = r_high.metrics["debye_length_nm"]
    # 100x more salt -> 10x shorter Debye length.
    assert math.isclose(debye_low / debye_high, 10.0, rel_tol=0.05)
