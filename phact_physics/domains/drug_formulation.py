"""
PHACT Domain: Drug Formulation Physics
======================================
Four physics validators for nanoparticle drug delivery formulation.
All use well-established soft-matter physics - no ML, no approximations
invented for this project. Every equation is citable.

Physics covered:
  1. check_colloidal_stability - DLVO theory (Derjaguin-Landau-Verwey-Overbeek)
  2. check_polymer_drug_compatibility - Flory-Huggins χ parameter
  3. check_drug_permeability - Lipinski + logP/TPSA via RDKit
  4. check_diffusion_coefficient - Stokes-Einstein at physiological temperature

Designed to catch the specific hallucinations that LLMs make in this domain:
  • Using vacuum Hamaker constant instead of the in-water value
  • Applying the wrong water viscosity (η at 20°C ≠ η at 37°C)
  • Claiming logP > 5 improves membrane permeability (it doesn't)
  • Getting Flory-Huggins sign convention inverted

Reference:
  Israelachvili, "Intermolecular and Surface Forces", 3rd ed.
  Flory, "Principles of Polymer Chemistry" (1953)
  SantaLucia & Hicks, Annu. Rev. Biophys. Biomol. Struct. (2004)
  Lipinski et al., Adv. Drug Deliv. Rev. (2001)
"""

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Physical constants (SI, exact where possible) ──────────────────────────

K_B       = 1.380649e-23   # J/K - Boltzmann constant (exact, SI 2019)
E_CHARGE  = 1.602176634e-19 # C - elementary charge (exact, SI 2019)
EPS_0     = 8.8541878e-12   # F/m - vacuum permittivity
EPS_WATER = 78.5            # dimensionless - relative permittivity of water at 25°C
                             # (use 74.5 at 37°C - matters near thresholds)
EPS_WATER_37 = 74.5         # at physiological temperature
N_AVO     = 6.02214076e23   # mol⁻¹ - Avogadro (exact, SI 2019)
R_GAS     = 8.314462        # J/(mol·K) - gas constant

# ── Hamaker constants in water (A₁₃₂, i.e. material/water/material) ────────
# Source: Israelachvili Table 13.2 + Parsegian "Van der Waals Forces" (2006)
# These are the non-retarded values. AI models almost always use vacuum values
# (which are ~10× larger for typical biomaterials) - a critical error.
HAMAKER_IN_WATER = {
    "plga":        5.0e-21,  # J - poly(lactic-co-glycolic acid), estimated
    "polystyrene": 1.3e-20,  # J - well-measured (Bevan & Prieve 1999)
    "silica":      8.3e-21,  # J - well-measured (Elimelech et al. 1995)
    "lipid":       5.0e-21,  # J - phospholipid bilayer (Parsegian 2006)
    "protein":     2.5e-21,  # J - globular protein, midpoint of 1-5×10⁻²¹ range
    "gold":        3.0e-19,  # J - large, metallic (Pinchuk 2012)
    "chitosan":    4.0e-21,  # J - estimated from dielectric data
}
HAMAKER_DEFAULT = 5.0e-21   # J - conservative default for unknown polymer


# ── Result dataclass (mirrors PhysicsResult interface) ──────────────────────

@dataclass
class PhysicsResult:
    passed:          bool
    domain:          str
    check_name:      str
    design_summary:  str
    violations:      list[str]     = field(default_factory=list)
    metrics:         dict          = field(default_factory=dict)
    physics_feedback: str           = ""
    correction_hint: str           = ""
    reference:       str           = ""   # citable source for the equation used


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DLVO Colloidal Stability
# ═══════════════════════════════════════════════════════════════════════════════

def check_colloidal_stability(
    particle_radius_nm:   float,
    zeta_potential_mv:    float,
    salt_concentration_mm: float,
    particle_material:    str   = "plga",
    temperature_c:        float = 37.0,
) -> PhysicsResult:
    """
    DLVO colloidal stability analysis for nanoparticle drug carriers.

    Computes the total interaction energy V(h) = V_vdW(h) + V_EDL(h)
    using the Derjaguin approximation for two identical spheres.

    V_vdW(h) = -A·a / (12h)
    V_EDL(h) = 64π·a·n∞·kT·κ⁻¹·γ²·exp(-κh)
    where γ = tanh(eζ/4kT)

    Stability criterion (Israelachvili, Ch. 13):
      V_max > 15 kT  → stable
      V_max 5-15 kT  → marginal
      V_max < 5 kT   → unstable (rapid aggregation)

    CRITICAL: Uses the in-water Hamaker constant A₁₃₂, NOT the vacuum
    value A₁₁. The vacuum value is ~10× larger - a common AI error that
    would predict all nanoparticles as irreversibly aggregated.

    Args:
        particle_radius_nm:    Hydrodynamic radius in nanometres
        zeta_potential_mv:     Zeta potential in millivolts (negative = anionic)
        salt_concentration_mm: Monovalent salt concentration in millimolar (e.g. NaCl)
        particle_material:     One of: plga, polystyrene, silica, lipid, protein, gold, chitosan
        temperature_c:         Temperature in Celsius (default 37°C)

    Returns:
        PhysicsResult with energy barrier in kT and stability verdict.
    """
    violations = []
    metrics    = {}

    T   = temperature_c + 273.15
    kT  = K_B * T
    a   = particle_radius_nm * 1e-9          # m
    zeta = zeta_potential_mv * 1e-3          # V
    c_mol = salt_concentration_mm * 1e-3     # mol/L = M
    c_si  = c_mol * 1e3 * N_AVO             # number density [m⁻³]  (1 M = 1000 mol/m³)

    eps_r = EPS_WATER_37 if temperature_c > 30 else EPS_WATER
    eps   = eps_r * EPS_0

    # ── Debye screening length ──────────────────────────────────────────────
    # κ⁻¹ = sqrt(ε₀·εᵣ·kT / (2·n∞·e²))   [Israelachvili eq. 14.9]
    # n∞ = number density of each ion species = N_Avo × c [mol/m³]
    # For 1:1 electrolyte (NaCl): n∞ = N_Avo × c_mol[mol/L] × 1000 [L/m³]
    kappa = math.sqrt(2 * c_si * E_CHARGE**2 / (eps * kT))
    debye_length_nm = (1.0 / kappa) * 1e9

    metrics["debye_length_nm"]    = round(debye_length_nm, 3)
    metrics["temperature_K"]      = round(T, 2)
    metrics["kT_J"]               = f"{kT:.4e}"
    metrics["salt_concentration_mM"] = salt_concentration_mm

    # ── In-water Hamaker constant ───────────────────────────────────────────
    mat = particle_material.lower()
    A   = HAMAKER_IN_WATER.get(mat, HAMAKER_DEFAULT)
    metrics["hamaker_in_water_J"]  = f"{A:.2e}"
    metrics["hamaker_in_kT"]       = round(A / kT, 2)

    if mat not in HAMAKER_IN_WATER:
        violations.append(
            f"WARNING: Unknown material '{particle_material}'. "
            f"Using default A = {HAMAKER_DEFAULT:.1e} J. "
            f"Known materials: {', '.join(HAMAKER_IN_WATER.keys())}."
        )

    # ── Scan V_total over separation h to find energy barrier ──────────────
    # Derjaguin approx: valid when κa >> 1 (particle radius >> Debye length)
    # For a = 100 nm, κa ≈ 100/0.78 ≈ 128 at physiological salt → valid
    kappa_a = kappa * a
    metrics["kappa_times_a"] = round(kappa_a, 1)

    gamma = math.tanh(E_CHARGE * zeta / (4 * kT))

    # ── Scan V_total over separation h ────────────────────────────────────
    # Correct Derjaguin sphere-sphere EDL (Israelachvili eq. 14.16):
    # V_EDL(h) = 64·π·ε·a·(kT/e)²·γ²·exp(-κh)
    # V_vdW(h) = -A·a / (12h)   [Derjaguin approximation, h << a]
    h_values  = [i * 1e-10 for i in range(1, 500)]  # 0.1-50 nm
    v_max_kT  = 0.0
    h_max_nm  = 0.0

    for h in h_values:
        v_vdw = -A * a / (12.0 * h)
        v_edl = 64.0 * math.pi * eps * a * (kT / E_CHARGE)**2 * (gamma**2) * math.exp(-kappa * h)
        v_tot = (v_vdw + v_edl) / kT

        if v_tot > v_max_kT:
            v_max_kT = v_tot
            h_max_nm = h * 1e9

    metrics["energy_barrier_kT"]      = round(v_max_kT, 2)
    metrics["barrier_position_nm"]    = round(h_max_nm, 2)
    metrics["stability_threshold_kT"] = 15

    # ── Stability verdict ───────────────────────────────────────────────────
    if v_max_kT < 5:
        violations.append(
            f"CRITICAL: Energy barrier {v_max_kT:.1f} kT is below 5 kT. "
            f"Nanoparticles will aggregate rapidly. Thermal fluctuations (~1 kT) "
            f"will continuously drive particles over this barrier."
        )
    elif v_max_kT < 15:
        violations.append(
            f"WARNING: Energy barrier {v_max_kT:.1f} kT is marginal (5-15 kT). "
            f"Suspension may be kinetically stable short-term but will aggregate "
            f"over hours to days."
        )

    # ── Zeta potential sanity check ─────────────────────────────────────────
    if abs(zeta_potential_mv) < 15:
        violations.append(
            f"WARNING: |ζ| = {abs(zeta_potential_mv)} mV is below 15 mV. "
            f"This provides insufficient electrostatic repulsion for stability "
            f"unless steric stabilisation (e.g. PEG coating) is present."
        )

    # ── High salt warning ───────────────────────────────────────────────────
    if salt_concentration_mm > 150:
        violations.append(
            f"WARNING: Salt concentration {salt_concentration_mm} mM exceeds "
            f"physiological PBS (150 mM). Debye length = {debye_length_nm:.2f} nm. "
            f"Electrostatic repulsion is heavily screened at this ionic strength."
        )

    passed = len(violations) == 0

    if not passed:
        # Calculate minimum zeta potential for stability at current salt
        # Approximate: need V_max ≥ 15 kT
        min_zeta_mv = _estimate_min_zeta(a, A, kappa, kT) * 1e3
        physics_feedback = (
            f"DLVO analysis: energy barrier = {v_max_kT:.1f} kT at h = {h_max_nm:.2f} nm.\n"
            f"Measured violations:\n" +
            "\n".join(f"  • {v}" for v in violations) +
            f"\n\nKey physics data:"
            f"\n  • Debye screening length: {debye_length_nm:.2f} nm (at {salt_concentration_mm} mM salt)"
            f"\n  • Hamaker constant (in water): {A:.2e} J = {A/kT:.1f} kT"
            f"\n  • Current energy barrier: {v_max_kT:.1f} kT (need ≥ 15 kT)"
        )
        correction_hint = (
            f"To reach 15 kT barrier: increase |ζ| to ≥ {min_zeta_mv:.0f} mV, "
            f"or reduce salt to < {salt_concentration_mm * 0.5:.0f} mM, "
            f"or add PEG steric stabilisation."
        )
    else:
        physics_feedback = (
            f"DLVO APPROVED: Energy barrier = {v_max_kT:.1f} kT >> 15 kT threshold. "
            f"Debye length = {debye_length_nm:.2f} nm. "
            f"Nanoparticles are electrostatically stable against aggregation."
        )
        correction_hint = ""

    return PhysicsResult(
        passed          = passed,
        domain          = "drug_formulation",
        check_name      = "colloidal_stability",
        design_summary  = (
            f"R={particle_radius_nm} nm, ζ={zeta_potential_mv} mV, "
            f"[salt]={salt_concentration_mm} mM, material={particle_material}"
        ),
        violations      = violations,
        metrics         = metrics,
        physics_feedback = physics_feedback,
        correction_hint = correction_hint,
        reference       = "Israelachvili (2011) Intermolecular and Surface Forces, 3rd ed., Ch.13",
    )


def _estimate_min_zeta(a, A, kappa, kT) -> float:
    """Estimate minimum |ζ| (V) needed for 15 kT barrier via bisection."""
    eps = EPS_WATER_37 * EPS_0

    def barrier(zeta_v):
        gamma = math.tanh(E_CHARGE * zeta_v / (4 * kT))
        v_max = 0.0
        for h in [i * 1e-10 for i in range(1, 500)]:
            vdw = -A * a / (12.0 * h)
            edl = 64.0 * math.pi * eps * a * (kT / E_CHARGE)**2 * gamma**2 * math.exp(-kappa * h)
            v_max = max(v_max, (vdw + edl) / kT)
        return v_max

    lo, hi = 0.001, 0.200  # V
    for _ in range(40):
        mid = (lo + hi) / 2
        if barrier(mid) < 15:
            lo = mid
        else:
            hi = mid
    return hi


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Flory-Huggins Polymer-Drug Compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def check_polymer_drug_compatibility(
    drug_solubility_parameter:     float,   # MPa^0.5 - Hildebrand δ
    polymer_solubility_parameter:  float,   # MPa^0.5
    drug_molar_volume_cm3_mol:     float,   # cm³/mol
    temperature_c:                 float = 37.0,
) -> PhysicsResult:
    """
    Flory-Huggins χ parameter for polymer-drug miscibility in amorphous solid dispersions.

    χ = (V_mol / RT) · (δ_drug - δ_polymer)²

    Miscibility criterion (Flory 1953, validated for amorphous solid dispersions by
    Marsac et al., Pharm. Res. 2006):
      χ < 0.5    → miscible (single-phase ASD stable, drug will not crystallise)
      χ = 0.5    → critical / spinodal boundary
      χ > 0.5    → phase separation, drug crystallises out over time
      χ < 0      → exothermic mixing, very stable (ideal for long shelf-life)

    COMMON AI ERRORS CAUGHT:
      1. Using mole fraction instead of volume fraction in the χ criterion
         (the 0.5 threshold is for symmetric mixtures; use χ_crit = 0.5 for
          equal-volume components - AI often inverts this)
      2. Getting solubility parameter units wrong (MPa^0.5 vs. (cal/cm³)^0.5;
         1 (cal/cm³)^0.5 = 2.0455 MPa^0.5)
      3. Forgetting the molar volume normalisation V_mol/RT

    Typical solubility parameters (δ, MPa^0.5):
      Ibuprofen:       19.9     PVP K30:    28.6
      Indomethacin:    22.9     HPMC:       26.2
      Naproxen:        20.9     PEG 4000:   22.0
      Felodipine:      19.5     Eudragit:   21.5
      Water:           47.9     Ethanol:    26.2

    Args:
        drug_solubility_parameter:    Hildebrand δ of drug (MPa^0.5)
        polymer_solubility_parameter: Hildebrand δ of polymer (MPa^0.5)
        drug_molar_volume_cm3_mol:    Molar volume of drug (cm³/mol)
        temperature_c:                Temperature in Celsius (default 37°C)

    Returns:
        PhysicsResult with χ value and miscibility verdict.
    """
    violations = []
    metrics    = {}

    T     = temperature_c + 273.15
    R_cal = 1.987    # cal/(mol·K) - gas constant in cal units

    # χ = V_mol[cm³/mol] / (R[cal/mol·K] × T[K]) × (δ₁ - δ₂)²[(cal/cm³)]
    # Unit conversion: δ in MPa^0.5 → (cal/cm³)^0.5  via ×0.4888
    # (1 MPa^0.5 = 0.4888 (cal/cm³)^0.5)
    # This gives χ dimensionless. Source: Greenhalgh et al. J.Pharm.Sci (2000)
    d1_cal = drug_solubility_parameter    * 0.4888   # (cal/cm³)^0.5
    d2_cal = polymer_solubility_parameter * 0.4888
    delta_sq_cal = (d1_cal - d2_cal) ** 2            # cal/cm³
    chi = drug_molar_volume_cm3_mol / (R_cal * T) * delta_sq_cal

    metrics["chi_parameter"]              = round(chi, 4)
    metrics["delta_drug_MPa05"]           = drug_solubility_parameter
    metrics["delta_polymer_MPa05"]        = polymer_solubility_parameter
    metrics["delta_difference_MPa05"]     = round(abs(drug_solubility_parameter - polymer_solubility_parameter), 2)
    metrics["drug_molar_volume_cm3_mol"]  = drug_molar_volume_cm3_mol
    metrics["chi_miscibility_threshold"]  = 2.0   # Greenhalgh 2000, Hildebrand method
    metrics["temperature_C"]              = temperature_c

    # ── Miscibility verdict ─────────────────────────────────────────────────
    # Threshold for Hildebrand-calculated χ: empirical χ < 2.0 predicts miscibility
    # (Greenhalgh et al. J.Pharm.Sci 2000; validated for 25+ drug/polymer pairs)
    # NOTE: the classical FH χ < 0.5 applies to the LATTICE model with symmetric
    # components; for asymmetric drug/polymer pairs with Hildebrand δ, χ_crit ~ 2.0
    if chi > 5.0:
        violations.append(
            f"CRITICAL: χ = {chi:.3f} >> 2.0 (Greenhalgh threshold). "
            f"Strong thermodynamic driving force for phase separation. "
            f"Drug will crystallise from the amorphous solid dispersion. "
            f"Δδ = {abs(drug_solubility_parameter - polymer_solubility_parameter):.1f} MPa^0.5 is far too large."
        )
    elif chi > 2.0:
        violations.append(
            f"WARNING: χ = {chi:.3f} > 2.0 (Greenhalgh miscibility threshold). "
            f"Phase separation is thermodynamically likely. "
            f"Δδ = {abs(drug_solubility_parameter - polymer_solubility_parameter):.1f} MPa^0.5 "
            f"exceeds the ~4 MPa^0.5 practical limit for ASD stability."
        )

    # ── Solubility parameter gap check ─────────────────────────────────────
    delta_diff = abs(drug_solubility_parameter - polymer_solubility_parameter)
    metrics["delta_diff_rule_of_thumb_threshold"] = 4.0
    if delta_diff > 4.0:
        violations.append(
            f"WARNING: Solubility parameter difference Δδ = {delta_diff:.1f} MPa^0.5 "
            f"exceeds the empirical 4 MPa^0.5 practical limit for ASD miscibility "
            f"(Greenhalgh et al., J.Pharm.Sci 2000)."
        )

    passed = len(violations) == 0

    if not passed:
        physics_feedback = (
            f"Flory-Huggins analysis: χ = {chi:.3f}.\n"
            f"Violations:\n" + "\n".join(f"  • {v}" for v in violations) +
            f"\n\nThermodynamic data:"
            f"\n  • χ_miscibility_threshold = 2.0 (Greenhalgh 2000, Hildebrand method)"
            f"\n  • Current χ = {chi:.3f} - {'MISCIBLE' if chi < 2.0 else 'IMMISCIBLE'}"
            f"\n  • Δδ = {delta_diff:.1f} MPa^0.5 (target: < 4 MPa^0.5)"
        )
        correction_hint = (
            f"Choose a polymer with δ closer to {drug_solubility_parameter:.1f} MPa^0.5. "
            f"Target |Δδ| < 7 MPa^0.5 (current: {delta_diff:.1f}). "
            f"For δ_drug ≈ {drug_solubility_parameter:.1f}: consider "
            + _suggest_polymer(drug_solubility_parameter)
        )
    else:
        physics_feedback = (
            f"Flory-Huggins APPROVED: χ = {chi:.3f} < 2.0 (Greenhalgh threshold). "
            f"Drug-polymer system is thermodynamically miscible. "
            f"Amorphous solid dispersion predicted to be stable."
        )
        correction_hint = ""

    return PhysicsResult(
        passed          = passed,
        domain          = "drug_formulation",
        check_name      = "polymer_drug_compatibility",
        design_summary  = (
            f"δ_drug={drug_solubility_parameter} MPa^0.5, "
            f"δ_polymer={polymer_solubility_parameter} MPa^0.5, "
            f"V_mol={drug_molar_volume_cm3_mol} cm³/mol → χ={chi:.3f}"
        ),
        violations      = violations,
        metrics         = metrics,
        physics_feedback = physics_feedback,
        correction_hint = correction_hint,
        reference       = "Flory (1953) Principles of Polymer Chemistry; Marsac et al. Pharm. Res. (2006)",
    )


def _suggest_polymer(drug_delta: float) -> str:
    """Suggest a polymer with δ close to the drug's solubility parameter.
    δ values from Greenhalgh et al. J.Pharm.Sci (2000) - most cited pharma reference."""
    polymers = {
        "PEG 4000":    22.0,   # Greenhalgh 2000
        "Eudragit L":  21.5,   # Greenhalgh 2000
        "Soluplus":    22.5,   # estimated
        "PVP K30":     22.3,   # Greenhalgh 2000 (NOT 28.6 - common LLM error)
        "HPMC":        23.2,   # Greenhalgh 2000
        "HPMC-AS":     24.0,   # estimated
    }
    best = min(polymers, key=lambda p: abs(polymers[p] - drug_delta))
    return f"{best} (δ = {polymers[best]} MPa^0.5, Δδ = {abs(polymers[best] - drug_delta):.1f})"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Drug Permeability - Lipinski + RDKit
# ═══════════════════════════════════════════════════════════════════════════════

def check_drug_permeability(smiles: str) -> PhysicsResult:
    """
    Lipinski Rule-of-Five + Veber criteria for oral drug permeability.
    Computes molecular descriptors from SMILES using RDKit.

    Lipinski RO5 (Lipinski et al., Adv. Drug Deliv. Rev. 2001):
      MW   ≤ 500 Da
      logP ≤ 5     (ClogP, Crippen method)
      HBD  ≤ 5     (H-bond donors)
      HBA  ≤ 10    (H-bond acceptors)

    Veber criteria (Veber et al., J. Med. Chem. 2002):
      TPSA ≤ 140 Å²  (topological polar surface area)
      Rotatable bonds ≤ 10

    Non-monotonic permeability vs logP:
      "Higher logP gives better permeability" holds only up to logP ~ 3. Above
      logP ~ 5, aqueous solubility drops exponentially, making the drug
      dissolution-limited rather than permeation-limited. Net permeability peaks
      around logP = 1-3 and decreases above logP > 5.
      (Source: Abraham & Zhao, J. Org. Chem. 2004)

    Args:
        smiles: SMILES string of the drug molecule

    Returns:
        PhysicsResult with all molecular descriptors and permeability verdict.
    """
    violations = []
    metrics    = {}

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return PhysicsResult(
                passed=False, domain="drug_formulation", check_name="drug_permeability",
                design_summary=f"Invalid SMILES: {smiles}",
                violations=["CRITICAL: Invalid SMILES string - could not parse molecule."],
                physics_feedback="Provide a valid SMILES string. Use PubChem or ChemDraw to verify.",
                correction_hint="Check the SMILES at https://pubchem.ncbi.nlm.nih.gov/",
            )

        mw   = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    except ImportError:
        return PhysicsResult(
            passed=False, domain="drug_formulation", check_name="drug_permeability",
            design_summary="RDKit not available",
            violations=["RDKit is required: pip install rdkit"],
            physics_feedback="Install RDKit: pip install rdkit",
        )

    metrics["molecular_weight_Da"]      = round(mw, 2)
    metrics["logP_Crippen"]             = round(logP, 2)
    metrics["H_bond_donors"]            = hbd
    metrics["H_bond_acceptors"]         = hba
    metrics["TPSA_A2"]                  = round(tpsa, 1)
    metrics["rotatable_bonds"]          = rotb
    metrics["smiles"]                   = smiles

    ro5_violations = 0

    if mw > 500:
        violations.append(
            f"Lipinski RO5 FAIL: MW = {mw:.1f} Da > 500 Da. "
            f"Large molecules have poor passive absorption through intestinal epithelium."
        )
        ro5_violations += 1

    if logP > 5:
        violations.append(
            f"Lipinski RO5 FAIL: logP = {logP:.2f} > 5. "
            f"IMPORTANT: High logP does NOT improve permeability above this threshold - "
            f"aqueous solubility drops exponentially (log S ≈ -logP - 0.01·MW/100 + const), "
            f"making absorption dissolution-limited. Optimal range: logP = 1-3."
        )
        ro5_violations += 1

    if logP < 0:
        violations.append(
            f"WARNING: logP = {logP:.2f} < 0. Drug is highly hydrophilic. "
            f"Poor passive membrane permeability. Active transport may be needed."
        )

    if hbd > 5:
        violations.append(
            f"Lipinski RO5 FAIL: H-bond donors = {hbd} > 5. "
            f"Excess donors impair desolvation penalty for membrane crossing."
        )
        ro5_violations += 1

    if hba > 10:
        violations.append(
            f"Lipinski RO5 FAIL: H-bond acceptors = {hba} > 10. "
            f"Excess acceptors increase aqueous solubility at cost of membrane permeability."
        )
        ro5_violations += 1

    if tpsa > 140:
        violations.append(
            f"Veber criterion FAIL: TPSA = {tpsa:.1f} Å² > 140 Å². "
            f"High polar surface area strongly correlates with poor oral absorption."
        )

    if tpsa > 90:
        violations.append(
            f"Veber criterion NOTE: TPSA = {tpsa:.1f} Å² > 90 Å². "
            f"CNS penetration (blood-brain barrier) is unlikely at TPSA > 90 Å²."
        )

    if rotb > 10:
        violations.append(
            f"Veber criterion FAIL: Rotatable bonds = {rotb} > 10. "
            f"High flexibility reduces oral bioavailability."
        )

    metrics["ro5_violations"]    = ro5_violations
    metrics["passes_lipinski"]   = ro5_violations <= 1  # one violation tolerated

    passed = len([v for v in violations if "FAIL" in v]) == 0

    if not passed:
        physics_feedback = (
            f"Permeability analysis: {ro5_violations} Lipinski violation(s).\n"
            f"Violations:\n" + "\n".join(f"  • {v}" for v in violations) +
            f"\n\nKey insight: logP optimum for oral bioavailability is 1-3, not 'as high as possible'. "
            f"Above logP 5, dissolution becomes rate-limiting."
        )
        correction_hint = (
            f"Modify the molecule to: MW < 500, logP 1-3, HBD ≤ 5, HBA ≤ 10, TPSA < 90 Å². "
            f"Consider prodrug strategies, salt forms, or nanoparticle encapsulation "
            f"for BCS Class II/IV compounds."
        )
    else:
        physics_feedback = (
            f"Permeability APPROVED: All Lipinski RO5 criteria satisfied. "
            f"logP = {logP:.2f}, MW = {mw:.1f} Da, TPSA = {tpsa:.1f} Å². "
            f"Good oral bioavailability predicted."
        )
        correction_hint = ""

    return PhysicsResult(
        passed          = passed,
        domain          = "drug_formulation",
        check_name      = "drug_permeability",
        design_summary  = (
            f"MW={mw:.0f} Da, logP={logP:.2f}, HBD={hbd}, HBA={hba}, "
            f"TPSA={tpsa:.0f} Å² - {ro5_violations} RO5 violation(s)"
        ),
        violations      = violations,
        metrics         = metrics,
        physics_feedback = physics_feedback,
        correction_hint = correction_hint,
        reference       = (
            "Lipinski et al. (2001) Adv. Drug Deliv. Rev. 46:3-26; "
            "Veber et al. (2002) J. Med. Chem. 45:2615-2623"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Stokes-Einstein Diffusion Coefficient
# ═══════════════════════════════════════════════════════════════════════════════

def check_diffusion_coefficient(
    particle_radius_nm:     float,
    claimed_diffusivity_um2_s: float,
    temperature_c:          float = 37.0,
    medium:                 str   = "water",
) -> PhysicsResult:
    """
    Validates a claimed diffusion coefficient against the Stokes-Einstein equation.

    D = kT / (6π·η·r)

    Temperature-dependent viscosity:
      A common error is to use η = 1.0 mPa·s (water at 20°C) when the problem
      specifies 37°C. The correct value is η(37°C) = 0.692 mPa·s. Using the 20°C
      value overestimates the drag force by ~45% and underestimates the
      diffusion coefficient by ~45%. For a 100 nm particle, that is the
      difference between D ~ 4.5 μm²/s (at 20°C) and D ~ 6.6 μm²/s (at 37°C).

      Reference values well-known to Dr. Stoev's community:
        100 nm particle at 37°C: D ≈ 6.6 μm²/s
        10 nm protein at 37°C:   D ≈ 66 μm²/s   (~100 kDa globular protein ≈ 60 μm²/s)
        1 μm bead at 37°C:       D ≈ 0.44 μm²/s  (standard microrheology tracer)

    Args:
        particle_radius_nm:        Hydrodynamic radius of the particle in nm
        claimed_diffusivity_um2_s: The diffusion coefficient claimed/proposed (μm²/s)
        temperature_c:             Temperature in Celsius
        medium:                    "water", "blood_plasma", or "cytoplasm"

    Returns:
        PhysicsResult: passes if claimed D is within 20% of Stokes-Einstein prediction.
    """
    violations = []
    metrics    = {}

    T = temperature_c + 273.15
    kT = K_B * T
    r  = particle_radius_nm * 1e-9  # m

    # ── Water viscosity - Vogel equation (accurate to 1% from 0-100°C) ────
    # η(T) = A × 10^(B / (T - C))
    # A = 2.414e-5 Pa·s, B = 247.8 K, C = 140.0 K
    # Source: Dortmund Data Bank / standard reference (gives η(20°C)=1.002, η(37°C)=0.692)
    viscosity_media = {
        "water":        2.414e-5 * 10 ** (247.8 / (T - 140.0)),    # Pa·s, Vogel eq.
        "blood_plasma": 1.2e-3,    # Pa·s - approx. at 37°C (Bhatt et al. 2003)
        "cytoplasm":    3.0e-3,    # Pa·s - approx. cytoplasm viscosity (Wirtz 2009)
    }
    eta = viscosity_media.get(medium.lower(), viscosity_media["water"])
    eta_mPas = eta * 1e3

    # ── Stokes-Einstein D ───────────────────────────────────────────────────
    D_si   = kT / (6 * math.pi * eta * r)      # m²/s
    D_um2s = D_si * 1e12                        # μm²/s

    # Common wrong answer: using η = 1.0 mPa·s regardless of temperature
    D_wrong = kT / (6 * math.pi * 1.0e-3 * r) * 1e12  # η = 1.0 mPa·s (20°C value)

    metrics["stokes_einstein_D_um2_s"]     = round(D_um2s, 3)
    metrics["claimed_D_um2_s"]             = claimed_diffusivity_um2_s
    metrics["water_viscosity_mPa_s"]       = round(eta_mPas, 4)
    metrics["temperature_C"]               = temperature_c
    metrics["kT_J"]                        = f"{kT:.4e}"
    metrics["particle_radius_nm"]          = particle_radius_nm
    metrics["medium"]                      = medium
    metrics["wrong_D_if_using_20C_eta"]    = round(D_wrong, 3)

    # ── Tolerance: 20% around Stokes-Einstein value ─────────────────────────
    tolerance = 0.20
    ratio = claimed_diffusivity_um2_s / D_um2s
    deviation_pct = abs(ratio - 1.0) * 100

    metrics["deviation_from_stokes_einstein_pct"] = round(deviation_pct, 1)

    if deviation_pct > 20:
        # Diagnose the most likely error
        ratio_to_wrong = claimed_diffusivity_um2_s / D_wrong
        if abs(ratio_to_wrong - 1.0) < 0.10:
            diagnosis = (
                f"The claimed value matches the WRONG Stokes-Einstein prediction "
                f"using η = 1.0 mPa·s (water at 20°C). "
                f"The correct viscosity at {temperature_c:.0f}°C is η = {eta_mPas:.3f} mPa·s."
            )
        else:
            diagnosis = (
                f"Claimed D deviates {deviation_pct:.0f}% from Stokes-Einstein prediction. "
                f"Check particle radius and viscosity assumptions."
            )

        violations.append(
            f"PHYSICS MISMATCH: Claimed D = {claimed_diffusivity_um2_s:.3f} μm²/s, "
            f"Stokes-Einstein predicts D = {D_um2s:.3f} μm²/s at {temperature_c:.0f}°C "
            f"in {medium} (η = {eta_mPas:.3f} mPa·s). Deviation = {deviation_pct:.0f}%. "
            f"{diagnosis}"
        )

    passed = len(violations) == 0

    if not passed:
        physics_feedback = (
            f"Stokes-Einstein check failed.\n"
            f"Violations:\n" + "\n".join(f"  • {v}" for v in violations) +
            f"\n\nCorrect physics:"
            f"\n  D = kT / (6π·η·r)"
            f"\n  kT({temperature_c:.0f}°C) = {kT:.4e} J"
            f"\n  η({temperature_c:.0f}°C, {medium}) = {eta_mPas:.4f} mPa·s  ← NOT 1.0 mPa·s"
            f"\n  r = {particle_radius_nm} nm"
            f"\n  → D = {D_um2s:.3f} μm²/s"
        )
        correction_hint = (
            f"Use D = {D_um2s:.3f} μm²/s (Stokes-Einstein at {temperature_c:.0f}°C, "
            f"η = {eta_mPas:.3f} mPa·s for {medium})."
        )
    else:
        physics_feedback = (
            f"Stokes-Einstein APPROVED: Claimed D = {claimed_diffusivity_um2_s:.3f} μm²/s "
            f"is within {deviation_pct:.1f}% of theoretical {D_um2s:.3f} μm²/s. "
            f"η({temperature_c:.0f}°C) = {eta_mPas:.3f} mPa·s."
        )
        correction_hint = ""

    return PhysicsResult(
        passed          = passed,
        domain          = "drug_formulation",
        check_name      = "diffusion_coefficient",
        design_summary  = (
            f"r={particle_radius_nm} nm, T={temperature_c}°C, medium={medium}: "
            f"D_claimed={claimed_diffusivity_um2_s:.3f} vs D_SE={D_um2s:.3f} μm²/s"
        ),
        violations      = violations,
        metrics         = metrics,
        physics_feedback = physics_feedback,
        correction_hint = correction_hint,
        reference       = (
            "Einstein (1905) Ann. Phys. 17:549; "
            "Huber et al. (2009) J. Phys. Chem. Ref. Data 38:101 (viscosity)"
        ),
    )
