"""
Drug Formulation Structural contract
==========================================
Hand-coded contract for nanoparticle drug delivery formulation.
Derived directly from governing equations, not fitted to data.

Dependency graph (DAG):

  EXOGENOUS (designer-controlled):
    zeta_potential_mv          ─┐
    particle_radius_nm          ├─→ energy_barrier_kT ──→ colloidal_stable
    salt_concentration_mm      ─┘
    particle_radius_nm         ─────────────────────────→ diffusion_coeff_um2s
    temperature_c              ─┬─→ energy_barrier_kT
                                └─→ diffusion_coeff_um2s
    drug_delta_mpa05           ─┐
    polymer_delta_mpa05         ├─→ chi_parameter ───────→ polymer_compatible
    drug_molar_volume_cm3mol   ─┘

  ENDOGENOUS (computed by physics):
    debye_length_nm            ← f(salt_concentration_mm, temperature_c)
    gamma_reduced_zeta         ← f(zeta_potential_mv, temperature_c)
    energy_barrier_kT          ← f(debye_length_nm, gamma_reduced_zeta,
                                    particle_radius_nm, temperature_c)
    chi_parameter              ← f(drug_delta_mpa05, polymer_delta_mpa05,
                                    drug_molar_volume_cm3mol, temperature_c)
    water_viscosity_mPas       ← f(temperature_c)
    diffusion_coeff_um2s       ← f(particle_radius_nm, temperature_c,
                                    water_viscosity_mPas)

  OUTCOMES (pass/fail targets):
    colloidal_stable           ← energy_barrier_kT ≥ 15 kT
    polymer_compatible         ← chi_parameter < 2.0
    diffusion_physical         ← |D_claimed - D_SE| / D_SE < 0.20

Key dependency claims:
  1. zeta_potential_mv → energy_barrier_kT
     Mechanism: DLVO EDL term V_EDL ∝ γ²(ζ), where γ = tanh(eζ/4kT)
     Direction: |ζ| ↑ → barrier ↑  (more repulsion)

  2. salt_concentration_mm → debye_length_nm → energy_barrier_kT
     Mechanism: κ⁻¹ = √(εkT/2n∞e²), V_EDL ∝ exp(-κh)
     Direction: salt ↑ → Debye ↓ → barrier ↓  (screening)

  3. particle_radius_nm → energy_barrier_kT  AND  diffusion_coeff_um2s
     Mechanism: DLVO Derjaguin: V ∝ a; Stokes-Einstein: D ∝ 1/r
     Direction: r ↑ → barrier ↑ (more EDL area) but D ↓ (slower diffusion)
     Note: CONFOUNDED - r has OPPOSITE effects on stability vs diffusion
           This is the key structural insight that iterative search misses.

  4. temperature_c → water_viscosity_mPas → diffusion_coeff_um2s
     Mechanism: Vogel equation η(T) = 2.414e-5 × 10^(247.8/(T-140))
     Direction: T ↑ → η ↓ → D ↑  (the 37°C vs 20°C hallucination lives here)

  5. drug_delta_mpa05, polymer_delta_mpa05 → chi_parameter
     Mechanism: χ = V_mol/RT × (δ₁ - δ₂)² (Greenhalgh 2000)
     Direction: |Δδ| ↑ → χ ↑ → less compatible

References:
  Israelachvili (2011) Intermolecular and Surface Forces, 3rd ed.
  Greenhalgh et al. (2000) J. Pharm. Sci. 89:1461-1470
  Einstein (1905) Ann. Phys. 17:549
  Vogel (1921) / Huber et al. (2009) J. Phys. Chem. Ref. Data 38:101
"""

from __future__ import annotations
import math
from phact_physics.contracts.engine import DependencyGraph, GraphNode, GraphEdge

# ── Physical constants ────────────────────────────────────────────────────────
K_B      = 1.380649e-23    # J/K
E_CHARGE = 1.602176634e-19 # C
EPS_0    = 8.8541878e-12   # F/m
N_AVO    = 6.02214076e23   # mol⁻¹
HAMAKER_PLGA_WATER = 5.0e-21  # J  (Israelachvili Table 13.2, in-water value)


def build_drug_formulation_contract() -> DependencyGraph:
    """
    Build and return the contract for nanoparticle drug formulation.

    All structural equations are derived from first principles -     not fitted to data. The contract is exact for the physics domain.
    """
    g = DependencyGraph(
        domain      = "drug_formulation",
        description = (
            "Structural contract for PLGA nanoparticle formulation. "
            "Encodes DLVO stability, Flory-Huggins compatibility, and "
            "Stokes-Einstein diffusion as a DAG with structural equations."
        ),
    )

    # ── Exogenous nodes (designer-controlled) ─────────────────────────────────

    g.add_node(GraphNode(
        name         = "zeta_potential_mv",
        node_type    = "exogenous",
        unit         = "mV",
        description  = (
            "Surface zeta potential of the nanoparticle. "
            "Negative values = anionic surface (typical for PLGA). "
            "Controls electrostatic repulsion via DLVO EDL term."
        ),
        bounds       = (-80.0, 80.0),
    ))

    g.add_node(GraphNode(
        name         = "particle_radius_nm",
        node_type    = "exogenous",
        unit         = "nm",
        description  = (
            "Hydrodynamic radius of the nanoparticle. "
            "Larger radius → more EDL area (better stability) "
            "BUT slower diffusion (D ∝ 1/r). "
            "This is the key structural confound in the design space."
        ),
        bounds       = (10.0, 500.0),
    ))

    g.add_node(GraphNode(
        name         = "salt_concentration_mm",
        node_type    = "exogenous",
        unit         = "mM",
        description  = (
            "Monovalent salt concentration (NaCl). "
            "Physiological PBS = 150 mM. "
            "Higher salt screens electrostatic repulsion (shorter Debye length)."
        ),
        bounds       = (1.0, 500.0),
    ))

    g.add_node(GraphNode(
        name         = "temperature_c",
        node_type    = "exogenous",
        unit         = "°C",
        description  = (
            "System temperature. "
            "Physiological = 37°C. "
            "Affects both DLVO (via kT) and diffusion (via Vogel viscosity). "
            "Common error: using η(20°C) = 1.0 mPa·s instead of η(37°C) = 0.692 mPa·s."
        ),
        bounds       = (0.0, 100.0),
    ))

    g.add_node(GraphNode(
        name         = "drug_delta_mpa05",
        node_type    = "exogenous",
        unit         = "MPa^0.5",
        description  = (
            "Hildebrand solubility parameter of the drug. "
            "Ibuprofen: δ = 19.9 MPa^0.5. "
            "Used in Flory-Huggins χ calculation."
        ),
        bounds       = (10.0, 40.0),
    ))

    g.add_node(GraphNode(
        name         = "polymer_delta_mpa05",
        node_type    = "exogenous",
        unit         = "MPa^0.5",
        description  = (
            "Hildebrand solubility parameter of the polymer carrier. "
            "PVP K30: δ = 22.3 MPa^0.5 (Greenhalgh 2000). "
            "The value 28.6 sometimes cited is from solubility experiments, "
            "not calorimetry, and is not used here."
        ),
        bounds       = (15.0, 35.0),
    ))

    g.add_node(GraphNode(
        name         = "drug_molar_volume_cm3mol",
        node_type    = "exogenous",
        unit         = "cm³/mol",
        description  = (
            "Molar volume of the drug (V_mol = MW / density). "
            "Ibuprofen: V_mol ≈ 180 cm³/mol. "
            "Normalises the χ parameter for molecule size."
        ),
        bounds       = (50.0, 500.0),
    ))

    # ── Endogenous nodes (structural equations from physics) ──────────────────

    def compute_debye_length(p: dict) -> float:
        """
        Debye screening length κ⁻¹ [nm].
        κ⁻¹ = √(ε₀·εᵣ·kT / (2·n∞·e²))
        Israelachvili eq. 14.9

        Parents: salt_concentration_mm, temperature_c
        Direction: salt ↑ → κ⁻¹ ↓  (more screening)
                   T ↑   → κ⁻¹ ↑  (slightly, via kT)
        """
        T     = p["temperature_c"] + 273.15
        kT    = K_B * T
        c_si  = p["salt_concentration_mm"] * 1e-3 * 1e3 * N_AVO  # number density m⁻³
        eps_r = 74.5 if p["temperature_c"] > 30 else 78.5
        eps   = eps_r * EPS_0
        kappa = math.sqrt(2 * c_si * E_CHARGE**2 / (eps * kT))
        return (1.0 / kappa) * 1e9  # nm

    g.add_node(GraphNode(
        name          = "debye_length_nm",
        node_type     = "endogenous",
        unit          = "nm",
        description   = "Electrostatic screening length. 150 mM NaCl, 37°C → 0.78 nm.",
        bounds        = (0.1, 100.0),
        structural_eq = compute_debye_length,
        parent_names  = ["salt_concentration_mm", "temperature_c"],
    ))

    def compute_gamma(p: dict) -> float:
        """
        Reduced surface potential γ = tanh(eζ/4kT).
        Appears in DLVO EDL term as γ².
        Israelachvili eq. 14.16

        Parents: zeta_potential_mv, temperature_c
        Direction: |ζ| ↑ → γ → 1  (saturation; γ² enters V_EDL)
        """
        T    = p["temperature_c"] + 273.15
        kT   = K_B * T
        zeta = p["zeta_potential_mv"] * 1e-3  # V
        return math.tanh(E_CHARGE * zeta / (4 * kT))

    g.add_node(GraphNode(
        name          = "gamma_reduced_zeta",
        node_type     = "endogenous",
        unit          = "dimensionless",
        description   = "Reduced surface potential γ = tanh(eζ/4kT). Range: -1 to +1.",
        bounds        = (-1.0, 1.0),
        structural_eq = compute_gamma,
        parent_names  = ["zeta_potential_mv", "temperature_c"],
    ))

    def compute_energy_barrier(p: dict) -> float:
        """
        DLVO energy barrier V_max [kT].
        V_total(h) = V_vdW(h) + V_EDL(h)
        V_vdW(h) = -A·a / (12h)
        V_EDL(h) = 64·π·ε·a·(kT/e)²·γ²·exp(-κh)

        Scans h = 0.1-50 nm; returns maximum.
        Israelachvili eq. 14.16 + Derjaguin approximation.

        Parent inputs: debye_length_nm, gamma_reduced_zeta,
                       particle_radius_nm, temperature_c
        Dependencies: increasing |ζ| → γ ↑ → V_EDL ↑ → barrier ↑
                      increasing salt → Debye ↓ → V_EDL decays faster → barrier ↓
                      These are INDEPENDENT dependency paths through the graph.
        """
        T     = p["temperature_c"] + 273.15
        kT    = K_B * T
        a     = p["particle_radius_nm"] * 1e-9
        gamma = p["gamma_reduced_zeta"]
        eps_r = 74.5 if p["temperature_c"] > 30 else 78.5
        eps   = eps_r * EPS_0
        kappa = 1.0 / (p["debye_length_nm"] * 1e-9)  # m⁻¹

        v_max_kT = 0.0
        A        = HAMAKER_PLGA_WATER

        for i in range(1, 500):
            h     = i * 1e-10  # 0.1-50 nm
            v_vdw = -A * a / (12.0 * h)
            v_edl = (64.0 * math.pi * eps * a
                     * (kT / E_CHARGE)**2
                     * gamma**2
                     * math.exp(-kappa * h))
            v_tot = (v_vdw + v_edl) / kT
            if v_tot > v_max_kT:
                v_max_kT = v_tot

        return v_max_kT

    g.add_node(GraphNode(
        name          = "energy_barrier_kT",
        node_type     = "endogenous",
        unit          = "kT",
        description   = (
            "DLVO energy barrier height. "
            "≥ 15 kT → stable. 5-15 kT → marginal. < 5 kT → rapid aggregation."
        ),
        bounds        = (0.0, 1000.0),
        structural_eq = compute_energy_barrier,
        parent_names  = ["debye_length_nm", "gamma_reduced_zeta",
                         "particle_radius_nm", "temperature_c"],
    ))

    def compute_chi(p: dict) -> float:
        """
        Flory-Huggins χ parameter.
        χ = V_mol / (R·T) × (δ_drug - δ_polymer)²
        Units: V_mol [cm³/mol], R [cal/mol·K], δ [(cal/cm³)^0.5]
        Conversion: 1 MPa^0.5 = 0.4888 (cal/cm³)^0.5

        Source: Greenhalgh et al. J. Pharm. Sci. (2000)
        Threshold: χ < 2.0 for miscibility (not 0.5, which is for a
                   symmetric lattice; the Hildebrand method uses 2.0)

        Parents: drug_delta_mpa05, polymer_delta_mpa05,
                        drug_molar_volume_cm3mol, temperature_c
        Direction: |Δδ| ↑ → χ ↑ → less miscible
        """
        T     = p["temperature_c"] + 273.15
        R_cal = 1.987  # cal/(mol·K)
        d1    = p["drug_delta_mpa05"]    * 0.4888  # (cal/cm³)^0.5
        d2    = p["polymer_delta_mpa05"] * 0.4888
        return p["drug_molar_volume_cm3mol"] / (R_cal * T) * (d1 - d2)**2

    g.add_node(GraphNode(
        name          = "chi_parameter",
        node_type     = "endogenous",
        unit          = "dimensionless",
        description   = (
            "Flory-Huggins interaction parameter χ. "
            "< 2.0: miscible (Greenhalgh 2000). > 2.0: phase separation likely."
        ),
        bounds        = (0.0, 50.0),
        structural_eq = compute_chi,
        parent_names  = ["drug_delta_mpa05", "polymer_delta_mpa05",
                         "drug_molar_volume_cm3mol", "temperature_c"],
    ))

    def compute_viscosity(p: dict) -> float:
        """
        Water viscosity [mPa·s] via Vogel equation.
        η(T) = 2.414e-5 × 10^(247.8 / (T_K - 140.0))  [Pa·s]

        Source: Huber et al. J. Phys. Chem. Ref. Data (2009)
        Verified: η(20°C) = 1.002 mPa·s, η(37°C) = 0.692 mPa·s

        Common error: using η = 1.0 mPa·s regardless of T.
        At 37°C that is a 45% error in the drag force.

        Parent: temperature_c
        Direction: T ↑ → η ↓  (water becomes less viscous when heated)
        """
        T_K = p["temperature_c"] + 273.15
        eta_pa = 2.414e-5 * (10 ** (247.8 / (T_K - 140.0)))
        return eta_pa * 1e3  # mPa·s

    g.add_node(GraphNode(
        name          = "water_viscosity_mPas",
        node_type     = "endogenous",
        unit          = "mPa·s",
        description   = (
            "Water viscosity at temperature T (Vogel equation). "
            "η(37°C) = 0.692 mPa·s, not the 1.0 mPa·s value for 20°C water."
        ),
        bounds        = (0.1, 5.0),
        structural_eq = compute_viscosity,
        parent_names  = ["temperature_c"],
    ))

    def compute_diffusion(p: dict) -> float:
        """
        Stokes-Einstein diffusion coefficient [μm²/s].
        D = kT / (6π·η·r)

        Source: Einstein (1905) Ann. Phys. 17:549

        Parents: particle_radius_nm, temperature_c, water_viscosity_mPas
        Direction:
          r ↑  → D ↓   (larger particles diffuse slower)
          T ↑  → D ↑   (via both kT and lower η)
          η ↓  → D ↑   (less viscous medium → faster diffusion)

        NOTE: temperature_c → water_viscosity_mPas → diffusion_coeff_um2s
              is the MEDIATED path - the effect of T on D is MEDIATED by η.
              the graph separates the direct kT effect from the η effect.
        """
        T_K  = p["temperature_c"] + 273.15
        kT   = K_B * T_K
        r    = p["particle_radius_nm"] * 1e-9
        eta  = p["water_viscosity_mPas"] * 1e-3  # Pa·s
        D_si = kT / (6 * math.pi * eta * r)
        return D_si * 1e12  # μm²/s

    g.add_node(GraphNode(
        name          = "diffusion_coeff_um2s",
        node_type     = "endogenous",
        unit          = "μm²/s",
        description   = (
            "Stokes-Einstein diffusion coefficient. "
            "100 nm PLGA at 37°C: D ≈ 6.6 μm²/s."
        ),
        bounds        = (0.01, 1000.0),
        structural_eq = compute_diffusion,
        parent_names  = ["particle_radius_nm", "temperature_c", "water_viscosity_mPas"],
    ))

    # ── Outcome nodes ─────────────────────────────────────────────────────────

    def colloidal_stable(p: dict) -> float:
        """1.0 if energy_barrier_kT ≥ 15, else 0.0"""
        return 1.0 if p["energy_barrier_kT"] >= 15.0 else 0.0

    g.add_node(GraphNode(
        name          = "colloidal_stable",
        node_type     = "outcome",
        unit          = "bool",
        description   = "1 if DLVO energy barrier ≥ 15 kT (Israelachvili stability criterion).",
        bounds        = (0.0, 1.0),
        structural_eq = colloidal_stable,
        parent_names  = ["energy_barrier_kT"],
    ))

    def polymer_compatible(p: dict) -> float:
        """1.0 if chi_parameter < 2.0 (Greenhalgh 2000)"""
        return 1.0 if p["chi_parameter"] < 2.0 else 0.0

    g.add_node(GraphNode(
        name          = "polymer_compatible",
        node_type     = "outcome",
        unit          = "bool",
        description   = "1 if Flory-Huggins χ < 2.0 (Greenhalgh miscibility threshold).",
        bounds        = (0.0, 1.0),
        structural_eq = polymer_compatible,
        parent_names  = ["chi_parameter"],
    ))

    # ── Dependency edges (with mechanism descriptions) ────────────────────────────

    # DLVO pathway
    g.add_edge(GraphEdge("zeta_potential_mv",    "gamma_reduced_zeta",
        +1, "γ = tanh(eζ/4kT): |ζ| ↑ → γ → 1 (saturating)"))
    g.add_edge(GraphEdge("temperature_c",         "gamma_reduced_zeta",
        -1, "Higher T increases kT denominator → weaker ζ effect on γ"))
    g.add_edge(GraphEdge("salt_concentration_mm", "debye_length_nm",
        -1, "κ ∝ √c_salt: more salt → shorter Debye length (more screening)"))
    g.add_edge(GraphEdge("temperature_c",         "debye_length_nm",
        +1, "κ⁻¹ ∝ √(kT): higher T → slightly longer Debye length"))
    g.add_edge(GraphEdge("gamma_reduced_zeta",    "energy_barrier_kT",
        +1, "V_EDL ∝ γ²: higher γ → stronger repulsion → higher barrier"))
    g.add_edge(GraphEdge("debye_length_nm",       "energy_barrier_kT",
        +1, "V_EDL ∝ exp(-κh): longer Debye → slower decay → higher barrier"))
    g.add_edge(GraphEdge("particle_radius_nm",    "energy_barrier_kT",
        +1, "Derjaguin: V ∝ a: larger particle → more EDL area → higher barrier"))
    g.add_edge(GraphEdge("temperature_c",         "energy_barrier_kT",
        0,  "T affects both kT and ε (opposing effects - net near-zero)"))
    g.add_edge(GraphEdge("energy_barrier_kT",     "colloidal_stable",
        +1, "Barrier ≥ 15 kT → stable (Israelachvili criterion)"))

    # Diffusion pathway
    g.add_edge(GraphEdge("temperature_c",         "water_viscosity_mPas",
        -1, "Vogel equation: T ↑ → η ↓ (water less viscous when heated)"))
    g.add_edge(GraphEdge("water_viscosity_mPas",  "diffusion_coeff_um2s",
        -1, "Stokes-Einstein: D = kT/6πηr - η ↑ → D ↓"))
    g.add_edge(GraphEdge("particle_radius_nm",    "diffusion_coeff_um2s",
        -1, "Stokes-Einstein: D ∝ 1/r - larger particle → slower diffusion"))
    g.add_edge(GraphEdge("temperature_c",         "diffusion_coeff_um2s",
        +1, "Direct kT effect: T ↑ → kT ↑ → D ↑ (mediated path via η dominates)"))

    # Flory-Huggins pathway
    g.add_edge(GraphEdge("drug_delta_mpa05",          "chi_parameter",
        0,  "χ ∝ (δ_drug - δ_polymer)²: effect depends on sign of Δδ"))
    g.add_edge(GraphEdge("polymer_delta_mpa05",       "chi_parameter",
        0,  "χ ∝ (δ_drug - δ_polymer)²: closer δ → lower χ"))
    g.add_edge(GraphEdge("drug_molar_volume_cm3mol",  "chi_parameter",
        +1, "χ = V_mol/RT × Δδ²: larger molecule → higher χ at same Δδ"))
    g.add_edge(GraphEdge("temperature_c",             "chi_parameter",
        -1, "χ ∝ 1/T: higher temperature → more miscible (entropy wins)"))
    g.add_edge(GraphEdge("chi_parameter",             "polymer_compatible",
        -1, "χ < 2.0 → compatible; χ > 2.0 → phase separation"))

    return g
