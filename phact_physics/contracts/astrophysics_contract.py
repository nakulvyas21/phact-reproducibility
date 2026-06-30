"""
Structural contract - Gravitational-wave astrophysics
=========================================================
Exact SCMs for compact-binary inspiral, matching the deterministic checks in
``phact_physics.domains.astrophysics`` but expressed as dependency graphs in which the
*certified quantity is a derived outcome node*, never a model-supplied input.

Why this matters for certification integrity
---------------------------------------------
The bare astrophysics tools take the answer as an argument, e.g.
``check_chirp_mass(m1, m2, chirp_mass_msun)`` validates a *claimed* chirp mass.
A proposer can satisfy a "verify the ISCO frequency is 300 Hz" goal by silently
substituting a different total mass that genuinely has that frequency - the
engine then certifies correct physics for an off-goal design. The English
constraint "the mass is fixed" is invisible to the tool.

Routing certification through the contract removes this loophole structurally. The
fixed quantities of a goal are bound as *exogenous* nodes; the certified
quantity (chirp mass, ISCO frequency, strain, merger time) is an *outcome* node
computed by the structural equation from those exogenous parents. The proposer
never supplies the outcome - it is derived - so it cannot substitute a
self-consistent alternative. A goal that fixes the masses and asks for an
impossible derived value is therefore impossible by construction: the contract
computes the true value and the target cannot be met without changing an
exogenous node the goal forbids.

Physical constants (SI, CODATA 2018) - identical to the domain checks.

References
----------
  Peters (1964) Phys. Rev. 136, B1224 - merger timescale
  Maggiore (2007) "Gravitational Waves Vol.1", Oxford - chirp mass, strain, ISCO
  Abbott et al. (2016) PRL 116, 061102 - GW150914
"""

from __future__ import annotations

import math

from phact_physics.contracts.engine import DependencyGraph, GraphNode, GraphEdge

# ── Physical constants (match phact_physics.domains.astrophysics) ────────────────
G     = 6.67430e-11    # m^3 / (kg s^2)
C     = 2.99792458e8   # m/s (exact)
M_SUN = 1.98892e30     # kg
PC    = 3.085677581e16 # m
MPC   = PC * 1e6       # m
AU    = 1.495978707e11 # m
YR    = 3.15575520e7   # s


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chirp mass:  exogenous m1, m2  ->  outcome chirp_mass_msun
# ═══════════════════════════════════════════════════════════════════════════════

def build_chirp_mass_contract() -> DependencyGraph:
    """contract for the chirp mass. The masses are exogenous (designer-controlled);
    the chirp mass is a derived outcome, so it cannot be supplied independently.

        M_c = (m1 m2)^(3/5) / (m1 + m2)^(1/5)
    """
    g = DependencyGraph(
        domain      = "astrophysics",
        description = (
            "contract for binary chirp mass. Exogenous component masses determine the "
            "derived chirp mass M_c = (m1 m2)^(3/5)/(m1+m2)^(1/5)."
        ),
    )
    g.add_node(GraphNode(
        name="m1_msun", node_type="exogenous", unit="Msun",
        description="Primary mass (designer-controlled).", bounds=(0.1, 1.0e3)))
    g.add_node(GraphNode(
        name="m2_msun", node_type="exogenous", unit="Msun",
        description="Secondary mass (designer-controlled).", bounds=(0.1, 1.0e3)))

    def compute_chirp(p: dict) -> float:
        m1 = p["m1_msun"] * M_SUN
        m2 = p["m2_msun"] * M_SUN
        return ((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2) / M_SUN

    g.add_node(GraphNode(
        name="chirp_mass_msun", node_type="outcome", unit="Msun",
        description="Derived chirp mass M_c.",
        bounds=(0.0, 1.0e3),
        structural_eq=compute_chirp,
        parent_names=["m1_msun", "m2_msun"]))

    for parent in ("m1_msun", "m2_msun"):
        g.add_edge(GraphEdge(
            parent=parent, child="chirp_mass_msun", effect_direction=+1,
            mechanism="M_c = (m1 m2)^(3/5)/(m1+m2)^(1/5): increasing a mass raises M_c"))
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ISCO frequency:  exogenous total_mass  ->  outcome f_isco_hz
# ═══════════════════════════════════════════════════════════════════════════════

def build_isco_contract() -> DependencyGraph:
    """contract for the Schwarzschild ISCO (peak) GW frequency. The total mass is
    exogenous; the ISCO frequency is derived and inversely proportional to mass.

        f_ISCO = c^3 / (6^(3/2) pi G M)   (GW = 2 x orbital)
    """
    g = DependencyGraph(
        domain      = "astrophysics",
        description = (
            "contract for the ISCO peak GW frequency. Exogenous total mass determines "
            "the derived f_ISCO = c^3/(6^(3/2) pi G M)."
        ),
    )
    g.add_node(GraphNode(
        name="total_mass_msun", node_type="exogenous", unit="Msun",
        description="Total binary mass M = m1 + m2 (designer-controlled).",
        bounds=(0.1, 1.0e4)))

    def compute_fisco(p: dict) -> float:
        M = p["total_mass_msun"] * M_SUN
        r_isco = 6.0 * G * M / C**2
        f_orb = (1.0 / (2.0 * math.pi)) * math.sqrt(G * M / r_isco**3)
        return 2.0 * f_orb  # GW frequency = 2 x orbital

    g.add_node(GraphNode(
        name="f_isco_hz", node_type="outcome", unit="Hz",
        description="Derived ISCO peak GW frequency.",
        bounds=(0.0, 1.0e6),
        structural_eq=compute_fisco,
        parent_names=["total_mass_msun"]))

    g.add_edge(GraphEdge(
        parent="total_mass_msun", child="f_isco_hz", effect_direction=-1,
        mechanism="f_ISCO = c^3/(6^(3/2) pi G M): inversely proportional to mass"))
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GW strain:  exogenous chirp_mass, distance, freq  ->  outcome strain
# ═══════════════════════════════════════════════════════════════════════════════

def build_strain_contract() -> DependencyGraph:
    """contract for the quadrupole GW strain amplitude. Chirp mass, luminosity
    distance, and GW frequency are exogenous; the strain is derived.

        h = (4/D_L)(G M_c/c^2)(pi G M_c f/c^3)^(2/3)
    """
    g = DependencyGraph(
        domain      = "astrophysics",
        description = (
            "contract for quadrupole GW strain. Exogenous chirp mass, luminosity "
            "distance, and GW frequency determine the derived strain amplitude."
        ),
    )
    g.add_node(GraphNode(
        name="chirp_mass_msun", node_type="exogenous", unit="Msun",
        description="Chirp mass (designer-controlled here).", bounds=(0.1, 1.0e3)))
    g.add_node(GraphNode(
        name="distance_mpc", node_type="exogenous", unit="Mpc",
        description="Luminosity distance.", bounds=(0.1, 1.0e5)))
    g.add_node(GraphNode(
        name="gw_frequency_hz", node_type="exogenous", unit="Hz",
        description="Gravitational-wave frequency.", bounds=(1.0, 1.0e4)))

    def compute_strain(p: dict) -> float:
        M_c = p["chirp_mass_msun"] * M_SUN
        D_L = p["distance_mpc"] * MPC
        f   = p["gw_frequency_hz"]
        inner = (math.pi * G * M_c * f / C**3) ** (2.0 / 3.0)
        return (4.0 / D_L) * (G * M_c / C**2) * inner

    g.add_node(GraphNode(
        name="strain", node_type="outcome", unit="dimensionless",
        description="Derived GW strain amplitude h.",
        bounds=(0.0, 1.0),
        structural_eq=compute_strain,
        parent_names=["chirp_mass_msun", "distance_mpc", "gw_frequency_hz"]))

    g.add_edge(GraphEdge(parent="chirp_mass_msun", child="strain", effect_direction=+1,
        mechanism="h grows with chirp mass (M_c^(5/3) overall)"))
    g.add_edge(GraphEdge(parent="distance_mpc", child="strain", effect_direction=-1,
        mechanism="h ~ 1/D_L: inversely proportional to luminosity distance"))
    g.add_edge(GraphEdge(parent="gw_frequency_hz", child="strain", effect_direction=+1,
        mechanism="h ~ f^(2/3): grows with frequency"))
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Merger time (Peters):  exogenous m1, m2, separation  ->  outcome time
# ═══════════════════════════════════════════════════════════════════════════════

def build_merger_time_contract() -> DependencyGraph:
    """contract for the Peters (1964) circular-orbit merger timescale. Component
    masses and separation are exogenous; the merger time is derived.

        T = a0^4 / (4 beta),  beta = (64/5) G^3 m1 m2 (m1+m2) / c^5
    """
    g = DependencyGraph(
        domain      = "astrophysics",
        description = (
            "contract for the Peters merger timescale. Exogenous masses and initial "
            "separation determine the derived coalescence time."
        ),
    )
    g.add_node(GraphNode(
        name="m1_msun", node_type="exogenous", unit="Msun",
        description="Primary mass.", bounds=(0.1, 1.0e3)))
    g.add_node(GraphNode(
        name="m2_msun", node_type="exogenous", unit="Msun",
        description="Secondary mass.", bounds=(0.1, 1.0e3)))
    g.add_node(GraphNode(
        name="separation_au", node_type="exogenous", unit="AU",
        description="Initial circular orbital separation.", bounds=(1e-4, 1e4)))

    def compute_merger_myr(p: dict) -> float:
        m1 = p["m1_msun"] * M_SUN
        m2 = p["m2_msun"] * M_SUN
        a0 = p["separation_au"] * AU
        beta = 64.0 / 5.0 * G**3 * m1 * m2 * (m1 + m2) / C**5
        if beta <= 0:
            return float("inf")
        return (a0**4 / (4.0 * beta)) / (YR * 1e6)

    g.add_node(GraphNode(
        name="merger_time_myr", node_type="outcome", unit="Myr",
        description="Derived Peters coalescence time.",
        bounds=(0.0, 1.0e18),
        structural_eq=compute_merger_myr,
        parent_names=["m1_msun", "m2_msun", "separation_au"]))

    g.add_edge(GraphEdge(parent="separation_au", child="merger_time_myr", effect_direction=+1,
        mechanism="T ~ a0^4: strongly increasing in separation"))
    for parent in ("m1_msun", "m2_msun"):
        g.add_edge(GraphEdge(parent=parent, child="merger_time_myr", effect_direction=-1,
            mechanism="T ~ 1/(m1 m2 (m1+m2)): heavier binaries merge faster"))
    return g


# Registry: goal-quantity -> (builder, outcome node name)
ASTROPHYSICS_CONTRACTS = {
    "chirp_mass":  (build_chirp_mass_contract,  "chirp_mass_msun"),
    "isco":        (build_isco_contract,        "f_isco_hz"),
    "strain":      (build_strain_contract,      "strain"),
    "merger_time": (build_merger_time_contract, "merger_time_myr"),
}
