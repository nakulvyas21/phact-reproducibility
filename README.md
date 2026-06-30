# PHACT - Reproducibility

Deterministic physics engines and reproducibility scripts for the paper
**"Structural Certification for Reliable Physical Design with Language Models"**
(Physics-Anchored Certification, PHACT).

This repository lets anyone verify the scientific core of the paper with **no
credentials, no network access, and no language model**. Everything here runs
on the Python standard library alone.

---

## What this repository is, and what it is not

The paper has two layers:

1. **A deterministic physics engine** with a structural certification contract.
   This is the scientific ground truth. It implements published equations
   (SantaLucia 1998 for DNA melting, the LIGO/Virgo post-Newtonian expressions
   for gravitational waves, `f = 1/(2 pi R C)` for filters, DLVO theory for
   colloids), together with the structural contracts -- directed dependency
   graphs that derive the certified quantity from a goal's fixed inputs
   (`phact_physics/contracts/`). It contains no machine learning.

2. **A language-model layer.** A model (Gemini 2.5 Flash; Llama 3.3 70B and
   Llama 3.1 8B for the cross-model results) proposes designs that the engine
   certifies. This layer is stochastic and requires API access.

This repository releases layer 1 in full, plus the archived outputs of the runs
reported in the paper. It does not include the live agent stack, API keys, or
any user interface. The deterministic engine is what the paper's claims rest on
and what a reviewer needs to check; the language-model procedure is described in
the paper's Methods, and its outputs are preserved here as archived data so the
tables regenerate from them.

---

## Quick start

```bash
git clone <this-repo>
cd phact-reproducibility
python3 -m pip install -e ".[dev]"   # installs pytest; the engines need nothing

make verify                          # full reproducibility check
```

For a 30-second tour of the certification surface - the exact tool
specifications the proposing model is bound by, a live engine verdict, and the
contract's anti-forgery property - run `make demo` (or `python3 demo.py`). It
needs no API key. The provenance of the archived model runs is documented in
[PROVENANCE.md](PROVENANCE.md).

`make verify` runs three checks:

| Target           | What it checks |
|------------------|----------------|
| `make calibrate` | Each engine reproduces published literature values within a stated tolerance (DNA melting temperature within +/-1 C of SantaLucia 1998; GW150914 chirp mass within +/-1 Msun of the LIGO catalog). |
| `make test`      | The impossible goals in the adversarial experiment are genuinely impossible: the engine rejects every one, and the structural contract rejects a forged target while accepting the true one. |
| `make tables`    | The paper's tables regenerate from the archived runs. |

---

## Reproducing the paper's tables

```bash
make tables          # or: python3 regenerate_tables.py
```

`regenerate_tables.py` reads the archived verdicts under `results/` and prints
the paper's tables: bare-model validity and PHACT feasible certification, the
bare-validator-vs-structural-contract impossible-goal results, the forgeability
ablation (S1-S4c), the cross-model comparison, the capability-floor probe
(Llama 3.1 8B), the feedback-form ablation, and the two fairness baselines. The
printed numbers match the manuscript.

Because the live experiment calls a stochastic language model, re-running it
from scratch produces numbers within the reported variance but not bit-identical.
The archived runs are the canonical reference for the specific figures reported.

---

## Using the engines directly

```python
# Chirp mass (GW150914 component masses)
from phact_physics.domains.astrophysics import check_chirp_mass

r = check_chirp_mass(m1_msun=35.6, m2_msun=30.6, chirp_mass_msun=28.6)
print(r.passed, r.metrics["chirp_mass_msun"])
```

```python
# The structural contract: you supply only the fixed inputs and the target.
# The contract derives the true value; you cannot supply it yourself.
from phact_physics.domains.astrophysics import verify_binary_target

r = verify_binary_target(quantity="chirp_mass", m1_msun=36, m2_msun=29,
                         target_value=50.0)
print(r.passed)   # False: the true chirp mass of (36, 29) is 28.1 Msun, not 50
```

```python
# DNA melting temperature (SantaLucia 1998 + Owczarzy 2004 salt correction)
from phact_physics import get_validator

v = get_validator("biochem")
r = v.validate({"sequence": "GCGCGCATGCGCGC", "target_temp_c": 65.0,
                "strand_conc_nm": 250.0})
print(r.passed, r.metrics["calculated_tm_C"])
```

```python
# RC filter cutoff
from phact_physics.domains.electrical import check_rc_filter

r = check_rc_filter(resistance_ohm=1000, capacitance_uf=0.159,
                    target_cutoff_hz=1000)
print(r.passed, r.metrics)
```

```python
# Colloidal stability / DLVO (drug-formulation domain)
from phact_physics.domains.drug_formulation import check_colloidal_stability

r = check_colloidal_stability(particle_radius_nm=100.0, zeta_potential_mv=-30.0,
                              salt_concentration_mm=150.0)
print(r.passed, r.metrics)
```

---

## Repository layout

```
phact-reproducibility/
├── phact_physics/              Deterministic physics engines (stdlib only)
│   ├── physics_engine.py       Drone + DNA validators (get_validator)
│   ├── domains/
│   │   ├── astrophysics.py     Chirp mass, GW strain, Peters time, ISCO,
│   │   │                         and verify_binary_target (structural contract)
│   │   ├── electrical.py       RC filter, PCB thermal, power, antenna
│   │   ├── drug_formulation.py DLVO stability, Flory-Huggins, permeability
│   │   └── nrt.py              Additional structural-record validators
│   └── contracts/              Structural contracts: dependency graphs (derive-from-inputs)
├── calibration/
│   └── test_calibration.py     Engine-vs-literature validation
├── tests/
│   └── test_adversarial_impossible.py  Impossible goals are truly impossible
├── results/                    Archived run verdicts (rev3_*, rev4_*, ablation_*)
├── demo.py                     30-second tour of the certification surface
├── regenerate_tables.py        Archived verdicts -> the paper's tables
├── PROVENANCE.md               Models, decoding settings, and archive provenance
├── Makefile                    make demo / verify / tables / calibrate / test
└── pyproject.toml              No runtime dependencies
```

---

## For the manuscript

**Code availability.** The deterministic physics engines and structural
contracts (directed dependency graphs) that constitute the certification oracle,
together with the calibration tests and the scripts that regenerate the reported
tables, are available at this repository under the Apache License, Version 2.0.
The engines depend only on the Python standard library.

**Data availability.** The archived output of the experimental runs reported in
the paper is under `results/` and is sufficient to regenerate the tables via
`make tables`. The data is available under CC BY 4.0.

**Reproducibility note.** The engines are deterministic and reproduce the cited
literature within the tolerances stated in `calibration/`. The language-model
layer is stochastic; its procedure is described in Methods, and the specific
results reported in the paper are preserved in the archived runs.

---

## License

- **Code** - Apache License, Version 2.0. See [LICENSE](LICENSE).
- **Data and documentation** (`results/`, README, PROVENANCE) - Creative Commons
  Attribution 4.0 International (CC BY 4.0). See [LICENSE-DATA](LICENSE-DATA).

The Physics-Anchored Certification (PHACT) method is the subject of patent rights
held by Heysuvi Labs, LLC. The Apache-2.0 patent grant extends only to the code
released here; see [NOTICE](NOTICE).
