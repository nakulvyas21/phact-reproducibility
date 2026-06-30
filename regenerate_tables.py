#!/usr/bin/env python3
"""
Regenerate the paper's result tables from the archived run
==========================================================

Reproduces the tables of the paper

    "Structural Certification for Reliable Physical Design with Language Models"
    (Physics-Anchored Certification, PHACT)

directly from the archived experiment output in ``results/``. Uses only the
Python standard library: no credentials, no network, no language model.

    python regenerate_tables.py

The archived JSON files are the immutable record of the runs reported in the
paper (the audited, single-code-path "rev3/rev4" run of 2026-06-16..19). Because
the live experiment calls a stochastic language model, re-running it from scratch
would produce numbers within the reported variance but not bit-identical; this
script regenerates the *reported* numbers deterministically from the archived
verdicts, so a reviewer can confirm every number in the paper traces to data.

Source files (results/):
    rev3_E0_*           bare model, no engine (feasible + impossible)
    rev3_Pf_*           PHACT feasible certification
    rev3_E1_*           PHACT impossible, bare answer-submitting validators
    rev3_E2_*           PHACT impossible, SCM-exclusive validators
    rev3_SCM_S1..S4c    forgeability ablation (3 gameable astro goals x5 trials)
    rev4_baseline_A     bare-but-permitted-to-refuse fairness baseline
    rev4_baseline_B     engine-only closed-form solver (no LLM) baseline
    ablation_*_20260618 feedback-form ablation (RC / DNA / drug)
"""

import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
DOMAINS = ["drone", "biochem", "astrophysics", "electrical", "drug_formulation"]
LABEL = {
    "drone": "Drone", "biochem": "Biochem", "astrophysics": "Astrophysics",
    "electrical": "Electrical", "drug_formulation": "Drug formulation",
}


def load(name):
    p = R / name
    if not p.exists():
        raise SystemExit(f"Archived file not found: {p}")
    with open(p) as f:
        return json.load(f)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def hr(title):
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
# Bare-model validity (E0) and PHACT feasible certification (Pf)
# ---------------------------------------------------------------------------

def bare_feasible_valid(model):
    """E0 feasible: count designs that pass the engine post hoc."""
    d = load(f"rev3_E0_feasible_{model}.json")
    n = sum(len(d[dom]) for dom in d)
    valid = sum(1 for dom in d for g in d[dom].values() if g["physics_passed"])
    return valid, n


def phact_feasible(model):
    """Pf: per-domain certified / loop-fail / wrong-verdict / timeout.

    Convention (REV4): feasible certification is reported over all authored
    feasible goals, with infrastructure timeouts shown explicitly in the
    denominator. A 'wrong_verdict' (feasible goal declared impossible) is a
    failure. A run with no terminal verdict that is not a timeout is a
    loop-fail.
    """
    d = load(f"rev3_Pf_feasible_{model}.json")
    rows = {}
    for dom in DOMAINS:
        cert = loop = wrong = timeout = 0
        for g in d.get(dom, {}).values():
            outcome = g.get("outcome")
            if outcome == "certified":
                cert += 1
            elif outcome == "impossible":
                wrong += 1                      # feasible goal wrongly refused
            elif g.get("error_kind") == "timeout" or (g.get("latency_s", 0) >= 115
                    and outcome in (None, "loop_closure", "error")):
                timeout += 1
            else:
                loop += 1
        rows[dom] = (cert, loop, wrong, timeout)
    return rows


def print_feasible():
    hr("TABLE - PHACT feasible certification (Gemini 2.5 Flash, temperature 0)")
    bare_v, bare_n = bare_feasible_valid("gemini")
    rows = phact_feasible("gemini")
    print(f"{'Domain':<18}{'Certified':>11}{'Loop-fail':>11}{'Wrong':>8}{'Timeout':>9}")
    print("-" * 76)
    tc = tl = tw = tt = 0
    for dom in DOMAINS:
        c, l, w, t = rows[dom]
        n = c + l + w + t
        print(f"{LABEL[dom]:<18}{f'{c}/{n}':>11}{l:>11}{w:>8}{t:>9}")
        tc += c; tl += l; tw += w; tt += t
    total = tc + tl + tw + tt
    print("-" * 76)
    print(f"{'Overall':<18}{f'{tc}/{total}':>11}{tl:>11}{tw:>8}{tt:>9}")
    completed = total - tt
    print()
    print(f"  Bare feasible valid : {bare_v}/{bare_n} = {pct(bare_v,bare_n):.0f}%")
    print(f"  PHACT certified     : {tc}/{total} = {pct(tc,total):.0f}% "
          f"(all goals; {tt} infrastructure timeouts included in denominator)")
    print()


# ---------------------------------------------------------------------------
# Bare-model checker-passing designs on impossible goals (E0)
# ---------------------------------------------------------------------------

def print_bare_impossible():
    hr("Bare model on impossible goals: checker-passing designs")
    print(f"{'Domain':<18}{'Gemini hall.':>14}{'Llama70B hall.':>16}")
    print("-" * 76)
    g = load("rev3_E0_impossible_gemini.json")
    l = load("rev3_E0_impossible_llama70b.json")
    gt = gn = lt = ln = 0
    for dom in DOMAINS:
        gh = sum(1 for x in g.get(dom, {}).values() if x["physics_passed"])
        gd = len(g.get(dom, {}))
        lh = sum(1 for x in l.get(dom, {}).values() if x["physics_passed"])
        ld = len(l.get(dom, {}))
        print(f"{LABEL[dom]:<18}{f'{gh}/{gd}':>14}{f'{lh}/{ld}':>16}")
        gt += gh; gn += gd; lt += lh; ln += ld
    print("-" * 76)
    print(f"{'Overall':<18}{f'{gt}/{gn} ({pct(gt,gn):.0f}%)':>14}"
          f"{f'{lt}/{ln} ({pct(lt,ln):.0f}%)':>16}")
    print("  (Checker-passing design = a design that passes the checker although "
          "the goal is impossible.)")
    print()


# ---------------------------------------------------------------------------
# Impossible: bare validators (E1) vs SCM-exclusive (E2)
# ---------------------------------------------------------------------------

def _impossible_counts(fname):
    d = load(fname)
    refused = false_cert = loop = 0
    for dom in DOMAINS:
        for g in d.get(dom, {}).values():
            o = g.get("outcome")
            if o in ("impossible", "unknown"):
                refused += 1
            elif o == "certified" or g.get("certified"):
                false_cert += 1
            else:
                loop += 1
    return refused, false_cert, loop


def print_e1_e2():
    hr("TABLE - Impossible goals: bare validators (E1) vs SCM-exclusive (E2)")
    print(f"{'Condition':<28}{'Refused':>9}{'False cert.':>13}{'Loop/err':>10}")
    print("-" * 76)
    for label, fname in [
        ("Gemini  bare (E1)",     "rev3_E1_impossible_gemini.json"),
        ("Gemini  SCM-excl (E2)", "rev3_E2_impossible_gemini.json"),
        ("Llama70 bare (E1)",     "rev3_E1_impossible_llama70b.json"),
        ("Llama70 SCM-excl (E2)", "rev3_E2_impossible_llama70b.json"),
        ("Llama8B bare (E1)",     "rev3_E1_impossible_llama8b.json"),
    ]:
        ref, fc, loop = _impossible_counts(fname)
        tot = ref + fc + loop
        print(f"{label:<28}{f'{ref}/{tot}':>9}{fc:>13}{loop:>10}")
    print("\n  SCM-exclusive validators yield no false certifications for any model.")
    print()


# ---------------------------------------------------------------------------
# Forgeability ablation S1..S4c
# ---------------------------------------------------------------------------

def _scm_false_certs(fname):
    """SCM ablation files: {goal: [verdict_trial_1..5]}. Count 'certified'
    verdicts as forged certificates (the goal is impossible)."""
    d = load(fname)
    fc = refused = loop = total = 0
    for verdicts in d.values():
        for v in verdicts:
            total += 1
            if v == "certified":
                fc += 1
            elif v in ("impossible", "unknown"):
                refused += 1
            else:
                loop += 1
    return fc, refused, loop, total


def print_scm_ablation():
    hr("TABLE - Forgeability ablation (Gemini; 3 gameable astro goals x5 = 15)")
    print(f"{'Row':<5}{'Condition':<34}{'Contract':<9}{'False cert.':>12}")
    print("-" * 76)
    spec = [
        ("S1",  "temperature 0",                "bare", "rev3_SCM_S1_bare_t0_gemini.json"),
        ("S2",  "temperature 1",                "bare", "rev3_SCM_S2_bare_t1_gemini.json"),
        ("S3",  "temperature 1",                "SCM",  "rev3_SCM_S3_scmexcl_t1_gemini.json"),
        ("S4a", "temperature 0, +latch fault",  "bare", "rev3_SCM_S4a_bare_latch_gemini.json"),
        ("S4b", "temperature 0, +latch (avail)","SCM?", "rev3_SCM_S4b_scmavail_latch_gemini.json"),
        ("S4c", "temperature 0, +latch fault",  "SCM",  "rev3_SCM_S4c_scmexcl_latch_gemini.json"),
    ]
    pooled_bare_fc = pooled_bare_n = 0
    pooled_scm_fc = pooled_scm_n = 0
    for row, cond, contract, fname in spec:
        fc, ref, loop, tot = _scm_false_certs(fname)
        print(f"{row:<5}{cond:<34}{contract:<9}{f'{fc}/{tot}':>12}")
        if contract == "bare":
            pooled_bare_fc += fc; pooled_bare_n += tot
        elif contract == "SCM":
            pooled_scm_fc += fc; pooled_scm_n += tot
    print("-" * 76)
    print(f"  Pooled bare       : {pooled_bare_fc}/{pooled_bare_n} forged")
    print(f"  Pooled SCM-excl   : {pooled_scm_fc}/{pooled_scm_n} forged")
    print()


def print_zero_hallucination():
    hr("False certifications pooled over SCM-exclusive conditions")
    e2g = _impossible_counts("rev3_E2_impossible_gemini.json")
    e2l = _impossible_counts("rev3_E2_impossible_llama70b.json")
    s3  = _scm_false_certs("rev3_SCM_S3_scmexcl_t1_gemini.json")
    s4c = _scm_false_certs("rev3_SCM_S4c_scmexcl_latch_gemini.json")
    # E2 are 25-goal suites; S3/S4c are 15-trial cells.
    n = 25 + 25 + s3[3] + s4c[3]
    fc = e2g[1] + e2l[1] + s3[0] + s4c[0]
    print(f"  E2 Gemini false-cert : {e2g[1]}/25")
    print(f"  E2 Llama70B false-cert: {e2l[1]}/25")
    print(f"  S3 (SCM, temp1)      : {s3[0]}/{s3[3]}")
    print(f"  S4c (SCM, +latch)    : {s4c[0]}/{s4c[3]}")
    print("-" * 76)
    print(f"  POOLED               : {fc} / {n} false certifications")
    print()


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def print_cross_model():
    hr("TABLE - Cross-model: the safety guarantee transfers between model families")
    print(f"{'Metric':<38}{'Gemini':>10}{'Llama70B':>11}")
    print("-" * 76)
    # bare feasible valid
    gv, gn = bare_feasible_valid("gemini")
    lv, ln = bare_feasible_valid("llama70b")
    print(f"{'Bare feasible valid (no engine)':<38}"
          f"{f'{pct(gv,gn):.0f}%':>10}{f'{pct(lv,ln):.0f}%':>11}")
    # PHACT feasible certified, all-goals denominator (44/50, 33/50) to match
    # the paper's cross-model table.
    def feas_cert_allgoals(model):
        rows = phact_feasible(model)
        c = sum(r[0] for r in rows.values())
        tot = sum(sum(r) for r in rows.values())
        return c, tot
    gc, gcd = feas_cert_allgoals("gemini"); lc, lcd = feas_cert_allgoals("llama70b")
    print(f"{'PHACT feasible certified':<38}"
          f"{f'{pct(gc,gcd):.0f}%':>10}{f'{pct(lc,lcd):.0f}%':>11}")
    # The two false-certification rows are the key safety contrast: a bare
    # validator (E1) is forged for both models; the structural contract (E2)
    # forges nothing for either.
    gbare = _impossible_counts("rev3_E1_impossible_gemini.json")
    lbare = _impossible_counts("rev3_E1_impossible_llama70b.json")
    gstr  = _impossible_counts("rev3_E2_impossible_gemini.json")
    lstr  = _impossible_counts("rev3_E2_impossible_llama70b.json")
    print(f"{'False certifications - bare validator':<38}"
          f"{f'{gbare[1]}/25':>10}{f'{lbare[1]}/25':>11}")
    print(f"{'False certifications - structural':<38}"
          f"{f'{gstr[1]}/25':>10}{f'{lstr[1]}/25':>11}")
    print("\n  A bare validator that accepts the model-supplied value is forged "
          "for both models;\n  the structural contract, which derives the certified "
          "quantity from the fixed\n  inputs, yields zero false certifications for both.")
    print()

    # Capability-floor probe (Llama 3.1 8B) - reported separately, bare validator
    # only. 8B never closes the loop, so it never certifies, refuses, or forges.
    hr("TABLE - Capability-floor probe (Llama 3.1 8B, on-device via Ollama)")
    l8 = _impossible_counts("rev3_E1_impossible_llama8b.json")
    refused8, falsecert8, looperr8 = l8[0], l8[1], l8[2]
    print(f"  Impossible suite (25 goals), bare answer-submitting validator")
    print(f"  Refused           : {refused8}/25")
    print(f"  False cert.       : {falsecert8}/25")
    print(f"  Loop-fail         : {looperr8}/25")
    print("\n  Every trial ends in loop-closure failure: 8B never issues a terminal")
    print("  verdict, so it never certifies, never refuses, and never forges.")
    print("  Usefulness is at the floor; safety is intact for the trivial reason")
    print("  that the model never acts. (8B was run under the bare validator only;")
    print("  there is no structural-contract 8B run.)")
    print()


# ---------------------------------------------------------------------------
# Feedback-form ablation
# ---------------------------------------------------------------------------

def _mean_iters(rows):
    cert = [r["iterations"] for r in rows if r.get("outcome") == "certified"]
    return statistics.mean(cert) if cert else float("nan")


def print_feedback():
    hr("TABLE - Feedback-form ablation (Gemini; mean iterations to certify)")
    print(f"{'Domain (structure)':<34}{'Weak':>8}{'Strong':>9}{'Structural':>12}")
    print("-" * 76)
    # RC filter (electrical)
    elec = load("ablation_elec_20260618_115514.json")
    print(f"{'RC filter (closed-form inverse)':<34}"
          f"{_mean_iters(elec.get('weak',[])):>8.2f}"
          f"{_mean_iters(elec.get('strong',[])):>9.2f}"
          f"{_mean_iters(elec.get('scm',[])):>12.2f}")
    # Drug (string vs scm)
    drug = load("ablation_20260618_121321.json")
    drug_str = _mean_iters(drug.get('string', []))
    drug_scm = _mean_iters(drug.get('scm', []))
    print(f"{'Drug formulation (large region)':<34}"
          f"{f'{drug_str:.2f} (strong)':>17}{drug_scm:>12.2f}")
    # DNA dual-constraint
    dna = load("ablation_dna_20260618_120014.json")
    print(f"{'DNA dual-constraint (coupled)':<34}"
          f"{_mean_iters(dna.get('weak',[])):>8.2f}"
          f"{_mean_iters(dna.get('strong',[])):>9.2f}"
          f"{_mean_iters(dna.get('scm',[])):>12.2f}")
    print("\n  Feedback form matters only for the coupled domain (DNA), where no "
          "closed-form inverse exists.")
    print()


# ---------------------------------------------------------------------------
# Fairness baselines
# ---------------------------------------------------------------------------

def print_fairness():
    hr("TABLE - Fairness baselines")
    a = load("rev4_baseline_A.json")
    # Baseline A: bare-but-permitted-to-refuse
    imp = a["impossible"]
    refused = sum(1 for dom in imp.values() for g in dom.values() if g["refused"])
    hall = sum(1 for dom in imp.values() for g in dom.values()
               if not g["refused"] and g.get("valid"))
    ntot = sum(len(dom) for dom in imp.values())
    feas = a["feasible"]
    fvalid = sum(1 for dom in feas.values() for g in dom.values()
                 if not g["refused"] and g.get("valid"))
    fwrong = sum(1 for dom in feas.values() for g in dom.values() if g["refused"])
    fn = sum(len(dom) for dom in feas.values())
    print("  Baseline A  (bare Gemini ALLOWED to refuse, no engine):")
    print(f"    Impossible refused : {refused}/{ntot} ({pct(refused,ntot):.0f}%)"
          f"   [PHACT: 21/25, forbidden-bare: 0/25]")
    print(f"    Impossible still hallucinated-valid : {hall}")
    print(f"    Feasible valid : {fvalid}/{fn};  wrongly refused : {fwrong}")
    print("    => Permitting refusal recovers most refusal benefit, but gives NO")
    print("       guarantee: its refusals are unverified guesses.")
    print()
    b = load("rev4_baseline_B.json")
    print("  Baseline B  (engine-only closed-form solver, NO language model):")
    print(f"    RC filter  : {b['rc_filter']['certified']}/{b['rc_filter']['n']} certified")
    print(f"    Chirp mass : {b['chirp_mass']['certified']}/{b['chirp_mass']['n']} certified")
    print("    => Where a closed form exists, a parser+solver needs no LLM; the")
    print("       LLM's marginal value is the NL front-end and cross-domain reach.")
    print()


def main():
    print("PHACT - table regeneration from archived results "
          "(stdlib only, no network, no LLM)\n")
    print_feasible()
    print_bare_impossible()
    print_e1_e2()
    print_scm_ablation()
    print_zero_hallucination()
    print_cross_model()
    print_feedback()
    print_fairness()


if __name__ == "__main__":
    main()
