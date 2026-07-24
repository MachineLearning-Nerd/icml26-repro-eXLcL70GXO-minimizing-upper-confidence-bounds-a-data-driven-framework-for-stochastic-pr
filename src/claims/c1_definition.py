"""C1 verifier: APUB definition (Definition 2.2).

Claim: U^apub[mu|P_hat_N] := (1/alpha) * integral_0^alpha U^efron_tau d_tau,
integrating Efron's percentile upper bound over tau in [0, alpha].

Verification approach:
  1. Show the implementation faithfully computes the integral (Definition 2.2)
     via fine-grid numerical quadrature of Efron's quantile function.
  2. Show it equals the CVaR reformulation (Proposition 2.3 / Rockafellar-Uryasev)
     to within Monte-Carlo error — an independent implementation of the same
     mathematical object via a different formula.
  3. Verify structural properties derived from the definition:
     - APUB >= sample mean  (Remark 2.4: APUB^alpha >= APUB^1 = mu_hat_N)
     - APUB >= Efron's bound at level alpha  (averaging over [0,alpha] >= value at alpha)
     - APUB monotonically decreasing in alpha  (Remark 2.4)
  4. Verify across multiple distributions (Gamma, Normal, Exponential, Uniform,
     Bimodal mixture) — not just the paper's single Gamma(2,1).
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..apub_core import (
    apub,
    apub_cvar,
    apub_integral,
    bootstrap_means,
    clt_upper_bound,
    efron_upper_bound,
)


ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", ".openresearch", "artifacts", "c1")
os.makedirs(ARTIFACTS, exist_ok=True)


def _verify_integral_equals_cvar():
    """The integral form (Definition 2.2) must match the CVaR form (Prop. 2.3)."""
    rng = np.random.default_rng(2024)
    results = []
    configs = [
        ("Gamma(2,1)", lambda r: r.gamma(2, 1, size=500)),
        ("Normal(0,1)", lambda r: r.standard_normal(500)),
        ("Exponential(1)", lambda r: r.exponential(1, size=500)),
        ("Uniform(0,1)", lambda r: r.uniform(0, 1, size=500)),
        ("Bimodal", lambda r: np.where(r.random(500) < 0.5,
                                       r.normal(-2, 1, 500), r.normal(2, 1, 500))),
    ]
    alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
    max_diff = 0.0
    for name, sampler in configs:
        for alpha in alphas:
            data = sampler(rng)
            means = bootstrap_means(data, M=5000, rng=np.random.default_rng(100 + hash(name) % 1000))
            v_integral = apub_integral(means, alpha, n_grid=5000)
            v_cvar = apub_cvar(means, alpha)
            diff = abs(v_integral - v_cvar)
            max_diff = max(max_diff, diff)
            results.append({
                "distribution": name,
                "alpha": alpha,
                "apub_integral": round(v_integral, 8),
                "apub_cvar": round(v_cvar, 8),
                "abs_difference": round(diff, 8),
            })
    return results, max_diff


def _verify_properties():
    """Verify structural properties from Remark 2.4."""
    rng = np.random.default_rng(2024)
    results = []
    configs = [
        ("Gamma(2,1)", lambda r: r.gamma(2, 1, size=200)),
        ("Normal(3,2)", lambda r: r.normal(3, 2, size=200)),
        ("Exponential(1)", lambda r: r.exponential(1, size=200)),
    ]
    alphas = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]
    all_geq_mean = True
    all_geq_efron = True
    all_monotone = True
    for name, sampler in configs:
        data = sampler(rng)
        sm = float(data.mean())
        prev = float("inf")
        for alpha in alphas:
            res = apub(data, alpha, M=3000, rng=np.random.default_rng(42))
            v = res["apub"]
            efron = res["efron"]
            geq_mean = v >= sm - 1e-10
            geq_efron = v >= efron - 1e-10
            monotone = v <= prev + 1e-10
            all_geq_mean = all_geq_mean and geq_mean
            all_geq_efron = all_geq_efron and geq_efron
            all_monotone = all_monotone and monotone
            results.append({
                "distribution": name,
                "alpha": alpha,
                "apub": round(v, 6),
                "efron": round(efron, 6),
                "sample_mean": round(sm, 6),
                "apub_geq_mean": bool(geq_mean),
                "apub_geq_efron": bool(geq_efron),
                "monotone_decr_alpha": bool(monotone),
            })
            prev = v
    return results, all_geq_mean, all_geq_efron, all_monotone


def _verify_apub_equals_one_at_alpha1():
    """At alpha=1, APUB must equal the sample mean (Remark 2.4).

    U^APUB^1 = (1/1) integral_0^1 U^Efron_tau d_tau = E[mu*] = mu_hat_N.
    With finite M bootstrap resamples the bootstrap-mean average has Monte-Carlo
    noise O(sigma / (sqrt(N) * sqrt(M))), so we use a tolerance calibrated to
    M=20000 (SE ~ 0.0005 for Gamma(2,1), N=300).
    """
    rng = np.random.default_rng(99)
    data = rng.gamma(2, 1, size=300)
    res = apub(data, 1.0, M=20000, rng=np.random.default_rng(7))
    diff = abs(res["apub"] - res["sample_mean"])
    return {"apub_alpha1": round(res["apub"], 8),
            "sample_mean": round(res["sample_mean"], 8),
            "abs_diff": round(diff, 10),
            "equals_sample_mean": bool(diff < 0.002)}


def run() -> dict:
    print("=" * 70)
    print("CLAIM C1: APUB Definition (Definition 2.2)")
    print("=" * 70)

    integral_results, max_diff = _verify_integral_equals_cvar()
    print(f"\n[1] Integral form vs CVaR form (Proposition 2.3):")
    print(f"    Max absolute difference across 25 configs: {max_diff:.2e}")
    integral_ok = max_diff < 0.01

    prop_results, geq_mean, geq_efron, monotone = _verify_properties()
    print(f"\n[2] Structural properties (Remark 2.4):")
    print(f"    APUB >= sample mean (all configs):   {geq_mean}")
    print(f"    APUB >= Efron bound (all configs):   {geq_efron}")
    print(f"    APUB monotonically decreasing in a:  {monotone}")

    alpha1_result = _verify_apub_equals_one_at_alpha1()
    print(f"\n[3] APUB at alpha=1 equals sample mean:")
    print(f"    APUB(1)={alpha1_result['apub_alpha1']}, "
          f"mean={alpha1_result['sample_mean']}, diff={alpha1_result['abs_diff']}")

    all_ok = integral_ok and geq_mean and geq_efron and monotone and alpha1_result["equals_sample_mean"]

    result = {
        "claim": "C1: APUB Definition 2.2",
        "source": "U^apub = (1/alpha) integral_0^alpha U^efron_tau d_tau",
        "verdict": "VERIFIED" if all_ok else "FAIL",
        "integral_vs_cvar_max_diff": round(max_diff, 8),
        "integral_matches_cvar": bool(integral_ok),
        "apub_geq_sample_mean": bool(geq_mean),
        "apub_geq_efron": bool(geq_efron),
        "apub_monotone_decreasing_alpha": bool(monotone),
        "apub_equals_mean_at_alpha1": bool(alpha1_result["equals_sample_mean"]),
        "n_distributions_tested": 5,
        "n_alphas_tested": 5,
        "detail_integral_vs_cvar": integral_results,
        "detail_properties": prop_results,
        "detail_alpha1": alpha1_result,
    }

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> C1 VERDICT: {result['verdict']}")
    return result
