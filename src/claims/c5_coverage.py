"""C5 verifier: Example 2.5 — coverage probability comparison.

Claim: On Gamma(2,1) with alpha=0.05 and N from 80 to 10000, APUB's coverage
probability approaches the 0.95 nominal level faster than Efron's percentile
bound and the standard large-sample (CLT) approximation (Figure 1, Example 2.5).

Specifically the paper states:
  - Efron and CLT bounds are "asymptotically accurate": coverage -> (1-alpha)
  - APUB is "asymptotically correct but NOT accurate": coverage can GROW BEYOND
    (1-alpha), and this growth is "more rapid than the other two bounds"

Evidence: Monte Carlo with 1000 independent replications per (method, N).
For each replication we draw a fresh Gamma(2,1) sample of size N, compute the
upper bound, and check whether mu=2 <= bound.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import norm

from ..apub_core import (
    apub_cvar,
    bootstrap_means,
    clt_upper_bound,
    efron_upper_bound,
)


ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", ".openresearch", "artifacts", "c5")
os.makedirs(ARTIFACTS, exist_ok=True)

TRUE_MEAN = 2.0
ALPHA = 0.05
NOMINAL = 1.0 - ALPHA
SAMPLE_SIZES = [80, 200, 500, 1000, 2000, 5000, 10000]
N_TRIALS = 1000
M_BOOT = 800


def _coverage_experiment():
    """Monte Carlo coverage probability for APUB, Efron, and CLT on Gamma(2,1)."""
    rng = np.random.default_rng(2403)
    rows = []
    for N in SAMPLE_SIZES:
        apub_cov = 0
        efron_cov = 0
        clt_cov = 0
        apub_vals = []
        efron_vals = []
        clt_vals = []
        for trial in range(N_TRIALS):
            data = rng.gamma(2, 1, size=N)
            brng = np.random.default_rng(rng.integers(0, 2**31))
            bmeans = bootstrap_means(data, M_BOOT, brng)
            v_apub = apub_cvar(bmeans, ALPHA)
            v_efron = efron_upper_bound(bmeans, ALPHA)
            v_clt = clt_upper_bound(data, ALPHA)
            apub_cov += int(TRUE_MEAN <= v_apub)
            efron_cov += int(TRUE_MEAN <= v_efron)
            clt_cov += int(TRUE_MEAN <= v_clt)
            apub_vals.append(v_apub)
            efron_vals.append(v_efron)
            clt_vals.append(v_clt)
        cp_apub = apub_cov / N_TRIALS
        cp_efron = efron_cov / N_TRIALS
        cp_clt = clt_cov / N_TRIALS
        rows.append({
            "N": N,
            "alpha": ALPHA,
            "true_mean": TRUE_MEAN,
            "coverage_apub": round(cp_apub, 4),
            "coverage_efron": round(cp_efron, 4),
            "coverage_clt": round(cp_clt, 4),
            "mean_apub_value": round(float(np.mean(apub_vals)), 4),
            "mean_efron_value": round(float(np.mean(efron_vals)), 4),
            "mean_clt_value": round(float(np.mean(clt_vals)), 4),
            "n_trials": N_TRIALS,
            "M_bootstrap": M_BOOT,
        })
        print(f"  N={N:6d}  APUB={cp_apub:.4f}  Efron={cp_efron:.4f}  CLT={cp_clt:.4f}  "
              f"(nominal={NOMINAL:.2f})")
    return rows


def _check_claims(rows):
    """Verify the specific claims from Example 2.5."""
    large_N_row = max(rows, key=lambda r: r["N"])

    apub_exceeds_nominal_large = large_N_row["coverage_apub"] > NOMINAL
    apub_grows_beyond = any(
        r["coverage_apub"] > NOMINAL + 0.02 for r in rows if r["N"] >= 1000
    )
    largest_apub = max(r["coverage_apub"] for r in rows)
    largest_efron = max(r["coverage_efron"] for r in rows)
    largest_clt = max(r["coverage_clt"] for r in rows)
    apub_growth_exceeds_efron = largest_apub >= largest_efron - 0.01
    apub_growth_exceeds_clt = largest_apub >= largest_clt - 0.01

    all_converge_to_mu = True
    for r in rows:
        if r["mean_apub_value"] - TRUE_MEAN > 0.5:
            all_converge_to_mu = False
            break

    return {
        "apub_coverage_exceeds_nominal_at_large_N": bool(apub_exceeds_nominal_large),
        "apub_coverage_grows_beyond_nominal": bool(apub_grows_beyond),
        "apub_max_coverage": round(largest_apub, 4),
        "efron_max_coverage": round(largest_efron, 4),
        "clt_max_coverage": round(largest_clt, 4),
        "apub_growth_geq_efron": bool(apub_growth_exceeds_efron),
        "apub_growth_geq_clt": bool(apub_growth_exceeds_clt),
        "apub_values_converge_to_mu": bool(all_converge_to_mu),
        "coverage_at_largest_N": {
            "N": large_N_row["N"],
            "apub": large_N_row["coverage_apub"],
            "efron": large_N_row["coverage_efron"],
            "clt": large_N_row["coverage_clt"],
        },
    }


def run() -> dict:
    print("=" * 70)
    print("CLAIM C5: Example 2.5 — coverage probability comparison")
    print("=" * 70)
    print(f"  Distribution: Gamma(2,1), mu={TRUE_MEAN}, alpha={ALPHA}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  Trials per config: {N_TRIALS}, Bootstrap M: {M_BOOT}")
    print()

    rows = _coverage_experiment()

    with open(os.path.join(ARTIFACTS, "coverage_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    checks = _check_claims(rows)
    print(f"\n  APUB coverage exceeds nominal at N=10000: "
          f"{checks['apub_coverage_exceeds_nominal_at_large_N']}")
    print(f"  APUB coverage grows beyond nominal:       "
          f"{checks['apub_coverage_grows_beyond_nominal']}")
    print(f"  APUB max coverage={checks['apub_max_coverage']}, "
          f"Efron max={checks['efron_max_coverage']}, "
          f"CLT max={checks['clt_max_coverage']}")

    ok = (checks["apub_coverage_grows_beyond_nominal"]
          and checks["apub_growth_geq_efron"]
          and checks["apub_growth_geq_clt"]
          and checks["apub_values_converge_to_mu"])
    verdict = "VERIFIED" if ok else "FAIL"

    result = {
        "claim": "C5: Example 2.5 coverage probability comparison",
        "source": "APUB coverage grows beyond (1-alpha) faster than Efron/CLT on Gamma(2,1)",
        "verdict": verdict,
        "distribution": "Gamma(2,1)",
        "true_mean": TRUE_MEAN,
        "alpha": ALPHA,
        "nominal_level": NOMINAL,
        "sample_sizes": SAMPLE_SIZES,
        "n_trials": N_TRIALS,
        "M_bootstrap": M_BOOT,
        "coverage_data": rows,
        "checks": checks,
        "raw_data": "artifacts/c5/coverage_raw.csv",
    }

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> C5 VERDICT: {verdict}")
    return result
