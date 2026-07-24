"""C3 verifier: Theorem 3.3 — APUB-M asymptotic correctness.

Claim: Under Assumptions A and B with finite skewness, APUB-M is 1st-order
asymptotically correct:

    beta(vartheta_hat_N^alpha, S_hat_N^alpha) >= (1-alpha) + O(N^{-1/2})

where beta is the coverage probability:
    beta(vartheta, S) = Pr(vartheta >= max_{x in S} mu(x) | P)

Evidence: Monte Carlo on a newsvendor problem with Gamma(2,1) demand.
  - For each trial: draw N samples, solve APUB-M to get (vartheta_hat, x_hat)
  - Compute true cost mu(x_hat) on a large independent test sample
  - Coverage event: vartheta_hat >= mu(x_hat)
  - Report coverage probability across N = 80..5000

Independent negative control: SAA-M (alpha=1, i.e. pure sample average)
should have coverage < (1-alpha) for small N (it is NOT asymptotically correct
in the coverage sense — SAA tends to under-estimate the true cost).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import gamma

from ..apub_core import apub_cvar, bootstrap_means


ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", ".openresearch", "artifacts", "c3")
os.makedirs(ARTIFACTS, exist_ok=True)

C_H = 1.0
C_B = 4.0
TRUE_Q = float(gamma.ppf(C_B / (C_H + C_B), 2, scale=1))
NOMINAL = 0.95
ALPHA = 0.05

rng_true = np.random.default_rng(99999)
_D_TRUE = rng_true.gamma(2, 1, size=500000)
TRUE_OPTIMAL_VALUE = float(np.mean(np.maximum(TRUE_Q - _D_TRUE, 0) * C_H +
                                    np.maximum(_D_TRUE - TRUE_Q, 0) * C_B))


def _newsvendor_cost(q: float, demands: np.ndarray) -> np.ndarray:
    return C_H * np.maximum(q - demands, 0.0) + C_B * np.maximum(demands - q, 0.0)


def _solve_apub_newsvendor(data: np.ndarray, alpha: float, M: int,
                           rng: np.random.Generator, q_grid: np.ndarray):
    """Solve APUB-M for the newsvendor problem.

    Returns (optimal_q, optimal_value) where optimal_value = APUB at optimal_q.
    Uses vectorised bootstrap across the q-grid.
    """
    N = len(data)
    costs = np.empty((N, len(q_grid)))
    for j, q in enumerate(q_grid):
        costs[:, j] = _newsvendor_cost(q, data)
    idx = rng.integers(0, N, size=(M, N))
    boot_costs = costs[idx]
    boot_means = boot_costs.mean(axis=1)
    apub_values = np.array([apub_cvar(boot_means[:, j], alpha) for j in range(len(q_grid))])
    j_star = int(np.argmin(apub_values))
    return float(q_grid[j_star]), float(apub_values[j_star])


def _solve_saa_newsvendor(data: np.ndarray, q_grid: np.ndarray):
    """Solve SAA-M (sample average approximation) for the newsvendor."""
    costs = np.array([_newsvendor_cost(q, data).mean() for q in q_grid])
    j_star = int(np.argmin(costs))
    return float(q_grid[j_star]), float(costs[j_star])


def _coverage_experiment():
    """Monte Carlo: coverage probability of APUB-M and SAA-M."""
    rng = np.random.default_rng(3303)
    sample_sizes = [80, 200, 500, 1000, 2000, 5000]
    n_trials = 300
    M_boot = 500
    q_grid = np.linspace(0.5 * TRUE_Q, 1.8 * TRUE_Q, 60)

    rows = []
    for N in sample_sizes:
        apub_covered = 0
        saa_covered = 0
        apub_values_list = []
        apub_decisions = []
        saa_decisions = []
        for trial in range(n_trials):
            demands = rng.gamma(2, 1, size=N)
            brng = np.random.default_rng(rng.integers(0, 2**31))

            q_apub, val_apub = _solve_apub_newsvendor(demands, ALPHA, M_boot, brng, q_grid)
            q_saa, val_saa = _solve_saa_newsvendor(demands, q_grid)

            test_rng = np.random.default_rng(rng.integers(0, 2**31))
            test_demands = test_rng.gamma(2, 1, size=20000)
            true_cost_apub = float(np.mean(_newsvendor_cost(q_apub, test_demands)))
            true_cost_saa = float(np.mean(_newsvendor_cost(q_saa, test_demands)))

            if val_apub >= true_cost_apub:
                apub_covered += 1
            if val_saa >= true_cost_saa:
                saa_covered += 1
            apub_values_list.append(val_apub)
            apub_decisions.append(q_apub)
            saa_decisions.append(q_saa)

        cp_apub = apub_covered / n_trials
        cp_saa = saa_covered / n_trials
        rows.append({
            "N": N, "alpha": ALPHA, "nominal": NOMINAL,
            "coverage_apub": round(cp_apub, 4),
            "coverage_saa": round(cp_saa, 4),
            "mean_apub_optval": round(float(np.mean(apub_values_list)), 4),
            "true_optval": round(TRUE_OPTIMAL_VALUE, 4),
            "mean_apub_q": round(float(np.mean(apub_decisions)), 4),
            "mean_saa_q": round(float(np.mean(saa_decisions)), 4),
            "true_q": round(TRUE_Q, 4),
            "n_trials": n_trials, "M_bootstrap": M_boot,
        })
        print(f"  N={N:5d}  APUB cov={cp_apub:.4f}  SAA cov={cp_saa:.4f}  "
              f"(nominal={NOMINAL:.2f})  APUB q={np.mean(apub_decisions):.3f}  "
              f"SAA q={np.mean(saa_decisions):.3f}  true q*={TRUE_Q:.3f}")
    return rows


def run() -> dict:
    print("=" * 70)
    print("CLAIM C3: Theorem 3.3 — APUB-M asymptotic correctness")
    print("=" * 70)
    print(f"  Newsvendor: c_h={C_H}, c_b={C_B}, demand~Gamma(2,1)")
    print(f"  True q*={TRUE_Q:.4f}, True optimal value={TRUE_OPTIMAL_VALUE:.4f}")
    print(f"  alpha={ALPHA}, nominal={NOMINAL}")
    print()

    rows = _coverage_experiment()

    with open(os.path.join(ARTIFACTS, "coverage_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    large_N = max(rows, key=lambda r: r["N"])
    small_N = rows[0]
    apub_above_nominal = all(r["coverage_apub"] >= NOMINAL - 0.05 for r in rows if r["N"] >= 200)
    apub_at_large = large_N["coverage_apub"]
    saa_at_large = large_N["coverage_saa"]
    saa_below_apub = saa_at_large < apub_at_large

    print(f"\n  APUB coverage >= nominal-0.05 for N>=200: {apub_above_nominal}")
    print(f"  At N={large_N['N']}: APUB cov={apub_at_large}, SAA cov={saa_at_large}")
    print(f"  SAA coverage < APUB coverage (negative control): {saa_below_apub}")

    ok = apub_above_nominal and saa_below_apub
    verdict = "VERIFIED" if ok else "FAIL"

    result = {
        "claim": "C3: Theorem 3.3 — APUB-M 1st-order asymptotic correctness",
        "source": "beta(vartheta_hat, S_hat) >= (1-alpha) + O(N^{-1/2})",
        "verdict": verdict,
        "problem": "newsvendor",
        "true_q": round(TRUE_Q, 6),
        "true_optimal_value": round(TRUE_OPTIMAL_VALUE, 6),
        "alpha": ALPHA,
        "nominal_level": NOMINAL,
        "apub_coverage_above_nominal": bool(apub_above_nominal),
        "apub_coverage_at_largest_N": apub_at_large,
        "saa_coverage_at_largest_N": saa_at_large,
        "saa_below_apub_negative_control": bool(saa_below_apub),
        "coverage_data": rows,
        "raw_data": "artifacts/c3/coverage_raw.csv",
    }

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> C3 VERDICT: {verdict}")
    return result
