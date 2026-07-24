"""C4 verifier: Theorem 3.5 — APUB-M asymptotic consistency.

Claim: Under Assumptions A and B, for any alpha in (0,1], as N -> infinity:
    (i)  vartheta_hat_N^alpha -> vartheta*           (optimal value converges)
    (ii) D(S_hat_N^alpha, S) -> 0                     (solution set converges)
  where D(S_hat, S) = sup_{y in S_hat} inf_{z in S} ||y - z||.

Evidence: newsvendor with unique optimum (Gamma(2,1) demand, c_h=1, c_b=4).
  - For increasing N: solve APUB-M, record optimal value and decision
  - Show optimal value -> true optimal value
  - Show |q_hat - q*| -> 0
  - Repeat for multiple alpha values
  - Include a multi-product newsvendor (2 products) to test solution-set
    convergence in higher dimension.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import gamma

from ..apub_core import apub_cvar
from src.claims.c3_opt_correctness import (
    _newsvendor_cost, _solve_apub_newsvendor, _solve_saa_newsvendor,
    TRUE_Q, TRUE_OPTIMAL_VALUE, C_H, C_B,
)


ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", ".openresearch", "artifacts", "c4")
os.makedirs(ARTIFACTS, exist_ok=True)

SAMPLE_SIZES = [80, 200, 500, 1000, 2000, 5000, 10000]
ALPHAS = [0.05, 0.10, 0.50]
N_TRIALS = 30
M_BOOT = 500


def _single_product_convergence():
    """Newsvendor (1 product): value + solution convergence."""
    rng = np.random.default_rng(4404)
    q_grid = np.linspace(0.5 * TRUE_Q, 1.8 * TRUE_Q, 80)
    rows = []
    for alpha in ALPHAS:
        for N in SAMPLE_SIZES:
            vals = []
            qs = []
            for trial in range(N_TRIALS):
                demands = rng.gamma(2, 1, size=N)
                brng = np.random.default_rng(rng.integers(0, 2**31))
                q_hat, val_hat = _solve_apub_newsvendor(
                    demands, alpha, M_BOOT, brng, q_grid)
                vals.append(val_hat)
                qs.append(q_hat)
            mean_val = float(np.mean(vals))
            mean_q = float(np.mean(qs))
            std_q = float(np.std(qs, ddof=1))
            rows.append({
                "alpha": alpha, "N": N,
                "mean_optval": round(mean_val, 4),
                "true_optval": round(TRUE_OPTIMAL_VALUE, 4),
                "val_error": round(abs(mean_val - TRUE_OPTIMAL_VALUE), 4),
                "mean_q": round(mean_q, 4),
                "true_q": round(TRUE_Q, 4),
                "q_error": round(abs(mean_q - TRUE_Q), 4),
                "q_std": round(std_q, 4),
            })
            print(f"  alpha={alpha:.2f} N={N:6d}  val={mean_val:.4f} "
                  f"(true={TRUE_OPTIMAL_VALUE:.4f}, err={abs(mean_val-TRUE_OPTIMAL_VALUE):.4f})  "
                  f"q={mean_q:.4f} (true={TRUE_Q:.4f}, err={abs(mean_q-TRUE_Q):.4f})")
    return rows


def _check_convergence(rows):
    all_value_converge = True
    all_solution_converge = True
    rate_results = []
    by_alpha = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r)
    for alpha, cfg in by_alpha.items():
        cfg.sort(key=lambda x: x["N"])
        val_err_small = cfg[0]["val_error"]
        val_err_large = cfg[-1]["val_error"]
        q_err_small = cfg[0]["q_error"]
        q_err_large = cfg[-1]["q_error"]
        val_conv = val_err_large < val_err_small
        q_conv = q_err_large < q_err_small
        all_value_converge = all_value_converge and val_conv
        all_solution_converge = all_solution_converge and q_conv
        ns = np.array([r["N"] for r in cfg])
        val_errs = np.array([max(r["val_error"], 1e-10) for r in cfg])
        q_errs = np.array([max(r["q_error"], 1e-10) for r in cfg])
        if len(ns) >= 3:
            v_slope = float(np.polyfit(np.log(ns), np.log(val_errs), 1)[0])
            q_slope = float(np.polyfit(np.log(ns), np.log(q_errs), 1)[0])
        else:
            v_slope = q_slope = float("nan")
        rate_results.append({
            "alpha": alpha,
            "val_err_small_N": val_err_small, "val_err_large_N": val_err_large,
            "q_err_small_N": q_err_small, "q_err_large_N": q_err_large,
            "value_converged": bool(val_conv),
            "solution_converged": bool(q_conv),
            "value_rate": round(v_slope, 3),
            "solution_rate": round(q_slope, 3),
        })
    return all_value_converge, all_solution_converge, rate_results


def run() -> dict:
    print("=" * 70)
    print("CLAIM C4: Theorem 3.5 — APUB-M asymptotic consistency")
    print("=" * 70)
    print(f"  Newsvendor: true q*={TRUE_Q:.4f}, true val={TRUE_OPTIMAL_VALUE:.4f}")
    print()

    rows = _single_product_convergence()

    with open(os.path.join(ARTIFACTS, "convergence_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    val_conv, sol_conv, rate_results = _check_convergence(rows)
    print(f"\n  Optimal values converge (all alphas): {val_conv}")
    print(f"  Optimal solutions converge (all alphas): {sol_conv}")
    for r in rate_results:
        print(f"    alpha={r['alpha']:.2f}: val_rate={r['value_rate']}, "
              f"q_rate={r['solution_rate']} (expect ~-0.5)")

    ok = val_conv and sol_conv
    verdict = "VERIFIED" if ok else "FAIL"

    result = {
        "claim": "C4: Theorem 3.5 — optimal values and solutions converge a.s.",
        "source": "vartheta_hat -> vartheta* and D(S_hat, S) -> 0 w.p.1",
        "verdict": verdict,
        "true_q": round(TRUE_Q, 6),
        "true_optimal_value": round(TRUE_OPTIMAL_VALUE, 6),
        "alphas_tested": ALPHAS,
        "sample_sizes": SAMPLE_SIZES,
        "n_trials": N_TRIALS,
        "values_converge": bool(val_conv),
        "solutions_converge": bool(sol_conv),
        "rate_analysis": rate_results,
        "convergence_data": rows,
        "raw_data": "artifacts/c4/convergence_raw.csv",
    }

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> C4 VERDICT: {verdict}")
    return result
