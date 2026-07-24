"""C6 verifier: Section 5 — application validation.

Validates the APUB framework on two benchmark problems from Section 5:
  (A) Two-stage product mix problem (Section 5.1, Dantzig benchmark)
  (B) Multi-product newsvendor problem (Section 5.3, Hanasusanto et al. 2015)

For each problem we compare APUB-M against SAA-M via Monte Carlo:
  - Out-of-sample performance (mean true cost of the optimal solution)
  - Coverage probability (fraction of trials where the bound covers true cost)
  - Robustness at small sample sizes

The paper's claim (Section 5): APUB balances robustness and practicality
compared to SAA, particularly at small N where distributional ambiguity is
most severe.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", ".openresearch", "artifacts", "c6")
os.makedirs(ARTIFACTS, exist_ok=True)


# ========================================================================== #
#  Part A: Two-Stage Product Mix Problem (Section 5.1)                       #
# ========================================================================== #

C_PROD = np.array([-12.0, -20.0, -18.0, -40.0])
Q_OUT = np.array([6.0, 12.0])


def _sample_gamma_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample gamma from the bimodal mixture in Section 5.1."""
    comp = rng.random(n) < 0.7
    n1 = comp.sum()
    n2 = n - n1
    mean_a = np.array([12.0, 8.0])
    cov_a = np.array([[5.76, 1.92], [1.92, 2.56]])
    mean_b = np.array([2.0, 1.0])
    cov_b = np.array([[0.16, 0.04], [0.04, 0.04]])
    samples = np.empty((n, 2))
    if n1 > 0:
        samples[comp] = rng.multivariate_normal(mean_a, cov_a, size=n1)
    if n2 > 0:
        samples[~comp] = rng.multivariate_normal(mean_b, cov_b, size=n2)
    return np.maximum(samples, 0.01)


def _second_stage_cost(x: np.ndarray, gamma: np.ndarray) -> float:
    """Closed-form second-stage cost for the product mix problem.

    Q(x, gamma) = sum_{w} (q_w / 0.9) * max(0, sum_j T_wj(gamma_w) * x_j - 500*gamma_w)
    """
    g1, g2 = gamma[0], gamma[1]
    T_row1 = np.array([4 - g1/4, 9 - g1/4, 7 - g1/4, 10 - g1/4])
    T_row2 = np.array([3 - g2/4, 1 - g2/4, 3 - g2/4, 6 - g2/4])
    labor_need_1 = T_row1 @ x - 500 * g1
    labor_need_2 = T_row2 @ x - 500 * g2
    cost = (Q_OUT[0] / 0.9) * max(labor_need_1, 0.0) + (Q_OUT[1] / 0.9) * max(labor_need_2, 0.0)
    return cost


def _total_cost(x: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Total cost c'x + Q(x, gamma_i) for each sample i."""
    base = C_PROD @ x
    second = np.array([_second_stage_cost(x, g) for g in gammas])
    return base + second


def _solve_saa_productmix(gammas: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve SAA-M for the product mix problem."""
    def obj(x):
        return float(np.mean(_total_cost(x, gammas)))
    x0 = np.array([80.0, 80.0, 80.0, 80.0])
    bounds = [(0.0, 300.0)] * 4
    res = minimize(obj, x0, method='Powell', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 0.01})
    return res.x, float(res.fun)


def _solve_apub_productmix(gammas: np.ndarray, alpha: float, M: int,
                           rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Solve APUB-M for the product mix problem."""
    N = len(gammas)
    from src.apub_core import apub_cvar
    idx = rng.integers(0, N, size=(M, N))

    def obj(x):
        costs = _total_cost(x, gammas)
        boot_means = costs[idx].mean(axis=1)
        return apub_cvar(boot_means, alpha)

    x0 = np.array([80.0, 80.0, 80.0, 80.0])
    bounds = [(0.0, 300.0)] * 4
    res = minimize(obj, x0, method='Powell', bounds=bounds,
                   options={'maxiter': 1000, 'ftol': 0.01})
    return res.x, float(res.fun)


def _productmix_experiment():
    """Monte Carlo comparison of APUB-M vs SAA-M on the product mix."""
    rng = np.random.default_rng(5606)
    sample_sizes = [30, 60, 120]
    n_trials = 100
    M_boot = 300
    alpha = 0.20
    nominal = 1.0 - alpha

    test_gammas = _sample_gamma_mixture(50000, np.random.default_rng(888))

    rows = []
    for N in sample_sizes:
        apub_costs = []
        saa_costs = []
        apub_covered = 0
        saa_covered = 0
        for trial in range(n_trials):
            train = _sample_gamma_mixture(N, rng)
            brng = np.random.default_rng(rng.integers(0, 2**31))

            x_apub, val_apub = _solve_apub_productmix(train, alpha, M_boot, brng)
            x_saa, val_saa = _solve_saa_productmix(train)

            tc_apub = float(np.mean(_total_cost(x_apub, test_gammas)))
            tc_saa = float(np.mean(_total_cost(x_saa, test_gammas)))
            apub_costs.append(tc_apub)
            saa_costs.append(tc_saa)
            if val_apub >= tc_apub:
                apub_covered += 1
            if val_saa >= tc_saa:
                saa_covered += 1

        rows.append({
            "N": N, "alpha": alpha, "nominal": nominal,
            "mean_oos_apub": round(float(np.mean(apub_costs)), 2),
            "mean_oos_saa": round(float(np.mean(saa_costs)), 2),
            "p10_oos_apub": round(float(np.percentile(apub_costs, 10)), 2),
            "p90_oos_apub": round(float(np.percentile(apub_costs, 90)), 2),
            "p10_oos_saa": round(float(np.percentile(saa_costs, 10)), 2),
            "p90_oos_saa": round(float(np.percentile(saa_costs, 90)), 2),
            "coverage_apub": round(apub_covered / n_trials, 4),
            "coverage_saa": round(saa_covered / n_trials, 4),
            "n_trials": n_trials,
        })
        print(f"  N={N:4d}  APUB oos={np.mean(apub_costs):.1f} "
              f"(cov={apub_covered/n_trials:.3f})  "
              f"SAA oos={np.mean(saa_costs):.1f} "
              f"(cov={saa_covered/n_trials:.3f})")
    return rows


# ========================================================================== #
#  Part B: Multi-Product Newsvendor (Section 5.3)                           #
# ========================================================================== #

P_NEW = -2.0
H_NEW = 9.0
B_NEW = 5.0
CRITICAL_RATIO = (B_NEW - P_NEW) / (H_NEW + B_NEW)

MU1 = np.array([60.89, 48.58, 46.81, 56.54, 61.58, 52.69, 69.42, 60.54, 54.43, 51.76])
MU2 = np.array([50.30, 61.87, 53.16, 41.79, 51.94, 62.14, 45.47, 45.26, 55.95, 55.95])

SIG1 = np.array([
    [9.27,2.84,-0.07,1.19,-0.48,1.40,2.87,4.06,-1.40,-1.96],
    [2.84,5.90,-2.83,0.21,2.27,-2.40,-0.89,4.22,3.43,2.78],
    [-0.07,-2.83,5.48,-0.30,0.90,3.54,-4.51,-2.45,-2.91,-4.95],
    [1.19,0.21,-0.30,7.99,-1.02,-1.27,-0.15,-1.55,-1.69,-0.36],
    [-0.48,2.27,0.90,-1.02,9.48,-0.08,-3.69,2.71,-0.69,-0.34],
    [1.40,-2.40,3.54,-1.27,-0.08,6.94,-1.26,-2.73,0.01,-5.19],
    [2.87,-0.89,-4.51,-0.15,-3.69,-1.26,12.05,-0.16,-0.16,2.44],
    [4.06,4.22,-2.45,-1.55,2.71,-2.73,-0.16,9.16,-0.77,1.94],
    [-1.40,3.43,-2.91,-1.69,-0.69,0.01,-0.16,-0.77,7.41,2.24],
    [-1.96,2.78,-4.95,-0.36,-0.34,-5.19,2.44,1.94,2.24,6.70],
])

SIG2 = np.array([
    [6.32,2.99,-0.06,0.73,-0.33,1.36,1.55,2.51,-1.19,-1.75],
    [2.99,9.57,-4.09,0.19,2.44,-3.60,-0.74,4.02,4.49,3.83],
    [-0.06,-4.09,7.06,-0.25,0.86,4.74,-3.35,-2.08,-3.40,-6.08],
    [0.73,0.19,-0.25,4.37,-0.64,-1.11,-0.07,-0.86,-1.29,-0.29],
    [-0.33,2.44,0.86,-0.64,6.74,-0.08,-2.04,1.71,-0.60,-0.31],
    [1.36,-3.60,4.74,-1.11,-0.08,9.65,-0.98,-2.41,0.01,-6.62],
    [1.55,-0.74,-3.35,-0.07,-2.04,-0.98,5.17,-0.08,-0.10,1.72],
    [2.51,4.02,-2.08,-0.86,1.71,-2.41,-0.08,5.12,-0.59,1.57],
    [-1.19,4.49,-3.40,-1.29,-0.60,0.01,-0.10,-0.59,7.83,2.49],
    [-1.75,3.83,-6.08,-0.29,-0.31,-6.62,1.72,1.57,2.49,7.83],
])

NOISE_LO = np.array([-5.37, 6.74, 3.22, -7.48, -4.89, -0.21, -12.14, -7.74, 0.77, 2.13])
NOISE_HI = np.array([26.27, 14.16, 17.68, 28.38, 25.79, 16.11, 32.99, 28.64, 20.13, 18.77])


def _sample_newsvendor_demand(n: int, rng: np.random.Generator, case: int = 1) -> np.ndarray:
    """Sample 10-product demand. Case I: mixed normal. Case II: + biased noise."""
    comp = rng.random(n) < 0.5
    n1 = comp.sum()
    n2 = n - n1
    demands = np.empty((n, 10))
    if n1 > 0:
        demands[comp] = rng.multivariate_normal(MU1, SIG1, size=n1)
    if n2 > 0:
        demands[~comp] = rng.multivariate_normal(MU2, SIG2, size=n2)
    demands = np.maximum(demands, 0.0)
    if case == 2:
        noise = rng.uniform(NOISE_LO, NOISE_HI, size=(n, 10))
        demands = demands + noise
    return demands


def _newsvendor_total_cost(x: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Per-sample cost F(x, xi) = p'x + h'(x-xi)_+ + b'(xi-x)_+ for each sample."""
    return (P_NEW * x.sum() +
            H_NEW * np.maximum(x[None, :] - demands, 0).sum(axis=1) +
            B_NEW * np.maximum(demands - x[None, :], 0).sum(axis=1))


def _solve_saa_newsvendor_multi(demands: np.ndarray) -> np.ndarray:
    """SAA optimal order quantities = critical-ratio quantile of empirical demand."""
    return np.quantile(demands, CRITICAL_RATIO, axis=0)


def _solve_apub_newsvendor_multi(demands: np.ndarray, alpha: float, M: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """APUB-M for multi-product newsvendor via coordinate descent.

    Starts from SAA solution, optimizes one product at a time.
    """
    from src.apub_core import apub_cvar
    N = len(demands)
    idx = rng.integers(0, N, size=(M, N))
    x = _solve_saa_newsvendor_multi(demands).copy()

    for pass_i in range(3):
        for i in range(10):
            grid = np.linspace(max(x[i] * 0.5, 1), x[i] * 2.0, 15)
            best_val = float('inf')
            best_q = x[i]
            for q in grid:
                x[i] = q
                costs = _newsvendor_total_cost(x, demands)
                boot_means = costs[idx].mean(axis=1)
                val = apub_cvar(boot_means, alpha)
                if val < best_val:
                    best_val = val
                    best_q = q
            x[i] = best_q
    return x


def _newsvendor_experiment():
    """Monte Carlo comparison of APUB-M vs SAA-M on multi-product newsvendor."""
    rng = np.random.default_rng(5607)
    sample_sizes = [30, 60, 120]
    n_trials = 100
    M_boot = 300
    alpha = 0.20

    test_demands = _sample_newsvendor_demand(50000, np.random.default_rng(777), case=1)

    rows = []
    for N in sample_sizes:
        apub_costs = []
        saa_costs = []
        for trial in range(n_trials):
            train = _sample_newsvendor_demand(N, rng, case=1)
            brng = np.random.default_rng(rng.integers(0, 2**31))

            x_apub = _solve_apub_newsvendor_multi(train, alpha, M_boot, brng)
            x_saa = _solve_saa_newsvendor_multi(train)

            tc_apub = float(np.mean(_newsvendor_total_cost(x_apub, test_demands)))
            tc_saa = float(np.mean(_newsvendor_total_cost(x_saa, test_demands)))
            apub_costs.append(tc_apub)
            saa_costs.append(tc_saa)

        mean_apub = float(np.mean(apub_costs))
        mean_saa = float(np.mean(saa_costs))
        rows.append({
            "N": N, "alpha": alpha,
            "mean_oos_apub": round(mean_apub, 2),
            "mean_oos_saa": round(mean_saa, 2),
            "p10_oos_apub": round(float(np.percentile(apub_costs, 10)), 2),
            "p90_oos_apub": round(float(np.percentile(apub_costs, 90)), 2),
            "p10_oos_saa": round(float(np.percentile(saa_costs, 10)), 2),
            "p90_oos_saa": round(float(np.percentile(saa_costs, 90)), 2),
            "apub_better": bool(mean_apub < mean_saa),
            "n_trials": n_trials,
        })
        print(f"  N={N:4d}  APUB oos={mean_apub:.1f} "
              f"[p10={np.percentile(apub_costs,10):.0f},p90={np.percentile(apub_costs,90):.0f}]  "
              f"SAA oos={mean_saa:.1f} "
              f"[p10={np.percentile(saa_costs,10):.0f},p90={np.percentile(saa_costs,90):.0f}]  "
              f"APUB better={mean_apub < mean_saa}")
    return rows


def run() -> dict:
    print("=" * 70)
    print("CLAIM C6: Section 5 — application validation")
    print("=" * 70)

    print("\n[A] Two-stage product mix (Section 5.1)")
    print("    Dantzig benchmark, bimodal normal labor uncertainty")
    pm_rows = _productmix_experiment()

    print("\n[B] Multi-product newsvendor (Section 5.3)")
    print("    10 products, mixed normal demand (Case I)")
    nv_rows = _newsvendor_experiment()

    with open(os.path.join(ARTIFACTS, "productmix_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pm_rows[0].keys()))
        w.writeheader()
        w.writerows(pm_rows)
    with open(os.path.join(ARTIFACTS, "newsvendor_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(nv_rows[0].keys()))
        w.writeheader()
        w.writerows(nv_rows)

    pm_apub_better = any(r["mean_oos_apub"] < r["mean_oos_saa"] for r in pm_rows)
    nv_apub_better = any(r["apub_better"] for r in nv_rows)
    pm_apub_cov = all(r["coverage_apub"] > r["coverage_saa"] for r in pm_rows if r["N"] <= 60)

    ok = pm_apub_better and nv_apub_better
    verdict = "VERIFIED" if ok else "FAIL"

    result = {
        "claim": "C6: Section 5 — APUB validated on product mix + newsvendor",
        "source": "APUB balances robustness and practicality vs SAA",
        "verdict": verdict,
        "productmix": {
            "problem": "two-stage product mix (Dantzig)",
            "apub_better_at_some_N": bool(pm_apub_better),
            "apub_higher_coverage_small_N": bool(pm_apub_cov),
            "data": pm_rows,
        },
        "newsvendor": {
            "problem": "multi-product newsvendor (Hanasusanto 2015)",
            "apub_better_at_some_N": bool(nv_apub_better),
            "data": nv_rows,
        },
        "raw_data": "artifacts/c6/productmix_raw.csv, artifacts/c6/newsvendor_raw.csv",
    }

    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n>>> C6 VERDICT: {verdict}")
    return result
