"""APUB Interactive Demo — Average Percentile Upper Bound

This notebook demonstrates the central claim of arXiv 2403.08966:
APUB is a statistical upper bound whose coverage grows beyond the nominal
level, unlike Efron's percentile bound and the CLT bound.

Run:  marimo edit notebooks/apub_demo.py  or  marimo run notebooks/apub_demo.py
"""

import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.stats import norm
    return mo, np, norm


@app.cell
def _(mo):
    mo.md(
        "# Average Percentile Upper Bound (APUB)\n"
        "Reproduction of arXiv 2403.08966 — all 6 claims VERIFIED.\n\n"
        "APUB averages Efron's bootstrap percentile bound over the tail, "
        "yielding a CVaR-like upper confidence bound whose coverage exceeds "
        "the nominal level."
    )
    return


@app.cell
def _(np):
    def bootstrap_means(data, M, rng):
        N = len(data)
        idx = rng.integers(0, N, size=(M, N))
        return data[idx].mean(axis=1)

    def apub_cvar(means, alpha):
        M = len(means)
        sx = np.sort(means)
        k = max(0, min(int(np.floor((1-alpha)*M)), M-1))
        t = sx[k]
        return float(t + np.maximum(means - t, 0).sum() / (alpha * M))

    def efron_bound(means, alpha):
        return float(np.quantile(means, 1 - alpha))

    def clt_bound(data, alpha):
        N = len(data)
        z = norm.ppf(1 - alpha)
        return float(data.mean() + z * data.std(ddof=1) / np.sqrt(N))

    return apub_cvar, bootstrap_means, clt_bound, efron_bound


@app.cell
def _(apub_cvar, bootstrap_means, clt_bound, efron_bound, mo, np, norm):
    TRUE_MEAN = 2.0
    ALPHA = 0.05
    N = mo.slider(10, 2000, value=200, label="Sample size N")
    M_BOOT = 1000

    rng = np.random.default_rng(42)
    data = rng.gamma(2, 1, size=N.value)
    brng = np.random.default_rng(99)
    bmeans = bootstrap_means(data, M_BOOT, brng)

    v_apub = apub_cvar(bmeans, ALPHA)
    v_efron = efron_bound(bmeans, ALPHA)
    v_clt = clt_bound(data, ALPHA)

    mo.md(
        f"**N = {N.value}, alpha = {ALPHA}**\n\n"
        f"| Bound | Value | Covers μ=2? |\n"
        f"|-------|-------|------------|\n"
        f"| APUB | {v_apub:.4f} | {'yes' if v_apub >= TRUE_MEAN else 'no'} |\n"
        f"| Efron | {v_efron:.4f} | {'yes' if v_efron >= TRUE_MEAN else 'no'} |\n"
        f"| CLT | {v_clt:.4f} | {'yes' if v_clt >= TRUE_MEAN else 'no'} |\n"
    )
    return


@app.cell
def _(mo):
    mo.md(
        "## Evidence from the full reproduction\n\n"
        "**C5 (Example 2.5):** On Gamma(2,1) with α=0.05, 1000 Monte Carlo trials:\n\n"
        "| N | APUB coverage | Efron coverage | CLT coverage |\n"
        "|-----|--------------|----------------|-------------|\n"
        "| 80 | 0.975 | 0.932 | 0.929 |\n"
        "| 500 | 0.982 | 0.949 | 0.947 |\n"
        "| 1000 | 0.977 | 0.944 | 0.946 |\n"
        "| 10000 | 0.980 | 0.939 | 0.938 |\n\n"
        "APUB coverage stays above 0.97 while Efron/CLT hover around 0.95.\n\n"
        "See [reports/apub/report.md](../reports/apub/report.md) for full details."
    )
    return


if __name__ == "__main__":
    app.run()
