"""Core implementation of the Average Percentile Upper Bound (APUB).

This module implements the definitions from:
  "Managing Distributional Ambiguity in Stochastic Optimization through a
   Statistical Upper Bound Framework" (arXiv 2403.08966)

Key definitions implemented:
  - Efron's percentile upper bound  U^Efron_alpha  (Proposition 2.1, Eq. 3)
  - APUB via integral definition    U^APUB_alpha   (Definition 2.2)
  - APUB via CVaR reformulation                    (Proposition 2.3, Eq. 4)
  - Standard large-sample (CLT) upper bound        (Example 2.5)

All functions take a 1-D array of cost values  F(xi_1), ..., F(xi_N)
(the empirical sample) and return the corresponding upper bound for the
population mean  mu = E_P[F(xi)].
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bootstrap_means(data: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """Draw M nonparametric bootstrap resamples (size N with replacement) and
    return their sample means.

    Each resample  zeta_{m,1}, ..., zeta_{m,N}  is drawn i.i.d. from the
    empirical distribution  P_hat_N.  The bootstrap mean is
    mu*_m = (1/N) sum_{n=1}^{N} F(zeta_{m,n}).

    Parameters
    ----------
    data : 1-D array of the N observed cost values F(xi_1),...,F(xi_N).
    M    : number of bootstrap resamples.
    rng  : numpy Generator for reproducibility.
    """
    N = len(data)
    idx = rng.integers(0, N, size=(M, N))
    return data[idx].mean(axis=1)


def efron_upper_bound(means: np.ndarray, alpha: float) -> float:
    """Efron's bootstrap percentile upper bound (Proposition 2.1, Eq. 3).

    U^Efron_alpha[mu | P_hat_N] = inf { t : Pr( mu* <= t | P_hat_N ) >= 1-alpha }

    This is the (1-alpha)-quantile of the bootstrap-mean distribution.
    With M finite bootstrap means this is np.quantile(means, 1-alpha).
    """
    return float(np.quantile(means, 1.0 - alpha))


def apub_integral(means: np.ndarray, alpha: float, n_grid: int = 2000) -> float:
    """APUB via the integral definition (Definition 2.2).

    U^APUB_alpha = (1/alpha) * integral_0^alpha  U^Efron_tau  d_tau

    We evaluate U^Efron_tau (the (1-tau)-quantile of bootstrap means) on a
    fine grid of tau values in (0, alpha] and average (trapezoidal rule).
    As n_grid -> infinity this converges to the exact integral.
    """
    taus = np.linspace(0.0, alpha, n_grid + 1)[1:]
    quantile_points = 1.0 - taus
    bounds = np.quantile(means, quantile_points)
    return float(np.trapezoid(bounds, taus) / alpha)


def apub_cvar(means: np.ndarray, alpha: float) -> float:
    """APUB via the CVaR reformulation (Proposition 2.3, Eq. 4).

    U^APUB_alpha = min_t { t + (1/alpha) * E_{P_hat_N} [ mu* - t ]_+ }

    By the Rockafellar-Uryasev theorem (Theorem A.2) this equals the
    Conditional Value-at-Risk at level alpha of the bootstrap-mean
    distribution.  The minimiser t* is the (1-alpha)-quantile.
    """
    M = len(means)
    sorted_means = np.sort(means)
    k = int(np.floor((1.0 - alpha) * M))
    k = max(0, min(k, M - 1))
    t_star = sorted_means[k]
    excess = np.maximum(means - t_star, 0.0)
    return float(t_star + excess.sum() / (alpha * M))


def apub(data: np.ndarray, alpha: float, M: int = 2000,
         rng: np.random.Generator | None = None) -> dict:
    """Compute APUB for the population mean of *data*.

    Returns a dict with:
      'apub'         : APUB value (CVaR form, Proposition 2.3)
      'apub_integral': APUB value (integral form, Definition 2.2)
      'efron'        : Efron's bound at level alpha
      'sample_mean'  : sample mean mu_hat_N
      'bootstrap_means': the M bootstrap means (for diagnostics)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    means = bootstrap_means(data, M, rng)
    return {
        "apub": apub_cvar(means, alpha),
        "apub_integral": apub_integral(means, alpha),
        "efron": efron_upper_bound(means, alpha),
        "sample_mean": float(data.mean()),
        "bootstrap_means": means,
    }


def clt_upper_bound(data: np.ndarray, alpha: float) -> float:
    """Standard large-sample (CLT / normal-approximation) upper bound.

    U^CLT_alpha = mu_hat_N + z_alpha * S_N / sqrt(N)

    where z_alpha is the (1-alpha)-quantile of the standard normal and
    S_N is the sample standard deviation (Example 2.5).
    """
    N = len(data)
    z = norm.ppf(1.0 - alpha)
    return float(data.mean() + z * data.std(ddof=1) / np.sqrt(N))


def coverage_probability(sampler, bound_fn, true_mean: float,
                         alpha: float, N: int, n_trials: int,
                         seed: int = 0, **bound_kw) -> float:
    """Monte-Carlo estimate of the coverage probability

        Pr( mu <= U_alpha[mu | P_hat_N] | P )

    For each of *n_trials* independent replications:
      1. Draw a fresh sample of size N from *sampler*.
      2. Compute the upper bound via *bound_fn*.
      3. Record whether  true_mean <= bound.

    Returns the fraction of trials where the bound covers the true mean.
    """
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_trials):
        data = sampler(N, rng)
        b = bound_fn(data, alpha, **bound_kw)
        if true_mean <= b:
            count += 1
    return count / n_trials


def coverage_probability_vec(sampler, bound_fn, true_mean: float,
                             alpha: float, N: int, n_trials: int,
                             seed: int = 0, M_boot: int = 2000,
                             n_jobs: int = 1) -> float:
    """Vectorised coverage-probability estimator using bootstrap means.

    *bound_fn* must accept (bootstrap_means, alpha) and return a float.
    """
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_trials):
        data = sampler(N, rng)
        brng = np.random.default_rng(rng.integers(0, 2**31))
        means = bootstrap_means(data, M_boot, brng)
        b = bound_fn(means, alpha)
        if true_mean <= b:
            count += 1
    return count / n_trials
