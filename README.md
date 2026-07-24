# icml26-repro-eXLcL70GXO — APUB Stochastic Optimization Reproduction

Reproduction of **"Managing Distributional Ambiguity in Stochastic Optimization through a Statistical Upper Bound Framework"** (arXiv [2403.08966](https://arxiv.org/abs/2403.08966), OpenReview [eXLcL70GXO](https://openreview.net/forum?id=eXLcL70GXO)).

## Reproduction Summary

**All 6 claims VERIFIED** with faithful Monte Carlo evidence on Hugging Face cpu-upgrade (CPU only, no GPU).

| Claim | Statement | Verdict | Key result |
|-------|-----------|---------|------------|
| C1 | APUB definition (Def 2.2) | ✅ VERIFIED | Integral ≡ CVaR form (diff < 0.002) |
| C2 | APUB → μ a.s. (Thm 2.7) | ✅ VERIFIED | 5 distributions, rate = −0.51 (expect −0.5) |
| C3 | APUB-M coverage ≥ (1−α) (Thm 3.3) | ✅ VERIFIED | Coverage 0.91–0.97; SAA control 0.37–0.51 |
| C4 | Opt values/solutions converge (Thm 3.5) | ✅ VERIFIED | Value err 0.59→0.06, solution err 0.35→0.03 |
| C5 | APUB coverage > Efron/CLT (Ex 2.5) | ✅ VERIFIED | APUB 0.98 vs Efron/CLT 0.95 |
| C6 | Applications: product mix + newsvendor | ✅ VERIFIED | APUB lower cost + higher coverage vs SAA |

**Full report:** [reports/apub/report.md](reports/apub/report.md)

## Experiment Log

| Branch | Purpose | Run command | Assessment | Compute |
|--------|---------|-------------|------------|---------|
| `main` | Publication surface | Not run as an experiment (publication surface) | — | — |
| [`orx/apub-baseline-core-implementation-c1`](../../tree/orx/apub-baseline-core-implementation-c1) | APUB core + C1 definition | `pip install -q uv && uv run python -m src.verify` | C1 VERIFIED | Local CPU, 6s |
| [`orx/c2-convergence-c5-coverage-v2`](../../tree/orx/c2-convergence-c5-coverage-v2) | C2 convergence + C5 coverage | `pip install -q uv && uv run python -m src.verify` | C2, C5 VERIFIED | HF cpu-upgrade, 35min |
| [`orx/c3-opt-correctness-c4-opt-consistency`](../../tree/orx/c3-opt-correctness-c4-opt-consistency) | C3 correctness + C4 consistency | `pip install -q uv && uv run python -m src.verify` | C3, C4 VERIFIED | HF cpu-upgrade, 25min |
| [`orx/c6-applications-product-mix-multi-product-newsve`](../../tree/orx/c6-applications-product-mix-multi-product-newsve) | C6 product mix + newsvendor | `pip install -q uv && uv run python -m src.verify` | C6 VERIFIED | HF cpu-upgrade, 50min |

## Key findings

- **APUB is correct but not accurate:** Its coverage probability exceeds the nominal level (0.98 vs 0.95), while Efron's and CLT bounds converge to exactly 0.95. This matches the paper's theoretical prediction.
- **APUB-M solves the optimizer's curse:** SAA-M coverage is only ~0.45 (it systematically under-estimates the true cost), while APUB-M achieves coverage ≥ 0.90 at all sample sizes tested.
- **APUB works on real problems:** On the two-stage product mix (Dantzig benchmark), APUB-M achieves lower mean cost (−3,584 vs −3,307) AND higher coverage (0.79 vs 0.47) than SAA-M at N=30.

## Reproducing locally

```bash
git clone https://github.com/MachineLearning-Nerd/icml26-repro-eXLcL70GXO-minimizing-upper-confidence-bounds-a-data-driven-framework-for-stochastic-pr.git
cd icml26-repro-eXLcL70GXO-minimizing-upper-confidence-bounds-a-data-driven-framework-for-stochastic-pr
git checkout orx/c6-applications-product-mix-multi-product-newsve
pip install uv && uv run python -m src.verify
```

## Interactive notebook

```bash
marimo edit notebooks/apub_demo.py
```

## Environment

- Python ≥ 3.12, numpy ≥ 2.0, scipy ≥ 1.14, matplotlib ≥ 3.9
- Managed by [uv](https://github.com/astral-sh/uv); `pyproject.toml` + `uv.lock` committed
- All runs use deterministic seeds (see each claim verifier)
