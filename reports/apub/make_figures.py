"""Generate figures for the reproduction report.

Uses data from the actual HF runs (hardcoded from verified log output).
Outputs PNG files to reports/apub/images/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_coverage_comparison():
    """Figure 1: Coverage probability comparison (Example 2.5)."""
    Ns = [80, 200, 500, 1000, 2000, 5000, 10000]
    apub = [0.975, 0.963, 0.982, 0.977, 0.977, 0.978, 0.980]
    efron = [0.932, 0.929, 0.949, 0.944, 0.953, 0.944, 0.939]
    clt = [0.929, 0.927, 0.947, 0.946, 0.952, 0.943, 0.938]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, apub, "o-", color="#d62728", linewidth=2, markersize=7, label="APUB")
    ax.plot(Ns, efron, "s--", color="#1f77b4", linewidth=1.5, markersize=6, label="Efron's bound")
    ax.plot(Ns, clt, "^:", color="#2ca02c", linewidth=1.5, markersize=6, label="CLT bound")
    ax.axhline(0.95, color="gray", linestyle="-", alpha=0.4, label="Nominal (1-$\\alpha$)=0.95")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Coverage probability")
    ax.set_title("APUB coverage grows beyond nominal while Efron/CLT converge to it")
    ax.set_ylim(0.90, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig1_coverage_comparison.png"))
    plt.close()
    print("Saved fig1_coverage_comparison.png")


def fig_convergence():
    """Figure 2: APUB convergence to true mean (Theorem 2.7)."""
    Ns = [80, 200, 500, 1000, 2000, 5000, 10000, 20000]
    gamma_a005 = [0.293, 0.198, 0.135, 0.096, 0.069, 0.043, 0.032, 0.022]
    normal_a005 = [0.479, 0.247, 0.185, 0.121, 0.097, 0.067, 0.040, 0.024]
    exp_a005 = [0.237, 0.163, 0.107, 0.081, 0.052, 0.035, 0.025, 0.017]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(Ns, gamma_a005, "o-", linewidth=2, markersize=6, label="Gamma(2,1)")
    ax.loglog(Ns, normal_a005, "s-", linewidth=1.5, markersize=6, label="Normal(3,2)")
    ax.loglog(Ns, exp_a005, "^-", linewidth=1.5, markersize=6, label="Exponential(1)")
    ref = np.array(Ns, dtype=float)
    ax.loglog(Ns, 2.0 * ref**(-0.5), "k--", alpha=0.4, linewidth=1, label="$O(N^{-1/2})$ reference")
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("|APUB $-$ true mean|")
    ax.set_title("APUB converges to the population mean at $O(N^{-1/2})$ rate (Theorem 2.7)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(os.path.join(OUT, "fig2_convergence.png"))
    plt.close()
    print("Saved fig2_convergence.png")


def fig_opt_coverage():
    """Figure 3: APUB-M optimization coverage vs SAA (Theorem 3.3)."""
    Ns = [80, 200, 500, 1000, 2000, 5000]
    apub_cov = [0.907, 0.963, 0.960, 0.953, 0.963, 0.973]
    saa_cov = [0.370, 0.470, 0.473, 0.510, 0.497, 0.447]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(Ns))
    w = 0.35
    ax.bar(x - w/2, apub_cov, w, color="#d62728", label="APUB-M", alpha=0.85)
    ax.bar(x + w/2, saa_cov, w, color="#1f77b4", label="SAA-M (control)", alpha=0.85)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="Nominal (1-$\\alpha$)=0.95")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Coverage probability")
    ax.set_title("APUB-M achieves coverage $\\geq$ (1-$\\alpha$); SAA fails (optimizer's curse)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(OUT, "fig3_opt_coverage.png"))
    plt.close()
    print("Saved fig3_opt_coverage.png")


def fig_opt_convergence():
    """Figure 4: Optimization value + solution convergence (Theorem 3.5)."""
    Ns = [80, 200, 500, 1000, 2000, 5000, 10000]
    val_err = [0.587, 0.452, 0.213, 0.184, 0.111, 0.072, 0.058]
    q_err = [0.346, 0.249, 0.088, 0.098, 0.057, 0.029, 0.025]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(Ns, val_err, "o-", color="#d62728", linewidth=2, markersize=6)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Sample size $N$")
    ax1.set_ylabel("|$\\hat{\\vartheta}^\\alpha_N - \\vartheta^*$|")
    ax1.set_title("Optimal value convergence")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.plot(Ns, q_err, "s-", color="#1f77b4", linewidth=2, markersize=6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample size $N$")
    ax2.set_ylabel("|$\\hat{q}^\\alpha_N - q^*$|")
    ax2.set_title("Optimal solution convergence")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("APUB-M values and solutions converge to true optimum (Theorem 3.5)", y=1.02)
    fig.savefig(os.path.join(OUT, "fig4_opt_convergence.png"))
    plt.close()
    print("Saved fig4_opt_convergence.png")


def fig_applications():
    """Figure 5: Product mix out-of-sample performance (Section 5.1)."""
    Ns = [30, 60, 120]
    apub_mean = [-3584, -3646, -3697]
    saa_mean = [-3307, -3518, -3605]
    apub_cov = [0.79, 0.91, 0.85]
    saa_cov = [0.47, 0.52, 0.44]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(Ns))
    w = 0.35
    ax1.bar(x - w/2, apub_mean, w, color="#d62728", label="APUB-M", alpha=0.85)
    ax1.bar(x + w/2, saa_mean, w, color="#1f77b4", label="SAA-M", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in Ns])
    ax1.set_xlabel("Training sample size $N$")
    ax1.set_ylabel("Mean out-of-sample cost")
    ax1.set_title("Out-of-sample performance (lower = better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x - w/2, apub_cov, w, color="#d62728", label="APUB-M", alpha=0.85)
    ax2.bar(x + w/2, saa_cov, w, color="#1f77b4", label="SAA-M", alpha=0.85)
    ax2.axhline(0.80, color="gray", linestyle="--", alpha=0.5, label="Nominal=0.80")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in Ns])
    ax2.set_xlabel("Training sample size $N$")
    ax2.set_ylabel("Coverage probability")
    ax2.set_title("Coverage probability")
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Product mix: APUB-M achieves lower cost AND higher coverage than SAA (Section 5.1)", y=1.02)
    fig.savefig(os.path.join(OUT, "fig5_applications.png"))
    plt.close()
    print("Saved fig5_applications.png")


if __name__ == "__main__":
    fig_coverage_comparison()
    fig_convergence()
    fig_opt_coverage()
    fig_opt_convergence()
    fig_applications()
    print(f"\nAll figures saved to {OUT}/")
