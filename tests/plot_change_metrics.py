"""
Visualize the D (frame difference) and S (1 - SSIM) distributions measured by
measure_change_metrics.py. Produces two figures in results/:

  change_dist.png     - distributions of D and S (hist + log-hist + box on shared axis)
  change_corr.png     - D-vs-S relationship (scatter, rank scatter, joint)

Usage: python tests/plot_change_metrics.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = RESULTS / "change_metrics.npz"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Run measure_change_metrics.py first; {DATA} not found.")
    npz = np.load(DATA)
    d, s = npz["d"], npz["s"]
    used, n_videos = int(npz["used"]), int(npz["n_videos"])
    subtitle = f"{used} videos, {len(d)} frame-pairs"

    # ---- Figure 1: distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Change-metric distributions  ({subtitle})", fontsize=13)

    # (a) overlaid histograms on the same [0,1] axis -> shows the scale mismatch
    bins = np.linspace(0, 1, 60)
    axes[0].hist(d, bins=bins, alpha=0.6, label="D (frame diff)", color="#1f77b4", density=True)
    axes[0].hist(s, bins=bins, alpha=0.6, label="S (1 - SSIM)", color="#d62728", density=True)
    axes[0].set_title("(a) Distributions on shared [0,1] axis")
    axes[0].set_xlabel("normalized value")
    axes[0].set_ylabel("density")
    axes[0].legend()

    # (b) box plot side by side -> spread comparison
    axes[1].boxplot([d, s], tick_labels=["D", "S"], showfliers=False, widths=0.5)
    axes[1].scatter(np.full(len(d), 1) + np.random.default_rng(0).normal(0, 0.03, len(d)),
                    d, s=3, alpha=0.08, color="#1f77b4")
    axes[1].scatter(np.full(len(s), 2) + np.random.default_rng(1).normal(0, 0.03, len(s)),
                    s, s=3, alpha=0.08, color="#d62728")
    axes[1].set_title(f"(b) Spread:  std(S)={s.std():.3f} ≈ {s.std()/d.std():.1f}× std(D)={d.std():.3f}")
    axes[1].set_ylabel("normalized value")

    # (c) variance contribution to C = 0.5D + 0.5S
    var_d = 0.25 * d.var()
    var_s = 0.25 * s.var()
    shares = [var_d / (var_d + var_s) * 100, var_s / (var_d + var_s) * 100]
    bars = axes[2].bar(["0.5·D", "0.5·S"], shares, color=["#1f77b4", "#d62728"])
    axes[2].set_title("(c) Variance share of C = 0.5D + 0.5S")
    axes[2].set_ylabel("% of Var(C)")
    axes[2].set_ylim(0, 100)
    for b, v in zip(bars, shares):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = RESULTS / "change_dist.png"
    fig.savefig(f1, dpi=130)
    print(f"saved {f1}")

    # ---- Figure 2: correlation ----
    pearson = float(np.corrcoef(d, s)[0, 1])
    dr = np.argsort(np.argsort(d))
    sr = np.argsort(np.argsort(s))
    spearman = float(np.corrcoef(dr, sr)[0, 1])

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f"D vs S relationship  ({subtitle})", fontsize=13)

    ax2[0].scatter(d, s, s=5, alpha=0.15, color="#6a3d9a")
    ax2[0].set_title(f"(a) Raw values  —  Pearson r = {pearson:.3f}")
    ax2[0].set_xlabel("D (frame difference)")
    ax2[0].set_ylabel("S (1 - SSIM)")

    ax2[1].scatter(dr / len(d), sr / len(s), s=5, alpha=0.12, color="#33a02c")
    ax2[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax2[1].set_title(f"(b) Rank-rank  —  Spearman ρ = {spearman:.3f}")
    ax2[1].set_xlabel("rank(D) percentile")
    ax2[1].set_ylabel("rank(S) percentile")

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    f2 = RESULTS / "change_corr.png"
    fig2.savefig(f2, dpi=130)
    print(f"saved {f2}")


if __name__ == "__main__":
    main()
