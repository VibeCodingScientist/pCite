"""
fig_coverage_threshold.py

Generate Figure 4: Precision@50 as a function of Physical-tier coverage.

Two series from MetaboLights boundary investigation (mean of 5 seeds)
plus one annotation point from the PRIDE v1 disease-query corpus.

Output: figures/fig_coverage_threshold.pdf + .png (300 DPI)
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style (matches fig_boundary.pdf / fig_sensitivity.pdf) ───────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

BOUNDARY_PATH = Path("data/boundary_results.json")
FIGURE_DIR = Path("figures")


def make_figure() -> plt.Figure:
    data = json.loads(BOUNDARY_PATH.read_text())

    # Separate MetaboLights subsamples (exclude PRIDE_v1 overlay)
    metabo = [r for r in data if r.get("source") != "PRIDE_v1"]
    metabo.sort(key=lambda r: r["target_coverage"])

    # Filter to the 6 coverage levels in the spec (skip 40%, 50%, 62.4%)
    target_set = {0.05, 0.10, 0.15, 0.20, 0.30, 0.624}
    points = [r for r in metabo if r["target_coverage"] in target_set]

    coverages  = [r["actual_coverage"] * 100 for r in points]
    p50_pcite  = [r["P@50_pcite"] for r in points]
    p50_trad   = [r["P@50_trad"] for r in points]
    std_pcite  = [r.get("P@50_pcite_std", 0) for r in points]
    std_trad   = [r.get("P@50_trad_std", 0) for r in points]

    fig, ax = plt.subplots(figsize=(8, 5))

    # pCite series: solid red, filled circles, ±1 std error bars
    ax.errorbar(coverages, p50_pcite, yerr=std_pcite, fmt="-o",
                color="#d62728", ms=7, lw=2, capsize=3, capthick=1,
                label="pCite (validation-weighted)", zorder=3)

    # Traditional series: dashed grey, filled squares, ±1 std error bars
    ax.errorbar(coverages, p50_trad, yerr=std_trad, fmt="--s",
                color="#7f7f7f", ms=7, lw=2, capsize=3, capthick=1,
                label="Traditional (citation count)", zorder=2)

    # Crossover threshold — interpolate between points where pCite overtakes trad
    for i in range(1, len(coverages)):
        if p50_pcite[i] > p50_trad[i] and p50_pcite[i - 1] <= p50_trad[i - 1]:
            x0, x1 = coverages[i - 1], coverages[i]
            diff0 = p50_pcite[i - 1] - p50_trad[i - 1]
            diff1 = p50_pcite[i] - p50_trad[i]
            crossover_x = x0 + (x1 - x0) * (-diff0) / (diff1 - diff0 + 1e-9)
            ax.axvline(crossover_x, ls=":", color="#888888", lw=1.5, alpha=0.7)
            ax.annotate(f"Crossover\nthreshold\n~{crossover_x:.0f}%",
                        xy=(crossover_x, 0.35),
                        xytext=(crossover_x + 4, 0.20),
                        fontsize=9, color="#888888",
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
            break

    # PRIDE v1 disease-query annotation point
    pride_x = 5.7   # Physical coverage %
    pride_y = 0.160  # P@50 pCite
    ax.scatter([pride_x], [pride_y], marker="^", c="none", edgecolors="#d62728",
               s=100, linewidths=1.5, zorder=5,
               label="PRIDE disease-query corpus")
    ax.annotate("PRIDE v1\n(disease-query)",
                xy=(pride_x, pride_y),
                xytext=(pride_x + 5, pride_y - 0.06),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

    ax.set_xlabel("Physical-tier coverage (%)")
    ax.set_ylabel("Precision@50")
    ax.set_title("Precision@50 as a function of Physical-tier coverage")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: pCite first, Traditional second, PRIDE third
    order = [labels.index("pCite (validation-weighted)"),
             labels.index("Traditional (citation count)"),
             labels.index("PRIDE disease-query corpus")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    pdf_path = FIGURE_DIR / "fig_coverage_threshold.pdf"
    png_path = FIGURE_DIR / "fig_coverage_threshold.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  {pdf_path}")
    print(f"  {png_path}")
