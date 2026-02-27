"""
Sensitivity analysis: Physical/TextDerived weight ratio vs Precision@50.

Addresses the reviewer critique that the 1000:1 weight ratio between
PhysicalMeasurement and TextDerived was chosen to produce the result.
Shows that Precision@50 lift over traditional citation count is robust
across a wide range of ratios (1:1 through 1000:1).

No API calls. No graph rebuild. Runs in seconds.

Output:
  data/sensitivity/sensitivity_table.json
  figures/fig_sensitivity.pdf
"""

import json, math, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Reuse existing evaluate functions
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pcite.evaluate import load_scores, precision_at_k

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RATIOS = [1, 2, 5, 10, 50, 100, 500, 1000]

# Log-positions of intermediate classes between TextDerived (0) and Physical (1).
# Derived from current production weights:
#   ClinicalObservation: log(4.0/0.01) / log(10.0/0.01) = 0.867
#   Replicated:          log(2.0/0.01) / log(10.0/0.01) = 0.767
#   DatabaseReferenced:  log(0.5/0.01) / log(10.0/0.01) = 0.567
LOG_POSITIONS = {
    "PhysicalMeasurement":  1.0,
    "ClinicalObservation":  0.867,
    "Replicated":           0.767,
    "DatabaseReferenced":   0.567,
    "TextDerived":          0.0,
    "Hypothesis":           None,   # always 0
}

PHYSICAL_WEIGHT = 10.0  # fixed anchor

OUTPUT_DIR  = Path("data/sensitivity")
FIGURES_DIR = Path("figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Weight interpolation
# ---------------------------------------------------------------------------
def compute_weights(ratio: float) -> dict[str, float]:
    """Compute validation weights for a given Physical:TextDerived ratio.

    Physical is fixed at 10.0.  TextDerived = 10.0 / ratio.
    Intermediate classes are log-interpolated between these endpoints,
    preserving rank order.  Hypothesis is always 0.
    """
    text_weight = PHYSICAL_WEIGHT / ratio
    weights = {}
    for cls, pos in LOG_POSITIONS.items():
        if pos is None:
            weights[cls] = 0.0
        elif ratio == 1:
            # All classes get the same weight (null hypothesis)
            weights[cls] = PHYSICAL_WEIGHT
        else:
            # Log-interpolate: w = text_weight * (Physical/text_weight)^pos
            weights[cls] = text_weight * (PHYSICAL_WEIGHT / text_weight) ** pos
    return weights


def reweight_records(records: list[dict], weights: dict[str, float]) -> list[dict]:
    """Return a copy of records with base_weight recomputed using new weights."""
    out = []
    for r in records:
        w = weights.get(r["validation_class"], 0.0)
        new_bw = w * math.log2(r["replication_count"] + 1)
        out.append({**r, "base_weight": new_bw})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_sensitivity() -> list[dict]:
    records = load_scores()
    print(f"Loaded {len(records)} records", file=sys.stderr)

    results = []
    for ratio in RATIOS:
        weights = compute_weights(ratio)
        reweighted = reweight_records(records, weights)
        p = precision_at_k(reweighted, k=50, score_key="base_weight")
        row = {
            "ratio":                 ratio,
            "weights":               {k: round(v, 4) for k, v in weights.items()},
            "precision_pcite":       p["precision_pcite"],
            "precision_traditional": p["precision_traditional"],
            "lift":                  p["lift"],
        }
        results.append(row)
        print(f"  ratio={ratio:>5d}  P@50_pcite={p['precision_pcite']:.3f}  "
              f"P@50_trad={p['precision_traditional']:.3f}  "
              f"lift={p['lift']:.2f}x", file=sys.stderr)

    # Write table
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "sensitivity_table.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nTable → {OUTPUT_DIR / 'sensitivity_table.json'}", file=sys.stderr)

    # Generate figure
    FIGURES_DIR.mkdir(exist_ok=True)
    fig = make_figure(results)
    fig.savefig(FIGURES_DIR / "fig_sensitivity.pdf")
    plt.close(fig)
    print(f"Figure → {FIGURES_DIR / 'fig_sensitivity.pdf'}", file=sys.stderr)

    return results


def make_figure(results: list[dict]) -> plt.Figure:
    ratios = [r["ratio"] for r in results]
    p_pcite = [r["precision_pcite"] for r in results]
    p_trad = [r["precision_traditional"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # pCite line
    ax.plot(ratios, p_pcite, "-o", color="#d62728", ms=7, lw=2,
            label="pCite (validation-weighted)", zorder=3)

    # Traditional baseline (flat)
    ax.axhline(p_trad[0], ls="--", color="#888888", lw=1.5,
               label="Traditional (replication count)")

    # Current production ratio
    ax.axvline(1000, ls=":", color="#555555", lw=1, alpha=0.7)
    ax.annotate("current\n(1000:1)", xy=(1000, p_pcite[-1]),
                xytext=(400, p_pcite[-1] - 0.06),
                fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Physical : TextDerived weight ratio")
    ax.set_ylabel("Precision@50")
    ax.set_title("Sensitivity of Precision@50 to weight ratio")
    ax.set_xticks(ratios)
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run_sensitivity()
