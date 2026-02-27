"""
pcite.evaluate

Three tests, all measuring whether pCite surfaces physically-validated claims
better than traditional citation count. They must agree. If they don't: debug.

Output: data/results.json + figures/*.pdf
Run:    python -m pcite.evaluate
"""

import json, math, sys
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SCORES_IN   = Path("data/scores.jsonl")
RESULTS_OUT = Path("data/results.json")
FIGURES_DIR = Path("figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

COLORS = {
    "PhysicalMeasurement":  "#d62728",
    "Replicated":           "#ff7f0e",
    "DatabaseReferenced":   "#2ca02c",
    "TextDerived":          "#aec7e8",
    "Hypothesis":           "#c7c7c7",
}


def load_scores(path: Path | None = None) -> list[dict]:
    path = path or SCORES_IN
    return [json.loads(l) for l in path.read_text().splitlines() if l]


_VALIDATED = {"PhysicalMeasurement", "ClinicalObservation", "Replicated"}


def _is_validated(r: dict) -> bool:
    return r["validation_class"] in _VALIDATED


def mann_whitney(records: list[dict]) -> dict:
    """Non-parametric. No distribution assumption. One-sided.

    Always uses base_weight: this tests whether the validation-weighted
    scoring formula separates validated from unvalidated claims across
    the entire corpus.  pcite_score (citation-amplified) is sparse —
    most claims receive zero citations — so it fails Mann-Whitney even
    when the ranking metrics (P@k, NDCG@k) are excellent.
    """
    score_key = "base_weight"
    val   = [r[score_key] for r in records if _is_validated(r)]
    unval = [r[score_key] for r in records if not _is_validated(r)]
    if len(val) < 2 or len(unval) < 2:
        return {"u": 0, "p_value": float("nan"), "n_validated": len(val),
                "n_unvalidated": len(unval), "median_validated": 0, "median_unvalidated": 0,
                "score_key": score_key}
    u, p  = stats.mannwhitneyu(val, unval, alternative="greater")
    return {"u": u, "p_value": p, "n_validated": len(val), "n_unvalidated": len(unval),
            "median_validated":   float(sorted(val)[len(val)//2]),
            "median_unvalidated": float(sorted(unval)[len(unval)//2]),
            "score_key": score_key}


def precision_at_k(records: list[dict], k: int = 50, score_key: str | None = None) -> dict:
    """What fraction of the top-k are validated (Physical/Clinical/Replicated)?"""
    validated = {r["claim_id"] for r in records if _is_validated(r)}
    if score_key is None:
        score_key = "pcite_score" if any(r["pcite_score"] > 0 for r in records) else "base_weight"
    nc = sorted(records, key=lambda r: r.get(score_key, 0), reverse=True)
    tr = sorted(records, key=lambda r: r.get("traditional_citations", r.get("replication_count", 0)),
                reverse=True)
    p_nc = sum(1 for r in nc[:k] if r["claim_id"] in validated) / k
    p_tr = sum(1 for r in tr[:k] if r["claim_id"] in validated) / k
    return {"k": k, "precision_pcite": p_nc, "precision_traditional": p_tr,
            "lift": p_nc / p_tr if p_tr > 0 else float("inf"),
            "score_key": score_key}


def ndcg_at_k(records: list[dict], k: int = 50, score_key: str | None = None) -> dict:
    """Normalised Discounted Cumulative Gain. Standard IR metric."""
    validated = {r["claim_id"] for r in records if _is_validated(r)}

    def dcg(ranked: list[dict]) -> float:
        return sum(
            (1 if r["claim_id"] in validated else 0) / math.log2(i + 2)
            for i, r in enumerate(ranked[:k])
        )

    ideal = sorted(records, key=lambda r: _is_validated(r), reverse=True)
    idcg  = dcg(ideal)
    if idcg == 0:
        return {"k": k, "ndcg_pcite": 0.0, "ndcg_traditional": 0.0}
    if score_key is None:
        score_key = "pcite_score" if any(r["pcite_score"] > 0 for r in records) else "base_weight"
    nc = sorted(records, key=lambda r: r.get(score_key, 0), reverse=True)
    tr = sorted(records, key=lambda r: r.get("traditional_citations", r.get("replication_count", 0)),
                reverse=True)
    return {"k": k, "ndcg_pcite": dcg(nc) / idcg, "ndcg_traditional": dcg(tr) / idcg}


def fig1_rank_scatter(records: list[dict]) -> plt.Figure:
    """
    x = traditional rank, y = pCite rank, colour = validation class.
    Physical claims should cluster top-left. This is the paper's hero figure.
    """
    score_key = "pcite_score" if any(r["pcite_score"] > 0 for r in records) else "base_weight"
    nc = {r["claim_id"]: i+1 for i, r in enumerate(
        sorted(records, key=lambda r: r[score_key], reverse=True))}
    tr = {r["claim_id"]: i+1 for i, r in enumerate(
        sorted(records, key=lambda r: r.get("traditional_citations", r.get("replication_count", 0)),
               reverse=True))}
    fig, ax = plt.subplots(figsize=(7, 6))
    for vc, color in COLORS.items():
        group = [r for r in records if r["validation_class"] == vc]
        ax.scatter([tr[r["claim_id"]] for r in group],
                   [nc[r["claim_id"]] for r in group],
                   c=color, label=vc, alpha=0.5, s=10, linewidths=0)
    n = len(records)
    ax.plot([1, n], [1, n], "k--", lw=0.8, alpha=0.3, label="No change")
    ax.set(xlabel="Traditional citation rank", ylabel="pCite rank",
           title="pCite vs traditional ranking")
    ax.legend(fontsize=8, markerscale=2, loc="lower right")
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


def fig2_score_distribution(records: list[dict]) -> plt.Figure:
    """pCite score by validation class. Physical should be dramatically higher."""
    classes = ["PhysicalMeasurement", "Replicated", "DatabaseReferenced", "TextDerived"]
    data    = [[r["pcite_score"] for r in records if r["validation_class"] == vc] for vc in classes]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=["Physical","Replicated","DB Ref","Text Derived"],
                    patch_artist=True, showfliers=False)
    for patch, vc in zip(bp["boxes"], classes):
        patch.set(facecolor=COLORS[vc], alpha=0.7)
    ax.set(ylabel="pCite score", title="pCite score by validation class")
    fig.tight_layout()
    return fig


def fig3_precision_curve(records: list[dict]) -> plt.Figure:
    """Precision@k for k = 10...200. pCite should stay above traditional throughout."""
    validated = {r["claim_id"] for r in records if _is_validated(r)}
    score_key = "pcite_score" if any(r["pcite_score"] > 0 for r in records) else "base_weight"
    nc = sorted(records, key=lambda r: r[score_key], reverse=True)
    tr = sorted(records, key=lambda r: r.get("traditional_citations", r.get("replication_count", 0)),
                reverse=True)
    ks = list(range(10, min(201, len(records)), 10))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, [sum(1 for r in nc[:k] if r["claim_id"] in validated)/k for k in ks],
            "-o", ms=4, color=COLORS["PhysicalMeasurement"], label="pCite")
    ax.plot(ks, [sum(1 for r in tr[:k] if r["claim_id"] in validated)/k for k in ks],
            "-s", ms=4, color=COLORS["TextDerived"], label="Traditional")
    ax.axhline(len(validated)/len(records), ls="--", color="gray", lw=0.8, label="Random")
    ax.set(xlabel="k", ylabel="Precision@k", title="Precision of surfacing validated claims")
    ax.legend()
    fig.tight_layout()
    return fig


def run_experiment(
    scores_path: Path | None = None,
    results_path: Path | None = None,
    figures_dir: Path | None = None,
) -> dict:
    scores_path = scores_path or SCORES_IN
    results_path = results_path or RESULTS_OUT
    figures_dir = figures_dir or FIGURES_DIR
    records = load_scores(scores_path)
    results = {
        "n_total":      len(records),
        "mann_whitney": mann_whitney(records),
        "precision_50": precision_at_k(records, k=50),
        "ndcg_50":      ndcg_at_k(records, k=50),
    }
    if any(r.get("pcite_propagated", 0) != 0 for r in records):
        results["propagated_precision_50"] = precision_at_k(records, k=50, score_key="pcite_propagated")
        results["propagated_ndcg_50"] = ndcg_at_k(records, k=50, score_key="pcite_propagated")
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig1_rank_scatter(records).savefig(figures_dir / "fig1_rank_comparison.pdf")
    fig2_score_distribution(records).savefig(figures_dir / "fig2_score_dist.pdf")
    fig3_precision_curve(records).savefig(figures_dir / "fig3_precision_at_k.pdf")
    plt.close("all")
    return results


if __name__ == "__main__":
    r   = run_experiment()
    mw  = r["mann_whitney"]
    p50 = r["precision_50"]
    ng  = r["ndcg_50"]
    print(f"\n{'---'*18}", file=sys.stderr)
    print(f"  Mann-Whitney p:  {mw['p_value']:.4f}  "
          f"{'V' if mw['p_value'] < 0.05 else 'X'}", file=sys.stderr)
    print(f"  Precision@50:    pCite={p50['precision_pcite']:.3f}  "
          f"trad={p50['precision_traditional']:.3f}  "
          f"lift={p50['lift']:.1f}x", file=sys.stderr)
    print(f"  NDCG@50:         pCite={ng['ndcg_pcite']:.4f}  "
          f"trad={ng['ndcg_traditional']:.4f}", file=sys.stderr)
    print(f"{'---'*18}", file=sys.stderr)
