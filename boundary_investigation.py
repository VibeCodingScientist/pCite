"""
Boundary investigation: Physical coverage threshold vs pCite advantage.

Subsamples existing MetaboLights data/scores.jsonl at decreasing Physical
coverage levels. At each level, computes P@k and NDCG@k for pCite vs
traditional vs embedding. Plots the crossover point where pCite starts winning.

No API calls. Runs locally in seconds (plus ~60s first time for embeddings).

Output:
  data/boundary_results.json
  figures/fig_boundary.pdf
"""

import hashlib, json, math, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Reuse existing helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pcite.evaluate import load_scores, _VALIDATED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COVERAGE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.624]
N_SEEDS = 5
K_VALUES = [10, 25, 50, 100, 200]
NDCG_K = 50

SCORES_PATH = Path("data/scores.jsonl")
PRIDE_SCORES_PATH = Path("data/pride/scores.jsonl")
OUTPUT_PATH = Path("data/boundary_results.json")
FIGURE_PATH = Path("figures/fig_boundary.pdf")

# Embedding setup (reuse from embedding_baseline.py)
MODEL_NAME = "FremyCompany/BioLORD-2023"
QUERY = (
    "metabolite measured by mass spectrometry instrument "
    "physical experiment validated biomarker"
)
EMBED_CACHE_DIR = Path("data/embedding")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Metric helpers (local — same logic as evaluate.py / embedding_baseline.py)
# ---------------------------------------------------------------------------

def precision_at_k(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(1 for cid in ranked_ids[:k] if cid in validated) / k


def dcg_at_k(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(
        (1.0 if ranked_ids[i] in validated else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(ranked_ids)))
    )


def ndcg_at_k(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    ideal_ids = (
        [cid for cid in ranked_ids if cid in validated]
        + [cid for cid in ranked_ids if cid not in validated]
    )
    idcg = dcg_at_k(ideal_ids, k, validated)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, k, validated) / idcg


# ---------------------------------------------------------------------------
# Embedding ranking
# ---------------------------------------------------------------------------

def _embed_cache_key(claim_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(claim_ids).encode()).hexdigest()[:16]


def get_embedding_scores(records: list[dict]) -> dict[str, float]:
    """Return {claim_id: cosine_similarity} for BioLORD-2023 embedding ranking."""
    claim_ids = [r["claim_id"] for r in records]
    key = _embed_cache_key(claim_ids)

    emb_path = EMBED_CACHE_DIR / f"embeddings_{key}.npy"
    qry_path = EMBED_CACHE_DIR / f"query_{key}.npy"

    if emb_path.exists() and qry_path.exists():
        claim_emb = np.load(emb_path)
        query_emb = np.load(qry_path)
    else:
        print("  Loading embedding model (first run)...", file=sys.stderr)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in records]
        claim_emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        query_emb = model.encode([QUERY], convert_to_numpy=True)
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, claim_emb)
        np.save(qry_path, query_emb)

    cosine = (claim_emb @ query_emb.T).flatten()
    return {cid: float(cosine[i]) for i, cid in enumerate(claim_ids)}


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_at_coverage(
    physical: list[dict],
    non_physical: list[dict],
    target_coverage: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Randomly drop Physical records until Physical/total == target_coverage."""
    n_non_phys = len(non_physical)
    # target = n_keep / (n_keep + n_non_phys)  =>  n_keep = target * n_non_phys / (1 - target)
    n_keep = int(round(target_coverage * n_non_phys / (1 - target_coverage)))
    n_keep = max(1, min(n_keep, len(physical)))
    indices = rng.choice(len(physical), size=n_keep, replace=False)
    sampled_physical = [physical[i] for i in indices]
    return sampled_physical + non_physical


def evaluate_subsample(records: list[dict], embedding_scores: dict[str, float]) -> dict:
    """Compute metrics for one subsample."""
    validated = {r["claim_id"] for r in records if r["validation_class"] in _VALIDATED}

    # pCite ranking (pcite_score if available, else base_weight)
    score_key = "pcite_score" if any(r["pcite_score"] > 0 for r in records) else "base_weight"
    pcite_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r.get(score_key, 0), reverse=True)
    ]

    # Traditional ranking (replication_count)
    trad_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r.get("replication_count", 0), reverse=True)
    ]

    # Embedding ranking (cosine similarity from full corpus — filter to present claims)
    present_ids = {r["claim_id"] for r in records}
    emb_ranked = sorted(
        [cid for cid in present_ids if cid in embedding_scores],
        key=lambda cid: embedding_scores[cid],
        reverse=True,
    )

    result = {}
    for k in K_VALUES:
        if k > len(records):
            continue
        result[f"P@{k}_pcite"] = precision_at_k(pcite_ranked, k, validated)
        result[f"P@{k}_trad"] = precision_at_k(trad_ranked, k, validated)
        result[f"P@{k}_embed"] = precision_at_k(emb_ranked, k, validated)

    if NDCG_K <= len(records):
        result[f"NDCG@{NDCG_K}_pcite"] = ndcg_at_k(pcite_ranked, NDCG_K, validated)
        result[f"NDCG@{NDCG_K}_trad"] = ndcg_at_k(trad_ranked, NDCG_K, validated)
        result[f"NDCG@{NDCG_K}_embed"] = ndcg_at_k(emb_ranked, NDCG_K, validated)

    return result


# ---------------------------------------------------------------------------
# Main investigation
# ---------------------------------------------------------------------------

def run_boundary_investigation() -> list[dict]:
    records = load_scores(SCORES_PATH)
    print(f"Loaded {len(records)} records from MetaboLights", file=sys.stderr)

    # Separate Physical from non-Physical
    physical = [r for r in records if r["validation_class"] == "PhysicalMeasurement"]
    non_physical = [r for r in records if r["validation_class"] != "PhysicalMeasurement"]
    actual_coverage = len(physical) / len(records)
    print(f"  Physical: {len(physical)} ({actual_coverage:.1%}), "
          f"Non-Physical: {len(non_physical)}", file=sys.stderr)

    # Pre-compute embedding scores on full corpus
    print("  Computing embedding scores...", file=sys.stderr)
    embedding_scores = get_embedding_scores(records)

    # Run subsampling
    results = []
    for target in COVERAGE_LEVELS:
        print(f"\n  Coverage target: {target:.0%}", file=sys.stderr)
        seed_results = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 42)
            subsample = subsample_at_coverage(physical, non_physical, target, rng)
            actual = sum(1 for r in subsample if r["validation_class"] == "PhysicalMeasurement")
            metrics = evaluate_subsample(subsample, embedding_scores)
            metrics["actual_coverage"] = actual / len(subsample)
            metrics["n_total"] = len(subsample)
            metrics["n_physical"] = actual
            seed_results.append(metrics)

        # Average across seeds
        avg = {"target_coverage": target}
        keys = [k for k in seed_results[0] if k not in ("actual_coverage", "n_total", "n_physical")]
        for k in keys:
            vals = [sr[k] for sr in seed_results if k in sr]
            avg[k] = float(np.mean(vals))
        avg["actual_coverage"] = float(np.mean([sr["actual_coverage"] for sr in seed_results]))
        avg["n_total_avg"] = float(np.mean([sr["n_total"] for sr in seed_results]))
        results.append(avg)

        print(f"    actual={avg['actual_coverage']:.1%}  "
              f"P@50: pcite={avg.get('P@50_pcite', 0):.3f} "
              f"trad={avg.get('P@50_trad', 0):.3f} "
              f"embed={avg.get('P@50_embed', 0):.3f}", file=sys.stderr)

    # Try to load PRIDE scores and add as overlay point
    pride_point = None
    if PRIDE_SCORES_PATH.exists():
        pride_records = load_scores(PRIDE_SCORES_PATH)
        if pride_records:
            n_phys = sum(1 for r in pride_records
                         if r["validation_class"] == "PhysicalMeasurement")
            coverage = n_phys / len(pride_records)
            pride_metrics = evaluate_subsample(pride_records, embedding_scores)
            pride_point = {"target_coverage": coverage, **pride_metrics,
                           "actual_coverage": coverage, "source": "PRIDE_v1"}
            results.append(pride_point)
            print(f"\n  PRIDE v1 overlay: coverage={coverage:.1%} "
                  f"P@50: pcite={pride_metrics.get('P@50_pcite', 0):.3f} "
                  f"trad={pride_metrics.get('P@50_trad', 0):.3f}",
                  file=sys.stderr)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {OUTPUT_PATH}", file=sys.stderr)

    # Generate figure
    fig = make_figure(results, pride_point)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH)
    plt.close(fig)
    print(f"Figure → {FIGURE_PATH}", file=sys.stderr)

    return results


def make_figure(results: list[dict], pride_point: dict | None = None) -> plt.Figure:
    """Line plot: P@50 vs Physical coverage for three ranking methods."""
    # Separate MetaboLights subsamples from PRIDE overlay
    metabo = [r for r in results if r.get("source") != "PRIDE_v1"]
    metabo.sort(key=lambda r: r["target_coverage"])

    coverages = [r["actual_coverage"] * 100 for r in metabo]
    p50_pcite = [r.get("P@50_pcite", 0) for r in metabo]
    p50_trad = [r.get("P@50_trad", 0) for r in metabo]
    p50_embed = [r.get("P@50_embed", 0) for r in metabo]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(coverages, p50_pcite, "-o", color="#d62728", ms=7, lw=2,
            label="pCite", zorder=3)
    ax.plot(coverages, p50_trad, "-s", color="#aec7e8", ms=7, lw=2,
            label="Traditional (replication count)", zorder=2)
    ax.plot(coverages, p50_embed, "-^", color="#2ca02c", ms=7, lw=2,
            label="Embedding (BioLORD-2023)", zorder=2)

    # Find and annotate crossover
    for i in range(1, len(coverages)):
        if p50_pcite[i] > p50_trad[i] and p50_pcite[i - 1] <= p50_trad[i - 1]:
            # Linear interpolation for crossover x
            x0, x1 = coverages[i - 1], coverages[i]
            diff0 = p50_pcite[i - 1] - p50_trad[i - 1]
            diff1 = p50_pcite[i] - p50_trad[i]
            crossover_x = x0 + (x1 - x0) * (-diff0) / (diff1 - diff0 + 1e-9)
            ax.axvline(crossover_x, ls=":", color="#888888", lw=1.5, alpha=0.7)
            ax.annotate(f"Crossover\n~{crossover_x:.0f}%",
                        xy=(crossover_x, ax.get_ylim()[0]),
                        xytext=(crossover_x + 3, 0.15),
                        fontsize=9, color="#888888",
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
            break

    # PRIDE overlay point
    if pride_point:
        px = pride_point["actual_coverage"] * 100
        py = pride_point.get("P@50_pcite", 0)
        ax.scatter([px], [py], marker="D", c="#ff7f0e", s=120, zorder=5,
                   edgecolors="black", linewidths=1)
        ax.annotate(f"PRIDE v1\n({px:.0f}% cov.)",
                    xy=(px, py), xytext=(px + 3, py + 0.04),
                    fontsize=9, color="#ff7f0e",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1))

    ax.set_xlabel("Physical-tier coverage (%)")
    ax.set_ylabel("Precision@50")
    ax.set_title("pCite advantage by Physical coverage level")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run_boundary_investigation()
