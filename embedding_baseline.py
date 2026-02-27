"""
Embedding baseline: BioLORD-2023 semantic similarity vs pCite vs traditional.

Addresses reviewer critique that traditional citation count is a deliberately
weak baseline.  Adds a domain-specific embedding model (BioLORD-2023) that
ranks claims by cosine similarity to a biomarker-validation query.

No API calls.  Model runs locally via sentence-transformers.
First run: ~60s (model download + encoding).  Cached reruns: <1s.

Output:
  data/embedding/embedding_table.json
"""

import hashlib, json, math, sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reuse existing evaluate helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pcite.evaluate import load_scores, _VALIDATED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "FremyCompany/BioLORD-2023"
QUERY = (
    "metabolite measured by mass spectrometry instrument "
    "physical experiment validated biomarker"
)
K_VALUES = [10, 25, 50, 100, 200]
CACHE_DIR = Path("data/embedding")

# ---------------------------------------------------------------------------
# Metric helpers (local — does not modify evaluate.py)
# ---------------------------------------------------------------------------

def precision_at_k_single(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(1 for cid in ranked_ids[:k] if cid in validated) / k


def dcg_at_k(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(
        (1.0 if ranked_ids[i] in validated else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(ranked_ids)))
    )


def ndcg_at_k_single(ranked_ids: list[str], k: int, validated: set[str], n_total: int) -> float:
    # Ideal: all validated claims first
    ideal_ids = (
        [cid for cid in ranked_ids if cid in validated]
        + [cid for cid in ranked_ids if cid not in validated]
    )
    idcg = dcg_at_k(ideal_ids, k, validated)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, k, validated) / idcg


# ---------------------------------------------------------------------------
# Embedding + caching
# ---------------------------------------------------------------------------

def _cache_key(claim_ids: list[str]) -> str:
    """SHA-256 of ordered claim IDs — invalidates if scores.jsonl changes."""
    return hashlib.sha256("\n".join(claim_ids).encode()).hexdigest()[:16]


def embed_claims(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (claim_embeddings, query_embedding), using cache when possible."""
    claim_ids = [r["claim_id"] for r in records]
    key = _cache_key(claim_ids)

    emb_path = CACHE_DIR / f"embeddings_{key}.npy"
    qry_path = CACHE_DIR / f"query_{key}.npy"

    if emb_path.exists() and qry_path.exists():
        print("  Using cached embeddings", file=sys.stderr)
        return np.load(emb_path), np.load(qry_path)

    # Cache miss — load model and encode
    print(f"  Loading model {MODEL_NAME}...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in records]
    print(f"  Encoding {len(texts)} claims...", file=sys.stderr)
    claim_emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    query_emb = model.encode([QUERY], convert_to_numpy=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, claim_emb)
    np.save(qry_path, query_emb)
    print(f"  Cached → {emb_path}", file=sys.stderr)

    return claim_emb, query_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_embedding_baseline() -> list[dict]:
    records = load_scores()
    print(f"Loaded {len(records)} records", file=sys.stderr)

    validated = {r["claim_id"] for r in records if r["validation_class"] in _VALIDATED}

    # --- Three rankings ---

    # 1. pCite (pcite_score — citation-amplified, matches Table 3)
    pcite_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r["pcite_score"], reverse=True)
    ]

    # 2. Traditional (replication_count)
    trad_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r["replication_count"], reverse=True)
    ]

    # 3. Embedding (cosine similarity)
    claim_emb, query_emb = embed_claims(records)
    cosine_scores = (claim_emb @ query_emb.T).flatten()
    emb_order = np.argsort(-cosine_scores)
    emb_ranked = [records[i]["claim_id"] for i in emb_order]

    # --- Evaluate ---
    n_total = len(records)
    results = []

    for k in K_VALUES:
        row = {
            "metric": f"P@{k}",
            "pcite": round(precision_at_k_single(pcite_ranked, k, validated), 4),
            "traditional": round(precision_at_k_single(trad_ranked, k, validated), 4),
            "embedding": round(precision_at_k_single(emb_ranked, k, validated), 4),
        }
        results.append(row)

    # NDCG@50
    k_ndcg = 50
    results.append({
        "metric": f"NDCG@{k_ndcg}",
        "pcite": round(ndcg_at_k_single(pcite_ranked, k_ndcg, validated, n_total), 4),
        "traditional": round(ndcg_at_k_single(trad_ranked, k_ndcg, validated, n_total), 4),
        "embedding": round(ndcg_at_k_single(emb_ranked, k_ndcg, validated, n_total), 4),
    })

    # --- Output ---
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = CACHE_DIR / "embedding_table.json"
    table_path.write_text(json.dumps(results, indent=2))

    # Pretty-print
    print(f"\n{'Metric':<10} {'pCite':>8} {'Trad.':>8} {'Embed.':>8}", file=sys.stderr)
    print("-" * 38, file=sys.stderr)
    for row in results:
        print(
            f"{row['metric']:<10} {row['pcite']:>8.4f} {row['traditional']:>8.4f} "
            f"{row['embedding']:>8.4f}",
            file=sys.stderr,
        )
    print(f"\nTable → {table_path}", file=sys.stderr)

    return results


if __name__ == "__main__":
    run_embedding_baseline()
