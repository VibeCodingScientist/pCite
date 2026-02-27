"""
pride_graded_eval.py

Post-processing evaluation: graded deposit quality (1.0-10.0) vs flat pCite
vs traditional vs embedding baseline.

For Physical claims, replaces flat base_weight (using VALIDATION_WEIGHT[PHYSICAL]=10.0)
with per-paper graded weight from PRIDE deposit quality scores. Re-ranks.
Evaluates all four ranking methods.

Table 4 — 4 columns x 6 rows:
  Columns: Graded pCite | Flat pCite | Traditional | Embedding (BioLORD-2023)
  Rows: P@10, P@25, P@50, P@100, P@200, NDCG@50

No API calls. Runs locally.

Output: data/pride/graded_table.json
"""

import hashlib, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pcite.evaluate import _VALIDATED
from pcite.extract import load_claims
from pcite.models import VALIDATION_WEIGHT, ValidationClass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = Path("data/pride")
SCORES_PATH  = DATA_DIR / "scores.jsonl"
CLAIMS_PATH  = DATA_DIR / "claims.jsonl"
QUALITY_PATH = DATA_DIR / "deposit_quality.json"
OUTPUT_PATH  = DATA_DIR / "graded_table.json"

K_VALUES = [10, 25, 50, 100, 200]
MODEL_NAME = "FremyCompany/BioLORD-2023"
QUERY = (
    "protein measured by mass spectrometry instrument "
    "physical experiment validated cancer biomarker"
)
EMB_CACHE_DIR = DATA_DIR / "embedding"

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def precision_at_k_single(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(1 for cid in ranked_ids[:k] if cid in validated) / k


def dcg_at_k(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    return sum(
        (1.0 if ranked_ids[i] in validated else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(ranked_ids)))
    )


def ndcg_at_k_single(ranked_ids: list[str], k: int, validated: set[str]) -> float:
    ideal_ids = (
        [cid for cid in ranked_ids if cid in validated]
        + [cid for cid in ranked_ids if cid not in validated]
    )
    idcg = dcg_at_k(ideal_ids, k, validated)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, k, validated) / idcg


# ---------------------------------------------------------------------------
# Graded reweight
# ---------------------------------------------------------------------------

def build_claim_quality_map(
    claims_path: Path, quality: dict[str, float],
) -> dict[str, float]:
    """Map claim_id -> mean deposit quality across Physical provenance entries.

    For claims with multiple Physical provenance entries (from different PRIDE
    projects), averages their quality scores. Returns only Physical claims.
    """
    claims = load_claims(claims_path)
    claim_quality: dict[str, float] = {}
    for c in claims:
        qualities = []
        for p in c.provenance:
            if p.deposit_id and p.deposit_id in quality:
                qualities.append(quality[p.deposit_id])
        if qualities:
            claim_quality[c.id] = sum(qualities) / len(qualities)
    return claim_quality


def reweight_graded(
    records: list[dict], claim_quality: dict[str, float],
) -> list[dict]:
    """Return records with Physical claims' base_weight using graded deposit quality."""
    out = []
    for r in records:
        if r["claim_id"] in claim_quality:
            # Replace VALIDATION_WEIGHT[PHYSICAL] (10.0) with graded quality
            graded_w = claim_quality[r["claim_id"]]
            new_bw = graded_w * math.log2(r["replication_count"] + 1)
            out.append({**r, "base_weight": new_bw})
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _cache_key(claim_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(claim_ids).encode()).hexdigest()[:16]


def embed_claims(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (claim_embeddings, query_embedding), using cache when possible."""
    claim_ids = [r["claim_id"] for r in records]
    key = _cache_key(claim_ids)

    emb_path = EMB_CACHE_DIR / f"embeddings_{key}.npy"
    qry_path = EMB_CACHE_DIR / f"query_{key}.npy"

    if emb_path.exists() and qry_path.exists():
        print("  Using cached PRIDE embeddings", file=sys.stderr)
        return np.load(emb_path), np.load(qry_path)

    print(f"  Loading model {MODEL_NAME}...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in records]
    print(f"  Encoding {len(texts)} claims...", file=sys.stderr)
    claim_emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    query_emb = model.encode([QUERY], convert_to_numpy=True)

    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, claim_emb)
    np.save(qry_path, query_emb)
    print(f"  Cached -> {emb_path}", file=sys.stderr)

    return claim_emb, query_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_graded_eval() -> list[dict]:
    records = [json.loads(l) for l in SCORES_PATH.read_text().splitlines() if l]
    quality = json.loads(QUALITY_PATH.read_text())
    print(f"Loaded {len(records)} records, {len(quality)} quality scores",
          file=sys.stderr)

    validated = {r["claim_id"] for r in records if r["validation_class"] in _VALIDATED}
    claim_quality = build_claim_quality_map(CLAIMS_PATH, quality)
    print(f"  {len(claim_quality)} claims with graded deposit quality",
          file=sys.stderr)

    # Graded records
    graded_records = reweight_graded(records, claim_quality)

    # --- Four rankings ---

    # 1. Graded pCite (base_weight uses graded deposit quality)
    graded_ranked = [
        r["claim_id"]
        for r in sorted(graded_records, key=lambda r: r["base_weight"], reverse=True)
    ]

    # 2. Flat pCite (original base_weight with flat 10.0)
    flat_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r["base_weight"], reverse=True)
    ]

    # 3. Traditional (replication count)
    trad_ranked = [
        r["claim_id"]
        for r in sorted(records, key=lambda r: r["replication_count"], reverse=True)
    ]

    # 4. Embedding (BioLORD-2023 cosine similarity)
    claim_emb, query_emb = embed_claims(records)
    cosine_scores = (claim_emb @ query_emb.T).flatten()
    emb_order = np.argsort(-cosine_scores)
    emb_ranked = [records[i]["claim_id"] for i in emb_order]

    # --- Evaluate ---
    results = []

    for k in K_VALUES:
        row = {
            "metric": f"P@{k}",
            "graded_pcite": round(precision_at_k_single(graded_ranked, k, validated), 4),
            "flat_pcite": round(precision_at_k_single(flat_ranked, k, validated), 4),
            "traditional": round(precision_at_k_single(trad_ranked, k, validated), 4),
            "embedding": round(precision_at_k_single(emb_ranked, k, validated), 4),
        }
        results.append(row)

    # NDCG@50
    k_ndcg = 50
    n_total = len(records)
    results.append({
        "metric": f"NDCG@{k_ndcg}",
        "graded_pcite": round(ndcg_at_k_single(graded_ranked, k_ndcg, validated), 4),
        "flat_pcite": round(ndcg_at_k_single(flat_ranked, k_ndcg, validated), 4),
        "traditional": round(ndcg_at_k_single(trad_ranked, k_ndcg, validated), 4),
        "embedding": round(ndcg_at_k_single(emb_ranked, k_ndcg, validated), 4),
    })

    # --- Output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # Pretty-print
    print(f"\n{'Metric':<10} {'Graded':>8} {'Flat':>8} {'Trad.':>8} {'Embed.':>8}",
          file=sys.stderr)
    print("-" * 48, file=sys.stderr)
    for row in results:
        print(
            f"{row['metric']:<10} {row['graded_pcite']:>8.4f} "
            f"{row['flat_pcite']:>8.4f} {row['traditional']:>8.4f} "
            f"{row['embedding']:>8.4f}",
            file=sys.stderr,
        )
    print(f"\nTable 4 -> {OUTPUT_PATH}", file=sys.stderr)

    return results


if __name__ == "__main__":
    run_graded_eval()
