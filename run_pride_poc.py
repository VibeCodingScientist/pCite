"""
run_pride_poc.py

PRIDE corpus PoC — cancer proteomics.
Mirrors run_poc.py with PRIDE-specific paths and domain="proteomics".

Usage:
  python run_pride_poc.py
  python run_pride_poc.py --dry-run      # evaluate cached data, skip API calls
  python run_pride_poc.py --skip-corpus  # skip corpus build, run extract+validate+graph+eval
"""

import asyncio, sys, argparse
from functools import partial
from pathlib import Path

import pride_corpus
from pcite import extract, validate, graph, evaluate

# All paths under data/pride/
DATA_DIR       = Path("data/pride")
CLAIMS_PATH    = DATA_DIR / "claims.jsonl"
NANOPUBS_PATH  = DATA_DIR / "nanopubs"
GRAPH_OUT      = DATA_DIR / "graph.graphml"
SCORES_OUT     = DATA_DIR / "scores.jsonl"
CLASSIFY_CACHE = DATA_DIR / "classify_cache.json"
CITATION_CACHE = DATA_DIR / "citation_cache.json"
RESULTS_OUT    = DATA_DIR / "results.json"
FIGURES_DIR    = Path("figures/pride")


def _load_pride_claims():
    return extract.load_claims(CLAIMS_PATH)


async def main(dry_run: bool = False, skip_corpus: bool = False) -> int:
    print("\npCite PoC — PRIDE Cancer Proteomics Corpus\n")

    if not dry_run:
        if not skip_corpus:
            print("1/5  Corpus (PRIDE-first + PubMed disease cluster)...")
            print(f"     {await pride_corpus.build_corpus()} papers\n")

        print("2/5  Extraction (Claude Sonnet — proteomics domain)...")
        print(f"     {await extract.process_corpus(
            papers_loader=pride_corpus.load_papers,
            output_path=CLAIMS_PATH,
            domain='proteomics',
        )} unique claims\n")

        print("3/5  Validation + nanopublications...")
        print(f"     {validate.process_claims(
            claims_path=CLAIMS_PATH,
            nanopubs_path=NANOPUBS_PATH,
            papers_loader=pride_corpus.load_papers,
            claims_loader=_load_pride_claims,
        )} claims classified\n")

        print("4/5  pCite graph (OpenAlex + Gemini Flash)...")
        G = await graph.build_full_graph(
            claims_loader=_load_pride_claims,
            graph_out=GRAPH_OUT,
            scores_out=SCORES_OUT,
            classify_cache=CLASSIFY_CACHE,
            citation_cache=CITATION_CACHE,
        )
        print(f"     {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("5/5  Experiment...")
    r   = evaluate.run_experiment(
        scores_path=SCORES_OUT,
        results_path=RESULTS_OUT,
        figures_dir=FIGURES_DIR,
    )
    mw  = r["mann_whitney"]
    p50 = r["precision_50"]
    ng  = r["ndcg_50"]

    print(f"\n{'='*55}")
    print(f"  n total:          {r['n_total']:,}")
    print(f"  n validated:      {mw['n_validated']:,}")
    print(f"  Mann-Whitney p:   {mw['p_value']:.4f}   "
          f"{'PASS' if mw['p_value'] < 0.05 else 'FAIL'}")
    print(f"  Precision@50:     {p50['precision_pcite']:.3f} vs "
          f"{p50['precision_traditional']:.3f}  ({p50['lift']:.1f}x lift)")
    print(f"  NDCG@50:          {ng['ndcg_pcite']:.4f} vs "
          f"{ng['ndcg_traditional']:.4f}")
    print(f"{'='*55}\n")

    holds = (
        mw["p_value"] < 0.05
        and p50["lift"] >= 1.0
        and ng["ndcg_pcite"] > ng["ndcg_traditional"]
    )
    print("PASS: Hypothesis holds.\n" if holds else "FAIL: Hypothesis did not hold.\n")
    return 0 if holds else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-corpus", action="store_true")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.dry_run, args.skip_corpus)))
