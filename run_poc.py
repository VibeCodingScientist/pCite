"""
run_poc.py

Exits 0 if hypothesis holds. Exits 1 if it does not.
This file is simultaneously the demo, the integration test,
and the reproducibility proof.

Usage:
  python run_poc.py
  python run_poc.py --dry-run   # evaluate cached data, skip API calls
"""

import asyncio, sys, argparse
from ncite import corpus, extract, validate, graph, evaluate


async def main(dry_run: bool = False) -> int:
    print("\nnCite PoC — Validation-Weighted Citation Graph for Metabolomics\n")

    if not dry_run:
        print("1/5  Corpus (PubMed + MetaboLights)...")
        print(f"     {await corpus.build_corpus()} papers\n")

        print("2/5  Extraction (Claude Sonnet)...")
        print(f"     {await extract.process_corpus()} unique claims\n")

        print("3/5  Validation + nanopublications...")
        print(f"     {validate.process_claims()} claims classified\n")

        print("4/5  nCite graph (OpenAlex + Claude Sonnet)...")
        G = await graph.build_full_graph()
        print(f"     {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("5/5  Experiment...")
    r   = evaluate.run_experiment()
    mw  = r["mann_whitney"]
    p50 = r["precision_50"]
    ng  = r["ndcg_50"]

    print(f"\n{'='*55}")
    print(f"  n total:          {r['n_total']:,}")
    print(f"  n validated:      {mw['n_validated']:,}")
    print(f"  Mann-Whitney p:   {mw['p_value']:.4f}   "
          f"{'PASS' if mw['p_value'] < 0.05 else 'FAIL'}")
    print(f"  Precision@50:     {p50['precision_ncite']:.3f} vs "
          f"{p50['precision_traditional']:.3f}  ({p50['lift']:.1f}x lift)")
    print(f"  NDCG@50:          {ng['ndcg_ncite']:.4f} vs "
          f"{ng['ndcg_traditional']:.4f}")
    print(f"{'='*55}\n")

    holds = (
        mw["p_value"] < 0.05
        and p50["lift"] >= 1.0
        and ng["ndcg_ncite"] > ng["ndcg_traditional"]
    )
    print("PASS: Hypothesis holds.\n" if holds else "FAIL: Hypothesis did not hold.\n")
    return 0 if holds else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    sys.exit(asyncio.run(main(p.parse_args().dry_run)))
