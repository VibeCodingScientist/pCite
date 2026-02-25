"""Run steps 3-5 only, using existing data/papers.jsonl and data/claims.jsonl."""
import asyncio, sys
from pcite import validate, graph, evaluate


async def main():
    print("3/5  Validation + nanopublications...")
    n = validate.process_claims()
    print(f"     {n} claims classified\n")

    print("4/5  pCite graph (OpenAlex + Claude Sonnet)...")
    G = await graph.build_full_graph()
    print(f"     {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("5/5  Experiment...")
    r   = evaluate.run_experiment()
    mw  = r["mann_whitney"]
    p50 = r["precision_50"]
    ng  = r["ndcg_50"]

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  n total:          {r['n_total']:,}")
    print(f"  n validated:      {mw['n_validated']:,}")
    pf = "PASS" if mw["p_value"] < 0.05 else "FAIL"
    print(f"  Mann-Whitney p:   {mw['p_value']:.4f}   {pf}")
    print(f"  Precision@50:     {p50['precision_pcite']:.3f} vs "
          f"{p50['precision_traditional']:.3f}  ({p50['lift']:.1f}x lift)")
    print(f"  NDCG@50:          {ng['ndcg_pcite']:.4f} vs "
          f"{ng['ndcg_traditional']:.4f}")
    print(f"{sep}\n")

    holds = (
        mw["p_value"] < 0.05
        and p50["lift"] > 1.0
        and ng["ndcg_pcite"] > ng["ndcg_traditional"]
    )
    print("PASS: Hypothesis holds.\n" if holds else "FAIL: Hypothesis did not hold.\n")
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
