"""
ncite.graph

Build nCite graph. Compute scores.

nCite score = Σ (NCiteType.weight × source.base_weight) for all incoming edges
base_weight = ValidationClass.weight × log₂(replication_count + 1)

The formula is 5 lines. The scientific argument is in the constants.

Output: data/graph.graphml + data/scores.jsonl
Run:    python -m ncite.graph
"""

import asyncio, functools, json, sys
from pathlib import Path
import httpx, networkx as nx, anthropic
from ncite.models import Claim, NCiteType, NCITE_WEIGHT

GRAPH_OUT  = Path("data/graph.graphml")
SCORES_OUT = Path("data/scores.jsonl")
OPENALEX   = "https://api.openalex.org/works"

_client = anthropic.Anthropic()
_sem    = asyncio.Semaphore(5)


def build_claim_nodes(claims: list[Claim]) -> nx.DiGraph:
    G = nx.DiGraph()
    for c in claims:
        G.add_node(c.id, **{
            "validation_class":  c.validation_class.value,
            "base_weight":       c.base_weight,
            "replication_count": c.replication_count,
            "predicate":         c.predicate.value,
            "subject":           c.subject.name,
            "object":            c.object.name,
        })
    return G


async def _citing_dois(doi: str, client: httpx.AsyncClient) -> list[str]:
    """OpenAlex citation lookup. Fully open, no auth, 100k req/day."""
    async with _sem:
        try:
            resp = await client.get(OPENALEX, params={
                "filter": f"cites:{doi}", "select": "doi",
                "per-page": 50, "mailto": "research@ncite.org",
            }, timeout=15)
            return [
                w["doi"].replace("https://doi.org/", "")
                for w in resp.json().get("results", []) if w.get("doi")
            ]
        except Exception:
            return []


@functools.lru_cache(maxsize=50_000)
def _classify(src_text: str, tgt_text: str) -> NCiteType:
    """
    Classify citation relationship. Claude Haiku — fast and cheap for single-word output.
    Cached: most claim pairs recur across the corpus.
    """
    resp = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content":
            f'Source: "{src_text}"\nTarget: "{tgt_text}"\n\n'
            f"One word:\nsupports | extends | replicates | contradicts | applies"}]
    )
    word = resp.content[0].text.strip().lower().split()[0]
    return NCiteType(word) if word in NCiteType._value2member_map_ else NCiteType.SUPPORTS


def compute_ncite_scores(G: nx.DiGraph) -> dict[str, float]:
    """
    nCite score = Σ incoming edge weights.

    weight per edge = NCiteType.weight × source.base_weight
    source.base_weight = ValidationClass.weight × log₂(replication_count + 1)

    Reading the graph structure is sufficient to compute trust.
    No manual scoring. No external labels.
    """
    return {
        node: sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        for node in G.nodes()
    }


async def build_full_graph() -> nx.DiGraph:
    from ncite.extract import load_claims
    claims = load_claims()
    by_doi: dict[str, list[Claim]] = {}
    for c in claims:
        for p in c.provenance:
            by_doi.setdefault(p.doi, []).append(c)

    G = build_claim_nodes(claims)
    print(f"  {G.number_of_nodes()} nodes", file=sys.stderr)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        dois       = list(by_doi.keys())
        citing_map = dict(zip(
            dois,
            await asyncio.gather(*[_citing_dois(doi, client) for doi in dois])
        ))

    edges = 0
    for target_doi, citing_dois_list in citing_map.items():
        for tgt in by_doi.get(target_doi, []):
            for citing_doi in citing_dois_list:
                for src in by_doi.get(citing_doi, []):
                    if src.id == tgt.id:
                        continue
                    ntype  = _classify(
                        f"{src.subject.name} {src.predicate.value} {src.object.name}",
                        f"{tgt.subject.name} {tgt.predicate.value} {tgt.object.name}",
                    )
                    G.add_edge(src.id, tgt.id, type=ntype.value,
                               weight=NCITE_WEIGHT[ntype] * src.base_weight,
                               source_weight=src.base_weight)
                    edges += 1

    print(f"  {edges} edges", file=sys.stderr)
    scores = compute_ncite_scores(G)
    nx.set_node_attributes(G, scores, "ncite_score")
    nx.write_graphml(G, GRAPH_OUT)

    with SCORES_OUT.open("w") as f:
        for c in claims:
            f.write(json.dumps({
                "claim_id":          c.id,
                "ncite_score":       scores.get(c.id, 0.0),
                "validation_class":  c.validation_class.value,
                "replication_count": c.replication_count,
                "base_weight":       c.base_weight,
                "predicate":         c.predicate.value,
                "subject":           c.subject.name,
                "object":            c.object.name,
            }) + "\n")
    return G


if __name__ == "__main__":
    G = asyncio.run(build_full_graph())
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)
