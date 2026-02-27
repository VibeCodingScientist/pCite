"""
pcite.graph

Build pCite graph. Compute scores.

pCite score = Σ (PCiteType.weight × source.base_weight) for all incoming edges
base_weight = ValidationClass.weight × log₂(replication_count + 1)

The formula is 5 lines. The scientific argument is in the constants.

Output: data/graph.graphml + data/scores.jsonl
Run:    python -m pcite.graph
"""

import asyncio, functools, json, sys, time
from pathlib import Path
import httpx, networkx as nx
from google import genai
from pcite.models import Claim, PCiteType, PCITE_WEIGHT
from pcite import config

GRAPH_OUT       = Path("data/graph.graphml")
SCORES_OUT      = Path("data/scores.jsonl")
CLASSIFY_CACHE  = Path("data/classify_cache.json")
CITATION_CACHE  = Path("data/citation_cache.json")
OPENALEX        = "https://api.openalex.org/works"

_gemini_client: genai.Client | None = None
_sem = asyncio.Semaphore(5)


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


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


async def _resolve_openalex_ids(
    dois: list[str], client: httpx.AsyncClient,
) -> dict[str, str]:
    """Batch-resolve bare DOIs to OpenAlex work IDs (50 per request)."""
    mapping: dict[str, str] = {}
    batch_size = 50
    for i in range(0, len(dois), batch_size):
        batch = dois[i : i + batch_size]
        pipe_filter = "|".join(f"https://doi.org/{d}" for d in batch)
        async with _sem:
            try:
                resp = await client.get(OPENALEX, params={
                    "filter": f"doi:{pipe_filter}",
                    "select": "id,doi",
                    "per-page": batch_size,
                    "mailto": config.OPENALEX_EMAIL,
                }, timeout=30)
                for w in resp.json().get("results", []):
                    if w.get("doi") and w.get("id"):
                        bare = w["doi"].replace("https://doi.org/", "")
                        mapping[bare] = w["id"]
            except Exception:
                pass
    print(f"  resolved {len(mapping)}/{len(dois)} DOIs to OpenAlex IDs",
          file=sys.stderr)
    return mapping


async def _citing_dois(openalex_id: str, client: httpx.AsyncClient) -> list[str]:
    """OpenAlex citation lookup using OpenAlex work ID."""
    async with _sem:
        try:
            resp = await client.get(OPENALEX, params={
                "filter": f"cites:{openalex_id}", "select": "doi",
                "per-page": 50, "mailto": config.OPENALEX_EMAIL,
            })
            return [
                w["doi"].replace("https://doi.org/", "")
                for w in resp.json().get("results", []) if w.get("doi")
            ]
        except Exception:
            return []


def _load_classify_cache(path: Path | None = None) -> dict[str, str]:
    path = path or CLASSIFY_CACHE
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_classify_cache(cache: dict[str, str], path: Path | None = None) -> None:
    path = path or CLASSIFY_CACHE
    path.write_text(json.dumps(cache))


_disk_cache: dict[str, str] = {}


@functools.lru_cache(maxsize=50_000)
def _classify(src_text: str, tgt_text: str) -> PCiteType:
    """
    Classify citation relationship via Gemini Flash.
    Cached in memory (lru_cache) and on disk (classify_cache.json).
    Retries up to 5 times with exponential backoff.
    """
    key = f"{src_text}||{tgt_text}"
    if key in _disk_cache:
        word = _disk_cache[key]
        return PCiteType(word) if word in PCiteType._value2member_map_ else PCiteType.SUPPORTS

    for attempt in range(5):
        try:
            resp = _get_gemini_client().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=(
                    f'Classify the relationship between these two biomedical claims.\n'
                    f'Source: "{src_text}"\nTarget: "{tgt_text}"\n\n'
                    f"Reply with EXACTLY one word, no formatting, no markdown, no asterisks:\n"
                    f"supports | extends | replicates | contradicts | applies"
                ),
                config={"max_output_tokens": 10},
            )
            word = resp.text.strip().lower().strip("*").split()[0].strip("*")
            _disk_cache[key] = word
            return PCiteType(word) if word in PCiteType._value2member_map_ else PCiteType.SUPPORTS
        except Exception as e:
            wait = 2 ** attempt
            print(f"  classify retry {attempt+1}/5 ({e}), waiting {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    # All retries exhausted — default to SUPPORTS
    print(f"  classify failed after 5 retries, defaulting to SUPPORTS",
          file=sys.stderr)
    _disk_cache[key] = "supports"
    return PCiteType.SUPPORTS


def compute_pcite_scores(G: nx.DiGraph) -> dict[str, float]:
    """
    pCite score = Σ incoming edge weights.

    weight per edge = PCiteType.weight × source.base_weight
    source.base_weight = ValidationClass.weight × log₂(replication_count + 1)

    Reading the graph structure is sufficient to compute trust.
    No manual scoring. No external labels.
    """
    return {
        node: sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        for node in G.nodes()
    }


def compute_propagated_score(G: nx.DiGraph, alpha: float = 0.1) -> dict[str, float]:
    """
    Neighbourhood-propagated pCite score.

    For each node, adds alpha × (mean pCite score of neighbours) to its own
    pCite score. Neighbours = all nodes connected by an edge in either direction.
    """
    pcite = {n: G.nodes[n].get("pcite_score", 0.0) for n in G.nodes()}
    propagated = {}
    for node in G.nodes():
        neighbours = set(G.predecessors(node)) | set(G.successors(node))
        if neighbours:
            neighbour_mean = sum(pcite.get(n, 0.0) for n in neighbours) / len(neighbours)
            propagated[node] = pcite[node] + alpha * neighbour_mean
        else:
            propagated[node] = pcite[node]
    return propagated


async def build_full_graph(
    claims_loader=None,
    graph_out: Path | None = None,
    scores_out: Path | None = None,
    classify_cache: Path | None = None,
    citation_cache: Path | None = None,
) -> nx.DiGraph:
    graph_out = graph_out or GRAPH_OUT
    scores_out = scores_out or SCORES_OUT
    classify_cache_path = classify_cache or CLASSIFY_CACHE
    citation_cache_path = citation_cache or CITATION_CACHE
    if claims_loader is None:
        from pcite.extract import load_claims
        claims_loader = load_claims
    claims = claims_loader()
    by_doi: dict[str, list[Claim]] = {}
    for c in claims:
        for p in c.provenance:
            by_doi.setdefault(p.doi, []).append(c)

    G = build_claim_nodes(claims)
    print(f"  {G.number_of_nodes()} nodes", file=sys.stderr)

    # Load cached citations if available, otherwise query OpenAlex
    if citation_cache_path.exists():
        citing_map = json.loads(citation_cache_path.read_text())
        print(f"  loaded {len(citing_map)} DOIs from citation cache",
              file=sys.stderr)
    else:
        timeout = httpx.Timeout(30.0, connect=10.0)
        limits  = httpx.Limits(max_connections=5, max_keepalive_connections=3)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                     limits=limits) as client:
            dois       = list(by_doi.keys())
            oa_ids     = await _resolve_openalex_ids(dois, client)
            resolved   = [(doi, oa_ids[doi]) for doi in dois if doi in oa_ids]

            citing_map: dict[str, list[str]] = {}
            batch_size = 50
            for i in range(0, len(resolved), batch_size):
                batch = resolved[i : i + batch_size]
                results = await asyncio.gather(*[
                    _citing_dois(oa_id, client) for _, oa_id in batch
                ])
                for (doi, _), result in zip(batch, results):
                    citing_map[doi] = result
                print(f"  citations: {min(i + batch_size, len(resolved))}"
                      f"/{len(resolved)} DOIs queried", file=sys.stderr)

        # Save citation cache
        citation_cache_path.parent.mkdir(parents=True, exist_ok=True)
        citation_cache_path.write_text(json.dumps(citing_map))
        print(f"  saved citation cache ({len(citing_map)} DOIs)",
              file=sys.stderr)

    # Count candidate edges first for progress tracking
    candidates = []
    for target_doi, citing_dois_list in citing_map.items():
        for tgt in by_doi.get(target_doi, []):
            for citing_doi in citing_dois_list:
                for src in by_doi.get(citing_doi, []):
                    if src.id != tgt.id:
                        candidates.append((src, tgt))

    total = len(candidates)
    print(f"  {total} candidate edges to classify", file=sys.stderr)

    global _disk_cache
    _disk_cache = _load_classify_cache(classify_cache_path)
    cached = sum(1 for s, t in candidates
                 if f"{s.subject.name} {s.predicate.value} {s.object.name}||"
                    f"{t.subject.name} {t.predicate.value} {t.object.name}" in _disk_cache)
    if cached:
        print(f"  {cached}/{total} already cached on disk", file=sys.stderr)

    edges = 0
    for i, (src, tgt) in enumerate(candidates):
        ntype = _classify(
            f"{src.subject.name} {src.predicate.value} {src.object.name}",
            f"{tgt.subject.name} {tgt.predicate.value} {tgt.object.name}",
        )
        G.add_edge(src.id, tgt.id, type=ntype.value,
                   weight=PCITE_WEIGHT[ntype] * src.base_weight,
                   source_weight=src.base_weight)
        edges += 1
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  classify: {i + 1}/{total} edges", file=sys.stderr)
            _save_classify_cache(_disk_cache, classify_cache_path)

    print(f"  {edges} edges added", file=sys.stderr)
    scores = compute_pcite_scores(G)
    nx.set_node_attributes(G, scores, "pcite_score")
    propagated = compute_propagated_score(G)
    nx.set_node_attributes(G, propagated, "pcite_propagated")
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graph_out)

    scores_out.parent.mkdir(parents=True, exist_ok=True)
    with scores_out.open("w") as f:
        for c in claims:
            f.write(json.dumps({
                "claim_id":          c.id,
                "pcite_score":       scores.get(c.id, 0.0),
                "pcite_propagated":  propagated.get(c.id, 0.0),
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
