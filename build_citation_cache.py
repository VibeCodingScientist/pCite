"""Run just the OpenAlex citation lookup and save citation_cache.json.

No LLM API keys needed — only hits OpenAlex.
Uses a fresh mailto to get polite pool budget.

Usage: python build_citation_cache.py
"""
import asyncio, json, sys
from pathlib import Path
import httpx

OPENALEX = "https://api.openalex.org/works"
MAILTO   = "pcite.framework@gmail.com"
CACHE_OUT = Path("data/citation_cache.json")
sem = asyncio.Semaphore(3)


async def resolve_openalex_ids(
    dois: list[str], client: httpx.AsyncClient,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    batch_size = 50
    for i in range(0, len(dois), batch_size):
        batch = dois[i : i + batch_size]
        pipe_filter = "|".join(f"https://doi.org/{d}" for d in batch)
        async with sem:
            try:
                resp = await client.get(OPENALEX, params={
                    "filter": f"doi:{pipe_filter}",
                    "select": "id,doi",
                    "per-page": batch_size,
                    "mailto": MAILTO,
                }, timeout=30)
                for w in resp.json().get("results", []):
                    if w.get("doi") and w.get("id"):
                        bare = w["doi"].replace("https://doi.org/", "")
                        mapping[bare] = w["id"]
            except Exception as e:
                print(f"  resolve batch error: {e}", file=sys.stderr)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  resolve: {min(i + batch_size, len(dois))}/{len(dois)}",
                  file=sys.stderr)
    print(f"  resolved {len(mapping)}/{len(dois)} DOIs to OpenAlex IDs",
          file=sys.stderr)
    return mapping


async def citing_dois(openalex_id: str, client: httpx.AsyncClient) -> list[str]:
    async with sem:
        try:
            resp = await client.get(OPENALEX, params={
                "filter": f"cites:{openalex_id}", "select": "doi",
                "per-page": 50, "mailto": MAILTO,
            }, timeout=30)
            return [
                w["doi"].replace("https://doi.org/", "")
                for w in resp.json().get("results", []) if w.get("doi")
            ]
        except Exception:
            return []


async def main():
    from pcite.extract import load_claims
    from pcite.models import Claim

    claims = load_claims()
    by_doi: dict[str, list[Claim]] = {}
    for c in claims:
        for p in c.provenance:
            by_doi.setdefault(p.doi, []).append(c)

    dois = list(by_doi.keys())
    print(f"  {len(dois)} unique DOIs from {len(claims)} claims", file=sys.stderr)

    timeout = httpx.Timeout(30.0, connect=10.0)
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=3)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                 limits=limits) as client:
        oa_ids = await resolve_openalex_ids(dois, client)
        resolved = [(doi, oa_ids[doi]) for doi in dois if doi in oa_ids]

        citing_map: dict[str, list[str]] = {}
        batch_size = 50
        for i in range(0, len(resolved), batch_size):
            batch = resolved[i : i + batch_size]
            results = await asyncio.gather(*[
                citing_dois(oa_id, client) for _, oa_id in batch
            ])
            for (doi, _), result in zip(batch, results):
                citing_map[doi] = result
            print(f"  citations: {min(i + batch_size, len(resolved))}"
                  f"/{len(resolved)} DOIs queried", file=sys.stderr)

    CACHE_OUT.write_text(json.dumps(citing_map))
    total_citations = sum(len(v) for v in citing_map.values())
    print(f"\n  Saved {len(citing_map)} DOIs, {total_citations} total citations"
          f" -> {CACHE_OUT}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
