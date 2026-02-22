"""
ncite.extract

Paper text -> structured Claims via Claude tool_use.
The merge step collapses identical assertions across papers into one node.
After this: len(claim.provenance) == replication count.

Output: data/claims.jsonl (deduplicated)
Run:    python -m ncite.extract
"""

import asyncio, functools, json, sys, urllib.request, urllib.parse
from pathlib import Path
import anthropic
from ncite.models import (
    Claim, Entity, Paper, Predicate, ProvenanceEntry,
    StatisticalQualifiers, ValidationClass,
)
from ncite import config

DATA_OUT = Path("data/claims.jsonl")
_client: anthropic.AsyncAnthropic | None = None
_sem     = asyncio.Semaphore(5)


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=config.require_api_key())
    return _client

_PROMPT = """\
Extract atomic scientific claims from this metabolomics paper.

A claim is one falsifiable assertion: subject -> predicate -> object, with quantitative evidence.
Only extract claims backed by numbers (p-values, fold changes, effect sizes, AUC).

predicate MUST be exactly one of:
  increases | decreases | is_biomarker_for | distinguishes | predicts |
  inhibits | activates | is_metabolite_of | correlates_with | causes | treats

Entity IDs: HMDB (compounds), MESH (diseases), CHEBI (chemicals).
Format: "HMDB:HMDB0000122" or "MESH:D006262". Use "UNKNOWN:name" if unmappable.

DOI: {doi}

{text}"""

_SCHEMA = {
    "name": "submit_claims",
    "description": "Submit extracted scientific claims",
    "input_schema": {
        "type": "object",
        "required": ["claims"],
        "properties": {"claims": {"type": "array", "items": {
            "type": "object",
            "required": ["subject_name","subject_type","subject_id",
                          "predicate","object_name","object_type","object_id"],
            "properties": {
                "subject_name": {"type": "string"},
                "subject_type": {"type": "string"},
                "subject_id":   {"type": "string"},
                "predicate":    {"enum": [p.value for p in Predicate]},
                "object_name":  {"type": "string"},
                "object_type":  {"type": "string"},
                "object_id":    {"type": "string"},
                "n":            {"type": ["integer","null"]},
                "p_value":      {"type": ["number","null"]},
                "effect_size":  {"type": ["number","null"]},
                "fold_change":  {"type": ["number","null"]},
                "method":       {"type": ["string","null"]},
            }
        }}}
    }
}


@functools.lru_cache(maxsize=10_000)
def _hmdb_lookup(name: str) -> str:
    """HMDB compound lookup. Cached — same compound appears in hundreds of papers."""
    try:
        url = f"https://hmdb.ca/metabolites/search.json?query={urllib.parse.quote(name)}"
        with urllib.request.urlopen(url, timeout=5) as r:
            if hits := json.loads(r.read()).get("metabolites"):
                return f"HMDB:{hits[0]['accession']}"
    except Exception:
        pass
    return f"UNKNOWN:{name.lower().strip()}"


async def _extract(paper: Paper) -> list[Claim]:
    """One Claude call per paper. tool_use guarantees valid JSON — no output parsing."""
    text = paper.title + "\n\n" + paper.abstract
    if paper.full_text:
        text += "\n\n" + paper.full_text[:3000]
    async with _sem:
        try:
            resp = await _get_client().messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                tools=[_SCHEMA],
                tool_choice={"type": "tool", "name": "submit_claims"},
                messages=[{"role": "user",
                            "content": _PROMPT.format(doi=paper.doi, text=text)}],
            )
            raw = resp.content[0].input["claims"]
        except Exception as e:
            print(f"  skip {paper.doi}: {e}", file=sys.stderr)
            return []

    claims = []
    for c in raw:
        try:
            subj_id = c["subject_id"] if not c["subject_id"].startswith("UNKNOWN") \
                      else _hmdb_lookup(c["subject_name"])
            claims.append(Claim(
                subject    = Entity(id=subj_id, name=c["subject_name"], type=c["subject_type"]),
                predicate  = Predicate(c["predicate"]),
                object     = Entity(id=c["object_id"], name=c["object_name"], type=c["object_type"]),
                qualifiers = StatisticalQualifiers(
                    n=c.get("n"), p_value=c.get("p_value"),
                    effect_size=c.get("effect_size"), fold_change=c.get("fold_change"),
                    method=c.get("method"),
                ),
                provenance = [ProvenanceEntry(
                    doi=paper.doi, metabo_id=paper.metabo_id,
                    validation_class=ValidationClass.AI_GENERATED,
                )],
            ))
        except Exception as e:
            print(f"  bad claim in {paper.doi}: {e}", file=sys.stderr)
    return claims


def _merge_all(claims: list[Claim]) -> list[Claim]:
    """
    Deduplicate by claim ID, merging provenance.
    After this call, len(claim.provenance) is the true replication count.
    This is the moment replication becomes a structural property.
    """
    merged: dict[str, Claim] = {}
    for c in claims:
        merged[c.id] = merged[c.id].merge(c) if c.id in merged else c
    return list(merged.values())


async def process_corpus() -> int:
    from ncite.corpus import load_papers
    papers, all_claims = load_papers(), []
    for i in range(0, len(papers), 20):
        results = await asyncio.gather(*[_extract(p) for p in papers[i:i+20]])
        for claims in results:
            all_claims.extend(claims)
        print(f"  {len(all_claims)} claims extracted...", file=sys.stderr)
    merged = _merge_all(all_claims)
    DATA_OUT.parent.mkdir(exist_ok=True)
    with DATA_OUT.open("w") as f:
        for c in merged:
            f.write(c.model_dump_json() + "\n")
    replicated = sum(1 for c in merged if c.replication_count > 1)
    print(f"  {len(merged)} unique claims, {replicated} seen in >1 paper", file=sys.stderr)
    return len(merged)


def load_claims() -> list[Claim]:
    return [Claim.model_validate_json(l) for l in DATA_OUT.read_text().splitlines() if l]


if __name__ == "__main__":
    asyncio.run(process_corpus())
