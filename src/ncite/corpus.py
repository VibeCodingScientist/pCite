"""
ncite.corpus

Fetch metabolomics papers from PubMed.
Check each for MetaboLights / MassIVE raw data deposits.
The deposit check is the Phase 3 measurement anchor embedded in Phase 1.

Output: data/papers.jsonl
Run:    python -m ncite.corpus
"""

import asyncio, json, sys, urllib.parse
from pathlib import Path
import httpx
from ncite.models import Paper

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
METABO_DOI    = "https://www.ebi.ac.uk/metabolights/ws/studies/doi"
DATA_FILE     = Path("data/papers.jsonl")
DEFAULT_QUERY = '("metabolomics"[MeSH] OR "metabolomics"[Title]) AND "biomarker"[Title/Abstract]'

_sem = asyncio.Semaphore(3)   # PubMed: 3 req/s without API key


async def _search_pmids(query: str, max_results: int, client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(PUBMED_SEARCH, params={
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "json", "datetype": "pdat", "mindate": "2015",
    })
    return resp.json()["esearchresult"]["idlist"]


async def _check_deposit(doi: str, client: httpx.AsyncClient) -> str | None:
    """
    Non-null return = physical instrument data exists in MetaboLights.
    This is the only field that determines ValidationClass.PHYSICAL eligibility.
    """
    async with _sem:
        try:
            resp = await client.get(
                f"{METABO_DOI}/{urllib.parse.quote(doi)}", timeout=10
            )
            if resp.status_code == 200 and (data := resp.json()):
                return data[0].get("accession")
        except Exception:
            pass
    return None


async def _fetch_paper(pmid: str, client: httpx.AsyncClient) -> Paper | None:
    async with _sem:
        try:
            resp = await client.get(PUBMED_FETCH, params={
                "db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"
            })
            xml   = resp.text
            doi   = _between(xml, '<ELocationID EIdType="doi"', "</ELocationID>").split(">")[-1]
            title = _between(xml, "<ArticleTitle>", "</ArticleTitle>")
            abst  = _between(xml, "<AbstractText>", "</AbstractText>")
            year  = _between(xml, "<PubDate><Year>", "</Year>")
            if not doi or not title:
                return None
            return Paper(
                doi=doi.strip(), pmid=pmid, title=title, abstract=abst,
                metabo_id=await _check_deposit(doi.strip(), client),
                year=int(year) if year.isdigit() else None,
            )
        except Exception:
            return None


def _between(text: str, start: str, end: str) -> str:
    try:
        s = text.index(start) + len(start)
        return text[s:text.index(end, s)].strip()
    except ValueError:
        return ""


async def build_corpus(query: str = DEFAULT_QUERY, max_results: int = 2000) -> int:
    DATA_FILE.parent.mkdir(exist_ok=True)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        pmids  = await _search_pmids(query, max_results, client)
        papers = await asyncio.gather(*[_fetch_paper(p, client) for p in pmids])
    valid = [p for p in papers if p]
    with DATA_FILE.open("w") as f:
        for p in valid:
            f.write(p.model_dump_json() + "\n")
    deposited = sum(1 for p in valid if p.metabo_id)
    print(f"  {len(valid)} papers, {deposited} with MetaboLights deposits", file=sys.stderr)
    return len(valid)


def load_papers() -> list[Paper]:
    return [Paper.model_validate_json(l) for l in DATA_FILE.read_text().splitlines() if l]


if __name__ == "__main__":
    asyncio.run(build_corpus())
