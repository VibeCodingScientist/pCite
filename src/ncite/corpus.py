"""
ncite.corpus

Fetch metabolomics papers from PubMed.
Check each for MetaboLights / MassIVE raw data deposits.
The deposit check is the Phase 3 measurement anchor embedded in Phase 1.

Output: data/papers.jsonl
Run:    python -m ncite.corpus
"""

import asyncio, json, re, sys, urllib.parse
from pathlib import Path
import httpx
from ncite.models import Paper
from ncite import config

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
METABO_DOI    = "https://www.ebi.ac.uk/metabolights/ws/studies/doi"
DATA_FILE     = Path("data/papers.jsonl")
REPO_QUERY = (
    '"MetaboLights"[All Fields] OR "MTBLS"[All Fields] '
    'OR "Metabolomics Workbench"[All Fields]'
)
DISEASE_QUERY = (
    '"colorectal neoplasms"[MeSH] AND '
    '("metabolomics"[MeSH] OR "metabolomics"[Title/Abstract])'
)
DEFAULT_QUERIES = [REPO_QUERY, DISEASE_QUERY]

_sem_metabo = asyncio.Semaphore(5)
_BATCH_SIZE = 200  # PubMed efetch max per request


def _ncbi_params(**kwargs) -> dict:
    """Add NCBI API key to params if configured."""
    if config.NCBI_API_KEY:
        kwargs["api_key"] = config.NCBI_API_KEY
    return kwargs


async def _search_pmids(query: str, max_results: int, client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(PUBMED_SEARCH, params=_ncbi_params(
        db="pubmed", term=query, retmax=max_results,
        retmode="json", datetype="pdat", mindate="2015",
    ))
    return resp.json()["esearchresult"]["idlist"]


async def _check_deposit(doi: str, client: httpx.AsyncClient) -> str | None:
    """
    Non-null return = physical instrument data exists in MetaboLights.
    This is the only field that determines ValidationClass.PHYSICAL eligibility.
    """
    async with _sem_metabo:
        try:
            resp = await client.get(
                f"{METABO_DOI}/{urllib.parse.quote(doi)}", timeout=10
            )
            if resp.status_code == 200 and (data := resp.json()):
                return data[0].get("accession")
        except Exception:
            pass
    return None


def _between(text: str, start: str, end: str) -> str:
    try:
        s = text.index(start) + len(start)
        return text[s:text.index(end, s)].strip()
    except ValueError:
        return ""


_ARTICLE_DOI_RE = re.compile(r'<ArticleId IdType="doi">([^<]+)</ArticleId>')
_ARTICLE_SPLIT_RE = re.compile(r'<PubmedArticle>')


def _parse_article(article_xml: str) -> dict | None:
    """Parse a single <PubmedArticle> block. Returns dict with doi/pmid/title/etc or None."""
    # Isolate main article from references
    ref_start = article_xml.find("<ReferenceList>")
    main_xml = article_xml[:ref_start] if ref_start > 0 else article_xml

    pmid = _between(main_xml, "<PMID Version=\"1\">", "</PMID>")
    if not pmid:
        pmid = _between(main_xml, "<PMID>", "</PMID>")

    doi = _between(main_xml, '<ELocationID EIdType="doi"', "</ELocationID>").split(">")[-1]
    if not doi:
        m = _ARTICLE_DOI_RE.search(main_xml)
        doi = m.group(1).strip() if m else ""

    title = _between(main_xml, "<ArticleTitle>", "</ArticleTitle>")
    abst  = _between(main_xml, "<AbstractText>", "</AbstractText>")
    year  = _between(main_xml, "<PubDate><Year>", "</Year>")

    if not doi or not title:
        return None
    return {
        "doi": doi.strip(), "pmid": pmid, "title": title,
        "abstract": abst, "year": int(year) if year.isdigit() else None,
    }


async def _fetch_batch(pmids: list[str], client: httpx.AsyncClient) -> list[dict]:
    """Fetch up to 200 PMIDs in a single efetch call, parse all articles."""
    resp = await client.get(PUBMED_FETCH, params=_ncbi_params(
        db="pubmed", id=",".join(pmids), retmode="xml", rettype="abstract",
    ), timeout=60)
    if resp.status_code != 200:
        print(f"  batch fetch failed: HTTP {resp.status_code}", file=sys.stderr)
        return []
    # Split on <PubmedArticle> boundaries
    parts = _ARTICLE_SPLIT_RE.split(resp.text)
    articles = []
    for part in parts[1:]:  # skip preamble before first <PubmedArticle>
        parsed = _parse_article(part)
        if parsed:
            articles.append(parsed)
    return articles


async def build_corpus(
    queries: list[str] | None = None,
    max_per_query: int | None = None,
) -> int:
    queries = queries or DEFAULT_QUERIES
    max_per_query = max_per_query or config.PUBMED_MAX_PER_QUERY
    DATA_FILE.parent.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        # Run each query and union PMIDs
        all_pmids: set[str] = set()
        for i, q in enumerate(queries, 1):
            pmids = await _search_pmids(q, max_per_query, client)
            new = len(set(pmids) - all_pmids)
            all_pmids.update(pmids)
            print(f"  query {i}/{len(queries)}: {len(pmids)} PMIDs ({new} new)",
                  file=sys.stderr)
            if i < len(queries):
                await asyncio.sleep(0.4)  # polite pause between queries
        pmid_list = sorted(all_pmids)
        print(f"  {len(pmid_list)} unique PMIDs, fetching in batches of {_BATCH_SIZE}...",
              file=sys.stderr)

        # Batch fetch
        all_articles: list[dict] = []
        for i in range(0, len(pmid_list), _BATCH_SIZE):
            batch = pmid_list[i:i + _BATCH_SIZE]
            articles = await _fetch_batch(batch, client)
            all_articles.extend(articles)
            print(f"  batch {i // _BATCH_SIZE + 1}: {len(articles)} papers parsed",
                  file=sys.stderr)
            if i + _BATCH_SIZE < len(pmid_list):
                await asyncio.sleep(0.5)  # polite pause between batches

        # MetaboLights deposit checks (parallel, rate-limited by _sem_metabo)
        deposit_tasks = [
            _check_deposit(a["doi"], client) for a in all_articles
        ]
        deposits = await asyncio.gather(*deposit_tasks)

        papers = []
        for art, metabo_id in zip(all_articles, deposits):
            papers.append(Paper(
                doi=art["doi"], pmid=art["pmid"], title=art["title"],
                abstract=art["abstract"], metabo_id=metabo_id,
                year=art["year"],
            ))

    with DATA_FILE.open("w") as f:
        for p in papers:
            f.write(p.model_dump_json() + "\n")
    deposited = sum(1 for p in papers if p.metabo_id)
    print(f"  {len(papers)} papers, {deposited} with MetaboLights deposits", file=sys.stderr)
    return len(papers)


def load_papers() -> list[Paper]:
    return [Paper.model_validate_json(l) for l in DATA_FILE.read_text().splitlines() if l]


if __name__ == "__main__":
    asyncio.run(build_corpus())
