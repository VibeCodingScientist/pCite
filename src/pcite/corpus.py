"""
pcite.corpus

MetaboLights-first corpus construction.
Every study in MetaboLights has deposited instrument data, so Physical-tier
classification is a structural property of corpus construction — not an inference.

Pipeline:
  1. Fetch all public MetaboLights study accessions
  2. Fetch publications (DOI + PMID) for each study
  3. Fetch paper abstracts from PubMed by PMID
  4. Add disease-focused PubMed papers for citation density

Output: data/papers.jsonl
Run:    python -m pcite.corpus
"""

import asyncio, json, re, sys, urllib.parse
from pathlib import Path
import httpx
from pcite.models import Paper
from pcite import config

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
METABO_STUDIES = "https://www.ebi.ac.uk/metabolights/ws/studies"
METABO_PUBS   = "https://www.ebi.ac.uk/metabolights/ws/studies/{accession}/publications"
DATA_FILE     = Path("data/papers.jsonl")

DISEASE_QUERY = (
    '"colorectal neoplasms"[MeSH] AND '
    '("metabolomics"[MeSH] OR "metabolomics"[Title/Abstract])'
)

_sem_metabo = asyncio.Semaphore(10)
_sem_pubmed = asyncio.Semaphore(5)
_BATCH_SIZE = 200  # PubMed efetch max per request


def _ncbi_params(**kwargs) -> dict:
    """Add NCBI API key to params if configured."""
    if config.NCBI_API_KEY:
        kwargs["api_key"] = config.NCBI_API_KEY
    return kwargs


# ── MetaboLights layer ───────────────────────────────────────────────

async def _fetch_all_accessions(client: httpx.AsyncClient) -> list[str]:
    """Get all public MetaboLights study accession IDs."""
    resp = await client.get(METABO_STUDIES, timeout=30)
    return resp.json()["content"]


async def _fetch_study_publications(
    accession: str, client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch publications for one study. Returns list of {accession, doi, pmid}."""
    async with _sem_metabo:
        try:
            resp = await client.get(
                METABO_PUBS.format(accession=accession), timeout=15,
            )
            if resp.status_code != 200:
                return []
            pubs = resp.json().get("publications", [])
            return [
                {"accession": accession, "doi": p["doi"], "pmid": p.get("pubMedID", "")}
                for p in pubs if p.get("doi")
            ]
        except Exception:
            return []


# ── PubMed layer ─────────────────────────────────────────────────────

async def _search_pmids(query: str, max_results: int, client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(PUBMED_SEARCH, params=_ncbi_params(
        db="pubmed", term=query, retmax=max_results,
        retmode="json", datetype="pdat", mindate="2015",
    ))
    return resp.json()["esearchresult"]["idlist"]


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


# ── Corpus builder ───────────────────────────────────────────────────

async def build_corpus(
    disease_query: str | None = DISEASE_QUERY,
    max_disease_papers: int | None = None,
) -> int:
    max_disease_papers = max_disease_papers or config.PUBMED_MAX_PER_QUERY
    DATA_FILE.parent.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:

        # ── Phase A: MetaboLights-first (all Physical-tier) ──────────
        print("  Phase A: MetaboLights studies...", file=sys.stderr)
        accessions = await _fetch_all_accessions(client)
        print(f"  {len(accessions)} public studies found", file=sys.stderr)

        # Fetch publications for every study (async, rate-limited)
        pub_tasks = [_fetch_study_publications(acc, client) for acc in accessions]
        pub_results = await asyncio.gather(*pub_tasks)
        # Flatten: {doi -> accession}, keep first accession per DOI
        doi_to_accession: dict[str, str] = {}
        pmid_from_metabo: dict[str, str] = {}  # doi -> pmid (if MetaboLights has it)
        for pubs in pub_results:
            for p in pubs:
                doi = p["doi"]
                if doi not in doi_to_accession:
                    doi_to_accession[doi] = p["accession"]
                    if p["pmid"]:
                        pmid_from_metabo[doi] = p["pmid"]
        print(f"  {len(doi_to_accession)} unique DOIs from MetaboLights "
              f"({len(pmid_from_metabo)} with PMIDs)", file=sys.stderr)

        # ── Phase B: disease-focused PubMed papers (citation density) ─
        disease_pmids: set[str] = set()
        if disease_query:
            print("  Phase B: disease-focused PubMed query...", file=sys.stderr)
            disease_pmids = set(
                await _search_pmids(disease_query, max_disease_papers, client)
            )
            print(f"  {len(disease_pmids)} disease-focused PMIDs", file=sys.stderr)

        # ── Combine PMIDs and batch fetch ────────────────────────────
        all_pmids = set(pmid_from_metabo.values()) | disease_pmids
        pmid_list = sorted(all_pmids)
        print(f"  {len(pmid_list)} unique PMIDs to fetch from PubMed...",
              file=sys.stderr)

        all_articles: list[dict] = []
        for i in range(0, len(pmid_list), _BATCH_SIZE):
            batch = pmid_list[i:i + _BATCH_SIZE]
            articles = await _fetch_batch(batch, client)
            all_articles.extend(articles)
            print(f"  batch {i // _BATCH_SIZE + 1}: {len(articles)} papers parsed",
                  file=sys.stderr)
            if i + _BATCH_SIZE < len(pmid_list):
                await asyncio.sleep(0.5)

        # ── Build Paper objects ──────────────────────────────────────
        papers = []
        for art in all_articles:
            metabo_id = doi_to_accession.get(art["doi"])
            papers.append(Paper(
                doi=art["doi"], pmid=art["pmid"], title=art["title"],
                abstract=art["abstract"], metabo_id=metabo_id,
                year=art["year"],
            ))

    with DATA_FILE.open("w") as f:
        for p in papers:
            f.write(p.model_dump_json() + "\n")
    deposited = sum(1 for p in papers if p.metabo_id)
    print(f"  {len(papers)} papers, {deposited} with MetaboLights deposits",
          file=sys.stderr)
    return len(papers)


def load_papers() -> list[Paper]:
    return [Paper.model_validate_json(l) for l in DATA_FILE.read_text().splitlines() if l]


if __name__ == "__main__":
    asyncio.run(build_corpus())
