"""
pride_corpus.py

PRIDE-first corpus construction for cancer proteomics.
Every COMPLETE submission in PRIDE has deposited instrument data, so Physical-tier
classification is a structural property of corpus construction — not an inference.

Pipeline:
  1. Search PRIDE API for cancer proteomics projects
  2. Filter: submissionType == COMPLETE, organism contains Homo sapiens
  3. Parse PubMed IDs from PRIDE reference strings
  4. Fetch paper metadata from PubMed (reuses corpus._fetch_batch + _parse_article)
  5. Add disease-focused PubMed query for citation density
  6. Compute deposit quality per project (1.0-10.0)

Output: data/pride/papers.jsonl + data/pride/deposit_quality.json
Run:    python pride_corpus.py
"""

import asyncio, json, math, re, sys
from pathlib import Path
import httpx
from pcite.models import Paper
from pcite.corpus import _fetch_batch, _parse_article, _search_pmids, _ncbi_params
from pcite import config

PRIDE_SEARCH = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects"
DATA_DIR = Path("data/pride")
PAPERS_FILE = DATA_DIR / "papers.jsonl"
QUALITY_FILE = DATA_DIR / "deposit_quality.json"

SEARCH_TERMS = [
    "cancer proteomics",
    "tumor proteomics",
    "carcinoma proteomics",
    "colorectal cancer",
    "breast cancer proteomics",
    "lung cancer proteomics",
    "hepatocellular carcinoma proteomics",
    "ovarian cancer proteomics",
    "prostate cancer proteomics",
    "pancreatic cancer proteomics",
]

# Broader terms for deposit-first mode — maximizes PRIDE project coverage
DEPOSIT_FIRST_TERMS = SEARCH_TERMS + [
    "proteomics mass spectrometry",
    "quantitative proteomics",
    "phosphoproteomics",
    "proteome profiling",
    "TMT proteomics",
    "SILAC proteomics",
    "DIA proteomics",
]

DISEASE_QUERY = (
    '"colorectal neoplasms"[MeSH] AND '
    '"proteomics"[Title/Abstract]'
)

HR_TERMS = {"orbitrap", "q exactive", "lumos", "exploris", "tof", "timstof"}

_PUBMED_RE = re.compile(r"pubMed:(\d+)")


def _parse_pride_pmids(references: list[str]) -> list[str]:
    """Extract PubMed IDs from PRIDE reference strings.

    PRIDE reference format uses --pubMed:12345-- or pubMed:12345.
    Filters out pubMed:0 (no PMID available).
    """
    pmids = []
    for ref in references:
        for m in _PUBMED_RE.finditer(ref):
            pmid = m.group(1)
            if pmid != "0":
                pmids.append(pmid)
    return pmids


async def _search_pride(
    keyword: str, client: httpx.AsyncClient, page_size: int = 100,
) -> list[dict]:
    """Search PRIDE API for projects matching a keyword.

    PRIDE v2 search returns a raw JSON list of project objects.
    Pagination: keep fetching until len(batch) < page_size.
    """
    projects = []
    page = 0
    while True:
        try:
            resp = await client.get(PRIDE_SEARCH, params={
                "keyword": keyword,
                "pageSize": page_size,
                "page": page,
            }, timeout=30)
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            projects.extend(batch)
            page += 1
            if len(batch) < page_size:
                break
        except Exception as e:
            print(f"  PRIDE search '{keyword}' page {page}: {e}", file=sys.stderr)
            break
    return projects


def _is_human_complete(project: dict) -> bool:
    """Filter: COMPLETE submission, Homo sapiens.

    In PRIDE v2 search results, organisms is list[str] (e.g. ["Homo sapiens (human)"]).
    """
    if project.get("submissionType") != "COMPLETE":
        return False
    organisms = project.get("organisms", [])
    return any(
        "homo sapiens" in (o.lower() if isinstance(o, str) else str(o).lower())
        for o in organisms
    )


def compute_deposit_quality(projects: dict[str, dict]) -> dict[str, float]:
    """Compute deposit quality score per project, normalized to 1.0-10.0.

    Formula:
      raw = log1p(n_files) * is_complete + (1.0 if high-res instrument else 0.0)
      weight = 1.0 + 9.0 * (raw - min) / (max - min + 1e-9)
    """
    raw_scores: dict[str, float] = {}
    for accession, proj in projects.items():
        file_names = proj.get("projectFileNames") or []
        n_files = len(file_names)
        is_complete = 1.0 if proj.get("submissionType") == "COMPLETE" else 0.5
        instruments = proj.get("instruments") or []
        has_hr = any(
            any(t in (inst.lower() if isinstance(inst, str) else str(inst).lower())
                for t in HR_TERMS)
            for inst in instruments
        )
        raw = math.log1p(n_files) * is_complete + (1.0 if has_hr else 0.0)
        raw_scores[accession] = raw

    if not raw_scores:
        return {}
    min_raw = min(raw_scores.values())
    max_raw = max(raw_scores.values())
    return {
        acc: round(1.0 + 9.0 * (raw - min_raw) / (max_raw - min_raw + 1e-9), 3)
        for acc, raw in raw_scores.items()
    }


async def build_corpus(max_disease_papers: int | None = None) -> int:
    max_disease_papers = max_disease_papers or config.PUBMED_MAX_PER_QUERY
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:

        # Phase A: Search PRIDE for cancer proteomics projects
        print("  Phase A: PRIDE project search...", file=sys.stderr)
        all_projects: dict[str, dict] = {}  # accession -> project
        for term in SEARCH_TERMS:
            projects = await _search_pride(term, client)
            for proj in projects:
                acc = proj.get("accession", "")
                if acc and acc not in all_projects and _is_human_complete(proj):
                    all_projects[acc] = proj
            print(f"  '{term}': {len(projects)} results, "
                  f"{len(all_projects)} unique COMPLETE/human total",
                  file=sys.stderr)
            await asyncio.sleep(0.3)

        print(f"  {len(all_projects)} total PRIDE projects after dedup",
              file=sys.stderr)

        # Phase B: Parse PubMed IDs from PRIDE references
        print("  Phase B: Parsing PubMed IDs from references...", file=sys.stderr)
        pmid_to_accession: dict[str, str] = {}  # pmid -> first accession
        for acc, proj in all_projects.items():
            refs = proj.get("references", [])
            for pmid in _parse_pride_pmids(refs):
                if pmid not in pmid_to_accession:
                    pmid_to_accession[pmid] = acc

        print(f"  {len(pmid_to_accession)} unique PMIDs from PRIDE references",
              file=sys.stderr)

        # Phase C: Disease-focused PubMed query for citation density
        disease_pmids: set[str] = set()
        print("  Phase C: disease-focused PubMed query...", file=sys.stderr)
        disease_pmids = set(
            await _search_pmids(DISEASE_QUERY, max_disease_papers, client)
        )
        print(f"  {len(disease_pmids)} disease-focused PMIDs", file=sys.stderr)

        # Combine and batch-fetch
        all_pmids = set(pmid_to_accession.keys()) | disease_pmids
        pmid_list = sorted(all_pmids)
        print(f"  {len(pmid_list)} unique PMIDs to fetch from PubMed...",
              file=sys.stderr)

        all_articles: list[dict] = []
        batch_size = 200
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i + batch_size]
            articles = await _fetch_batch(batch, client)
            all_articles.extend(articles)
            print(f"  batch {i // batch_size + 1}: {len(articles)} papers parsed",
                  file=sys.stderr)
            if i + batch_size < len(pmid_list):
                await asyncio.sleep(0.5)

        # Build Paper objects — deposit_id links to PRIDE accession
        papers = []
        for art in all_articles:
            pmid = art.get("pmid", "")
            deposit_id = pmid_to_accession.get(pmid)
            papers.append(Paper(
                doi=art["doi"], pmid=pmid, title=art["title"],
                abstract=art["abstract"], deposit_id=deposit_id,
                year=art["year"],
            ))

    # Write papers
    with PAPERS_FILE.open("w") as f:
        for p in papers:
            f.write(p.model_dump_json() + "\n")

    # Compute and write deposit quality
    quality = compute_deposit_quality(all_projects)
    QUALITY_FILE.write_text(json.dumps(quality, indent=2))

    deposited = sum(1 for p in papers if p.deposit_id)
    print(f"  {len(papers)} papers, {deposited} with PRIDE deposits",
          file=sys.stderr)
    if quality:
        print(f"  {len(quality)} projects with quality scores "
              f"(range {min(quality.values()):.1f}-{max(quality.values()):.1f})",
              file=sys.stderr)
    else:
        print("  0 projects with quality scores", file=sys.stderr)
    return len(papers)


async def build_deposit_first_corpus() -> int:
    """Deposit-first corpus: every paper must have a PRIDE accession.

    Differences from build_corpus():
      - No Phase C (disease-focused PubMed supplement)
      - Broader PRIDE search terms (DEPOSIT_FIRST_TERMS)
      - Only papers with deposit_id are kept
      - Fallback PubMed search by project title for projects with 0 PMIDs
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:

        # Phase A: Search PRIDE with broader terms
        print("  Phase A: PRIDE project search (deposit-first)...", file=sys.stderr)
        all_projects: dict[str, dict] = {}
        for term in DEPOSIT_FIRST_TERMS:
            projects = await _search_pride(term, client)
            for proj in projects:
                acc = proj.get("accession", "")
                if acc and acc not in all_projects and _is_human_complete(proj):
                    all_projects[acc] = proj
            print(f"  '{term}': {len(projects)} results, "
                  f"{len(all_projects)} unique COMPLETE/human total",
                  file=sys.stderr)
            await asyncio.sleep(0.3)

        print(f"  {len(all_projects)} total PRIDE projects after dedup",
              file=sys.stderr)

        # Phase B: Parse PubMed IDs from PRIDE references
        print("  Phase B: Parsing PubMed IDs from references...", file=sys.stderr)
        pmid_to_accession: dict[str, str] = {}
        projects_with_pmids: set[str] = set()
        for acc, proj in all_projects.items():
            refs = proj.get("references", [])
            pmids = _parse_pride_pmids(refs)
            for pmid in pmids:
                if pmid not in pmid_to_accession:
                    pmid_to_accession[pmid] = acc
                projects_with_pmids.add(acc)

        print(f"  {len(pmid_to_accession)} unique PMIDs from PRIDE references",
              file=sys.stderr)

        # Phase B2: Fallback — search PubMed by project title for projects with 0 PMIDs
        projects_without = [
            acc for acc in all_projects if acc not in projects_with_pmids
        ]
        if projects_without:
            print(f"  Phase B2: PubMed title search for {len(projects_without)} "
                  f"projects without PMIDs...", file=sys.stderr)
            found = 0
            for acc in projects_without:
                title = all_projects[acc].get("title", "")
                if not title:
                    continue
                # Search PubMed with project title
                query = f'"{title}"[Title]'
                try:
                    pmids = await _search_pmids(query, 5, client)
                    for pmid in pmids:
                        if pmid not in pmid_to_accession:
                            pmid_to_accession[pmid] = acc
                            found += 1
                except Exception as e:
                    print(f"    {acc} title search failed: {e}", file=sys.stderr)
                await asyncio.sleep(0.4)
            print(f"  Phase B2: found {found} additional PMIDs", file=sys.stderr)

        # No Phase C — deposit-first mode skips disease supplement

        # Batch-fetch from PubMed — only deposit-linked PMIDs
        pmid_list = sorted(pmid_to_accession.keys())
        print(f"  {len(pmid_list)} deposit-linked PMIDs to fetch from PubMed...",
              file=sys.stderr)

        all_articles: list[dict] = []
        batch_size = 200
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i + batch_size]
            articles = await _fetch_batch(batch, client)
            all_articles.extend(articles)
            print(f"  batch {i // batch_size + 1}: {len(articles)} papers parsed",
                  file=sys.stderr)
            if i + batch_size < len(pmid_list):
                await asyncio.sleep(0.5)

        # Build Paper objects — all have deposit_id by construction
        papers = []
        for art in all_articles:
            pmid = art.get("pmid", "")
            deposit_id = pmid_to_accession.get(pmid)
            if not deposit_id:
                continue  # safety — should not happen
            papers.append(Paper(
                doi=art["doi"], pmid=pmid, title=art["title"],
                abstract=art["abstract"], deposit_id=deposit_id,
                year=art["year"],
            ))

    # Write papers
    with PAPERS_FILE.open("w") as f:
        for p in papers:
            f.write(p.model_dump_json() + "\n")

    # Compute and write deposit quality
    quality = compute_deposit_quality(all_projects)
    QUALITY_FILE.write_text(json.dumps(quality, indent=2))

    deposited = sum(1 for p in papers if p.deposit_id)
    print(f"  {len(papers)} papers, {deposited} with PRIDE deposits "
          f"({deposited / len(papers) * 100:.0f}% coverage)" if papers
          else "  0 papers", file=sys.stderr)
    if quality:
        print(f"  {len(quality)} projects with quality scores "
              f"(range {min(quality.values()):.1f}-{max(quality.values()):.1f})",
              file=sys.stderr)
    return len(papers)


def load_papers() -> list[Paper]:
    return [Paper.model_validate_json(l) for l in PAPERS_FILE.read_text().splitlines() if l]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--deposit-first", action="store_true",
                   help="Build deposit-first corpus (no disease supplement)")
    args = p.parse_args()
    if args.deposit_first:
        asyncio.run(build_deposit_first_corpus())
    else:
        asyncio.run(build_corpus())
