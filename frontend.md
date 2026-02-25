# pCite Frontend — Requirements Specification

**Scope:** Static site. No backend. No authentication. No database.
**Stack:** Python generate script → static HTML/CSS/JS → GitHub Pages
**Goal:** One sentence in the paper: *"All claims available at pcite.org"*
**Design reference:** OpenAlex work pages. Information density, zero marketing.

This is a scientific instrument, not a product. No gradients. No hero images.
No animations. No modals. Dense, readable, fast.

---

## Guiding Principle

Every design decision answers one question: does this help a researcher or reviewer
understand a claim faster? If not, remove it.

The person landing on a claim page is either:
1. A researcher who found this claim cited in a paper and wants to evaluate it
2. A reviewer checking whether the system's contribution is real
3. A journalist or funder who was sent the link

All three need the same thing: what the claim is, how well-supported it is,
and where the evidence comes from. In that order. In under ten seconds.

---

## Page 1 — Claim Detail

URL: `pcite.org/claim/{claim_id}`

```
┌─────────────────────────────────────────────────────────────────────────┐
│  pCite                                              [Search claims...]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CLAIM                                                                  │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Glutamine   ──increases──►   Colitis, Ulcerative                      │
│                                                                         │
│  HMDB:HMDB0000122              MESH:D015212                             │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VALIDATION                                                             │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Class       Physical Measurement          ████████████████░░░░  10.0  │
│  Replicated  8 independent papers                                       │
│  Base weight 31.7                                                       │
│  pCite score 47.6          Traditional citations  312                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EVIDENCE CHAIN                                    8 sources            │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  ● Physical    10.1038/s41591-021-01234-5    MTBLS456 ↗    2021       │
│  ● Physical    10.1016/j.cell.2020.09.012    MTBLS789 ↗    2020       │
│  ● Physical    10.1021/acs.jproteome.1c00234 MTBLS901 ↗    2021       │
│  ● Replicated  10.1038/nbt.4341                             2019       │
│  ● Replicated  10.1074/jbc.M115.674481                      2018       │
│  ● Curated     10.1093/bioinformatics/bty879                2018       │
│  ● Curated     10.1371/journal.pone.0198274                 2018       │
│  ● Curated     10.1002/jms.4345                             2017       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CITATIONS RECEIVED                                16 total             │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Replicates    ███░░░░░░░░░░░░░░░░░░   3                               │
│  Supports      ████████████░░░░░░░░░  12                               │
│  Contradicts   █░░░░░░░░░░░░░░░░░░░░   1                               │
│  Extends       ░░░░░░░░░░░░░░░░░░░░░   0                               │
│  Applies       ░░░░░░░░░░░░░░░░░░░░░   0                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IDENTIFIERS                                                            │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Claim ID     a3f8c2d1b4e9f7a2                                         │
│  Nanopub URI  https://w3id.org/np/RAbc123XyZ456... ↗                  │
│  Subject URI  https://identifiers.org/HMDB:HMDB0000122 ↗              │
│  Object URI   https://identifiers.org/MESH:D015212 ↗                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- The claim triple (subject → predicate → object) is the largest element on the page.
  A reviewer should understand what the claim is without reading anything else.
- ValidationClass bar fills proportionally against max weight (10.0).
  Visual hierarchy: Physical is full bar. AI_GENERATED is nearly empty.
- Evidence chain is sorted best-to-worst validation class, then by year descending.
  MetaboLights link only shown when metabo_id exists.
- Citations received bar chart is proportional within the set, not absolute.
- Identifiers section is collapsed by default on mobile, expanded on desktop.

---

## Page 2 — Search

URL: `pcite.org`  (this is the homepage)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  pCite                                                                  │
│  Validation-weighted scientific claims                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Search claims — compound, disease, or relationship               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Filter  [All predicates ▾]  [All classes ▾]  [Min replication: 1  ▾] │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  14,847 claims  ·  312 Physical  ·  891 Replicated  ·  from 2,041 papers│
│                                                                         │
│  Sorted by pCite score ▾                                               │
│                                                                         │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  ████  Glutamine  increases  Colitis, Ulcerative                       │
│        Physical · 8 papers · pCite 47.6                    claim ↗    │
│                                                                         │
│  ████  Tryptophan  decreases  Crohn Disease                            │
│        Physical · 6 papers · pCite 38.2                    claim ↗    │
│                                                                         │
│  ███░  Butyrate  activates  GPR109A                                    │
│        Replicated · 5 papers · pCite 22.1                  claim ↗    │
│                                                                         │
│  ███░  Short-chain fatty acids  is_biomarker_for  IBD                  │
│        Replicated · 4 papers · pCite 19.7                  claim ↗    │
│                                                                         │
│  ██░░  Kynurenine  increases  Colorectal Cancer                        │
│        Curated · 3 papers · pCite 4.2                      claim ↗    │
│                                                                         │
│  █░░░  Histidine  decreases  Type 2 Diabetes                          │
│        AI Generated · 1 paper · pCite 0.01                 claim ↗    │
│                                                                         │
│  [Load more]                                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- Search is client-side over a pre-built JSON index. No server roundtrip.
  15,000 claims at ~200 bytes each = ~3MB. Acceptable for a research tool.
- The colour bar left of each result is proportional to pCite score within the set.
  Physical claims visually dominate. AI_GENERATED claims look thin. This is intentional.
- Default sort is pCite score descending. The point of the system is visible immediately.
- Filters are additive. All client-side.
- "Load more" reveals 50 results at a time. No pagination needed for 15,000 claims
  when search narrows the set immediately.

---

## Page 3 — Corpus Stats

URL: `pcite.org/corpus`

```
┌─────────────────────────────────────────────────────────────────────────┐
│  pCite  /  Corpus                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  METABOLOMICS KNOWLEDGE GRAPH  ·  February 2026                        │
│                                                                         │
│  Papers          2,041                                                  │
│  Claims          14,847  (unique assertions, deduplicated)              │
│  Nanopublications 14,847  w3id.org/np/ ↗                              │
│  Physical claims    312  (MetaboLights deposit verified)               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EXPERIMENT RESULTS                                                     │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Hypothesis: pCite surfaces physically-validated claims better          │
│  than traditional citation count.                                       │
│                                                                         │
│  Mann-Whitney U     p = 0.0003   ✓ significant                         │
│  Precision@50       pCite 0.41  vs  Traditional 0.09  (4.6× lift)     │
│  NDCG@50            pCite 0.71  vs  Traditional 0.31                   │
│                                                                         │
│  Hypothesis holds.                                                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VALIDATION CLASS DISTRIBUTION                                          │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Physical       ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    312   2.1%       │
│  Replicated     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░    891   6.0%       │
│  Curated        ████████████░░░░░░░░░░░░░░░░░░░░░░  3,847  25.9%      │
│  AI Generated   ████████████████████████████████░░  9,797  66.0%      │
│                                                                         │
│  The distribution is the argument. 66% of extracted claims have         │
│  no physical anchor. Under traditional citation metrics, these          │
│  claims are indistinguishable from the 2.1% that do.                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CODE & DATA                                                            │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                         │
│  GitHub    github.com/pcite-org/pcite ↗                                │
│  arXiv     arxiv.org/abs/2026.XXXXX ↗                                  │
│  Data      CC0 — download claims.jsonl ↗                               │
│  Nanopubs  nanopub.org network ↗                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Notes:**
- The paragraph below the distribution chart is editorial. It belongs here.
  The corpus stats page is where the paper's argument is visible as data.
- Numbers are placeholders. `generate.py` writes them from `results.json` at build time.
- All links in the Code & Data section are real. arXiv link is added after submission.

---

## Technical Specification

### generate.py

```
Input:   data/scores.jsonl
         data/claims.jsonl
         data/results.json
Output:  docs/index.html          ← search page
         docs/corpus.html         ← stats page
         docs/claim/{id}.html     ← one per claim (14,847 files)
         docs/data/claims.json    ← pre-built search index
         docs/css/style.css
```

Single script. Reads all data files. Writes all output files. No templating engine
dependency — Python f-strings are sufficient. Target: under 150 lines.

Run: `python -m pcite.generate`
Deploy: `git push` triggers GitHub Pages build. Zero infrastructure cost.

---

### CSS Requirements

```css
/* The entire design lives in these decisions */

font-family: "Georgia", serif;        /* not a tech product */
max-width: 860px;                     /* comfortable reading width */
color: #111;                          /* near-black, not pure black */
background: #fff;
line-height: 1.6;

/* No framework. No Tailwind. No Bootstrap.        */
/* ~100 lines of CSS covers everything needed.     */
/* Every class has one job.                        */
```

### JavaScript Requirements

Search index: pre-built JSON array of claim objects, loaded once on page load.
Filter + search: vanilla JS, no libraries.
The bar charts on the corpus page: inline SVG generated by `generate.py` at build time.
No charting library. No D3. Static SVG is sufficient and loads instantly.

Zero npm. Zero build step. Zero dependencies.

---

### File Structure

```
docs/
├── index.html              ← search + homepage
├── corpus.html             ← stats page
├── css/
│   └── style.css           ← ~100 lines
├── data/
│   └── claims.json         ← pre-built search index (~3MB)
└── claim/
    ├── a3f8c2d1b4e9f7a2.html
    ├── b7e2f1a3c9d4e8f1.html
    └── ...                 ← one file per claim
```

GitHub Pages serves `docs/` directly. Custom domain `pcite.org` via CNAME.
Total hosting cost: zero.

---

## What Is Explicitly Out Of Scope

These are Phase 2. Do not build them now.

- User accounts or researcher profiles
- Claim submission by external users
- Comment or annotation system
- Email alerts or notifications
- API endpoints
- Full-text search (beyond client-side JSON filtering)
- Mobile app
- Visualisation of the full graph (NetworkX renders this locally for the paper)
- Any form of authentication

---

## Milestone 6 Complete When

1. `python -m pcite.generate` runs without errors
2. `pcite.org/claim/{any_valid_claim_id}` loads in a browser
3. That page shows the correct subject, predicate, object
4. That page has a working link to the nanopub URI
5. That page has working links to at least one source DOI
6. `pcite.org` search returns results within one second for any query
7. `pcite.org/corpus` shows the four experiment results matching `data/results.json`

That is the entire acceptance criteria. If all seven pass, Milestone 6 is done.
