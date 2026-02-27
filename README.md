# pCite

[![CI](https://github.com/VibeCodingScientist/pCite/actions/workflows/ci.yml/badge.svg)](https://github.com/VibeCodingScientist/pCite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/VibeCodingScientist/pCite/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Site](https://img.shields.io/badge/demo-GitHub%20Pages-green.svg)](https://vibecodingscientist.github.io/pCite/)

**Physical measurement weighting for scientific claim retrieval in the age of AI-generated science.**

AI systems generate scientific claims at near-zero cost. Existing citation metrics assign equal weight to instrument-measured results and text-derived assertions. pCite addresses this by weighting every citation by the validation class of its source claim, with a 1,000-fold gap between PhysicalMeasurement (10.0) and TextDerived (0.01).

---

## Results

Evaluated on 8,761 metabolomics claims from 1,994 papers (2021–2026), with 30,759 typed citation edges.

**Main experiment — MetaboLights-first corpus (5,495 Physical-tier claims)**

| Metric | pCite | Traditional | Lift |
|---|---|---|---|
| Mann-Whitney p (base_weight) | < 2.2e-16 | — | validated median 10.0 vs 0.5 |
| Precision@50 | **0.94** | 0.50 | **1.88x** |
| NDCG@50 | **0.94** | 0.60 | — |

**Negative control — corpus with 0 Physical-tier claims**

| Metric | pCite | Traditional |
|---|---|---|
| Precision@50 | 0.02 | 0.14 |
| NDCG@50 | 0.27 | 0.97 |

pCite loses predictably when no physically-grounded claims exist. This is the expected behaviour: the 1,000-fold weight gap has nothing to act on. The negative control confirms the mechanism, not a failure. Raw data in `data/negative-control/`.

**Sensitivity analysis — weight ratio robustness**

| Ratio | Precision@50 |
|---|---|
| 1:1 | 0.50 |
| 2:1 | 0.60 |
| 5:1 | 0.94 |
| 10:1 | 1.00 |
| 50:1 | 1.00 |
| 100:1 | 1.00 |
| 500:1 | 1.00 |
| 1000:1 | 1.00 |

Precision@50 lift saturates at ratio ≥ 10:1 and is insensitive to the exact production value (1000:1). The result is not an artifact of a specific weight choice. Script: `sensitivity_analysis.py`; data: `data/sensitivity/`.

---

## How it works

**Hypothesis:** Weighting citations by the physical grounding of the source claim surfaces validated scientific claims more accurately than traditional citation count.

**Validation classes and weights:**

| Class | Weight | Definition |
|---|---|---|
| PhysicalMeasurement | 10.0 | Raw instrument data in a public repository (MetaboLights, PDB, PRIDE) |
| ClinicalObservation | 4.0 | EHR-verified patient data, IRB-approved trial outcomes |
| Replicated | 2.0 | Same assertion confirmed in 3+ independent sources |
| DatabaseReferenced | 0.5 | Structured database deposit, no raw data |
| TextDerived | 0.01 | Synthesised from literature text |
| Hypothesis | 0.0 | Proposed, untested |

**Scoring formula:**

```
base_weight  = ValidationWeight(class) × log₂(replication_count + 1)
edge_weight  = PCiteTypeWeight(type)   × source.base_weight
pcite_score  = Σ incoming edge weights
```

Citation edge types and multipliers: `replicates` 1.5 · `extends` 1.2 · `supports` 1.0 · `contradicts` -0.5 · `applies` 0.6.

A PhysicalMeasurement claim cited 35 times by other physical claims reaches a score of 364. A TextDerived claim cited 1,000 times peaks at 0.10. No manual scoring. Entirely from the data model.

---

## Pipeline

```
MetaboLights API + PubMed eUtils
        |
corpus.py       →  data/papers.jsonl        (1,994 papers)
        |
extract.py      →  data/claims.jsonl        (8,761 claims, Claude tool_use)
        |
graph.py        →  data/graph.graphml       (30,759 edges, Gemini Flash classification)
                →  data/scores.jsonl
        |
evaluate.py     →  data/results.json
                →  figures/*.pdf
```

Corpus construction is MetaboLights-first: papers retrieved via the EBI REST API are classified as PhysicalMeasurement-tier by construction, not by text inference. No classifier needed for ground truth.

---

## Quick start

```bash
pip install -e .
python run_poc.py --dry-run     # evaluate cached data, no API calls required
```

To reproduce from scratch (requires API keys):

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY + GEMINI_API_KEY
python -m pcite.corpus          # fetch papers
python -m pcite.extract         # extract claims via Claude
python -m pcite.graph           # build citation graph via Gemini + OpenAlex
python -m pcite.evaluate        # compute metrics and figures
```

Results are deterministic. Gemini classification responses are cached in `data/classify_cache.json`; rerunning without clearing the cache produces identical edge assignments.

---

## Repository layout

```
src/pcite/
  models.py       — ValidationClass, Claim, PCite, scoring formula
  corpus.py       — MetaboLights-first corpus construction
  extract.py      — Claude tool_use claim extraction
  graph.py        — OpenAlex citation graph + Gemini edge classification
  evaluate.py     — Mann-Whitney, Precision@k, NDCG@k
data/
  papers.jsonl
  claims.jsonl
  scores.jsonl
  graph.graphml
  results.json
  negative-control/
  sensitivity/
figures/
  fig1_rank_comparison.pdf
  fig2_score_dist.pdf
  fig3_precision_at_k.pdf
  fig_sensitivity.pdf
tests/             — 25 tests, no API keys needed
docs/              — static site (GitHub Pages)
```

---

## Requirements

- Python >= 3.11
- `ANTHROPIC_API_KEY` — claim extraction (Claude Sonnet 4.5)
- `GEMINI_API_KEY` — citation edge classification (Gemini 2.0 Flash)
- OpenAlex is used without authentication (mailto param recommended)
- See `.env.example` for all configuration

```bash
pytest tests/ -v
```

