# nCite

Validation-weighted citation framework for reproducible biomedical knowledge graphs.

## Results

Run 3 corpus: 8,761 claims from 1,287 papers, 5,495 Physical-tier (MetaboLights-verified), 30,759 citation edges.

| Metric | nCite | Traditional | |
|---|---|---|---|
| Mann-Whitney | p &asymp; 0 | &mdash; | validated median 10.0 vs unvalidated 0.5 |
| Precision@50 | **0.92** | 0.50 | **1.84x lift** |
| NDCG@50 | **0.92** | 0.60 | |

nCite surfaces physically-validated claims better than traditional citation count.

## Architecture

```
PubMed + MetaboLights APIs
        |
corpus.py       ->  data/papers.jsonl
        |
extract.py      ->  data/claims.jsonl
        |
validate.py     ->  data/claims.jsonl (upgraded)
                ->  data/nanopubs/*.trig
        |
graph.py        ->  data/graph.graphml
                ->  data/scores.jsonl
        |
evaluate.py     ->  data/results.json
                ->  figures/*.pdf
```

Six modules + orchestrator. Single `pyproject.toml`. No Docker. No services.

## Packages

- **`src/ncite/`** &mdash; pipeline: corpus fetch, claim extraction, validation, graph build, evaluation
- **`src/pcite/`** &mdash; static site generator for [pcite.org](https://pcite.org) (8,761 claim pages)

## Quick start

```bash
pip install -e .
python run_poc.py --dry-run     # evaluate cached data, no API calls
python -m pcite.generate        # generate static site in docs/
```

## Requirements

- Python >= 3.11
- API keys: `ANTHROPIC_API_KEY` (extraction), `GEMINI_API_KEY` (edge classification)
- See `.env.example` for all configuration

## Tests

```bash
pytest tests/ -v    # 23 tests, no API keys needed
```

## License

MIT
