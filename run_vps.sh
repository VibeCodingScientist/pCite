#!/usr/bin/env bash
# run_vps.sh — Run boundary investigation + deposit-first pipeline on VPS
# Usage:   nohup bash run_vps.sh &> run_vps.log &
# Resume:  nohup bash run_vps.sh &>> run_vps.log &   (completed steps are skipped)
# Fresh:   nohup bash run_vps.sh --fresh &> run_vps.log &   (delete all checkpoints first)
set -euo pipefail

cd ~/nCite
VENV=".venv/bin/python"
LOG="run_vps.log"
CKPT_DIR="data/pride"
STARTED=$(date '+%Y-%m-%d %H:%M:%S')

# ── --fresh flag: remove all checkpoint files ────────────────────────
if [[ "${1:-}" == "--fresh" ]]; then
    echo "Cleaning all checkpoint files..."
    rm -f "$CKPT_DIR"/.ckpt_step_* "$CKPT_DIR"/.ckpt_projects.json "$CKPT_DIR"/.ckpt_neighbours.json
    rm -f data/pride/claims.partial.jsonl
    echo "  Done — starting fresh run"
fi

# ── Helpers ──────────────────────────────────────────────────────────────
ok()   { echo "  ✅ $1"; }
fail() { echo "  ❌ $1"; }
warn() { echo "  ⚠️  $1"; }
hr()   { echo "────────────────────────────────────────────────────────"; }

# ── Preflight checks ────────────────────────────────────────────────────
hr
echo "pCite VPS Run — $STARTED"
hr

echo "1/7  Preflight checks..."

if [ ! -f "$VENV" ]; then
    fail "venv not found at $VENV"
    exit 1
fi
ok "venv exists"

if ! $VENV -c "import pcite; import pride_corpus; import boundary_investigation" 2>/dev/null; then
    fail "imports failed — running pip install"
    $VENV -m pip install -e . --quiet 2>&1 || { fail "pip install failed"; exit 1; }
    ok "pip install recovered"
else
    ok "imports OK"
fi

if [ ! -f "data/scores.jsonl" ]; then
    fail "data/scores.jsonl missing (MetaboLights)"
    exit 1
fi
ok "MetaboLights scores.jsonl present ($(wc -l < data/scores.jsonl) records)"

if [ -f "data/pride/scores.jsonl" ]; then
    ok "PRIDE v1 scores.jsonl present ($(wc -l < data/pride/scores.jsonl) records) — will overlay"
else
    warn "PRIDE v1 scores.jsonl not found — boundary plot will skip overlay"
fi

if [ -f ".env" ]; then
    ok ".env present"
else
    warn ".env missing — API calls may fail"
fi

# ── Tests ────────────────────────────────────────────────────────────────
echo ""
echo "2/7  Running tests..."
if $VENV -m pytest tests/ -q --tb=line 2>&1; then
    ok "All tests passed"
else
    fail "Tests failed — continuing anyway (boundary uses cached data)"
    warn "Check test output above for details"
fi

# ── Boundary investigation ───────────────────────────────────────────────
hr
echo ""
if [ -f "$CKPT_DIR/.ckpt_step_3" ]; then
    ok "3/7  Boundary investigation — SKIPPED (checkpoint exists)"
else
    echo "3/7  Boundary investigation (subsampling MetaboLights data)..."
    echo "     Coverage levels: 5% 10% 15% 20% 30% 40% 50% 62.4%"
    echo "     5 random seeds each × 3 ranking methods"
    echo ""

    if $VENV boundary_investigation.py 2>&1; then
        ok "Boundary investigation complete"
        if [ -f "data/boundary_results.json" ]; then
            ok "data/boundary_results.json written"
        fi
        if [ -f "figures/fig_boundary.pdf" ]; then
            ok "figures/fig_boundary.pdf written"
        fi
        touch "$CKPT_DIR/.ckpt_step_3"
    else
        BOUNDARY_EXIT=$?
        fail "Boundary investigation failed (exit $BOUNDARY_EXIT)"
        warn "Fallback: checking if partial results exist..."
        if [ -f "data/boundary_results.json" ]; then
            warn "Partial boundary_results.json exists — may be usable"
            touch "$CKPT_DIR/.ckpt_step_3"
        else
            warn "No boundary results — will continue with deposit-first pipeline"
        fi
    fi
fi

# ── Deposit-first corpus build ───────────────────────────────────────────
hr
echo ""
if [ -f "$CKPT_DIR/.ckpt_step_4" ]; then
    ok "4/7  Deposit-first corpus build — SKIPPED (checkpoint exists)"
else
    echo "4/7  Deposit-first corpus build (PRIDE API + PubMed + OpenAlex)..."
    echo "     Broader search terms + citation-neighbourhood sampling"
    echo "     Target ~50% deposit coverage (deposit papers + neighbour negatives)"
    echo ""

    if $VENV pride_corpus.py --deposit-first 2>&1; then
        ok "Deposit-first corpus built"
        if [ -f "data/pride/papers.jsonl" ]; then
            N_PAPERS=$(wc -l < data/pride/papers.jsonl)
            N_DEPOSIT=$(grep -c '"deposit_id"' data/pride/papers.jsonl || echo 0)
            ok "data/pride/papers.jsonl: $N_PAPERS papers"
        fi
        touch "$CKPT_DIR/.ckpt_step_4"
    else
        CORPUS_EXIT=$?
        fail "Corpus build failed (exit $CORPUS_EXIT)"
        if [ -f "data/pride/papers.jsonl" ]; then
            warn "Fallback: existing papers.jsonl found — attempting pipeline with cached corpus"
        else
            fail "No papers.jsonl — cannot continue pipeline"
            echo ""
            hr
            echo "PARTIAL RUN — boundary investigation may have succeeded"
            echo "Fix corpus build and re-run: nohup bash run_vps.sh &>> run_vps.log &"
            hr
            exit 1
        fi
    fi
fi

# ── Full pipeline (extract → validate → graph → evaluate) ───────────────
hr
echo ""
if [ -f "$CKPT_DIR/.ckpt_step_5" ]; then
    ok "5/7  Full pipeline — SKIPPED (checkpoint exists)"
else
    echo "5/7  Full pipeline (extract + validate + graph + evaluate)..."
    echo "     Using deposit-first corpus, --skip-corpus flag"
    echo "     This will take ~30-60 min (Claude + Gemini + OpenAlex API calls)"
    echo ""

    if $VENV run_pride_poc.py --deposit-first --skip-corpus 2>&1; then
        ok "Pipeline complete"
        touch "$CKPT_DIR/.ckpt_step_5"
    else
        PIPE_EXIT=$?
        fail "Pipeline failed (exit $PIPE_EXIT)"
        if [ -f "data/pride/scores.jsonl" ]; then
            warn "Fallback: scores.jsonl exists — attempting evaluation with cached scores"
        else
            fail "No scores.jsonl — cannot evaluate"
            echo ""
            hr
            echo "PARTIAL RUN — boundary + corpus may have succeeded"
            echo "Debug, then re-run: nohup bash run_vps.sh &>> run_vps.log &"
            hr
            exit 1
        fi
    fi
fi

# ── Graded evaluation ───────────────────────────────────────────────────
hr
echo ""
if [ -f "$CKPT_DIR/.ckpt_step_6" ]; then
    ok "6/7  Graded evaluation — SKIPPED (checkpoint exists)"
else
    echo "6/7  Graded deposit-quality evaluation..."

    if [ -f "pride_graded_eval.py" ]; then
        if $VENV pride_graded_eval.py 2>&1; then
            ok "Graded evaluation complete"
            touch "$CKPT_DIR/.ckpt_step_6"
        else
            fail "Graded eval failed — non-critical, main results already in step 5"
        fi
    else
        warn "pride_graded_eval.py not found — skipping graded eval"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────
hr
echo ""
echo "7/7  Summary"
echo ""

FINISHED=$(date '+%Y-%m-%d %H:%M:%S')
echo "  Started:  $STARTED"
echo "  Finished: $FINISHED"
echo ""

if [ -f "data/boundary_results.json" ]; then
    ok "Boundary results: data/boundary_results.json"
fi
if [ -f "figures/fig_boundary.pdf" ]; then
    ok "Boundary figure:  figures/fig_boundary.pdf"
fi
if [ -f "data/pride/papers.jsonl" ]; then
    ok "Corpus:           data/pride/papers.jsonl ($(wc -l < data/pride/papers.jsonl) papers)"
fi
if [ -f "data/pride/scores.jsonl" ]; then
    ok "Scores:           data/pride/scores.jsonl ($(wc -l < data/pride/scores.jsonl) records)"
fi
if [ -f "data/pride/results.json" ]; then
    ok "Results:          data/pride/results.json"
    echo ""
    echo "  Results contents:"
    $VENV -c "
import json
r = json.loads(open('data/pride/results.json').read())
mw = r['mann_whitney']
p50 = r['precision_50']
ng = r['ndcg_50']
print(f\"    n total:        {r['n_total']:,}\")
print(f\"    Mann-Whitney p: {mw['p_value']:.4f}  {'✅ PASS' if mw['p_value'] < 0.05 else '❌ FAIL'}\")
print(f\"    P@50:           pCite={p50['precision_pcite']:.3f} vs trad={p50['precision_traditional']:.3f}  ({p50['lift']:.1f}x lift)\")
print(f\"    NDCG@50:        pCite={ng['ndcg_pcite']:.4f} vs trad={ng['ndcg_traditional']:.4f}\")
holds = mw['p_value'] < 0.05 and p50['lift'] >= 1.0 and ng['ndcg_pcite'] > ng['ndcg_traditional']
print(f\"    Hypothesis:     {'✅ PASS' if holds else '❌ FAIL'}\")
" 2>&1
fi

hr
echo "Done. Pull results: rsync -avz deploy@45.55.129.88:~/nCite/data/pride/ data/pride/"
echo "Pull figures: rsync -avz deploy@45.55.129.88:~/nCite/figures/ figures/"
hr
