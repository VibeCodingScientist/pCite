#!/usr/bin/env python3
"""
validate_components.py — Component validation for pCite paper supplement.

Part 1: Edge typing reliability
  Sample 200 edges, re-classify with Claude Sonnet 4.6 + Gemini 3 Flash + GPT-5.2,
  compute pairwise Cohen's kappa.

Part 2: Claim extraction spot-check
  Sample 50 claims stratified by validation class, verify traceability
  against source abstracts via Claude.

Checkpoints after every API call → restart picks up where it left off.

Output → data/validation/
  - edge_typing_results.json
  - claim_extraction_results.json
  - validation_report.md

Usage:
  python validate_components.py            # run (resumes from checkpoint)
  python validate_components.py --reset    # wipe checkpoints and start fresh
"""

import json, os, random, subprocess, sys, time
from collections import Counter
from pathlib import Path

import anthropic
import networkx as nx
from google import genai
from sklearn.metrics import cohen_kappa_score

from pcite import config

# ── Configuration ────────────────────────────────────────────────────────────
N_EDGES = 200
N_CLAIMS = 50
RANDOM_SEED = 42
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2.0
SLEEP_BETWEEN_CALLS = 0.3

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "validation"
GRAPH_FILE = DATA_DIR / "graph.graphml"
CLAIMS_FILE = DATA_DIR / "claims.jsonl"
PAPERS_FILE = DATA_DIR / "papers.jsonl"
SCORES_FILE = DATA_DIR / "scores.jsonl"

# Checkpoint files
EDGE_CHECKPOINT = OUT_DIR / "_edge_checkpoint.json"
CLAIM_CHECKPOINT = OUT_DIR / "_claim_checkpoint.json"

EDGE_TYPES = ["supports", "extends", "replicates", "contradicts", "applies"]

CLASSIFY_PROMPT = (
    'Classify the relationship between these two biomedical claims.\n'
    'Source: "{src}"\nTarget: "{tgt}"\n\n'
    'Reply with EXACTLY one word, no formatting, no markdown, no asterisks:\n'
    'supports | extends | replicates | contradicts | applies'
)

GEMINI_VALIDATION_MODEL = "gemini-3-flash-preview"
GPT_VALIDATION_MODEL = "gpt-5.2"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _retry(fn, *args, **kwargs):
    """Call fn with retries and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    failed after {MAX_RETRIES} retries: {e}", file=sys.stderr)
                return None
            wait = RETRY_BASE_WAIT ** (attempt + 1)
            print(f"    retry {attempt + 1}/{MAX_RETRIES} ({e}), waiting {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    return None


def _parse_edge_label(text: str | None) -> str:
    """Normalise LLM response to a valid edge type."""
    if not text:
        return "supports"
    cleaned = text.strip().lower().strip("*").split()
    if not cleaned:
        return "supports"
    word = cleaned[0].strip("*")
    return word if word in EDGE_TYPES else "supports"


def _save_checkpoint(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


def _load_checkpoint(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── LLM classifiers ─────────────────────────────────────────────────────────

def _classify_claude(client: anthropic.Anthropic, src: str, tgt: str) -> str | None:
    def _call():
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(src=src, tgt=tgt)}],
        )
        return resp.content[0].text
    result = _retry(_call)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return _parse_edge_label(result) if result else None


def _classify_gemini(client: genai.Client, src: str, tgt: str) -> str | None:
    def _call():
        resp = client.models.generate_content(
            model=GEMINI_VALIDATION_MODEL,
            contents=CLASSIFY_PROMPT.format(src=src, tgt=tgt),
            config={"max_output_tokens": 100},
        )
        return resp.text
    result = _retry(_call)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return _parse_edge_label(result) if result else None


def _classify_gpt(client, src: str, tgt: str) -> str | None:
    def _call():
        resp = client.chat.completions.create(
            model=GPT_VALIDATION_MODEL,
            max_completion_tokens=100,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(src=src, tgt=tgt)}],
        )
        return resp.choices[0].message.content
    result = _retry(_call)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return _parse_edge_label(result) if result else None


# ── Part 1: Edge typing reliability ─────────────────────────────────────────

def run_edge_typing() -> dict:
    print("Part 1: Edge typing reliability", file=sys.stderr)

    # Check for checkpoint
    ckpt = _load_checkpoint(EDGE_CHECKPOINT)
    if ckpt and ckpt.get("done"):
        print("  Loaded completed checkpoint — skipping Part 1", file=sys.stderr)
        return ckpt["results"]

    if ckpt and ckpt.get("edge_data"):
        edge_data = ckpt["edge_data"]
        has_openai = ckpt.get("has_openai", False)
        completed = sum(1 for ed in edge_data if ed.get("claude") is not None)
        print(f"  Resuming from checkpoint: {completed}/{len(edge_data)} edges done",
              file=sys.stderr)
    else:
        # Fresh start: sample edges
        print(f"  Loading graph from {GRAPH_FILE}", file=sys.stderr)
        G = nx.read_graphml(GRAPH_FILE)
        edges = list(G.edges(data=True))
        print(f"  {len(edges)} total edges", file=sys.stderr)

        random.seed(RANDOM_SEED)
        sample = random.sample(edges, min(N_EDGES, len(edges)))
        print(f"  Sampled {len(sample)} edges", file=sys.stderr)

        edge_data = []
        for src_id, tgt_id, data in sample:
            src_node = G.nodes[src_id]
            tgt_node = G.nodes[tgt_id]
            src_text = f"{src_node.get('subject', '')} {src_node.get('predicate', '')} {src_node.get('object', '')}"
            tgt_text = f"{tgt_node.get('subject', '')} {tgt_node.get('predicate', '')} {tgt_node.get('object', '')}"
            original_label = data.get("type", "supports")
            edge_data.append({
                "src_id": src_id, "tgt_id": tgt_id,
                "src_text": src_text, "tgt_text": tgt_text,
                "original": original_label,
            })
        has_openai = bool(os.environ.get("OPENAI_API_KEY", ""))

    # Initialise LLM clients
    claude_client = anthropic.Anthropic(api_key=config.require_api_key())
    gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

    openai_client = None
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_key)
        has_openai = True
        print(f"  GPT ({GPT_VALIDATION_MODEL}): enabled", file=sys.stderr)
    else:
        print("  GPT: skipped (no OPENAI_API_KEY)", file=sys.stderr)

    # Classify each edge (skip already-done ones from checkpoint)
    for i, ed in enumerate(edge_data):
        if ed.get("claude") is not None:
            continue  # already classified in a previous run

        ed["claude"] = _classify_claude(claude_client, ed["src_text"], ed["tgt_text"])
        ed["gemini"] = _classify_gemini(gemini_client, ed["src_text"], ed["tgt_text"])
        if openai_client:
            ed["gpt52"] = _classify_gpt(openai_client, ed["src_text"], ed["tgt_text"])
        else:
            ed["gpt52"] = None

        # Checkpoint after every edge
        _save_checkpoint(EDGE_CHECKPOINT, {
            "edge_data": edge_data, "has_openai": has_openai, "done": False,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(edge_data):
            print(f"  classify: {i + 1}/{len(edge_data)}", file=sys.stderr)

    # ── Compute statistics ───────────────────────────────────────────────
    models = ["claude", "gemini"]
    if has_openai:
        models.append("gpt52")

    valid = [ed for ed in edge_data if all(ed.get(m) is not None for m in models)]
    print(f"  {len(valid)}/{len(edge_data)} edges with all models responding", file=sys.stderr)

    kappa_matrix = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            labels1 = [ed[m1] for ed in valid]
            labels2 = [ed[m2] for ed in valid]
            k = cohen_kappa_score(labels1, labels2)
            kappa_matrix[f"{m1}_vs_{m2}"] = round(k, 4)

    kappa_vs_original = {}
    for m in models:
        labels_m = [ed[m] for ed in valid]
        labels_orig = [ed["original"] for ed in valid]
        k = cohen_kappa_score(labels_m, labels_orig)
        kappa_vs_original[f"{m}_vs_original"] = round(k, 4)

    supports_frac = {}
    for m in ["original"] + models:
        vals = [ed[m] for ed in valid]
        supports_frac[m] = round(vals.count("supports") / len(vals), 4) if vals else 0.0

    per_class = {}
    for etype in EDGE_TYPES:
        subset = [ed for ed in valid if ed["original"] == etype]
        if not subset:
            continue
        agreement = {}
        for m in models:
            match = sum(1 for ed in subset if ed[m] == ed["original"])
            agreement[m] = round(match / len(subset), 4)
        per_class[etype] = {"n": len(subset), **agreement}

    results = {
        "n_sampled": len(edge_data),
        "n_valid": len(valid),
        "models": models,
        "kappa_matrix": kappa_matrix,
        "kappa_vs_original": kappa_vs_original,
        "supports_fraction": supports_frac,
        "per_class_agreement": per_class,
        "edges": edge_data,
    }

    # Mark checkpoint as done
    _save_checkpoint(EDGE_CHECKPOINT, {
        "edge_data": edge_data, "has_openai": has_openai, "done": True,
        "results": results,
    })
    return results


# ── Part 2: Claim extraction spot-check ─────────────────────────────────────

def run_claim_spotcheck() -> dict:
    print("\nPart 2: Claim extraction spot-check", file=sys.stderr)

    # Check for checkpoint
    ckpt = _load_checkpoint(CLAIM_CHECKPOINT)
    if ckpt and ckpt.get("done"):
        print("  Loaded completed checkpoint — skipping Part 2", file=sys.stderr)
        return ckpt["results"]

    # Load data
    scores = [json.loads(l) for l in SCORES_FILE.read_text().splitlines() if l]
    claims_raw = [json.loads(l) for l in CLAIMS_FILE.read_text().splitlines() if l]
    papers_map = {}
    for l in PAPERS_FILE.read_text().splitlines():
        if not l:
            continue
        p = json.loads(l)
        papers_map[p["doi"]] = p

    claims_by_id = {c["id"]: c for c in claims_raw}

    if ckpt and ckpt.get("sampled") and ckpt.get("results_list") is not None:
        sampled = ckpt["sampled"]
        results_list = ckpt["results_list"]
        by_class_keys = ckpt["by_class_keys"]
        completed_ids = {r["claim_id"] for r in results_list}
        print(f"  Resuming from checkpoint: {len(results_list)}/{len(sampled)} claims done",
              file=sys.stderr)
    else:
        # Stratified sample — support both old and new enum names
        classes = [
            "PhysicalMeasurement",
            "DatabaseReferenced", "HumanCurated",       # current/legacy name
            "TextDerived", "AIGenerated",                # current/legacy name
            "Replicated",
        ]
        by_class: dict[str, list[dict]] = {cls: [] for cls in classes}
        for s in scores:
            cls = s["validation_class"]
            if cls in by_class:
                by_class[cls].append(s)

        by_class = {cls: items for cls, items in by_class.items() if items}
        by_class_keys = list(by_class.keys())
        print(f"  Classes: {', '.join(f'{cls}={len(v)}' for cls, v in by_class.items())}",
              file=sys.stderr)

        random.seed(RANDOM_SEED)
        # Deterministic stratified sample: equal share per class, remainder to largest
        base_n = N_CLAIMS // len(by_class)
        remainder = N_CLAIMS - base_n * len(by_class)
        sampled: list[dict] = []
        for i, (cls, items) in enumerate(by_class.items()):
            n = base_n + (1 if i < remainder else 0)
            n = min(n, len(items))
            picked = random.sample(items, n)
            print(f"    {cls}: sampled {len(picked)}", file=sys.stderr)
            sampled.extend(picked)

        results_list = []
        completed_ids = set()
        print(f"  Sampled {len(sampled)} claims", file=sys.stderr)

    claude_client = anthropic.Anthropic(api_key=config.require_api_key())

    for i, score_entry in enumerate(sampled):
        claim_id = score_entry["claim_id"]
        if claim_id in completed_ids:
            continue  # already processed in a previous run

        claim = claims_by_id.get(claim_id)
        if not claim:
            results_list.append({
                "claim_id": claim_id, "status": "claim_not_found",
                "validation_class": score_entry["validation_class"],
            })
            _save_checkpoint(CLAIM_CHECKPOINT, {
                "sampled": sampled, "results_list": results_list,
                "by_class_keys": by_class_keys, "done": False,
            })
            continue

        # Find first provenance DOI with non-empty abstract
        abstract = None
        used_doi = None
        for prov in claim.get("provenance", []):
            doi = prov["doi"]
            paper = papers_map.get(doi)
            if paper and paper.get("abstract") and paper["abstract"].strip():
                abstract = paper["abstract"]
                used_doi = doi
                break

        if not abstract:
            results_list.append({
                "claim_id": claim_id, "status": "no_abstract",
                "validation_class": score_entry["validation_class"],
                "subject": score_entry["subject"],
                "predicate": score_entry["predicate"],
                "object": score_entry["object"],
            })
            _save_checkpoint(CLAIM_CHECKPOINT, {
                "sampled": sampled, "results_list": results_list,
                "by_class_keys": by_class_keys, "done": False,
            })
            continue

        claim_text = (f'{score_entry["subject"]} {score_entry["predicate"]} '
                      f'{score_entry["object"]}')

        verify_prompt = (
            f'A biomedical claim was extracted from a paper.\n\n'
            f'Claim: "{claim_text}"\n'
            f'Paper DOI: {used_doi}\n'
            f'Abstract: "{abstract[:3000]}"\n\n'
            f'Is this claim traceable to the abstract above? '
            f'Could a careful reader derive this claim from the abstract text?\n\n'
            f'Reply with EXACTLY one word: yes or no'
        )

        def _verify():
            resp = claude_client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": verify_prompt}],
            )
            return resp.content[0].text.strip().lower()

        answer = _retry(_verify)
        time.sleep(SLEEP_BETWEEN_CALLS)

        traceable = None
        if answer:
            traceable = answer.startswith("yes")

        results_list.append({
            "claim_id": claim_id,
            "status": "checked",
            "traceable": traceable,
            "validation_class": score_entry["validation_class"],
            "subject": score_entry["subject"],
            "predicate": score_entry["predicate"],
            "object": score_entry["object"],
            "doi": used_doi,
            "llm_response": answer,
        })

        # Checkpoint after every claim
        _save_checkpoint(CLAIM_CHECKPOINT, {
            "sampled": sampled, "results_list": results_list,
            "by_class_keys": by_class_keys, "done": False,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(sampled):
            print(f"  verify: {i + 1}/{len(sampled)}", file=sys.stderr)

    # ── Compute precision ────────────────────────────────────────────────
    checked_items = [r for r in results_list if r["status"] == "checked" and r["traceable"] is not None]
    overall_precision = (
        sum(1 for r in checked_items if r["traceable"]) / len(checked_items)
        if checked_items else 0.0
    )

    per_class_precision = {}
    for cls in by_class_keys:
        cls_items = [r for r in checked_items if r["validation_class"] == cls]
        if cls_items:
            per_class_precision[cls] = {
                "n": len(cls_items),
                "traceable": sum(1 for r in cls_items if r["traceable"]),
                "precision": round(
                    sum(1 for r in cls_items if r["traceable"]) / len(cls_items), 4
                ),
            }

    not_checked = [r for r in results_list if r["status"] != "checked"]

    final = {
        "n_sampled": len(sampled),
        "n_checked": len(checked_items),
        "n_not_checked": len(not_checked),
        "overall_precision": round(overall_precision, 4),
        "per_class_precision": per_class_precision,
        "not_checked_reasons": Counter(r["status"] for r in not_checked),
        "results": results_list,
    }

    # Mark checkpoint as done
    _save_checkpoint(CLAIM_CHECKPOINT, {
        "sampled": sampled, "results_list": results_list,
        "by_class_keys": by_class_keys, "done": True,
        "results": final,
    })
    return final


# ── Report generation ────────────────────────────────────────────────────────

def write_report(edge_results: dict, claim_results: dict) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    sha = _git_sha()

    lines = [
        f"# pCite Component Validation Report",
        f"",
        f"Generated: {ts}  ",
        f"Git SHA: `{sha}`  ",
        f"Seed: {RANDOM_SEED}",
        f"",
        f"---",
        f"",
        f"## Part 1: Edge Typing Reliability",
        f"",
        f"**Models:** Claude Sonnet 4.6 (`{config.CLAUDE_MODEL}`), "
        f"Gemini 3 Flash (`{GEMINI_VALIDATION_MODEL}`), "
        f"GPT-5.2 (`{GPT_VALIDATION_MODEL}`)  ",
        f"**Original labels:** Gemini 2.0 Flash (`{config.GEMINI_MODEL}`) at graph-build time",
        f"",
        f"**Method:** Sampled {edge_results['n_sampled']} edges from the citation graph. "
        f"Re-classified each edge independently with {', '.join(edge_results['models'])}. "
        f"Computed pairwise Cohen's kappa to measure inter-model agreement.",
        f"",
        f"### Kappa Matrix (inter-model)",
        f"",
        f"| Pair | Cohen's kappa |",
        f"|------|--------------|",
    ]
    for pair, k in edge_results["kappa_matrix"].items():
        lines.append(f"| {pair.replace('_', ' ')} | {k:.4f} |")

    lines += [
        f"",
        f"### Kappa vs Original Labels (Gemini at graph-build time)",
        f"",
        f"| Model | Cohen's kappa |",
        f"|-------|--------------|",
    ]
    for pair, k in edge_results["kappa_vs_original"].items():
        model = pair.replace("_vs_original", "")
        lines.append(f"| {model} | {k:.4f} |")

    lines += [
        f"",
        f"### Supports Fraction by Model",
        f"",
        f"| Model | Fraction |",
        f"|-------|----------|",
    ]
    for m, frac in edge_results["supports_fraction"].items():
        lines.append(f"| {m} | {frac:.4f} |")

    lines += [
        f"",
        f"### Per-Class Agreement (fraction matching original label)",
        f"",
        f"| Edge type | N |",
    ]
    models = edge_results["models"]
    header = "| Edge type | N | " + " | ".join(models) + " |"
    sep = "|-----------|---|" + "|".join(["------"] * len(models)) + "|"
    lines[-2:] = [header, sep]

    for etype, data in edge_results["per_class_agreement"].items():
        vals = " | ".join(f"{data.get(m, 0.0):.4f}" for m in models)
        lines.append(f"| {etype} | {data['n']} | {vals} |")

    lines += [
        f"",
        f"**Interpretation guide:**",
        f"- kappa < 0.20: poor agreement",
        f"- 0.21-0.40: fair",
        f"- 0.41-0.60: moderate",
        f"- 0.61-0.80: substantial",
        f"- 0.81-1.00: almost perfect",
        f"",
        f"---",
        f"",
        f"## Part 2: Claim Extraction Spot-Check",
        f"",
        f"**Method:** Stratified sample of {claim_results['n_sampled']} claims across "
        f"validation classes. For each claim, retrieved the source paper abstract and asked "
        f"Claude ({config.CLAUDE_MODEL}) whether the claim is traceable to the abstract.",
        f"",
        f"**Overall extraction precision:** {claim_results['overall_precision']:.4f} "
        f"({claim_results['n_checked']} checked)",
        f"",
    ]

    if claim_results["n_not_checked"] > 0:
        lines.append(f"**Not checked:** {claim_results['n_not_checked']} claims "
                      f"({dict(claim_results['not_checked_reasons'])})")
        lines.append("")

    lines += [
        f"### Per-Class Breakdown",
        f"",
        f"| Validation class | N | Traceable | Precision |",
        f"|------------------|---|-----------|-----------|",
    ]
    for cls, data in claim_results["per_class_precision"].items():
        lines.append(f"| {cls} | {data['n']} | {data['traceable']} | {data['precision']:.4f} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Limitations",
        f"",
        f"1. **Part 1** uses the same prompt template as the original graph construction. "
        f"Models may show inflated agreement due to shared prompt framing.",
        f"2. **Part 2** uses Claude to verify Claude's own extractions. "
        f"This measures internal consistency, not ground truth. "
        f"A manual expert review of a subset is recommended for full validation.",
        f"3. Edge type distribution is heavily skewed toward 'supports' (~80%), "
        f"so per-class kappa for rare types (replicates, contradicts) should be "
        f"interpreted with caution.",
        f"",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parts = set()
    if "--part1" in sys.argv:
        parts.add(1)
    if "--part2" in sys.argv:
        parts.add(2)
    if not parts:
        parts = {1, 2}  # default: run both

    if "--reset" in sys.argv:
        targets = []
        if 1 in parts:
            targets.append(EDGE_CHECKPOINT)
        if 2 in parts:
            targets.append(CLAIM_CHECKPOINT)
        for f in targets:
            if f.exists():
                f.unlink()
                print(f"  Deleted {f}", file=sys.stderr)
        print("  Checkpoints cleared — starting fresh", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1
    edge_results_file = OUT_DIR / "edge_typing_results.json"
    if 1 in parts:
        edge_results = run_edge_typing()
        edge_results_file.write_text(
            json.dumps(edge_results, indent=2, default=str)
        )
        print(f"\n  Saved edge typing results to {edge_results_file}",
              file=sys.stderr)
    elif edge_results_file.exists():
        edge_results = json.loads(edge_results_file.read_text())
        print(f"  Loaded existing Part 1 results from {edge_results_file}", file=sys.stderr)
    else:
        print("  ERROR: No Part 1 results found. Run without --part2 first.",
              file=sys.stderr)
        sys.exit(1)

    # Part 2
    if 2 in parts:
        claim_results = run_claim_spotcheck()
        (OUT_DIR / "claim_extraction_results.json").write_text(
            json.dumps(claim_results, indent=2, default=str)
        )
        print(f"  Saved claim results to {OUT_DIR / 'claim_extraction_results.json'}",
              file=sys.stderr)
    elif (OUT_DIR / "claim_extraction_results.json").exists():
        claim_results = json.loads((OUT_DIR / "claim_extraction_results.json").read_text())
        print(f"  Loaded existing Part 2 results", file=sys.stderr)
    else:
        print("  ERROR: No Part 2 results found. Run without --part1 first.",
              file=sys.stderr)
        sys.exit(1)

    # Report
    report = write_report(edge_results, claim_results)
    (OUT_DIR / "validation_report.md").write_text(report)
    print(f"  Saved report to {OUT_DIR / 'validation_report.md'}", file=sys.stderr)

    # Clean up checkpoint files
    for f in [EDGE_CHECKPOINT, CLAIM_CHECKPOINT]:
        if f.exists():
            f.unlink()
    print("  Cleaned up checkpoint files", file=sys.stderr)

    # Print summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Edge typing: {len(edge_results['kappa_matrix'])} kappa pairs computed",
          file=sys.stderr)
    for pair, k in edge_results["kappa_matrix"].items():
        print(f"    {pair}: {k:.4f}", file=sys.stderr)
    print(f"  Claim precision: {claim_results['overall_precision']:.4f} "
          f"({claim_results['n_checked']}/{claim_results['n_sampled']} checked)",
          file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


if __name__ == "__main__":
    main()
