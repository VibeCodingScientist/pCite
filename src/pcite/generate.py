"""
pcite.generate — static site generator for pcite.org

Reads data/{claims,scores}.jsonl + data/results.json
Writes docs/ with index.html, corpus.html, claim/{id}.html, css/style.css, data/claims.json

Run: python -m pcite.generate
"""

import html
import json
import sys
from pathlib import Path

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")

COLORS = {
    "PhysicalMeasurement": "#d62728",
    "ClinicalObservation":  "#e377c2",
    "Replicated":           "#ff7f0e",
    "HumanCurated":         "#2ca02c",
    "AIGenerated":          "#aec7e8",
    "Hypothesis":           "#c7c7c7",
}

SHORT_NAMES = {
    "PhysicalMeasurement": "Physical",
    "ClinicalObservation":  "Clinical",
    "Replicated":           "Replicated",
    "HumanCurated":         "Curated",
    "AIGenerated":          "AI Generated",
    "Hypothesis":           "Hypothesis",
}

WEIGHTS = {
    "PhysicalMeasurement": 10.0,
    "ClinicalObservation":  4.0,
    "Replicated":           2.0,
    "HumanCurated":         0.5,
    "AIGenerated":          0.01,
    "Hypothesis":           0.0,
}

CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Georgia",serif;color:#111;background:#fff;line-height:1.6;
  max-width:860px;margin:0 auto;padding:1rem 1.5rem}
a{color:#1a0dab;text-decoration:none}
a:hover{text-decoration:underline}
header{display:flex;justify-content:space-between;align-items:center;
  padding:.75rem 0;border-bottom:1px solid #ddd;margin-bottom:1.5rem}
header h1{font-size:1.1rem;font-weight:700}
header h1 a{color:#111}
header nav a{font-size:.85rem;margin-left:1rem;color:#555}
.section-label{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:#666;margin-bottom:.5rem;margin-top:1.5rem}
.section-rule{border:none;border-top:1px solid #e0e0e0;margin-bottom:1rem}
.triple{font-size:1.15rem;margin:.5rem 0}
.triple .pred{color:#555;font-size:.9rem;margin:0 .3rem}
.entity-id{font-size:.75rem;color:#888;margin-top:.25rem}
.val-bar{height:12px;border-radius:2px;margin:.4rem 0;
  background:linear-gradient(to right,var(--color) var(--pct),#eee var(--pct))}
.stat-row{display:flex;gap:2rem;font-size:.9rem;flex-wrap:wrap}
.stat-row span{white-space:nowrap}
.evidence-row{font-size:.85rem;margin:.3rem 0;display:flex;gap:.5rem;align-items:baseline}
.evidence-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0;
  position:relative;top:1px}
.id-table{font-size:.85rem;margin:.3rem 0}
.id-table td{padding:.15rem .5rem .15rem 0;vertical-align:top}
.id-table td:first-child{color:#666;white-space:nowrap}
input[type=text]{width:100%;padding:.5rem .75rem;font-size:1rem;font-family:inherit;
  border:1px solid #ccc;border-radius:3px;margin-bottom:.75rem}
.filters{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem}
.filters select{font-family:inherit;font-size:.85rem;padding:.3rem .5rem;
  border:1px solid #ccc;border-radius:3px}
.summary{font-size:.85rem;color:#666;margin-bottom:.75rem}
.result{display:flex;gap:.75rem;padding:.5rem 0;border-bottom:1px solid #f0f0f0;
  align-items:center}
.result-bar{width:40px;height:24px;border-radius:2px;flex-shrink:0;
  background:linear-gradient(to right,var(--color) var(--pct),#eee var(--pct))}
.result-info{flex:1;min-width:0}
.result-claim{font-size:.95rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.result-meta{font-size:.8rem;color:#666}
.result a.claim-link{font-size:.8rem;white-space:nowrap}
#load-more{display:none;margin:1rem auto;padding:.4rem 1.5rem;font-family:inherit;
  font-size:.9rem;border:1px solid #ccc;border-radius:3px;background:#fff;cursor:pointer}
.cite-bar{display:flex;align-items:center;gap:.5rem;font-size:.85rem;margin:.2rem 0}
.cite-bar-fill{height:10px;border-radius:2px;background:#888}
.corpus-stat{font-size:.95rem;margin:.2rem 0}
.dist-row{display:flex;align-items:center;gap:.5rem;font-size:.85rem;margin:.3rem 0}
.dist-bar{height:14px;border-radius:2px;flex-shrink:0}
.dist-label{width:80px;text-align:right;flex-shrink:0}
.dist-count{width:60px;font-size:.8rem;color:#666}
footer{margin-top:2rem;padding-top:.75rem;border-top:1px solid #ddd;
  font-size:.75rem;color:#888;text-align:center}
@media(max-width:600px){
  .filters{flex-direction:column}
  .result{flex-wrap:wrap}
  .stat-row{flex-direction:column;gap:.3rem}
}
"""


def _esc(s):
    return html.escape(str(s))


def _header(title, depth=0):
    prefix = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} — pCite</title>
<link rel="stylesheet" href="{prefix}css/style.css">
</head>
<body>
<header>
  <h1><a href="{prefix}index.html">pCite</a></h1>
  <nav><a href="{prefix}index.html">Search</a><a href="{prefix}corpus.html">Corpus</a></nav>
</header>
"""


def _footer():
    return """<footer>pCite — validation-weighted scientific claims</footer>
</body>
</html>
"""


def _score_val(score_data):
    """Use pcite_score when non-zero, fall back to base_weight."""
    ns = score_data.get("pcite_score", 0)
    return ns if ns > 0 else score_data.get("base_weight", 0)


def load_data():
    claims_path = DATA_DIR / "claims.jsonl"
    scores_path = DATA_DIR / "scores.jsonl"
    results_path = DATA_DIR / "results.json"

    claims = [json.loads(l) for l in claims_path.read_text().splitlines() if l]
    scores = [json.loads(l) for l in scores_path.read_text().splitlines() if l]
    scores_by_id = {s["claim_id"]: s for s in scores}

    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    graph_edges = {}  # target_id -> list of edge dicts
    graph_path = DATA_DIR / "graph.graphml"
    if graph_path.exists():
        import networkx as nx
        G = nx.read_graphml(graph_path)
        for u, v, d in G.edges(data=True):
            graph_edges.setdefault(v, []).append({
                "source": u, "target": v,
                "type": d.get("type", "supports"),
                "weight": float(d.get("weight", 0)),
            })

    return claims, scores_by_id, results, graph_edges


def build_search_index(claims, scores_by_id):
    max_score = max((_score_val(scores_by_id.get(c["id"], {})) for c in claims), default=1) or 1
    index = []
    for c in claims:
        sd = scores_by_id.get(c["id"], {})
        score = _score_val(sd)
        index.append({
            "id": c["id"],
            "s": c["subject"]["name"],
            "p": c["predicate"],
            "o": c["object"]["name"],
            "v": sd.get("validation_class", c.get("validation_class", "")),
            "r": sd.get("replication_count", c.get("replication_count", 1)),
            "b": round(score, 3),
            "n": round(score / max_score * 100, 1),
        })
    index.sort(key=lambda x: x["b"], reverse=True)
    return index


def render_claim(claim, score_data, graph_edges):
    c = claim
    sd = score_data
    vc = sd.get("validation_class", c.get("validation_class", "AIGenerated"))
    short = SHORT_NAMES.get(vc, vc)
    color = COLORS.get(vc, "#888")
    weight = WEIGHTS.get(vc, 0)
    pct = weight / 10.0 * 100
    score = _score_val(sd)
    rep = sd.get("replication_count", c.get("replication_count", 1))

    # Evidence chain sorted by validation class weight desc
    provenance = c.get("provenance", [])
    provenance_sorted = sorted(
        provenance,
        key=lambda p: WEIGHTS.get(p.get("validation_class", "AIGenerated"), 0),
        reverse=True,
    )

    evidence_html = ""
    for p in provenance_sorted:
        pvc = p.get("validation_class", "AIGenerated")
        pc = COLORS.get(pvc, "#888")
        ps = SHORT_NAMES.get(pvc, pvc)
        doi = _esc(p["doi"])
        metabo = ""
        if p.get("metabo_id"):
            mid = _esc(p["metabo_id"])
            metabo = f' <a href="https://www.ebi.ac.uk/metabolights/{mid}">{mid}</a>'
        evidence_html += (
            f'<div class="evidence-row">'
            f'<span class="evidence-dot" style="background:{pc}" title="{_esc(ps)}"></span>'
            f'<span>{_esc(ps)}</span>'
            f'<a href="https://doi.org/{doi}">{doi}</a>'
            f'{metabo}'
            f'</div>\n'
        )

    # Citations from graph
    edges = graph_edges.get(c["id"], [])
    citations_html = ""
    if edges:
        from collections import Counter
        by_type = Counter(e["type"] for e in edges)
        total = len(edges)
        max_ct = max(by_type.values()) if by_type else 1
        citations_html += f'<p class="section-label">CITATIONS RECEIVED &middot; {total} total</p>\n<hr class="section-rule">\n'
        for ct in ["replicates", "supports", "contradicts", "extends", "applies"]:
            count = by_type.get(ct, 0)
            bw = count / max_ct * 100 if max_ct else 0
            citations_html += (
                f'<div class="cite-bar">'
                f'<span style="width:70px;text-align:right">{ct.title()}</span>'
                f'<span class="cite-bar-fill" style="width:{bw:.0f}px"></span>'
                f'<span>{count}</span>'
                f'</div>\n'
            )
    else:
        citations_html = '<p style="font-size:.85rem;color:#888">No citations yet.</p>\n'

    subj = c["subject"]
    obj = c["object"]

    return _header(f"{_esc(subj['name'])} {_esc(c['predicate'])} {_esc(obj['name'])}", depth=1) + f"""
<p class="section-label">CLAIM</p>
<hr class="section-rule">
<div class="triple">
  <strong>{_esc(subj['name'])}</strong>
  <span class="pred">&mdash;{_esc(c['predicate'])}&rarr;</span>
  <strong>{_esc(obj['name'])}</strong>
</div>
<div class="entity-id">{_esc(subj['id'])} &nbsp;&middot;&nbsp; {_esc(obj['id'])}</div>

<p class="section-label">VALIDATION</p>
<hr class="section-rule">
<div class="val-bar" style="--pct:{pct:.0f}%;--color:{color}"></div>
<div class="stat-row">
  <span>Class: {_esc(short)} ({weight})</span>
  <span>Replicated: {rep} paper{"s" if rep != 1 else ""}</span>
  <span>Score: {score:.2f}</span>
</div>

<p class="section-label">EVIDENCE CHAIN &middot; {len(provenance)} source{"s" if len(provenance) != 1 else ""}</p>
<hr class="section-rule">
{evidence_html}

{citations_html}

<p class="section-label">IDENTIFIERS</p>
<hr class="section-rule">
<table class="id-table">
<tr><td>Claim ID</td><td><code>{_esc(c['id'])}</code></td></tr>
<tr><td>Subject URI</td><td><a href="{_esc(subj.get('uri', 'https://identifiers.org/' + subj['id']))}">{_esc(subj['id'])}</a></td></tr>
<tr><td>Object URI</td><td><a href="{_esc(obj.get('uri', 'https://identifiers.org/' + obj['id']))}">{_esc(obj['id'])}</a></td></tr>
</table>
""" + _footer()


def render_index(claims, scores_by_id):
    # Collect unique predicates and validation classes for filters
    predicates = sorted({c["predicate"] for c in claims})
    pred_options = '<option value="">All predicates</option>\n'
    for p in predicates:
        pred_options += f'    <option value="{_esc(p)}">{_esc(p)}</option>\n'

    classes = ["PhysicalMeasurement", "ClinicalObservation", "Replicated",
               "HumanCurated", "AIGenerated", "Hypothesis"]
    cls_options = '<option value="">All classes</option>\n'
    for vc in classes:
        short = SHORT_NAMES.get(vc, vc)
        cls_options += f'    <option value="{_esc(vc)}">{_esc(short)}</option>\n'

    # Summary stats
    from collections import Counter
    vc_counts = Counter(scores_by_id[c["id"]].get("validation_class", "") for c in claims if c["id"] in scores_by_id)
    n_papers = len({p["doi"] for c in claims for p in c.get("provenance", [])})

    return _header("Search claims", depth=0) + f"""
<p style="font-size:.9rem;color:#666;margin-bottom:.75rem">Validation-weighted scientific claims</p>

<input type="text" id="q" placeholder="Search claims — compound, disease, or relationship" autofocus>

<div class="filters">
  <select id="f-pred">{pred_options}</select>
  <select id="f-class">{cls_options}</select>
  <select id="f-rep">
    <option value="1">Min replication: 1</option>
    <option value="2">Min replication: 2</option>
    <option value="3">Min replication: 3</option>
    <option value="5">Min replication: 5</option>
  </select>
</div>

<div class="summary" id="summary">{len(claims):,} claims &middot; from {n_papers:,} papers</div>

<div id="results"></div>
<button id="load-more">Load more</button>

<script>
(function() {{
  const COLORS = {json.dumps(COLORS)};
  let allClaims = [], filtered = [], shown = 0;
  const BATCH = 50;

  fetch("data/claims.json")
    .then(r => r.json())
    .then(data => {{ allClaims = data; applyFilters(); }});

  const q = document.getElementById("q");
  const fPred = document.getElementById("f-pred");
  const fClass = document.getElementById("f-class");
  const fRep = document.getElementById("f-rep");
  const results = document.getElementById("results");
  const loadMore = document.getElementById("load-more");
  const summary = document.getElementById("summary");

  q.addEventListener("input", applyFilters);
  fPred.addEventListener("change", applyFilters);
  fClass.addEventListener("change", applyFilters);
  fRep.addEventListener("change", applyFilters);
  loadMore.addEventListener("click", showMore);

  function applyFilters() {{
    const query = q.value.toLowerCase();
    const pred = fPred.value;
    const vc = fClass.value;
    const minRep = parseInt(fRep.value) || 1;

    filtered = allClaims.filter(c => {{
      if (query && !(c.s.toLowerCase().includes(query) ||
          c.o.toLowerCase().includes(query) ||
          c.p.toLowerCase().includes(query))) return false;
      if (pred && c.p !== pred) return false;
      if (vc && c.v !== vc) return false;
      if (c.r < minRep) return false;
      return true;
    }});

    summary.textContent = filtered.length.toLocaleString() + " claims";
    shown = 0;
    results.innerHTML = "";
    showMore();
  }}

  function showMore() {{
    const end = Math.min(shown + BATCH, filtered.length);
    for (let i = shown; i < end; i++) {{
      const c = filtered[i];
      const color = COLORS[c.v] || "#888";
      const div = document.createElement("div");
      div.className = "result";
      div.innerHTML =
        '<div class="result-bar" style="--pct:' + c.n + '%;--color:' + color + '"></div>' +
        '<div class="result-info">' +
          '<div class="result-claim">' + esc(c.s) + ' <span style="color:#888">' + esc(c.p) + '</span> ' + esc(c.o) + '</div>' +
          '<div class="result-meta">' + shortName(c.v) + ' &middot; ' + c.r + ' paper' + (c.r !== 1 ? 's' : '') + ' &middot; score ' + c.b + '</div>' +
        '</div>' +
        '<a class="claim-link" href="claim/' + c.id + '.html">view &rarr;</a>';
      results.appendChild(div);
    }}
    shown = end;
    loadMore.style.display = shown < filtered.length ? "block" : "none";
  }}

  function esc(s) {{
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }}

  function shortName(v) {{
    const m = {{"PhysicalMeasurement":"Physical","ClinicalObservation":"Clinical",
      "Replicated":"Replicated","HumanCurated":"Curated","AIGenerated":"AI Generated",
      "Hypothesis":"Hypothesis"}};
    return m[v] || v;
  }}
}})();
</script>
""" + _footer()


def render_corpus(claims, scores_by_id, results):
    from collections import Counter
    vc_counts = Counter(
        scores_by_id[c["id"]].get("validation_class", "")
        for c in claims if c["id"] in scores_by_id
    )
    n_papers = len({p["doi"] for c in claims for p in c.get("provenance", [])})
    n_claims = len(claims)

    # Experiment results
    mw = results.get("mann_whitney", {})
    p50 = results.get("precision_50", {})
    ng = results.get("ndcg_50", {})

    p_val = mw.get("p_value", 0)
    p_str = f"{p_val:.2e}" if p_val < 0.01 else f"{p_val:.4f}"
    sig = "significant" if p_val < 0.05 else "not significant"

    p_nc = p50.get("precision_pcite", 0)
    p_tr = p50.get("precision_traditional", 0)
    lift = p50.get("lift", 0)
    ndcg_nc = ng.get("ndcg_pcite", 0)
    ndcg_tr = ng.get("ndcg_traditional", 0)

    # Validation class distribution bars
    classes_order = [
        "PhysicalMeasurement", "ClinicalObservation", "Replicated",
        "HumanCurated", "AIGenerated",
    ]
    max_count = max((vc_counts.get(vc, 0) for vc in classes_order), default=1) or 1
    dist_html = ""
    for vc in classes_order:
        ct = vc_counts.get(vc, 0)
        if ct == 0 and vc not in vc_counts:
            continue
        short = SHORT_NAMES.get(vc, vc)
        color = COLORS.get(vc, "#888")
        bw = ct / max_count * 200
        pct = ct / n_claims * 100 if n_claims else 0
        dist_html += (
            f'<div class="dist-row">'
            f'<span class="dist-label">{_esc(short)}</span>'
            f'<span class="dist-bar" style="width:{bw:.0f}px;background:{color}"></span>'
            f'<span class="dist-count">{ct:,} &nbsp; {pct:.1f}%</span>'
            f'</div>\n'
        )

    return _header("Corpus", depth=0) + f"""
<p class="section-label">METABOLOMICS KNOWLEDGE GRAPH</p>
<hr class="section-rule">
<div class="corpus-stat">Papers: <strong>{n_papers:,}</strong></div>
<div class="corpus-stat">Claims: <strong>{n_claims:,}</strong> (unique assertions, deduplicated)</div>
<div class="corpus-stat">Physical claims: <strong>{vc_counts.get("PhysicalMeasurement", 0):,}</strong> (MetaboLights deposit verified)</div>

<p class="section-label">EXPERIMENT RESULTS</p>
<hr class="section-rule">
<p style="font-size:.9rem;margin-bottom:.75rem">
  <em>Hypothesis: pCite surfaces physically-validated claims better than traditional citation count.</em>
</p>
<table class="id-table">
  <tr><td>Mann-Whitney U</td><td>p = {p_str} &nbsp; {"&#10003;" if p_val < 0.05 else "&#10007;"} {sig}</td></tr>
  <tr><td>Precision@50</td><td>pCite {p_nc:.2f} vs Traditional {p_tr:.2f} ({lift:.1f}&times; lift)</td></tr>
  <tr><td>NDCG@50</td><td>pCite {ndcg_nc:.2f} vs Traditional {ndcg_tr:.2f}</td></tr>
</table>
<p style="font-size:.9rem;margin-top:.5rem"><strong>{"Hypothesis holds." if p_val < 0.05 else "Hypothesis not supported."}</strong></p>

<p class="section-label">VALIDATION CLASS DISTRIBUTION</p>
<hr class="section-rule">
{dist_html}
<p style="font-size:.85rem;color:#555;margin-top:.75rem;max-width:640px">
  The distribution is the argument. {vc_counts.get("AIGenerated", 0) / n_claims * 100 if n_claims else 0:.0f}% of extracted claims have
  no physical anchor. Under traditional citation metrics, these
  claims are indistinguishable from the {vc_counts.get("PhysicalMeasurement", 0) / n_claims * 100 if n_claims else 0:.1f}% that do.
</p>

<p class="section-label">CODE &amp; DATA</p>
<hr class="section-rule">
<table class="id-table">
  <tr><td>GitHub</td><td><a href="https://github.com/VibeCodingScientist/pCite">github.com/VibeCodingScientist/pCite</a></td></tr>
  <tr><td>Data</td><td>CC0 &mdash; <a href="../data/claims.jsonl">download claims.jsonl</a></td></tr>
</table>
""" + _footer()


def main():
    print("pcite: loading data...", file=sys.stderr)
    claims, scores_by_id, results, graph_edges = load_data()
    print(f"  {len(claims):,} claims, {len(scores_by_id):,} scores", file=sys.stderr)

    # Create output dirs
    for d in [DOCS_DIR, DOCS_DIR / "css", DOCS_DIR / "data", DOCS_DIR / "claim"]:
        d.mkdir(parents=True, exist_ok=True)

    # Write .nojekyll
    (DOCS_DIR / ".nojekyll").write_text("")

    # Write CSS
    (DOCS_DIR / "css" / "style.css").write_text(CSS)
    print("  wrote css/style.css", file=sys.stderr)

    # Write search index
    index = build_search_index(claims, scores_by_id)
    (DOCS_DIR / "data" / "claims.json").write_text(json.dumps(index, separators=(",", ":")))
    size_mb = (DOCS_DIR / "data" / "claims.json").stat().st_size / 1024 / 1024
    print(f"  wrote data/claims.json ({size_mb:.1f} MB)", file=sys.stderr)

    # Write claim pages
    total = len(claims)
    for i, c in enumerate(claims):
        sd = scores_by_id.get(c["id"], {})
        page = render_claim(c, sd, graph_edges)
        (DOCS_DIR / "claim" / f"{c['id']}.html").write_text(page)
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  claims: {i + 1:,}/{total:,}", file=sys.stderr)

    # Write index + corpus
    (DOCS_DIR / "index.html").write_text(render_index(claims, scores_by_id))
    print("  wrote index.html", file=sys.stderr)

    (DOCS_DIR / "corpus.html").write_text(render_corpus(claims, scores_by_id, results))
    print("  wrote corpus.html", file=sys.stderr)

    print(f"pcite: done — {total:,} claim pages in docs/", file=sys.stderr)


if __name__ == "__main__":
    main()
