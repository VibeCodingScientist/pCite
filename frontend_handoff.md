# pCite Frontend — Dev Handoff

Three changes, in order of priority. All are copy/layout additions — no data model changes required.

---

## 1. Homepage: add explainer paragraph above search box

**Where:** Between the tagline "Validation-weighted scientific claims" and the search input.

**What to add:**

> AI systems now generate scientific claims faster than citation metrics can distinguish them from instrument-measured results. pCite weights every citation by the physical grounding of its source — a mass spectrometer reading outweighs a text-derived prediction by 1,000×. Search any compound, disease, or relationship to see how claims rank when evidence quality, not attention, determines the score.

This should render as plain body text, same width as the search box, ~16px, muted colour (not black). No heading needed — it sits between tagline and search.

---

## 2. Homepage: add colour + score legend

**Where:** Directly below the search filters row (below "All predicates / All classes / Min replication"), before the claim count line.

**What to add — a single inline legend row:**

| Swatch | Label | Weight |
|---|---|---|
| Red | Physical | 10.0 |
| Orange | Replicated | 2.0 |
| Green | Curated | 0.5 |
| Light blue | AI-generated | 0.01 |

Render as a compact horizontal strip of four labelled colour dots, e.g.:

`● Physical (10.0)   ● Replicated (2.0)   ● Curated (0.5)   ● AI-generated (0.01)`

Small text (~12px), same line, left-aligned. This answers "what does the colour mean" and "what does the score mean" in one line.

---

## 3. Claim detail page: score tooltip

**Where:** Next to "Score: 364.00" on any claim detail page.

**What to add:** A small ⓘ icon that on hover shows:

> Score = Σ incoming citation weights. Each citation is weighted by the validation class of the source claim (Physical 10.0 · Curated 0.5 · AI 0.01) multiplied by the citation type (Replicates 1.5 · Supports 1.0 · Contradicts 0.8). A score of 364 means this claim received strong physical-tier citation support.

Plain tooltip, no modal needed. The existing score display is otherwise correct — this just gives a first-time visitor a way to understand what the number means without reading the paper.

---

## What does not need changing

- Claim triple display (subject → predicate → object) is clear
- Red bar as Physical signal works well
- Evidence chain with DOI + MTBLS accession is the right detail level
- Citation type breakdown (Supports / Contradicts / Extends / Applies / Replicates) with bar chart is strong — keep exactly as is
- Identifiers section at the bottom is correct
