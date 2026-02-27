"""
pcite.validate

Two jobs:
  1. Upgrade ProvenanceEntry.validation_class based on checkable facts.
  2. Serialise each claim as a W3C TriG nanopublication.

The classifier is a pure function — testable with no mocks, no network.
Every rule has a comment explaining the epistemological reason.

Output: data/claims.jsonl (updated) + data/nanopubs/*.trig
Run:    python -m pcite.validate
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import sys, rdflib
from rdflib import RDF, XSD, Literal, Namespace, URIRef
from rdflib.graph import ConjunctiveGraph
from pcite.models import Claim, Paper, ProvenanceEntry, ValidationClass, VALIDATION_WEIGHT


def _safe_uri(url: str) -> URIRef:
    """Percent-encode spaces and other invalid chars for RDF URIs."""
    return URIRef(quote(url, safe=":/?#[]@!$&'()*+,;=-._~%"))

DATA_CLAIMS = Path("data/claims.jsonl")
NANOPUBS    = Path("data/nanopubs")

NP   = Namespace("https://w3id.org/np/")
NPX  = Namespace("http://purl.org/nanopub/x/")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCT  = Namespace("http://purl.org/dc/terms/")
BASE = Namespace("https://pcite.org/np/")


def classify_provenance(entry: ProvenanceEntry, papers: dict[str, Paper]) -> ProvenanceEntry:
    """
    Upgrade a ProvenanceEntry's ValidationClass based on checkable external facts.
    First match wins.

    PHYSICAL: paper has a MetaboLights accession.
      Reason: raw instrument files exist in a public repository. The claim is
      anchored to a physical measurement that anyone can download and reanalyse.

    CURATED: paper has a structured abstract record.
      Reason: a human bibliographer verified the record. Weaker than physical
      data but stronger than AI text synthesis.

    AI_GENERATED: no verifiable external anchor.
      Reason: the claim exists only as text. May be accurate but unverifiable.

    Replication is NOT classified here. It emerges from provenance count in upgrade_claim().
    """
    paper = papers.get(entry.doi)
    if not paper:
        return entry
    if paper.metabo_id:
        return entry.model_copy(update={
            "validation_class": ValidationClass.PHYSICAL,
            "metabo_id": paper.metabo_id,
        })
    if paper.deposit_id:
        return entry.model_copy(update={
            "validation_class": ValidationClass.PHYSICAL,
            "deposit_id": paper.deposit_id,
        })
    if paper.abstract:
        return entry.model_copy(update={"validation_class": ValidationClass.CURATED})
    return entry


def upgrade_claim(claim: Claim, papers: dict[str, Paper]) -> Claim:
    """
    Upgrade all entries, then apply structural replication rule:
    ≥3 independent CURATED-or-better sources → REPLICATED.
    """
    upgraded = [classify_provenance(p, papers) for p in claim.provenance]
    trusted  = sum(
        1 for p in upgraded
        if VALIDATION_WEIGHT[p.validation_class] >= VALIDATION_WEIGHT[ValidationClass.CURATED]
    )
    if trusted >= 3:
        upgraded = [
            p.model_copy(update={"validation_class": ValidationClass.REPLICATED})
            if VALIDATION_WEIGHT[p.validation_class] >= VALIDATION_WEIGHT[ValidationClass.CURATED]
            else p
            for p in upgraded
        ]
    return claim.model_copy(update={"provenance": upgraded})


def to_nanopub(claim: Claim) -> ConjunctiveGraph:
    """
    Four named graphs per W3C nanopublication spec.
    Assertion = exactly one RDF triple.
    Provenance links each paper + any MetaboLights deposit (Phase 3 anchor).
    """
    g, base = ConjunctiveGraph(), BASE[claim.id + "/"]
    head    = rdflib.Graph(g.store, identifier=base + "head")
    assn    = rdflib.Graph(g.store, identifier=base + "assertion")
    prov    = rdflib.Graph(g.store, identifier=base + "provenance")
    pubinfo = rdflib.Graph(g.store, identifier=base + "pubinfo")

    head.add((base, RDF.type,             NP.Nanopublication))
    head.add((base, NP.hasAssertion,      base + "assertion"))
    head.add((base, NP.hasProvenance,     base + "provenance"))
    head.add((base, NP.hasPublicationInfo, base + "pubinfo"))

    assn.add((_safe_uri(claim.subject.uri),
              BASE[f"predicate/{claim.predicate.value}"],
              _safe_uri(claim.object.uri)))

    assn_uri = base + "assertion"
    for entry in claim.provenance:
        doi_uri = _safe_uri(f"https://doi.org/{entry.doi}")
        prov.add((assn_uri, PROV.wasDerivedFrom, doi_uri))
        if entry.metabo_id:
            prov.add((doi_uri, PROV.hadPrimarySource,
                      _safe_uri(f"https://www.ebi.ac.uk/metabolights/{entry.metabo_id}")))
        elif entry.deposit_id:
            prov.add((doi_uri, PROV.hadPrimarySource,
                      _safe_uri(f"https://www.ebi.ac.uk/pride/archive/projects/{entry.deposit_id}")))
    prov.add((assn_uri, NPX.validationClass,
              Literal(claim.validation_class.value)))
    prov.add((assn_uri, NPX.replicationCount,
              Literal(claim.replication_count, datatype=XSD.integer)))

    pubinfo.add((base, DCT.created,
                 Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)))
    pubinfo.add((base, DCT.publisher, URIRef("https://pcite.org")))
    return g


def process_claims(
    claims_path: Path | None = None,
    nanopubs_path: Path | None = None,
    papers_loader=None,
    claims_loader=None,
) -> int:
    claims_path = claims_path or DATA_CLAIMS
    nanopubs_path = nanopubs_path or NANOPUBS
    if papers_loader is None:
        from pcite.corpus import load_papers
        papers_loader = load_papers
    if claims_loader is None:
        from pcite.extract import load_claims
        claims_loader = load_claims
    papers, claims = {p.doi: p for p in papers_loader()}, claims_loader()
    nanopubs_path.mkdir(parents=True, exist_ok=True)
    upgraded = [upgrade_claim(c, papers) for c in claims]
    for c in upgraded:
        to_nanopub(c).serialize(
            destination=str(nanopubs_path / f"{c.id}.trig"), format="trig"
        )
    with claims_path.open("w") as f:
        for c in upgraded:
            f.write(c.model_dump_json() + "\n")
    by_class: dict[str, int] = {}
    for c in upgraded:
        by_class[c.validation_class.value] = by_class.get(c.validation_class.value, 0) + 1
    for vc, n in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {vc:<25} {n:>5}", file=sys.stderr)
    return len(upgraded)


if __name__ == "__main__":
    process_claims()
