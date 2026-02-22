"""tests/test_all.py"""

import pytest
from ncite.models import (
    Claim, Entity, NCite, NCiteType, Paper, Predicate, ProvenanceEntry,
    StatisticalQualifiers, ValidationClass, VALIDATION_WEIGHT, NCITE_WEIGHT,
)


def claim(doi: str = "10.1000/test") -> Claim:
    return Claim(
        subject    = Entity(id="HMDB:HMDB0000122", name="Glutamine", type="compound"),
        predicate  = Predicate.INCREASES,
        object     = Entity(id="MESH:D015212", name="Colitis, Ulcerative", type="disease"),
        qualifiers = StatisticalQualifiers(n=150, p_value=0.001, method="LC-MS"),
        provenance = [ProvenanceEntry(doi=doi, validation_class=ValidationClass.CURATED)],
    )


# Axiom 1: Claims are universal

def test_claim_id_excludes_doi():
    assert claim("10.1000/a").id == claim("10.1000/b").id

def test_merge_accumulates_provenance():
    assert len(claim("10.1000/a").merge(claim("10.1000/b")).provenance) == 2

def test_merge_idempotent_on_same_doi():
    c = claim("10.1000/a")
    assert len(c.merge(c).provenance) == 1

def test_merge_keeps_best_validation_class():
    c_phys = Claim(
        subject=claim().subject, predicate=claim().predicate,
        object=claim().object,   qualifiers=claim().qualifiers,
        provenance=[ProvenanceEntry(doi="10.1000/b",
                                    validation_class=ValidationClass.PHYSICAL,
                                    metabo_id="MTBLS123")],
    )
    assert claim("10.1000/a").merge(c_phys).validation_class == ValidationClass.PHYSICAL


# Axiom 2: Replication is structural

def test_replication_count_is_provenance_length():
    c = claim().merge(claim("10.1000/b")).merge(claim("10.1000/c"))
    assert c.replication_count == 3

def test_base_weight_increases_with_replication():
    assert claim().merge(claim("10.1000/b")).base_weight > claim().base_weight

def test_log_scaling_diminishes():
    """8th replication adds less than 2nd."""
    c = claim()
    for i in range(7):
        c = c.merge(claim(f"10.1000/{i}"))
    d2 = claim().merge(claim("10.1000/x")).base_weight - claim().base_weight
    d8 = c.merge(claim("10.1000/y")).base_weight - c.base_weight
    assert d2 > d8


# Axiom 3: CI pipeline is a validator

def test_invalid_p_value_raises():
    with pytest.raises(Exception): StatisticalQualifiers(p_value=1.5)

def test_invalid_ci_raises():
    with pytest.raises(Exception): StatisticalQualifiers(confidence_interval=(0.8, 0.3))

def test_zero_n_raises():
    with pytest.raises(Exception): StatisticalQualifiers(n=0)

def test_valid_qualifiers_pass():
    assert StatisticalQualifiers(n=100, p_value=0.05).n == 100


# Weight constants

def test_physical_is_highest():
    assert VALIDATION_WEIGHT[ValidationClass.PHYSICAL] == max(VALIDATION_WEIGHT.values())

def test_hypothesis_is_zero():
    assert VALIDATION_WEIGHT[ValidationClass.HYPOTHESIS] == 0.0

def test_replication_ncite_is_highest():
    assert NCITE_WEIGHT[NCiteType.REPLICATES] == max(NCITE_WEIGHT.values())

def test_edge_weight_formula():
    n = NCite(source_id="a", target_id="b", type=NCiteType.REPLICATES, source_weight=5.0)
    assert n.weight == pytest.approx(7.5)


# Validate: pure function tests (no network)

def test_metabo_deposit_upgrades_to_physical():
    from ncite.validate import classify_provenance
    paper = Paper(doi="10.1000/x", title="T", abstract="A", metabo_id="MTBLS123")
    entry = ProvenanceEntry(doi="10.1000/x")
    assert classify_provenance(entry, {"10.1000/x": paper}).validation_class \
           == ValidationClass.PHYSICAL

def test_abstract_upgrades_to_curated():
    from ncite.validate import classify_provenance
    paper = Paper(doi="10.1000/x", title="T", abstract="A")
    entry = ProvenanceEntry(doi="10.1000/x")
    assert classify_provenance(entry, {"10.1000/x": paper}).validation_class \
           == ValidationClass.CURATED

def test_three_curated_become_replicated():
    from ncite.validate import upgrade_claim
    c      = claim("10.1000/a").merge(claim("10.1000/b")).merge(claim("10.1000/c"))
    papers = {p.doi: Paper(doi=p.doi, title="T", abstract="A") for p in c.provenance}
    assert upgrade_claim(c, papers).validation_class == ValidationClass.REPLICATED

def test_two_curated_stay_curated():
    from ncite.validate import upgrade_claim
    c      = claim("10.1000/a").merge(claim("10.1000/b"))
    papers = {p.doi: Paper(doi=p.doi, title="T", abstract="A") for p in c.provenance}
    assert upgrade_claim(c, papers).validation_class == ValidationClass.CURATED


# Graph: scoring

def test_isolated_node_scores_zero():
    import networkx as nx
    from ncite.graph import compute_ncite_scores
    G = nx.DiGraph()
    G.add_node("a")
    assert compute_ncite_scores(G)["a"] == 0.0

def test_score_sums_incoming_weights():
    import networkx as nx
    from ncite.graph import compute_ncite_scores
    G = nx.DiGraph()
    for n in ["a","b","c"]: G.add_node(n)
    G.add_edge("b", "a", weight=2.0)
    G.add_edge("c", "a", weight=3.0)
    assert compute_ncite_scores(G)["a"] == pytest.approx(5.0)


# Evaluate: metrics

def test_ndcg_perfect_ranking():
    from ncite.evaluate import ndcg_at_k
    records = [{"claim_id": str(i), "validation_class": "PhysicalMeasurement",
                "ncite_score": float(10-i)} for i in range(10)]
    assert ndcg_at_k(records, k=10)["ndcg_ncite"] == pytest.approx(1.0)

def test_precision_in_range():
    from ncite.evaluate import precision_at_k
    records = [{"claim_id": str(i),
                "validation_class": "PhysicalMeasurement" if i < 10 else "AIGenerated",
                "ncite_score": float(i)} for i in range(100)]
    r = precision_at_k(records, k=10)
    assert 0.0 <= r["precision_ncite"] <= 1.0
