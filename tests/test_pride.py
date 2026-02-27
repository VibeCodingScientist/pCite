"""tests/test_pride.py — PRIDE corpus + graded deposit quality tests (no network)."""

import math
import pytest
from pcite.models import (
    Claim, Entity, Paper, Predicate, ProvenanceEntry,
    StatisticalQualifiers, ValidationClass,
)
from pcite.validate import classify_provenance


def _claim(doi: str = "10.1000/test", deposit_id: str | None = None,
           metabo_id: str | None = None) -> Claim:
    return Claim(
        subject    = Entity(id="UNIPROT:P04637", name="TP53", type="protein"),
        predicate  = Predicate.INCREASES,
        object     = Entity(id="MESH:D009369", name="Neoplasms", type="disease"),
        qualifiers = StatisticalQualifiers(n=200, p_value=0.001, method="LC-MS/MS"),
        provenance = [ProvenanceEntry(
            doi=doi, deposit_id=deposit_id, metabo_id=metabo_id,
            validation_class=ValidationClass.AI_GENERATED,
        )],
    )


# deposit_id upgrades to Physical

def test_deposit_id_upgrades_to_physical():
    paper = Paper(doi="10.1000/x", title="T", abstract="A", deposit_id="PXD012345")
    entry = ProvenanceEntry(doi="10.1000/x")
    result = classify_provenance(entry, {"10.1000/x": paper})
    assert result.validation_class == ValidationClass.PHYSICAL
    assert result.deposit_id == "PXD012345"


# deposit_id backward compat — None default

def test_deposit_id_defaults_to_none_on_provenance():
    entry = ProvenanceEntry(doi="10.1000/x")
    assert entry.deposit_id is None


def test_deposit_id_defaults_to_none_on_paper():
    paper = Paper(doi="10.1000/x", title="T", abstract="A")
    assert paper.deposit_id is None


# metabo_id takes precedence over deposit_id

def test_metabo_id_takes_precedence():
    """When paper has both metabo_id and deposit_id, metabo_id wins."""
    paper = Paper(doi="10.1000/x", title="T", abstract="A",
                  metabo_id="MTBLS123", deposit_id="PXD012345")
    entry = ProvenanceEntry(doi="10.1000/x")
    result = classify_provenance(entry, {"10.1000/x": paper})
    assert result.validation_class == ValidationClass.PHYSICAL
    assert result.metabo_id == "MTBLS123"
    assert result.deposit_id is None  # metabo_id path doesn't set deposit_id


# PRIDE reference string parsing

def test_parse_pride_pmids_basic():
    from pride_corpus import _parse_pride_pmids
    refs = ["Smith et al. --pubMed:12345678--"]
    assert _parse_pride_pmids(refs) == ["12345678"]


def test_parse_pride_pmids_zero_filtered():
    from pride_corpus import _parse_pride_pmids
    refs = ["Unknown --pubMed:0--"]
    assert _parse_pride_pmids(refs) == []


def test_parse_pride_pmids_multiple():
    from pride_corpus import _parse_pride_pmids
    refs = [
        "Smith --pubMed:11111--",
        "Jones --pubMed:22222-- and --pubMed:33333--",
    ]
    assert _parse_pride_pmids(refs) == ["11111", "22222", "33333"]


def test_parse_pride_pmids_no_dashes():
    """PRIDE sometimes uses pubMed:12345 without dashes."""
    from pride_corpus import _parse_pride_pmids
    refs = ["pubMed:99999"]
    assert _parse_pride_pmids(refs) == ["99999"]


def test_parse_pride_pmids_empty():
    from pride_corpus import _parse_pride_pmids
    assert _parse_pride_pmids([]) == []
    assert _parse_pride_pmids(["no pmid here"]) == []


# Deposit quality normalization

def test_deposit_quality_range():
    from pride_corpus import compute_deposit_quality
    projects = {
        "PXD001": {"submissionType": "COMPLETE", "projectFileNames": ["a"],
                    "instruments": ["Orbitrap"]},
        "PXD002": {"submissionType": "COMPLETE", "projectFileNames": list("abcdefghij"),
                    "instruments": ["Q Exactive"]},
        "PXD003": {"submissionType": "COMPLETE", "projectFileNames": ["a", "b"],
                    "instruments": ["unknown"]},
    }
    quality = compute_deposit_quality(projects)
    assert len(quality) == 3
    for score in quality.values():
        assert 1.0 <= score <= 10.0


def test_deposit_quality_ordering():
    """More files + high-res instrument should score higher."""
    from pride_corpus import compute_deposit_quality
    projects = {
        "PXD_LOW": {"submissionType": "COMPLETE", "projectFileNames": ["a"],
                     "instruments": ["unknown"]},
        "PXD_HIGH": {"submissionType": "COMPLETE",
                      "projectFileNames": [f"f{i}" for i in range(100)],
                      "instruments": ["Orbitrap Exploris"]},
    }
    quality = compute_deposit_quality(projects)
    assert quality["PXD_HIGH"] > quality["PXD_LOW"]


def test_deposit_quality_empty():
    from pride_corpus import compute_deposit_quality
    assert compute_deposit_quality({}) == {}


# Deposit-first search terms

def test_deposit_first_terms_superset():
    """DEPOSIT_FIRST_TERMS includes all of SEARCH_TERMS plus extras."""
    from pride_corpus import SEARCH_TERMS, DEPOSIT_FIRST_TERMS
    for term in SEARCH_TERMS:
        assert term in DEPOSIT_FIRST_TERMS
    assert len(DEPOSIT_FIRST_TERMS) > len(SEARCH_TERMS)


def test_deposit_first_terms_has_broad_terms():
    """Deposit-first mode includes broader proteomics terms."""
    from pride_corpus import DEPOSIT_FIRST_TERMS
    broad = {"proteomics mass spectrometry", "quantitative proteomics",
             "phosphoproteomics", "DIA proteomics"}
    for term in broad:
        assert term in DEPOSIT_FIRST_TERMS


# Boundary investigation helpers

def test_boundary_subsample_at_coverage():
    """subsample_at_coverage returns correct approximate coverage."""
    import numpy as np
    from boundary_investigation import subsample_at_coverage

    physical = [{"claim_id": f"phys_{i}", "validation_class": "PhysicalMeasurement"}
                for i in range(100)]
    non_physical = [{"claim_id": f"text_{i}", "validation_class": "TextDerived"}
                    for i in range(100)]

    rng = np.random.default_rng(42)
    result = subsample_at_coverage(physical, non_physical, 0.20, rng)

    n_phys = sum(1 for r in result if r["validation_class"] == "PhysicalMeasurement")
    actual_cov = n_phys / len(result)
    assert 0.15 <= actual_cov <= 0.30  # within tolerance of rounding


def test_boundary_subsample_preserves_non_physical():
    """Non-physical records are always fully included."""
    import numpy as np
    from boundary_investigation import subsample_at_coverage

    physical = [{"claim_id": f"phys_{i}", "validation_class": "PhysicalMeasurement"}
                for i in range(200)]
    non_physical = [{"claim_id": f"text_{i}", "validation_class": "TextDerived"}
                    for i in range(50)]

    rng = np.random.default_rng(0)
    result = subsample_at_coverage(physical, non_physical, 0.10, rng)

    text_ids = {r["claim_id"] for r in result if r["validation_class"] == "TextDerived"}
    assert text_ids == {f"text_{i}" for i in range(50)}


def test_boundary_precision_at_k():
    """Local precision_at_k matches expected value."""
    from boundary_investigation import precision_at_k

    ranked = ["a", "b", "c", "d", "e"]
    validated = {"a", "c", "e"}
    assert precision_at_k(ranked, 5, validated) == pytest.approx(0.6)
    assert precision_at_k(ranked, 2, validated) == pytest.approx(0.5)


def test_boundary_ndcg_at_k():
    """NDCG@k returns 1.0 for perfect ranking."""
    from boundary_investigation import ndcg_at_k

    # Perfect ranking: all validated first
    ranked = ["a", "b", "c", "d", "e"]
    validated = {"a", "b"}
    assert ndcg_at_k(ranked, 5, validated) == pytest.approx(1.0)
