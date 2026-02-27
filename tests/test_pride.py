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
