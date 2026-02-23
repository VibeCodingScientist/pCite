"""
ncite.models — the shared contract

Three axioms encoded as design decisions:

  1. Claims are universal. ID = hash(subject, predicate, object) — never DOI.
     Same assertion in 50 papers = one node with 50 provenance entries.

  2. Replication is structural. len(claim.provenance) IS the replication count.
     No classifier. No edge type. Structural fact.

  3. CI pipeline is a validator. Statistically impossible claims cannot be
     constructed. Pydantic raises at creation, not at runtime.
"""

from __future__ import annotations
import hashlib, math
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, computed_field, model_validator


class ValidationClass(str, Enum):
    PHYSICAL     = "PhysicalMeasurement"   # raw instrument data in public repo
    CLINICAL     = "ClinicalObservation"   # EHR / IRB-verified patient data
    REPLICATED   = "Replicated"            # ≥3 independent CURATED sources
    CURATED      = "HumanCurated"          # structured deposit, expert record
    AI_GENERATED = "AIGenerated"           # synthesised from literature text
    HYPOTHESIS   = "Hypothesis"            # proposed, untested

VALIDATION_WEIGHT: dict[ValidationClass, float] = {
    ValidationClass.PHYSICAL:     10.0,  # categorically different — instrument data
    ValidationClass.CLINICAL:      4.0,  # physical patient outcomes
    ValidationClass.REPLICATED:    2.0,  # only meaningful if physically anchored
    ValidationClass.CURATED:       0.5,  # human agreement on text
    ValidationClass.AI_GENERATED:  0.01, # near-zero in AI-flood world
    ValidationClass.HYPOTHESIS:    0.0,
}


class Predicate(str, Enum):
    """
    Closed vocabulary. Every metabolomics claim maps to exactly one predicate.
    Closed = queryable and comparable across papers.
    Free strings fragment the graph — "increases", "is elevated in", "upregulates"
    would be three different predicates for the same relationship.
    """
    INCREASES        = "increases"
    DECREASES        = "decreases"
    IS_BIOMARKER_FOR = "is_biomarker_for"
    DISTINGUISHES    = "distinguishes"
    PREDICTS         = "predicts"
    INHIBITS         = "inhibits"
    ACTIVATES        = "activates"
    IS_METABOLITE_OF = "is_metabolite_of"
    CORRELATES_WITH  = "correlates_with"
    CAUSES           = "causes"
    TREATS           = "treats"


class NCiteType(str, Enum):
    SUPPORTS    = "supports"
    EXTENDS     = "extends"
    REPLICATES  = "replicates"   # explicit replication relationship — highest weight
    CONTRADICTS = "contradicts"
    APPLIES     = "applies"

NCITE_WEIGHT: dict[NCiteType, float] = {
    NCiteType.SUPPORTS:    1.0,
    NCiteType.EXTENDS:     1.2,
    NCiteType.REPLICATES:  1.5,  # replication earns more than novelty
    NCiteType.CONTRADICTS: 0.8,  # counter-evidence still earns credit
    NCiteType.APPLIES:     0.6,
}


class Entity(BaseModel):
    id:   str   # "HMDB:HMDB0000122" | "CHEBI:15422" | "MESH:D003920"
    name: str
    type: str   # compound | disease | gene | organism | process | pathway

    @property
    def uri(self) -> str:
        from urllib.parse import quote
        return f"https://identifiers.org/{quote(self.id, safe=':')}"


class StatisticalQualifiers(BaseModel):
    """
    Typed qualifiers that double as a CI pipeline.
    Inconsistent statistics raise at construction — not at runtime, not later.
    This is the entire validation pipeline in one model.
    """
    n:                   Optional[int]                 = None
    p_value:             Optional[float]               = None
    effect_size:         Optional[float]               = None
    confidence_interval: Optional[tuple[float, float]] = None
    fold_change:         Optional[float]               = None
    method:              Optional[str]                 = None

    @model_validator(mode="after")
    def check_consistency(self) -> "StatisticalQualifiers":
        if self.p_value is not None and not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"p={self.p_value} outside [0, 1]")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if lo >= hi:
                raise ValueError(f"CI [{lo}, {hi}]: lower ≥ upper")
        if self.n is not None and self.n < 1:
            raise ValueError(f"n={self.n}: must be ≥ 1")
        return self


class ProvenanceEntry(BaseModel):
    """
    Per-paper evidence record. One claim carries one entry per paper that made it.
    metabo_id is Phase 3's anchor: it links to actual raw instrument files.
    It is stored now, used for classification now, and becomes the measurement
    registry foundation later — without changing a line of code.
    """
    doi:              str
    validation_class: ValidationClass = ValidationClass.AI_GENERATED
    metabo_id:        Optional[str]   = None


class Claim(BaseModel):
    subject:    Entity
    predicate:  Predicate
    object:     Entity
    qualifiers: StatisticalQualifiers
    provenance: list[ProvenanceEntry] = []

    @computed_field
    @property
    def id(self) -> str:
        """
        Content-addressed from assertion alone — never from DOI.
        Same claim in 50 papers = one node with 50 provenance entries.
        This makes replication structural rather than classified.
        """
        payload = f"{self.subject.id}:{self.predicate.value}:{self.object.id}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @computed_field
    @property
    def replication_count(self) -> int:
        """Number of independent papers making this assertion. Not a classification. A count."""
        return len(self.provenance)

    @computed_field
    @property
    def validation_class(self) -> ValidationClass:
        """Best validation class across provenance entries. Ratchets upward on merge."""
        if not self.provenance:
            return ValidationClass.HYPOTHESIS
        return max(
            self.provenance,
            key=lambda p: VALIDATION_WEIGHT[p.validation_class]
        ).validation_class

    @computed_field
    @property
    def base_weight(self) -> float:
        """
        Evidence quality × replication count, log-scaled.

        The 1000× gap between PHYSICAL (10.0) and AI_GENERATED (0.01)
        is the paper's central claim made quantitative: a claim anchored
        to instrument data is categorically, not incrementally, more
        trustworthy than a claim that exists only as text.

        PHYSICAL, 8 papers:     10.0 × log₂(9) ≈ 31.7
        AI_GENERATED, 1 paper:  0.01 × log₂(2) =  0.01

        3170× difference. In a world of infinite AI-generated text,
        this asymmetry is not a design choice — it is an epistemological fact.
        """
        return VALIDATION_WEIGHT[self.validation_class] * math.log2(self.replication_count + 1)

    def merge(self, other: "Claim") -> "Claim":
        """
        Same assertion, different paper. Accumulate provenance.
        Best-wins per DOI: if a DOI appears in both, keep the higher-trust entry.
        Validation class updates automatically via computed field — no bookkeeping.
        """
        assert self.id == other.id
        by_doi = {p.doi: p for p in self.provenance}
        for p in other.provenance:
            existing = by_doi.get(p.doi)
            if not existing or (
                VALIDATION_WEIGHT[p.validation_class] > VALIDATION_WEIGHT[existing.validation_class]
            ):
                by_doi[p.doi] = p
        return self.model_copy(update={"provenance": list(by_doi.values())})


class NCite(BaseModel):
    source_id:     str
    target_id:     str
    type:          NCiteType
    source_weight: float

    @computed_field
    @property
    def weight(self) -> float:
        """
        Edge weight = citation type × source trustworthiness.

        Replication from a physically-validated, 8× replicated claim:  1.5 × 31.7 ≈ 47.6
        Support from an AI-generated singleton:                         1.0 × 0.01  =  0.01

        4760× difference. No manual scoring. Entirely from the data model.
        """
        return NCITE_WEIGHT[self.type] * self.source_weight


class Paper(BaseModel):
    doi:                   str
    pmid:                  Optional[str] = None
    title:                 str
    abstract:              str
    full_text:             Optional[str] = None
    metabo_id:             Optional[str] = None
    traditional_citations: int           = 0
    year:                  Optional[int] = None
