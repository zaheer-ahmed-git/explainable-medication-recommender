"""Fail-closed evidence contract for medication-ranking explanations.

The contract represents model evidence, safety decisions, external knowledge,
counterfactual replay, and uncertainty as separate typed records. It carries
references and version metadata only: raw clinical rows, source text, and
direct identifiers are explicitly forbidden.

This module is intentionally model- and framework-independent. It defines the
audit object that future Transformer, GNN, safety, retrieval, and language
components may populate; it does not generate or clinically validate an
explanation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

EXPLANATION_LEDGER_SCHEMA_VERSION = "medication-rank-evidence-ledger-v1"
CONSERVATION_TOLERANCE = 1e-8

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


class DecisionStatus(StrEnum):
    """Final status represented by one evidence ledger."""

    RANKED = "ranked"
    HARD_EXCLUDED = "hard_excluded"
    ABSTAINED = "abstained"


class MarginStage(StrEnum):
    """Pipeline stage at which the recorded score margin is defined."""

    FINAL_FEASIBLE = "final_feasible"
    PRE_SAFETY = "pre_safety"
    NONE = "none"


class EvidenceChannel(StrEnum):
    """Computational channel contributing to a recorded score margin."""

    TRANSFORMER = "transformer"
    GNN = "gnn"
    FUSION = "fusion"
    SOFT_RULE = "soft_rule"
    RETRIEVAL_RERANK = "retrieval_rerank"


class EvidenceDirection(StrEnum):
    """Candidate favoured by a signed contribution."""

    SUPPORTS_SELECTED = "supports_selected"
    SUPPORTS_COMPARATOR = "supports_comparator"
    NEUTRAL = "neutral"


class ConstraintStatus(StrEnum):
    """Result of replaying one hard safety constraint."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ConstraintAction(StrEnum):
    """Required action for a hard-constraint result."""

    ALLOW = "allow"
    EXCLUDE = "exclude"
    ABSTAIN = "abstain"


class KnowledgeSupportStatus(StrEnum):
    """Relationship between external knowledge and an explanation claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"


def _strict_mapping(
    payload: Any,
    *,
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object")
    optional = optional or set()
    keys = set(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    return payload


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _probability(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter and contain only letters, digits, "
            "'.', '_', ':', or '-' (maximum 128 characters)"
        )
    return value


def _reference(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 256 or "\n" in cleaned or "\r" in cleaned:
        raise ValueError(
            f"{name} must be a non-empty single-line reference of at most 256 characters"
        )
    return cleaned


def _references(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of references")
    result = tuple(
        _reference(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate references")
    return result


def _positive_rank(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc_timestamp(value: Any, *, name: str) -> str:
    text = _reference(value, name=name)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must include a UTC offset")
    return text


def _enum(enum_type: type[StrEnum], value: Any, *, name: str) -> StrEnum:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from error


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=CONSERVATION_TOLERANCE,
        abs_tol=CONSERVATION_TOLERANCE,
    )


@dataclass(frozen=True)
class ContributionInterval:
    """Uncertainty interval for one signed evidence contribution."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _finite(self.lower, name="interval.lower"))
        object.__setattr__(self, "upper", _finite(self.upper, name="interval.upper"))
        if self.lower > self.upper:
            raise ValueError("interval.lower must not exceed interval.upper")

    def to_dict(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="contribution_interval",
            required={"lower", "upper"},
        )
        return cls(lower=data["lower"], upper=data["upper"])


@dataclass(frozen=True)
class ComponentVersion:
    """Version and optional immutable artifact digest for one component."""

    name: str
    version: str
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, name="component.name"))
        object.__setattr__(
            self,
            "version",
            _reference(self.version, name=f"component[{self.name}].version"),
        )
        if self.artifact_digest is not None:
            if not isinstance(self.artifact_digest, str):
                raise TypeError("component.artifact_digest must be a string")
            digest = self.artifact_digest.lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    "component.artifact_digest must be a SHA-256 hex digest"
                )
            object.__setattr__(self, "artifact_digest", digest)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="component_version",
            required={"name", "version", "artifact_digest"},
        )
        return cls(**data)


@dataclass(frozen=True)
class DecisionRecord:
    """Final decision or abstention that the ledger explains."""

    status: DecisionStatus
    selected_candidate: str | None
    comparator_candidate: str | None
    margin: float | None
    margin_stage: MarginStage
    selected_rank: int | None = None
    comparator_rank: int | None = None
    rank_stability: float | None = None

    def __post_init__(self) -> None:
        status = _enum(DecisionStatus, self.status, name="decision.status")
        stage = _enum(MarginStage, self.margin_stage, name="decision.margin_stage")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "margin_stage", stage)

        if self.selected_candidate is not None:
            object.__setattr__(
                self,
                "selected_candidate",
                _identifier(
                    self.selected_candidate,
                    name="decision.selected_candidate",
                ),
            )
        if self.comparator_candidate is not None:
            object.__setattr__(
                self,
                "comparator_candidate",
                _identifier(
                    self.comparator_candidate,
                    name="decision.comparator_candidate",
                ),
            )
        if (
            self.selected_candidate is not None
            and self.selected_candidate == self.comparator_candidate
        ):
            raise ValueError("selected and comparator candidates must differ")

        margin = _optional_finite(self.margin, name="decision.margin")
        object.__setattr__(self, "margin", margin)
        if self.rank_stability is not None:
            object.__setattr__(
                self,
                "rank_stability",
                _probability(
                    self.rank_stability,
                    name="decision.rank_stability",
                ),
            )

        if status is DecisionStatus.RANKED:
            if self.selected_candidate is None or self.comparator_candidate is None:
                raise ValueError("ranked decisions require two candidate identifiers")
            if margin is None or margin <= 0:
                raise ValueError("ranked decisions require a positive final margin")
            if stage is not MarginStage.FINAL_FEASIBLE:
                raise ValueError("ranked decisions require margin_stage=final_feasible")
            selected_rank = _positive_rank(
                self.selected_rank,
                name="decision.selected_rank",
            )
            comparator_rank = _positive_rank(
                self.comparator_rank,
                name="decision.comparator_rank",
            )
            if selected_rank >= comparator_rank:
                raise ValueError("selected_rank must be better than comparator_rank")
        elif status is DecisionStatus.HARD_EXCLUDED:
            if self.selected_candidate is None or self.comparator_candidate is None:
                raise ValueError(
                    "hard exclusions require selected and excluded candidates"
                )
            if self.selected_rank is not None or self.comparator_rank is not None:
                raise ValueError(
                    "hard exclusions must not assign a rank to the excluded pair"
                )
            expected_stage = (
                MarginStage.PRE_SAFETY if margin is not None else MarginStage.NONE
            )
            if stage is not expected_stage:
                raise ValueError(
                    "hard exclusions require pre_safety for a recorded margin, "
                    "otherwise none"
                )
        else:
            if margin is not None or stage is not MarginStage.NONE:
                raise ValueError("abstained decisions must not record a score margin")
            if self.selected_rank is not None or self.comparator_rank is not None:
                raise ValueError("abstained decisions must not assign candidate ranks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "comparator_candidate": self.comparator_candidate,
            "margin": self.margin,
            "margin_stage": self.margin_stage.value,
            "selected_rank": self.selected_rank,
            "comparator_rank": self.comparator_rank,
            "rank_stability": self.rank_stability,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="decision",
            required={
                "status",
                "selected_candidate",
                "comparator_candidate",
                "margin",
                "margin_stage",
                "selected_rank",
                "comparator_rank",
                "rank_stability",
            },
        )
        return cls(**data)


@dataclass(frozen=True)
class EvidenceAtom:
    """One signed computational contribution with provenance and uncertainty."""

    evidence_id: str
    channel: EvidenceChannel
    direction: EvidenceDirection
    contribution: float
    provenance_refs: tuple[str, ...]
    parent_id: str | None = None
    interval: ContributionInterval | None = None
    stability: float | None = None
    necessity: float | None = None
    sufficiency_error: float | None = None
    detail_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _identifier(self.evidence_id, name="evidence.evidence_id"),
        )
        channel = _enum(EvidenceChannel, self.channel, name="evidence.channel")
        direction = _enum(EvidenceDirection, self.direction, name="evidence.direction")
        contribution = _finite(self.contribution, name="evidence.contribution")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "contribution", contribution)
        object.__setattr__(
            self,
            "provenance_refs",
            _references(self.provenance_refs, name="evidence.provenance_refs"),
        )
        if self.parent_id is not None:
            object.__setattr__(
                self,
                "parent_id",
                _identifier(self.parent_id, name="evidence.parent_id"),
            )
        expected_direction = (
            EvidenceDirection.SUPPORTS_SELECTED
            if contribution > 0
            else EvidenceDirection.SUPPORTS_COMPARATOR
            if contribution < 0
            else EvidenceDirection.NEUTRAL
        )
        if direction is not expected_direction:
            raise ValueError("evidence.direction must match the contribution sign")
        if self.interval is not None:
            if not isinstance(self.interval, ContributionInterval):
                raise TypeError("evidence.interval must be a ContributionInterval")
            if not self.interval.lower <= contribution <= self.interval.upper:
                raise ValueError("evidence.interval must contain the contribution")
        if self.stability is not None:
            object.__setattr__(
                self,
                "stability",
                _probability(self.stability, name="evidence.stability"),
            )
        object.__setattr__(
            self,
            "necessity",
            _optional_finite(self.necessity, name="evidence.necessity"),
        )
        sufficiency_error = _optional_finite(
            self.sufficiency_error,
            name="evidence.sufficiency_error",
        )
        if sufficiency_error is not None and sufficiency_error < 0:
            raise ValueError("evidence.sufficiency_error must be non-negative")
        object.__setattr__(self, "sufficiency_error", sufficiency_error)
        object.__setattr__(
            self,
            "detail_residual",
            _finite(self.detail_residual, name="evidence.detail_residual"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "channel": self.channel.value,
            "direction": self.direction.value,
            "contribution": self.contribution,
            "provenance_refs": list(self.provenance_refs),
            "parent_id": self.parent_id,
            "interval": None if self.interval is None else self.interval.to_dict(),
            "stability": self.stability,
            "necessity": self.necessity,
            "sufficiency_error": self.sufficiency_error,
            "detail_residual": self.detail_residual,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="evidence_atom",
            required={
                "evidence_id",
                "channel",
                "direction",
                "contribution",
                "provenance_refs",
                "parent_id",
                "interval",
                "stability",
                "necessity",
                "sufficiency_error",
                "detail_residual",
            },
        )
        interval = data["interval"]
        return cls(
            **{
                **data,
                "interval": None
                if interval is None
                else ContributionInterval.from_dict(interval),
            }
        )


@dataclass(frozen=True)
class SafetyConstraintCertificate:
    """Exact replay result for one hard safety rule."""

    certificate_id: str
    candidate_id: str
    rule_id: str
    rule_version: str
    status: ConstraintStatus
    action: ConstraintAction
    source_ref: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certificate_id",
            _identifier(self.certificate_id, name="constraint.certificate_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, name="constraint.candidate_id"),
        )
        object.__setattr__(
            self,
            "rule_id",
            _identifier(self.rule_id, name="constraint.rule_id"),
        )
        object.__setattr__(
            self,
            "rule_version",
            _reference(self.rule_version, name="constraint.rule_version"),
        )
        status = _enum(ConstraintStatus, self.status, name="constraint.status")
        action = _enum(ConstraintAction, self.action, name="constraint.action")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action", action)
        expected_action = {
            ConstraintStatus.PASS: ConstraintAction.ALLOW,
            ConstraintStatus.FAIL: ConstraintAction.EXCLUDE,
            ConstraintStatus.UNKNOWN: ConstraintAction.ABSTAIN,
        }[status]
        if action is not expected_action:
            raise ValueError("constraint.action must match the hard-constraint status")
        object.__setattr__(
            self,
            "source_ref",
            _reference(self.source_ref, name="constraint.source_ref"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _references(self.evidence_refs, name="constraint.evidence_refs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "candidate_id": self.candidate_id,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "status": self.status.value,
            "action": self.action.value,
            "source_ref": self.source_ref,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="safety_constraint_certificate",
            required={
                "certificate_id",
                "candidate_id",
                "rule_id",
                "rule_version",
                "status",
                "action",
                "source_ref",
                "evidence_refs",
            },
        )
        return cls(**data)


@dataclass(frozen=True)
class ExternalKnowledgeEvidence:
    """Versioned external support kept separate from model attribution."""

    knowledge_id: str
    claim_ref: str
    source_ref: str
    source_version: str
    support_status: KnowledgeSupportStatus
    effective_at: str | None = None
    rerank_evidence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "knowledge_id",
            _identifier(self.knowledge_id, name="knowledge.knowledge_id"),
        )
        object.__setattr__(
            self,
            "claim_ref",
            _reference(self.claim_ref, name="knowledge.claim_ref"),
        )
        object.__setattr__(
            self,
            "source_ref",
            _reference(self.source_ref, name="knowledge.source_ref"),
        )
        object.__setattr__(
            self,
            "source_version",
            _reference(self.source_version, name="knowledge.source_version"),
        )
        object.__setattr__(
            self,
            "support_status",
            _enum(
                KnowledgeSupportStatus,
                self.support_status,
                name="knowledge.support_status",
            ),
        )
        if self.effective_at is not None:
            object.__setattr__(
                self,
                "effective_at",
                _utc_timestamp(self.effective_at, name="knowledge.effective_at"),
            )
        if self.rerank_evidence_id is not None:
            object.__setattr__(
                self,
                "rerank_evidence_id",
                _identifier(
                    self.rerank_evidence_id,
                    name="knowledge.rerank_evidence_id",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "claim_ref": self.claim_ref,
            "source_ref": self.source_ref,
            "source_version": self.source_version,
            "support_status": self.support_status.value,
            "effective_at": self.effective_at,
            "rerank_evidence_id": self.rerank_evidence_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="external_knowledge_evidence",
            required={
                "knowledge_id",
                "claim_ref",
                "source_ref",
                "source_version",
                "support_status",
                "effective_at",
                "rerank_evidence_id",
            },
        )
        return cls(**data)


@dataclass(frozen=True)
class CounterfactualEvidence:
    """Validated model-rank counterfactual represented only by safe references."""

    counterfactual_id: str
    target_candidate: str
    cost: float
    edit_refs: tuple[str, ...]
    constraint_refs: tuple[str, ...]
    replay_succeeded: bool
    rank_flipped: bool
    clinically_permitted: bool
    causal_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterfactual_id",
            _identifier(
                self.counterfactual_id,
                name="counterfactual.counterfactual_id",
            ),
        )
        object.__setattr__(
            self,
            "target_candidate",
            _identifier(
                self.target_candidate,
                name="counterfactual.target_candidate",
            ),
        )
        cost = _finite(self.cost, name="counterfactual.cost")
        if cost < 0:
            raise ValueError("counterfactual.cost must be non-negative")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(
            self,
            "edit_refs",
            _references(self.edit_refs, name="counterfactual.edit_refs"),
        )
        object.__setattr__(
            self,
            "constraint_refs",
            _references(
                self.constraint_refs,
                name="counterfactual.constraint_refs",
            ),
        )
        for name in (
            "replay_succeeded",
            "rank_flipped",
            "clinically_permitted",
            "causal_claim",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"counterfactual.{name} must be a boolean")
        if not self.replay_succeeded or not self.rank_flipped:
            raise ValueError(
                "displayable counterfactuals must replay and flip the rank"
            )
        if not self.clinically_permitted:
            raise ValueError(
                "displayable counterfactuals must pass clinical constraints"
            )
        if self.causal_claim:
            raise ValueError(
                "v1 counterfactuals must not be represented as causal claims"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "target_candidate": self.target_candidate,
            "cost": self.cost,
            "edit_refs": list(self.edit_refs),
            "constraint_refs": list(self.constraint_refs),
            "replay_succeeded": self.replay_succeeded,
            "rank_flipped": self.rank_flipped,
            "clinically_permitted": self.clinically_permitted,
            "causal_claim": self.causal_claim,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="counterfactual_evidence",
            required={
                "counterfactual_id",
                "target_candidate",
                "cost",
                "edit_refs",
                "constraint_refs",
                "replay_succeeded",
                "rank_flipped",
                "clinically_permitted",
                "causal_claim",
            },
        )
        return cls(**data)


@dataclass(frozen=True)
class DataSafetyDeclaration:
    """Required declaration preventing clinical rows from entering the ledger."""

    contains_patient_rows: bool = False
    contains_source_text: bool = False
    contains_direct_identifiers: bool = False
    external_reference_resolution_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "contains_patient_rows",
            "contains_source_text",
            "contains_direct_identifiers",
            "external_reference_resolution_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"data_safety.{name} must be a boolean")
        if (
            self.contains_patient_rows
            or self.contains_source_text
            or self.contains_direct_identifiers
        ):
            raise ValueError(
                "explanation ledgers must not contain patient rows, source text, "
                "or direct identifiers"
            )
        if not self.external_reference_resolution_required:
            raise ValueError(
                "v1 ledgers must keep clinical content behind protected references"
            )

    def to_dict(self) -> dict[str, bool]:
        return {
            "contains_patient_rows": self.contains_patient_rows,
            "contains_source_text": self.contains_source_text,
            "contains_direct_identifiers": self.contains_direct_identifiers,
            "external_reference_resolution_required": (
                self.external_reference_resolution_required
            ),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        data = _strict_mapping(
            payload,
            context="data_safety",
            required={
                "contains_patient_rows",
                "contains_source_text",
                "contains_direct_identifiers",
                "external_reference_resolution_required",
            },
        )
        return cls(**data)


@dataclass(frozen=True)
class ExplanationLedger:
    """Versioned, evidence-conserving explanation packet."""

    decision: DecisionRecord
    components: tuple[ComponentVersion, ...]
    evidence: tuple[EvidenceAtom, ...]
    generated_at: str
    baseline_contribution: float = 0.0
    conservation_residual: float = 0.0
    hard_constraints: tuple[SafetyConstraintCertificate, ...] = ()
    external_knowledge: tuple[ExternalKnowledgeEvidence, ...] = ()
    counterfactuals: tuple[CounterfactualEvidence, ...] = ()
    warning_codes: tuple[str, ...] = ()
    data_safety: DataSafetyDeclaration = field(default_factory=DataSafetyDeclaration)
    schema_version: str = EXPLANATION_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPLANATION_LEDGER_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EXPLANATION_LEDGER_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.decision, DecisionRecord):
            raise TypeError("decision must be a DecisionRecord")
        if not isinstance(self.data_safety, DataSafetyDeclaration):
            raise TypeError("data_safety must be a DataSafetyDeclaration")

        object.__setattr__(
            self, "generated_at", _utc_timestamp(self.generated_at, name="generated_at")
        )
        object.__setattr__(
            self,
            "baseline_contribution",
            _finite(self.baseline_contribution, name="baseline_contribution"),
        )
        object.__setattr__(
            self,
            "conservation_residual",
            _finite(self.conservation_residual, name="conservation_residual"),
        )

        component_tuple = tuple(self.components)
        evidence_tuple = tuple(self.evidence)
        constraint_tuple = tuple(self.hard_constraints)
        knowledge_tuple = tuple(self.external_knowledge)
        counterfactual_tuple = tuple(self.counterfactuals)
        warning_tuple = _references(
            self.warning_codes,
            name="warning_codes",
            allow_empty=True,
        )
        object.__setattr__(self, "components", component_tuple)
        object.__setattr__(self, "evidence", evidence_tuple)
        object.__setattr__(self, "hard_constraints", constraint_tuple)
        object.__setattr__(self, "external_knowledge", knowledge_tuple)
        object.__setattr__(self, "counterfactuals", counterfactual_tuple)
        object.__setattr__(self, "warning_codes", warning_tuple)

        self._validate_types()
        self._validate_unique_ids()
        self._validate_evidence_hierarchy()
        self._validate_margin_conservation()
        self._validate_decision_constraints()
        self._validate_knowledge_links()
        self._validate_counterfactual_targets()

    def _validate_types(self) -> None:
        collections = (
            (self.components, ComponentVersion, "components"),
            (self.evidence, EvidenceAtom, "evidence"),
            (self.hard_constraints, SafetyConstraintCertificate, "hard_constraints"),
            (self.external_knowledge, ExternalKnowledgeEvidence, "external_knowledge"),
            (self.counterfactuals, CounterfactualEvidence, "counterfactuals"),
        )
        for values, expected_type, name in collections:
            if any(not isinstance(value, expected_type) for value in values):
                raise TypeError(f"{name} contains an invalid record type")
        if not self.components:
            raise ValueError("components must contain at least one versioned component")
        component_names = [component.name for component in self.components]
        if len(set(component_names)) != len(component_names):
            raise ValueError("component names must be unique")

    def _validate_unique_ids(self) -> None:
        identifiers = [atom.evidence_id for atom in self.evidence]
        identifiers.extend(
            certificate.certificate_id for certificate in self.hard_constraints
        )
        identifiers.extend(item.knowledge_id for item in self.external_knowledge)
        identifiers.extend(item.counterfactual_id for item in self.counterfactuals)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                "all evidence, constraint, knowledge, and counterfactual IDs must be unique"
            )

    def _validate_evidence_hierarchy(self) -> None:
        atoms = {atom.evidence_id: atom for atom in self.evidence}
        children: dict[str, list[EvidenceAtom]] = {
            evidence_id: [] for evidence_id in atoms
        }
        for atom in self.evidence:
            if atom.parent_id is None:
                continue
            if atom.parent_id == atom.evidence_id:
                raise ValueError("evidence atoms must not be their own parent")
            parent = atoms.get(atom.parent_id)
            if parent is None:
                raise ValueError(f"evidence parent {atom.parent_id!r} does not exist")
            if parent.channel is not atom.channel:
                raise ValueError("parent and child evidence must use the same channel")
            children[parent.evidence_id].append(atom)

        for atom in self.evidence:
            seen = {atom.evidence_id}
            current = atom
            while current.parent_id is not None:
                if current.parent_id in seen:
                    raise ValueError("evidence hierarchy contains a cycle")
                seen.add(current.parent_id)
                current = atoms[current.parent_id]

        for atom in self.evidence:
            atom_children = children[atom.evidence_id]
            if not atom_children:
                if not _close(atom.detail_residual, 0.0):
                    raise ValueError("leaf evidence must have zero detail_residual")
                continue
            detailed = sum(child.contribution for child in atom_children)
            detailed += atom.detail_residual
            if not _close(detailed, atom.contribution):
                raise ValueError(
                    f"child evidence does not conserve parent contribution for "
                    f"{atom.evidence_id!r}"
                )

    def _validate_margin_conservation(self) -> None:
        margin = self.decision.margin
        if margin is None:
            if self.evidence:
                raise ValueError(
                    "decisions without a margin must not contain attribution evidence"
                )
            if not _close(self.baseline_contribution, 0.0) or not _close(
                self.conservation_residual,
                0.0,
            ):
                raise ValueError(
                    "decisions without a margin require zero contribution terms"
                )
            return
        if not self.evidence:
            raise ValueError("decisions with a margin require attribution evidence")
        root_contribution = sum(
            atom.contribution for atom in self.evidence if atom.parent_id is None
        )
        reconstructed = (
            self.baseline_contribution + root_contribution + self.conservation_residual
        )
        if not _close(reconstructed, margin):
            raise ValueError(
                "root evidence, baseline, and residual do not conserve the decision margin"
            )

    def _validate_decision_constraints(self) -> None:
        decision = self.decision
        compared = {
            candidate
            for candidate in (
                decision.selected_candidate,
                decision.comparator_candidate,
            )
            if candidate is not None
        }
        if decision.status is DecisionStatus.RANKED:
            invalid = [
                certificate
                for certificate in self.hard_constraints
                if certificate.candidate_id in compared
                and certificate.status is not ConstraintStatus.PASS
            ]
            if invalid:
                raise ValueError(
                    "ranked candidates must not have failed or unknown hard constraints"
                )
        elif decision.status is DecisionStatus.HARD_EXCLUDED:
            matching = [
                certificate
                for certificate in self.hard_constraints
                if certificate.candidate_id == decision.comparator_candidate
                and certificate.status is ConstraintStatus.FAIL
                and certificate.action is ConstraintAction.EXCLUDE
            ]
            if not matching:
                raise ValueError(
                    "hard_excluded decisions require a failed certificate for the comparator"
                )
        elif not self.warning_codes:
            raise ValueError("abstained decisions require at least one warning code")

    def _validate_knowledge_links(self) -> None:
        atoms = {atom.evidence_id: atom for atom in self.evidence}
        for item in self.external_knowledge:
            if item.rerank_evidence_id is None:
                continue
            atom = atoms.get(item.rerank_evidence_id)
            if atom is None or atom.channel is not EvidenceChannel.RETRIEVAL_RERANK:
                raise ValueError(
                    "knowledge rerank links must reference retrieval_rerank evidence"
                )

    def _validate_counterfactual_targets(self) -> None:
        candidates = {
            candidate
            for candidate in (
                self.decision.selected_candidate,
                self.decision.comparator_candidate,
            )
            if candidate is not None
        }
        for item in self.counterfactuals:
            if item.target_candidate not in candidates:
                raise ValueError(
                    "counterfactual target must be part of the explained decision"
                )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest of the canonical ledger JSON."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the ledger."""

        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "decision": self.decision.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "evidence": [atom.to_dict() for atom in self.evidence],
            "baseline_contribution": self.baseline_contribution,
            "conservation_residual": self.conservation_residual,
            "hard_constraints": [
                certificate.to_dict() for certificate in self.hard_constraints
            ],
            "external_knowledge": [item.to_dict() for item in self.external_knowledge],
            "counterfactuals": [item.to_dict() for item in self.counterfactuals],
            "warning_codes": list(self.warning_codes),
            "data_safety": self.data_safety.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the ledger deterministically and reject non-finite values."""

        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> Self:
        """Validate and construct a ledger from a strict JSON-style mapping."""

        data = _strict_mapping(
            payload,
            context="explanation_ledger",
            required={
                "schema_version",
                "generated_at",
                "decision",
                "components",
                "evidence",
                "baseline_contribution",
                "conservation_residual",
                "hard_constraints",
                "external_knowledge",
                "counterfactuals",
                "warning_codes",
                "data_safety",
            },
        )
        list_fields = (
            "components",
            "evidence",
            "hard_constraints",
            "external_knowledge",
            "counterfactuals",
            "warning_codes",
        )
        for name in list_fields:
            if not isinstance(data[name], list):
                raise TypeError(f"explanation_ledger.{name} must be a JSON array")
        return cls(
            schema_version=data["schema_version"],
            generated_at=data["generated_at"],
            decision=DecisionRecord.from_dict(data["decision"]),
            components=tuple(
                ComponentVersion.from_dict(item) for item in data["components"]
            ),
            evidence=tuple(EvidenceAtom.from_dict(item) for item in data["evidence"]),
            baseline_contribution=data["baseline_contribution"],
            conservation_residual=data["conservation_residual"],
            hard_constraints=tuple(
                SafetyConstraintCertificate.from_dict(item)
                for item in data["hard_constraints"]
            ),
            external_knowledge=tuple(
                ExternalKnowledgeEvidence.from_dict(item)
                for item in data["external_knowledge"]
            ),
            counterfactuals=tuple(
                CounterfactualEvidence.from_dict(item)
                for item in data["counterfactuals"]
            ),
            warning_codes=tuple(data["warning_codes"]),
            data_safety=DataSafetyDeclaration.from_dict(data["data_safety"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Validate and construct a ledger from JSON text."""

        if not isinstance(payload, str):
            raise TypeError("payload must be JSON text")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("payload is not valid JSON") from error
        return cls.from_dict(data)
