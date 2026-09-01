"""Synthetic tests for the medication explanation evidence contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.explainability import (
    ComponentVersion,
    ConstraintAction,
    ConstraintStatus,
    ContributionInterval,
    CounterfactualEvidence,
    DataSafetyDeclaration,
    DecisionRecord,
    DecisionStatus,
    EvidenceAtom,
    EvidenceChannel,
    EvidenceDirection,
    ExplanationLedger,
    ExternalKnowledgeEvidence,
    KnowledgeSupportStatus,
    MarginStage,
    SafetyConstraintCertificate,
)


def _components() -> tuple[ComponentVersion, ...]:
    return (
        ComponentVersion(name="transformer", version="synthetic-v1"),
        ComponentVersion(name="gnn", version="synthetic-v1"),
        ComponentVersion(name="safety_rules", version="synthetic-v1"),
        ComponentVersion(name="knowledge_retrieval", version="synthetic-v1"),
    )


def _pass_certificate(candidate_id: str, suffix: str) -> SafetyConstraintCertificate:
    return SafetyConstraintCertificate(
        certificate_id=f"constraint.{suffix}",
        candidate_id=candidate_id,
        rule_id="rule.allergy",
        rule_version="synthetic-v1",
        status=ConstraintStatus.PASS,
        action=ConstraintAction.ALLOW,
        source_ref="ruleset:allergy:v1",
        evidence_refs=(f"profile:allergy:{suffix}",),
    )


def _ranked_ledger() -> ExplanationLedger:
    evidence = (
        EvidenceAtom(
            evidence_id="transformer.root",
            channel=EvidenceChannel.TRANSFORMER,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.8,
            provenance_refs=("model:transformer:output",),
            interval=ContributionInterval(lower=0.7, upper=0.9),
            stability=0.92,
            necessity=0.61,
            sufficiency_error=0.04,
        ),
        EvidenceAtom(
            evidence_id="transformer.diagnosis",
            channel=EvidenceChannel.TRANSFORMER,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.5,
            provenance_refs=("profile:diagnosis:synthetic",),
            parent_id="transformer.root",
        ),
        EvidenceAtom(
            evidence_id="transformer.laboratory",
            channel=EvidenceChannel.TRANSFORMER,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.3,
            provenance_refs=("profile:laboratory:synthetic",),
            parent_id="transformer.root",
        ),
        EvidenceAtom(
            evidence_id="gnn.root",
            channel=EvidenceChannel.GNN,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.4,
            provenance_refs=("graph:subgraph:synthetic",),
            detail_residual=0.05,
        ),
        EvidenceAtom(
            evidence_id="gnn.path",
            channel=EvidenceChannel.GNN,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.35,
            provenance_refs=("graph:path:diagnosis-medication",),
            parent_id="gnn.root",
        ),
        EvidenceAtom(
            evidence_id="soft_rule.root",
            channel=EvidenceChannel.SOFT_RULE,
            direction=EvidenceDirection.SUPPORTS_COMPARATOR,
            contribution=-0.1,
            provenance_refs=("ruleset:preference:v1",),
        ),
    )
    return ExplanationLedger(
        generated_at="2026-08-31T12:00:00Z",
        decision=DecisionRecord(
            status=DecisionStatus.RANKED,
            selected_candidate="candidate_a",
            comparator_candidate="candidate_b",
            margin=1.0,
            margin_stage=MarginStage.FINAL_FEASIBLE,
            selected_rank=1,
            comparator_rank=2,
            rank_stability=0.84,
        ),
        components=_components(),
        evidence=evidence,
        conservation_residual=-0.1,
        hard_constraints=(
            _pass_certificate("candidate_a", "selected"),
            _pass_certificate("candidate_b", "comparator"),
        ),
        external_knowledge=(
            ExternalKnowledgeEvidence(
                knowledge_id="knowledge.guideline",
                claim_ref="claim:selection:synthetic",
                source_ref="guideline:synthetic:section-1",
                source_version="synthetic-v1",
                support_status=KnowledgeSupportStatus.SUPPORTS,
                effective_at="2026-01-01T00:00:00Z",
            ),
        ),
        counterfactuals=(
            CounterfactualEvidence(
                counterfactual_id="counterfactual.rank_flip",
                target_candidate="candidate_b",
                cost=0.25,
                edit_refs=("edit:synthetic-laboratory-threshold",),
                constraint_refs=("constraint:plausibility:v1",),
                replay_succeeded=True,
                rank_flipped=True,
                clinically_permitted=True,
            ),
        ),
    )


def test_ranked_ledger_round_trips_with_stable_digest() -> None:
    ledger = _ranked_ledger()

    restored = ExplanationLedger.from_json(ledger.to_json())

    assert restored == ledger
    assert restored.digest == ledger.digest
    assert restored.to_json() == ledger.to_json()
    assert restored.data_safety == DataSafetyDeclaration()


def test_root_evidence_must_conserve_decision_margin() -> None:
    ledger = _ranked_ledger()

    with pytest.raises(ValueError, match="conserve the decision margin"):
        replace(ledger, conservation_residual=0.0)

    with pytest.raises(ValueError, match="require attribution evidence"):
        replace(
            ledger,
            evidence=(),
            baseline_contribution=ledger.decision.margin,
            conservation_residual=0.0,
        )


def test_child_evidence_must_conserve_parent_contribution() -> None:
    ledger = _ranked_ledger()
    changed = tuple(
        replace(atom, contribution=0.4)
        if atom.evidence_id == "transformer.diagnosis"
        else atom
        for atom in ledger.evidence
    )

    with pytest.raises(ValueError, match="child evidence does not conserve"):
        replace(ledger, evidence=changed)


def test_evidence_direction_and_interval_must_match_contribution() -> None:
    with pytest.raises(ValueError, match="direction must match"):
        EvidenceAtom(
            evidence_id="evidence.invalid_direction",
            channel=EvidenceChannel.TRANSFORMER,
            direction=EvidenceDirection.SUPPORTS_COMPARATOR,
            contribution=0.2,
            provenance_refs=("model:synthetic",),
        )

    with pytest.raises(ValueError, match="interval must contain"):
        EvidenceAtom(
            evidence_id="evidence.invalid_interval",
            channel=EvidenceChannel.GNN,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.2,
            provenance_refs=("graph:synthetic",),
            interval=ContributionInterval(lower=0.3, upper=0.4),
        )


def test_hard_exclusion_requires_replayed_failure_certificate() -> None:
    decision = DecisionRecord(
        status=DecisionStatus.HARD_EXCLUDED,
        selected_candidate="candidate_a",
        comparator_candidate="candidate_b",
        margin=0.2,
        margin_stage=MarginStage.PRE_SAFETY,
    )
    evidence = (
        EvidenceAtom(
            evidence_id="transformer.pre_safety",
            channel=EvidenceChannel.TRANSFORMER,
            direction=EvidenceDirection.SUPPORTS_SELECTED,
            contribution=0.2,
            provenance_refs=("model:pre-safety:synthetic",),
        ),
    )

    with pytest.raises(ValueError, match="failed certificate"):
        ExplanationLedger(
            generated_at="2026-08-31T12:00:00Z",
            decision=decision,
            components=_components(),
            evidence=evidence,
        )

    certificate = SafetyConstraintCertificate(
        certificate_id="constraint.failed_interaction",
        candidate_id="candidate_b",
        rule_id="rule.drug_interaction",
        rule_version="synthetic-v1",
        status=ConstraintStatus.FAIL,
        action=ConstraintAction.EXCLUDE,
        source_ref="ruleset:interaction:v1",
        evidence_refs=("profile:medication:synthetic",),
    )
    ledger = ExplanationLedger(
        generated_at="2026-08-31T12:00:00Z",
        decision=decision,
        components=_components(),
        evidence=evidence,
        hard_constraints=(certificate,),
    )

    assert ledger.hard_constraints == (certificate,)


def test_abstention_requires_warning_and_has_no_attribution() -> None:
    decision = DecisionRecord(
        status=DecisionStatus.ABSTAINED,
        selected_candidate=None,
        comparator_candidate=None,
        margin=None,
        margin_stage=MarginStage.NONE,
    )

    with pytest.raises(ValueError, match="at least one warning"):
        ExplanationLedger(
            generated_at="2026-08-31T12:00:00Z",
            decision=decision,
            components=_components(),
            evidence=(),
        )

    ledger = ExplanationLedger(
        generated_at="2026-08-31T12:00:00Z",
        decision=decision,
        components=_components(),
        evidence=(),
        warning_codes=("warning.insufficient_context",),
    )
    assert ledger.warning_codes == ("warning.insufficient_context",)


@pytest.mark.parametrize(
    "field_name",
    ["contains_patient_rows", "contains_source_text", "contains_direct_identifiers"],
)
def test_data_safety_declaration_rejects_embedded_clinical_content(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="must not contain patient rows"):
        DataSafetyDeclaration(**{field_name: True})


def test_knowledge_rerank_link_must_reference_retrieval_channel() -> None:
    ledger = _ranked_ledger()
    invalid_knowledge = replace(
        ledger.external_knowledge[0],
        rerank_evidence_id="transformer.root",
    )

    with pytest.raises(ValueError, match="retrieval_rerank evidence"):
        replace(ledger, external_knowledge=(invalid_knowledge,))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("replay_succeeded", False, "must replay and flip"),
        ("rank_flipped", False, "must replay and flip"),
        ("clinically_permitted", False, "must pass clinical constraints"),
        ("causal_claim", True, "must not be represented as causal"),
    ],
)
def test_counterfactuals_are_replayed_constrained_and_noncausal(
    field_name: str,
    value: bool,
    message: str,
) -> None:
    fields = {
        "counterfactual_id": "counterfactual.invalid",
        "target_candidate": "candidate_b",
        "cost": 0.1,
        "edit_refs": ("edit:synthetic",),
        "constraint_refs": ("constraint:synthetic",),
        "replay_succeeded": True,
        "rank_flipped": True,
        "clinically_permitted": True,
        "causal_claim": False,
    }
    fields[field_name] = value

    with pytest.raises(ValueError, match=message):
        CounterfactualEvidence(**fields)


def test_deserialization_rejects_unknown_fields() -> None:
    payload = _ranked_ledger().to_dict()
    payload["unreviewed_narrative"] = "not permitted"

    with pytest.raises(ValueError, match="unknown fields"):
        ExplanationLedger.from_dict(payload)


def test_identifiers_are_unique_across_evidence_categories() -> None:
    ledger = _ranked_ledger()
    duplicate = replace(
        ledger.hard_constraints[0],
        certificate_id=ledger.evidence[0].evidence_id,
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        replace(ledger, hard_constraints=(duplicate, *ledger.hard_constraints[1:]))
