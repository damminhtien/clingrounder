from __future__ import annotations

import json
from pathlib import Path

from clingrounder.benchmarks.phase1.phase1_selective_calibration import (
    CandidateCalibrationOptions,
    build_assertion_calibration_report,
    build_candidate_calibration_report,
    write_candidate_calibration_report,
    write_calibrated_assertion_map,
)
from clingrounder.benchmarks.phase1.phase1 import load_calibrated_assertion_map
from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.schema.annotation import (
    AssertionEvidence,
    CandidateConcept,
    EntityAnnotation,
)
from clingrounder.schema.output import ClinicalPrediction
from clingrounder.schema.types import AssertionStatus, CodeSystem, EntityType


def test_candidate_calibration_promotes_exact_and_rejects_harmful_fuzzy() -> None:
    predictions: list[ClinicalPrediction] = []
    gold: dict[str, list[dict[str, object]]] = {}
    reviewed = frozenset({("tăng huyết áp", "CHẨN_ĐOÁN", "I10")})
    for index in range(1, 21):
        document_id = str(index)
        exact = index <= 10
        mention = "tăng huyết áp" if exact else "bệnh mạch"
        code = "I10" if exact else "I99"
        source = "exact" if exact else "fuzzy"
        predictions.append(_prediction(document_id, mention, code, source))
        gold[document_id] = [
            _gold_row(mention, candidates=["I10"] if exact else [])
        ]

    report = build_candidate_calibration_report(
        predictions,
        gold,
        reviewed_candidates=reviewed,
        options=CandidateCalibrationOptions(
            minimum_support=5,
            minimum_train_support=4,
            abstention_margin=0.05,
        ),
    )

    groups = {
        (row["code_system"], row["source"]): row for row in report["source_groups"]
    }
    assert groups[("ICD-10", "exact")]["recommended"] is True
    assert groups[("ICD-10", "exact")]["mean_emitted_jaccard"] == 1.0
    assert groups[("ICD-10", "fuzzy")]["recommended"] is False
    assert groups[("ICD-10", "fuzzy")]["mean_abstained_jaccard"] == 1.0
    assert report["recommended_link_emit_probabilities_by_source"] == {
        "ICD-10:exact": groups[("ICD-10", "exact")][
            "smoothed_correct_probability"
        ]
    }


def test_candidate_calibration_report_writes_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    predictions = [_prediction(str(index), "tăng huyết áp", "I10", "exact") for index in range(8)]
    gold = {
        str(index): [_gold_row("tăng huyết áp", candidates=["I10"])]
        for index in range(8)
    }
    report = build_candidate_calibration_report(
        predictions,
        gold,
        options=CandidateCalibrationOptions(minimum_support=3, minimum_train_support=2),
    )

    write_candidate_calibration_report(report, tmp_path)

    payload = json.loads((tmp_path / "candidate_calibration.json").read_text(encoding="utf-8"))
    rows = (tmp_path / "candidate_calibration.jsonl").read_text(encoding="utf-8").splitlines()
    assert payload["schema_version"] == "phase1-candidate-calibration.v3"
    assert len(rows) == len(report["policy_groups"])
    assert report["expected_jaccard_policy"]["empty_probabilities"]["ICD-10"] == 0.1
    assert report["expected_jaccard_policy"]["rank_probabilities"]["ICD-10"][
        "exact"
    ] == [0.9]


def test_candidate_coverage_separates_null_rxnorm_and_icd_granularity() -> None:
    predictions = [
        _prediction("1", "aspirin", "1191", "exact", entity_type=EntityType.DRUG),
        _prediction("2", "tăng huyết áp", "I10", "exact"),
        _prediction("3", "bệnh mạch", "I99", "exact"),
    ]
    gold = {
        "1": [_gold_row("aspirin", candidates=["1191"], entity_type="THUỐC")],
        "2": [_gold_row("tăng huyết áp", candidates=["I10"])],
        "3": [_gold_row("bệnh mạch", candidates=[])],
    }
    entries = [
        ConceptEntry(
            concept_id="RX:1191",
            code="1191",
            code_system=CodeSystem.RXNORM,
            canonical_name="aspirin",
            semantic_type=EntityType.DRUG,
            rxnorm_tty="IN",
        ),
        ConceptEntry(
            concept_id="ICD:I10",
            code="I10",
            code_system=CodeSystem.ICD10,
            canonical_name="hypertension",
            semantic_type=EntityType.DISEASE,
        ),
        ConceptEntry(
            concept_id="ICD:I11",
            code="I11",
            code_system=CodeSystem.ICD10,
            canonical_name="hypertensive heart disease",
            semantic_type=EntityType.DISEASE,
            parent_code="I10",
        ),
    ]

    report = build_candidate_calibration_report(
        predictions,
        gold,
        terminology_entries=entries,
        options=CandidateCalibrationOptions(minimum_support=1, minimum_train_support=1),
    )

    buckets = {
        (row["code_system"], row["bucket"]): row
        for row in report["coverage_buckets"]["buckets"]
    }
    assert buckets[("RxNorm", "rxnorm_ingredient")]["mean_jaccard"] == 1.0
    assert buckets[("ICD-10", "icd_parent")]["mean_jaccard"] == 1.0
    assert buckets[("ICD-10", "gold_empty")]["mean_jaccard"] == 0.0


def test_candidate_calibration_estimates_contiguous_rank_probabilities() -> None:
    predictions: list[ClinicalPrediction] = []
    gold: dict[str, list[dict[str, object]]] = {}
    for index in range(10):
        document_id = str(index)
        prediction = _prediction(document_id, "tăng huyết áp", "I10", "exact")
        prediction.entities[0].candidates.append(
            CandidateConcept(
                code_system=CodeSystem.ICD10,
                code="I11",
                name="I11",
                retrieval_score=0.8,
                emit_probability=0.0,
                concept_id="ICD-10:I11",
                source="exact",
                evidence_sources=("exact",),
                matched_alias="tăng huyết áp",
                qualified=True,
                qualification_reason="test_candidate",
            )
        )
        predictions.append(prediction)
        gold[document_id] = [
            _gold_row(
                "tăng huyết áp",
                candidates=["I10", "I11"] if index < 5 else ["I10"],
            )
        ]

    report = build_candidate_calibration_report(
        predictions,
        gold,
        options=CandidateCalibrationOptions(minimum_support=5, minimum_train_support=4),
    )

    policy = report["expected_jaccard_policy"]
    probabilities = policy["rank_probabilities"]["ICD-10"]["exact"]
    assert len(probabilities) == 2
    assert probabilities[0] > 0.8
    assert 0.4 < probabilities[1] < 0.6


def test_assertion_calibration_promotes_precise_rule_and_rejects_false_rule(
    tmp_path: Path,
) -> None:
    predictions: list[ClinicalPrediction] = []
    gold: dict[str, list[dict[str, object]]] = {}
    for index in range(1, 21):
        document_id = str(index)
        correct = index <= 10
        assertion = AssertionStatus.NEGATED if correct else AssertionStatus.FAMILY
        rule_id = "neg.no" if correct else "family.noise"
        label = "isNegated" if correct else "isFamily"
        entity = EntityAnnotation(
            id="E1",
            span=(0, len("viêm phổi")),
            text="viêm phổi",
            normalized_text="viêm phổi",
            type=EntityType.DISEASE,
            assertion_evidence=(
                AssertionEvidence(rule_id, assertion, "cue", "left"),
            ),
        )
        predictions.append(
            ClinicalPrediction.from_text(
                document_id,
                "viêm phổi",
                [entity],
                [],
                "test",
            )
        )
        row = _gold_row("viêm phổi", candidates=[])
        row["assertions"] = [label] if correct else []
        gold[document_id] = [row]

    report = build_assertion_calibration_report(
        predictions,
        gold,
        options=CandidateCalibrationOptions(
            minimum_support=5,
            minimum_train_support=4,
        ),
    )

    groups = {row["rule_id"]: row for row in report["evidence_groups"]}
    assert groups["neg.no"]["recommended"] is True
    assert groups["family.noise"]["recommended"] is False
    assert report["recommended_rule_ids"] == ["neg.no"]
    map_path = tmp_path / "assertion_evidence_map.jsonl"
    rows = write_calibrated_assertion_map(report, map_path)
    assert len(rows) == 1
    assert load_calibrated_assertion_map(map_path) == frozenset(
        {("neg.no", "isNegated", "CHẨN_ĐOÁN")}
    )


def _prediction(
    document_id: str,
    mention: str,
    code: str,
    source: str,
    *,
    entity_type: EntityType = EntityType.DISEASE,
) -> ClinicalPrediction:
    code_system = CodeSystem.RXNORM if entity_type == EntityType.DRUG else CodeSystem.ICD10
    entity = EntityAnnotation(
        id="E1",
        span=(0, len(mention)),
        text=mention,
        normalized_text=mention,
        type=entity_type,
        candidates=[
            CandidateConcept(
                code_system=code_system,
                code=code,
                name=mention,
                retrieval_score=0.95,
                emit_probability=0.0,
                concept_id=f"{code_system.value}:{code}",
                source=source,
                evidence_sources=(source,),
                matched_alias=mention,
                qualified=True,
                qualification_reason="test_candidate",
            )
        ],
    )
    return ClinicalPrediction.from_text(document_id, mention, [entity], [], "test")


def _gold_row(
    mention: str,
    *,
    candidates: list[str],
    entity_type: str = "CHẨN_ĐOÁN",
) -> dict[str, object]:
    return {
        "text": mention,
        "type": entity_type,
        "assertions": [],
        "candidates": candidates,
        "position": [0, len(mention)],
    }
