"""Training, ranking, and artifact tests for the Phase 1 boundary verifier."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from medical_kg_nlp.benchmarks.phase1.boundary_variants import (
    PHASE1_BOUNDARY_FEATURE_CONTRACT,
    BoundaryErrorLabel,
    boundary_cross_encoder_text,
    extract_phase1_boundary_features,
    generate_phase1_boundary_variants,
    label_phase1_boundary_variant,
)
from medical_kg_nlp.benchmarks.phase1.boundary_verifier import (
    Phase1BoundaryDataset,
    Phase1BoundaryExample,
    Phase1BoundaryVerifier,
    fit_phase1_boundary_verifier,
    load_phase1_boundary_dataset,
    resolve_phase1_boundary_rows,
    write_phase1_boundary_dataset,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProbabilityCalibrator,
    Phase1ProposalVerifier,
)
from medical_kg_nlp.evaluation.sparse_logistic import SparseLogisticModel
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES


def test_boundary_verifier_ranks_full_span_and_round_trips() -> None:
    dataset = _dataset()

    verifier, report = fit_phase1_boundary_verifier(dataset)
    restored = Phase1BoundaryVerifier.from_dict(verifier.to_dict())
    output, scored = resolve_phase1_boundary_rows(
        [_row("1", "đau", 0)],
        {"1": "đau ngực"},
        restored,
        source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
    )

    assert report["holdout_opened"] is False
    assert report["development_selection"]["learned"]["f1"] == 1.0
    assert report["development_selection"]["family_top1"]["accuracy"] == 1.0
    assert output["1"][0]["text"] == "đau ngực"
    assert sum(item.selected for item in scored) == 1
    start, end = output["1"][0]["position"]
    assert "đau ngực"[start:end] == output["1"][0]["text"]


def test_boundary_verifier_rejects_missing_base_probability_contract() -> None:
    dataset = _dataset(requires_base_probability=True)
    verifier, _ = fit_phase1_boundary_verifier(dataset)

    try:
        resolve_phase1_boundary_rows(
            [_row("1", "đau", 0)],
            {"1": "đau ngực"},
            verifier,
            source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
        )
    except ValueError as exc:
        assert "base-probability contract" in str(exc)
    else:
        raise AssertionError("Expected the frozen base-probability contract to fail closed")

    output, scored = resolve_phase1_boundary_rows(
        [_row("1", "đau", 0)],
        {"1": "đau ngực"},
        verifier,
        source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
        proposal_verifier=_always_keep_proposal_verifier(),
    )

    assert verifier.resolution_policy == "conservative_replacement"
    assert len(output["1"]) == 1
    assert sum(item.selected for item in scored) == 1
    assert all(
        item.resolution_policy == "conservative_replacement" for item in scored
    )


def test_boundary_dataset_round_trips_and_checks_sha256(tmp_path) -> None:
    dataset = _dataset()
    output = tmp_path / "dataset"
    write_phase1_boundary_dataset(dataset, output)

    restored = load_phase1_boundary_dataset(output)

    assert restored.manifest == dataset.manifest
    assert restored.examples == dataset.examples

    with (output / "examples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    try:
        load_phase1_boundary_dataset(output)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("Expected a modified dataset artifact to fail closed")


def _dataset(*, requires_base_probability: bool = False) -> Phase1BoundaryDataset:
    examples: list[Phase1BoundaryExample] = []
    split_by_document = {
        "1": "train",
        "2": "train",
        "3": "train",
        "4": "train",
        "5": "development",
        "6": "development",
    }
    for document_id, split in split_by_document.items():
        text = "đau ngực"
        row = _row(document_id, "đau", 0)
        structure = DocumentStructureAnalyzer().analyze(text)
        variants = generate_phase1_boundary_variants(
            document_id,
            text,
            [row],
            source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
            structure=structure,
        )
        sizes = Counter(variant.family_id for variant in variants)
        gold = [
            {
                "text": text,
                "type": "TRIỆU_CHỨNG",
                "position": [0, len(text)],
            }
        ]
        for variant in variants:
            error = label_phase1_boundary_variant(variant, gold)
            features = extract_phase1_boundary_features(
                variant,
                text,
                {"xlmr": ProposalSourceRole.TOKEN_MODEL},
                family_size=sizes[variant.family_id],
                base_probability=0.8 if requires_base_probability else None,
                structure=structure,
            )
            examples.append(
                Phase1BoundaryExample(
                    variant=variant,
                    split=split,
                    label=int(error is BoundaryErrorLabel.CORRECT),
                    error_label=error.value,
                    genre="unknown",
                    section="none",
                    cross_encoder_text=boundary_cross_encoder_text(
                        variant,
                        text,
                        structure=structure,
                    ),
                    features=tuple(sorted(features.items())),
                    base_probability=0.8 if requires_base_probability else None,
                    base_selected=False,
                )
            )
    serialized = "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )
    gold_counts: dict[str, int] = {}
    for split, count in (("train", 4), ("development", 2)):
        for entity_type in PHASE1_ALLOWED_TYPES:
            value = count if entity_type == "TRIỆU_CHỨNG" else 0
            gold_counts[f"{split}:{entity_type}"] = value
            gold_counts[f"{split}:unknown:{entity_type}"] = value
    return Phase1BoundaryDataset(
        examples=tuple(examples),
        manifest={
            "schema_version": "phase1-boundary-dataset.v1",
            "feature_contract": PHASE1_BOUNDARY_FEATURE_CONTRACT,
            "example_count": len(examples),
            "examples_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
            "requires_base_probability": requires_base_probability,
            "gold_entity_counts": gold_counts,
        },
    )


def _row(
    document_id: str,
    text: str,
    start: int,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "proposal_id": f"{document_id}:xlmr:{start}",
        "text": text,
        "type": "TRIỆU_CHỨNG",
        "position": [start, start + len(text)],
        "sources": ["xlmr"],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
        "source_evidence": {
            "xlmr": {
                "confidence": 0.8,
                "source_labels": ["TRIỆU_CHỨNG"],
                "support_only": False,
            }
        },
    }


def _always_keep_proposal_verifier() -> Phase1ProposalVerifier:
    return Phase1ProposalVerifier(
        model=SparseLogisticModel((), (), 10.0),
        probability_calibrator=Phase1ProbabilityCalibrator(
            method="identity_logistic",
            model=None,
            fold_count=2,
            assignment_sha256="a" * 64,
        ),
        thresholds=tuple(
            (entity_type, 0.0) for entity_type in sorted(PHASE1_ALLOWED_TYPES)
        ),
        training_dataset_sha256="b" * 64,
    )
