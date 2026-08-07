"""Runner helpers keep support evidence typed, offset-safe, and model-neutral."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clingrounder.adapters.generative import GenerationConfig
from clingrounder.benchmarks.phase1.qwen_proposals import Phase1QwenPassResult
from clingrounder.benchmarks.phase1.qwen_runner import (
    Phase1QwenProposalRunConfig,
    _adjudication_candidates,
    _has_document_outputs,
    _load_completed_document,
    _merge_review_rows,
    _prepare_resume_state,
    _proposals_to_rows,
    _qwen_proposals_to_source_rows,
    _rows_to_review_entities,
    _rows_to_proposals,
    _run_document_passes,
    _write_document_rows,
    materialize_phase1_qwen_pass_source,
)
from clingrounder.ner.proposal import EntityProposal
from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.types import EntityType


def test_runner_round_trips_support_rows_without_assertion_or_candidates() -> None:
    text = "Bệnh nhân ho và tăng huyết áp"
    rows = [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [10, 12],
            "assertions": ["isNegated"],
            "candidates": [],
        },
        {
            "text": "tăng huyết áp",
            "type": "CHẨN_ĐOÁN",
            "position": [16, 29],
            "assertions": [],
            "candidates": ["I10"],
        },
    ]

    proposals = _rows_to_proposals(rows, text, source="rule")
    output = _proposals_to_rows(proposals, text)

    assert [row["text"] for row in output] == ["ho", "tăng huyết áp"]
    assert all(row["assertions"] == [] and row["candidates"] == [] for row in output)


def test_exact_quote_source_retains_independent_pass_evidence() -> None:
    text = "Bệnh nhân ho"
    proposals = (
        EntityProposal(
            span=(10, 12),
            candidate_types=(EntityType.SYMPTOM,),
            source="qwen.recall",
            score=0.6,
        ),
        EntityProposal(
            span=(10, 12),
            candidate_types=(EntityType.SYMPTOM,),
            source="qwen.targeted.TRIỆU_CHỨNG",
            score=0.9,
        ),
    )

    rows = _qwen_proposals_to_source_rows(proposals, text)

    assert rows == [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
            "position": [10, 12],
            "confidence": 0.6,
            "source_label": "qwen.recall",
        },
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
            "position": [10, 12],
            "confidence": 0.9,
            "source_label": "qwen.targeted.TRIỆU_CHỨNG",
        }
    ]


def test_materialize_stored_qwen_pass_projects_offsets_and_verifies_hash(
    tmp_path: Path,
) -> None:
    document = ClinicalDocument(document_id="1", text="Bệnh nhân ho và sốt.")
    response = (
        '{"entities":[{"text":"ho","type":"TRIỆU_CHỨNG"},{"text":"sốt","type":"TRIỆU_CHỨNG"}]}'
    )
    manifest = materialize_phase1_qwen_pass_source(
        (document,),
        (
            {
                "document_id": "1",
                "pass_id": "recall",
                "window_index": 0,
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            },
        ),
        tmp_path / "source",
        pass_id="recall",
        max_window_characters=12_000,
        window_overlap_characters=800,
    )

    rows = json.loads((tmp_path / "source" / "1.json").read_text())
    assert [(row["text"], row["position"]) for row in rows] == [
        ("ho", [10, 12]),
        ("sốt", [16, 19]),
    ]
    assert manifest["counters"]["entity.total"] == 2
    assert manifest["counters"]["response.complete"] == 1


def test_runner_rejects_support_offset_mismatch() -> None:
    with pytest.raises(ValueError, match="violates raw offsets"):
        _rows_to_proposals(
            (
                {
                    "text": "ho",
                    "type": "TRIỆU_CHỨNG",
                    "position": [0, 2],
                },
            ),
            "Không ho",
            source="xlmr",
        )


def test_adjudication_candidates_merge_exact_source_evidence() -> None:
    text = "ho"
    rule = _rows_to_proposals(
        ({"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},),
        text,
        source="rule",
    )
    qwen = _rows_to_proposals(
        ({"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},),
        text,
        source="qwen.recall",
    )

    candidates = _adjudication_candidates(
        {"rule": rule, "qwen.recall": qwen},
        text,
    )

    assert candidates[0].sources == ("qwen.recall", "rule")


def test_review_rows_preserve_baseline_metadata_and_reject_overlap() -> None:
    baseline = [
        {
            "text": "aspirin 81 mg",
            "type": "THUỐC",
            "position": [0, 13],
            "assertions": ["isHistorical"],
            "candidates": ["243670"],
        }
    ]
    additions = [
        {
            "text": "aspirin",
            "type": "THUỐC",
            "position": [0, 7],
            "assertions": [],
            "candidates": [],
        },
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [18, 20],
            "assertions": [],
            "candidates": [],
        },
    ]

    reviewed, rejected = _merge_review_rows(baseline, additions)

    assert rejected == 1
    assert reviewed == [
        baseline[0],
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [18, 20],
            "assertions": [],
            "candidates": [],
        },
    ]


def test_review_entities_keep_every_raw_occurrence() -> None:
    text = "ho rồi ho"
    entities = _rows_to_review_entities(
        (
            {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},
            {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [7, 9]},
        ),
        text,
        source="baseline",
    )

    assert [entity.span for entity in entities] == [(0, 2), (7, 9)]


def test_review_only_config_requires_a_complete_source(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires a review source"):
        Phase1QwenProposalRunConfig(
            documents_path=documents,
            expected_source_archive_sha256="a" * 64,
            output_dir=tmp_path / "output",
            review_only=True,
        )


def test_resume_reuses_only_complete_offset_validated_document(
    tmp_path: Path,
) -> None:
    document = ClinicalDocument(document_id="1", text="Bệnh nhân ho")
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    _write_document_rows(
        consensus / "1.json",
        (
            {
                "text": "ho",
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
                "position": [10, 12],
            },
        ),
    )

    resumed = _load_completed_document(
        document,
        consensus_dir=consensus,
        adjudicated_dir=None,
        review_additions_dir=None,
        reviewed_dir=None,
        support_source_names=("vietmed",),
    )

    assert resumed is not None
    assert resumed["counters"]["resume.documents"] == 1
    assert resumed["counters"]["consensus.entities"] == 1
    assert resumed["trace"]["resume"]["source"] == "validated_document_outputs"


def test_resume_rejects_existing_offset_mismatch(tmp_path: Path) -> None:
    document = ClinicalDocument(document_id="1", text="Bệnh nhân ho")
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    _write_document_rows(
        consensus / "1.json",
        (
            {
                "text": "ho",
                "type": "TRIỆU_CHỨNG",
                "assertions": [],
                "candidates": [],
                "position": [0, 2],
            },
        ),
    )

    with pytest.raises(ValueError, match="violates raw offsets"):
        _load_completed_document(
            document,
            consensus_dir=consensus,
            adjudicated_dir=None,
            review_additions_dir=None,
            reviewed_dir=None,
            support_source_names=(),
        )


def test_resume_state_blocks_mixed_run_fingerprints(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "consensus").mkdir(parents=True)
    (output / "consensus" / "1.json").write_text("[]\n", encoding="utf-8")

    adopted = _prepare_resume_state(
        output,
        run_fingerprint="a" * 64,
        resume=True,
    )

    assert adopted["adopted_existing_outputs"] is True
    with pytest.raises(ValueError, match="fingerprint differs"):
        _prepare_resume_state(
            output,
            run_fingerprint="b" * 64,
            resume=True,
        )


def test_resume_detects_authorized_source_prefixed_document_output(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    (consensus / "authorized_gt:1.json").write_text("[]\n", encoding="utf-8")

    assert _has_document_outputs(tmp_path) is True


class _RecordingAdapter:
    def __init__(self) -> None:
        self.pass_ids: list[str] = []

    def extract(self, source_text: str, *, pass_id: str, target_types, generation):
        del source_text, target_types, generation
        self.pass_ids.append(pass_id)
        return Phase1QwenPassResult(
            pass_id=pass_id,
            prompt_hash="a" * 64,
            proposals=(),
            rejected=(),
            response_sha256=(),
            raw_responses=(),
        )


@pytest.mark.parametrize(
    ("mode", "expected_passes"),
    [
        ("recall_only", ["recall"]),
        (
            "recall_and_targeted",
            [
                "recall",
                "targeted.TRIỆU_CHỨNG",
                "targeted.TÊN_XÉT_NGHIỆM",
                "targeted.KẾT_QUẢ_XÉT_NGHIỆM",
                "targeted.CHẨN_ĐOÁN",
                "targeted.THUỐC",
            ],
        ),
    ],
)
def test_document_passes_respect_bounded_extraction_mode(
    mode: str,
    expected_passes: list[str],
) -> None:
    adapter = _RecordingAdapter()
    run_spec = SimpleNamespace(
        recall_generation=GenerationConfig(),
        targeted_generation=GenerationConfig(),
    )

    _run_document_passes(
        adapter,  # type: ignore[arg-type]
        run_spec,  # type: ignore[arg-type]
        ClinicalDocument(document_id="1", text="Bệnh nhân ho"),
        extraction_mode=mode,  # type: ignore[arg-type]
    )

    assert adapter.pass_ids == expected_passes
