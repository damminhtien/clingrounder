"""Public Pipeline facade behavior and lifecycle contracts."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from clingrounder import (
    ClinicalDocument,
    Pipeline,
    PipelineClosedError,
    PipelineConfigurationError,
    UnknownProfileError,
    load_pipeline,
)
from clingrounder.artifacts import ArtifactNotFoundError, BuiltinArtifact, get_builtin_artifact
from clingrounder.pipeline import PipelineComponents, PipelineOptions, RuntimeCapabilities
from clingrounder.schema.annotation import EntityAnnotation


def test_pipeline_from_profile_predicts_and_closes() -> None:
    with Pipeline.from_profile("clinical-baseline") as pipeline:
        prediction = pipeline.predict("Bệnh nhân khó thở.", document_id="note-001")

    assert prediction.document_id == "note-001"
    with pytest.raises(PipelineClosedError):
        pipeline.predict("Bệnh nhân ho.", document_id="note-002")


def test_pipeline_from_config_and_trace() -> None:
    with Pipeline.from_config("configs/pipeline/clinical-baseline.yaml") as pipeline:
        result = pipeline.predict_with_trace("Bệnh nhân khó thở.", document_id="note-001")

    assert result.prediction.document_id == "note-001"
    assert result.trace.document_id == "note-001"


def test_bundled_vietnamese_artifact_is_offline_and_callable(tmp_path: Path) -> None:
    with load_pipeline("vi-clinical-small", offline=True) as pipeline:
        prediction = pipeline(
            "Bệnh nhân không sốt. Tiền sử tăng huyết áp. Đang dùng metformin."
        )

    by_text = {entity.text: entity for entity in prediction.entities}
    assert prediction.document_id.startswith("text-")
    assert by_text["sốt"].assertion.value == "NEGATED"
    assert by_text["tăng huyết áp"].assertion.value == "HISTORICAL"
    assert by_text["metformin"].code == "6809"

    cached = Pipeline.download("vi-clinical-small", cache_dir=tmp_path)
    assert (cached / "seed_concepts.jsonl").is_file()
    assert (cached / "manifest.json").is_file()

    with load_pipeline(cached, offline=True) as pipeline:
        cached_prediction = pipeline("Bệnh nhân không sốt.")
    assert cached_prediction.document_id.startswith("text-")


def test_bundled_artifact_trace_exposes_latency_and_candidate_evidence() -> None:
    with load_pipeline("vi-clinical-small", offline=True) as pipeline:
        result = pipeline.predict_with_trace(
            "Bệnh nhân đang dùng metformin.",
            document_id="trace-evidence",
        )

    assert result.trace.stages
    assert result.trace.total_ms >= 0.0
    assert all(stage.elapsed_ms >= 0.0 for stage in result.trace.stages)
    metformin = next(
        entity for entity in result.prediction.entities if entity.text == "metformin"
    )
    assert metformin.candidates
    candidate = metformin.candidates[0]
    assert candidate.source in candidate.evidence_sources
    assert candidate.matched_alias
    assert candidate.qualification_reason


def test_artifact_manifest_rejects_payload_tampering(tmp_path: Path) -> None:
    source = get_builtin_artifact("vi-clinical-small")
    source.install(tmp_path)
    copied_root = tmp_path / "vi-clinical-small" / source.revision
    payload = copied_root / "seed_concepts.jsonl"
    payload.write_text(payload.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = BuiltinArtifact(source.artifact_id, source.revision, copied_root)

    with pytest.raises(ArtifactNotFoundError, match="checksum/size"):
        tampered.verify_manifest()


def test_pipeline_from_components_preserves_batch_order() -> None:
    components = PipelineComponents(
        entity_extractor=_Extractor(),
        options=_minimal_options(),
        runtime_capabilities=RuntimeCapabilities(thread_safe=True),
    )

    with Pipeline.from_components(components) as pipeline:
        predictions = pipeline.predict_many(
            [
                ClinicalDocument("doc-1", "first"),
                ClinicalDocument("doc-2", "second"),
            ],
            workers=2,
        )

    assert [prediction.document_id for prediction in predictions] == ["doc-1", "doc-2"]


def test_pipeline_close_is_idempotent() -> None:
    pipeline = Pipeline.from_components(
        PipelineComponents(entity_extractor=_Extractor(), options=_minimal_options())
    )

    pipeline.close()
    pipeline.close()
    with pytest.raises(PipelineClosedError):
        pipeline.predict("text", document_id="doc")


def test_pipeline_closes_after_prediction_failure() -> None:
    pipeline = Pipeline.from_components(
        PipelineComponents(
            entity_extractor=_Extractor(fail=True),
            options=_minimal_options(),
        )
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with pipeline:
            pipeline.predict("FAIL", document_id="doc")
    with pytest.raises(PipelineClosedError):
        pipeline.predict("text", document_id="doc")


def test_pipeline_profile_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(UnknownProfileError, match="Unknown pipeline profile"):
        Pipeline.from_profile("does-not-exist")

    profile = tmp_path / "missing.yaml"
    profile.write_text(
        """\
schema_version: clingrounder.pipeline-profile
profile:
  id: missing
  title: Missing resources
  description: Test profile
  maturity: stable
  portability: template
  support_status: setup_required
terminology:
  recognition_path: missing.jsonl
pipeline:
  enable_context: false
  enable_linking: false
  enable_candidate_reranking: false
  enable_entity_kg_validation: false
  enable_relations: false
  enable_relation_kg_validation: false
""",
        encoding="utf-8",
    )
    with pytest.raises(PipelineConfigurationError, match="Unable to compose"):
        Pipeline.from_config(profile)


def test_top_level_import_does_not_load_optional_ml_dependencies() -> None:
    code = """
import sys
import clingrounder
assert not any(name in sys.modules for name in ("torch", "transformers", "faiss"))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _minimal_options() -> PipelineOptions:
    return PipelineOptions(
        enable_context=False,
        enable_linking=False,
        enable_candidate_reranking=False,
        enable_entity_kg_validation=False,
        enable_relations=False,
        enable_relation_kg_validation=False,
    )


@dataclass(frozen=True)
class _Extractor:
    fail: bool = False

    def extract(self, source_text: str) -> list[EntityAnnotation]:
        if self.fail:
            raise RuntimeError("synthetic failure")
        return []
