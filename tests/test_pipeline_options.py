from __future__ import annotations

import pytest

from clingrounder.pipeline.factory import PipelineConfig
from clingrounder.pipeline.options import PipelineOptions


def test_pipeline_options_parse_candidate_calibration_thresholds() -> None:
    options = PipelineOptions.from_mapping(
        {
            "link_candidate_thresholds_by_type": {"DISEASE": 0.82},
            "link_candidate_thresholds_by_source": {"exact": 0.65, "fuzzy": 0.88},
            "link_emit_probabilities_by_source": {
                "ICD-10:dictionary_exact": 0.99505,
            },
        }
    )

    assert options.link_candidate_thresholds_by_type == (("DISEASE", 0.82),)
    assert options.link_candidate_thresholds_by_source == (("exact", 0.65), ("fuzzy", 0.88))
    assert options.link_emit_probabilities_by_source == (
        ("ICD-10:dictionary_exact", 0.99505),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"link_candidate_thresholds_by_type": {"NOT_A_TYPE": 0.8}},
        {"link_candidate_thresholds_by_source": {"fuzzy": 1.1}},
        {"link_candidate_thresholds_by_source": []},
        {"link_max_qualified_candidates": 6},
    ],
)
def test_pipeline_options_reject_invalid_candidate_qualification_config(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PipelineOptions.from_mapping(payload)


def test_pipeline_options_rejects_unknown_source_and_negative_context() -> None:
    with pytest.raises(ValueError, match="Unknown candidate source"):
        PipelineOptions.from_mapping({"candidate_sources": ["not_a_source"]})
    with pytest.raises(ValueError, match="context_window"):
        PipelineOptions.from_mapping({"context_window": -1})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"enable_linking": False, "enable_candidate_reranking": True}, "reranking"),
        (
            {
                "enable_linking": False,
                "enable_candidate_reranking": False,
                "enable_graph_evidence_reranking": True,
            },
            "Graph evidence",
        ),
        ({"enable_relations": False, "enable_relation_kg_validation": True}, "relations"),
    ],
)
def test_pipeline_options_reject_incompatible_subsystems(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PipelineOptions.from_mapping(payload)


def test_grouped_subsystem_config_compiles_to_one_runtime_policy() -> None:
    config = PipelineConfig.from_mapping(
        {
            "pipeline": {
                "context": {"provider": "rules", "context_window": 120},
                "linking": {
                    "provider": "rules",
                    "max_candidates": 12,
                    "candidate_sources": ["exact", "bm25"],
                    "reranker": {"provider": "rules"},
                },
                "graph": {"provider": "disabled"},
                "relations": {"provider": "disabled", "validate_with_kg": False},
                "validation": {"entities_with_kg": False, "relations_with_kg": False},
                "runtime": {"backend": "serial", "workers": 1},
            }
        }
    )

    assert config.context.context_window == 120
    assert config.linking.max_candidates == 12
    assert config.runtime.backend == "serial"
    assert config.options.max_candidates == 12
    assert config.options.enable_context is True
    assert config.options.enable_linking is True
    assert config.options.enable_relations is False
    assert config.options.enable_relation_kg_validation is False


def test_grouped_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValueError, match="pipeline.context.enable_contex"):
        PipelineConfig.from_mapping(
            {"pipeline": {"context": {"enable_contex": True}}}
        )


def test_grouped_config_rejects_disabled_linking_with_reranker() -> None:
    with pytest.raises(ValueError, match="Candidate reranking requires linking"):
        PipelineConfig.from_mapping(
            {
                "pipeline": {
                    "linking": {
                        "provider": "disabled",
                        "reranker": {"provider": "rules"},
                    }
                }
            }
        )
