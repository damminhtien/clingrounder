from __future__ import annotations

import pytest

from medical_kg_nlp.pipeline.options import PipelineOptions


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
