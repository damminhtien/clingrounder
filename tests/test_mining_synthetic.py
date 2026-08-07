"""Regression tests for graph-first synthetic scenario rendering."""

from __future__ import annotations

import pytest

from clingrounder.mining.policy import MiningQualityGate
from clingrounder.mining.synthetic import (
    MinimalPairGenerator,
    ScenarioEntity,
    ScenarioGraph,
    SentinelScenarioRenderer,
)


def test_all_minimal_pair_cases_preserve_projected_offsets() -> None:
    generator = MinimalPairGenerator()
    renderer = SentinelScenarioRenderer()

    for case_name in generator.case_names:
        graph, template, note_type = generator.build(case_name)
        rendered = renderer.render_template(graph, template, note_type=note_type)
        for annotation in rendered.annotations:
            annotation.validate_offsets(rendered.document)
        assert "<<KG:" not in rendered.document.text
        assert not MiningQualityGate().validate(
            [rendered.document], list(rendered.annotations)
        )


def test_repeated_entity_creates_distinct_occurrence_annotations() -> None:
    graph, template, note_type = MinimalPairGenerator().build("repeated_mention")

    rendered = SentinelScenarioRenderer().render_template(
        graph, template, note_type=note_type
    )

    assert len(rendered.annotations) == 2
    assert rendered.annotations[0].span != rendered.annotations[1].span
    assert {item.text for item in rendered.annotations} == {"đau bụng"}


def test_projection_accepts_paraphrased_surface_inside_preserved_sentinel() -> None:
    graph = ScenarioGraph(
        "paraphrase",
        (ScenarioEntity("symptom", "đau ngực", "SYMPTOM"),),
    )

    rendered = SentinelScenarioRenderer().project(
        graph,
        "Bệnh nhân <<KG:symptom>>đau tức ngực<</KG:symptom>>.",
        note_type="progress_note",
    )

    assert rendered.annotations[0].text == "đau tức ngực"
    rendered.annotations[0].validate_offsets(rendered.document)


def test_projection_rejects_missing_or_malformed_sentinels() -> None:
    graph = ScenarioGraph("bad", (ScenarioEntity("disease", "cúm", "DISEASE"),))
    renderer = SentinelScenarioRenderer()

    with pytest.raises(ValueError, match="missing"):
        renderer.project(graph, "Không có sentinel.", note_type="progress_note")
    with pytest.raises(ValueError, match="Malformed sentinel"):
        renderer.project(
            graph,
            "<<KG:disease>>cúm<</KG:other>>",
            note_type="progress_note",
        )
