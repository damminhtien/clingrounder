from importlib.resources import files
from pathlib import Path

from medical_kg_nlp.context.cue_loader import load_default_assertion_cues
from medical_kg_nlp.ontology.false_positive import load_false_positive_rules


def test_assertion_cues_are_packaged_and_match_source_table() -> None:
    packaged = files("medical_kg_nlp").joinpath("resources/assertion_cues.jsonl")
    source = Path("data/heuristics/assertion_cues.jsonl")

    assert packaged.is_file()
    assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert len(load_default_assertion_cues()) >= 70


def test_false_positive_rules_are_packaged_and_match_source_table() -> None:
    packaged = files("medical_kg_nlp").joinpath("resources/false_positive_blacklist.jsonl")
    source = Path("data/heuristics/false_positive_blacklist.jsonl")

    assert packaged.is_file()
    assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    rules = load_false_positive_rules()
    assert len(rules) >= 5
    assert len({rule.rule_id for rule in rules}) == len(rules)
    assert all(rule.source and rule.examples_positive and rule.examples_negative for rule in rules)
