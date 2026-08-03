from importlib.resources import files
from pathlib import Path

from medical_kg_nlp.context.cue_loader import load_default_assertion_cues


def test_assertion_cues_are_packaged_and_match_source_table() -> None:
    packaged = files("medical_kg_nlp").joinpath("resources/assertion_cues.jsonl")
    source = Path("data/heuristics/assertion_cues.jsonl")

    assert packaged.is_file()
    assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert len(load_default_assertion_cues()) >= 70
