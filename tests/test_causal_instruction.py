"""Deterministic dataset and assistant-only loss contracts for causal training."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from clingrounder.training.causal_instruction import (
    CausalInstructionRecord,
    CausalInstructionSource,
    InstructionTooLongError,
    load_causal_instruction_records,
    tokenize_causal_instruction,
)
from clingrounder.utils.hashing import sha256_file


class _CharacterTokenizer:
    """Small chat tokenizer whose prefix behavior is explicit in unit tests."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        elif conversation and conversation[-1]["role"] == "assistant":
            prefix = rendered.rsplit("<assistant>", maxsplit=1)[0]
            target = conversation[-1]["content"]
            rendered = f"{prefix}<assistant>{target}</assistant>"
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Mapping[str, Sequence[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def test_loader_filters_samples_deduplicates_then_repeats(tmp_path: Path) -> None:
    source_path = tmp_path / "instructions.jsonl"
    rows = [
        _row("keep-a", "train", "synthetic:a", "A"),
        _row("keep-a-copy", "train", "synthetic:b", "A"),
        _row("keep-b", "train", "synthetic:c", "B"),
        _row("wrong-prefix", "train", "human:1", "C"),
        _row("wrong-split", "development", "synthetic:d", "D"),
    ]
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    source = CausalInstructionSource(
        path=source_path,
        sha256=sha256_file(source_path),
        split="train",
        repeat=2,
        document_id_prefix="synthetic:",
    )

    first, first_report = load_causal_instruction_records(
        [source],
        sample_seed="fixed",
    )
    second, second_report = load_causal_instruction_records(
        [source],
        sample_seed="fixed",
    )

    assert [record.record_id for record in first] == [
        record.record_id for record in second
    ]
    assert len(first) == 4
    assert first_report.selected_unique_records == 2
    assert first_report.repeated_training_records == 4
    assert first_report.duplicate_content_records == 1
    assert first_report.dataset_sha256 == second_report.dataset_sha256


def test_tokenizer_masks_prompt_and_keeps_complete_assistant_target() -> None:
    tokenizer = _CharacterTokenizer()
    record = _record("one", "target")

    tokenized = tokenize_causal_instruction(tokenizer, record, max_length=512)

    first_target = tokenized["labels"].index(next(value for value in tokenized["labels"] if value >= 0))
    assert all(label == -100 for label in tokenized["labels"][:first_target])
    assert tokenized["labels"][first_target:] == tokenized["input_ids"][first_target:]
    assert len(tokenized["attention_mask"]) == len(tokenized["input_ids"])


def test_tokenizer_rejects_overlength_instead_of_truncating() -> None:
    with pytest.raises(InstructionTooLongError, match="max_length=256"):
        tokenize_causal_instruction(
            _CharacterTokenizer(),
            _record("long", "x" * 300),
            max_length=256,
        )


def test_source_rejects_changed_fingerprint(tmp_path: Path) -> None:
    source_path = tmp_path / "instructions.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    source = CausalInstructionSource(
        path=source_path,
        sha256="0" * 64,
        split="train",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        source.verify()


def _row(
    record_id: str,
    split: str,
    document_id: str,
    target: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "document_id": document_id,
        "split": split,
        "task": "test",
        "messages": [
            {"role": "system", "content": "extract"},
            {"role": "user", "content": "source"},
            {"role": "assistant", "content": target},
        ],
    }


def _record(record_id: str, target: str) -> CausalInstructionRecord:
    return CausalInstructionRecord(
        record_id=record_id,
        messages=(
            {"role": "system", "content": "extract"},
            {"role": "user", "content": "source"},
            {"role": "assistant", "content": target},
        ),
        split="train",
        task="test",
        source_path="fixture",
    )
