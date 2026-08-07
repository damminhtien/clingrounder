"""Deterministic chat-instruction loading and causal-loss projection.

This module intentionally has no Torch or Transformers import. Dataset identity,
sampling, leakage filters, and prompt/assistant masking can therefore be tested
on CPU and inspected before an expensive GPU process starts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from clingrounder.utils.hashing import sha256_file

__all__ = [
    "CausalInstructionRecord",
    "CausalInstructionSource",
    "ChatTokenizerPort",
    "InstructionDatasetReport",
    "InstructionTooLongError",
    "load_causal_instruction_records",
    "tokenize_causal_instruction",
]


class ChatTokenizerPort(Protocol):
    """Small tokenizer surface needed to build assistant-only causal labels."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Mapping[str, Sequence[int]]: ...


class InstructionTooLongError(ValueError):
    """Raised instead of silently truncating source text or gold output."""


@dataclass(frozen=True, slots=True)
class CausalInstructionSource:
    """One immutable JSONL source and its deterministic sampling policy."""

    path: Path
    sha256: str
    split: str
    maximum_records: int | None = None
    repeat: int = 1
    document_id_prefix: str | None = None

    def __post_init__(self) -> None:
        if not self.split.strip():
            raise ValueError("Instruction source split must be non-empty")
        if self.maximum_records is not None and self.maximum_records < 1:
            raise ValueError("maximum_records must be positive when provided")
        if self.repeat < 1 or self.repeat > 20:
            raise ValueError("Instruction source repeat must be between 1 and 20")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("Instruction source SHA-256 must be lowercase hexadecimal")

    def verify(self) -> None:
        """Reject missing or changed derived data before training."""

        if not self.path.is_file():
            raise ValueError(f"Instruction source does not exist: {self.path}")
        observed = sha256_file(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"Instruction source SHA-256 mismatch for {self.path}: "
                f"expected {self.sha256}, observed {observed}"
            )


@dataclass(frozen=True, slots=True)
class CausalInstructionRecord:
    """Validated chat record with stable identity and provenance."""

    record_id: str
    messages: tuple[dict[str, str], ...]
    split: str
    task: str
    source_path: str

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.messages:
            raise ValueError("Instruction record requires record_id and messages")
        if self.messages[-1].get("role") != "assistant":
            raise ValueError("Instruction record must end with one assistant target")
        if any(
            message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in self.messages
        ):
            raise ValueError("Instruction messages require supported roles and text content")

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(self.messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InstructionDatasetReport:
    """Counts and fingerprints written into every QLoRA run manifest."""

    source_counts: dict[str, int]
    selected_unique_records: int
    repeated_training_records: int
    duplicate_content_records: int
    dataset_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_counts": dict(sorted(self.source_counts.items())),
            "selected_unique_records": self.selected_unique_records,
            "repeated_training_records": self.repeated_training_records,
            "duplicate_content_records": self.duplicate_content_records,
            "dataset_sha256": self.dataset_sha256,
        }


def load_causal_instruction_records(
    sources: Sequence[CausalInstructionSource],
    *,
    sample_seed: str,
) -> tuple[list[CausalInstructionRecord], InstructionDatasetReport]:
    """Load, split-filter, sample, deduplicate, and intentionally repeat records.

    SCALING: sampling ranks hashes instead of materializing random state, so
    another machine selects the same records regardless of filesystem order.
    """

    if not sources or not sample_seed.strip():
        raise ValueError("Instruction loading requires sources and sample_seed")
    selected: list[tuple[CausalInstructionRecord, int]] = []
    source_counts: dict[str, int] = {}
    for source in sources:
        source.verify()
        rows = _read_source(source)
        rows.sort(key=lambda row: _sample_rank(sample_seed, row.record_id))
        if source.maximum_records is not None:
            rows = rows[: source.maximum_records]
        source_counts[str(source.path)] = len(rows)
        selected.extend((row, source.repeat) for row in rows)

    unique: dict[str, tuple[CausalInstructionRecord, int]] = {}
    duplicate_count = 0
    for record, repeat in selected:
        key = record.content_sha256
        if key in unique:
            duplicate_count += 1
            existing, existing_repeat = unique[key]
            unique[key] = (existing, max(existing_repeat, repeat))
            continue
        unique[key] = (record, repeat)

    repeated: list[CausalInstructionRecord] = []
    for key in sorted(unique):
        record, repeat = unique[key]
        repeated.extend(record for _ in range(repeat))
    dataset_payload = "\n".join(record.content_sha256 for record in repeated)
    report = InstructionDatasetReport(
        source_counts=source_counts,
        selected_unique_records=len(unique),
        repeated_training_records=len(repeated),
        duplicate_content_records=duplicate_count,
        dataset_sha256=hashlib.sha256(dataset_payload.encode("ascii")).hexdigest(),
    )
    return repeated, report


def tokenize_causal_instruction(
    tokenizer: ChatTokenizerPort,
    record: CausalInstructionRecord,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat record and mask every non-assistant token from loss.

    INVARIANT: overlength records fail rather than dropping the beginning of
    SOURCE or the end of the gold JSON. Either truncation would train a target
    that is no longer supported by the visible text.
    """

    if max_length < 256:
        raise ValueError("Causal max_length must be at least 256")
    prompt_messages = record.messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        record.messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Tokenizer chat template does not preserve the generation-prompt prefix"
        )
    prompt_ids = list(
        tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    )
    input_ids = list(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenized full conversation does not preserve prompt tokens")
    if len(input_ids) > max_length:
        raise InstructionTooLongError(
            f"{record.record_id} requires {len(input_ids)} tokens; max_length={max_length}"
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"{record.record_id} has no assistant target tokens")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _read_source(source: CausalInstructionSource) -> list[CausalInstructionRecord]:
    records: list[CausalInstructionRecord] = []
    with source.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split", "")) != source.split:
                continue
            document_id = str(raw.get("document_id", raw.get("source_record_id", "")))
            if source.document_id_prefix is not None and not document_id.startswith(
                source.document_id_prefix
            ):
                continue
            messages_raw = raw.get("messages")
            if not isinstance(messages_raw, list):
                raise ValueError(
                    f"{source.path}:{line_number} messages must be a list"
                )
            messages: list[dict[str, str]] = []
            for message in messages_raw:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{source.path}:{line_number} message must be a mapping"
                    )
                messages.append(
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                )
            records.append(
                CausalInstructionRecord(
                    record_id=str(
                        raw.get(
                            "record_id",
                            f"{source.path.name}:{line_number}",
                        )
                    ),
                    messages=tuple(messages),
                    split=source.split,
                    task=str(raw.get("task", "causal_instruction")),
                    source_path=str(source.path),
                )
            )
    return records


def _sample_rank(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{record_id}".encode("utf-8")).hexdigest()
