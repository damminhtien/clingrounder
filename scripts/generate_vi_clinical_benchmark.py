#!/usr/bin/env python3
"""Generate a deterministic synthetic benchmark snapshot for development.

The generator is intentionally separate from the benchmark runner.  It creates a larger,
redistributable fixture with split-specific templates, but marks the result as synthetic and
human-review pending.  It must never be used to claim clinical validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Callable, Iterable, Literal, Sequence

import yaml

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class Mention:
    text: str
    entity_type: str
    assertion: str
    code_system: str
    code: str


@dataclass(frozen=True, slots=True)
class Template:
    name: str
    language: str
    genre: str
    render: Callable[[Sequence[Mention]], tuple[str, list[dict[str, object]]]]


CONCEPTS = (
    Mention("sốt", "SYMPTOM", "NEGATED", "LOCAL", "SYMPTOM_FEVER"),
    Mention("ho", "SYMPTOM", "PRESENT", "LOCAL", "SYMPTOM_COUGH"),
    Mention("khó thở", "SYMPTOM", "PRESENT", "LOCAL", "SYMPTOM_DYSPNEA"),
    Mention("tăng huyết áp", "DISEASE", "HISTORICAL", "ICD-10", "I10"),
    Mention("đái tháo đường type 2", "DISEASE", "PRESENT", "ICD-10", "E11"),
    Mention("metformin", "DRUG", "PRESENT", "RxNorm", "6809"),
    Mention("đường huyết", "LAB_TEST", "PRESENT", "LOCAL", "LAB_TEST_GLUCOSE"),
    Mention("tăng cao", "LAB_RESULT", "PRESENT", "LOCAL", "LAB_RESULT_HIGH"),
)


EntityPayload = dict[str, object]


def _render(parts: Iterable[str | Mention]) -> tuple[str, list[EntityPayload]]:
    """Render mention objects while owning offsets at the point the text is assembled."""

    chunks: list[str] = []
    entities: list[EntityPayload] = []
    cursor = 0
    entity_number = 1
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
            cursor += len(part)
            continue
        start = cursor
        chunks.append(part.text)
        cursor += len(part.text)
        entities.append(
            {
                "id": f"g{entity_number}",
                "span": [start, cursor],
                "text": part.text,
                "type": part.entity_type,
                "assertion": part.assertion,
                "code_system": part.code_system,
                "code": part.code,
            }
        )
        entity_number += 1
    text = "".join(chunks)
    for entity in entities:
        span = entity["span"]
        entity_text = entity["text"]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
            or not isinstance(entity_text, str)
        ):
            raise AssertionError("generator produced an invalid entity payload")
        start, end = span
        assert text[start:end] == entity_text
    return text, entities


def _templates(split: Split) -> tuple[Template, ...]:
    """Use disjoint surface templates so test templates never leak into training."""

    def concept(concepts: Sequence[Mention], entity_type: str) -> Mention:
        return next(item for item in concepts if item.entity_type == entity_type)

    if split == "train":
        return (
            Template("train.negation", "vi", "clinical", lambda c: _render(["Không ghi nhận ", c[0], "."])),
            Template("train.history", "vi", "clinical", lambda c: _render(["Tiền sử ", c[3], "."])),
            Template("train.medication", "vi-en", "medication_list", lambda c: _render(["Đang dùng ", c[5], " 500 mg po bid."])),
            Template("train.symptoms", "vi", "clinical", lambda c: _render(["Hiện tại bệnh nhân ", c[1], " và ", c[2], "."])),
            Template("train.lab", "vi", "clinical", lambda c: _render(["Xét nghiệm ", concept(c, "LAB_TEST"), " cho kết quả ", concept(c, "LAB_RESULT"), "."])),
        )
    if split == "validation":
        return (
            Template("validation.negation", "vi", "clinical", lambda c: _render(["Bệnh nhân phủ nhận ", c[0], "."])),
            Template("validation.history", "vi-en", "clinical", lambda c: _render(["PMH: ", c[4], "; history of ", c[3], "."])),
            Template("validation.lab_like", "vi", "clinical", lambda c: _render(["Lý do khám: ", c[2], ", không kèm ", c[0], "."])),
            Template("validation.lab", "vi", "clinical", lambda c: _render(["Kết quả xét nghiệm ", concept(c, "LAB_TEST"), ": ", concept(c, "LAB_RESULT"), "."])),
        )
    return (
        Template("test.question", "vi", "question_answer", lambda c: _render(["Hỏi: Có ", c[1], " không? Đáp: Có, hiện có ", c[2], "."])),
        Template("test.repeated", "vi", "clinical", lambda c: _render(["Không sốt; theo dõi ", c[0], ". Hiện ", c[1], "."])),
        Template("test.mixed", "vi-en", "educational", lambda c: _render(["Clinical note: ", c[4], ", đang dùng ", c[5], "; symptoms: ", c[2], "."])),
        Template("test.lab", "vi-en", "clinical", lambda c: _render(["Lab panel: ", concept(c, "LAB_TEST"), " = ", concept(c, "LAB_RESULT"), "."])),
    )


def _relations_for(template_name: str, entities: Sequence[EntityPayload]) -> list[dict[str, str]]:
    """Add one explicit lab relation only to templates that render a test and its result."""

    if not template_name.endswith(".lab"):
        return []
    test = next(entity for entity in entities if entity["type"] == "LAB_TEST")
    result = next(entity for entity in entities if entity["type"] == "LAB_RESULT")
    return [
        {
            "id": "r1",
            "head": str(test["id"]),
            "tail": str(result["id"]),
            "type": "HAS_VALUE",
        }
    ]


def generate_snapshot(
    output_dir: str | Path,
    *,
    train_documents: int = 600,
    validation_documents: int = 100,
    test_documents: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Write a synthetic snapshot and return its deterministic manifest."""

    if min(train_documents, validation_documents, test_documents) < 1:
        raise ValueError("every split must contain at least one document")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[Split, int] = {
        "train": train_documents,
        "validation": validation_documents,
        "test": test_documents,
    }
    split_fingerprints: dict[str, str] = {}
    split_templates: dict[str, list[str]] = {}
    for split, count in counts.items():
        rows: list[str] = []
        templates = _templates(split)
        split_templates[split] = [template.name for template in templates]
        # SCALING: use one deterministic RNG per split so changing train size does not perturb
        # validation or test records.
        rng = random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split])
        for index in range(count):
            template = templates[index % len(templates)]
            concepts = list(CONCEPTS)
            rng.shuffle(concepts)
            text, entities = template.render(concepts)
            relations = _relations_for(template.name, entities)
            rows.append(
                json.dumps(
                    {
                        "document_id": f"synthetic-{split}-{index:04d}",
                        "text": text,
                        "metadata": {
                            "language": template.language,
                            "genre": template.genre,
                            "template_group": template.name,
                            "synthetic": "true",
                            "human_reviewed": "false",
                        },
                        "entities": entities,
                        "relations": relations,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        path = root / f"{split}.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        split_fingerprints[split] = _sha256(path)
    manifest: dict[str, object] = {
        "schema_version": "clingrounder.dataset-manifest.v1",
        "dataset": {
            "id": "vi-clinical-grounding-synthetic-v1",
            "version": "0.1.0",
            "status": "synthetic_pending_human_review",
            "license": "MIT",
            "human_reviewed": False,
            "seed": seed,
        },
        "splits": {
            split: {
                "path": f"{split}.jsonl",
                "documents": count,
                "sha256": split_fingerprints[split],
                "template_groups": split_templates[split],
            }
            for split, count in counts.items()
        },
        "entities": sorted({concept.entity_type for concept in CONCEPTS}),
        "assertions": sorted({concept.assertion for concept in CONCEPTS}),
        "code_systems": sorted({concept.code_system for concept in CONCEPTS}),
        "policy": {
            "template_groups_disjoint": True,
            "test_used_for_development": False,
            "private_data": False,
            "human_review_required_before_clinical_claim": True,
        },
    }
    (root / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-documents", type=int, default=600)
    parser.add_argument("--validation-documents", type=int, default=100)
    parser.add_argument("--test-documents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_snapshot(
        args.output_dir,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        test_documents=args.test_documents,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
