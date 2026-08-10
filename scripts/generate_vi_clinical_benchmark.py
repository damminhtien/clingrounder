#!/usr/bin/env python3
"""Generate a deterministic synthetic benchmark snapshot for development.

The generator is intentionally separate from the benchmark runner.  It creates a larger,
redistributable fixture with split-specific templates, but marks the result as synthetic and
human-review pending.  It must never be used to claim clinical validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
    code_system: str
    code: str | None
    assertion: str = "PRESENT"


@dataclass(frozen=True, slots=True)
class Template:
    name: str
    language: str
    genre: str
    render: Callable[[Sequence[Mention], int], tuple[str, list[dict[str, object]]]]


CONCEPTS = (
    Mention("sốt", "SYMPTOM", "LOCAL", "SYMPTOM_FEVER"),
    Mention("ho", "SYMPTOM", "LOCAL", "SYMPTOM_COUGH"),
    Mention("khó thở", "SYMPTOM", "LOCAL", "SYMPTOM_DYSPNEA"),
    Mention("đau ngực", "SYMPTOM", "LOCAL", "SYMPTOM_CHEST_PAIN"),
    Mention("đau đầu", "SYMPTOM", "LOCAL", "SYMPTOM_HEADACHE"),
    Mention("buồn nôn và nôn", "SYMPTOM", "LOCAL", "SYMPTOM_NAUSEA_VOMITING"),
    Mention("tăng huyết áp", "DISEASE", "ICD-10", "I10"),
    Mention("đái tháo đường type 2", "DISEASE", "ICD-10", "E11"),
    Mention("viêm phổi", "DISEASE", "ICD-10", "J18.9"),
    Mention("hen phế quản", "DISEASE", "ICD-10", "J45"),
    Mention("ung thư phổi", "DISEASE", "ICD-10", "C34"),
    Mention("nhồi máu cơ tim cấp", "DISEASE", "ICD-10", "I21.9"),
    Mention("metformin", "DRUG", "RxNorm", "6809"),
    Mention("salbutamol", "DRUG", "RxNorm", "435"),
    Mention("aspirin", "DRUG", "RxNorm", "1191"),
    Mention("amoxicillin", "DRUG", "RxNorm", "723"),
    Mention("atorvastatin", "DRUG", "RxNorm", "83367"),
    Mention("omeprazole", "DRUG", "RxNorm", "7646"),
    Mention("đường huyết", "LAB_TEST", "LOCAL", "GLUCOSE"),
    Mention("creatinin", "LAB_TEST", "LOCAL", "CREATININE"),
    Mention("HbA1c", "LAB_TEST", "LOCAL", "HBA1C"),
    Mention("điện tâm đồ", "LAB_TEST", "LOCAL", "ECG"),
    Mention("tăng", "LAB_RESULT", "NONE", None),
    Mention("giảm", "LAB_RESULT", "NONE", None),
    Mention("bình thường", "LAB_RESULT", "NONE", None),
    Mention("bất thường", "LAB_RESULT", "NONE", None),
)
SUPPORTED_ASSERTIONS = ("PRESENT", "NEGATED", "HISTORICAL", "FAMILY", "POSSIBLE")


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
    """Use role-safe, split-disjoint templates with assertions owned by their cues."""

    def select(concepts: Sequence[Mention], entity_type: str, offset: int = 0) -> Mention:
        matches = [item for item in concepts if item.entity_type == entity_type]
        return matches[offset % len(matches)]

    def annotated(mention: Mention, assertion: str) -> Mention:
        return replace(mention, assertion=assertion)

    def context(index: int) -> tuple[str, int]:
        # The age/day pair keeps large generated splits text-distinct without changing labels.
        age = 18 + index % 73
        duration_days = 1 + (index // 73) % 30
        return f"Bệnh nhân {age} tuổi", duration_days

    def current_symptoms(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, days = context(index)
        first = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        second = annotated(select(concepts, "SYMPTOM", 1), "PRESENT")
        return _render([subject, " hiện ", first, ", kèm ", second, f" trong {days} ngày."])

    def negated_symptom(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, days = context(index)
        symptom = annotated(select(concepts, "SYMPTOM"), "NEGATED")
        return _render([subject, " không ghi nhận ", symptom, f" trong {days} ngày qua."])

    def historical_disease(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "HISTORICAL")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        return _render([subject, " có tiền sử ", disease, ", hiện xuất hiện ", symptom, "."])

    def family_disease(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "FAMILY")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        return _render(["Mẹ của ", subject.casefold(), " mắc ", disease, "; người bệnh hiện ", symptom, "."])

    def medication(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        drug = annotated(select(concepts, "DRUG"), "PRESENT")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        if split == "train":
            return _render(
                [subject, " có thuốc hiện tại là ", drug, "; đồng thời ghi nhận ", symptom, "."]
            )
        return _render(
            ["Danh sách thuốc của ", subject.casefold(), " gồm ", drug, "; chỉ định do ", symptom, "."]
        )

    def lab(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        test = annotated(select(concepts, "LAB_TEST"), "PRESENT")
        result_options = [
            item
            for item in concepts
            if item.entity_type == "LAB_RESULT"
            and (
                item.text in {"bình thường", "bất thường"}
                if test.text == "điện tâm đồ"
                else item.text in {"tăng", "giảm", "bình thường"}
            )
        ]
        result = annotated(result_options[0], "PRESENT")
        lead_by_split = {
            "train": "Kết quả thường quy của ",
            "validation": "Đánh giá cận lâm sàng ở ",
            "test": "Bảng xét nghiệm của ",
        }
        return _render([lead_by_split[split], subject.casefold(), ": ", test, " ", result, "."])

    def validation_negation(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        symptom = annotated(select(concepts, "SYMPTOM"), "NEGATED")
        return _render([subject, " phủ nhận triệu chứng ", symptom, "."])

    def validation_history(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        first = annotated(select(concepts, "DISEASE"), "HISTORICAL")
        second = annotated(select(concepts, "DISEASE", 1), "HISTORICAL")
        return _render([subject, " có PMH gồm ", first, " và history of ", second, "."])

    def possible_disease(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "POSSIBLE")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        if split == "validation":
            return _render(
                [subject, " được đánh giá khả năng ", disease, "; đồng thời ghi nhận ", symptom, "."]
            )
        return _render(
            [subject, " đang được theo dõi vì có thể mắc ", disease, "; biểu hiện kèm theo là ", symptom, "."]
        )

    def validation_family(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "FAMILY")
        symptom = annotated(select(concepts, "SYMPTOM"), "NEGATED")
        return _render(["Cha của ", subject.casefold(), " từng mắc ", disease, "; bệnh nhân không ", symptom, "."])

    def question_answer(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, days = context(index)
        symptom = select(concepts, "SYMPTOM")
        first = annotated(symptom, "PRESENT")
        repeated = annotated(symptom, "PRESENT")
        return _render(
            ["Hỏi: ", subject, " có ", first, " không? Đáp: Có ", repeated, f" từ {days} ngày nay."]
        )

    def repeated_context(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, days = context(index)
        symptom = select(concepts, "SYMPTOM")
        negated = annotated(symptom, "NEGATED")
        present = annotated(symptom, "PRESENT")
        return _render([subject, " ban đầu không ", negated, "; sau ", str(days), " ngày xuất hiện ", present, "."])

    def mixed_note(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "PRESENT")
        drug = annotated(select(concepts, "DRUG"), "PRESENT")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        return _render(["Clinical note: ", subject.casefold(), " có ", disease, ", dùng ", drug, "; symptom: ", symptom, "."])

    def test_family(concepts: Sequence[Mention], index: int) -> tuple[str, list[EntityPayload]]:
        subject, _ = context(index)
        disease = annotated(select(concepts, "DISEASE"), "FAMILY")
        symptom = annotated(select(concepts, "SYMPTOM"), "PRESENT")
        return _render([subject, " hiện ", symptom, "; chị gái có tiền sử ", disease, "."])

    if split == "train":
        return (
            Template("train.current", "vi", "clinical", current_symptoms),
            Template("train.negation", "vi", "clinical", negated_symptom),
            Template("train.history", "vi", "clinical", historical_disease),
            Template("train.family", "vi", "family_history", family_disease),
            Template("train.medication", "vi-en", "medication_list", medication),
            Template("train.lab", "vi", "lab_report", lab),
        )
    if split == "validation":
        return (
            Template("validation.negation", "vi", "clinical", validation_negation),
            Template("validation.history", "vi-en", "clinical", validation_history),
            Template("validation.possible", "vi", "clinical", possible_disease),
            Template("validation.family", "vi", "family_history", validation_family),
            Template("validation.lab", "vi", "lab_report", lab),
        )
    return (
        Template("test.question", "vi", "question_answer", question_answer),
        Template("test.repeated", "vi", "clinical", repeated_context),
        Template("test.mixed", "vi-en", "educational", mixed_note),
        Template("test.possible", "vi", "clinical", possible_disease),
        Template("test.family", "vi", "family_history", test_family),
        Template("test.medication", "vi-en", "medication_list", medication),
        Template("test.lab", "vi-en", "lab_report", lab),
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
            text, entities = template.render(concepts, index)
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
            "version": "0.2.0",
            "status": "synthetic_pending_human_review",
            "language": ["vi", "vi-en"],
            "license": "MIT",
            "license_url": "https://opensource.org/license/mit",
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
        "assertions": list(SUPPORTED_ASSERTIONS),
        "code_systems": sorted({concept.code_system for concept in CONCEPTS}),
        "policy": {
            "template_groups_disjoint": True,
            "test_used_for_development": False,
            "private_data": False,
            "human_review_required_before_clinical_claim": True,
        },
        "review": {
            "status": "pending",
            "reviewers_required": 2,
            "double_review_fraction": 0.1,
            "agreement_targets": {
                "span_type": 0.90,
                "assertion": 0.85,
                "relation": 0.80,
            },
            "adjudication_required": True,
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
