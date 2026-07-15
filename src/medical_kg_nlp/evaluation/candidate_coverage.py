from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_CODABLE_TYPES,
    PHASE1_TYPE_BY_ENTITY_TYPE,
    expected_code_system,
)
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.types import CodeSystem


_RXNORM_INGREDIENT_TTYS = frozenset({"IN", "MIN", "PIN"})
_RXNORM_BRAND_TTYS = frozenset({"BN"})
_RXNORM_CLINICAL_TTYS = frozenset({"SCD", "SBD"})


@dataclass(frozen=True)
class TerminologyCoverageIndex:
    """Minimal terminology metadata needed by evaluation, without building alias indexes."""

    entries_by_code: Mapping[tuple[CodeSystem, str], tuple[ConceptEntry, ...]]
    icd_parent_codes: frozenset[str]

    @classmethod
    def from_entries(cls, entries: Iterable[ConceptEntry]) -> "TerminologyCoverageIndex":
        by_code: dict[tuple[CodeSystem, str], list[ConceptEntry]] = defaultdict(list)
        parent_codes: set[str] = set()
        for entry in entries:
            if entry.code is not None:
                by_code[(entry.code_system, str(entry.code))].append(entry)
            if entry.code_system == CodeSystem.ICD10:
                if entry.parent_code:
                    parent_codes.add(entry.parent_code)
                parent_codes.update(entry.parents)
        return cls(
            entries_by_code={key: tuple(values) for key, values in by_code.items()},
            icd_parent_codes=frozenset(parent_codes),
        )

    def bucket(self, code_system: CodeSystem, codes: frozenset[str]) -> str:
        if not codes:
            return "gold_empty"
        categories = {self._code_bucket(code_system, code) for code in codes}
        return next(iter(categories)) if len(categories) == 1 else "mixed"

    def _code_bucket(self, code_system: CodeSystem, code: str) -> str:
        entries = self.entries_by_code.get((code_system, code), ())
        if code_system == CodeSystem.RXNORM:
            ttys = {str(entry.rxnorm_tty or "").upper() for entry in entries}
            if ttys & _RXNORM_INGREDIENT_TTYS:
                return "rxnorm_ingredient"
            if ttys & _RXNORM_BRAND_TTYS:
                return "rxnorm_brand"
            if ttys & _RXNORM_CLINICAL_TTYS:
                return "rxnorm_scd_sbd"
            return "rxnorm_other" if entries else "terminology_unknown"
        if code_system == CodeSystem.ICD10:
            if code in self.icd_parent_codes:
                return "icd_parent"
            return "icd_leaf" if entries else "terminology_unknown"
        return "other_code_system" if entries else "terminology_unknown"


def build_candidate_coverage_report(
    predictions: Iterable[ClinicalPrediction],
    gold_by_document: Mapping[str, list[dict[str, Any]]],
    *,
    terminology_entries: Iterable[ConceptEntry] = (),
) -> dict[str, Any]:
    """Measure candidate behavior by nullness and terminology granularity.

    Only exact span/type matches are included so entity boundary errors cannot masquerade as
    candidate-policy errors. This report is diagnostic and never promotes a policy by itself.
    """

    prediction_by_document = {prediction.document_id: prediction for prediction in predictions}
    index = TerminologyCoverageIndex.from_entries(terminology_entries)
    aggregate: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    counters: Counter[str] = Counter()

    for document_id, gold_rows in gold_by_document.items():
        prediction = prediction_by_document.get(document_id)
        if prediction is None:
            counters["missing_prediction_document"] += 1
            continue
        gold_index = {
            (int(row["position"][0]), int(row["position"][1]), str(row["type"])): row
            for row in gold_rows
            if isinstance(row.get("position"), list | tuple)
            and len(row["position"]) == 2
        }
        for entity in prediction.entities:
            phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type)
            if phase1_type not in PHASE1_CODABLE_TYPES:
                continue
            gold = gold_index.get((entity.span[0], entity.span[1], phase1_type))
            if gold is None:
                counters["entity_without_exact_gold_match"] += 1
                continue
            code_system = expected_code_system(phase1_type)
            if code_system is None:
                continue
            gold_codes = _string_set(gold.get("candidates"))
            predicted_codes = frozenset(
                str(candidate.code)
                for candidate in entity.candidates
                if candidate.qualified
                and candidate.code is not None
                and candidate.code_system == code_system
            )
            bucket = index.bucket(code_system, gold_codes)
            row = aggregate[(code_system.value, bucket)]
            row["support"] += 1
            row["gold_nonempty"] += int(bool(gold_codes))
            row["prediction_nonempty"] += int(bool(predicted_codes))
            row["exact_set_correct"] += int(predicted_codes == gold_codes)
            row["jaccard_micros"] += round(_jaccard(gold_codes, predicted_codes) * 1_000_000)
            if gold_codes:
                row["top1_support"] += 1
                top = next(
                    (
                        str(candidate.code)
                        for candidate in entity.candidates
                        if candidate.qualified
                        and candidate.code is not None
                        and candidate.code_system == code_system
                    ),
                    None,
                )
                row["top1_correct"] += int(top in gold_codes)

    buckets = []
    for (code_system_name, bucket), values in sorted(aggregate.items()):
        support = values["support"]
        top1_support = values["top1_support"]
        buckets.append(
            {
                "code_system": code_system_name,
                "bucket": bucket,
                "support": support,
                "gold_nonempty": values["gold_nonempty"],
                "prediction_coverage": round(values["prediction_nonempty"] / support, 6),
                "exact_set_accuracy": round(values["exact_set_correct"] / support, 6),
                "mean_jaccard": round(values["jaccard_micros"] / support / 1_000_000, 6),
                "top1_accuracy": (
                    round(values["top1_correct"] / top1_support, 6)
                    if top1_support
                    else None
                ),
            }
        )
    return {
        "schema_version": "candidate-coverage.v1",
        "exact_match_only": True,
        "buckets": buckets,
        "counters": dict(sorted(counters.items())),
    }


def _string_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return frozenset()
    return frozenset(str(item) for item in value if item is not None and str(item).strip())


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
