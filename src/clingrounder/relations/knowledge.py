"""Versioned, code-based relation knowledge for the rule baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clingrounder.schema.types import CodeSystem, RelationType

__all__ = ["KnownRelation", "KnownRelationRepository"]


@dataclass(frozen=True)
class KnownRelation:
    """One reviewed ontology relation loaded from an external resource."""

    subject_code_system: CodeSystem
    subject_code: str
    relation_type: RelationType
    object_code_system: CodeSystem
    object_code: str
    source: str
    source_version: str
    review_status: str
    valid_from: str | None = None
    valid_to: str | None = None


class KnownRelationRepository:
    """Read-only lookup for reviewed code-to-code relation records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._relations = self._load()

    def find(
        self,
        subject_code_system: CodeSystem,
        subject_code: str,
        relation_type: RelationType,
        object_code_system: CodeSystem,
        object_code: str,
    ) -> KnownRelation | None:
        return self._relations.get(
            (
                subject_code_system,
                subject_code,
                relation_type,
                object_code_system,
                object_code,
            )
        )

    def _load(self) -> dict[tuple[CodeSystem, str, RelationType, CodeSystem, str], KnownRelation]:
        relations: dict[
            tuple[CodeSystem, str, RelationType, CodeSystem, str], KnownRelation
        ] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    relation = KnownRelation(
                        subject_code_system=CodeSystem(raw["subject_code_system"]),
                        subject_code=str(raw["subject_code"]),
                        relation_type=RelationType(raw["relation_type"]),
                        object_code_system=CodeSystem(raw["object_code_system"]),
                        object_code=str(raw["object_code"]),
                        source=str(raw["source"]),
                        source_version=str(raw["source_version"]),
                        review_status=str(raw["review_status"]),
                        valid_from=raw.get("valid_from"),
                        valid_to=raw.get("valid_to"),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid relation record at {self.path}:{line_number}"
                    ) from error
                if relation.review_status not in {"reviewed", "approved"}:
                    continue
                key = (
                    relation.subject_code_system,
                    relation.subject_code,
                    relation.relation_type,
                    relation.object_code_system,
                    relation.object_code,
                )
                if key in relations:
                    raise ValueError(f"Duplicate relation record at {self.path}:{line_number}")
                relations[key] = relation
        return relations
