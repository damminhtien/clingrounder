from __future__ import annotations
from pathlib import Path
from typing import Protocol

from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction


class DatasetAdapter(Protocol):
    def load_documents(self, path: str | Path) -> list[ClinicalDocument]:
        ...

    def load_gold(self, path: str | Path) -> list[ClinicalPrediction]:
        ...

