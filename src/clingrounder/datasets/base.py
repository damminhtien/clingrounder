from __future__ import annotations
from pathlib import Path
from typing import Protocol

from clingrounder.schema.document import ClinicalDocument
from clingrounder.schema.output import ClinicalPrediction


class DatasetAdapter(Protocol):
    def load_documents(self, path: str | Path) -> list[ClinicalDocument]:
        ...

    def load_gold(self, path: str | Path) -> list[ClinicalPrediction]:
        ...

