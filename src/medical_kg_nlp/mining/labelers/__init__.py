"""Source-native and model-backed proposal labelers for mined documents."""

from medical_kg_nlp.mining.labelers.brat import (
    BratArchiveLabelerAdapter,
    create_brat_archive_labeler,
)

__all__ = ["BratArchiveLabelerAdapter", "create_brat_archive_labeler"]
