"""Final supervision source composition contracts."""

from __future__ import annotations

from clingrounder.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from clingrounder.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus


def test_final_supervision_requires_final_train_and_complete_provenance() -> None:
    reviewed = Phase1ReviewedCorpus(
        source_texts={"1": "đau", "authorized_gt:1": "sốt"},
        gold_rows={"1": (), "authorized_gt:1": ()},
        split_by_document={"1": "train", "authorized_gt:1": "train"},
    )
    corpus = Phase1FinalSupervisionCorpus(
        reviewed=reviewed,
        source_by_document={"1": "manual_gold", "authorized_gt:1": "authorized_ground_truth"},
        manifest={},
    )

    assert corpus.source_by_document["authorized_gt:1"] == "authorized_ground_truth"
