"""Verified train/development/holdout boundaries for Phase 1 experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.utils.hashing import sha256_file

__all__ = [
    "Phase1SplitContract",
    "load_phase1_split_contract",
    "phase1_document_sort_key",
]


@dataclass(frozen=True, slots=True)
class Phase1SplitContract:
    """Immutable split IDs after both manifests pass leakage checks."""

    train_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    split_groups: tuple[tuple[str, str], ...]
    corpus_fingerprint_sha256: str
    model_manifest_sha256: str
    frozen_manifest_sha256: str

    def ids(self, split: str) -> tuple[str, ...]:
        """Return IDs for a public calibration split; holdout access is explicit."""

        if split == "train":
            return self.train_ids
        if split == "development":
            return self.development_ids
        if split == "holdout":
            return self.holdout_ids
        raise ValueError(f"Unsupported Phase 1 split {split!r}")

    @property
    def group_by_document(self) -> dict[str, str]:
        return dict(self.split_groups)


def load_phase1_split_contract(
    model_manifest_path: str | Path,
    frozen_manifest_path: str | Path,
) -> Phase1SplitContract:
    """Load and cross-check the model split without reading any annotation file.

    INVARIANT: model train/development IDs must partition the frozen source-train IDs, and neither
    may overlap the sealed holdout. The source manifest hash and corpus fingerprint fail closed.
    """

    model_path = Path(model_manifest_path)
    frozen_path = Path(frozen_manifest_path)
    model = _read_mapping(model_path)
    frozen = _read_mapping(frozen_path)
    if model.get("schema_version") != "phase1-model-training-split.v1":
        raise ValueError("Unsupported Phase 1 model split manifest")
    if frozen.get("schema_version") != "phase1-manual-gold-split.v1":
        raise ValueError("Unsupported frozen Phase 1 split manifest")
    if bool(model.get("round2_included", True)):
        raise ValueError("Round 2 cannot enter Phase 1 calibration")

    frozen_sha256 = sha256_file(frozen_path)
    if model.get("source_split_manifest_sha256") != frozen_sha256:
        raise ValueError("Model split does not reference the current frozen manifest")

    source_ids = _required_mapping(model, "source_document_ids")
    frozen_splits = _required_mapping(frozen, "splits")
    train_ids = _string_ids(source_ids.get("train"), "model train")
    development_ids = _string_ids(
        source_ids.get("development"),
        "model development",
    )
    frozen_train_ids = _section_ids(frozen_splits, "train")
    holdout_ids = _section_ids(frozen_splits, "holdout")
    if train_ids & development_ids:
        raise ValueError("Model train and development IDs overlap")
    if train_ids | development_ids != frozen_train_ids:
        raise ValueError(
            "Model train/development IDs do not partition frozen source train"
        )
    if (train_ids | development_ids) & holdout_ids:
        raise ValueError("Model calibration IDs overlap frozen holdout")

    excluded = _required_mapping(model, "excluded_holdout")
    if excluded.get("document_ids_sha256") != _ids_sha256(holdout_ids):
        raise ValueError("Model split excluded-holdout fingerprint is stale")
    frozen_corpus = _required_mapping(frozen, "corpus")
    corpus_fingerprint = frozen_corpus.get("fingerprint_sha256")
    if (
        not isinstance(corpus_fingerprint, str)
        or len(corpus_fingerprint) != 64
        or model.get("source_corpus_fingerprint_sha256") != corpus_fingerprint
    ):
        raise ValueError("Model and frozen manifests describe different corpora")

    selected_ids = train_ids | development_ids
    split_groups = _split_groups(model, selected_ids)
    return Phase1SplitContract(
        train_ids=tuple(sorted(train_ids, key=phase1_document_sort_key)),
        development_ids=tuple(
            sorted(development_ids, key=phase1_document_sort_key)
        ),
        holdout_ids=tuple(sorted(holdout_ids, key=phase1_document_sort_key)),
        split_groups=tuple(sorted(split_groups.items(), key=lambda item: phase1_document_sort_key(item[0]))),
        corpus_fingerprint_sha256=corpus_fingerprint,
        model_manifest_sha256=sha256_file(model_path),
        frozen_manifest_sha256=frozen_sha256,
    )


def phase1_document_sort_key(document_id: str) -> tuple[int, int | str]:
    """Sort numeric benchmark IDs numerically while supporting generic string IDs."""

    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)


def _split_groups(
    model: Mapping[str, Any],
    selected_ids: set[str],
) -> dict[str, str]:
    raw_groups = model.get("split_groups")
    if not isinstance(raw_groups, Mapping):
        return {document_id: f"document:{document_id}" for document_id in selected_ids}
    groups: dict[str, str] = {}
    for document_id in selected_ids:
        value = raw_groups.get(document_id)
        if value is None:
            value = raw_groups.get(f"phase1-manual-gold:{document_id}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Model split has no duplicate group for {document_id}")
        groups[document_id] = value
    return groups


def _section_ids(splits: Mapping[str, Any], name: str) -> set[str]:
    section = splits.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Frozen split manifest is missing {name}")
    return _string_ids(section.get("document_ids"), f"frozen {name}")


def _string_ids(value: Any, name: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} document IDs must be a string list")
    values = set(value)
    if len(values) != len(value):
        raise ValueError(f"{name} document IDs contain duplicates")
    return values


def _ids_sha256(values: Sequence[str] | set[str]) -> str:
    ordered = sorted(values, key=phase1_document_sort_key)
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase 1 manifest is missing {key}")
    return value


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return payload
