"""Deterministic exact and near-duplicate grouping for leakage-safe splits."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence

from medical_kg_nlp.mining.records import MinedDocument

__all__ = ["StableTextDeduplicator"]

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class StableTextDeduplicator:
    """Group normalized duplicates and SimHash-near documents without quadratic scans."""

    def __init__(self, *, hamming_threshold: int = 3, bands: int = 4) -> None:
        if not 0 <= hamming_threshold <= 16:
            raise ValueError("hamming_threshold must be between 0 and 16")
        if bands <= 0 or 64 % bands:
            raise ValueError("bands must be a positive divisor of 64")
        self.hamming_threshold = hamming_threshold
        self.bands = bands

    def group(self, documents: Sequence[MinedDocument]) -> Mapping[str, str]:
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Cannot deduplicate documents with duplicate IDs")
        ordered = sorted(documents, key=lambda item: item.document_id)
        parent = list(range(len(ordered)))
        normalized_hashes: list[str] = []
        fingerprints: list[int] = []
        exact_buckets: dict[str, list[int]] = defaultdict(list)
        lsh_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        band_width = 64 // self.bands

        for index, document in enumerate(ordered):
            normalized = _normalize(document.text)
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            fingerprint = _simhash(normalized)
            normalized_hashes.append(normalized_hash)
            fingerprints.append(fingerprint)

            for candidate in exact_buckets[normalized_hash]:
                _union(parent, index, candidate)
            exact_buckets[normalized_hash].append(index)

            candidates: set[int] = set()
            for band in range(self.bands):
                shift = band * band_width
                key = (band, (fingerprint >> shift) & ((1 << band_width) - 1))
                candidates.update(lsh_buckets[key])
            for candidate in sorted(candidates):
                if (fingerprint ^ fingerprints[candidate]).bit_count() <= self.hamming_threshold:
                    _union(parent, index, candidate)
            for band in range(self.bands):
                shift = band * band_width
                key = (band, (fingerprint >> shift) & ((1 << band_width) - 1))
                lsh_buckets[key].append(index)

        members: dict[int, list[int]] = defaultdict(list)
        for index in range(len(ordered)):
            members[_find(parent, index)].append(index)
        group_ids: dict[int, str] = {}
        for root, indexes in members.items():
            # INVARIANT: every member of a duplicate cluster receives the same split group.
            identity = "\n".join(sorted(normalized_hashes[index] for index in indexes))
            group_ids[root] = f"duplicate:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        return {
            document.document_id: group_ids[_find(parent, index)]
            for index, document in enumerate(ordered)
        }


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _simhash(text: str) -> int:
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return 0
    shingles = (
        tokens
        if len(tokens) < 3
        else [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    )
    vector = [0] * 64
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    parent[max(left_root, right_root)] = min(left_root, right_root)
