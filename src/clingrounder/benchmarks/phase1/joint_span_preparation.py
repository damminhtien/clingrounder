"""Shared preparation of joint span/type supervision from governed proposal sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.joint_span_dataset import (
    build_phase1_joint_span_dataset,
    write_phase1_joint_span_dataset,
)
from clingrounder.benchmarks.phase1.joint_span_sources import (
    build_phase1_dictionary_trie_source_rows,
    build_phase1_joint_span_proposal_matrix,
    build_phase1_medication_parser_source_rows,
    build_phase1_rule_source_rows,
    load_phase1_joint_span_source_rows,
    verify_phase1_joint_span_source_artifact,
)
from clingrounder.benchmarks.phase1.proposal_features import ProposalSourceRole
from clingrounder.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.mining.io import write_json
from clingrounder.ner.dictionary_matcher import DictionaryMatcher
from clingrounder.utils.hashing import sha256_directory, sha256_file

__all__ = ["prepare_phase1_joint_span_supervision"]


def prepare_phase1_joint_span_supervision(
    corpus: Phase1ReviewedCorpus,
    dictionary: DictionaryStore,
    *,
    source_dataset_by_document: Mapping[str, str],
    oof_group_by_document: Mapping[str, str],
    genre_by_document: Mapping[str, str] | None,
    supervision_manifest: Mapping[str, Any],
    model_sources: Mapping[str, tuple[Path, ProposalSourceRole | str]],
    output_dir: str | Path,
    purpose: str,
    require_model_source: bool,
) -> dict[str, Any]:
    """Prepare a reproducible lattice dataset from a corpus and independently pinned sources.

    RuleNER, the independent dictionary trie, and the medication parser are separate evidence
    sources.  A final production verifier may require a model source, while bootstrap OOF data
    can be constructed first to validate the mixed-genre data contract before GPU artifacts exist.
    """

    if not purpose.strip():
        raise ValueError("Joint span preparation purpose is required")
    if require_model_source and not model_sources:
        raise ValueError("Joint span preparation requires at least one model proposal source")
    if {"rule", "dictionary_trie", "medication_parser"} & set(model_sources):
        raise ValueError("Joint span built-in source names cannot be overridden")
    if any(not name.strip() for name in model_sources):
        raise ValueError("Joint span model source names must be non-empty")
    _validate_document_mappings(
        corpus,
        source_dataset_by_document,
        oof_group_by_document,
        genre_by_document,
    )

    source_roles: dict[str, ProposalSourceRole] = {
        "rule": ProposalSourceRole.RULE,
        "dictionary_trie": ProposalSourceRole.RULE,
        "medication_parser": ProposalSourceRole.RULE,
    }
    source_rows = {
        "rule": build_phase1_rule_source_rows(corpus, dictionary),
        "dictionary_trie": build_phase1_dictionary_trie_source_rows(corpus, dictionary),
        "medication_parser": build_phase1_medication_parser_source_rows(corpus, dictionary),
    }
    source_descriptors: dict[str, dict[str, str]] = {}
    for name, (path, role) in sorted(model_sources.items()):
        source_path = Path(path)
        if not source_path.is_dir():
            raise FileNotFoundError(source_path)
        parsed_role = ProposalSourceRole(role)
        if parsed_role is ProposalSourceRole.VERIFIER:
            raise ValueError("Verifier-only evidence cannot be a joint span lattice source")
        artifact = verify_phase1_joint_span_source_artifact(source_path, corpus)
        source_roles[name] = parsed_role
        source_rows[name] = load_phase1_joint_span_source_rows(source_path, corpus)
        # INVARIANT: directory contents, not a user-provided label, pin the source revision.
        source_descriptors[name] = {
            "path": str(source_path),
            "sha256": sha256_directory(source_path),
            "role": parsed_role.value,
            "manifest_sha256": artifact["manifest_sha256"],
            "schema_version": artifact["schema_version"],
        }

    matrix = build_phase1_joint_span_proposal_matrix(
        corpus,
        source_rows,
        source_roles=source_roles,
    )
    dataset = build_phase1_joint_span_dataset(
        corpus,
        matrix,
        source_roles=source_roles,
        source_dataset_by_document=source_dataset_by_document,
        oof_group_by_document=oof_group_by_document,
        genre_by_document=genre_by_document,
        # Recognition aliases provide bounded candidates around independent source proposals only.
        dictionary_matcher=DictionaryMatcher(dictionary.aliases_for_ner()),
    )
    output = Path(output_dir)
    paths = write_phase1_joint_span_dataset(dataset, output)
    preparation_manifest = {
        "schema_version": "phase1-joint-span-preparation.v1",
        "purpose": purpose,
        "promotion": "official_submission_metrics_only",
        "supervision": dict(supervision_manifest),
        "sources": {
            "rule": {
                "role": ProposalSourceRole.RULE.value,
                "dictionary_entry_count": len(dictionary.entries),
            },
            "dictionary_trie": {
                "role": ProposalSourceRole.RULE.value,
                "kind": "independent_typed_lexical_proposals",
                "dictionary_entry_count": len(dictionary.entries),
            },
            "medication_parser": {
                "role": ProposalSourceRole.RULE.value,
                "kind": "structured_full_sig",
            },
            **source_descriptors,
        },
        "dataset": {
            **paths,
            "manifest_sha256": sha256_file(output / "manifest.json"),
        },
        "policy": {
            "round2_included": False,
            "friend31_included": False,
            "selection": "deferred_to_joint_span_verifier",
            "raw_offsets": "immutable",
            "model_source_required": require_model_source,
        },
    }
    write_json(output / "preparation_manifest.json", preparation_manifest)
    return preparation_manifest


def _validate_document_mappings(
    corpus: Phase1ReviewedCorpus,
    source_dataset_by_document: Mapping[str, str],
    oof_group_by_document: Mapping[str, str],
    genre_by_document: Mapping[str, str] | None,
) -> None:
    document_ids = set(corpus.source_texts)
    if document_ids != set(source_dataset_by_document):
        raise ValueError("Joint span source dataset mapping must cover the corpus exactly")
    if document_ids != set(oof_group_by_document):
        raise ValueError("Joint span OOF group mapping must cover the corpus exactly")
    if any(not value.strip() for value in source_dataset_by_document.values()):
        raise ValueError("Joint span source dataset values must be non-empty")
    if any(not value.strip() for value in oof_group_by_document.values()):
        raise ValueError("Joint span OOF group values must be non-empty")
    if genre_by_document is not None and document_ids != set(genre_by_document):
        raise ValueError("Joint span genre mapping must cover the corpus exactly")
