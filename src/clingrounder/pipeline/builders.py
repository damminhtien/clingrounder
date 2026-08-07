"""Subsystem builders used by the pipeline composition root.

Each builder owns one concern of composition and returns already-wired ports.  The factory keeps
the ordering and resource ownership policy; this module keeps concrete construction details out
of the runner and out of application code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

from clingrounder.adapters.generative import (
    GenerativeListwiseRerankerAdapter,
    TransformersCausalLMRuntime,
)
from clingrounder.adapters.huggingface import (
    HuggingFaceCrossEncoderAdapter,
    HuggingFaceTextEncoderAdapter,
    HuggingFaceTokenClassifierAdapter,
)
from clingrounder.adapters.hybrid import HybridEntityExtractorAdapter
from clingrounder.adapters.medication import MedicationMentionEntityExtractorAdapter
from clingrounder.adapters.rules import (
    DictionaryCandidateAdapter,
    RuleEntityExtractorAdapter,
)
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.dictionaries.merge import merge_concept_entries
from clingrounder.governance import fingerprint_artifact
from clingrounder.kg.sqlite_repository import SQLiteKnowledgeGraphRepository
from clingrounder.linking.graph_evidence import GraphEvidenceReranker
from clingrounder.linking.graph_second_pass import GraphEvidenceSecondPass
from clingrounder.linking.learned_edits import load_learned_edit_model
from clingrounder.linking.linker import EntityLinker
from clingrounder.linking.mention_code_memory import load_mention_code_memory
from clingrounder.ner.extractors.contextual_alias import (
    ContextualAliasRule,
    load_contextual_alias_rules,
)
from clingrounder.ner.rule_ner import RuleBasedNER
from clingrounder.pipeline.ports import (
    CandidateRerankerPort,
    EntityExtractorPort,
)
from clingrounder.retrieval.dense_retriever import DenseRetrieverAdapter
from clingrounder.retrieval.rule_factory import build_rule_retrieval_pipeline
from clingrounder.retrieval.synonym_index import FaissSynonymVectorIndex
from clingrounder.schema.types import EntityType
from clingrounder.terminology import (
    CachedTerminologyRepository,
    CompositeTerminologyRepository,
    InMemoryTerminologyRepository,
    SQLiteTerminologyRepository,
    TerminologyRepository,
    terminology_cache_path,
)

if TYPE_CHECKING:
    from clingrounder.pipeline.factory import PipelineConfig

__all__ = [
    "EntityBuildResult",
    "GraphBuildResult",
    "LinkingBuildResult",
    "TerminologyBuildResult",
    "build_entity_extractor",
    "build_graph_repository",
    "build_linking",
    "build_terminology",
]


@dataclass(frozen=True)
class TerminologyBuildResult:
    """Terminology ports and recognition resources required by downstream builders."""

    repository: TerminologyRepository
    recognition_store: DictionaryStore
    contextual_alias_rules: tuple[ContextualAliasRule, ...]
    uses_sqlite_normalization: bool
    fingerprint: str


@dataclass(frozen=True)
class EntityBuildResult:
    """Entity extraction port plus the resources it may own."""

    extractor: EntityExtractorPort


@dataclass(frozen=True)
class GraphBuildResult:
    """Optional graph repository and document-level reranker."""

    repository: SQLiteKnowledgeGraphRepository | None
    document_reranker: GraphEvidenceSecondPass | None


@dataclass(frozen=True)
class LinkingBuildResult:
    """Candidate ports and optional dense retriever created as one linking subsystem."""

    candidate_adapter: DictionaryCandidateAdapter | None
    candidate_reranker: CandidateRerankerPort | None
    dense_retriever: DenseRetrieverAdapter | None


def build_terminology(config: PipelineConfig) -> TerminologyBuildResult:
    """Load recognition terminology and optionally compose the full SQLite release."""

    recognition_entries = DictionaryStore.load_entries_jsonl(
        config.terminology.recognition_dictionary_path,
        alias_overlay_path=config.terminology.alias_overlay_path,
    )
    if config.terminology.additional_recognition_dictionary_path is not None:
        recognition_paths = (
            config.terminology.additional_recognition_dictionary_path,
            *config.terminology.additional_recognition_dictionary_paths,
        )
    else:
        recognition_paths = config.terminology.additional_recognition_dictionary_paths
    for recognition_path in recognition_paths:
        recognition_entries.extend(DictionaryStore.load_entries_jsonl(recognition_path))
    recognition_store = DictionaryStore(merge_concept_entries(recognition_entries))
    recognition_repository = InMemoryTerminologyRepository(recognition_store)
    contextual_alias_rules = (
        load_contextual_alias_rules(config.terminology.contextual_alias_path)
        if config.terminology.contextual_alias_path is not None
        else ()
    )

    terminology_repository: TerminologyRepository = recognition_repository
    uses_sqlite_normalization = bool(config.terminology.normalization_dictionary_paths)
    if config.terminology.normalization_dictionary_paths:
        source_paths = config.terminology.normalization_dictionary_paths
        index_path = config.terminology.normalization_index_path or str(
            terminology_cache_path(
                config.terminology.terminology_cache_dir,
                source_paths,
                alias_overlay_paths=config.terminology.normalization_alias_overlay_paths,
            )
        )
        sqlite_repository = SQLiteTerminologyRepository(
            index_path,
            expected_source_paths=source_paths,
            expected_alias_overlay_paths=config.terminology.normalization_alias_overlay_paths,
            expected_normalization_version=config.normalization_contract.version,
        )
        terminology_repository = CompositeTerminologyRepository(
            (recognition_repository, sqlite_repository)
        )
    if config.terminology.terminology_query_cache_size:
        terminology_repository = CachedTerminologyRepository(
            terminology_repository,
            max_size=config.terminology.terminology_query_cache_size,
        )

    return TerminologyBuildResult(
        repository=terminology_repository,
        recognition_store=recognition_store,
        contextual_alias_rules=contextual_alias_rules,
        uses_sqlite_normalization=uses_sqlite_normalization,
        fingerprint=_terminology_fingerprint(terminology_repository, config),
    )


def build_entity_extractor(
    config: PipelineConfig,
    terminology: TerminologyBuildResult,
) -> EntityBuildResult:
    """Compose rule, model, and medication structure adapters for entity extraction."""

    model_config = config.models
    dictionary = RuleEntityExtractorAdapter(
        RuleBasedNER(
            terminology.recognition_store,
            contextual_alias_rules=terminology.contextual_alias_rules,
            false_positive_path=config.terminology.false_positive_path,
        )
    )
    if model_config.entity_extractor is None:
        return EntityBuildResult(dictionary)

    model: EntityExtractorPort = MedicationMentionEntityExtractorAdapter(
        HuggingFaceTokenClassifierAdapter(
            model_config.entity_extractor,
            label_map=dict(model_config.entity_label_map),
            stride=model_config.entity_stride,
            confidence_thresholds=dict(model_config.entity_confidence_thresholds),
            default_confidence_threshold=model_config.entity_default_confidence_threshold,
        )
    )
    if model_config.entity_combine_with_dictionary:
        return EntityBuildResult(HybridEntityExtractorAdapter(model=model, dictionary=dictionary))
    return EntityBuildResult(model)


def build_graph_repository(config: PipelineConfig) -> GraphBuildResult:
    """Build graph-backed retrieval and evidence only when the profile explicitly enables it."""

    options = config.options
    needs_graph = "kg_exact" in options.candidate_sources or options.enable_graph_evidence_reranking
    if not needs_graph:
        return GraphBuildResult(None, None)
    if config.terminology.knowledge_graph_index_path is None:
        raise ValueError(
            "Graph-backed retrieval/reranking requires terminology.knowledge_graph_index_path"
        )
    repository = SQLiteKnowledgeGraphRepository(config.terminology.knowledge_graph_index_path)
    if not options.enable_graph_evidence_reranking:
        return GraphBuildResult(repository, None)
    reranker = GraphEvidenceSecondPass(
        GraphEvidenceReranker(
            repository,
            relation_types=options.graph_evidence_relation_types,
            min_support=options.graph_evidence_min_support,
            max_bonus=options.graph_evidence_max_bonus,
            cache_size=options.graph_evidence_cache_size,
        )
    )
    return GraphBuildResult(repository, reranker)


def build_linking(
    config: PipelineConfig,
    terminology: TerminologyBuildResult,
    graph: GraphBuildResult,
) -> LinkingBuildResult:
    """Compose retrieval, structured linking, and optional model reranking."""

    options = config.options
    if not options.enable_linking:
        return LinkingBuildResult(None, None, None)

    mention_code_memory = (
        load_mention_code_memory(config.terminology.mention_code_memory_path)
        if "mention_memory" in options.candidate_sources
        and config.terminology.mention_code_memory_path is not None
        else None
    )
    learned_edit_model = (
        load_learned_edit_model(config.terminology.learned_edit_path)
        if "learned_edit" in options.candidate_sources and config.terminology.learned_edit_path is not None
        else None
    )
    dense_retriever = _build_dense_retriever(config, terminology.repository)
    linker = EntityLinker(
        build_rule_retrieval_pipeline(
            terminology.repository,
            approximate_store=terminology.recognition_store,
            abbreviation_path=config.terminology.abbreviation_path,
            max_candidates=options.max_candidates,
            retrieval_sources=options.candidate_sources,
            mention_memory_path=config.terminology.reviewed_mention_path,
            use_fts_for_bm25=terminology.uses_sqlite_normalization,
            knowledge_graph_repository=graph.repository,
            mention_code_memory=mention_code_memory,
            learned_edit_model=learned_edit_model,
            dense_retriever=dense_retriever,
        ),
        terminology.repository,
        assignment_threshold=options.link_assignment_threshold,
        assignment_margin=options.link_assignment_margin,
        candidate_threshold=options.link_candidate_threshold,
        candidate_relative_margin=options.link_candidate_relative_margin,
        max_qualified_candidates=options.link_max_qualified_candidates,
        candidate_thresholds_by_entity_type={
            EntityType(entity_type): threshold
            for entity_type, threshold in options.link_candidate_thresholds_by_type
        },
        candidate_thresholds_by_source=dict(options.link_candidate_thresholds_by_source),
        emit_probabilities_by_source=dict(options.link_emit_probabilities_by_source),
        enforce_rxnorm_structure=options.link_enforce_rxnorm_structure,
    )
    candidate_adapter = DictionaryCandidateAdapter(linker)
    candidate_reranker: CandidateRerankerPort = candidate_adapter
    if config.models.candidate_reranker is not None:
        candidate_reranker = HuggingFaceCrossEncoderAdapter(
            config.models.candidate_reranker,
            model_weight=config.models.candidate_reranker_weight,
            positive_label_index=config.models.candidate_positive_label_index,
            base_reranker=candidate_adapter,
            max_pairs_per_batch=config.models.candidate_reranker.max_pairs_per_batch,
            max_tokens=config.models.candidate_reranker.max_tokens,
        )
    elif config.models.candidate_listwise_reranker is not None:
        listwise = config.models.candidate_listwise_reranker
        runtime = TransformersCausalLMRuntime(
            model_id=listwise.model.model_id,
            revision=listwise.model.revision,
            device=listwise.model.device,
            dtype=listwise.dtype,
            local_files_only=listwise.local_files_only,
        )
        candidate_reranker = GenerativeListwiseRerankerAdapter(
            runtime,
            terminology.repository,
            generation=listwise.generation,
            base_reranker=candidate_adapter,
            candidate_limit=listwise.candidate_limit,
            model_weight=listwise.model_weight,
            shuffle_seed=listwise.shuffle_seed,
            structured_retries=listwise.structured_retries,
        )
    return LinkingBuildResult(candidate_adapter, candidate_reranker, dense_retriever)


def _build_dense_retriever(
    config: PipelineConfig,
    repository: TerminologyRepository,
) -> DenseRetrieverAdapter | None:
    if "dense" not in config.options.candidate_sources:
        return None
    model = config.models.candidate_dense_encoder
    if model is None or config.terminology.synonym_index_path is None:
        raise ValueError(
            "Dense retrieval requires models.candidate_dense_encoder and "
            "terminology.synonym_index_path"
        )
    if config.terminology.synonym_index_terminology_fingerprint is None:
        raise ValueError("Dense retrieval requires a terminology fingerprint")
    return DenseRetrieverAdapter(
        encoder=HuggingFaceTextEncoderAdapter(model),
        index=FaissSynonymVectorIndex(
            config.terminology.synonym_index_path,
            expected_model_id=model.model_id,
            expected_revision=model.revision,
            expected_terminology_fingerprint=config.terminology.synonym_index_terminology_fingerprint,
        ),
        repository=repository,
    )


def _terminology_fingerprint(repository: object, config: PipelineConfig) -> str:
    metadata = getattr(repository, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("input_fingerprint", "source_fingerprint"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    if config.terminology.synonym_index_terminology_fingerprint:
        return config.terminology.synonym_index_terminology_fingerprint
    paths = (
        config.terminology.recognition_dictionary_path,
        config.terminology.abbreviation_path,
        config.terminology.alias_overlay_path,
        *config.terminology.additional_recognition_dictionary_paths,
        *config.terminology.normalization_dictionary_paths,
    )
    digests: list[str] = []
    for path in paths:
        if path is None:
            continue
        try:
            digests.append(fingerprint_artifact(path))
        except (OSError, ValueError):
            return "unknown"
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest() if digests else "unknown"
