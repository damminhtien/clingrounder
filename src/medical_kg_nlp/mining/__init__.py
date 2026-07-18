"""Reusable acquisition, curation, and dataset-snapshot primitives."""

from medical_kg_nlp.mining.catalog import DuckDBMiningCatalog, ParquetSnapshotWriter
from medical_kg_nlp.mining.connectors import connector_from_definition
from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget, ReviewPriority
from medical_kg_nlp.mining.curation import (
    AnnotationCurationPolicy,
    AnnotationCurationResult,
    curate_annotations,
    load_annotation_curation_policy,
)
from medical_kg_nlp.mining.crosswalk import (
    MentionCrosswalkPolicy,
    MentionCrosswalkRecord,
    MentionCrosswalkResult,
    crosswalk_mentions,
    load_crosswalk_policies,
)
from medical_kg_nlp.mining.dedup import (
    DuplicateGroup,
    DuplicateGroupKind,
    StableTextDeduplicator,
)
from medical_kg_nlp.mining.labeling import (
    BatchedProposalLabelerAdapter,
    ConsensusProposalLabeler,
    PolicyAwareProposalLabelerAdapter,
)
from medical_kg_nlp.mining.graph_knowledge import (
    GraphCompilationConfig,
    compile_knowledge_graph,
)
from medical_kg_nlp.mining.knowledge import (
    AliasKnowledgeCompilationResult,
    MinedAliasPromotionPolicy,
    compile_mined_aliases,
    load_alias_promotion_policy,
)
from medical_kg_nlp.mining.lexicon import (
    MentionInventoryEntry,
    MentionInventoryResult,
    build_mention_inventory,
    load_mention_inventory,
)
from medical_kg_nlp.mining.linked_aliases import (
    LinkedAliasProposalPolicy,
    LinkedAliasProposalResult,
    build_linked_alias_proposals,
    load_linked_alias_policy,
)
from medical_kg_nlp.mining.model_dataset import (
    SpanDatasetConfig,
    export_span_dataset,
    iter_span_training_records,
    load_dataset_splits,
)
from medical_kg_nlp.mining.parsers import parser_from_definition
from medical_kg_nlp.mining.ports import (
    ArtifactStorePort,
    CoveragePlannerPort,
    DeduplicatorPort,
    DocumentParserPort,
    ProposalLabelerPort,
    QualityGatePort,
    RelationLabelerPort,
    ReviewBackendPort,
    SourceConnectorPort,
)
from medical_kg_nlp.mining.policy import MiningQualityGate, PolicyDecision, SourcePolicyGate
from medical_kg_nlp.mining.profile import build_dataset_profile, profile_blocking_issue_count
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    CoverageCell,
    CoverageReport,
    DatasetSnapshot,
    DiscoveredArtifact,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
    SourceArtifact,
    SourceRequest,
    StoredObject,
)
from medical_kg_nlp.mining.reconciliation import (
    DocumentCanonicalMapping,
    DuplicateReconciliationReport,
    ExactDuplicateReconciliationResult,
    reconcile_exact_duplicates,
)
from medical_kg_nlp.mining.recognition_benchmark import (
    benchmark_recognition_dictionary,
)
from medical_kg_nlp.mining.recognition_knowledge import (
    RecognitionKnowledgeCompilationResult,
    RecognitionKnowledgePolicy,
    compile_recognition_knowledge,
    load_recognition_knowledge_policy,
)
from medical_kg_nlp.mining.storage import FsspecArtifactStore, LocalArtifactStore
from medical_kg_nlp.mining.registry import (
    LicenseMode,
    RetentionPolicy,
    SourceDefinition,
    SourceRegistry,
    VersionPolicy,
    load_source_registry,
)
from medical_kg_nlp.mining.review import JsonlReviewBackend
from medical_kg_nlp.mining.quality import (
    AgreementThresholds,
    GoldAgreementGate,
    ReviewAgreementEvaluator,
    ReviewAgreementReport,
)
from medical_kg_nlp.mining.runner import (
    MiningPlan,
    MiningPlanResult,
    SourceJob,
    load_mining_plan,
    run_mining_plan,
)
from medical_kg_nlp.mining.snapshot import SnapshotBuilder, SnapshotSplitConfig
from medical_kg_nlp.mining.synthetic import (
    MinimalPairGenerator,
    RenderedScenario,
    ScenarioEntity,
    ScenarioGraph,
    ScenarioRelation,
    SentinelScenarioRenderer,
)

__all__ = [
    "AccessClass",
    "AgreementThresholds",
    "AnnotationLayer",
    "AnnotationProposal",
    "AnnotationCurationPolicy",
    "AnnotationCurationResult",
    "ArtifactStorePort",
    "AliasKnowledgeCompilationResult",
    "BatchedProposalLabelerAdapter",
    "ConceptLink",
    "CoverageCell",
    "CoverageCubePlanner",
    "CoveragePlannerPort",
    "CoverageReport",
    "CoverageTarget",
    "DatasetSnapshot",
    "DeduplicatorPort",
    "DiscoveredArtifact",
    "GraphCompilationConfig",
    "DocumentParserPort",
    "DocumentCanonicalMapping",
    "DuckDBMiningCatalog",
    "DuplicateGroup",
    "DuplicateGroupKind",
    "DuplicateReconciliationReport",
    "ExactDuplicateReconciliationResult",
    "FsspecArtifactStore",
    "GoldAgreementGate",
    "LocalArtifactStore",
    "LicenseMode",
    "LinkedAliasProposalPolicy",
    "LinkedAliasProposalResult",
    "MinedDocument",
    "MinimalPairGenerator",
    "MiningQualityGate",
    "MinedAliasPromotionPolicy",
    "MiningPlan",
    "MiningPlanResult",
    "MentionInventoryEntry",
    "MentionInventoryResult",
    "MentionCrosswalkPolicy",
    "MentionCrosswalkRecord",
    "MentionCrosswalkResult",
    "ParquetSnapshotWriter",
    "ProposalLabelerPort",
    "PolicyDecision",
    "PolicyAwareProposalLabelerAdapter",
    "QualityGatePort",
    "RedistributionPolicy",
    "RecognitionKnowledgeCompilationResult",
    "RecognitionKnowledgePolicy",
    "RelationLabelerPort",
    "RetentionPolicy",
    "RelationProposal",
    "RenderedScenario",
    "ReviewPriority",
    "ReviewAgreementEvaluator",
    "ReviewAgreementReport",
    "ReviewBackendPort",
    "ReviewStatus",
    "SourceArtifact",
    "SourceJob",
    "SourceConnectorPort",
    "SourceDefinition",
    "SourcePolicyGate",
    "SourceRegistry",
    "SourceRequest",
    "ScenarioEntity",
    "ScenarioGraph",
    "ScenarioRelation",
    "SentinelScenarioRenderer",
    "StableTextDeduplicator",
    "SnapshotBuilder",
    "SnapshotSplitConfig",
    "StoredObject",
    "SpanDatasetConfig",
    "VersionPolicy",
    "build_dataset_profile",
    "build_linked_alias_proposals",
    "compile_knowledge_graph",
    "benchmark_recognition_dictionary",
    "build_mention_inventory",
    "crosswalk_mentions",
    "compile_mined_aliases",
    "compile_recognition_knowledge",
    "curate_annotations",
    "connector_from_definition",
    "ConsensusProposalLabeler",
    "JsonlReviewBackend",
    "load_source_registry",
    "load_crosswalk_policies",
    "load_annotation_curation_policy",
    "load_mention_inventory",
    "load_mining_plan",
    "load_linked_alias_policy",
    "export_span_dataset",
    "iter_span_training_records",
    "load_dataset_splits",
    "load_recognition_knowledge_policy",
    "load_alias_promotion_policy",
    "parser_from_definition",
    "profile_blocking_issue_count",
    "reconcile_exact_duplicates",
    "run_mining_plan",
]
