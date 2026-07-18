"""Reusable acquisition, curation, and dataset-snapshot primitives."""

from medical_kg_nlp.mining.catalog import DuckDBMiningCatalog, ParquetSnapshotWriter
from medical_kg_nlp.mining.connectors import connector_from_definition
from medical_kg_nlp.mining.coverage import CoverageCubePlanner, CoverageTarget, ReviewPriority
from medical_kg_nlp.mining.dedup import StableTextDeduplicator
from medical_kg_nlp.mining.labeling import (
    BatchedProposalLabelerAdapter,
    ConsensusProposalLabeler,
    PolicyAwareProposalLabelerAdapter,
)
from medical_kg_nlp.mining.parsers import parser_from_definition
from medical_kg_nlp.mining.ports import (
    ArtifactStorePort,
    CoveragePlannerPort,
    DeduplicatorPort,
    DocumentParserPort,
    ProposalLabelerPort,
    QualityGatePort,
    ReviewBackendPort,
    SourceConnectorPort,
)
from medical_kg_nlp.mining.policy import MiningQualityGate, PolicyDecision, SourcePolicyGate
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
    "AnnotationLayer",
    "AnnotationProposal",
    "ArtifactStorePort",
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
    "DocumentParserPort",
    "DuckDBMiningCatalog",
    "FsspecArtifactStore",
    "LocalArtifactStore",
    "LicenseMode",
    "MinedDocument",
    "MinimalPairGenerator",
    "MiningQualityGate",
    "MiningPlan",
    "MiningPlanResult",
    "ParquetSnapshotWriter",
    "ProposalLabelerPort",
    "PolicyDecision",
    "PolicyAwareProposalLabelerAdapter",
    "QualityGatePort",
    "RedistributionPolicy",
    "RetentionPolicy",
    "RelationProposal",
    "RenderedScenario",
    "ReviewPriority",
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
    "VersionPolicy",
    "connector_from_definition",
    "ConsensusProposalLabeler",
    "JsonlReviewBackend",
    "load_source_registry",
    "load_mining_plan",
    "parser_from_definition",
    "run_mining_plan",
]
