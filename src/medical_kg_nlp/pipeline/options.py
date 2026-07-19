from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.schema.types import EntityType


DEFAULT_CANDIDATE_SOURCES = ("exact", "abbreviation", "fuzzy", "char_ngram", "bm25")


@dataclass(frozen=True)
class PipelineOptions:
    max_candidates: int = 20
    context_window: int = 80
    link_assignment_threshold: float = 0.75
    link_assignment_margin: float = 0.05
    link_candidate_threshold: float = 0.75
    link_candidate_relative_margin: float = 0.05
    link_max_qualified_candidates: int = 5
    link_candidate_thresholds_by_type: tuple[tuple[str, float], ...] = ()
    link_candidate_thresholds_by_source: tuple[tuple[str, float], ...] = ()
    link_emit_probabilities_by_source: tuple[tuple[str, float], ...] = ()
    link_enforce_rxnorm_structure: bool = True
    candidate_sources: tuple[str, ...] = DEFAULT_CANDIDATE_SOURCES
    enable_context: bool = True
    enable_linking: bool = True
    enable_candidate_reranking: bool = True
    enable_graph_evidence_reranking: bool = False
    graph_evidence_max_bonus: float = 0.04
    graph_evidence_min_support: int = 2
    graph_evidence_relation_types: tuple[str, ...] = ("CO_OCCURS_WITH",)
    enable_entity_kg_validation: bool = True
    enable_relations: bool = True
    enable_relation_kg_validation: bool = True

    def __post_init__(self) -> None:
        if self.enable_graph_evidence_reranking and not self.enable_linking:
            raise ValueError("Graph evidence reranking requires linking")
        if not 0.0 <= self.graph_evidence_max_bonus <= 1.0:
            raise ValueError("graph_evidence_max_bonus must be between 0 and 1")
        if self.graph_evidence_min_support < 1:
            raise ValueError("graph_evidence_min_support must be at least 1")
        if not self.graph_evidence_relation_types or any(
            not relation_type.strip()
            for relation_type in self.graph_evidence_relation_types
        ):
            raise ValueError("graph_evidence_relation_types must be non-empty")

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "PipelineOptions":
        sources = payload.get("candidate_sources", DEFAULT_CANDIDATE_SOURCES)
        if not isinstance(sources, list | tuple):
            raise ValueError("candidate_sources must be a list of retrieval source names")
        return cls(
            max_candidates=_int_value(payload, "max_candidates", cls.max_candidates),
            context_window=_int_value(payload, "context_window", cls.context_window),
            link_assignment_threshold=_probability_value(
                payload,
                "link_assignment_threshold",
                cls.link_assignment_threshold,
            ),
            link_assignment_margin=_probability_value(
                payload,
                "link_assignment_margin",
                cls.link_assignment_margin,
            ),
            link_candidate_threshold=_probability_value(
                payload,
                "link_candidate_threshold",
                cls.link_candidate_threshold,
            ),
            link_candidate_relative_margin=_probability_value(
                payload,
                "link_candidate_relative_margin",
                cls.link_candidate_relative_margin,
            ),
            link_max_qualified_candidates=_positive_int_value(
                payload,
                "link_max_qualified_candidates",
                cls.link_max_qualified_candidates,
                maximum=5,
            ),
            link_candidate_thresholds_by_type=_threshold_items(
                payload,
                "link_candidate_thresholds_by_type",
                allowed_keys={entity_type.value for entity_type in EntityType},
            ),
            link_candidate_thresholds_by_source=_threshold_items(
                payload,
                "link_candidate_thresholds_by_source",
            ),
            link_emit_probabilities_by_source=_threshold_items(
                payload,
                "link_emit_probabilities_by_source",
            ),
            link_enforce_rxnorm_structure=_bool_value(
                payload,
                "link_enforce_rxnorm_structure",
                cls.link_enforce_rxnorm_structure,
            ),
            candidate_sources=tuple(str(source) for source in sources),
            enable_context=_bool_value(payload, "enable_context", cls.enable_context),
            enable_linking=_bool_value(payload, "enable_linking", cls.enable_linking),
            enable_candidate_reranking=_bool_value(
                payload,
                "enable_candidate_reranking",
                cls.enable_candidate_reranking,
            ),
            enable_graph_evidence_reranking=_bool_value(
                payload,
                "enable_graph_evidence_reranking",
                cls.enable_graph_evidence_reranking,
            ),
            graph_evidence_max_bonus=_probability_value(
                payload,
                "graph_evidence_max_bonus",
                cls.graph_evidence_max_bonus,
            ),
            graph_evidence_min_support=_positive_int_value(
                payload,
                "graph_evidence_min_support",
                cls.graph_evidence_min_support,
            ),
            graph_evidence_relation_types=_string_tuple_value(
                payload,
                "graph_evidence_relation_types",
                cls.graph_evidence_relation_types,
            ),
            enable_entity_kg_validation=_bool_value(
                payload,
                "enable_entity_kg_validation",
                cls.enable_entity_kg_validation,
            ),
            enable_relations=_bool_value(payload, "enable_relations", cls.enable_relations),
            enable_relation_kg_validation=_bool_value(
                payload,
                "enable_relation_kg_validation",
                cls.enable_relation_kg_validation,
            ),
        )


def _bool_value(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int_value(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _positive_int_value(
    payload: dict[str, object],
    key: str,
    default: int,
    *,
    maximum: int | None = None,
) -> int:
    value = _int_value(payload, key, default)
    if value < 1:
        raise ValueError(f"{key} must be at least 1")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be at most {maximum}")
    return value


def _probability_value(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return result


def _threshold_items(
    payload: dict[str, object],
    key: str,
    *,
    allowed_keys: set[str] | None = None,
) -> tuple[tuple[str, float], ...]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    items: list[tuple[str, float]] = []
    for raw_name, raw_threshold in value.items():
        name = str(raw_name)
        if allowed_keys is not None and name not in allowed_keys:
            raise ValueError(f"Unknown key {name!r} in {key}")
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int | float):
            raise ValueError(f"{key}.{name} must be a number")
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{key}.{name} must be between 0 and 1")
        items.append((name, threshold))
    return tuple(sorted(items))


def _string_tuple_value(
    payload: dict[str, object],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = payload.get(key, default)
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)
