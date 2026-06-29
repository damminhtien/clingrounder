from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CANDIDATE_SOURCES = ("exact", "abbreviation", "fuzzy", "char_ngram", "bm25")


@dataclass(frozen=True)
class PipelineOptions:
    max_candidates: int = 20
    context_window: int = 80
    candidate_sources: tuple[str, ...] = DEFAULT_CANDIDATE_SOURCES
    enable_context: bool = True
    enable_linking: bool = True
    enable_candidate_reranking: bool = True
    enable_entity_kg_validation: bool = True
    enable_relations: bool = True
    enable_relation_kg_validation: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "PipelineOptions":
        sources = payload.get("candidate_sources", DEFAULT_CANDIDATE_SOURCES)
        if not isinstance(sources, list | tuple):
            raise ValueError("candidate_sources must be a list of retrieval source names")
        return cls(
            max_candidates=_int_value(payload, "max_candidates", cls.max_candidates),
            context_window=_int_value(payload, "context_window", cls.context_window),
            candidate_sources=tuple(str(source) for source in sources),
            enable_context=_bool_value(payload, "enable_context", cls.enable_context),
            enable_linking=_bool_value(payload, "enable_linking", cls.enable_linking),
            enable_candidate_reranking=_bool_value(
                payload,
                "enable_candidate_reranking",
                cls.enable_candidate_reranking,
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
