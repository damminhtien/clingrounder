"""Local ClinGrounder demo for inspecting grounded spans and evidence."""

from __future__ import annotations

import hashlib
from html import escape
from typing import Any

import streamlit as st

from clingrounder import load_pipeline


_SAMPLE_TEXT = (
    "Bệnh nhân không sốt, tiền sử tăng huyết áp và đang dùng metformin 500 mg "
    "ngày hai lần."
)
_TYPE_COLORS = {
    "DISEASE": "#fee2e2",
    "SYMPTOM": "#fef3c7",
    "DRUG": "#dcfce7",
    "LAB_TEST": "#dbeafe",
    "LAB_RESULT": "#ede9fe",
}


@st.cache_resource
def _pipeline() -> Any:
    """Cache one offline runtime for the Streamlit process."""

    # INVARIANT: the demo uses only the package-bundled, checksum-verified pack. It never
    # downloads a model or sends clinical text to a hosted service.
    return load_pipeline("vi-clinical-small", offline=True)


def _highlighted_text(text: str, entities: list[Any]) -> str:
    """Render non-overlapping entity spans without changing source offsets."""

    fragments: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: (item.span[0], item.span[1], item.id)):
        start, end = entity.span
        if start < cursor:
            continue
        fragments.append(escape(text[cursor:start]))
        color = _TYPE_COLORS.get(entity.type.value, "#e5e7eb")
        label = f"{entity.type.value} · {entity.assertion.value}"
        fragments.append(
            '<mark style="background:{color};padding:0.12rem 0.25rem;border-radius:0.25rem;" '
            'title="{label}">{mention}</mark>'.format(
                color=color,
                label=escape(label),
                mention=escape(text[start:end]),
            )
        )
        cursor = end
    fragments.append(escape(text[cursor:]))
    return "".join(fragments)


def _candidate_rows(entity: Any) -> list[dict[str, object]]:
    """Project candidate evidence into a compact, inspectable table."""

    return [
        {
            "code_system": candidate.code_system.value,
            "code": candidate.code,
            "name": candidate.name,
            "retrieval_score": round(candidate.retrieval_score, 4),
            "emit_probability": round(candidate.emit_probability, 4),
            "primary_source": candidate.source,
            "evidence_sources": ", ".join(candidate.evidence_sources),
            "matched_alias": candidate.matched_alias,
            "qualified": candidate.qualified,
            "qualification_reason": candidate.qualification_reason,
        }
        for candidate in entity.candidates
    ]


def _relation_rows(relations: list[Any]) -> list[dict[str, object]]:
    """Expose relation confidence and provenance without rendering source text."""

    rows: list[dict[str, object]] = []
    for relation in relations:
        evidence = relation.evidence
        rows.append(
            {
                "type": relation.type.value,
                "head": relation.head,
                "tail": relation.tail,
                "confidence": round(relation.confidence, 4),
                "evidence_span": relation.evidence_span,
                "evidence_source": evidence.source if evidence else None,
                "rule_id": evidence.rule_id if evidence else None,
                "support_score": (
                    round(evidence.support_score, 4) if evidence else None
                ),
                "provenance": evidence.provenance if evidence else None,
            }
        )
    return rows


def _stage_rows(trace: Any) -> list[dict[str, object]]:
    """Return stable stage timings and bounded counters from a pipeline trace."""

    return [
        {
            "stage": stage.name,
            "status": stage.status,
            "elapsed_ms": round(stage.elapsed_ms, 3),
            "entity_count": stage.entity_count,
            "counters": dict(stage.counters),
        }
        for stage in trace.stages
    ]


def _document_id(text: str) -> str:
    """Create the same deterministic, PHI-free identifier shape as the public facade."""

    return f"demo-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def main() -> None:
    """Run the interactive local demo."""

    st.set_page_config(page_title="ClinGrounder", page_icon="🩺", layout="wide")
    st.title("ClinGrounder")
    st.caption("Offline Vietnamese clinical text grounding")
    text = st.text_area("Clinical text", value=_SAMPLE_TEXT, height=150)

    if not st.button("Analyze", type="primary"):
        return
    if not text.strip():
        st.warning("Enter clinical text first.")
        return

    with st.spinner("Running local pipeline..."):
        result = _pipeline().predict_with_trace(text, document_id=_document_id(text))
        prediction = result.prediction

    st.subheader("Grounded text")
    st.markdown(_highlighted_text(text, prediction.entities), unsafe_allow_html=True)

    st.subheader("Entities")
    rows = []
    for entity in prediction.entities:
        rows.append(
            {
                "text": entity.text,
                "type": entity.type.value,
                "assertion": entity.assertion.value,
                "span": list(entity.span),
                "assigned_code": entity.code,
                "candidate_count": len(entity.candidates),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for entity in prediction.entities:
        if entity.candidates:
            with st.expander(f"Candidates: {entity.text}"):
                st.dataframe(_candidate_rows(entity), use_container_width=True, hide_index=True)

    if prediction.relations:
        st.subheader("Relations")
        st.dataframe(
            _relation_rows(prediction.relations),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Stage latency")
    bottleneck = result.trace.bottleneck()
    latency_columns = st.columns(3)
    latency_columns[0].metric("Total", f"{result.trace.total_ms:.2f} ms")
    latency_columns[1].metric("Stages", len(result.trace.stages))
    latency_columns[2].metric("Bottleneck", bottleneck.name if bottleneck else "n/a")
    st.dataframe(_stage_rows(result.trace), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
