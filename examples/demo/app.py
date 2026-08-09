"""Local ClinGrounder demo for inspecting grounded spans and evidence."""

from __future__ import annotations

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
    return [
        {
            "code_system": candidate.code_system.value,
            "code": candidate.code,
            "name": candidate.name,
            "source": candidate.source,
            "score": round(candidate.retrieval_score, 4),
            "qualified": candidate.qualified,
        }
        for candidate in entity.candidates
    ]


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
        prediction = _pipeline()(text)

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
            [
                {
                    "type": relation.type.value,
                    "head": relation.head,
                    "tail": relation.tail,
                    "evidence_span": relation.evidence_span,
                }
                for relation in prediction.relations
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
