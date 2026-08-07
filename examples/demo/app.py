"""Small optional Streamlit viewer for the bundled ClinGrounder pipeline.

The demo is intentionally outside the core package. It shows the inspectable output contract
without adding a web framework to the runtime or silently enabling remote model execution.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from clingrounder import load_pipeline


@st.cache_resource
def get_pipeline():
    """Keep one local, deterministic runtime for the Streamlit process."""

    return load_pipeline("vi-clinical-small", offline=True)


def _entity_rows(prediction: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in prediction.entities:
        rows.append(
            {
                "text": entity.text,
                "span": f"[{entity.span[0]}, {entity.span[1]})",
                "type": entity.type.value,
                "assertion": entity.assertion.value,
                "code_system": entity.code_system.value,
                "code": entity.code or "",
                "candidates": ", ".join(
                    f"{candidate.code_system.value}:{candidate.code}"
                    for candidate in entity.candidates
                    if candidate.code
                ),
            }
        )
    return rows


st.set_page_config(page_title="ClinGrounder demo", layout="wide")
st.title("ClinGrounder")
st.caption("Local, deterministic clinical text grounding for Vietnamese and mixed Vietnamese-English text")
text = st.text_area(
    "Clinical text",
    value="Bệnh nhân không sốt. Tiền sử tăng huyết áp. Đang dùng metformin.",
    height=140,
)

if st.button("Run pipeline", type="primary"):
    if not text.strip():
        st.warning("Enter clinical text first.")
    else:
        with st.spinner("Running local pipeline"):
            prediction = get_pipeline()(text)
        st.subheader("Entities")
        if prediction.entities:
            st.dataframe(_entity_rows(prediction), use_container_width=True, hide_index=True)
        else:
            st.info("No supported entities were found.")
        st.subheader("Relations")
        st.json([relation.to_json() for relation in prediction.relations])

