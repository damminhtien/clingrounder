"""Pure format readers shared by mining parsers and source-label adapters."""

from medical_kg_nlp.mining.formats.brat import (
    BratDocumentBundle,
    BratTextBoundAnnotation,
    parse_brat_text_bound_annotations,
    read_brat_archive,
)
from medical_kg_nlp.mining.formats.codiesp import (
    CodiEspArchiveBundle,
    CodiEspDocumentBundle,
    CodiEspSpanAnnotation,
    read_codiesp_archive,
)
from medical_kg_nlp.mining.formats.clinicaltrials import (
    ClinicalTrialRenderedField,
    ClinicalTrialRenderedStudy,
    render_clinical_trial,
)

__all__ = [
    "BratDocumentBundle",
    "BratTextBoundAnnotation",
    "CodiEspArchiveBundle",
    "CodiEspDocumentBundle",
    "CodiEspSpanAnnotation",
    "ClinicalTrialRenderedField",
    "ClinicalTrialRenderedStudy",
    "parse_brat_text_bound_annotations",
    "read_brat_archive",
    "read_codiesp_archive",
    "render_clinical_trial",
]
from medical_kg_nlp.mining.formats.jats import (
    RenderedJatsArticle,
    RenderedJatsBlock,
    render_jats_article,
)

__all__ = ["RenderedJatsArticle", "RenderedJatsBlock", "render_jats_article"]
