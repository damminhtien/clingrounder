"""Pure format readers shared by mining parsers and source-label adapters."""

from clingrounder.mining.formats.brat import (
    BratDocumentBundle,
    BratTextBoundAnnotation,
    parse_brat_text_bound_annotations,
    read_brat_archive,
)
from clingrounder.mining.formats.codiesp import (
    CodiEspArchiveBundle,
    CodiEspDocumentBundle,
    CodiEspSpanAnnotation,
    read_codiesp_archive,
)
from clingrounder.mining.formats.clinicaltrials import (
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
from clingrounder.mining.formats.jats import (
    RenderedJatsArticle,
    RenderedJatsBlock,
    render_jats_article,
)

__all__ = ["RenderedJatsArticle", "RenderedJatsBlock", "render_jats_article"]
