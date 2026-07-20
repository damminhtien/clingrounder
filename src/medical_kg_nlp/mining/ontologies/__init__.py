"""Streaming compilers for ontology releases and association tables."""

from medical_kg_nlp.mining.ontologies.obo_graph import (
    OBOGraphCompilationConfig,
    compile_obo_graph_release,
)

__all__ = ["OBOGraphCompilationConfig", "compile_obo_graph_release"]
