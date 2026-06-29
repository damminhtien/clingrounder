from __future__ import annotations
from medical_kg_nlp.pipeline.runner import PipelineRunner
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction


def run_batch(runner: PipelineRunner, documents: list[ClinicalDocument]) -> list[ClinicalPrediction]:
    return [runner.process_document(document) for document in documents]

