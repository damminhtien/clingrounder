from __future__ import annotations
import csv
from pathlib import Path

from medical_kg_nlp.schema.output import ClinicalPrediction


def write_error_analysis(
    gold: list[ClinicalPrediction],
    pred: list[ClinicalPrediction],
    output_path: str | Path,
) -> None:
    pred_by_doc = {item.document_id: item for item in pred}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["document_id", "error_type", "text_window", "gold", "prediction", "candidate_list", "notes"],
        )
        writer.writeheader()
        for gold_doc in gold:
            pred_doc = pred_by_doc.get(gold_doc.document_id)
            if pred_doc is None:
                writer.writerow(
                    {
                        "document_id": gold_doc.document_id,
                        "error_type": "missing_document",
                        "text_window": "",
                        "gold": gold_doc.to_json(),
                        "prediction": "",
                        "candidate_list": "",
                        "notes": "No prediction for document",
                    }
                )
                continue
            pred_by_span = {entity.span: entity for entity in pred_doc.entities}
            for gold_entity in gold_doc.entities:
                prediction = pred_by_span.get(gold_entity.span)
                if prediction is None:
                    error_type = "missing_entity"
                elif prediction.type != gold_entity.type:
                    error_type = "type_error"
                elif prediction.code != gold_entity.code:
                    error_type = "linking_error"
                elif prediction.assertion != gold_entity.assertion:
                    error_type = "context_error"
                else:
                    continue
                writer.writerow(
                    {
                        "document_id": gold_doc.document_id,
                        "error_type": error_type,
                        "text_window": gold_entity.text,
                        "gold": gold_entity.to_json(),
                        "prediction": prediction.to_json() if prediction else "",
                        "candidate_list": prediction.candidates if prediction else "",
                        "notes": "",
                    }
                )

