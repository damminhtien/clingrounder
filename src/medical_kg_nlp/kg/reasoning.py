from __future__ import annotations
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import AssertionStatus


def is_confirmed_patient_condition(entity: EntityAnnotation) -> bool:
    return entity.assertion not in {AssertionStatus.NEGATED, AssertionStatus.FAMILY, AssertionStatus.POSSIBLE}

