from __future__ import annotations
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.types import AssertionStatus


def is_confirmed_patient_condition(entity: EntityAnnotation) -> bool:
    return entity.assertion == AssertionStatus.PRESENT
