"""Advanced pipeline APIs.

The implementation and export list live in :mod:`medical_kg_nlp.pipeline.advanced`.  Keeping
this package initializer as a forwarding module prevents two competing public compositions from
drifting apart while retaining the conventional ``medical_kg_nlp.pipeline`` import path for
advanced integrations.
"""

from __future__ import annotations

from medical_kg_nlp.pipeline.advanced import *  # noqa: F403
from medical_kg_nlp.pipeline.advanced import __all__ as __all__
