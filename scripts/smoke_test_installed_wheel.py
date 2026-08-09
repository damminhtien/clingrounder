"""Run the smallest supported smoke test from an installed ClinGrounder wheel.

This script is deliberately independent of the source checkout. CI installs the built wheel
into a fresh virtual environment before invoking it, which catches missing package data and
imports that only work because ``src/`` is present.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

import clingrounder
from clingrounder import Pipeline, load_pipeline


def main() -> None:
    """Load the bundled offline pack and verify the public prediction contract."""

    assert clingrounder.__version__, "installed package did not expose a version"
    source = "Bệnh nhân không sốt. Tiền sử tăng huyết áp. Đang dùng metformin."
    with load_pipeline("vi-clinical-small", offline=True) as pipeline:
        prediction = pipeline(source)

    assert prediction.document_id.startswith("text-"), prediction.document_id
    assert prediction.entities, "bundled pack returned no entities"
    for entity in prediction.entities:
        start, end = entity.span
        # INVARIANT: an installed wheel must preserve raw offsets exactly like a source checkout.
        assert source[start:end] == entity.text, (entity.text, entity.span)

    by_text = {entity.text: entity for entity in prediction.entities}
    assert by_text["sốt"].assertion.value == "NEGATED"
    assert by_text["tăng huyết áp"].assertion.value == "HISTORICAL"
    assert by_text["metformin"].code == "6809"

    # INVARIANT: the documented download cache must be loadable without source-checkout paths.
    with TemporaryDirectory(prefix="clingrounder-wheel-artifact-") as cache_dir:
        cached_path = Pipeline.download("vi-clinical-small", cache_dir=cache_dir)
        with load_pipeline(cached_path, offline=True) as cached_pipeline:
            cached_prediction = cached_pipeline(source)
        assert [item.span for item in cached_prediction.entities] == [
            item.span for item in prediction.entities
        ]
    print(f"installed ClinGrounder {clingrounder.__version__}: smoke test passed")


if __name__ == "__main__":
    main()
