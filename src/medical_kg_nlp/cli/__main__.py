"""Operational module entrypoint for ``python -m medical_kg_nlp.cli``."""

from medical_kg_nlp.cli.main import operational_main


if __name__ == "__main__":
    raise SystemExit(operational_main())
