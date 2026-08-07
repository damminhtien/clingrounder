"""Operational module entrypoint for ``python -m clingrounder.cli``."""

from clingrounder.cli.main import operational_main


if __name__ == "__main__":
    raise SystemExit(operational_main())
