"""Developer commands should resolve to the same supported CLI hierarchy as CI."""

from pathlib import Path


def test_make_phase1_submit_uses_current_benchmark_subcommand() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    command = next(
        line.strip()
        for line in makefile.splitlines()
        if "clingrounder.cli benchmark phase1" in line
    )

    assert "benchmark phase1 submission --input-dir" in command
