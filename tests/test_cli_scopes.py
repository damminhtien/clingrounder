from __future__ import annotations

import pytest

from clingrounder.cli.parser import build_parser


def test_operational_parser_excludes_research_and_benchmark_commands() -> None:
    parser = build_parser("operational")

    assert parser.parse_args(["pipeline", "list-profiles"]).command == "pipeline"
    assert parser.parse_args(["evaluate", "--gold", "g", "--pred", "p"]).command == "evaluate"
    with pytest.raises(SystemExit):
        parser.parse_args(["model", "inspect-inference-budget", "--config", "run.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "list"])


def test_research_and_benchmark_scopes_are_separate() -> None:
    research = build_parser("research")
    benchmark = build_parser("benchmark")

    with pytest.raises(SystemExit):
        research.parse_args(["evaluate", "--gold", "g", "--pred", "p"])
    assert benchmark.parse_args(["benchmark", "list"]).command == "benchmark"
    with pytest.raises(SystemExit):
        benchmark.parse_args(["pipeline", "list-profiles"])
