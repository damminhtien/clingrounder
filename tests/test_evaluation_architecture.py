"""Dependency and neutral-record contracts for reusable evaluation code."""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from clingrounder import __version__
from clingrounder.benchmarks.phase1.adapter import (
    Phase1EvaluationAdapter,
    Phase1Record,
)
from clingrounder.evaluation import (
    EvaluationDocument,
    EvaluationEntity,
    adapt_evaluation_records,
)


def test_generic_evaluation_does_not_import_benchmarks_or_experiments() -> None:
    evaluation_root = Path("src/clingrounder/evaluation")
    forbidden: list[tuple[str, str]] = []
    for path in evaluation_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module.startswith(
                    ("clingrounder.benchmarks", "clingrounder.experiments")
                ):
                    forbidden.append((str(path), module))

    assert forbidden == []


def test_public_packages_declare_docs_and_exports() -> None:
    package_root = Path("src/clingrounder")
    violations: list[str] = []
    for path in sorted(package_root.rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_exports = any(
            (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            )
            for node in tree.body
        )
        if ast.get_docstring(tree) is None:
            violations.append(f"{path}: missing module docstring")
        if not has_exports:
            violations.append(f"{path}: missing explicit __all__")

    assert violations == []


def test_package_versions_remain_synchronized() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == __version__


def test_evaluation_adapter_validates_offsets_and_duplicate_documents() -> None:
    records = [_RawRecord("1", "đau ngực")]
    adapted = adapt_evaluation_records(records, _Adapter())

    assert adapted[0].entities[0].text == "đau ngực"
    with pytest.raises(ValueError, match="Duplicate evaluation document ID"):
        adapt_evaluation_records(records * 2, _Adapter())


def test_phase1_plugin_adapts_to_neutral_records() -> None:
    source = "đau ngực"
    documents = adapt_evaluation_records(
        [
            Phase1Record(
                document_id="1",
                source_text=source,
                entities=[
                    {
                        "text": source,
                        "position": [0, len(source)],
                        "type": "TRIỆU_CHỨNG",
                        "assertions": [],
                        "candidates": [],
                    }
                ],
            )
        ],
        Phase1EvaluationAdapter(),
    )

    assert documents[0].entities[0].entity_type == "TRIỆU_CHỨNG"


@dataclass(frozen=True)
class _RawRecord:
    document_id: str
    text: str


class _Adapter:
    def adapt(self, record: _RawRecord) -> EvaluationDocument:
        return EvaluationDocument(
            document_id=record.document_id,
            source_text=record.text,
            entities=(
                EvaluationEntity(
                    entity_id="E1",
                    span=(0, len(record.text)),
                    text=record.text,
                    entity_type="SYMPTOM",
                ),
            ),
        )
