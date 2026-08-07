"""Public repository policy and credential-audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clingrounder.cli.parser import build_parser
from clingrounder.governance.public_release import (
    audit_public_repository,
    build_local_artifact_inventory,
    load_public_repository_policy,
)


def test_public_release_accepts_explicit_small_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "data" / "samples" / "note.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"text":"synthetic"}\n', encoding="utf-8")
    policy = _write_policy(
        tmp_path,
        rules=[
            {
                "id": "fixtures",
                "patterns": ["data/samples/**"],
                "disposition": "redistributable",
                "rationale": "Synthetic public fixtures.",
            }
        ],
    )

    report = audit_public_repository(
        tmp_path,
        policy,
        tracked_paths=["data/samples/note.jsonl"],
    )

    assert report.valid is True
    assert report.disposition_counts == {"redistributable": 1}
    assert report.issues == ()


def test_public_release_rejects_restricted_and_unclassified_tracked_data(
    tmp_path: Path,
) -> None:
    private = tmp_path / "data" / "private" / "gold.json"
    unknown = tmp_path / "data" / "unknown.bin"
    private.parent.mkdir(parents=True)
    private.write_text("[]\n", encoding="utf-8")
    unknown.write_bytes(b"unknown")
    policy = _write_policy(
        tmp_path,
        rules=[
            {
                "id": "private-gold",
                "patterns": ["data/private/**"],
                "disposition": "local_only",
                "rationale": "Private labels stay outside Git.",
            }
        ],
    )

    report = audit_public_repository(
        tmp_path,
        policy,
        tracked_paths=["data/private/gold.json", "data/unknown.bin"],
    )

    assert report.valid is False
    assert {(issue.code, issue.path) for issue in report.issues} == {
        ("restricted_path_tracked", "data/private/gold.json"),
        ("unclassified_protected_path", "data/unknown.bin"),
    }


def test_public_release_reports_secret_location_without_secret_value(tmp_path: Path) -> None:
    source = tmp_path / "src" / "settings.py"
    source.parent.mkdir(parents=True)
    secret = "hf_" + "A" * 30
    source.write_text(f"TOKEN = '{secret}'\n", encoding="utf-8")
    policy = _write_policy(tmp_path, rules=[_fallback_rule()])

    report = audit_public_repository(
        tmp_path,
        policy,
        tracked_paths=["src/settings.py"],
    )

    assert report.valid is False
    assert report.issues[0].code == "potential_secret"
    assert report.issues[0].line == 1
    assert secret not in report.model_dump_json()


def test_public_release_policy_requires_notice_only_for_attributed_rules(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.yaml"
    payload = _policy_payload(
        rules=[
            {
                "id": "attributed",
                "patterns": ["data/open/**"],
                "disposition": "redistributable_with_notice",
                "rationale": "Upstream attribution is required.",
            }
        ]
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="require notice_path"):
        load_public_repository_policy(path)


def test_local_inventory_hashes_only_explicit_restricted_roots(tmp_path: Path) -> None:
    private = tmp_path / "data" / "private" / "gold.json"
    transient = tmp_path / "outputs" / "run.json"
    private.parent.mkdir(parents=True)
    transient.parent.mkdir(parents=True)
    private.write_text("[]\n", encoding="utf-8")
    transient.write_text("{}\n", encoding="utf-8")
    policy = _write_policy(
        tmp_path,
        rules=[
            {
                "id": "private-gold",
                "patterns": ["data/private/**"],
                "disposition": "local_only",
                "rationale": "Private labels stay outside Git.",
                "inventory_local_bytes": True,
                "source_ids": [],
            },
            {
                "id": "transient-output",
                "patterns": ["outputs/**"],
                "disposition": "local_only",
                "rationale": "Transient runs are not stable release inputs.",
            },
        ],
    )

    first = build_local_artifact_inventory(tmp_path, policy)
    second = build_local_artifact_inventory(tmp_path, policy)

    assert first == second
    assert first.artifact_count == 1
    assert first.artifacts[0].path == "data/private/gold.json"
    assert first.artifacts[0].byte_size == 3
    assert len(first.artifacts[0].sha256) == 64


def test_release_audit_parser_is_task_neutral() -> None:
    args = build_parser().parse_args(
        ["release", "audit", "--policy", "policy.yaml", "--root", "."]
    )

    assert args.handler == "release_audit"
    assert args.output is None

    inventory = build_parser().parse_args(
        ["release", "inventory", "--output", "local-artifacts.json"]
    )
    assert inventory.handler == "release_inventory"


@pytest.mark.release
def test_checked_in_public_repository_policy_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit_public_repository(
        root,
        root / "configs/repository/public-release.yaml",
    )

    assert report.valid is True, report.model_dump(mode="json")


def _write_policy(
    root: Path,
    *,
    rules: list[dict[str, object]],
) -> Path:
    (root / "source-registry.yaml").write_text(
        "schema_version: medical-source-registry.v2\nresources: []\n",
        encoding="utf-8",
    )
    path = root / "policy.yaml"
    path.write_text(
        yaml.safe_dump(_policy_payload(rules=rules), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _policy_payload(*, rules: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "clingrounder.public-repository-policy.v1",
        "protected_roots": ["data", "outputs", "models", "checkpoints"],
        "max_tracked_file_bytes": 1024,
        "source_registry_path": "source-registry.yaml",
        "rules": rules,
        "required_tracked_paths": [],
        "secret_scan_excludes": [],
    }


def _fallback_rule() -> dict[str, object]:
    return {
        "id": "public-source",
        "patterns": ["src/**"],
        "disposition": "redistributable",
        "rationale": "Project-authored source code.",
    }
