from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.annotation_knowledge import (
    compile_annotation_knowledge,
    write_annotation_knowledge,
)
from medical_kg_nlp.evaluation.manual_gold import (
    load_phase1_directory,
    manual_gold_split,
)
from medical_kg_nlp.evaluation.phase1 import (
    score_phase1_documents,
    validate_phase1_submission_dir,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.evaluation.phase1_ensemble import load_phase1_output_source
from medical_kg_nlp.evaluation.phase1_entity_gates import (
    Phase1EntityGateConfig,
    apply_phase1_entity_gates,
    compile_boundary_rule_candidates,
)
from medical_kg_nlp.evaluation.phase1_proposals import (
    build_phase1_proposal_matrix,
    proposal_consensus_keys,
    write_phase1_proposal_matrix,
)
from medical_kg_nlp.evaluation.phase1_rule_registry import (
    Phase1RuleRegistry,
    load_phase1_rule_registry,
    write_phase1_rule_registry,
)
from medical_kg_nlp.evaluation.phase1_selective_overlays import (
    AssertionRegime,
    CandidateRegime,
    apply_selective_assertions,
    apply_selective_candidates,
    compile_reviewed_candidate_registry,
    validate_probe_isolation,
)


@dataclass(frozen=True)
class Phase1Top10ProbeConfig:
    base: Path
    proposal_sources: Mapping[str, Path]
    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    review_manifest: Path = Path("data/manual_gold/review_manifest.jsonl")
    dictionary: Path = Path("data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl")
    output_root: Path = Path("outputs/phase1/top10_probes")
    journal_dir: Path = Path("outputs/loops/journal")
    rule_registry: Path | None = None
    minimum_boundary_document_support: int = 2
    open_holdout: bool = False


def build_phase1_top10_probe_suite(config: Phase1Top10ProbeConfig) -> dict[str, Any]:
    if len(config.proposal_sources) < 2:
        raise ValueError("At least two independent proposal sources are required.")
    input_hashes = _input_hashes(config)
    run_hash = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = config.output_root / f"top10_{run_hash}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing probe run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    source_text_by_doc = {
        path.stem: path.read_text(encoding="utf-8") for path in config.input_dir.glob("*.txt")
    }
    base = load_phase1_output_source(config.base)
    gold = load_phase1_directory(config.gold_dir)
    proposal_sources = {
        name: load_phase1_output_source(path) for name, path in config.proposal_sources.items()
    }
    dictionary = DictionaryStore.from_jsonl(config.dictionary)
    expected_count = len(source_text_by_doc)

    train_document_ids = {
        document_id for document_id in gold if manual_gold_split(document_id) == "train"
    }
    annotation_report = compile_annotation_knowledge(
        gold_dir=config.gold_dir,
        manifest_path=config.review_manifest,
        document_ids=train_document_ids,
    )
    write_annotation_knowledge(annotation_report, run_dir / "knowledge_train")
    annotation_policy = annotation_report["policy"]

    proposal_report = build_phase1_proposal_matrix(proposal_sources, source_text_by_doc)
    write_phase1_proposal_matrix(proposal_report, run_dir / "proposals")
    tri_source_ready = len(config.proposal_sources) >= 3
    consensus = (
        proposal_consensus_keys(proposal_report, minimum_sources=2)
        if tri_source_ready
        else set()
    )

    boundary_registry, boundary_audit = compile_boundary_rule_candidates(
        gold,
        base,
        split="train",
        minimum_document_support=config.minimum_boundary_document_support,
        review_status="draft",
    )
    write_phase1_rule_registry(boundary_registry, run_dir / "boundary_rule_candidates.yaml")
    _write_json(run_dir / "boundary_rule_audit.json", boundary_audit)

    candidate_registry, candidate_audit = compile_reviewed_candidate_registry(
        gold,
        config.dictionary,
        split="train",
    )
    write_phase1_rule_registry(candidate_registry, run_dir / "reviewed_candidate_registry.yaml")
    _write_json(run_dir / "reviewed_candidate_audit.json", candidate_audit)

    runtime_registry = (
        load_phase1_rule_registry(config.rule_registry)
        if config.rule_registry is not None
        else Phase1RuleRegistry(())
    )
    if config.rule_registry is not None:
        write_phase1_rule_registry(runtime_registry, run_dir / "runtime_rule_registry.yaml")

    variants: list[dict[str, Any]] = []
    _materialize_variant(
        name="E0_BASE",
        module="entity",
        rows=base,
        base=base,
        decisions=[],
        counters={"output_entity_total": sum(len(rows) for rows in base.values())},
        policy_diff={},
        config=config,
        run_dir=run_dir,
        gold=gold,
        dictionary=dictionary,
        expected_count=expected_count,
        variants=variants,
    )

    entity_variants = {
        "E_LAB": Phase1EntityGateConfig(lab_gate=True, resolve_overlaps=False),
        "E_EXCLUSION": Phase1EntityGateConfig(strict_exclusions=True, resolve_overlaps=False),
        "E_OVERLAP": Phase1EntityGateConfig(resolve_overlaps=True),
        "E_LAB_EXCLUSION": Phase1EntityGateConfig(
            lab_gate=True,
            strict_exclusions=True,
            resolve_overlaps=False,
        ),
    }
    for stage in (
        "boundary_diagnosis",
        "boundary_symptom_prefix",
        "boundary_symptom_course",
        "boundary_imaging_test",
    ):
        entity_variants[f"E_{stage.upper()}"] = Phase1EntityGateConfig(
            boundary_stages=(stage,),
            resolve_overlaps=False,
        )
    entity_variants["E_ALL"] = Phase1EntityGateConfig(
        lab_gate=True,
        strict_exclusions=True,
        boundary_stages=(
            "boundary_diagnosis",
            "boundary_symptom_prefix",
            "boundary_symptom_course",
            "boundary_imaging_test",
        ),
        resolve_overlaps=True,
    )
    for name, gate_config in entity_variants.items():
        rows, decisions, counters = apply_phase1_entity_gates(
            base,
            source_text_by_doc,
            config=gate_config,
            annotation_policy=annotation_policy,
            registry=runtime_registry,
        )
        _materialize_variant(
            name=name,
            module="entity",
            rows=rows,
            base=base,
            decisions=decisions,
            counters=counters,
            policy_diff={
                "lab_gate": gate_config.lab_gate,
                "strict_exclusions": gate_config.strict_exclusions,
                "boundary_stages": list(gate_config.boundary_stages),
                "resolve_overlaps": gate_config.resolve_overlaps,
            },
            config=config,
            run_dir=run_dir,
            gold=gold,
            dictionary=dictionary,
            expected_count=expected_count,
            variants=variants,
        )

    assertion_variants: tuple[tuple[str, tuple[AssertionRegime, ...]], ...] = (
        ("A_HIST", ("history",)),
        ("A_NEG", ("negation",)),
        ("A_NEG_HIST", ("negation", "history")),
        ("A_FAM", ("family",)),
    )
    for name, regimes in assertion_variants:
        rows, decisions, counters = apply_selective_assertions(
            base,
            source_text_by_doc,
            regimes=regimes,
            registry=runtime_registry,
        )
        _materialize_variant(
            name=name,
            module="assertion",
            rows=rows,
            base=base,
            decisions=decisions,
            counters=counters,
            policy_diff={"assertion_regimes": list(regimes)},
            config=config,
            run_dir=run_dir,
            gold=gold,
            dictionary=dictionary,
            expected_count=expected_count,
            variants=variants,
        )

    candidate_variants: tuple[tuple[str, CandidateRegime, int], ...] = (
        ("C_ICD20", "icd", 20),
        ("C_ICD100", "icd", 100),
        ("C_RX_ING", "rxnorm_ingredient", 100),
        ("C_RX_SCD", "rxnorm_clinical_drug", 100),
    )
    for name, regime, limit in candidate_variants:
        rows, decisions, counters = apply_selective_candidates(
            base,
            candidate_registry,
            regime=regime,
            consensus_keys=consensus,
            mention_limit=limit,
        )
        _materialize_variant(
            name=name,
            module="candidate",
            rows=rows,
            base=base,
            decisions=decisions,
            counters=counters,
            policy_diff={
                "candidate_regime": regime,
                "mention_limit": limit,
                "minimum_exact_proposal_sources": 2,
            },
            config=config,
            run_dir=run_dir,
            gold=gold,
            dictionary=dictionary,
            expected_count=expected_count,
            variants=variants,
        )

    missing_gold_ids = sorted(
        set(source_text_by_doc) - set(gold),
        key=_document_sort_key,
    )
    _write_jsonl(
        run_dir / "missing_gold_review_queue.jsonl",
        [
            {"document_id": document_id, "text": source_text_by_doc[document_id]}
            for document_id in missing_gold_ids
        ],
    )
    manifest = {
        "schema_version": "phase1-top10-probe-suite.v1",
        "run_hash": run_hash,
        "run_dir": str(run_dir),
        "holdout_status": "opened" if config.open_holdout else "sealed",
        "tri_source_ready": tri_source_ready,
        "proposal_source_count": len(config.proposal_sources),
        "missing_gold_document_ids": missing_gold_ids,
        "input_hashes": input_hashes,
        "knowledge": {
            "train_document_count": len(train_document_ids),
            "strict_exclusion_count": annotation_report["summary"]["strict_exclusion_count"],
            "boundary_rule_candidates": boundary_audit,
            "candidate_rules": candidate_audit,
        },
        "proposal_summary": proposal_report["summary"],
        "candidate_probe_blocked_reason": None
        if tri_source_ready
        else "Candidate assignment requires exact agreement from at least 2 of 3 independent sources.",
        "variants": variants,
        "public_probe_order": [
            "E_LAB",
            "E_EXCLUSION",
            "E_OVERLAP",
            "one reviewed boundary family at a time",
            "A_HIST",
            "A_NEG",
            "C_ICD20",
            "C_ICD100",
            "C_RX_ING",
            "C_RX_SCD",
        ],
        "promotion_policy": (
            "No variant is auto-promoted from local metrics. Submit isolated probes and record "
            "public results with scripts/record_phase1_public_probe.py."
        ),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "summary.md").write_text(_render_summary(manifest), encoding="utf-8")
    _append_suite_journal(manifest, config.journal_dir)
    return manifest


def _materialize_variant(
    *,
    name: str,
    module: str,
    rows: Mapping[str, list[dict[str, Any]]],
    base: Mapping[str, list[dict[str, Any]]],
    decisions: list[dict[str, Any]],
    counters: Mapping[str, int],
    policy_diff: Mapping[str, Any],
    config: Phase1Top10ProbeConfig,
    run_dir: Path,
    gold: Mapping[str, list[dict[str, Any]]],
    dictionary: DictionaryStore,
    expected_count: int,
    variants: list[dict[str, Any]],
) -> None:
    variant_dir = run_dir / "variants" / name
    output_dir = variant_dir / "output"
    _write_phase1_rows(rows, output_dir)
    isolation_issues = validate_probe_isolation(base, rows, module=module)  # type: ignore[arg-type]
    validation_issues = validate_phase1_submission_dir(
        config.input_dir,
        output_dir,
        dictionary=dictionary,
    )
    if validation_issues:
        raise ValueError(
            f"{name} validation failed: {[issue.to_json() for issue in validation_issues[:5]]}"
        )
    zip_path = variant_dir / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)
    zip_issues = validate_phase1_submission_zip(
        zip_path,
        input_dir=config.input_dir,
        dictionary=dictionary,
        expected_count=expected_count,
    )
    if zip_issues:
        raise ValueError(f"{name} ZIP validation failed: {[issue.to_json() for issue in zip_issues[:5]]}")
    train_report = _score_split(gold, rows, split="train")
    holdout_report = _score_split(gold, rows, split="holdout") if config.open_holdout else None
    _write_jsonl(variant_dir / "decisions.jsonl", decisions)
    _write_json(variant_dir / "counters.json", dict(counters))
    changed = _change_counts(base, rows)
    report = {
        "name": name,
        "module": module,
        "policy_diff": dict(policy_diff),
        "counters": dict(counters),
        "changed": changed,
        "isolation_issues": isolation_issues,
        "validation_issue_count": 0,
        "train": train_report,
        "holdout": holdout_report,
        "probe_ready": not isolation_issues and changed["changed_row_count"] > 0,
        "zip": str(zip_path),
        "zip_sha256": _path_sha256(zip_path),
    }
    _write_json(variant_dir / "variant_report.json", report)
    variants.append(report)


def _score_split(
    gold: Mapping[str, list[dict[str, Any]]],
    predictions: Mapping[str, list[dict[str, Any]]],
    *,
    split: str,
) -> dict[str, Any]:
    ids = [document_id for document_id in gold if manual_gold_split(document_id) == split]
    split_gold = {document_id: gold[document_id] for document_id in ids}
    split_predictions = {document_id: predictions.get(document_id, []) for document_id in ids}
    metrics, errors = score_phase1_documents(split_gold, split_predictions)
    return {
        "document_count": len(ids),
        "metrics": metrics,
        "error_counts": dict(sorted(Counter(row["error_type"] for row in errors).items())),
    }


def _change_counts(
    base: Mapping[str, list[dict[str, Any]]],
    trial: Mapping[str, list[dict[str, Any]]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for document_id in set(base) | set(trial):
        base_rows = base.get(document_id, [])
        trial_rows = trial.get(document_id, [])
        base_identity = {_identity_key(row): row for row in base_rows}
        trial_identity = {_identity_key(row): row for row in trial_rows}
        counts["entity_added"] += len(set(trial_identity) - set(base_identity))
        counts["entity_removed"] += len(set(base_identity) - set(trial_identity))
        for key in set(base_identity) & set(trial_identity):
            before = base_identity[key]
            after = trial_identity[key]
            if before.get("assertions", []) != after.get("assertions", []):
                counts["assertion_changed"] += 1
            if before.get("candidates", []) != after.get("candidates", []):
                counts["candidate_changed"] += 1
    counts["changed_row_count"] = sum(
        counts[key]
        for key in ("entity_added", "entity_removed", "assertion_changed", "candidate_changed")
    )
    return dict(sorted(counts.items()))


def _input_hashes(config: Phase1Top10ProbeConfig) -> dict[str, Any]:
    implementation_paths = [
        Path(__file__),
        Path(__file__).with_name("phase1_entity_gates.py"),
        Path(__file__).with_name("phase1_proposals.py"),
        Path(__file__).with_name("phase1_rule_registry.py"),
        Path(__file__).with_name("phase1_selective_overlays.py"),
    ]
    return {
        "base": {"path": str(config.base), "sha256": _path_sha256(config.base)},
        "proposal_sources": {
            name: {"path": str(path), "sha256": _path_sha256(path)}
            for name, path in sorted(config.proposal_sources.items())
        },
        "dictionary": {"path": str(config.dictionary), "sha256": _path_sha256(config.dictionary)},
        "review_manifest": {
            "path": str(config.review_manifest),
            "sha256": _path_sha256(config.review_manifest),
        },
        "rule_registry": {
            "path": str(config.rule_registry),
            "sha256": _path_sha256(config.rule_registry),
        }
        if config.rule_registry is not None
        else None,
        "minimum_boundary_document_support": config.minimum_boundary_document_support,
        "open_holdout": config.open_holdout,
        "implementation": {
            str(path.name): _path_sha256(path) for path in implementation_paths
        },
    }


def _write_phase1_rows(rows_by_doc: Mapping[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, rows in sorted(rows_by_doc.items(), key=lambda item: _document_sort_key(item[0])):
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    position = row.get("position")
    return (
        row.get("text"),
        row.get("type"),
        tuple(position) if isinstance(position, list) else (),
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = rows if isinstance(rows, list) else []
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )


def _render_summary(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 1 Top 10 Probe Suite",
        "",
        f"- Run: `{manifest['run_hash']}`",
        f"- Holdout: **{manifest['holdout_status']}**",
        f"- Proposal sources: {manifest['proposal_source_count']}",
        f"- Tri-source ready: {manifest['tri_source_ready']}",
        f"- Missing manual gold: {len(manifest['missing_gold_document_ids'])}",
        "",
        "## Variants",
        "",
        "| Variant | Module | Changed rows | Train score | Probe ready | ZIP SHA |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for variant in manifest["variants"]:
        lines.append(
            "| `{name}` | {module} | {changed} | {score:.4f} | {ready} | `{sha}` |".format(
                name=variant["name"],
                module=variant["module"],
                changed=variant["changed"]["changed_row_count"],
                score=float(variant["train"]["metrics"]["score"]),
                ready=variant["probe_ready"],
                sha=variant["zip_sha256"][:12],
            )
        )
    lines.extend(
        [
            "",
            "Local train metrics are diagnostic only. Public promotion must use an isolated probe and "
            "`scripts/record_phase1_public_probe.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_suite_journal(manifest: Mapping[str, Any], journal_dir: Path) -> None:
    journal_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_hash": manifest["run_hash"],
        "run_dir": manifest["run_dir"],
        "holdout_status": manifest["holdout_status"],
        "tri_source_ready": manifest["tri_source_ready"],
        "proposal_source_count": manifest["proposal_source_count"],
        "input_hashes": manifest["input_hashes"],
        "variants": [
            {
                "name": row["name"],
                "module": row["module"],
                "changed": row["changed"],
                "train_metrics": row["train"]["metrics"],
                "probe_ready": row["probe_ready"],
                "zip": row["zip"],
                "zip_sha256": row["zip_sha256"],
            }
            for row in manifest["variants"]
        ],
    }
    jsonl_path = journal_dir / "phase1_top10_probe_runs.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [
        "# Phase 1 Top 10 Probe Runs",
        "",
        "| Run | Holdout | Sources | Ready probes | Best train score |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for item in records:
        variants = item.get("variants", [])
        ready = [row["name"] for row in variants if row.get("probe_ready")]
        best_score = max(
            (float(row["train_metrics"]["score"]) for row in variants),
            default=0.0,
        )
        lines.append(
            f"| `{item['run_hash']}` | {item['holdout_status']} | "
            f"{item['proposal_source_count']} | {', '.join(ready) or '-'} | {best_score:.4f} |"
        )
    lines.append("")
    (journal_dir / "phase1_top10_probe_runs.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
