from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.entity_wer_report import (
    build_entity_wer_report,
    write_entity_wer_report,
)
from medical_kg_nlp.evaluation.manual_gold import (
    evaluate_manual_gold,
    load_phase1_directory,
    verify_manual_gold_split_manifest,
)
from medical_kg_nlp.evaluation.phase1 import (
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
from medical_kg_nlp.evaluation.phase1_rule_registry import write_phase1_rule_registry
from medical_kg_nlp.utils.io import read_source_text, read_yaml


@dataclass(frozen=True)
class Phase1EntityAblationConfig:
    base: Path
    expected_base_sha256: str
    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    split_manifest: Path = Path("data/manual_gold/holdout_manifest.json")
    annotation_policy: Path = Path(
        "data/manual_gold/compiled/phase1_annotation_policy.yaml"
    )
    dictionary_paths: tuple[Path, ...] = (
        Path("data/standards/phase1_seed_tt06_controlled_concepts.jsonl"),
        Path("data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl"),
    )
    source_stages: tuple[tuple[str, Path], ...] = ()
    output_root: Path = Path("outputs/evaluation/phase1_entity_ablations")
    journal_dir: Path = Path("outputs/loops/journal")
    public_wer: float | None = None
    minimum_boundary_document_support: int = 2


def run_phase1_entity_ablations(config: Phase1EntityAblationConfig) -> dict[str, Any]:
    if config.minimum_boundary_document_support < 2:
        raise ValueError("Boundary rules require support from at least two training documents.")
    base_sha256 = _path_sha256(config.base)
    if base_sha256 != config.expected_base_sha256:
        raise ValueError(
            "Frozen baseline SHA-256 mismatch: "
            f"expected {config.expected_base_sha256}, got {base_sha256}."
        )

    split_manifest = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    verify_manual_gold_split_manifest(split_manifest, config.gold_dir, config.input_dir)
    gold = load_phase1_directory(config.gold_dir)
    source_text_by_doc = {
        path.stem: read_source_text(path)
        for path in config.input_dir.glob("*.txt")
        if path.stem.isdigit()
    }
    base = load_phase1_output_source(config.base)
    annotation_policy = read_yaml(config.annotation_policy)
    dictionary = _load_combined_dictionary(config.dictionary_paths)
    source_stages = [
        (name, load_phase1_output_source(path)) for name, path in config.source_stages
    ]

    input_fingerprint = _input_fingerprint(config, base_sha256, split_manifest)
    run_hash = hashlib.sha256(
        json.dumps(input_fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = config.output_root / f"entity_ablation_{run_hash}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing entity ablation run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Rules are discovered only on train. They are activated inside this diagnostic run so the
    # frozen holdout can measure generalization; this does not modify the production registry.
    boundary_registry, boundary_audit = compile_boundary_rule_candidates(
        gold,
        base,
        split="train",
        minimum_document_support=config.minimum_boundary_document_support,
        review_status="reviewed",
    )
    boundary_audit["activation_scope"] = "experiment_only"
    write_phase1_rule_registry(boundary_registry, run_dir / "boundary_registry_experiment.yaml")
    _write_json(run_dir / "boundary_rule_audit.json", boundary_audit)

    variants = _variant_configs()
    variant_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    traces: dict[str, tuple[list[dict[str, Any]], dict[str, int]]] = {}
    for name, gate_config in variants:
        if name == "E0_BASE":
            rows = {document_id: [dict(row) for row in values] for document_id, values in base.items()}
            decisions: list[dict[str, Any]] = []
            counters = {"output_entity_total": sum(len(values) for values in rows.values())}
        else:
            rows, decisions, counters = apply_phase1_entity_gates(
                base,
                source_text_by_doc,
                config=gate_config,
                annotation_policy=annotation_policy,
                registry=boundary_registry,
            )
        variant_rows[name] = rows
        traces[name] = (decisions, counters)

    baseline_report = build_entity_wer_report(
        gold_by_doc=gold,
        pred_by_doc=base,
        documents_by_doc=source_text_by_doc,
        stages=source_stages,
        annotation_policy=annotation_policy,
        public_wer=config.public_wer,
        final_source_name="baseline_only",
    )
    baseline_splits = baseline_report["splits"]
    reports: list[dict[str, Any]] = []
    for name, gate_config in variants:
        rows = variant_rows[name]
        decisions, counters = traces[name]
        report = _materialize_variant(
            name=name,
            gate_config=gate_config,
            rows=rows,
            base=base,
            decisions=decisions,
            counters=counters,
            gold=gold,
            source_text_by_doc=source_text_by_doc,
            source_stages=source_stages,
            annotation_policy=annotation_policy,
            dictionary=dictionary,
            config=config,
            run_dir=run_dir,
            baseline_splits=baseline_splits,
        )
        reports.append(report)

    manifest = {
        "schema_version": "phase1-entity-ablation.v1",
        "run_hash": run_hash,
        "run_dir": str(run_dir),
        "holdout_status": "frozen_then_opened_for_final_evaluation",
        "input_fingerprint": input_fingerprint,
        "split_manifest": split_manifest,
        "boundary_rule_audit": boundary_audit,
        "variants": reports,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_summary_csv(reports, run_dir / "summary.csv")
    (run_dir / "summary.md").write_text(
        _render_summary(manifest),
        encoding="utf-8",
    )
    _append_journal(manifest, config.journal_dir)
    return manifest


def _variant_configs() -> tuple[tuple[str, Phase1EntityGateConfig], ...]:
    return (
        ("E0_BASE", Phase1EntityGateConfig(resolve_overlaps=False)),
        (
            "E_LAB_RESULT_RETYPE",
            Phase1EntityGateConfig(lab_gate=True, resolve_overlaps=False),
        ),
        (
            "E_MEDICATION_FULL_SPAN",
            Phase1EntityGateConfig(
                medication_full_span=True,
                resolve_overlaps=False,
            ),
        ),
        (
            "E_BOUNDARY_DIAGNOSIS",
            Phase1EntityGateConfig(
                boundary_stages=("boundary_diagnosis",),
                resolve_overlaps=False,
            ),
        ),
        (
            "E_BOUNDARY_SYMPTOM_PREFIX",
            Phase1EntityGateConfig(
                boundary_stages=("boundary_symptom_prefix",),
                resolve_overlaps=False,
            ),
        ),
        (
            "E_BOUNDARY_SYMPTOM_COURSE",
            Phase1EntityGateConfig(
                boundary_stages=("boundary_symptom_course",),
                resolve_overlaps=False,
            ),
        ),
        (
            "E_BOUNDARY_SYMPTOM_ALL",
            Phase1EntityGateConfig(
                boundary_stages=("boundary_symptom_prefix", "boundary_symptom_course"),
                resolve_overlaps=False,
            ),
        ),
    )


def _materialize_variant(
    *,
    name: str,
    gate_config: Phase1EntityGateConfig,
    rows: Mapping[str, list[dict[str, Any]]],
    base: Mapping[str, list[dict[str, Any]]],
    decisions: list[dict[str, Any]],
    counters: Mapping[str, int],
    gold: dict[str, list[dict[str, Any]]],
    source_text_by_doc: Mapping[str, str],
    source_stages: Sequence[tuple[str, dict[str, list[dict[str, Any]]]]],
    annotation_policy: Mapping[str, Any],
    dictionary: DictionaryStore,
    config: Phase1EntityAblationConfig,
    run_dir: Path,
    baseline_splits: Mapping[str, Any],
) -> dict[str, Any]:
    variant_dir = run_dir / "variants" / name
    output_dir = variant_dir / "output"
    _write_phase1_rows(rows, output_dir)
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
        expected_count=len(source_text_by_doc),
    )
    if zip_issues:
        raise ValueError(
            f"{name} ZIP validation failed: {[issue.to_json() for issue in zip_issues[:5]]}"
        )

    wer_report = build_entity_wer_report(
        gold_by_doc=gold,
        pred_by_doc=dict(rows),
        documents_by_doc=source_text_by_doc,
        stages=source_stages,
        annotation_policy=annotation_policy,
        public_wer=config.public_wer if name == "E0_BASE" else None,
        final_source_name=name.lower(),
    )
    write_entity_wer_report(wer_report, variant_dir / "wer")
    manual_report = evaluate_manual_gold(gold, dict(rows))
    _write_json(variant_dir / "manual_gold_metrics.json", manual_report)
    _write_jsonl(variant_dir / "decisions.jsonl", decisions)
    _write_json(variant_dir / "counters.json", dict(counters))

    changes = _entity_change_counts(base, rows)
    split_deltas = {
        split: _split_delta(wer_report["splits"][split], baseline_splits[split])
        for split in ("all", "train", "holdout")
    }
    holdout_delta = split_deltas["holdout"]
    all_delta = split_deltas["all"]
    changed = changes["changed_identity_count"] > 0
    gate_passed = (
        name != "E0_BASE"
        and changed
        and holdout_delta["wer_reduction"] > 0.0
        and all_delta["wer_reduction"] >= 0.0
    )
    return {
        "name": name,
        "policy": {
            "lab_gate": gate_config.lab_gate,
            "medication_full_span": gate_config.medication_full_span,
            "boundary_stages": list(gate_config.boundary_stages),
            "resolve_overlaps": gate_config.resolve_overlaps,
        },
        "changes": changes,
        "decision_count": len(decisions),
        "counters": dict(counters),
        "splits": wer_report["splits"],
        "split_deltas": split_deltas,
        "gate_passed": gate_passed,
        "decision": "keep_candidate" if gate_passed else "baseline" if name == "E0_BASE" else "reject",
        "validation_issue_count": 0,
        "zip": str(zip_path),
        "zip_sha256": _path_sha256(zip_path),
        "wer_report": str(variant_dir / "wer" / "metrics.json"),
    }


def _split_delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    current_errors = current["alignment_error_counts"]
    baseline_errors = baseline["alignment_error_counts"]
    return {
        "text_score_gain": round(
            float(current["text_score"]) - float(baseline["text_score"]), 6
        ),
        "wer_reduction": round(
            float(baseline["micro_wer_proxy"]) - float(current["micro_wer_proxy"]), 6
        ),
        "missing_reduction": int(baseline_errors["missing"]) - int(current_errors["missing"]),
        "spurious_reduction": int(baseline_errors["spurious"]) - int(current_errors["spurious"]),
        "boundary_reduction": int(baseline_errors["boundary"]) - int(current_errors["boundary"]),
    }


def _entity_change_counts(
    base: Mapping[str, list[dict[str, Any]]],
    trial: Mapping[str, list[dict[str, Any]]],
) -> dict[str, int]:
    removed = 0
    added = 0
    for document_id in set(base) | set(trial):
        base_keys = {_identity_key(row) for row in base.get(document_id, [])}
        trial_keys = {_identity_key(row) for row in trial.get(document_id, [])}
        removed += len(base_keys - trial_keys)
        added += len(trial_keys - base_keys)
    return {
        "entity_removed": removed,
        "entity_added": added,
        "changed_identity_count": removed + added,
    }


def _load_combined_dictionary(paths: Sequence[Path]) -> DictionaryStore:
    entries = []
    for path in paths:
        entries.extend(DictionaryStore.from_jsonl(path).entries)
    return DictionaryStore(entries)


def _input_fingerprint(
    config: Phase1EntityAblationConfig,
    base_sha256: str,
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = Path(__file__)
    gate_implementation = implementation.with_name("phase1_entity_gates.py")
    phase1_implementation = implementation.with_name("phase1.py")
    return {
        "base": {"path": str(config.base), "sha256": base_sha256},
        "manual_gold_corpus_sha256": split_manifest["corpus"]["fingerprint_sha256"],
        "split_manifest": {
            "path": str(config.split_manifest),
            "sha256": _path_sha256(config.split_manifest),
        },
        "annotation_policy": {
            "path": str(config.annotation_policy),
            "sha256": _path_sha256(config.annotation_policy),
        },
        "dictionaries": [
            {"path": str(path), "sha256": _path_sha256(path)}
            for path in config.dictionary_paths
        ],
        "source_stages": [
            {"name": name, "path": str(path), "sha256": _path_sha256(path)}
            for name, path in config.source_stages
        ],
        "minimum_boundary_document_support": config.minimum_boundary_document_support,
        "implementation": {
            implementation.name: _path_sha256(implementation),
            gate_implementation.name: _path_sha256(gate_implementation),
            phase1_implementation.name: _path_sha256(phase1_implementation),
        },
    }


def _write_phase1_rows(
    rows_by_doc: Mapping[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, rows in sorted(
        rows_by_doc.items(), key=lambda item: _document_sort_key(item[0])
    ):
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_summary_csv(reports: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "name",
        "decision_count",
        "changed_identity_count",
        "all_wer",
        "all_wer_reduction",
        "holdout_wer",
        "holdout_wer_reduction",
        "holdout_missing_reduction",
        "holdout_spurious_reduction",
        "holdout_boundary_reduction",
        "decision",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "name": report["name"],
                    "decision_count": report["decision_count"],
                    "changed_identity_count": report["changes"]["changed_identity_count"],
                    "all_wer": report["splits"]["all"]["micro_wer_proxy"],
                    "all_wer_reduction": report["split_deltas"]["all"]["wer_reduction"],
                    "holdout_wer": report["splits"]["holdout"]["micro_wer_proxy"],
                    "holdout_wer_reduction": report["split_deltas"]["holdout"]["wer_reduction"],
                    "holdout_missing_reduction": report["split_deltas"]["holdout"]["missing_reduction"],
                    "holdout_spurious_reduction": report["split_deltas"]["holdout"]["spurious_reduction"],
                    "holdout_boundary_reduction": report["split_deltas"]["holdout"]["boundary_reduction"],
                    "decision": report["decision"],
                }
            )


def _render_summary(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 1 Entity Ablations",
        "",
        f"- Run: `{manifest['run_hash']}`",
        f"- Holdout: `{manifest['holdout_status']}`",
        f"- Gold corpus: `{manifest['split_manifest']['corpus']['fingerprint_sha256']}`",
        f"- Boundary rules compiled from train: {manifest['boundary_rule_audit']['compiled_rule_count']}",
        "",
        "| Variant | Changes | All WER | Delta | Holdout WER | Delta | Missing | Spurious | Boundary | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in manifest["variants"]:
        holdout = report["split_deltas"]["holdout"]
        lines.append(
            f"| `{report['name']}` | {report['changes']['changed_identity_count']} | "
            f"{report['splits']['all']['micro_wer_proxy']:.4f} | "
            f"{report['split_deltas']['all']['wer_reduction']:+.4f} | "
            f"{report['splits']['holdout']['micro_wer_proxy']:.4f} | "
            f"{holdout['wer_reduction']:+.4f} | {holdout['missing_reduction']:+d} | "
            f"{holdout['spurious_reduction']:+d} | {holdout['boundary_reduction']:+d} | "
            f"**{report['decision']}** |"
        )
    lines.extend(
        [
            "",
            "A positive delta is a WER reduction. `keep_candidate` is a local diagnostic gate, not automatic public promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_journal(manifest: Mapping[str, Any], journal_dir: Path) -> None:
    journal_dir.mkdir(parents=True, exist_ok=True)
    compact = {
        "schema_version": manifest["schema_version"],
        "run_hash": manifest["run_hash"],
        "run_dir": manifest["run_dir"],
        "holdout_status": manifest["holdout_status"],
        "manual_gold_corpus_sha256": manifest["split_manifest"]["corpus"][
            "fingerprint_sha256"
        ],
        "variants": [
            {
                "name": row["name"],
                "decision": row["decision"],
                "all_wer": row["splits"]["all"]["micro_wer_proxy"],
                "holdout_wer": row["splits"]["holdout"]["micro_wer_proxy"],
                "holdout_wer_reduction": row["split_deltas"]["holdout"]["wer_reduction"],
                "zip_sha256": row["zip_sha256"],
            }
            for row in manifest["variants"]
        ],
    }
    path = journal_dir / "phase1_entity_ablation_runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(compact, ensure_ascii=False, sort_keys=True) + "\n")


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
