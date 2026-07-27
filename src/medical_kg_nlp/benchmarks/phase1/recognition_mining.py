"""Run a leakage-safe Phase 1 recognition-knowledge mining experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.benchmarks.phase1.contextual_alias_mining import (
    compile_phase1_contextual_alias_rules,
)
from medical_kg_nlp.benchmarks.phase1.manual_gold_mining import (
    build_phase1_reviewed_recognition_policy,
    load_phase1_manual_gold_mining_corpus,
    recognition_policy_to_data,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import write_json, write_jsonl, write_text
from medical_kg_nlp.mining.lexicon import build_mention_inventory
from medical_kg_nlp.mining.recognition_benchmark import benchmark_recognition_dictionary
from medical_kg_nlp.mining.recognition_knowledge import compile_recognition_knowledge
from medical_kg_nlp.ontology.phase1 import PHASE1_RULE_BY_TYPE
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "Phase1RecognitionMiningConfig",
    "run_phase1_recognition_mining",
]


@dataclass(frozen=True)
class Phase1RecognitionMiningConfig:
    """Inputs and promotion gates for one deterministic mining run."""

    input_dir: Path = Path("data/raw/input")
    gold_dir: Path = Path("data/manual_gold")
    split_manifest: Path = Path("data/manual_gold/holdout_manifest.json")
    annotation_policy: Path = Path(
        "data/manual_gold/compiled/phase1_annotation_policy.yaml"
    )
    baseline_recognition: Path = Path(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    output_root: Path = Path("outputs/mining/knowledge")
    minimum_exact_f1_gain: float = 0.005
    minimum_true_positive_gain: int = 5
    maximum_false_positive_increase: int = 5

    def __post_init__(self) -> None:
        if self.minimum_exact_f1_gain < 0:
            raise ValueError("minimum_exact_f1_gain must be non-negative")
        if self.minimum_true_positive_gain < 1:
            raise ValueError("minimum_true_positive_gain must be positive")
        if self.maximum_false_positive_increase < 0:
            raise ValueError("maximum_false_positive_increase must be non-negative")


def run_phase1_recognition_mining(
    config: Phase1RecognitionMiningConfig,
) -> dict[str, Any]:
    """Mine train-only terms, benchmark on holdout, and write auditable artifacts."""

    train = load_phase1_manual_gold_mining_corpus(
        config.input_dir,
        config.gold_dir,
        config.split_manifest,
        split="train",
    )
    holdout = load_phase1_manual_gold_mining_corpus(
        config.input_dir,
        config.gold_dir,
        config.split_manifest,
        split="holdout",
    )
    inventory = build_mention_inventory(
        train.documents,
        train.annotations,
        min_occurrences=1,
        min_documents=1,
    )
    inventory_rows = tuple(entry.to_dict() for entry in inventory.entries)
    inventory_sha256 = _jsonl_sha256(inventory_rows)
    annotation_policy = read_yaml(config.annotation_policy)
    policy = build_phase1_reviewed_recognition_policy(
        annotation_policy,
        inventory_sha256=inventory_sha256,
    )
    baseline_entries = DictionaryStore.load_entries_jsonl(config.baseline_recognition)
    compilation = compile_recognition_knowledge(
        inventory.entries,
        policy,
        inventory_sha256=inventory_sha256,
        baseline_entries=baseline_entries,
    )
    contextual_aliases = compile_phase1_contextual_alias_rules(
        annotation_policy,
        inventory.entries,
        inventory_sha256=inventory_sha256,
    )

    input_fingerprint = _input_fingerprint(
        config,
        corpus_fingerprint=train.corpus_fingerprint,
        inventory_sha256=inventory_sha256,
    )
    run_hash = hashlib.sha256(
        json.dumps(input_fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = config.output_root / f"phase1-recognition-{run_hash}"
    existing_manifest = run_dir / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Cached run manifest is not an object: {run_dir}")
        if existing.get("input_fingerprint") != input_fingerprint:
            raise ValueError(f"Cached run fingerprint mismatch: {run_dir}")
        return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Refusing to reuse incomplete mining run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = _write_corpus_and_knowledge(
        run_dir,
        train=train,
        holdout=holdout,
        inventory_rows=inventory_rows,
        inventory=inventory,
        policy_data=recognition_policy_to_data(policy),
        compilation=compilation,
        contextual_aliases=contextual_aliases,
    )
    if outputs["inventory_sha256"] != inventory_sha256:
        raise ValueError("Written inventory fingerprint changed during serialization")

    additional_entries = DictionaryStore.load_entries_jsonl(
        run_dir / "recognition_concepts.jsonl"
    )
    benchmark = benchmark_recognition_dictionary(
        holdout.documents,
        holdout.annotations,
        DictionaryStore(baseline_entries),
        DictionaryStore(additional_entries),
        entity_types=tuple(
            rule.internal_type for rule in PHASE1_RULE_BY_TYPE.values()
        ),
    )
    gate = _promotion_gate(benchmark, config)
    write_json(run_dir / "holdout_benchmark.json", benchmark)
    write_json(run_dir / "promotion_gate.json", gate)
    if gate["passed"]:
        terminology_fragment: dict[str, Any] = {
            "additional_recognition_paths": [
                str(run_dir / "recognition_concepts.jsonl")
            ]
        }
        if contextual_aliases.artifact["rules"]:
            terminology_fragment["contextual_alias_path"] = str(
                run_dir / "contextual_alias_rules.yaml"
            )
        write_text(
            run_dir / "pipeline_profile_fragment.yaml",
            yaml.safe_dump(
                {"terminology": terminology_fragment},
                allow_unicode=True,
                sort_keys=False,
            ),
        )

    manifest = {
        "schema_version": "phase1-recognition-mining-run.v1",
        "run_hash": run_hash,
        "run_dir": str(run_dir),
        "input_fingerprint": input_fingerprint,
        "train": {
            "document_count": len(train.documents),
            "annotation_count": len(train.annotations),
        },
        "holdout": {
            "document_count": len(holdout.documents),
            "annotation_count": len(holdout.annotations),
        },
        "inventory": inventory.report,
        "compilation": compilation.report,
        "contextual_aliases": contextual_aliases.report,
        "benchmark": benchmark,
        "promotion_gate": gate,
        "outputs": outputs,
    }
    # The manifest cannot contain its own digest without creating a recursive hash. The
    # caller can hash this file from the stable path when publishing an artifact bundle.
    write_json(existing_manifest, manifest)
    return manifest


def _write_corpus_and_knowledge(
    run_dir: Path,
    *,
    train: Any,
    holdout: Any,
    inventory_rows: tuple[dict[str, Any], ...],
    inventory: Any,
    policy_data: dict[str, Any],
    compilation: Any,
    contextual_aliases: Any,
) -> dict[str, str]:
    """Write immutable inputs and compiler outputs using atomic mining IO."""

    outputs = {
        "train_documents_sha256": write_jsonl(
            run_dir / "train" / "documents.jsonl",
            (document.to_dict() for document in train.documents),
        ),
        "train_annotations_sha256": write_jsonl(
            run_dir / "train" / "annotations.jsonl",
            (annotation.to_dict() for annotation in train.annotations),
        ),
        "holdout_documents_sha256": write_jsonl(
            run_dir / "holdout" / "documents.jsonl",
            (document.to_dict() for document in holdout.documents),
        ),
        "holdout_annotations_sha256": write_jsonl(
            run_dir / "holdout" / "annotations.jsonl",
            (annotation.to_dict() for annotation in holdout.annotations),
        ),
        "inventory_sha256": write_jsonl(run_dir / "inventory.jsonl", inventory_rows),
        "inventory_conflicts_sha256": write_jsonl(
            run_dir / "inventory_conflicts.jsonl", inventory.conflicts
        ),
        "inventory_report_sha256": write_json(
            run_dir / "inventory_report.json", inventory.report
        ),
        "policy_sha256": write_text(
            run_dir / "recognition_policy.yaml",
            yaml.safe_dump(policy_data, allow_unicode=True, sort_keys=False),
        ),
        "recognition_concepts_sha256": write_jsonl(
            run_dir / "recognition_concepts.jsonl", compilation.concepts
        ),
        "decisions_sha256": write_jsonl(
            run_dir / "compilation_decisions.jsonl", compilation.decisions
        ),
        "compilation_report_sha256": write_json(
            run_dir / "compilation_report.json", compilation.report
        ),
        "contextual_alias_rules_sha256": write_text(
            run_dir / "contextual_alias_rules.yaml",
            yaml.safe_dump(
                contextual_aliases.artifact,
                allow_unicode=True,
                sort_keys=False,
            ),
        ),
        "contextual_alias_decisions_sha256": write_jsonl(
            run_dir / "contextual_alias_decisions.jsonl",
            contextual_aliases.decisions,
        ),
        "contextual_alias_report_sha256": write_json(
            run_dir / "contextual_alias_report.json",
            contextual_aliases.report,
        ),
    }
    return outputs


def _promotion_gate(
    benchmark: dict[str, Any],
    config: Phase1RecognitionMiningConfig,
) -> dict[str, Any]:
    baseline = benchmark["baseline"]["metrics"]
    enriched = benchmark["enriched"]["metrics"]
    f1_gain = float(enriched["f1"]) - float(baseline["f1"])
    precision_gain = float(enriched["precision"]) - float(baseline["precision"])
    true_positive_gain = int(enriched["true_positive_count"]) - int(
        baseline["true_positive_count"]
    )
    false_positive_increase = int(enriched["false_positive_count"]) - int(
        baseline["false_positive_count"]
    )
    checks = {
        "minimum_exact_f1_gain": f1_gain >= config.minimum_exact_f1_gain,
        "precision_not_lower": precision_gain >= 0.0,
        "minimum_true_positive_gain": (
            true_positive_gain >= config.minimum_true_positive_gain
        ),
        "maximum_false_positive_increase": (
            false_positive_increase <= config.maximum_false_positive_increase
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "exact_f1": f1_gain,
            "exact_precision": precision_gain,
            "true_positive_count": true_positive_gain,
            "false_positive_count": false_positive_increase,
        },
        "thresholds": {
            "minimum_exact_f1_gain": config.minimum_exact_f1_gain,
            "minimum_true_positive_gain": config.minimum_true_positive_gain,
            "maximum_false_positive_increase": config.maximum_false_positive_increase,
        },
    }


def _input_fingerprint(
    config: Phase1RecognitionMiningConfig,
    *,
    corpus_fingerprint: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    package_root = Path(__file__).parents[2]
    return {
        "corpus_fingerprint": corpus_fingerprint,
        "inventory_sha256": inventory_sha256,
        "split_manifest_sha256": sha256_file(config.split_manifest),
        "annotation_policy_sha256": sha256_file(config.annotation_policy),
        "baseline_recognition_sha256": sha256_file(config.baseline_recognition),
        "implementation": {
            "manual_gold_mining.py": sha256_file(
                Path(__file__).with_name("manual_gold_mining.py")
            ),
            "recognition_mining.py": sha256_file(__file__),
            "recognition_knowledge.py": sha256_file(
                package_root / "mining" / "recognition_knowledge.py"
            ),
            "recognition_benchmark.py": sha256_file(
                package_root / "mining" / "recognition_benchmark.py"
            ),
            "contextual_alias_mining.py": sha256_file(
                Path(__file__).with_name("contextual_alias_mining.py")
            ),
        },
        "gates": {
            "minimum_exact_f1_gain": config.minimum_exact_f1_gain,
            "minimum_true_positive_gain": config.minimum_true_positive_gain,
            "maximum_false_positive_increase": config.maximum_false_positive_increase,
        },
    }


def _jsonl_sha256(rows: tuple[dict[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    return digest.hexdigest()
