"""Joint XLM-R masked-language and terminology-contrastive pretraining.

Heavy dependencies are imported only inside the training entry point. The core
package can validate DAPT data and provenance without Torch or Transformers.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.training.dapt_run_spec import XlmrDaptTrainingConfig
from medical_kg_nlp.training.huggingface_token_classifier import (
    fingerprint_model_directory,
)
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import collect_git_metadata

__all__ = ["train_xlmr_dapt", "xlmr_dapt_input_provenance"]


@dataclass(frozen=True, slots=True)
class _ResumeState:
    global_step: int
    consumed_micro_batches: int
    completed_updates: int
    totals: dict[str, float]


@dataclass(slots=True)
class _MlmTextCollator:
    tokenizer: Any
    mlm_collator: Any
    max_length: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        tokenized = self.tokenizer(
            [str(row["text"]) for row in rows],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_special_tokens_mask=True,
        )
        features = [
            {key: values[index] for key, values in tokenized.items()}
            for index in range(len(rows))
        ]
        return dict(self.mlm_collator(features))


@dataclass(slots=True)
class _SynonymPairCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        concepts: dict[str, int] = {}
        concept_ids = []
        for row in rows:
            concept = str(row["concept_id"])
            if concept not in concepts:
                concepts[concept] = len(concepts)
            concept_ids.append(concepts[concept])
        return {
            "left": self.tokenizer(
                [str(row["left"]) for row in rows],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ),
            "right": self.tokenizer(
                [str(row["right"]) for row in rows],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ),
            "concept_ids": concept_ids,
        }


def train_xlmr_dapt(
    config: XlmrDaptTrainingConfig,
    *,
    mixed_precision: str,
    manifest_root: Path,
    resume_from_checkpoint: Path | None = None,
    max_steps_override: int | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Run joint DAPT and save one reusable XLM-R checkpoint.

    Round 2 rows enter only the MLM loader. The contrastive loader is built from
    the separately fingerprinted terminology-pair artifact.
    """

    if mixed_precision not in {"bf16", "fp16"}:
        raise ValueError("DAPT mixed_precision must be bf16 or fp16")
    source_control = _clean_source_control()
    dependencies = _load_dependencies()
    torch = dependencies["torch"]
    accelerator = dependencies["Accelerator"](
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    dependencies["set_seed"](config.seed)
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        use_fast=True,
    )
    mlm_model = dependencies["AutoModelForMaskedLM"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    joint_model = _joint_model(
        dependencies,
        mlm_model,
        contrastive_weight=config.contrastive_weight,
        contrastive_temperature=config.contrastive_temperature,
    )
    mlm_loader = _mlm_loader(config, tokenizer, dependencies)
    pair_loader = _pair_loader(config, tokenizer, dependencies)
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    max_steps = config.max_steps if max_steps_override is None else max_steps_override
    if max_steps < 1 or max_steps > config.max_steps:
        raise ValueError("DAPT max_steps_override must be in [1, configured max_steps]")
    scheduler = dependencies["get_linear_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=math.ceil(max_steps * config.warmup_ratio),
        num_training_steps=max_steps,
    )
    joint_model, optimizer, mlm_loader, pair_loader, scheduler = accelerator.prepare(
        joint_model,
        optimizer,
        mlm_loader,
        pair_loader,
        scheduler,
    )

    output = config.output_dir if output_dir_override is None else output_dir_override
    _relative_path(output, manifest_root)
    output.mkdir(parents=True, exist_ok=True)
    training_identity = _training_identity(
        config,
        mixed_precision=mixed_precision,
        manifest_root=manifest_root,
    )
    resume_state = _ResumeState(
        global_step=0,
        consumed_micro_batches=0,
        completed_updates=0,
        totals={"loss": 0.0, "mlm_loss": 0.0, "contrastive_loss": 0.0},
    )
    if resume_from_checkpoint is not None:
        resume_state = _load_checkpoint(
            accelerator,
            resume_from_checkpoint,
            maximum_step=max_steps,
            expected_training_identity=training_identity,
        )
    mlm_iterator = iter(mlm_loader)
    pair_iterator = iter(pair_loader)
    mlm_iterator, pair_iterator = _fast_forward_loaders(
        mlm_iterator,
        mlm_loader,
        pair_iterator,
        pair_loader,
        batches=resume_state.consumed_micro_batches,
    )
    global_step = resume_state.global_step
    totals = dict(resume_state.totals)
    completed_updates = resume_state.completed_updates
    joint_model.train()

    while global_step < max_steps:
        with accelerator.accumulate(joint_model):
            batch, mlm_iterator = _next_batch(mlm_iterator, mlm_loader)
            pair_batch, pair_iterator = _next_batch(pair_iterator, pair_loader)
            result = joint_model(batch, pair_batch)
            loss = result["loss"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(joint_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        completed_updates += 1
        for key in totals:
            reduced = accelerator.reduce(result[key].detach(), reduction="mean")
            totals[key] += float(reduced.item())
        if global_step % config.checkpoint_interval == 0:
            _save_checkpoint(
                accelerator,
                output,
                global_step=global_step,
                consumed_micro_batches=(
                    global_step * config.gradient_accumulation_steps
                ),
                completed_updates=completed_updates,
                totals=totals,
                training_identity=training_identity,
            )

    final_model = output / "final-model"
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(joint_model)
    if accelerator.is_main_process:
        unwrapped.mlm_model.save_pretrained(
            final_model,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(final_model)
    accelerator.wait_for_everyone()
    metrics = {
        key: value / completed_updates if completed_updates else 0.0
        for key, value in totals.items()
    }
    manifest = {
        "schema_version": "xlmr-dapt-artifact.v1",
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "parameter_count": sum(
                parameter.numel()
                for parameter in unwrapped.mlm_model.parameters()
            ),
            "output": _relative_path(final_model, manifest_root),
            "fingerprint": (
                fingerprint_model_directory(final_model)
                if accelerator.is_main_process
                else None
            ),
        },
        "objectives": {
            "masked_language_modeling": {
                "weight": 1.0,
                "probability": config.mlm_probability,
                "lanes": [lane.lane_id for lane in config.lanes],
            },
            "synonym_contrastive": {
                "weight": config.contrastive_weight,
                "temperature": config.contrastive_temperature,
                "round2_included": False,
            },
        },
        "inputs": xlmr_dapt_input_provenance(config, manifest_root=manifest_root),
        "training": {
            "global_step": global_step,
            "configured_max_steps": config.max_steps,
            "executed_max_steps": max_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "seed": config.seed,
            "smoke": max_steps_override is not None,
            "training_identity_sha256": training_identity,
        },
        "metrics": metrics,
        "source_control": source_control,
    }
    if accelerator.is_main_process:
        write_json(output / "run_manifest.json", manifest)
    accelerator.wait_for_everyone()
    return manifest


def xlmr_dapt_input_provenance(
    config: XlmrDaptTrainingConfig,
    *,
    manifest_root: Path,
) -> dict[str, Any]:
    """Describe objective-isolated bytes consumed by one DAPT run.

    Round 2 appears only among MLM lanes. Contrastive provenance is an exact
    allowlist of terminology sources from the run specification.
    """

    return {
        "corpus_manifest": {
            "path": _relative_path(config.corpus_manifest_path, manifest_root),
            "sha256": sha256_file(config.corpus_manifest_path),
        },
        "mlm_lanes": [
            {
                "lane_id": lane.lane_id,
                "kind": lane.kind,
                "path": _relative_path(lane.path, manifest_root),
                "sha256": lane.sha256,
                "record_count": lane.record_count,
                "sampling_weight": lane.sampling_weight,
                "supervision": "none",
                "objective": "masked_language_modeling",
            }
            for lane in config.lanes
        ],
        "synonym_pairs": {
            "path": _relative_path(config.synonym_pairs_path, manifest_root),
            "sha256": sha256_file(config.synonym_pairs_path),
            "manifest": _relative_path(
                config.synonym_manifest_path,
                manifest_root,
            ),
            "manifest_sha256": sha256_file(config.synonym_manifest_path),
            "round2_included": False,
            "sources": [
                {
                    "path": _relative_path(path, manifest_root),
                    "sha256": source_sha256,
                }
                for path, source_sha256 in config.synonym_source_fingerprints
            ],
        },
    }


def _mlm_loader(
    config: XlmrDaptTrainingConfig,
    tokenizer: Any,
    dependencies: dict[str, Any],
) -> Any:
    datasets = []
    weights: list[float] = []
    for lane in config.lanes:
        dataset = dependencies["load_dataset"](
            "json",
            data_files=str(lane.path),
            split="train",
        )
        datasets.append(dataset)
        weights.extend([lane.sampling_weight] * len(dataset))
    combined = dependencies["concatenate_datasets"](datasets)
    sampler = dependencies["WeightedRandomSampler"](
        weights,
        num_samples=max(
            len(combined),
            config.max_steps
            * config.gradient_accumulation_steps
            * config.mlm_batch_size,
        ),
        replacement=True,
        generator=dependencies["torch"].Generator().manual_seed(config.seed),
    )
    mlm_collator = dependencies["DataCollatorForLanguageModeling"](
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
        seed=config.seed,
    )
    return dependencies["DataLoader"](
        combined,
        batch_size=config.mlm_batch_size,
        sampler=sampler,
        collate_fn=_MlmTextCollator(
            tokenizer=tokenizer,
            mlm_collator=mlm_collator,
            max_length=config.max_length,
        ),
        num_workers=config.preprocessing_workers,
        pin_memory=True,
    )


def _pair_loader(
    config: XlmrDaptTrainingConfig,
    tokenizer: Any,
    dependencies: dict[str, Any],
) -> Any:
    dataset = dependencies["load_dataset"](
        "json",
        data_files=str(config.synonym_pairs_path),
        split="train",
    )
    if len(dataset) < 2:
        raise ValueError("DAPT synonym contrastive training requires at least two pairs")
    return dependencies["DataLoader"](
        dataset,
        batch_size=config.contrastive_batch_size,
        shuffle=True,
        collate_fn=_SynonymPairCollator(
            tokenizer=tokenizer,
            max_length=min(config.max_length, 128),
        ),
        num_workers=config.preprocessing_workers,
        pin_memory=True,
        drop_last=len(dataset) >= config.contrastive_batch_size,
        generator=dependencies["torch"].Generator().manual_seed(config.seed + 1),
    )


def _joint_model(
    dependencies: dict[str, Any],
    mlm_model: Any,
    *,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> Any:
    torch = dependencies["torch"]
    functional = dependencies["functional"]

    # MODEL: Torch remains a lazy optional dependency; the runtime base class is
    # intentionally unavailable to static analysis in a core-only environment.
    class JointDaptModel(torch.nn.Module):  # type: ignore[misc, name-defined]
        """One shared encoder optimized by MLM and supervised contrastive loss."""

        def __init__(self) -> None:
            super().__init__()
            self.mlm_model = mlm_model

        def forward(
            self,
            mlm_batch: dict[str, Any],
            pair_batch: dict[str, Any],
        ) -> dict[str, Any]:
            mlm_output = self.mlm_model(**mlm_batch)
            left_output = self.mlm_model.base_model(
                **pair_batch["left"],
                return_dict=True,
            )
            right_output = self.mlm_model.base_model(
                **pair_batch["right"],
                return_dict=True,
            )
            left = _mean_pool(
                left_output.last_hidden_state,
                pair_batch["left"]["attention_mask"],
                functional,
            )
            right = _mean_pool(
                right_output.last_hidden_state,
                pair_batch["right"]["attention_mask"],
                functional,
            )
            contrastive_loss = _supervised_contrastive_loss(
                torch,
                functional,
                left,
                right,
                pair_batch["concept_ids"],
                temperature=contrastive_temperature,
            )
            loss = mlm_output.loss + contrastive_weight * contrastive_loss
            return {
                "loss": loss,
                "mlm_loss": mlm_output.loss,
                "contrastive_loss": contrastive_loss,
            }

    return JointDaptModel()


def _mean_pool(hidden: Any, attention_mask: Any, functional: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return functional.normalize(pooled, p=2, dim=-1)


def _supervised_contrastive_loss(
    torch: Any,
    functional: Any,
    left: Any,
    right: Any,
    concept_ids: list[int],
    *,
    temperature: float,
) -> Any:
    embeddings = torch.cat((left, right), dim=0)
    labels = torch.tensor(
        concept_ids + concept_ids,
        device=embeddings.device,
    )
    similarity = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(
        similarity.shape[0],
        dtype=torch.bool,
        device=similarity.device,
    )
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    denominator = similarity.masked_fill(diagonal, float("-inf"))
    numerator = similarity.masked_fill(~positives, float("-inf"))
    return -(
        torch.logsumexp(numerator, dim=1)
        - torch.logsumexp(denominator, dim=1)
    ).mean()


def _save_checkpoint(
    accelerator: Any,
    output_dir: Path,
    *,
    global_step: int,
    consumed_micro_batches: int,
    completed_updates: int,
    totals: dict[str, float],
    training_identity: str,
) -> None:
    checkpoint = output_dir / "checkpoints" / f"step-{global_step:08d}"
    accelerator.save_state(checkpoint)
    if accelerator.is_main_process:
        write_json(
            checkpoint / "training_state.json",
            {
                "schema_version": "xlmr-dapt-checkpoint.v1",
                "global_step": global_step,
                "consumed_micro_batches": consumed_micro_batches,
                "completed_updates": completed_updates,
                "loss_totals": totals,
                "training_identity_sha256": training_identity,
            },
        )


def _load_checkpoint(
    accelerator: Any,
    checkpoint: Path,
    *,
    maximum_step: int,
    expected_training_identity: str,
) -> _ResumeState:
    state_path = checkpoint / "training_state.json"
    if not state_path.is_file():
        raise ValueError("DAPT checkpoint training_state.json is absent")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "xlmr-dapt-checkpoint.v1":
        raise ValueError("Unsupported DAPT checkpoint schema")
    if state.get("training_identity_sha256") != expected_training_identity:
        raise ValueError("DAPT checkpoint belongs to a different immutable run")
    step = int(state.get("global_step", -1))
    if step < 0 or step >= maximum_step:
        raise ValueError("DAPT checkpoint global_step is invalid")
    micro_batches = int(state.get("consumed_micro_batches", -1))
    completed_updates = int(state.get("completed_updates", -1))
    if micro_batches < step or completed_updates != step:
        raise ValueError("DAPT checkpoint progress metadata is invalid")
    raw_totals = state.get("loss_totals")
    if not isinstance(raw_totals, dict):
        raise ValueError("DAPT checkpoint loss totals are absent")
    totals = {
        key: float(raw_totals[key])
        for key in ("loss", "mlm_loss", "contrastive_loss")
    }
    accelerator.load_state(checkpoint)
    return _ResumeState(
        global_step=step,
        consumed_micro_batches=micro_batches,
        completed_updates=completed_updates,
        totals=totals,
    )


def _fast_forward_loaders(
    mlm_iterator: Any,
    mlm_loader: Any,
    pair_iterator: Any,
    pair_loader: Any,
    *,
    batches: int,
) -> tuple[Any, Any]:
    """Replay deterministic collators so a resumed update sees the next batch."""

    for _ in range(batches):
        _, mlm_iterator = _next_batch(mlm_iterator, mlm_loader)
        _, pair_iterator = _next_batch(pair_iterator, pair_loader)
    return mlm_iterator, pair_iterator


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    """Restart a finite loader without caching prior GPU batches."""

    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"DAPT artifact path escapes manifest root: {path}") from error


def _training_identity(
    config: XlmrDaptTrainingConfig,
    *,
    mixed_precision: str,
    manifest_root: Path,
) -> str:
    """Hash all bytes and controls that affect resumable optimization."""

    payload = {
        "model_id": config.model_id,
        "revision": config.revision,
        "inputs": xlmr_dapt_input_provenance(
            config,
            manifest_root=manifest_root,
        ),
        "training": {
            "max_length": config.max_length,
            "mlm_probability": config.mlm_probability,
            "mlm_batch_size": config.mlm_batch_size,
            "contrastive_batch_size": config.contrastive_batch_size,
            "contrastive_weight": config.contrastive_weight,
            "contrastive_temperature": config.contrastive_temperature,
            "max_steps": config.max_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
            "mixed_precision": mixed_precision,
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _clean_source_control() -> dict[str, Any]:
    """Fail before GPU allocation when code provenance is not immutable."""

    source_control = collect_git_metadata()
    commit = source_control.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("DAPT training requires a Git commit")
    if source_control.get("git_dirty") is not False:
        raise RuntimeError("DAPT training requires a clean Git worktree")
    return source_control


def _load_dependencies() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as functional  # type: ignore[import-not-found]
        from accelerate import Accelerator  # type: ignore[import-not-found]
        from datasets import concatenate_datasets, load_dataset  # type: ignore[import-not-found]
        from torch.utils.data import (  # type: ignore[import-not-found]
            DataLoader,
            WeightedRandomSampler,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            get_linear_schedule_with_warmup,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "XLM-R DAPT requires the optional ML dependencies; "
            "install with `uv sync --extra ml`."
        ) from error
    return {
        "torch": torch,
        "functional": functional,
        "Accelerator": Accelerator,
        "concatenate_datasets": concatenate_datasets,
        "load_dataset": load_dataset,
        "DataLoader": DataLoader,
        "WeightedRandomSampler": WeightedRandomSampler,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
        "set_seed": set_seed,
    }
