"""Executable two-step LoRA training and checkpoint-resume gate.

This is an infrastructure test, not a paper model. It trains on one locked
question from each of 20 training source images, saves checkpoint 1, resumes
the complete trainer state to checkpoint 2, independently reloads the final
PEFT adapter, and writes a machine-readable PASS/FAIL report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import config_sha256, load_config, validate_config
from .identifiers import question_text, source_image_id, stable_item_id
from .image_cache import materialize_image_cache, verify_image_cache
from .jsonl import iter_jsonl, write_jsonl_atomic
from .model_spec import PaliGemmaModelSpec
from .provenance import file_sha256
from .splits import (
    materialize_grouped_split_artifacts,
    verify_grouped_split_artifacts,
)
from .training import (
    EXPECTED_MS_SWIFT_VERSION,
    STUDY1_SWIFT_TEMPLATE_TYPE,
    SWIFT_TOKEN_TYPE_CORRECTION_ID,
)

TRAINING_GATE_SCHEMA_VERSION = "gi-vqa-colab-t4-training-gate-v1"
TRAINING_GATE_SELECTION_ALGORITHM = "sha256-one-item-per-training-source-v1"
EXPECTED_PACKAGES = {
    "accelerate": "1.9.0",
    "bitsandbytes": "0.47.0",
    "datasets": "3.3.2",
    "huggingface-hub": "0.34.3",
    "ms-swift": EXPECTED_MS_SWIFT_VERSION,
    "numpy": "2.0.2",
    "peft": "0.16.0",
    "Pillow": "11.3.0",
    "PyYAML": "6.0.2",
    "sentencepiece": "0.2.0",
    "transformers": "4.55.0",
    "wandb": "0.21.0",
}
_TOKEN_PATTERN = re.compile(r"\b(?:hf|api)_[A-Za-z0-9_-]{12,}\b")


class TrainingGateFailure(RuntimeError):
    """Raised when the bounded training gate cannot prove its contract."""


def build_training_gate_subset(
    *,
    train_jsonl: str | Path,
    image_cache_manifest: str | Path,
    output_jsonl: str | Path,
    project_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Select one deterministic question for each locked training cache image."""

    root = Path(project_root).resolve()
    cache_path = _resolve_under(root, image_cache_manifest)
    cache_manifest = _read_json_object(cache_path)
    selection = _required_mapping(
        cache_manifest.get("selection"),
        name="image cache selection",
    )
    required_sources = _required_string_list(
        selection.get("training_source_img_ids"),
        name="training_source_img_ids",
    )
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(_resolve_under(root, train_jsonl)):
        source_id = source_image_id(record)
        if source_id in set(required_sources):
            candidates[source_id].append(record)
    missing = sorted(set(required_sources) - set(candidates))
    if missing:
        raise TrainingGateFailure(
            f"training split is missing locked cache sources: {missing}"
        )

    selected: list[dict[str, Any]] = []
    for source_id in required_sources:
        records = candidates[source_id]

        def item_key(record: Mapping[str, Any]) -> tuple[str, str]:
            item_id = stable_item_id(record)
            payload = (
                f"{TRAINING_GATE_SELECTION_ALGORITHM}\0{seed}\0{item_id}"
            ).encode()
            return hashlib.sha256(payload).hexdigest(), item_id

        chosen = deepcopy(min(records, key=item_key))
        observed_item_id = stable_item_id(chosen)
        if chosen.get("item_id") != observed_item_id:
            raise TrainingGateFailure(
                f"training record item identity changed for {source_id}"
            )
        metadata = chosen.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("partition") != "train":
            raise TrainingGateFailure(
                f"training gate source is not labelled train: {source_id}"
            )
        images = chosen.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise TrainingGateFailure(
                f"training gate record must contain one image: {observed_item_id}"
            )
        image_path = _resolve_under(root, str(images[0]))
        cache_descriptor = _required_mapping(
            cache_manifest["images"].get(source_id),
            name=f"image cache entry {source_id}",
        )
        if image_path != _resolve_under(root, str(cache_descriptor["path"])):
            raise TrainingGateFailure(
                f"training record/cache image path mismatch for {source_id}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"training gate image is missing: {image_path}"
            )
        selected.append(chosen)

    output = Path(output_jsonl)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training subset: {output}")
    write_jsonl_atomic(output, selected, sort_keys=True)
    return {
        "algorithm": TRAINING_GATE_SELECTION_ALGORITHM,
        "seed": seed,
        "records": len(selected),
        "unique_source_images": len({source_image_id(row) for row in selected}),
        "source_img_ids": [source_image_id(row) for row in selected],
        "item_ids": [stable_item_id(row) for row in selected],
        "jsonl_path": str(output),
        "jsonl_sha256": file_sha256(output),
    }


def build_training_phase_command(
    *,
    config: Mapping[str, Any],
    dataset_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    resume_from_checkpoint: str | Path | None = None,
    python_executable: str = sys.executable,
) -> list[str]:
    """Return the fixed ms-swift command for one gate phase."""

    if max_steps not in (1, 2):
        raise TrainingGateFailure("training gate max_steps must be 1 or 2")
    if max_steps == 1 and resume_from_checkpoint is not None:
        raise TrainingGateFailure("phase 1 must not resume a checkpoint")
    if max_steps == 2 and resume_from_checkpoint is None:
        raise TrainingGateFailure("phase 2 must resume checkpoint 1")
    model = _required_mapping(config.get("model"), name="model")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TrainingGateFailure("training gate requires an integer seed")

    command = [
        python_executable,
        "-m",
        "gi_vqa.training",
        "--dataset",
        str(dataset_path),
        "--model",
        str(model["base_model"]),
        "--model_revision",
        str(model["base_model_revision"]),
        "--model_type",
        "paligemma",
        "--template",
        STUDY1_SWIFT_TEMPLATE_TYPE,
        "--use_hf",
        "true",
        "--train_type",
        "lora",
        "--tuner_backend",
        "peft",
        "--torch_dtype",
        "float16",
        "--quant_method",
        "bnb",
        "--quant_bits",
        "4",
        "--bnb_4bit_compute_dtype",
        "float16",
        "--bnb_4bit_quant_type",
        "nf4",
        "--bnb_4bit_use_double_quant",
        "true",
        "--max_length",
        "384",
        "--max_steps",
        str(max_steps),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "2e-5",
        "--lr_scheduler_type",
        "constant",
        "--warmup_ratio",
        "0",
        "--weight_decay",
        "0",
        "--optim",
        "adamw_torch",
        "--max_grad_norm",
        "1",
        "--lora_rank",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--gradient_checkpointing",
        "true",
        "--attn_impl",
        "eager",
        "--split_dataset_ratio",
        "0",
        "--dataset_num_proc",
        "1",
        "--dataloader_num_workers",
        "0",
        "--save_strategy",
        "steps",
        "--save_steps",
        "1",
        "--save_total_limit",
        "2",
        "--save_safetensors",
        "true",
        "--logging_steps",
        "1",
        "--logging_first_step",
        "true",
        "--report_to",
        "none",
        "--output_dir",
        str(output_dir),
        "--add_version",
        "false",
        "--seed",
        str(seed),
        "--data_seed",
        str(seed),
    ]
    if resume_from_checkpoint is not None:
        command.extend(
            ["--resume_from_checkpoint", str(resume_from_checkpoint)]
        )
    return command


def inspect_training_checkpoint(
    checkpoint: str | Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Require complete adapter and trainer-resume state at one checkpoint."""

    checkpoint_path = Path(checkpoint).resolve()
    if checkpoint_path.name != f"checkpoint-{expected_step}":
        raise TrainingGateFailure(
            f"unexpected checkpoint path for step {expected_step}: "
            f"{checkpoint_path}"
        )
    required_files = {
        "adapter_config": checkpoint_path / "adapter_config.json",
        "optimizer": checkpoint_path / "optimizer.pt",
        "scheduler": checkpoint_path / "scheduler.pt",
        "trainer_state": checkpoint_path / "trainer_state.json",
        "training_args": checkpoint_path / "training_args.bin",
    }
    adapter_candidates = [
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "adapter_model.bin",
    ]
    adapter_weights = next(
        (path for path in adapter_candidates if path.is_file()),
        adapter_candidates[0],
    )
    required_files["adapter_weights"] = adapter_weights
    missing = sorted(
        name for name, path in required_files.items() if not path.is_file()
    )
    if missing:
        raise TrainingGateFailure(
            f"checkpoint {expected_step} is missing resume files: {missing}"
        )
    trainer_state = _read_json_object(required_files["trainer_state"])
    if trainer_state.get("global_step") != expected_step:
        raise TrainingGateFailure(
            f"checkpoint {expected_step} trainer global_step is "
            f"{trainer_state.get('global_step')!r}"
        )
    losses = _finite_training_losses(trainer_state.get("log_history"))
    if not losses:
        raise TrainingGateFailure(
            f"checkpoint {expected_step} recorded no finite training loss"
        )
    adapter_config = _read_json_object(required_files["adapter_config"])
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise TrainingGateFailure(
            f"checkpoint {expected_step} is not a PEFT LoRA adapter"
        )
    if adapter_config.get("r") != 16 or adapter_config.get("lora_alpha") != 32:
        raise TrainingGateFailure(
            f"checkpoint {expected_step} has unexpected LoRA dimensions: "
            f"r={adapter_config.get('r')}, "
            f"alpha={adapter_config.get('lora_alpha')}"
        )
    target_modules = adapter_config.get("target_modules")
    has_target_modules = (
        isinstance(target_modules, str) and bool(target_modules.strip())
    ) or (
        isinstance(target_modules, list)
        and bool(target_modules)
        and all(
            isinstance(module, str) and bool(module.strip())
            for module in target_modules
        )
    )
    if not has_target_modules:
        raise TrainingGateFailure(
            f"checkpoint {expected_step} records invalid or empty LoRA "
            f"target modules: {target_modules!r}"
        )
    return {
        "path": str(checkpoint_path),
        "global_step": expected_step,
        "finite_training_losses": losses,
        "adapter_config": {
            "peft_type": adapter_config["peft_type"],
            "r": adapter_config["r"],
            "lora_alpha": adapter_config["lora_alpha"],
            "lora_dropout": adapter_config.get("lora_dropout"),
            "target_modules": target_modules,
            "base_model_name_or_path": adapter_config.get(
                "base_model_name_or_path"
            ),
        },
        "trainer_state": {
            "global_step": trainer_state["global_step"],
            "max_steps": trainer_state.get("max_steps"),
            "epoch": trainer_state.get("epoch"),
            "log_history_entries": len(
                trainer_state.get("log_history", [])
            ),
        },
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in sorted(required_files.items())
        },
    }


def verify_checkpoint_resume(
    phase_one: Mapping[str, Any],
    phase_two: Mapping[str, Any],
    *,
    expected_first_step: int = 1,
    expected_second_step: int = 2,
) -> dict[str, Any]:
    """Prove that phase 2 continued and changed the saved adapter."""

    if expected_first_step < 1 or expected_second_step <= expected_first_step:
        raise TrainingGateFailure("checkpoint resume step boundaries are invalid")
    if phase_one.get("global_step") != expected_first_step:
        raise TrainingGateFailure(
            f"phase 1 did not finish at global step {expected_first_step}"
        )
    if phase_two.get("global_step") != expected_second_step:
        raise TrainingGateFailure(
            f"phase 2 did not finish at global step {expected_second_step}"
        )
    first_hash = phase_one["files"]["adapter_weights"]["sha256"]
    second_hash = phase_two["files"]["adapter_weights"]["sha256"]
    if first_hash == second_hash:
        raise TrainingGateFailure(
            "adapter weights did not change after the resumed optimizer step"
        )
    second_losses = phase_two.get("finite_training_losses", [])
    if not second_losses:
        raise TrainingGateFailure("resumed phase has no finite training loss")
    return {
        "resumed_from_global_step": expected_first_step,
        "finished_global_step": expected_second_step,
        "adapter_changed_after_resume": True,
        "phase_one_adapter_sha256": first_hash,
        "phase_two_adapter_sha256": second_hash,
    }


def run_training_gate(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    expected_commit: str,
    split_manifest_path: str | Path,
    image_cache_manifest_path: str | Path,
    work_dir: str | Path,
    artifact_dir: str | Path,
    required_gpu_substring: str = "T4",
) -> dict[str, Any]:
    """Execute the complete two-step gate and always write its report."""

    started = _utc_now()
    artifacts = Path(artifact_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "training_gate_report.json"
    report: dict[str, Any] = {
        "schema_version": TRAINING_GATE_SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": started,
        "finished_at_utc": None,
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "checks": {},
        "artifact_paths": {"report": str(report_path)},
    }
    try:
        root = Path(repository_root).resolve()
        work = Path(work_dir).resolve()
        if work.exists() and any(work.iterdir()):
            raise TrainingGateFailure(
                f"training gate work directory is not empty: {work}"
            )
        work.mkdir(parents=True, exist_ok=True)
        config_file = _resolve_under(root, config_path)
        project_root = config_file.parents[2]
        if project_root.parent != root:
            raise TrainingGateFailure(
                "training config must be under "
                "<repository>/gi_vqa_research/configs/study1"
            )
        split_manifest = _resolve_under(root, split_manifest_path)
        image_manifest = _resolve_under(root, image_cache_manifest_path)
        for name, path in (
            ("split manifest", split_manifest),
            ("image cache manifest", image_manifest),
        ):
            try:
                path.relative_to(project_root)
            except ValueError as exc:
                raise TrainingGateFailure(
                    f"{name} must be inside the GI-VQA project root"
                ) from exc
        config = validate_config(
            load_config(config_file),
            require_resolved=True,
            require_model_execution=True,
        )

        _validate_repository(
            report,
            repository_root=root,
            expected_commit=expected_commit,
        )
        _validate_runtime(
            report,
            required_gpu_substring=required_gpu_substring,
        )
        report["configuration"] = {
            "path": str(config_file),
            "sha256": config_sha256(config),
            "seed": config["seed"],
            "project_root": str(project_root),
        }

        split_result = materialize_grouped_split_artifacts(
            manifest_path=split_manifest,
            project_root=project_root,
        )
        _require(
            report,
            "grouped_split_materialized_and_verified",
            split_result.get("status") == "PASS",
            detail=split_result,
        )
        cache_result = materialize_image_cache(
            manifest_path=image_manifest,
            project_root=project_root,
            token=os.getenv("HF_TOKEN"),
        )
        _require(
            report,
            "locked_image_cache_materialized_and_verified",
            cache_result.get("status") == "PASS",
            detail=cache_result,
        )
        report["inputs"] = {
            "split_manifest": {
                "path": str(split_manifest),
                "sha256": file_sha256(split_manifest),
            },
            "image_cache_manifest": {
                "path": str(image_manifest),
                "sha256": file_sha256(image_manifest),
            },
            "split_verification": verify_grouped_split_artifacts(
                manifest_path=split_manifest,
                project_root=project_root,
            ),
            "image_cache_verification": verify_image_cache(
                manifest_path=image_manifest,
                project_root=project_root,
            ),
        }

        split_payload = _read_json_object(split_manifest)
        train_path = _resolve_under(
            project_root,
            split_payload["artifacts"]["train"]["path"],
        )
        subset_path = artifacts / "training_gate_train.jsonl"
        subset = build_training_gate_subset(
            train_jsonl=train_path,
            image_cache_manifest=image_manifest,
            output_jsonl=subset_path,
            project_root=project_root,
            seed=config["seed"],
        )
        _require(
            report,
            "training_subset_has_20_unique_training_sources",
            subset["records"] == 20
            and subset["unique_source_images"] == 20,
            detail=subset,
        )
        report["training_subset"] = subset
        report["artifact_paths"]["training_subset"] = str(subset_path)

        training_output = work / "training_output"
        phase_one_command = build_training_phase_command(
            config=config,
            dataset_path=subset_path,
            output_dir=training_output,
            max_steps=1,
        )
        phase_one_log = artifacts / "phase_1_training.log"
        phase_one_process = _run_logged(
            phase_one_command,
            cwd=project_root,
            log_path=phase_one_log,
        )
        _require(
            report,
            "phase_1_process_succeeded",
            phase_one_process["returncode"] == 0,
            detail=phase_one_process,
        )
        checkpoint_one = training_output / "checkpoint-1"
        phase_one = inspect_training_checkpoint(
            checkpoint_one,
            expected_step=1,
        )
        _require(
            report,
            "phase_1_checkpoint_is_complete",
            phase_one["global_step"] == 1,
            detail=phase_one,
        )

        phase_two_command = build_training_phase_command(
            config=config,
            dataset_path=subset_path,
            output_dir=training_output,
            max_steps=2,
            resume_from_checkpoint=checkpoint_one,
        )
        phase_two_log = artifacts / "phase_2_resume.log"
        phase_two_process = _run_logged(
            phase_two_command,
            cwd=project_root,
            log_path=phase_two_log,
        )
        _require(
            report,
            "phase_2_resume_process_succeeded",
            phase_two_process["returncode"] == 0,
            detail=phase_two_process,
        )
        checkpoint_two = training_output / "checkpoint-2"
        phase_two = inspect_training_checkpoint(
            checkpoint_two,
            expected_step=2,
        )
        resume = verify_checkpoint_resume(phase_one, phase_two)
        _require(
            report,
            "checkpoint_resume_advanced_and_changed_adapter",
            resume["adapter_changed_after_resume"],
            detail=resume,
        )

        reload_log = artifacts / "adapter_reload.log"
        reload_probe = adapter_reload_probe(
            config=config,
            checkpoint=checkpoint_two,
            subset_path=subset_path,
            project_root=project_root,
            log_path=reload_log,
        )
        _require(
            report,
            "final_adapter_independently_reloads_with_finite_loss",
            reload_probe["finite_loss"],
            detail=reload_probe,
        )
        report["training"] = {
            "entrypoint": "python -m gi_vqa.training",
            "template": STUDY1_SWIFT_TEMPLATE_TYPE,
            "token_type_correction": SWIFT_TOKEN_TYPE_CORRECTION_ID,
            "phase_1": {
                "command": phase_one_command,
                "process": phase_one_process,
                "checkpoint": phase_one,
            },
            "phase_2": {
                "command": phase_two_command,
                "process": phase_two_process,
                "checkpoint": phase_two,
            },
            "resume_verification": resume,
            "adapter_reload": reload_probe,
        }
        report["artifact_paths"].update(
            {
                "phase_1_log": str(phase_one_log),
                "phase_2_log": str(phase_two_log),
                "adapter_reload_log": str(reload_log),
            }
        )
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": _redact(str(exc)),
            "traceback": _redact(traceback.format_exc()),
        }
    finally:
        report["finished_at_utc"] = _utc_now()
        _write_json_atomic(report_path, report)
    return report


def adapter_reload_probe(
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    subset_path: Path,
    project_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Independently load the final PEFT adapter and compute one finite loss."""

    started = time.monotonic()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            sys.stdout = log_handle
            sys.stderr = log_handle
            import torch
            import transformers
            from peft import PeftConfig, PeftModel
            from PIL import Image

            model = config["model"]
            spec = PaliGemmaModelSpec.from_config(config)
            adapter_config = PeftConfig.from_pretrained(str(checkpoint))
            base_model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
                str(model["base_model"]),
                revision=str(model["base_model_revision"]),
                torch_dtype=torch.float16,
                attn_implementation="eager",
                trust_remote_code=False,
                device_map="cuda:0",
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=[
                        "model.vision_tower",
                        "model.multi_modal_projector",
                        "lm_head",
                    ],
                ),
            )
            loaded = PeftModel.from_pretrained(
                base_model,
                str(checkpoint),
                is_trainable=False,
            )
            loaded.eval()
            processor = transformers.AutoProcessor.from_pretrained(
                spec.resolved_processor_id,
                revision=spec.resolved_processor_revision,
                use_fast=False,
                trust_remote_code=False,
            )
            record = next(iter_jsonl(subset_path))
            image_reference = _resolve_under(
                project_root,
                str(record["images"][0]),
            )
            with Image.open(image_reference) as opened:
                image = opened.convert("RGB")
                batch = processor(
                    images=image,
                    text=spec.format_prompt(question_text(record)),
                    suffix=_answer_text(record),
                    return_tensors="pt",
                )
            input_device = next(loaded.parameters()).device
            batch = {
                name: value.to(input_device)
                if hasattr(value, "to")
                else value
                for name, value in batch.items()
            }
            with torch.inference_mode():
                outputs = loaded(**batch, return_dict=True)
            loss = float(outputs.loss.detach().float().cpu().item())
            adapter_parameters = sum(
                parameter.numel()
                for name, parameter in loaded.named_parameters()
                if "lora_" in name
            )
            if adapter_parameters < 1:
                raise TrainingGateFailure(
                    "independent reload exposed no LoRA parameters"
                )
            if not math.isfinite(loss):
                raise TrainingGateFailure(
                    f"independent adapter reload produced non-finite loss: {loss}"
                )
            return {
                "checkpoint": str(checkpoint),
                "peft_type": str(adapter_config.peft_type),
                "adapter_parameters": adapter_parameters,
                "probe_item_id": stable_item_id(record),
                "loss": loss,
                "finite_loss": True,
                "elapsed_seconds": time.monotonic() - started,
                "peak_cuda_memory_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
            }
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                del loaded
                del base_model
            except UnboundLocalError:
                pass
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass


def _validate_repository(
    report: dict[str, Any],
    *,
    repository_root: Path,
    expected_commit: str,
) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise TrainingGateFailure(
            "expected_commit must be a lowercase 40-character Git SHA"
        )
    resolved = _git_output(repository_root, "rev-parse", "HEAD")
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    _require(
        report,
        "repository_commit_matches",
        resolved == expected_commit,
        detail={"expected": expected_commit, "observed": resolved},
    )
    _require(
        report,
        "repository_checkout_is_clean",
        not status,
        detail={"status": status.splitlines()},
    )
    report["repository"] = {
        "root": str(repository_root),
        "expected_commit": expected_commit,
        "resolved_commit": resolved,
        "clean": True,
    }


def _validate_runtime(
    report: dict[str, Any],
    *,
    required_gpu_substring: str,
) -> None:
    _require(
        report,
        "python_3_11",
        sys.version_info[:2] == (3, 11),
        detail={"python": sys.version},
    )
    try:
        import torch
    except ImportError as exc:
        raise TrainingGateFailure("PyTorch is required") from exc
    _require(
        report,
        "cuda_available",
        torch.cuda.is_available(),
        detail={"cuda_available": torch.cuda.is_available()},
    )
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    _require(
        report,
        "exactly_one_required_gpu",
        len(gpu_names) == 1 and required_gpu_substring in gpu_names[0],
        detail={
            "required_substring": required_gpu_substring,
            "gpu_names": gpu_names,
        },
    )
    observed_packages = {
        name: _installed_version(name) for name in EXPECTED_PACKAGES
    }
    mismatches = {
        name: {
            "expected": expected,
            "observed": observed_packages[name],
        }
        for name, expected in EXPECTED_PACKAGES.items()
        if observed_packages[name] != expected
    }
    _require(
        report,
        "pinned_training_packages",
        not mismatches,
        detail={"mismatches": mismatches, "observed": observed_packages},
    )
    _require(
        report,
        "pinned_torch_runtime",
        str(torch.__version__).startswith("2.6.0"),
        detail={
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    )
    report["runtime"] = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "gpu_names": gpu_names,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": observed_packages,
    }


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONHASHSEED": "42",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sanitized = _redact(line)
            print(sanitized, end="")
            log_handle.write(sanitized)
        returncode = process.wait()
        log_handle.flush()
        os.fsync(log_handle.fileno())
    result = {
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path),
    }
    if returncode != 0:
        result["log_tail"] = _tail(log_path, lines=40)
    return result


def _finite_training_losses(value: Any) -> list[dict[str, float | int]]:
    if not isinstance(value, list):
        return []
    results = []
    for entry in value:
        if not isinstance(entry, Mapping) or "loss" not in entry:
            continue
        loss = entry["loss"]
        if (
            isinstance(loss, int | float)
            and not isinstance(loss, bool)
            and math.isfinite(float(loss))
        ):
            results.append(
                {
                    "step": int(entry.get("step", -1)),
                    "loss": float(loss),
                }
            )
    return results


def _answer_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                return str(message["content"])
    raise TrainingGateFailure("training gate record has no assistant answer")


def _require(
    report: dict[str, Any],
    name: str,
    passed: bool,
    *,
    detail: Any,
) -> None:
    report["checks"][name] = {"passed": bool(passed), "detail": detail}
    if not passed:
        raise TrainingGateFailure(f"training gate check failed: {name}: {detail}")


def _git_output(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    output = process.stdout.strip()
    return (
        output.lower()
        if arguments[:2] == ("rev-parse", "HEAD")
        else output
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _read_json_object(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TrainingGateFailure(f"JSON root must be an object: {json_path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingGateFailure(f"{name} must be a mapping")
    return value


def _required_string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TrainingGateFailure(f"{name} must be a list of strings")
    if len(value) != len(set(value)):
        raise TrainingGateFailure(f"{name} contains duplicates")
    return list(value)


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainingGateFailure(
            f"path escapes the repository root: {path}"
        ) from exc
    return resolved


def _redact(value: str) -> str:
    token = os.getenv("HF_TOKEN")
    redacted = value.replace(token, "<redacted-token>") if token else value
    return _TOKEN_PATTERN.sub("<redacted-token>", redacted)


def _tail(path: Path, *, lines: int) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Study 1 Colab T4 tiny-LoRA training gate"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--image-cache-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--required-gpu-substring", default="T4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_training_gate(
        config_path=args.config,
        repository_root=args.repository_root,
        expected_commit=args.expected_commit.lower(),
        split_manifest_path=args.split_manifest,
        image_cache_manifest_path=args.image_cache_manifest,
        work_dir=args.work_dir,
        artifact_dir=args.artifact_dir,
        required_gpu_substring=args.required_gpu_substring,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
