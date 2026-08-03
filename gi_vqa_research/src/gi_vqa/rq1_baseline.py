"""Validate and plan the versioned RQ1 full-dataset baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256, file_sha256
from .training import STUDY1_SWIFT_TEMPLATE_TYPE

PROTOCOL_SCHEMA_VERSION = "gi-vqa-rq1-full-baseline-protocol-v1"
MODEL_PROFILE_SCHEMA_VERSION = "gi-vqa-model-profile-v1"
COMPUTE_PROFILE_SCHEMA_VERSION = "gi-vqa-compute-profile-v1"
PLAN_SCHEMA_VERSION = "gi-vqa-rq1-execution-plan-v1"
ALLOWED_STATUSES = {"DRAFT_REVIEW_REQUIRED", "LOCKED"}


class RQ1BaselineError(RuntimeError):
    """Raised when the RQ1 baseline would depart from its versioned plan."""


def validate_rq1_protocol(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    require_locked: bool = False,
) -> dict[str, Any]:
    """Validate bindings and derived values without resolving the test artifact."""

    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise RQ1BaselineError("unexpected RQ1 protocol schema")
    status = protocol.get("status")
    if status not in ALLOWED_STATUSES:
        raise RQ1BaselineError("RQ1 protocol has an unsupported status")
    if require_locked and status != "LOCKED":
        raise RQ1BaselineError("authoritative execution requires status=LOCKED")

    split_file = _bound_file(
        root, protocol["data"]["grouped_split_manifest"], "grouped split manifest"
    )
    split = _object(split_file)
    if split.get("dataset", {}).get("id") != protocol["data"]["dataset_id"]:
        raise RQ1BaselineError("dataset ID differs from grouped split lock")
    if (
        split.get("dataset", {}).get("revision")
        != protocol["data"]["dataset_revision"]
    ):
        raise RQ1BaselineError("dataset revision differs from grouped split lock")
    for name in ("training", "test"):
        descriptor = protocol["data"][name]
        artifact = split.get("artifacts", {}).get(descriptor["split"])
        if not isinstance(artifact, Mapping):
            raise RQ1BaselineError(f"missing split artifact: {name}")
        if artifact.get("path") != descriptor["path"]:
            raise RQ1BaselineError(f"{name} path differs from split lock")
        if artifact.get("sha256") != descriptor["sha256"]:
            raise RQ1BaselineError(f"{name} hash differs from split lock")
        observed_count = split.get("record_counts", {}).get(descriptor["split"])
        if observed_count != descriptor["expected_records"]:
            raise RQ1BaselineError(f"{name} record count differs from split lock")

    model_file = _bound_file(root, protocol["model_profile"], "model profile")
    model_profile = _object(model_file)
    if model_profile.get("schema_version") != MODEL_PROFILE_SCHEMA_VERSION:
        raise RQ1BaselineError("unexpected model profile schema")
    if model_profile.get("backend", {}).get("supports_full_test_runner") is not True:
        raise RQ1BaselineError("model profile has no accepted full-test backend")
    if (
        model_profile.get("model", {}).get("training_template")
        != STUDY1_SWIFT_TEMPLATE_TYPE
    ):
        raise RQ1BaselineError("model profile bypasses the audited training template")
    if not _immutable_revision(
        model_profile.get("model", {}).get("base_model_revision")
    ):
        raise RQ1BaselineError("base model revision is not immutable")
    if not _immutable_revision(
        model_profile.get("model", {}).get("processor_revision")
    ):
        raise RQ1BaselineError("processor revision is not immutable")

    compute_profiles = {}
    for name in (
        "authoritative_profile",
        "smoke_profile",
        "larger_model_candidate_profile",
    ):
        compute_file = _bound_file(root, protocol["compute"][name], name)
        compute = _object(compute_file)
        if compute.get("schema_version") != COMPUTE_PROFILE_SCHEMA_VERSION:
            raise RQ1BaselineError(f"unexpected compute profile schema: {name}")
        compute_profiles[name] = compute
    if compute_profiles["authoritative_profile"].get("platform") != "gcp_gpu_vm":
        raise RQ1BaselineError("authoritative RQ1 compute must be a GCP GPU VM")
    if compute_profiles["authoritative_profile"].get("authoritative") is not True:
        raise RQ1BaselineError("authoritative compute profile is not authoritative")
    if compute_profiles["smoke_profile"].get("authoritative") is not False:
        raise RQ1BaselineError("Colab smoke profile must be non-authoritative")

    training = protocol["training"]
    effective_batch = (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
    )
    if effective_batch != training["effective_batch_size"]:
        raise RQ1BaselineError("effective batch size is inconsistent")
    derived_steps = math.ceil(
        protocol["data"]["training"]["expected_records"]
        * int(training["full_passes_over_training_records"])
        / effective_batch
    )
    if derived_steps != training["expected_optimizer_steps"]:
        raise RQ1BaselineError("expected optimizer steps are inconsistent")
    if protocol["evaluation"]["expected_items_per_condition"] != protocol["data"][
        "test"
    ]["expected_records"]:
        raise RQ1BaselineError("evaluation item count differs from full test split")
    if protocol["governance"]["official_test_access"][
        "allowed_while_status_is_draft"
    ] is not False:
        raise RQ1BaselineError("draft protocol must not authorize test access")
    if status != "LOCKED" and protocol["evaluation"]["partition"] != "official_test":
        raise RQ1BaselineError("RQ1 target partition must remain explicit")
    implementation = protocol.get("implementation", {})
    for name in ("execution_planner", "report_visualizations"):
        descriptor = implementation.get(name)
        if not isinstance(descriptor, Mapping):
            raise RQ1BaselineError(f"missing implementation binding: {name}")
        _bound_file(root, descriptor, name)
    for name in (
        "restart_safe_training_runner",
        "full_test_evaluator",
        "benchmark_analysis",
    ):
        descriptor = implementation.get(name)
        if not isinstance(descriptor, Mapping):
            raise RQ1BaselineError(f"missing implementation gate: {name}")
        if status == "LOCKED" and descriptor.get("status") != "BOUND":
            raise RQ1BaselineError(
                f"cannot lock before implementation is bound: {name}"
            )
        if descriptor.get("status") == "BOUND":
            _bound_file(root, descriptor, name)

    return {
        "schema_version": "gi-vqa-rq1-protocol-check-v1",
        "status": "PASS",
        "protocol_status": status,
        "protocol": str(protocol_file.relative_to(root)),
        "protocol_sha256": file_sha256(protocol_file),
        "model_profile": str(model_file.relative_to(root)),
        "model_profile_sha256": file_sha256(model_file),
        "model_profile_id": model_profile["profile_id"],
        "authoritative_compute": compute_profiles["authoritative_profile"][
            "compute_profile_id"
        ],
        "smoke_compute": compute_profiles["smoke_profile"]["compute_profile_id"],
        "training_records": protocol["data"]["training"]["expected_records"],
        "test_records": protocol["data"]["test"]["expected_records"],
        "effective_batch_size": effective_batch,
        "optimizer_steps": derived_steps,
        "test_access_authorized": status == "LOCKED",
        "test_partition_accessed": False,
    }


def build_execution_plan(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    """Build commands and gates; never materialize or inspect the test split."""

    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    check = validate_rq1_protocol(
        project_root=root, protocol_path=protocol_file, require_locked=False
    )
    accepted_modes = protocol["monitoring"]["wandb"]["accepted_modes"]
    if wandb_mode not in accepted_modes:
        raise RQ1BaselineError(
            f"wandb mode must be one of {accepted_modes}"
        )
    model_profile = _object(_under(root, protocol["model_profile"]["path"]))
    command = _training_command(
        protocol=protocol,
        model_profile=model_profile,
        dataset="<prepared-official-train-jsonl>",
        output_dir="<persistent-run-dir>/training",
        wandb_mode=wandb_mode,
    )
    smoke_command = _training_command(
        protocol=protocol,
        model_profile=model_profile,
        dataset="<prepared-32-record-training-smoke-jsonl>",
        output_dir="<temporary-smoke-run-dir>/training",
        wandb_mode="offline" if wandb_mode != "disabled" else "disabled",
        maximum_steps=1,
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "PLAN_READY" if check["protocol_status"] == "LOCKED" else "DRAFT",
        "protocol_check": check,
        "decision": {
            "authoritative_platform": "GCP GPU VM",
            "colab_role": "32-record non-authoritative smoke only",
            "reason": (
                "full training, checkpoint resume, two-condition full-test "
                "inference and durable artifacts exceed a disposable notebook workflow"
            ),
        },
        "training": {
            "command": command,
            "expected_optimizer_steps": check["optimizer_steps"],
            "checkpoint_every_steps": protocol["training"]["save_steps"],
            "resume": (
                "restart the same versioned training runner from its latest "
                "complete checkpoint; never overwrite a run identity"
            ),
        },
        "smoke": {
            "maximum_records": 32,
            "maximum_steps": 1,
            "command": smoke_command,
        },
        "test_gate": {
            "authorized_now": check["protocol_status"] == "LOCKED",
            "requirements": [
                "reviewed protocol status changed to LOCKED in a clean commit",
                "training completed at the locked final optimizer step",
                "adapter reload probe passed and final adapter bytes were hashed",
                "checkpoint freeze receipt was committed",
                "evaluation and analysis implementation hashes were frozen",
                "one new GCS prefix was allocated for the full test benchmark",
            ],
            "test_materialized_by_this_plan": False,
        },
        "wandb": {
            "mode": wandb_mode,
            "project": protocol["monitoring"]["wandb"]["project"],
            "group": protocol["monitoring"]["wandb"]["group"],
            "authoritative": False,
            "note": (
                "W&B is monitoring only; local/GCS logs, checkpoints, predictions "
                "and hash receipts remain canonical"
            ),
        },
        "parameter_updates": protocol["parameter_change_policy"],
        "larger_models": {
            "mechanism": (
                "copy the model profile, assign a new profile_id and immutable "
                "model/processor revisions, select a suitable compute profile, "
                "then pass the same smoke and protocol checks"
            ),
            "current_backend_limit": (
                "the accepted full-test backend is PaliGemma; another architecture "
                "requires an explicit backend implementation and acceptance tests"
            ),
        },
        "test_partition_accessed": False,
    }


def _training_command(
    *,
    protocol: Mapping[str, Any],
    model_profile: Mapping[str, Any],
    dataset: str,
    output_dir: str,
    wandb_mode: str,
    maximum_steps: int | None = None,
) -> list[str]:
    training = protocol["training"]
    model = model_profile["model"]
    adaptation = model_profile["training"]
    steps = maximum_steps or int(training["expected_optimizer_steps"])
    report_to = "none" if wandb_mode == "disabled" else "wandb"
    return [
        "python3",
        "-m",
        "gi_vqa.training",
        "--dataset",
        dataset,
        "--model",
        model["base_model"],
        "--model_revision",
        model["base_model_revision"],
        "--model_type",
        model["architecture"],
        "--template",
        model["training_template"],
        "--use_hf",
        "true",
        "--train_type",
        adaptation["train_type"],
        "--tuner_backend",
        adaptation["tuner_backend"],
        "--torch_dtype",
        adaptation["precision"],
        "--quant_method",
        "bnb",
        "--quant_bits",
        "4",
        "--bnb_4bit_compute_dtype",
        adaptation["bnb_4bit_compute_dtype"],
        "--bnb_4bit_quant_type",
        adaptation["bnb_4bit_quant_type"],
        "--bnb_4bit_use_double_quant",
        _bool_text(adaptation["bnb_4bit_use_double_quant"]),
        "--max_length",
        str(adaptation["max_length"]),
        "--max_steps",
        str(steps),
        "--per_device_train_batch_size",
        str(training["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--lr_scheduler_type",
        training["lr_scheduler"],
        "--warmup_ratio",
        str(training["warmup_ratio"]),
        "--weight_decay",
        str(training["weight_decay"]),
        "--optim",
        training["optimizer"],
        "--max_grad_norm",
        str(training["max_grad_norm"]),
        "--lora_rank",
        str(adaptation["lora_rank"]),
        "--lora_alpha",
        str(adaptation["lora_alpha"]),
        "--lora_dropout",
        str(adaptation["lora_dropout"]),
        "--target_modules",
        adaptation["lora_target_modules"],
        "--freeze_vit",
        _bool_text(adaptation["freeze_vision_tower"]),
        "--freeze_aligner",
        _bool_text(adaptation["freeze_aligner"]),
        "--gradient_checkpointing",
        _bool_text(adaptation["gradient_checkpointing"]),
        "--attn_impl",
        adaptation["attention_implementation"],
        "--split_dataset_ratio",
        "0",
        "--dataset_shuffle",
        "false",
        "--train_dataloader_shuffle",
        "true",
        "--dataset_num_proc",
        "1",
        "--dataloader_num_workers",
        str(training["dataloader_workers"]),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(training["save_steps"]),
        "--save_total_limit",
        str(training["save_total_limit"]),
        "--save_safetensors",
        "true",
        "--save_only_model",
        "false",
        "--logging_steps",
        str(training["logging_steps"]),
        "--logging_first_step",
        "true",
        "--report_to",
        report_to,
        "--output_dir",
        output_dir,
        "--run_name",
        protocol["baseline_id"],
        "--add_version",
        "false",
        "--seed",
        str(training["seed"]),
        "--data_seed",
        str(training["data_seed"]),
    ]


def _bound_file(root: Path, descriptor: Mapping[str, Any], name: str) -> Path:
    path = _under(root, descriptor.get("path"))
    if not path.is_file() or file_sha256(path) != descriptor.get("sha256"):
        raise RQ1BaselineError(f"bound file missing or changed: {name}")
    return path


def _immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _under(root: Path, value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        raise RQ1BaselineError("path must be a string")
    path = Path(value)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise RQ1BaselineError(f"path escapes project root: {value}") from exc
    return result


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RQ1BaselineError(f"expected JSON object: {path}")
    return value


def _bool_text(value: Any) -> str:
    if not isinstance(value, bool):
        raise RQ1BaselineError("expected a boolean model parameter")
    return "true" if value else "false"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "plan"):
        child = subparsers.add_parser(name)
        child.add_argument("--project-root", type=Path, default=Path("."))
        child.add_argument(
            "--protocol",
            type=Path,
            default=Path(
                "protocols/study1/rq1_full_baseline_protocol_v1.draft.json"
            ),
        )
        if name == "check":
            child.add_argument("--require-locked", action="store_true")
        else:
            child.add_argument(
                "--wandb-mode",
                choices=("online", "offline", "disabled"),
                default="online",
            )
            child.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check":
        result = validate_rq1_protocol(
            project_root=args.project_root,
            protocol_path=args.protocol,
            require_locked=args.require_locked,
        )
    else:
        result = build_execution_plan(
            project_root=args.project_root,
            protocol_path=args.protocol,
            wandb_mode=args.wandb_mode,
        )
        if args.output is not None:
            _write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
