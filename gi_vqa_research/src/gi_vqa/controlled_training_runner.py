"""Restart-safe execution gate for the first controlled Study 1 training pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .controlled_training import (
    CONTROLLED_TRAINING_CONDITIONS,
    adapter_artifact_sha256,
    build_controlled_training_command,
    load_controlled_training_protocol,
    prepare_controlled_training_data,
)
from .provenance import canonical_json_sha256, file_sha256
from .training import STUDY1_SWIFT_TEMPLATE_TYPE, SWIFT_TOKEN_TYPE_CORRECTION_ID
from .training_gate import (
    TrainingGateFailure,
    _redact,
    _run_logged,
    _utc_now,
    _validate_repository,
    _validate_runtime,
    adapter_reload_probe,
    inspect_training_checkpoint,
    verify_checkpoint_resume,
)

CONTROLLED_TRAINING_RUN_SCHEMA_VERSION = "gi-vqa-controlled-training-run-v1"


class ControlledTrainingRunError(RuntimeError):
    """Raised when a controlled training run violates its locked execution gate."""


def run_controlled_training_pilot(
    *,
    repository_root: str | Path,
    expected_commit: str,
    protocol_path: str | Path,
    split_manifest_path: str | Path,
    data_dir: str | Path,
    work_dir: str | Path,
    artifact_dir: str | Path,
    required_gpu_substring: str = "T4",
) -> dict[str, Any]:
    """Prepare and train both locked arms, resuming only complete checkpoints.

    Failed or interrupted attempts are retained. A later invocation starts a new
    numbered phase attempt while reusing the latest complete checkpoint at the
    preceding locked boundary.
    """

    started = _utc_now()
    artifacts = Path(artifact_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "controlled_training_report.json"
    report: dict[str, Any] = {
        "schema_version": CONTROLLED_TRAINING_RUN_SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": started,
        "finished_at_utc": None,
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "test_partition_accessed": False,
        "checks": {},
        "conditions": {},
        "artifact_paths": {"report": str(report_path)},
    }
    try:
        repository = Path(repository_root).resolve()
        protocol_file = _resolve_under(repository, protocol_path)
        split_file = _resolve_under(repository, split_manifest_path)
        project_root = protocol_file.parents[2]
        if project_root.parent != repository:
            raise ControlledTrainingRunError(
                "protocol must be under <repository>/gi_vqa_research/protocols/study1"
            )
        data = _resolve_under(project_root, data_dir)
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)

        _validate_repository(
            report,
            repository_root=repository,
            expected_commit=expected_commit,
        )
        _validate_runtime(report, required_gpu_substring=required_gpu_substring)

        protocol = load_controlled_training_protocol(protocol_file)
        first_step = int(protocol["optimisation"]["phase_one_steps"])
        final_step = int(protocol["optimisation"]["phase_two_total_steps"])
        report["repository"]["project_root"] = str(project_root)
        report["protocol"] = {
            "path": str(protocol_file),
            "sha256": file_sha256(protocol_file),
            "pilot_id": protocol["pilot_id"],
            "phase_boundaries": [first_step, final_step],
        }
        report["split_manifest"] = {
            "path": str(split_file),
            "sha256": file_sha256(split_file),
        }
        report["training_backend"] = {
            "entrypoint": "python -m gi_vqa.training",
            "template": STUDY1_SWIFT_TEMPLATE_TYPE,
            "token_type_correction": SWIFT_TOKEN_TYPE_CORRECTION_ID,
        }

        prepared = prepare_controlled_training_data(
            protocol_path=protocol_file,
            split_manifest_path=split_file,
            project_root=project_root,
            output_dir=data,
            token=os.getenv("HF_TOKEN"),
        )
        _require(
            report,
            "train_only_controlled_data_prepared_and_verified",
            prepared.get("status") == "PASS"
            and prepared.get("test_partition_accessed") is False,
            detail=prepared,
        )
        data_manifest_path = Path(str(prepared["manifest"]))
        data_manifest = _read_json_object(data_manifest_path)
        report["prepared_data"] = prepared
        run_identity = {
            "schema_version": "gi-vqa-controlled-training-work-identity-v1",
            "repository_commit": expected_commit,
            "protocol_sha256": file_sha256(protocol_file),
            "split_manifest_sha256": file_sha256(split_file),
            "prepared_data_manifest_sha256": file_sha256(data_manifest_path),
            "ordered_item_ids_sha256": prepared["ordered_item_ids_sha256"],
        }
        identity_path = work / "run_identity.json"
        _publish_or_verify_json(identity_path, run_identity)
        report["work_identity"] = {
            **run_identity,
            "path": str(identity_path),
            "sha256": file_sha256(identity_path),
        }
        evidence_data_dir = artifacts / "prepared_data"
        for name, source in _prepared_evidence_files(
            data_manifest=data_manifest,
            project_root=project_root,
            manifest_path=data_manifest_path,
        ).items():
            copied = evidence_data_dir / name
            _publish_or_verify_file(source, copied)
            report["artifact_paths"][f"prepared_data/{name}"] = str(copied)

        for condition in CONTROLLED_TRAINING_CONDITIONS:
            condition_result = _run_condition(
                condition=condition,
                protocol=protocol,
                project_root=project_root,
                dataset_path=_resolve_under(
                    project_root,
                    data_manifest["conditions"][condition]["path"],
                ),
                work_root=work / condition,
                artifact_root=artifacts,
                first_step=first_step,
                final_step=final_step,
            )
            report["conditions"][condition] = condition_result
            _write_json_atomic(report_path, report)

        paired = report["conditions"]["paired_image"]
        constant = report["conditions"]["constant_image"]
        _require(
            report,
            "both_conditions_used_identical_ordered_training_items",
            paired["dataset_item_ids_sha256"]
            == constant["dataset_item_ids_sha256"]
            == prepared["ordered_item_ids_sha256"],
            detail={
                "paired_image": paired["dataset_item_ids_sha256"],
                "constant_image": constant["dataset_item_ids_sha256"],
                "prepared_data": prepared["ordered_item_ids_sha256"],
            },
        )
        _require(
            report,
            "both_final_adapters_have_distinct_content",
            paired["final_adapter_sha256"] != constant["final_adapter_sha256"],
            detail={
                "paired_image": paired["final_adapter_sha256"],
                "constant_image": constant["final_adapter_sha256"],
            },
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


def _run_condition(
    *,
    condition: str,
    protocol: Mapping[str, Any],
    project_root: Path,
    dataset_path: Path,
    work_root: Path,
    artifact_root: Path,
    first_step: int,
    final_step: int,
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "condition": condition,
        "dataset": {
            "path": str(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "sha256": file_sha256(dataset_path),
        },
        "dataset_item_ids_sha256": _ordered_item_ids_sha256(dataset_path),
        "phase_1": {},
        "phase_2": {},
    }

    checkpoint_one = _latest_complete_checkpoint(
        work_root / "phase_1",
        expected_step=first_step,
    )
    if checkpoint_one is None:
        attempt = _next_attempt(work_root / "phase_1")
        output = attempt / "training_output"
        command = build_controlled_training_command(
            protocol=protocol,
            condition=condition,
            dataset_path=dataset_path,
            output_dir=output,
            max_steps=first_step,
        )
        log = artifact_root / "logs" / f"{condition}_phase_1_{attempt.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        process = _run_logged(command, cwd=project_root, log_path=log)
        if process["returncode"] != 0:
            raise ControlledTrainingRunError(
                f"{condition} phase 1 failed; see {log}"
            )
        checkpoint_one = output / f"checkpoint-{first_step}"
        phase_one = inspect_training_checkpoint(checkpoint_one, expected_step=first_step)
        result["phase_1"] = {
            "reused": False,
            "attempt": attempt.name,
            "command": command,
            "process": process,
            "checkpoint": phase_one,
        }
    else:
        phase_one = inspect_training_checkpoint(checkpoint_one, expected_step=first_step)
        result["phase_1"] = {
            "reused": True,
            "attempt": checkpoint_one.parents[1].name,
            "checkpoint": phase_one,
        }

    checkpoint_two = _latest_complete_checkpoint(
        work_root / "phase_2",
        expected_step=final_step,
    )
    if checkpoint_two is None:
        attempt = _next_attempt(work_root / "phase_2")
        output = attempt / "training_output"
        command = build_controlled_training_command(
            protocol=protocol,
            condition=condition,
            dataset_path=dataset_path,
            output_dir=output,
            max_steps=final_step,
            resume_from_checkpoint=checkpoint_one,
        )
        log = artifact_root / "logs" / f"{condition}_phase_2_{attempt.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        process = _run_logged(command, cwd=project_root, log_path=log)
        if process["returncode"] != 0:
            raise ControlledTrainingRunError(
                f"{condition} phase 2 failed; see {log}"
            )
        checkpoint_two = output / f"checkpoint-{final_step}"
        phase_two = inspect_training_checkpoint(checkpoint_two, expected_step=final_step)
        result["phase_2"] = {
            "reused": False,
            "attempt": attempt.name,
            "command": command,
            "process": process,
            "checkpoint": phase_two,
        }
    else:
        phase_two = inspect_training_checkpoint(checkpoint_two, expected_step=final_step)
        result["phase_2"] = {
            "reused": True,
            "attempt": checkpoint_two.parents[1].name,
            "checkpoint": phase_two,
        }

    result["resume_verification"] = verify_checkpoint_resume(
        phase_one,
        phase_two,
        expected_first_step=first_step,
        expected_second_step=final_step,
    )
    reload_log = artifact_root / "logs" / f"{condition}_adapter_reload.log"
    reload_log.parent.mkdir(parents=True, exist_ok=True)
    reload_probe = adapter_reload_probe(
        config=_reload_config(protocol),
        checkpoint=checkpoint_two,
        subset_path=dataset_path,
        project_root=project_root,
        log_path=reload_log,
    )
    if not reload_probe.get("finite_loss"):
        raise ControlledTrainingRunError(
            f"{condition} final adapter failed its independent reload probe"
        )
    result["adapter_reload"] = reload_probe

    published = artifact_root / "adapters" / condition
    result["final_adapter"] = _publish_adapter(checkpoint_two, published)
    result["final_adapter_sha256"] = result["final_adapter"]["artifact_sha256"]
    return result


def _latest_complete_checkpoint(root: Path, *, expected_step: int) -> Path | None:
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob(f"attempt-*/training_output/checkpoint-{expected_step}"),
        reverse=True,
    )
    for candidate in candidates:
        try:
            inspect_training_checkpoint(candidate, expected_step=expected_step)
        except (TrainingGateFailure, FileNotFoundError, json.JSONDecodeError):
            continue
        return candidate
    return None


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    observed = []
    for child in root.glob("attempt-*"):
        try:
            observed.append(int(child.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = root / f"attempt-{max(observed, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _reload_config(protocol: Mapping[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    return {
        "seed": protocol["optimisation"]["seed"],
        "model": {
            "base_model": model["base_model"],
            "base_model_revision": model["base_model_revision"],
            "backend": "transformers-paligemma",
            "device": "cuda",
            "precision": "float16",
            "quantization": model["quantization"],
            "attn_implementation": "eager",
            "processor_use_fast": False,
            "trust_remote_code": False,
            "prompt_template": "paligemma",
        },
        "generation": {
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": None,
            "num_beams": 1,
            "return_token_logprobs": True,
        },
    }


def _publish_adapter(source: Path, destination: Path) -> dict[str, Any]:
    source_digest = adapter_artifact_sha256(source)
    if destination.is_dir():
        observed = adapter_artifact_sha256(destination)
        if observed != source_digest:
            raise ControlledTrainingRunError(
                f"published adapter differs from final checkpoint: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            _copy_adapter_files(source, temporary)
            try:
                os.rename(temporary, destination)
            except FileExistsError as exc:
                raise ControlledTrainingRunError(
                    f"refusing to overwrite adapter: {destination}"
                ) from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return {
        "path": str(destination),
        "artifact_sha256": source_digest,
        "files": files,
    }


def _copy_adapter_files(source: Path, destination: Path) -> None:
    names = ["adapter_config.json"]
    if (source / "adapter_model.safetensors").is_file():
        names.append("adapter_model.safetensors")
    elif (source / "adapter_model.bin").is_file():
        names.append("adapter_model.bin")
    else:
        raise ControlledTrainingRunError(f"adapter weights are missing: {source}")
    for name in names:
        shutil.copy2(source / name, destination / name)


def _prepared_evidence_files(
    *,
    data_manifest: Mapping[str, Any],
    project_root: Path,
    manifest_path: Path,
) -> dict[str, Path]:
    return {
        "controlled_training_data_manifest.json": manifest_path,
        "selection.json": _resolve_under(project_root, data_manifest["selection"]["path"]),
        "paired_image_train.jsonl": _resolve_under(
            project_root,
            data_manifest["conditions"]["paired_image"]["path"],
        ),
        "constant_image_train.jsonl": _resolve_under(
            project_root,
            data_manifest["conditions"]["constant_image"]["path"],
        ),
        "constant_image.png": _resolve_under(
            project_root,
            data_manifest["constant_image"]["path"],
        ),
    }


def _publish_or_verify_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if file_sha256(destination) != file_sha256(source):
            raise ControlledTrainingRunError(
                f"existing evidence artifact differs: {destination}"
            )
        return
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copy2(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ControlledTrainingRunError(
                f"refusing to overwrite evidence artifact: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _publish_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ControlledTrainingRunError(
                f"work directory belongs to a different controlled run: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ControlledTrainingRunError(
                f"refusing to overwrite work identity: {path}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ordered_item_ids_sha256(path: Path) -> str:
    item_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            item_id = value.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise ControlledTrainingRunError(f"dataset has invalid item_id: {path}")
            item_ids.append(item_id)
    return canonical_json_sha256(item_ids)


def _require(
    report: dict[str, Any],
    name: str,
    passed: bool,
    *,
    detail: Any,
) -> None:
    report["checks"][name] = {"passed": bool(passed), "detail": detail}
    if not passed:
        raise ControlledTrainingRunError(
            f"controlled training check failed: {name}: {detail}"
        )


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ControlledTrainingRunError(f"path escapes the required root: {path}") from exc
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ControlledTrainingRunError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked paired-image versus constant-image QLoRA pilot"
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/controlled_training_pilot"),
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--required-gpu-substring", default="T4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_controlled_training_pilot(
        repository_root=args.repository_root,
        expected_commit=args.expected_commit.lower(),
        protocol_path=args.protocol,
        split_manifest_path=args.split_manifest,
        data_dir=args.data_dir,
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


__all__ = [
    "CONTROLLED_TRAINING_RUN_SCHEMA_VERSION",
    "ControlledTrainingRunError",
    "run_controlled_training_pilot",
]
