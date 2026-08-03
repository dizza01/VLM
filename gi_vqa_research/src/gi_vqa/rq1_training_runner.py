"""Restart-safe full-training runner and checkpoint freeze gate for RQ1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from .controlled_training import adapter_artifact_sha256
from .controlled_training_runner import _publish_adapter
from .identifiers import source_image_id
from .jsonl import iter_jsonl
from .provenance import canonical_json_sha256, file_sha256
from .rq1_baseline import (
    RQ1BaselineError,
    _object,
    _training_command,
    _under,
    validate_rq1_protocol,
)
from .splits import _normalise_official_records
from .training_gate import (
    TrainingGateFailure,
    _redact,
    _utc_now,
    _validate_repository,
    _validate_runtime,
    adapter_reload_probe,
    inspect_training_checkpoint,
)

RUN_SCHEMA_VERSION = "gi-vqa-rq1-training-run-v1"
DATA_SCHEMA_VERSION = "gi-vqa-rq1-training-data-v1"
FREEZE_SCHEMA_VERSION = "gi-vqa-rq1-checkpoint-freeze-v1"
OfficialTrainLoader = Callable[[str, str], list[dict[str, Any]]]
ImageFetcher = Callable[[str, str, str, Optional[str]], Union[str, Path]]


class RQ1TrainingError(RuntimeError):
    """Raised when full RQ1 training violates its protocol or resume identity."""


def prepare_rq1_training_data(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    maximum_records: int | None = None,
    official_train_loader: OfficialTrainLoader | None = None,
    fetch_image: ImageFetcher | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Materialize official train only; this function cannot resolve test."""

    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    check = validate_rq1_protocol(
        project_root=root, protocol_path=protocol_file, require_locked=False
    )
    output = _under(root, output_dir)
    if maximum_records is not None and (
        isinstance(maximum_records, bool) or not 1 <= maximum_records <= 32
    ):
        raise RQ1TrainingError("smoke maximum_records must be between 1 and 32")
    official_path = _under(root, protocol["data"]["training"]["path"])
    if official_path.is_file():
        if file_sha256(official_path) != protocol["data"]["training"]["sha256"]:
            raise RQ1TrainingError("existing official training artifact changed")
    else:
        records = (
            official_train_loader(
                protocol["data"]["dataset_id"],
                protocol["data"]["dataset_revision"],
            )
            if official_train_loader is not None
            else _load_official_train(
                protocol["data"]["dataset_id"],
                protocol["data"]["dataset_revision"],
            )
        )
        split_manifest = _object(
            _under(root, protocol["data"]["grouped_split_manifest"]["path"])
        )
        normalized = _normalise_official_records(
            records,
            official_split="train",
            dataset_revision=protocol["data"]["dataset_revision"],
            image_dataset_revision=split_manifest["image_dataset"]["revision"],
            image_dir=root / "data/images",
            project_root=root,
        )
        _publish_jsonl(official_path, normalized)
        if file_sha256(official_path) != protocol["data"]["training"]["sha256"]:
            official_path.unlink(missing_ok=True)
            raise RQ1TrainingError(
                "train-only reconstruction differs from locked official train"
            )

    all_rows = list(iter_jsonl(official_path))
    if len(all_rows) != check["training_records"]:
        raise RQ1TrainingError("official training record count changed")
    selected = all_rows[:maximum_records] if maximum_records else all_rows
    mode = "smoke" if maximum_records else "full"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{mode}_training_data_manifest.json"
    prepared_path = output / f"{mode}_training.jsonl"
    if manifest_path.is_file():
        return _verify_prepared_data(
            manifest_path=manifest_path,
            prepared_path=prepared_path,
            protocol_file=protocol_file,
            expected_records=len(selected),
            expected_mode=mode,
        )

    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_ids = sorted({source_image_id(row) for row in selected})
    fetcher = fetch_image or _fetch_image
    images = {}
    for source_id in source_ids:
        downloaded = Path(
            fetcher(
                protocol["data"]["dataset_id"],
                protocol["data"]["dataset_revision"],
                f"images/{source_id}.jpg",
                token,
            )
        )
        if not downloaded.is_file():
            raise RQ1TrainingError(f"image fetch returned no file: {source_id}")
        destination = image_dir / f"{source_id}.jpg"
        if destination.is_file():
            if file_sha256(destination) != file_sha256(downloaded):
                raise RQ1TrainingError(f"cached training image changed: {source_id}")
        else:
            _copy_atomic(downloaded, destination)
        images[source_id] = {
            "path": str(destination.relative_to(root)),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        }
    prepared = []
    for source in selected:
        row = json.loads(json.dumps(source))
        source_id = source_image_id(row)
        row["images"] = [images[source_id]["path"]]
        prepared.append(row)
    _publish_jsonl(prepared_path, prepared)
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "status": "PASS",
        "mode": mode,
        "protocol_sha256": file_sha256(protocol_file),
        "official_training_sha256": file_sha256(official_path),
        "records": len(prepared),
        "unique_source_images": len(images),
        "ordered_item_ids_sha256": canonical_json_sha256(
            [row["item_id"] for row in prepared]
        ),
        "prepared_training": {
            "path": str(prepared_path.relative_to(root)),
            "sha256": file_sha256(prepared_path),
            "bytes": prepared_path.stat().st_size,
        },
        "images": images,
        "test_partition_accessed": False,
    }
    _publish_json(manifest_path, manifest)
    return _verify_prepared_data(
        manifest_path=manifest_path,
        prepared_path=prepared_path,
        protocol_file=protocol_file,
        expected_records=len(selected),
        expected_mode=mode,
    )


def run_rq1_training(
    *,
    repository_root: str | Path,
    project_root: str | Path,
    protocol_path: str | Path,
    data_dir: str | Path,
    work_dir: str | Path,
    artifact_dir: str | Path,
    expected_commit: str,
    wandb_mode: str = "online",
    smoke: bool = False,
    required_gpu_substring: str | None = None,
) -> dict[str, Any]:
    """Run/resume smoke or authoritative training and freeze the final adapter."""

    started = _utc_now()
    repository = Path(repository_root).resolve()
    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    check = validate_rq1_protocol(
        project_root=root,
        protocol_path=protocol_file,
        require_locked=not smoke,
    )
    if wandb_mode not in protocol["monitoring"]["wandb"]["accepted_modes"]:
        raise RQ1TrainingError("unsupported W&B mode")
    expected_gpu = required_gpu_substring or ("T4" if smoke else "L4")
    artifacts = Path(artifact_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "rq1_training_report.json"
    report: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "RUNNING",
        "mode": "smoke" if smoke else "authoritative",
        "started_at_utc": started,
        "finished_at_utc": None,
        "protocol_sha256": file_sha256(protocol_file),
        "repository_commit": expected_commit,
        "test_partition_accessed": False,
        "wandb": {
            "mode": wandb_mode,
            "authoritative": False,
            "project": protocol["monitoring"]["wandb"]["project"],
            "group": protocol["monitoring"]["wandb"]["group"],
        },
    }
    try:
        gate_report: dict[str, Any] = {"checks": {}}
        _validate_repository(
            gate_report,
            repository_root=repository,
            expected_commit=expected_commit,
        )
        _validate_runtime(gate_report, required_gpu_substring=expected_gpu)
        report["runtime"] = gate_report["runtime"]
        prepared = prepare_rq1_training_data(
            project_root=root,
            protocol_path=protocol_file,
            output_dir=data_dir,
            maximum_records=32 if smoke else None,
            token=os.getenv("HF_TOKEN"),
        )
        report["prepared_data"] = prepared
        data_path = _under(root, prepared["prepared_training"]["path"])
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema_version": "gi-vqa-rq1-training-identity-v1",
            "mode": report["mode"],
            "repository_commit": expected_commit,
            "protocol_sha256": file_sha256(protocol_file),
            "model_profile_sha256": check["model_profile_sha256"],
            "prepared_data_manifest_sha256": file_sha256(
                _under(root, data_dir)
                / f"{'smoke' if smoke else 'full'}_training_data_manifest.json"
            ),
            "ordered_item_ids_sha256": prepared["ordered_item_ids_sha256"],
        }
        _publish_or_match(work / "identity.json", identity)
        final_step = 1 if smoke else int(protocol["training"]["expected_optimizer_steps"])
        checkpoint = _complete_checkpoint(work / "training_output", final_step)
        resume_proof_path = work / "resume_proof.json"
        if checkpoint is None:
            resume = _latest_complete_checkpoint(work / "training_output", final_step)
            model_profile = _object(
                _under(root, protocol["model_profile"]["path"])
            )
            log_path = artifacts / "logs/rq1_training.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            processes = []
            resume_source = None
            resume_source_report = None
            while checkpoint is None:
                if not smoke and resume is None:
                    target_step = int(protocol["training"]["save_steps"])
                else:
                    target_step = final_step
                command = _training_command(
                    protocol=protocol,
                    model_profile=model_profile,
                    dataset=str(data_path),
                    output_dir=str(work / "training_output"),
                    wandb_mode=wandb_mode,
                    maximum_steps=target_step,
                )
                if resume is not None:
                    command.extend(["--resume_from_checkpoint", str(resume)])
                    resume_source = resume
                    resume_source_report = inspect_training_checkpoint(
                        resume,
                        expected_step=int(resume.name.split("-")[-1]),
                    )
                process = _run_logged(
                    command,
                    cwd=root,
                    log_path=log_path,
                    wandb_mode=wandb_mode,
                    wandb_project=protocol["monitoring"]["wandb"]["project"],
                    wandb_group=protocol["monitoring"]["wandb"]["group"],
                )
                processes.append(process)
                if process["returncode"] != 0:
                    raise RQ1TrainingError(f"training failed; see {log_path}")
                completed = _complete_checkpoint(
                    work / "training_output", target_step
                )
                if completed is None:
                    raise RQ1TrainingError(
                        f"checkpoint-{target_step} is incomplete"
                    )
                if target_step == final_step:
                    checkpoint = completed
                else:
                    resume = completed
            report["processes"] = processes
            report["resume_from_checkpoint"] = (
                str(resume_source) if resume_source is not None else None
            )
            if not smoke:
                if resume_source is None:
                    raise RQ1TrainingError(
                        "authoritative training did not prove checkpoint resume"
                    )
                _publish_or_match(
                    resume_proof_path,
                    {
                        "schema_version": "gi-vqa-rq1-resume-proof-v1",
                        "status": "PASS",
                        "source_checkpoint": str(resume_source),
                        "source_checkpoint_step": int(
                            resume_source.name.split("-")[-1]
                        ),
                        "source_checkpoint_sha256": canonical_json_sha256(
                            resume_source_report
                        ),
                        "final_checkpoint_step": final_step,
                    },
                )
        report["checkpoint"] = inspect_training_checkpoint(
            checkpoint, expected_step=final_step
        )
        if smoke:
            report["status"] = "PASS"
            report["test_evaluation_authorized"] = False
        else:
            if not resume_proof_path.is_file():
                raise RQ1TrainingError(
                    "authoritative checkpoint resume proof is missing"
                )
            report["resume_proof"] = _object(resume_proof_path)
            reload_log = artifacts / "logs/rq1_adapter_reload.log"
            reload_probe = adapter_reload_probe(
                config=_reload_config(protocol, root),
                checkpoint=checkpoint,
                subset_path=data_path,
                project_root=root,
                log_path=reload_log,
            )
            if not reload_probe.get("finite_loss"):
                raise RQ1TrainingError("final adapter reload probe failed")
            report["adapter_reload"] = reload_probe
            published = _publish_adapter(
                checkpoint, artifacts / "adapter/fine_tuned_baseline"
            )
            report["final_adapter"] = published
            freeze = {
                "schema_version": FREEZE_SCHEMA_VERSION,
                "status": "PASS",
                "baseline_id": protocol["baseline_id"],
                "repository_commit": expected_commit,
                "protocol_sha256": file_sha256(protocol_file),
                "model_profile_sha256": check["model_profile_sha256"],
                "training_data_manifest_sha256": identity[
                    "prepared_data_manifest_sha256"
                ],
                "final_optimizer_step": final_step,
                "resume_proof_sha256": file_sha256(resume_proof_path),
                "final_adapter": published,
                "adapter_reload_finite_loss": True,
                "test_evaluation_authorized": True,
                "test_partition_accessed": False,
            }
            _publish_or_match(artifacts / "checkpoint_freeze_receipt.json", freeze)
            report["freeze_receipt_sha256"] = file_sha256(
                artifacts / "checkpoint_freeze_receipt.json"
            )
            report["test_evaluation_authorized"] = True
            report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["test_evaluation_authorized"] = False
        report["error"] = {
            "type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": _redact(str(exc)),
            "traceback": _redact(traceback.format_exc()),
        }
    finally:
        report["finished_at_utc"] = _utc_now()
        _write_replace(report_path, report)
    return report


def _load_official_train(dataset_id: str, revision: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RQ1TrainingError("install the data extra") from exc
    dataset = load_dataset(dataset_id, split="train", revision=revision)
    return [dict(row) for row in dataset]


def _fetch_image(dataset_id, revision, filename, token):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        token=token,
    )


def _verify_prepared_data(
    *, manifest_path, prepared_path, protocol_file, expected_records, expected_mode
):
    manifest = _object(manifest_path)
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RQ1TrainingError("unexpected prepared-data schema")
    if manifest.get("mode") != expected_mode:
        raise RQ1TrainingError("prepared-data mode changed")
    if manifest.get("protocol_sha256") != file_sha256(protocol_file):
        raise RQ1TrainingError("prepared data belongs to another protocol")
    if manifest.get("records") != expected_records:
        raise RQ1TrainingError("prepared training record count changed")
    if manifest.get("test_partition_accessed") is not False:
        raise RQ1TrainingError("prepared training data accessed test")
    if (
        not prepared_path.is_file()
        or file_sha256(prepared_path)
        != manifest["prepared_training"]["sha256"]
    ):
        raise RQ1TrainingError("prepared training JSONL changed")
    return manifest


def _latest_complete_checkpoint(output: Path, final_step: int) -> Path | None:
    candidates = []
    for path in output.glob("checkpoint-*") if output.is_dir() else ():
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if step <= final_step:
            candidates.append((step, path))
    for step, path in sorted(candidates, reverse=True):
        try:
            inspect_training_checkpoint(path, expected_step=step)
        except (TrainingGateFailure, FileNotFoundError, json.JSONDecodeError):
            continue
        return path
    return None


def _complete_checkpoint(output: Path, step: int) -> Path | None:
    candidate = output / f"checkpoint-{step}"
    try:
        inspect_training_checkpoint(candidate, expected_step=step)
    except (TrainingGateFailure, FileNotFoundError, json.JSONDecodeError):
        return None
    return candidate


def _reload_config(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    profile = _object(_under(root, protocol["model_profile"]["path"]))
    model = profile["model"]
    adaptation = profile["training"]
    return {
        "seed": protocol["training"]["seed"],
        "model": {
            "base_model": model["base_model"],
            "base_model_revision": model["base_model_revision"],
            "backend": "transformers-paligemma",
            "device": "cuda",
            "precision": adaptation["precision"],
            "quantization": adaptation["quantization"],
            "attn_implementation": adaptation["attention_implementation"],
            "processor_use_fast": model["processor_use_fast"],
            "trust_remote_code": False,
            "prompt_template": model["prompt_template"],
        },
        "generation": {
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": None,
            "num_beams": 1,
            "return_token_logprobs": True,
        },
    }


def _run_logged(
    command, *, cwd, log_path, wandb_mode, wandb_project, wandb_group
):
    environment = os.environ.copy()
    environment.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": wandb_mode,
            "WANDB_PROJECT": wandb_project,
            "WANDB_RUN_GROUP": wandb_group,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONHASHSEED": "42",
        }
    )
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
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
            sys.stdout.write(sanitized)
            log.write(sanitized)
            log.flush()
        returncode = process.wait()
    return {
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "command": [str(value) for value in command],
        "log": str(log_path),
    }


def _publish_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            if file_sha256(path) != file_sha256(temporary):
                raise RQ1TrainingError(f"existing JSONL differs: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _object(path) != value:
            raise RQ1TrainingError(f"existing JSON differs: {path}")
        return
    _write_replace(path, value)


def _publish_or_match(path: Path, value: Mapping[str, Any]) -> None:
    _publish_json(path, value)


def _write_replace(path: Path, value: Mapping[str, Any]) -> None:
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--project-root", type=Path, default=Path("."))
    prepare.add_argument("--protocol", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--maximum-records", type=int)
    run = sub.add_parser("run")
    run.add_argument("--repository-root", required=True, type=Path)
    run.add_argument("--project-root", type=Path, default=Path("."))
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--work-dir", required=True, type=Path)
    run.add_argument("--artifact-dir", required=True, type=Path)
    run.add_argument("--expected-commit", required=True)
    run.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--required-gpu-substring")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_rq1_training_data(
            project_root=args.project_root,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            maximum_records=args.maximum_records,
            token=os.getenv("HF_TOKEN"),
        )
    else:
        result = run_rq1_training(
            repository_root=args.repository_root,
            project_root=args.project_root,
            protocol_path=args.protocol,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            artifact_dir=args.artifact_dir,
            expected_commit=args.expected_commit,
            wandb_mode=args.wandb_mode,
            smoke=args.smoke,
            required_gpu_substring=args.required_gpu_substring,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
