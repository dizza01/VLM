"""Locked three-condition evaluation of the controlled-training adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import PaliGemmaBackend, VisionLanguageBackend
from .config import load_config, validate_config
from .identifiers import canonical_text, question_text, source_image_id
from .image_cache import materialize_image_cache, verify_image_cache
from .jsonl import read_jsonl
from .provenance import canonical_json_sha256, file_fingerprint, file_sha256
from .splits import materialize_grouped_split_artifacts, verify_grouped_split_artifacts
from .training_gate import _validate_runtime

EVALUATION_SCHEMA_VERSION = "gi-vqa-controlled-evaluation-report-v1"
CONDITIONS = (
    "unadapted_paired_image",
    "paired_image_adapter",
    "constant_image_adapter",
)


class ControlledEvaluationError(RuntimeError):
    """Raised when the locked controlled-evaluation gate is violated."""


BackendFactory = Callable[[Mapping[str, Any]], VisionLanguageBackend]


def run_controlled_evaluation(
    *,
    project_root: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    training_receipt_path: str | Path,
    training_bundle_dir: str | Path,
    split_manifest_path: str | Path,
    image_cache_manifest_path: str | Path,
    run_dir: str | Path,
    expected_commit: str | None = None,
    require_clean_git: bool = False,
    required_gpu_substring: str = "T4",
    materialize_inputs: bool = True,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    """Run all three conditions on the same locked 20 development items."""

    root = Path(project_root).resolve()
    config_file = _under(root, config_path)
    protocol_file = _under(root, protocol_path)
    receipt_file = _under(root, training_receipt_path)
    bundle = _under(root, training_bundle_dir)
    split_file = _under(root, split_manifest_path)
    cache_file = _under(root, image_cache_manifest_path)
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    git = _git_state(root)
    if expected_commit is not None and git["commit"] != expected_commit:
        raise ControlledEvaluationError("repository commit differs from expected commit")
    if require_clean_git and git["dirty"]:
        raise ControlledEvaluationError("controlled evaluation requires a clean checkout")

    runtime_report: dict[str, Any] = {"checks": {}}
    _validate_runtime(
        runtime_report,
        required_gpu_substring=required_gpu_substring,
    )

    protocol = _read_object(protocol_file)
    receipt = _read_object(receipt_file)
    _validate_protocol(protocol, receipt_file)
    _validate_training_bundle(bundle, receipt)

    base_config = validate_config(
        load_config(config_file), require_resolved=True, require_model_execution=True
    )
    if base_config["execution"].get("evaluation_partition") != "development":
        raise ControlledEvaluationError("controlled evaluation is development-only")
    if base_config["execution"].get("max_items") != 20:
        raise ControlledEvaluationError("controlled evaluation requires exactly 20 items")

    if materialize_inputs:
        materialize_grouped_split_artifacts(manifest_path=split_file, project_root=root)
        materialize_image_cache(
            manifest_path=cache_file, project_root=root, token=os.getenv("HF_TOKEN")
        )
    split_check = verify_grouped_split_artifacts(manifest_path=split_file, project_root=root)
    cache_check = verify_image_cache(manifest_path=cache_file, project_root=root)
    records, images = _load_development_inputs(root, split_file, cache_file)

    factory = backend_factory or (lambda value: PaliGemmaBackend.from_config(value))
    condition_reports: dict[str, Any] = {}
    expected_item_ids = [str(record["item_id"]) for record in records]
    for condition in CONDITIONS:
        condition_config = _condition_config(
            base_config, condition=condition, bundle=bundle, receipt=receipt
        )
        backend = factory(condition_config)
        try:
            condition_reports[condition] = _run_condition(
                condition=condition,
                records=records,
                images=images,
                project_root=root,
                output_dir=output / condition,
                backend=backend,
            )
        finally:
            backend.close()

    if any(report["item_ids"] != expected_item_ids for report in condition_reports.values()):
        raise ControlledEvaluationError("conditions did not evaluate identical ordered items")
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "PASS",
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "test_partition_accessed": False,
        "repository": git,
        "runtime": runtime_report["runtime"],
        "runtime_checks": runtime_report["checks"],
        "protocol": file_fingerprint(protocol_file),
        "controlled_training_pass": file_fingerprint(receipt_file),
        "input_gates": {"split": split_check, "image_cache": cache_check},
        "conditions": condition_reports,
        "comparisons": _comparisons(condition_reports),
    }
    report_path = output / "controlled_evaluation_report.json"
    _publish_or_validate(report_path, report)
    return report


def _run_condition(*, condition, records, images, project_root, output_dir, backend):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reused = 0
    for rank, record in enumerate(records):
        item_id = str(record["item_id"])
        path = output_dir / "items" / f"{rank:03d}-{item_id}.json"
        source_id = source_image_id(record)
        descriptor = images[source_id]
        image_path = _under(project_root, descriptor["path"])
        if descriptor.get("scope") != "development_smoke":
            raise ControlledEvaluationError("evaluation attempted a non-development image")
        if file_sha256(image_path) != descriptor["sha256"]:
            raise ControlledEvaluationError(f"cached image hash changed for {source_id}")
        identity = canonical_json_sha256(record)
        if path.is_file():
            row = _read_object(path)
            if row.get("record_sha256") != identity or row.get("condition") != condition:
                raise ControlledEvaluationError(f"stale completion: {path}")
            reused += 1
        else:
            prepared = backend.prepare(
                item_id=item_id, image=image_path, question=question_text(record)
            )
            generation = backend.generate(prepared)
            reference = _answer(record)
            row = {
                "schema_version": "gi-vqa-controlled-evaluation-item-v1",
                "condition": condition,
                "item_id": item_id,
                "record_sha256": identity,
                "source_img_id": source_id,
                "reference_answer": reference,
                "prediction": generation.text,
                "normalized_exact_match": canonical_text(generation.text, casefold=True)
                == canonical_text(reference, casefold=True),
                "generated_token_count": len(generation.token_ids),
                "mean_generated_token_logprob": generation.mean_token_logprob,
                "sequence_confidence": generation.sequence_confidence,
                "backend": backend.provenance.as_dict(),
            }
            _publish_or_validate(path, row)
        rows.append(row)
    predictions = output_dir / "predictions.jsonl"
    serialized = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _publish_text_or_validate(predictions, serialized)
    count = len(rows)
    return {
        "status": "PASS",
        "items": count,
        "item_ids": [row["item_id"] for row in rows],
        "reused_items": reused,
        "new_items": count - reused,
        "backend": backend.provenance.as_dict(),
        "predictions": file_fingerprint(predictions),
        "metrics": {
            "normalized_exact_match": sum(bool(row["normalized_exact_match"]) for row in rows)
            / count,
            "mean_sequence_confidence": sum(float(row["sequence_confidence"]) for row in rows)
            / count,
            "mean_generated_token_logprob": sum(
                float(row["mean_generated_token_logprob"]) for row in rows
            )
            / count,
        },
    }


def _condition_config(base, *, condition, bundle, receipt):
    value = json.loads(json.dumps(base))
    if condition == "unadapted_paired_image":
        value["model"].update(condition="base", adapter=None, adapter_revision=None)
    else:
        adapter_name = condition.removesuffix("_adapter")
        value["model"].update(
            condition="adapter",
            adapter=str(bundle / "adapters" / adapter_name),
            adapter_revision=receipt["repository_commit"],
        )
    return validate_config(value, require_resolved=True, require_model_execution=True)


def _load_development_inputs(root, split_file, cache_file):
    split = _read_object(split_file)
    smoke = split.get("smoke", {})
    artifact = split.get("artifacts", {}).get("smoke_20", {})
    smoke_path = _under(root, artifact["path"])
    if file_sha256(smoke_path) != artifact.get("sha256"):
        raise ControlledEvaluationError("development JSONL differs from split lock")
    records = read_jsonl(smoke_path)
    if len(records) != 20 or any(
        record.get("metadata", {}).get("partition") != "development" for record in records
    ):
        raise ControlledEvaluationError("locked inputs are not 20 development records")
    if [record["item_id"] for record in records] != smoke.get("item_ids"):
        raise ControlledEvaluationError("development item order differs from split lock")
    cache = _read_object(cache_file)
    return records, cache["images"]


def _validate_protocol(protocol, receipt_file):
    if protocol.get("status") != "LOCKED" or protocol.get("test_partition_access") is not False:
        raise ControlledEvaluationError("evaluation protocol is not locked development-only")
    if tuple(protocol.get("conditions", ())) != CONDITIONS:
        raise ControlledEvaluationError("evaluation conditions differ from the locked order")
    if protocol.get("development_items") != 20:
        raise ControlledEvaluationError("evaluation protocol must lock 20 items")
    if file_sha256(receipt_file) != protocol.get("controlled_training_pass_sha256"):
        raise ControlledEvaluationError("controlled-training PASS receipt hash changed")


def _validate_training_bundle(bundle, receipt):
    manifest_path = bundle / "bundle_manifest.json"
    report_path = bundle / "controlled_training_report.json"
    if file_sha256(manifest_path) != receipt.get("bundle_manifest_sha256"):
        raise ControlledEvaluationError("training bundle manifest hash changed")
    if file_sha256(report_path) != receipt.get("controlled_training_report_sha256"):
        raise ControlledEvaluationError("training report hash changed")
    manifest, report = _read_object(manifest_path), _read_object(report_path)
    if manifest.get("controlled_training_status") != "PASS" or report.get("status") != "PASS":
        raise ControlledEvaluationError("controlled training did not pass")
    if report.get("test_partition_accessed") is not False:
        raise ControlledEvaluationError("controlled training accessed test data")
    members = manifest.get("members_sha256", {})
    for name, descriptor in receipt.get("adapters", {}).items():
        files = (
            ("adapter_config.json", "config_sha256"),
            ("adapter_model.safetensors", "weights_sha256"),
        )
        for filename, key in files:
            relative = f"adapters/{name}/{filename}"
            path = bundle / relative
            expected = descriptor[key]
            if members.get(relative) != expected or file_sha256(path) != expected:
                raise ControlledEvaluationError(f"training adapter hash changed: {relative}")


def _comparisons(reports):
    base = reports["unadapted_paired_image"]["metrics"]
    return {
        name: {metric: float(report["metrics"][metric]) - float(base[metric]) for metric in base}
        for name, report in reports.items()
        if name != "unadapted_paired_image"
    }


def _answer(record):
    messages = record.get("messages", [])
    answers = [m.get("content") for m in messages if m.get("role") == "assistant"]
    if len(answers) != 1 or not isinstance(answers[0], str):
        raise ControlledEvaluationError("record must contain one assistant answer")
    return answers[0]


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControlledEvaluationError(f"expected JSON object: {path}")
    return value


def _publish_or_validate(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_object(path) != value:
            raise ControlledEvaluationError(f"existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_text_or_validate(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ControlledEvaluationError(f"existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _under(root, path):
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ControlledEvaluationError(f"path escapes project root: {path}") from exc
    return resolved


def _git_state(root):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _parser():
    parser = argparse.ArgumentParser(description="Run the locked controlled evaluation gate")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/study1/smoke.yaml"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/study1/controlled_evaluation_pilot.json"),
    )
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=Path("protocols/study1/controlled_training_pass.json"),
    )
    parser.add_argument("--training-bundle", required=True, type=Path)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("protocols/study1/grouped_split_manifest.json"),
    )
    parser.add_argument(
        "--image-cache-manifest",
        type=Path,
        default=Path("protocols/study1/smoke_training_image_cache_manifest.json"),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--required-gpu-substring", default="T4")
    parser.add_argument("--no-materialize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_controlled_evaluation(
        project_root=args.project_root,
        config_path=args.config,
        protocol_path=args.protocol,
        training_receipt_path=args.training_receipt,
        training_bundle_dir=args.training_bundle,
        split_manifest_path=args.split_manifest,
        image_cache_manifest_path=args.image_cache_manifest,
        run_dir=args.run_dir,
        expected_commit=args.expected_commit,
        require_clean_git=args.require_clean_git,
        required_gpu_substring=args.required_gpu_substring,
        materialize_inputs=not args.no_materialize,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
