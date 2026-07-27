"""Restart-safe inference foundation for the locked larger development gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import PaliGemmaBackend, VisionLanguageBackend
from .config import load_config, validate_config
from .controlled_evaluation_runner import (
    _condition_config,
    _validate_training_bundle,
)
from .identifiers import canonical_text, question_text, source_image_id
from .jsonl import iter_jsonl
from .larger_development import CONDITIONS, validate_locked_protocol
from .provenance import canonical_json_sha256, file_sha256
from .training_gate import _validate_runtime

RUN_SCHEMA_VERSION = "gi-vqa-larger-development-inference-v1"


class LargerDevelopmentRunError(RuntimeError):
    """Raised when locked larger-development inference cannot proceed safely."""


BackendFactory = Callable[[Mapping[str, Any]], VisionLanguageBackend]
ImageFetcher = Callable[[str, str, str, Any], Any]


def run_larger_development_inference(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    training_bundle_dir: str | Path,
    run_dir: str | Path,
    config_path: str | Path = "configs/study1/smoke.yaml",
    expected_commit: str | None = None,
    require_clean_git: bool = False,
    required_gpu_substring: str = "T4",
    backend_factory: BackendFactory | None = None,
    fetch_image: ImageFetcher | None = None,
    max_new_items: int | None = None,
) -> dict[str, Any]:
    """Run or resume the five locked conditions without computing final metrics."""

    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    bundle = _under(root, training_bundle_dir)
    output = Path(run_dir).resolve()
    lock = validate_locked_protocol(project_root=root, protocol_path=protocol_file)
    protocol = _object(protocol_file)
    receipt_path = _under(
        root, protocol["inputs"]["controlled_training_pass"]["path"]
    )
    receipt = _object(receipt_path)
    _validate_training_bundle(bundle, receipt)
    git = _git(root)
    if expected_commit is not None and git["commit"] != expected_commit:
        raise LargerDevelopmentRunError("repository commit differs from expected")
    if require_clean_git and git["dirty"]:
        raise LargerDevelopmentRunError("larger development requires a clean checkout")
    if max_new_items is not None and max_new_items < 1:
        raise ValueError("max_new_items must be positive")

    runtime: dict[str, Any] = {"checks": {}}
    if backend_factory is None:
        _validate_runtime(runtime, required_gpu_substring=required_gpu_substring)
    else:
        runtime["runtime"] = {"test_backend": True}

    selection = _object(
        _under(root, protocol["inputs"]["selection_manifest"]["path"])
    )
    records = _selected_records(root, protocol, selection)
    images = _materialize_images(
        root=root,
        protocol=protocol,
        selection=selection,
        fetch_image=fetch_image,
    )
    constant_image = bundle / "prepared_data/constant_image.png"
    if not constant_image.is_file():
        raise LargerDevelopmentRunError("training neutral image is missing")
    bundle_manifest = _object(bundle / "bundle_manifest.json")
    expected_constant_hash = bundle_manifest.get("members_sha256", {}).get(
        "prepared_data/constant_image.png"
    )
    if (
        not isinstance(expected_constant_hash, str)
        or file_sha256(constant_image) != expected_constant_hash
    ):
        raise LargerDevelopmentRunError("training neutral image hash changed")

    base_config = validate_config(
        load_config(_under(root, config_path)),
        require_resolved=True,
        require_model_execution=True,
    )
    factory = backend_factory or (lambda value: PaliGemmaBackend.from_config(value))
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_sha256": file_sha256(protocol_file),
        "selection_sha256": lock["selection_sha256"],
        "training_receipt_sha256": file_sha256(receipt_path),
        "repository_commit": git["commit"],
        "ordered_item_ids_sha256": selection["selection"][
            "ordered_item_ids_sha256"
        ],
    }
    _publish_or_match(output / "identity.json", identity)

    condition_reports: dict[str, Any] = {}
    remaining_budget = max_new_items
    for condition in CONDITIONS:
        config = _config_for_condition(
            base_config, condition=condition, bundle=bundle, receipt=receipt
        )
        backend = factory(config)
        try:
            report = _run_condition(
                condition=condition,
                records=records,
                selection=selection,
                images=images,
                constant_image=constant_image,
                output=output / condition,
                backend=backend,
                max_new_items=remaining_budget,
            )
        finally:
            backend.close()
        condition_reports[condition] = report
        if remaining_budget is not None:
            remaining_budget -= report["new_items"]
            remaining_budget = max(0, remaining_budget)

    completed = sum(report["completed_items"] for report in condition_reports.values())
    expected = len(records) * len(CONDITIONS)
    status = "INFERENCE_COMPLETE" if completed == expected else "INCOMPLETE"
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": status,
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "test_partition_accessed": False,
        "expected_item_conditions": expected,
        "completed_item_conditions": completed,
        "repository": git,
        "runtime": runtime["runtime"],
        "identity": identity,
        "conditions": condition_reports,
        "analysis_complete": False,
        "promotion_decision": None,
    }
    _write_replace(output / "inference_status.json", summary)
    return summary


def _run_condition(
    *,
    condition: str,
    records: list[dict[str, Any]],
    selection: Mapping[str, Any],
    images: Mapping[str, Path],
    constant_image: Path,
    output: Path,
    backend: VisionLanguageBackend,
    max_new_items: int | None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    selection_rows = {
        row["item_id"]: row for row in selection["records"]
    }
    new_items = reused_items = 0
    for rank, record in enumerate(records):
        item_id = str(record["item_id"])
        completion = output / "items" / f"{rank:03d}-{item_id}.json"
        row = selection_rows[item_id]
        image = _condition_image(
            condition,
            source_id=row["source_img_id"],
            shuffled_source_id=row["shuffled_source_img_id"],
            images=images,
            constant_image=constant_image,
        )
        expected = {
            "schema_version": "gi-vqa-larger-development-item-v1",
            "condition": condition,
            "rank": rank,
            "item_id": item_id,
            "record_sha256": row["record_sha256"],
            "input_source_img_id": (
                row["shuffled_source_img_id"]
                if condition == "paired_shuffled"
                else row["source_img_id"]
            ),
            "image_sha256": file_sha256(image),
            "backend": backend.provenance.as_dict(),
        }
        if completion.is_file():
            saved = _object(completion)
            if any(saved.get(key) != value for key, value in expected.items()):
                raise LargerDevelopmentRunError(f"stale completion: {completion}")
            reused_items += 1
            continue
        if max_new_items is not None and new_items >= max_new_items:
            continue
        prepared = backend.prepare(
            item_id=item_id,
            image=image,
            question=question_text(record),
        )
        generated = backend.generate(prepared)
        reference = _answer(record)
        result = {
            **expected,
            "source_img_id": row["source_img_id"],
            "complexity": record.get("metadata", {}).get("complexity"),
            "question_class": record.get("metadata", {}).get(
                "question_class", []
            ),
            "reference_answer": reference,
            "prediction": generated.text,
            "normalized_exact_match": (
                canonical_text(generated.text, casefold=True)
                == canonical_text(reference, casefold=True)
            ),
            "generated_token_count": len(generated.token_ids),
            "mean_generated_token_logprob": generated.mean_token_logprob,
            "sequence_confidence": generated.sequence_confidence,
        }
        _publish_or_match(completion, result)
        new_items += 1
    completed = len(list((output / "items").glob("*.json"))) if (output / "items").exists() else 0
    return {
        "status": "COMPLETE" if completed == len(records) else "INCOMPLETE",
        "expected_items": len(records),
        "completed_items": completed,
        "new_items": new_items,
        "reused_items": reused_items,
        "backend": backend.provenance.as_dict(),
    }


def _condition_image(
    condition: str,
    *,
    source_id: str,
    shuffled_source_id: str,
    images: Mapping[str, Path],
    constant_image: Path,
) -> Path:
    if condition in {"constant_control", "paired_neutral_ablation"}:
        return constant_image
    if condition == "paired_shuffled":
        return images[shuffled_source_id]
    return images[source_id]


def _config_for_condition(base, *, condition, bundle, receipt):
    if condition == "base_correct_descriptive":
        return _condition_config(
            base,
            condition="unadapted_paired_image",
            bundle=bundle,
            receipt=receipt,
        )
    adapter = (
        "constant_image_adapter"
        if condition == "constant_control"
        else "paired_image_adapter"
    )
    return _condition_config(base, condition=adapter, bundle=bundle, receipt=receipt)


def _selected_records(root, protocol, selection):
    split = _object(_under(root, protocol["inputs"]["grouped_split_manifest"]["path"]))
    descriptor = split["artifacts"]["development"]
    path = _under(root, descriptor["path"])
    if file_sha256(path) != descriptor["sha256"]:
        raise LargerDevelopmentRunError("development artifact differs from lock")
    wanted = {row["item_id"]: row for row in selection["records"]}
    found = {}
    for record in iter_jsonl(path):
        item_id = str(record.get("item_id"))
        if item_id in wanted:
            if record.get("metadata", {}).get("partition") != "development":
                raise LargerDevelopmentRunError("selected record is not development")
            if canonical_json_sha256(record) != wanted[item_id]["record_sha256"]:
                raise LargerDevelopmentRunError("selected record hash changed")
            found[item_id] = dict(record)
    ordered = [found.get(row["item_id"]) for row in selection["records"]]
    if any(record is None for record in ordered):
        raise LargerDevelopmentRunError("development selection is incomplete")
    return ordered


def _materialize_images(*, root, protocol, selection, fetch_image):
    split = _object(_under(root, protocol["inputs"]["grouped_split_manifest"]["path"]))
    dataset = split["dataset"]
    source_ids = {
        value
        for row in selection["records"]
        for value in (row["source_img_id"], row["shuffled_source_img_id"])
    }
    cache = root / "data/larger_development_images"
    cache.mkdir(parents=True, exist_ok=True)
    fetcher = fetch_image or _hf_fetch
    result = {}
    for source_id in sorted(source_ids):
        destination = cache / f"{source_id}.jpg"
        if not destination.is_file():
            source = Path(
                fetcher(
                    dataset["id"],
                    dataset["revision"],
                    f"images/{source_id}.jpg",
                    os.getenv("HF_TOKEN"),
                )
            )
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        result[source_id] = destination
    return result


def _hf_fetch(dataset_id, revision, filename, token):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        token=token,
    )


def _answer(record):
    answers = [
        message.get("content")
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    ]
    if len(answers) != 1 or not isinstance(answers[0], str):
        raise LargerDevelopmentRunError("record must contain one reference answer")
    return answers[0]


def _git(root):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _under(root, value):
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LargerDevelopmentRunError(f"path escapes project root: {value}") from exc
    return resolved


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LargerDevelopmentRunError(f"expected JSON object: {path}")
    return value


def _publish_or_match(path, value):
    if path.exists():
        if _object(path) != value:
            raise LargerDevelopmentRunError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_replace(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/study1/larger_development_protocol.json"),
    )
    parser.add_argument("--training-bundle", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--required-gpu-substring", default="T4")
    parser.add_argument("--max-new-items", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_larger_development_inference(
            project_root=args.project_root,
            protocol_path=args.protocol,
            training_bundle_dir=args.training_bundle,
            run_dir=args.run_dir,
            expected_commit=args.expected_commit,
            require_clean_git=args.require_clean_git,
            required_gpu_substring=args.required_gpu_substring,
            max_new_items=args.max_new_items,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "INFERENCE_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
