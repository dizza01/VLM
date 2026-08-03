"""Restart-safe frozen two-condition full-test evaluator for RQ1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from .backends import PaliGemmaBackend, VisionLanguageBackend
from .controlled_training import adapter_artifact_sha256
from .identifiers import canonical_text, question_text, source_image_id
from .jsonl import iter_jsonl
from .provenance import canonical_json_sha256, file_sha256
from .rq1_baseline import _object, _under, validate_rq1_protocol
from .rq1_training_runner import FREEZE_SCHEMA_VERSION, _fetch_image, _publish_jsonl
from .splits import _normalise_official_records
from .training_gate import _validate_runtime

SCHEMA_VERSION = "gi-vqa-rq1-full-test-inference-v1"
ITEM_SCHEMA_VERSION = "gi-vqa-rq1-full-test-item-v1"
CONDITIONS = ("fine_tuned_baseline", "unadapted_base")
BackendFactory = Callable[[str, Mapping[str, Any]], VisionLanguageBackend]
OfficialTestLoader = Callable[[str, str], list[dict[str, Any]]]
ImageFetcher = Callable[[str, str, str, Optional[str]], Union[str, Path]]


class RQ1TestError(RuntimeError):
    """Raised when the frozen RQ1 test benchmark violates its one-run gate."""


def run_rq1_full_test(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    freeze_receipt_path: str | Path,
    adapter_dir: str | Path,
    run_dir: str | Path,
    expected_commit: str,
    authorize_test: bool,
    require_clean_git: bool = True,
    required_gpu_substring: str = "L4",
    backend_factory: BackendFactory | None = None,
    official_test_loader: OfficialTestLoader | None = None,
    fetch_image: ImageFetcher | None = None,
    maximum_items: int | None = None,
) -> dict[str, Any]:
    """Run the complete test only after explicit protocol and checkpoint gates."""

    if authorize_test is not True:
        raise RQ1TestError("full test requires explicit --authorize-test")
    if maximum_items is not None:
        raise RQ1TestError("authoritative full-test item limits are forbidden")
    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    check = validate_rq1_protocol(
        project_root=root, protocol_path=protocol_file, require_locked=True
    )
    git = _git(root)
    if git["commit"] != expected_commit:
        raise RQ1TestError("repository commit differs from expected")
    if require_clean_git and git["dirty"]:
        raise RQ1TestError("full-test evaluation requires a clean checkout")
    freeze_file = Path(freeze_receipt_path).resolve()
    freeze = _object(freeze_file)
    adapter = Path(adapter_dir).resolve()
    _validate_freeze(
        freeze,
        protocol=protocol,
        protocol_file=protocol_file,
        adapter=adapter,
        expected_commit=expected_commit,
    )
    runtime: dict[str, Any] = {"checks": {}}
    if backend_factory is None:
        _validate_runtime(runtime, required_gpu_substring=required_gpu_substring)
    else:
        runtime["runtime"] = {"test_backend": True}

    # This is the first point at which official test metadata may be resolved.
    records = _materialize_official_test(
        root=root,
        protocol=protocol,
        loader=official_test_loader,
    )
    images = _materialize_test_images(
        root=root,
        protocol=protocol,
        records=records,
        fetcher=fetch_image,
    )
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": protocol["baseline_id"],
        "repository_commit": expected_commit,
        "protocol_sha256": file_sha256(protocol_file),
        "model_profile_sha256": check["model_profile_sha256"],
        "checkpoint_freeze_receipt_sha256": file_sha256(freeze_file),
        "adapter_artifact_sha256": adapter_artifact_sha256(adapter),
        "official_test_sha256": protocol["data"]["test"]["sha256"],
        "ordered_item_ids_sha256": canonical_json_sha256(
            [row["item_id"] for row in records]
        ),
        "conditions": list(CONDITIONS),
    }
    _publish_or_match(output / "identity.json", identity)
    factory = backend_factory or (
        lambda condition, config: PaliGemmaBackend.from_config(config)
    )
    condition_reports = {}
    for condition in CONDITIONS:
        config = _backend_config(
            protocol=protocol,
            root=root,
            condition=condition,
            adapter=adapter,
            freeze=freeze,
        )
        backend = factory(condition, config)
        try:
            condition_reports[condition] = _run_condition(
                condition=condition,
                records=records,
                images=images,
                output=output / condition,
                backend=backend,
            )
        finally:
            backend.close()
    completed = sum(
        report["completed_items"] for report in condition_reports.values()
    )
    expected = len(records) * len(CONDITIONS)
    status = "INFERENCE_COMPLETE" if completed == expected else "INCOMPLETE"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "test_partition_accessed": True,
        "expected_item_conditions": expected,
        "completed_item_conditions": completed,
        "repository": git,
        "runtime": runtime["runtime"],
        "identity": identity,
        "conditions": condition_reports,
        "analysis_complete": False,
    }
    _write_replace(output / "inference_status.json", report)
    return report


def _run_condition(*, condition, records, images, output, backend):
    output.mkdir(parents=True, exist_ok=True)
    new_items = reused_items = 0
    for rank, record in enumerate(records):
        item_id = str(record["item_id"])
        source_id = source_image_id(record)
        image = images[source_id]
        path = output / "items" / f"{rank:05d}-{item_id}.json"
        expected = {
            "schema_version": ITEM_SCHEMA_VERSION,
            "condition": condition,
            "rank": rank,
            "item_id": item_id,
            "record_sha256": canonical_json_sha256(record),
            "source_img_id": source_id,
            "image_sha256": file_sha256(image),
            "backend": backend.provenance.as_dict(),
        }
        if path.is_file():
            value = _object(path)
            if any(value.get(key) != expected_value for key, expected_value in expected.items()):
                raise RQ1TestError(f"stale test completion: {path}")
            reused_items += 1
            continue
        started = time.monotonic()
        prepared = backend.prepare(
            item_id=item_id,
            image=image,
            question=question_text(record),
        )
        generation = backend.generate(prepared)
        inference_seconds = time.monotonic() - started
        reference = _answer(record)
        value = {
            **expected,
            "question": question_text(record),
            "reference_answer": reference,
            "prediction": generation.text,
            "complexity": record.get("metadata", {}).get("complexity"),
            "question_class": record.get("metadata", {}).get(
                "question_class", []
            ),
            "generated_token_count": len(generation.token_ids),
            "mean_generated_token_logprob": generation.mean_token_logprob,
            "sequence_confidence": generation.sequence_confidence,
            "inference_seconds": inference_seconds,
            "normalized_exact_match": (
                canonical_text(generation.text, casefold=True)
                == canonical_text(reference, casefold=True)
            ),
        }
        _publish_or_match(path, value)
        new_items += 1
    completed = (
        len(list((output / "items").glob("*.json")))
        if (output / "items").is_dir()
        else 0
    )
    return {
        "status": "COMPLETE" if completed == len(records) else "INCOMPLETE",
        "expected_items": len(records),
        "completed_items": completed,
        "new_items": new_items,
        "reused_items": reused_items,
        "backend": backend.provenance.as_dict(),
    }


def _materialize_official_test(*, root, protocol, loader):
    descriptor = protocol["data"]["test"]
    path = _under(root, descriptor["path"])
    if path.is_file():
        if file_sha256(path) != descriptor["sha256"]:
            raise RQ1TestError("official test artifact changed")
    else:
        raw = (
            loader(protocol["data"]["dataset_id"], protocol["data"]["dataset_revision"])
            if loader is not None
            else _load_official_test(
                protocol["data"]["dataset_id"],
                protocol["data"]["dataset_revision"],
            )
        )
        split = _object(
            _under(root, protocol["data"]["grouped_split_manifest"]["path"])
        )
        normalized = _normalise_official_records(
            raw,
            official_split="test",
            dataset_revision=protocol["data"]["dataset_revision"],
            image_dataset_revision=split["image_dataset"]["revision"],
            image_dir=root / "data/images",
            project_root=root,
        )
        _publish_jsonl(path, normalized)
        if file_sha256(path) != descriptor["sha256"]:
            path.unlink(missing_ok=True)
            raise RQ1TestError("test-only reconstruction differs from lock")
    records = list(iter_jsonl(path))
    if len(records) != descriptor["expected_records"]:
        raise RQ1TestError("official test record count changed")
    if any(
        row.get("metadata", {}).get("official_split") != "test"
        for row in records
    ):
        raise RQ1TestError("non-test record entered official test benchmark")
    return records


def _materialize_test_images(*, root, protocol, records, fetcher):
    image_dir = root / "data/rq1_full_test/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    downloader = fetcher or _fetch_image
    result = {}
    for source_id in sorted({source_image_id(row) for row in records}):
        destination = image_dir / f"{source_id}.jpg"
        downloaded = Path(
            downloader(
                protocol["data"]["dataset_id"],
                protocol["data"]["dataset_revision"],
                f"images/{source_id}.jpg",
                os.getenv("HF_TOKEN"),
            )
        )
        if not downloaded.is_file():
            raise RQ1TestError(f"test image fetch returned no file: {source_id}")
        if destination.is_file():
            if file_sha256(destination) != file_sha256(downloaded):
                raise RQ1TestError(f"cached test image changed: {source_id}")
        else:
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(downloaded, temporary)
            os.replace(temporary, destination)
        result[source_id] = destination
    return result


def _load_official_test(dataset_id, revision):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RQ1TestError("install the data extra") from exc
    dataset = load_dataset(dataset_id, split="test", revision=revision)
    return [dict(row) for row in dataset]


def _validate_freeze(freeze, *, protocol, protocol_file, adapter, expected_commit):
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise RQ1TestError("unexpected checkpoint freeze schema")
    if freeze.get("status") != "PASS":
        raise RQ1TestError("checkpoint freeze did not pass")
    if freeze.get("test_evaluation_authorized") is not True:
        raise RQ1TestError("checkpoint freeze does not authorize test")
    if freeze.get("test_partition_accessed") is not False:
        raise RQ1TestError("checkpoint freeze accessed test")
    if freeze.get("repository_commit") != expected_commit:
        raise RQ1TestError("freeze receipt belongs to another commit")
    if freeze.get("protocol_sha256") != file_sha256(protocol_file):
        raise RQ1TestError("freeze receipt belongs to another protocol")
    if freeze.get("baseline_id") != protocol["baseline_id"]:
        raise RQ1TestError("freeze receipt belongs to another baseline")
    if not adapter.is_dir():
        raise RQ1TestError("frozen adapter directory is missing")
    if (
        adapter_artifact_sha256(adapter)
        != freeze.get("final_adapter", {}).get("artifact_sha256")
    ):
        raise RQ1TestError("adapter bytes differ from freeze receipt")


def _backend_config(*, protocol, root, condition, adapter, freeze):
    profile = _object(_under(root, protocol["model_profile"]["path"]))
    model = profile["model"]
    evaluation = profile["evaluation"]
    use_adapter = condition == "fine_tuned_baseline"
    return {
        "seed": protocol["training"]["seed"],
        "model": {
            "base_model": model["base_model"],
            "base_model_revision": model["base_model_revision"],
            "backend": "transformers-paligemma",
            "condition": "adapter" if use_adapter else "base",
            "adapter": str(adapter) if use_adapter else None,
            "adapter_revision": (
                freeze["repository_commit"] if use_adapter else None
            ),
            "device": "cuda",
            "precision": evaluation["precision"],
            "quantization": "none",
            "attn_implementation": evaluation["attention_implementation"],
            "trust_remote_code": False,
            "processor_use_fast": model["processor_use_fast"],
            "prompt_template": model["prompt_template"],
        },
        "generation": {
            "max_new_tokens": evaluation["max_new_tokens"],
            "do_sample": evaluation["do_sample"],
            "temperature": None,
            "num_beams": evaluation["num_beams"],
            "batch_size": evaluation["batch_size"],
            "return_token_logprobs": True,
        },
        "target_scoring": {
            "target_source": "saved_prediction",
            "reduction": "mean_logprob",
            "include_eos": False,
            "batch_size": 1,
            "verify_generation_score": True,
            "absolute_tolerance": 0.02,
        },
    }


def _answer(record):
    values = [
        message.get("content")
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise RQ1TestError("test record must contain one reference answer")
    return values[0]


def _git(root):
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


def _publish_or_match(path, value):
    if path.exists():
        if _object(path) != value:
            raise RQ1TestError(f"existing artifact differs: {path}")
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
    os.replace(temporary, path)


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
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--authorize-test", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--required-gpu-substring", default="L4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_rq1_full_test(
            project_root=args.project_root,
            protocol_path=args.protocol,
            freeze_receipt_path=args.freeze_receipt,
            adapter_dir=args.adapter_dir,
            run_dir=args.run_dir,
            expected_commit=args.expected_commit,
            authorize_test=args.authorize_test,
            require_clean_git=not args.allow_dirty,
            required_gpu_substring=args.required_gpu_substring,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "INFERENCE_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
