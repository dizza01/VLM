"""One-item CUDA contract runner for the provisional Study 1 backend.

This module is intentionally callable outside a notebook. The Colab notebook
only prepares an exact checkout and authentication, then invokes this runner.
The selected item is diagnostic-only and must never be reported as a research
evaluation result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import (
    GenerationResult,
    PaliGemmaBackend,
    PreparedInput,
    TargetScore,
)
from .config import config_sha256, load_config, validate_config
from .identifiers import stable_item_id
from .model_spec import PaliGemmaModelSpec
from .provenance import file_sha256
from .training import (
    STUDY1_SWIFT_TEMPLATE_TYPE,
    TrainingCompatibilityError,
    correct_ms_swift_paligemma_training_encoding,
)

CONTRACT_SCHEMA_VERSION = "gi-vqa-colab-t4-backend-contract-v2"
EXPECTED_PYTHON = (3, 11)
EXPECTED_TORCH_PREFIX = "2.6.0"
EXPECTED_CUDA_PREFIX = "12.4"
DIAGNOSTIC_SPLIT = "train"
DIAGNOSTIC_ROW_INDEX = 143_500
DIAGNOSTIC_IMG_ID = "cl8k2u1pv1e4z08320vbv6jzb"
DIAGNOSTIC_QUESTION = (
    "Where in the image is the abnormality and what is the size of the polyp?"
)
DIAGNOSTIC_REFERENCE_ANSWER = (
    "Lesion observed in central region measuring less than 5 millimeters"
)
DIAGNOSTIC_COMPLEXITY = 2
DIAGNOSTIC_IMAGE_SHA256 = (
    "f2926b17c9caf0774fd8597ed35c7c2b22139d3c9d2965974af32ba5c9173e3f"
)
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.9.0",
    "bitsandbytes": "0.47.0",
    "datasets": "3.3.2",
    "huggingface-hub": "0.34.3",
    "ms-swift": "3.7.0",
    "numpy": "2.0.2",
    "peft": "0.16.0",
    "Pillow": "11.3.0",
    "PyYAML": "6.0.2",
    "sentencepiece": "0.2.0",
    "transformers": "4.55.0",
    "wandb": "0.21.0",
}
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOKEN_PATTERN = re.compile(r"\b(?:hf|api)_[A-Za-z0-9_-]{12,}\b")
_BEARER_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:access_token|token|signature|sig|"
    r"x-amz-signature|x-amz-credential|x-goog-signature)="
    r")([^&\s]+)"
)


class ContractFailure(RuntimeError):
    """Raised when a required runtime or numerical contract check fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _sanitise_text(value: str) -> str:
    value = _TOKEN_PATTERN.sub("<redacted-token>", value)
    value = _BEARER_PATTERN.sub(r"\1<redacted-token>", value)
    return _QUERY_SECRET_PATTERN.sub(r"\1<redacted-token>", value)


def _require(
    report: dict[str, Any],
    name: str,
    condition: bool,
    *,
    detail: Any,
) -> None:
    report.setdefault("checks", {})[name] = {
        "passed": bool(condition),
        "detail": detail,
    }
    if not condition:
        raise ContractFailure(f"contract check failed: {name}: {detail}")


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ContractFailure(
            f"command failed ({' '.join(command)}): {_sanitise_text(message)}"
        )
    return completed.stdout.strip()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _package_versions() -> dict[str, str | None]:
    names = sorted(
        {
            *EXPECTED_PACKAGE_VERSIONS,
            "huggingface-hub",
            "torch",
            "gi-vqa-research",
        }
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_runtime(
    report: dict[str, Any],
    *,
    required_gpu_substring: str,
) -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ContractFailure(
            "the contract runtime is incomplete; install the checked-out project "
            "with its 'gpu' extra before running"
        ) from exc

    versions = _package_versions()
    runtime = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    report["runtime"] = runtime

    _require(
        report,
        "python_3_11",
        sys.version_info[:2] == EXPECTED_PYTHON,
        detail={"expected": list(EXPECTED_PYTHON), "observed": list(sys.version_info[:2])},
    )
    _require(
        report,
        "torch_reference_version",
        torch.__version__.startswith(EXPECTED_TORCH_PREFIX),
        detail={"expected_prefix": EXPECTED_TORCH_PREFIX, "observed": torch.__version__},
    )
    _require(
        report,
        "torch_cuda_reference_version",
        str(torch.version.cuda or "").startswith(EXPECTED_CUDA_PREFIX),
        detail={"expected_prefix": EXPECTED_CUDA_PREFIX, "observed": torch.version.cuda},
    )
    _require(
        report,
        "cuda_available",
        bool(torch.cuda.is_available()),
        detail=bool(torch.cuda.is_available()),
    )
    _require(
        report,
        "single_cuda_device",
        torch.cuda.device_count() == 1,
        detail={"observed": int(torch.cuda.device_count())},
    )
    gpu_name = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)
    runtime["gpu"] = {
        "name": gpu_name,
        "total_memory_bytes": int(gpu_properties.total_memory),
        "capability": [
            int(gpu_properties.major),
            int(gpu_properties.minor),
        ],
    }
    _require(
        report,
        "required_gpu",
        required_gpu_substring.casefold() in gpu_name.casefold(),
        detail={"required_substring": required_gpu_substring, "observed": gpu_name},
    )

    for distribution, expected in EXPECTED_PACKAGE_VERSIONS.items():
        observed = versions.get(distribution)
        _require(
            report,
            f"package_{distribution.casefold().replace('-', '_')}",
            observed == expected,
            detail={"expected": expected, "observed": observed},
        )
    _require(
        report,
        "numpy_import_matches_distribution",
        np.__version__ == EXPECTED_PACKAGE_VERSIONS["numpy"],
        detail={
            "expected": EXPECTED_PACKAGE_VERSIONS["numpy"],
            "observed": np.__version__,
        },
    )
    return torch, np


def _verify_repository(
    report: dict[str, Any],
    *,
    repository_root: Path,
    expected_commit: str,
) -> None:
    normalized_commit = expected_commit.strip().casefold()
    if _GIT_COMMIT.fullmatch(normalized_commit) is None:
        raise ContractFailure("expected_commit must be a complete 40-character Git SHA")
    if not repository_root.is_dir():
        raise ContractFailure(f"repository root does not exist: {repository_root}")

    resolved = _run_command(["git", "rev-parse", "HEAD"], cwd=repository_root).casefold()
    status = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
    )
    report["repository"] = {
        "root": str(repository_root),
        "expected_commit": normalized_commit,
        "resolved_commit": resolved,
        "clean": not bool(status),
        "status": status.splitlines(),
    }
    _require(
        report,
        "exact_repository_commit",
        resolved == normalized_commit,
        detail={"expected": normalized_commit, "observed": resolved},
    )
    _require(
        report,
        "clean_repository",
        not bool(status),
        detail=status.splitlines(),
    )


def _validate_diagnostic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = ("img_id", "question", "answer", "complexity", "image")
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ContractFailure(
            f"pinned diagnostic row is missing required fields: {missing}"
        )
    try:
        complexity = int(row["complexity"])
    except (TypeError, ValueError) as exc:
        raise ContractFailure("pinned diagnostic complexity is not an integer") from exc

    normalized = {
        "img_id": str(row["img_id"]).strip(),
        "question": str(row["question"]).strip(),
        "answer": str(row["answer"]).strip(),
        "complexity": complexity,
        "image": str(row["image"]).strip(),
        "question_class": list(row.get("question_class") or []),
    }
    expected = {
        "img_id": DIAGNOSTIC_IMG_ID,
        "question": DIAGNOSTIC_QUESTION,
        "answer": DIAGNOSTIC_REFERENCE_ANSWER,
        "complexity": DIAGNOSTIC_COMPLEXITY,
    }
    observed = {name: normalized[name] for name in expected}
    if observed != expected:
        raise ContractFailure(
            "pinned diagnostic row content changed: "
            f"expected {expected!r}, observed {observed!r}"
        )
    if not normalized["image"]:
        raise ContractFailure("pinned diagnostic row has an empty image reference")
    return normalized


def _load_diagnostic_item(
    report: dict[str, Any],
    *,
    config: Mapping[str, Any],
    artifact_dir: Path,
) -> tuple[dict[str, Any], Path]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download, whoami
        from PIL import Image
    except ImportError as exc:
        raise ContractFailure(
            "datasets, huggingface_hub and Pillow are required"
        ) from exc

    dataset_id = str(config["data"]["dataset"])
    dataset_revision = str(config["data"]["dataset_revision"])
    image_dataset_id = str(config["data"]["image_dataset"])
    image_dataset_revision = str(config["data"]["image_dataset_revision"])
    identity = whoami()
    username = identity.get("name") or identity.get("fullname") or "authenticated-user"
    report["authentication"] = {"hugging_face_username": str(username)}

    resolved_image_dataset_revision = HfApi().dataset_info(
        repo_id=image_dataset_id,
        revision=image_dataset_revision,
        token=True,
    ).sha
    _require(
        report,
        "canonical_image_dataset_revision_resolves",
        resolved_image_dataset_revision == image_dataset_revision,
        detail={
            "expected": image_dataset_revision,
            "observed": resolved_image_dataset_revision,
        },
    )

    row_slice = (
        f"{DIAGNOSTIC_SPLIT}[{DIAGNOSTIC_ROW_INDEX}:{DIAGNOSTIC_ROW_INDEX + 1}]"
    )
    rows = load_dataset(
        dataset_id,
        split=row_slice,
        revision=dataset_revision,
        token=True,
    )
    if len(rows) != 1:
        raise ContractFailure(
            f"pinned diagnostic slice returned {len(rows)} rows instead of one"
        )
    row = _validate_diagnostic_row(rows[0])
    expected_suffix = f"/{row['img_id']}.jpg"
    _require(
        report,
        "diagnostic_image_reference_matches_id",
        row["image"].split("?", 1)[0].endswith(expected_suffix),
        detail={"img_id": row["img_id"], "image_reference": row["image"]},
    )

    image_filename = f"images/{row['img_id']}.jpg"
    cached_image = Path(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=image_filename,
            revision=dataset_revision,
            token=True,
        )
    )
    image_copy = artifact_dir / "diagnostic_image.jpg"
    image_copy.write_bytes(cached_image.read_bytes())
    observed_image_sha256 = file_sha256(image_copy)
    _require(
        report,
        "diagnostic_image_sha256",
        observed_image_sha256 == DIAGNOSTIC_IMAGE_SHA256,
        detail={
            "expected": DIAGNOSTIC_IMAGE_SHA256,
            "observed": observed_image_sha256,
        },
    )
    with Image.open(image_copy) as source_image:
        source_image.load()
        image_metadata = {
            "format": source_image.format,
            "mode": source_image.mode,
            "size": [int(value) for value in source_image.size],
        }
    expected_image_metadata = {
        "format": "JPEG",
        "mode": "RGB",
        "size": [594, 530],
    }
    _require(
        report,
        "diagnostic_image_is_expected_jpeg",
        image_metadata == expected_image_metadata,
        detail={"expected": expected_image_metadata, "observed": image_metadata},
    )

    official_test_overlap_indices: list[int] = []
    test_rows = load_dataset(
        dataset_id,
        split="test",
        revision=dataset_revision,
        streaming=True,
        token=True,
    )
    for test_index, test_row in enumerate(test_rows):
        if str(test_row.get("img_id", "")).strip() == row["img_id"]:
            official_test_overlap_indices.append(test_index)
    _require(
        report,
        "diagnostic_source_absent_from_official_test",
        not official_test_overlap_indices,
        detail={"matching_test_row_indices": official_test_overlap_indices},
    )

    item_id = stable_item_id(
        {
            "img_id": row["img_id"],
            "question": row["question"],
        }
    )
    report["diagnostic_item"] = {
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "reserved_from_future_research_splits": True,
        "selection": "fixed_versioned_contract_fixture",
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "split": DIAGNOSTIC_SPLIT,
        "row_index": DIAGNOSTIC_ROW_INDEX,
        "item_id": item_id,
        "img_id": row["img_id"],
        "question": row["question"],
        "reference_answer": row["answer"],
        "complexity": row["complexity"],
        "question_class": row["question_class"],
        "source_image_filename": image_filename,
        "source_image_sha256": observed_image_sha256,
        "source_image": image_metadata,
        "official_test_overlap_count": len(official_test_overlap_indices),
        "canonical_image_dataset": image_dataset_id,
        "canonical_image_dataset_revision": image_dataset_revision,
        "resolved_canonical_image_dataset_revision": (
            resolved_image_dataset_revision
        ),
        "image_resolution_note": (
            "This backend-only contract uses the exact JPEG committed with the "
            "pinned QA dataset. It verifies that the configured canonical image "
            "dataset revision exists, but the later data-stage smoke must compare "
            "and audit the canonical image bytes."
        ),
    }
    return row, image_copy


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in values],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_swift_template(
    report: dict[str, Any],
    torch: Any,
    *,
    spec: PaliGemmaModelSpec,
    prepared: PreparedInput,
    target_score: TargetScore,
    image_path: Path,
) -> None:
    """Require project training preprocessing to equal the direct processor."""

    try:
        from PIL import Image
        from swift.llm import (
            InferRequest,
            TemplateType,
            get_model_tokenizer,
            get_template,
        )
    except ImportError as exc:
        raise ContractFailure(
            "ms-swift and Pillow are required for template equivalence"
        ) from exc
    try:
        from .swift_paligemma_plugin import (
            get_study1_paligemma_template,
        )
    except (ImportError, TrainingCompatibilityError) as exc:
        raise ContractFailure(
            "the versioned Study 1 ms-swift PaliGemma template could not be "
            f"loaded: {exc}"
        ) from exc

    model, processor = get_model_tokenizer(
        spec.resolved_processor_id,
        torch_dtype=torch.float16,
        load_model=False,
        use_hf=True,
        revision=spec.resolved_processor_revision,
        download_model=False,
        model_type="paligemma",
    )
    _require(
        report,
        "swift_did_not_load_second_model",
        model is None,
        detail={"model_is_none": model is None},
    )
    builtin_template = get_template(
        TemplateType.paligemma,
        processor,
        max_length=512,
        template_backend="swift",
    )
    template = get_study1_paligemma_template(
        processor,
        max_length=512,
    )
    with Image.open(image_path) as source:
        rgb_image = source.convert("RGB")

    direct_prefix = processor(
        images=rgb_image,
        text=prepared.prompt,
        return_tensors="pt",
    )
    direct_prefix_ids = tuple(
        int(value) for value in direct_prefix["input_ids"][0].tolist()
    )
    _require(
        report,
        "direct_processor_matches_backend_prefix",
        direct_prefix_ids == prepared.input_token_ids,
        detail={
            "backend_sha256": _integer_sequence_sha256(
                prepared.input_token_ids
            ),
            "direct_sha256": _integer_sequence_sha256(direct_prefix_ids),
            "token_count": len(direct_prefix_ids),
        },
    )
    direct_prefix_pixel_sha256 = _tensor_sha256(
        direct_prefix["pixel_values"]
    )
    backend_pixel_sha256 = str(
        prepared.preprocessing["processed_pixel_values_sha256"]
    )
    _require(
        report,
        "direct_processor_pixels_match_backend",
        direct_prefix_pixel_sha256 == backend_pixel_sha256,
        detail={
            "backend_sha256": backend_pixel_sha256,
            "direct_sha256": direct_prefix_pixel_sha256,
        },
    )

    template.set_mode("pt")
    swift_prefix = template.encode(
        InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": f"<image>{prepared.question}",
                }
            ],
            images=[str(image_path)],
        )
    )
    swift_prefix_ids = tuple(int(value) for value in swift_prefix["input_ids"])
    _require(
        report,
        "swift_inference_template_matches_backend_prefix",
        swift_prefix_ids == prepared.input_token_ids,
        detail={
            "backend_sha256": _integer_sequence_sha256(
                prepared.input_token_ids
            ),
            "swift_sha256": _integer_sequence_sha256(swift_prefix_ids),
            "token_count": len(swift_prefix_ids),
        },
    )
    _require(
        report,
        "direct_inference_prefix_omits_token_types",
        "token_type_ids" not in direct_prefix,
        detail={"keys": sorted(direct_prefix)},
    )
    expected_prefix_token_types = (0,) * len(prepared.input_token_ids)
    swift_prefix_token_types = tuple(
        int(value) for value in swift_prefix["token_type_ids"]
    )
    _require(
        report,
        "swift_inference_token_types_are_prefix_only",
        swift_prefix_token_types == expected_prefix_token_types,
        detail={
            "expected_sha256": _integer_sequence_sha256(
                expected_prefix_token_types
            ),
            "swift_sha256": _integer_sequence_sha256(
                swift_prefix_token_types
            ),
        },
    )

    direct_target = processor(
        images=rgb_image,
        text=prepared.prompt,
        suffix=target_score.target_text,
        return_tensors="pt",
    )
    builtin_template.set_mode("train")
    builtin_swift_target = builtin_template.encode(
        InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": f"<image>{prepared.question}",
                },
                {
                    "role": "assistant",
                    "content": target_score.target_text,
                },
            ],
            images=[str(image_path)],
        )
    )
    template.set_mode("train")
    swift_target = template.encode(
        InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": f"<image>{prepared.question}",
                },
                {
                    "role": "assistant",
                    "content": target_score.target_text,
                },
            ],
            images=[str(image_path)],
        )
    )

    for field in ("input_ids", "labels"):
        direct_values = tuple(
            int(value) for value in direct_target[field][0].tolist()
        )
        builtin_values = tuple(
            int(value) for value in builtin_swift_target[field]
        )
        _require(
            report,
            f"swift_builtin_training_{field}_match_direct_processor",
            builtin_values == direct_values,
            detail={
                "direct_sha256": _integer_sequence_sha256(direct_values),
                "swift_builtin_sha256": _integer_sequence_sha256(
                    builtin_values
                ),
                "length": len(direct_values),
            },
        )

    try:
        corrected_builtin_target, correction = (
            correct_ms_swift_paligemma_training_encoding(
                builtin_swift_target,
                package_version=EXPECTED_PACKAGE_VERSIONS["ms-swift"],
            )
        )
    except TrainingCompatibilityError as exc:
        raise ContractFailure(
            "the built-in ms-swift PaliGemma token-type difference is not the "
            f"version-pinned compatibility pattern: {exc}"
        ) from exc

    raw_token_types = tuple(
        int(value) for value in builtin_swift_target["token_type_ids"]
    )
    corrected_builtin_token_types = tuple(
        int(value)
        for value in corrected_builtin_target["token_type_ids"]
    )
    direct_token_types = tuple(
        int(value) for value in direct_target["token_type_ids"][0].tolist()
    )
    mismatch_details: list[dict[str, Any]] = []
    for index in correction.mismatch_indices:
        input_id = int(builtin_swift_target["input_ids"][index])
        mismatch_details.append(
            {
                "index": index,
                "input_id": input_id,
                "decoded_token": processor.tokenizer.decode(
                    [input_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "label": int(builtin_swift_target["labels"][index]),
                "direct_token_type": direct_token_types[index],
                "swift_builtin_token_type": raw_token_types[index],
                "corrected_token_type": corrected_builtin_token_types[index],
            }
        )
    correction_detail = correction.as_dict()
    correction_detail.update(
        {
            "direct_sha256": _integer_sequence_sha256(
                direct_token_types
            ),
            "swift_builtin_sha256": _integer_sequence_sha256(
                raw_token_types
            ),
            "corrected_sha256": _integer_sequence_sha256(
                corrected_builtin_token_types
            ),
            "swift_builtin_prefix_zero_count": raw_token_types.count(0),
            "direct_prefix_zero_count": direct_token_types.count(0),
            "mismatches": mismatch_details,
        }
    )
    _require(
        report,
        "swift_builtin_token_type_difference_is_known_and_isolated",
        correction.action
        in {
            "already_canonical",
            "corrected_known_boundary_off_by_one",
        }
        and corrected_builtin_token_types == direct_token_types,
        detail=correction_detail,
    )

    exact_sequence_fields: dict[str, dict[str, Any]] = {}
    for field in ("input_ids", "labels", "token_type_ids"):
        direct_values = tuple(
            int(value) for value in direct_target[field][0].tolist()
        )
        swift_values = tuple(int(value) for value in swift_target[field])
        exact_sequence_fields[field] = {
            "direct_sha256": _integer_sequence_sha256(direct_values),
            "swift_sha256": _integer_sequence_sha256(swift_values),
            "length": len(direct_values),
        }
        _require(
            report,
            f"study1_swift_training_{field}_match_direct_processor",
            swift_values == direct_values,
            detail=exact_sequence_fields[field],
        )

    swift_pixels = torch.as_tensor(swift_target["pixel_values"])
    direct_pixels = direct_target["pixel_values"].to(dtype=swift_pixels.dtype)
    pixel_equal = bool(torch.equal(swift_pixels.cpu(), direct_pixels.cpu()))
    _require(
        report,
        "study1_swift_training_pixels_match_direct_processor",
        pixel_equal,
        detail={
            "shape": [int(value) for value in swift_pixels.shape],
            "dtype": str(swift_pixels.dtype),
            "equal": pixel_equal,
        },
    )

    swift_labels = tuple(int(value) for value in swift_target["labels"])
    scored_label_ids = [value for value in swift_labels if value != -100]
    eos_id = processor.tokenizer.eos_token_id
    _require(
        report,
        "study1_swift_training_labels_include_terminal_eos",
        bool(scored_label_ids)
        and eos_id is not None
        and scored_label_ids[-1] == int(eos_id),
        detail={
            "eos_token_id": eos_id,
            "last_scored_label_id": (
                scored_label_ids[-1] if scored_label_ids else None
            ),
        },
    )
    scored_label_ids = scored_label_ids[:-1]
    _require(
        report,
        "study1_swift_training_target_matches_saved_generation",
        tuple(scored_label_ids) == target_score.token_ids,
        detail={
            "swift_sha256": _integer_sequence_sha256(scored_label_ids),
            "target_sha256": _integer_sequence_sha256(
                target_score.token_ids
            ),
            "token_count": len(scored_label_ids),
        },
    )
    report["swift_template_equivalence"] = {
        "package_version": EXPECTED_PACKAGE_VERSIONS["ms-swift"],
        "builtin_template_type": str(TemplateType.paligemma),
        "study1_template_type": STUDY1_SWIFT_TEMPLATE_TYPE,
        "template_backend": "swift",
        "processor_id": spec.resolved_processor_id,
        "processor_revision": spec.resolved_processor_revision,
        "max_length": 512,
        "builtin_token_type_correction": correction_detail,
        "training_sequence_fields": exact_sequence_fields,
        "pixel_shape": [int(value) for value in swift_pixels.shape],
        "pixel_dtype": str(swift_pixels.dtype),
    }


def _memory_snapshot(torch: Any) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _measure_stage(
    report: dict[str, Any],
    torch: Any,
    stage: str,
    operation: Callable[[], Any],
) -> Any:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        value = operation()
    except BaseException:
        try:
            snapshot = _memory_snapshot(torch)
            snapshot["elapsed_seconds"] = round(
                time.perf_counter() - started,
                6,
            )
            snapshot["status"] = "FAIL"
            report.setdefault("memory_by_stage", {})[stage] = snapshot
        except Exception:
            report.setdefault("memory_by_stage", {})[stage] = {
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "status": "FAIL",
                "memory_snapshot_unavailable": True,
            }
        raise
    torch.cuda.synchronize()
    snapshot = _memory_snapshot(torch)
    snapshot["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    snapshot["status"] = "PASS"
    report.setdefault("memory_by_stage", {})[stage] = snapshot
    return value


def _generation_summary(result: GenerationResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "display_text": result.metadata.get("display_text"),
        "token_ids": list(result.token_ids),
        "token_logprobs": list(result.token_logprobs),
        "mean_token_logprob": result.mean_token_logprob,
        "sequence_confidence": result.sequence_confidence,
        "finish_reason": result.finish_reason,
        "metadata": dict(result.metadata),
    }


def _target_score_summary(result: TargetScore) -> dict[str, Any]:
    return {
        "target_text": result.target_text,
        "token_ids": list(result.token_ids),
        "token_logprobs": list(result.token_logprobs),
        "mean_token_logprob": result.mean_token_logprob,
        "total_logprob": result.total_logprob,
        "includes_eos": result.includes_eos,
        "metadata": dict(result.metadata),
    }


def _array_sha256(array: Any) -> str:
    contiguous = array.astype("float32", copy=False)
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "dtype": str(contiguous.dtype),
            "shape": [int(value) for value in contiguous.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_summary(
    report: dict[str, Any],
    np: Any,
    *,
    name: str,
    values: Any,
    expected_shape: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    array = np.asarray(values)
    finite = bool(np.isfinite(array).all())
    nonconstant = bool(array.size and float(array.max() - array.min()) > 1e-12)
    normalized = bool(
        finite
        and np.isclose(float(array.min()), 0.0, atol=1e-6)
        and np.isclose(float(array.max()), 1.0, atol=1e-6)
    )
    _require(
        report,
        f"{name}_shape",
        tuple(array.shape) == expected_shape,
        detail={"expected": list(expected_shape), "observed": list(array.shape)},
    )
    _require(
        report,
        f"{name}_dtype",
        str(array.dtype) == "float32",
        detail={"expected": "float32", "observed": str(array.dtype)},
    )
    _require(report, f"{name}_finite", finite, detail=finite)
    _require(report, f"{name}_nonconstant", nonconstant, detail=nonconstant)
    _require(report, f"{name}_minmax_normalized", normalized, detail=normalized)
    summary = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "sha256": _array_sha256(array),
    }
    return array, summary


def _repeat_logprob_difference(
    first: GenerationResult,
    second: GenerationResult,
) -> float:
    if len(first.token_logprobs) != len(second.token_logprobs):
        return math.inf
    return max(
        (
            abs(left - right)
            for left, right in zip(  # noqa: B905 - lengths checked above
                first.token_logprobs,
                second.token_logprobs,
            )
        ),
        default=0.0,
    )


def run_contract(
    *,
    config_path: Path,
    repository_root: Path,
    expected_commit: str,
    artifact_dir: Path,
    required_gpu_substring: str = "T4",
) -> dict[str, Any]:
    """Run the complete one-item backend contract and always write a report."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "contract_report.json"
    arrays_path = artifact_dir / "attribution_maps.npz"
    if arrays_path.exists():
        arrays_path.unlink()
    report: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": _utc_now(),
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "artifact_paths": {
            "report": str(report_path),
        },
        "checks": {},
    }
    backend: PaliGemmaBackend | None = None
    prepared = None
    torch = None
    failure: BaseException | None = None

    try:
        if not isinstance(required_gpu_substring, str) or not required_gpu_substring.strip():
            raise ContractFailure("required_gpu_substring must be a non-empty string")
        _verify_repository(
            report,
            repository_root=repository_root,
            expected_commit=expected_commit,
        )
        resolved_config_path = config_path.resolve()
        try:
            resolved_config_path.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ContractFailure(
                "the contract configuration must come from the verified repository"
            ) from exc

        raw_config = load_config(resolved_config_path)
        config = validate_config(
            raw_config,
            require_resolved=True,
            require_model_execution=True,
        )
        spec = PaliGemmaModelSpec.from_config(config)
        report["configuration"] = {
            "path": str(resolved_config_path),
            "file_sha256": file_sha256(resolved_config_path),
            "canonical_sha256": config_sha256(config),
            "backend_spec_sha256": spec.fingerprint(),
            "profile": config["profile"],
        }
        torch, np = _validate_runtime(
            report,
            required_gpu_substring=required_gpu_substring,
        )
        row, image_path = _load_diagnostic_item(
            report,
            config=config,
            artifact_dir=artifact_dir,
        )
        report["artifact_paths"]["diagnostic_image"] = str(image_path)

        backend = _measure_stage(
            report,
            torch,
            "load_backend",
            lambda: PaliGemmaBackend.from_config(config, load=True),
        )
        report["backend_provenance"] = backend.provenance.as_dict()
        prepared = _measure_stage(
            report,
            torch,
            "prepare",
            lambda: backend.prepare(
                item_id=report["diagnostic_item"]["item_id"],
                image=image_path,
                question=row["question"],
            ),
        )
        expected_grid = (16, 16)
        _require(
            report,
            "image_token_count_256",
            len(prepared.image_token_indices) == 256,
            detail={"observed": len(prepared.image_token_indices)},
        )
        _require(
            report,
            "patch_grid_16x16",
            tuple(prepared.preprocessing["patch_grid_shape"]) == expected_grid,
            detail=prepared.preprocessing["patch_grid_shape"],
        )
        _require(
            report,
            "processed_tensor_1x3x224x224",
            prepared.preprocessing["pixel_values_shape"] == [1, 3, 224, 224],
            detail=prepared.preprocessing["pixel_values_shape"],
        )
        report["prepared_input"] = {
            "prompt": prepared.prompt,
            "input_token_count": len(prepared.input_token_ids),
            "image_token_count": len(prepared.image_token_indices),
            "image_token_first": prepared.image_token_indices[0],
            "image_token_last": prepared.image_token_indices[-1],
            "preprocessing": dict(prepared.preprocessing),
        }

        generation = _measure_stage(
            report,
            torch,
            "generate_first",
            lambda: backend.generate(prepared),
        )
        repeated_generation = _measure_stage(
            report,
            torch,
            "generate_repeat",
            lambda: backend.generate(prepared),
        )
        repeat_difference = _repeat_logprob_difference(
            generation,
            repeated_generation,
        )
        _require(
            report,
            "repeat_generation_token_ids",
            generation.token_ids == repeated_generation.token_ids,
            detail={
                "first": list(generation.token_ids),
                "second": list(repeated_generation.token_ids),
            },
        )
        _require(
            report,
            "repeat_generation_text",
            generation.text == repeated_generation.text,
            detail={
                "first": generation.text,
                "second": repeated_generation.text,
            },
        )
        _require(
            report,
            "repeat_generation_logprobs",
            repeat_difference <= spec.score_absolute_tolerance,
            detail={
                "absolute_tolerance": spec.score_absolute_tolerance,
                "maximum_absolute_difference": repeat_difference,
            },
        )
        report["generation"] = _generation_summary(generation)
        report["repeated_generation"] = _generation_summary(repeated_generation)

        target_score = _measure_stage(
            report,
            torch,
            "teacher_forced_score",
            lambda: backend.score_target(
                prepared,
                generation.text,
                expected_token_ids=generation.token_ids,
            ),
        )
        verification = backend.verify_generation_score(generation, target_score)
        report["target_score"] = _target_score_summary(target_score)
        report["generation_score_verification"] = verification.as_dict()
        _require(
            report,
            "teacher_forced_token_identity",
            target_score.token_ids == generation.token_ids,
            detail=list(target_score.token_ids),
        )
        _require(
            report,
            "generation_teacher_forcing_score_parity",
            verification.maximum_absolute_difference
            <= spec.score_absolute_tolerance,
            detail=verification.as_dict(),
        )

        _measure_stage(
            report,
            torch,
            "swift_template_equivalence",
            lambda: _validate_swift_template(
                report,
                torch,
                spec=spec,
                prepared=prepared,
                target_score=target_score,
                image_path=image_path,
            ),
        )

        attention = _measure_stage(
            report,
            torch,
            "attention_attribution",
            lambda: backend.attribute(
                prepared,
                generation,
                method="decoder_answer_to_image_attention",
            ),
        )
        attention_array, attention_summary = _array_summary(
            report,
            np,
            name="attention",
            values=attention.values,
            expected_shape=expected_grid,
        )
        attention_summary.update(
            {
                "aggregation": attention.aggregation,
                "metadata": dict(attention.metadata),
                "target_score": _target_score_summary(attention.target_score),
            }
        )

        grad_cam = _measure_stage(
            report,
            torch,
            "grad_cam_attribution",
            lambda: backend.attribute(
                prepared,
                generation,
                method="answer_conditioned_grad_cam",
            ),
        )
        grad_cam_array, grad_cam_summary = _array_summary(
            report,
            np,
            name="grad_cam",
            values=grad_cam.values,
            expected_shape=expected_grid,
        )
        grad_cam_summary.update(
            {
                "aggregation": grad_cam.aggregation,
                "metadata": dict(grad_cam.metadata),
                "target_score": _target_score_summary(grad_cam.target_score),
            }
        )
        report["attributions"] = {
            "decoder_answer_to_image_attention": attention_summary,
            "answer_conditioned_grad_cam": grad_cam_summary,
        }
        np.savez_compressed(
            arrays_path,
            decoder_answer_to_image_attention=attention_array,
            answer_conditioned_grad_cam=grad_cam_array,
        )
        report["artifact_paths"]["attribution_maps"] = str(arrays_path)
        report["artifact_paths"]["attribution_maps_sha256"] = file_sha256(
            arrays_path
        )

        post_attribution_generation = _measure_stage(
            report,
            torch,
            "generate_after_attribution",
            lambda: backend.generate(prepared),
        )
        _require(
            report,
            "attribution_did_not_mutate_generation",
            post_attribution_generation.token_ids == generation.token_ids
            and post_attribution_generation.text == generation.text
            and _repeat_logprob_difference(
                generation,
                post_attribution_generation,
            )
            <= spec.score_absolute_tolerance,
            detail={
                "before_token_ids": list(generation.token_ids),
                "after_token_ids": list(post_attribution_generation.token_ids),
                "maximum_absolute_logprob_difference": (
                    _repeat_logprob_difference(
                        generation,
                        post_attribution_generation,
                    )
                ),
            },
        )
        report["post_attribution_generation"] = _generation_summary(
            post_attribution_generation
        )

        gpu_total_memory = int(report["runtime"]["gpu"]["total_memory_bytes"])
        peak_reserved = max(
            int(stage["peak_reserved_bytes"])
            for stage in report["memory_by_stage"].values()
        )
        _require(
            report,
            "peak_reserved_memory_within_gpu_capacity",
            peak_reserved < gpu_total_memory,
            detail={
                "peak_reserved_bytes": peak_reserved,
                "gpu_total_memory_bytes": gpu_total_memory,
                "fraction": peak_reserved / gpu_total_memory,
            },
        )
        report["status"] = "PASS"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["error"] = {
            "type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": _sanitise_text(str(exc)),
            "traceback": _sanitise_text(traceback.format_exc()),
        }
        if torch is not None and getattr(torch, "cuda", None) is not None:
            try:
                report["failure_memory"] = _memory_snapshot(torch)
            except Exception:
                pass
    finally:
        prepared = None
        if backend is not None:
            try:
                backend.close()
            except Exception as cleanup_exc:
                report["cleanup_error"] = {
                    "type": (
                        f"{type(cleanup_exc).__module__}."
                        f"{type(cleanup_exc).__name__}"
                    ),
                    "message": _sanitise_text(str(cleanup_exc)),
                }
                if failure is None:
                    failure = cleanup_exc
                    report["status"] = "FAIL"
                    report["error"] = {
                        "type": report["cleanup_error"]["type"],
                        "message": report["cleanup_error"]["message"],
                        "traceback": _sanitise_text(traceback.format_exc()),
                    }
        backend = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                report["memory_after_cleanup"] = _memory_snapshot(torch)
            except Exception:
                pass
        report["finished_at_utc"] = _utc_now()
        _atomic_write_json(report_path, report)

    if failure is not None:
        raise failure
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-item Colab T4 PaliGemma backend contract"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--required-gpu-substring", default="T4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_contract(
            config_path=args.config,
            repository_root=args.repository_root,
            expected_commit=args.expected_commit,
            artifact_dir=args.artifact_dir,
            required_gpu_substring=args.required_gpu_substring,
        )
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "report": str(args.artifact_dir / "contract_report.json"),
                    "error": _sanitise_text(str(exc)),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "attribution_maps": report["artifact_paths"].get(
                    "attribution_maps"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
