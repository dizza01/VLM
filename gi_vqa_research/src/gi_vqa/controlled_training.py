"""Locked preparation and commands for the first controlled Study 1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .identifiers import source_image_id, stable_item_id
from .jsonl import iter_jsonl, read_jsonl
from .provenance import canonical_json_sha256, file_sha256
from .splits import (
    materialize_grouped_split_artifacts,
    verify_grouped_split_artifacts,
)
from .training import STUDY1_SWIFT_TEMPLATE_TYPE

CONTROLLED_TRAINING_SCHEMA_VERSION = "gi-vqa-controlled-training-pilot-v1"
CONTROLLED_TRAINING_DATA_SCHEMA_VERSION = "gi-vqa-controlled-training-data-v1"
CONTROLLED_TRAINING_SELECTION_ALGORITHM = (
    "sha256-controlled-pilot-source-and-record-selection-v1"
)
CONTROLLED_TRAINING_CONDITIONS = ("paired_image", "constant_image")


class ControlledTrainingError(RuntimeError):
    """Raised when the controlled pilot would depart from its locked design."""


def load_controlled_training_protocol(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the controlled-pilot protocol."""

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ControlledTrainingError("controlled training protocol must be an object")
    _validate_protocol(value)
    return json.loads(json.dumps(value, sort_keys=True))


def prepare_controlled_training_data(
    *,
    protocol_path: str | Path,
    split_manifest_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    token: str | None = None,
    fetch_image: Callable[[str, str, str, str | None], str | Path] | None = None,
    materialize_splits: bool = True,
) -> dict[str, Any]:
    """Prepare identical paired/constant training arms without test contact.

    The function is restart-safe at file granularity. Existing immutable
    artifacts must match their reconstructed content; they are never silently
    overwritten.
    """

    root = Path(project_root).resolve()
    protocol_file = _resolve_under(root, protocol_path)
    split_file = _resolve_under(root, split_manifest_path)
    output = _resolve_under(root, output_dir)
    protocol = load_controlled_training_protocol(protocol_file)
    _validate_required_receipts(protocol, protocol_file.parent)
    if file_sha256(split_file) != protocol["split_manifest_sha256"]:
        raise ControlledTrainingError(
            "grouped split manifest differs from the controlled pilot lock"
        )
    if materialize_splits:
        materialize_grouped_split_artifacts(
            manifest_path=split_file,
            project_root=root,
        )
    split_check = verify_grouped_split_artifacts(
        manifest_path=split_file,
        project_root=root,
    )
    split_manifest = _read_json_object(split_file)
    train_descriptor = _required_mapping(
        _required_mapping(split_manifest.get("artifacts"), "split artifacts").get(
            "train"
        ),
        "training split artifact",
    )
    train_path = _resolve_under(
        root,
        _required_string(train_descriptor.get("path"), "training split path"),
    )
    if file_sha256(train_path) != train_descriptor.get("sha256"):
        raise ControlledTrainingError("training JSONL differs from the split manifest")

    output.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output / "controlled_training_data_manifest.json"
    if final_manifest_path.is_file():
        return verify_controlled_training_data(
            manifest_path=final_manifest_path,
            protocol_path=protocol_file,
            split_manifest_path=split_file,
            project_root=root,
        )

    selection = _select_training_records(
        train_path,
        source_count=int(protocol["data"]["source_images"]),
        records_per_source=int(protocol["data"]["records_per_source"]),
        seed=int(protocol["optimisation"]["data_seed"]),
    )
    selection_payload = {
        "schema_version": "gi-vqa-controlled-training-selection-v1",
        "algorithm": CONTROLLED_TRAINING_SELECTION_ALGORITHM,
        "protocol_sha256": canonical_json_sha256(protocol),
        "split_manifest_sha256": file_sha256(split_file),
        "training_jsonl_sha256": file_sha256(train_path),
        "source_img_ids": selection["source_img_ids"],
        "item_ids": selection["item_ids"],
        "source_images": len(selection["source_img_ids"]),
        "records": len(selection["records"]),
    }
    selection_path = output / "selection.json"
    _publish_or_validate_json(selection_path, selection_payload)

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset = _required_mapping(
        split_manifest.get("image_dataset"),
        "split image dataset",
    )
    dataset_id = _required_string(dataset.get("id"), "dataset ID")
    dataset_revision = _required_string(dataset.get("revision"), "dataset revision")
    downloader = fetch_image or _fetch_huggingface_image
    image_descriptors: dict[str, dict[str, Any]] = {}
    for source_id in selection["source_img_ids"]:
        remote_filename = f"images/{source_id}.jpg"
        source = Path(
            downloader(dataset_id, dataset_revision, remote_filename, token)
        )
        if not source.is_file():
            raise FileNotFoundError(f"image downloader returned no file: {source}")
        destination = images_dir / f"{source_id}.jpg"
        _publish_or_validate_file(source, destination)
        image_info = _inspect_rgb_image(destination)
        image_descriptors[source_id] = {
            "path": _portable_path(destination, root),
            "repo_filename": remote_filename,
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            **image_info,
        }

    constant_path = output / "constant_image.png"
    _publish_or_validate_constant_image(
        constant_path,
        width=int(protocol["data"]["constant_image"]["width"]),
        height=int(protocol["data"]["constant_image"]["height"]),
        rgb=tuple(int(value) for value in protocol["data"]["constant_image"]["rgb"]),
    )

    paired_rows = [
        _condition_record(
            record,
            condition="paired_image",
            image_path=images_dir / f"{source_image_id(record)}.jpg",
            project_root=root,
            pilot_id=str(protocol["pilot_id"]),
        )
        for record in selection["records"]
    ]
    constant_rows = [
        _condition_record(
            record,
            condition="constant_image",
            image_path=constant_path,
            project_root=root,
            pilot_id=str(protocol["pilot_id"]),
        )
        for record in selection["records"]
    ]
    paired_path = output / "paired_image_train.jsonl"
    constant_dataset_path = output / "constant_image_train.jsonl"
    _publish_or_validate_jsonl(paired_path, paired_rows)
    _publish_or_validate_jsonl(constant_dataset_path, constant_rows)
    _validate_condition_pair(paired_rows, constant_rows)

    manifest = {
        "schema_version": CONTROLLED_TRAINING_DATA_SCHEMA_VERSION,
        "status": "PASS",
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "protocol": _fingerprint(protocol_file, root),
        "split_manifest": _fingerprint(split_file, root),
        "split_verification": split_check,
        "selection": _fingerprint(selection_path, root),
        "training_source": _fingerprint(train_path, root),
        "source_images": len(image_descriptors),
        "records_per_condition": len(paired_rows),
        "ordered_item_ids_sha256": canonical_json_sha256(selection["item_ids"]),
        "conditions": {
            "paired_image": _fingerprint(paired_path, root),
            "constant_image": _fingerprint(constant_dataset_path, root),
        },
        "constant_image": _fingerprint(constant_path, root),
        "images": image_descriptors,
        "test_partition_accessed": False,
    }
    _publish_or_validate_json(final_manifest_path, manifest)
    return verify_controlled_training_data(
        manifest_path=final_manifest_path,
        protocol_path=protocol_file,
        split_manifest_path=split_file,
        project_root=root,
    )


def verify_controlled_training_data(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    split_manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verify the prepared pilot arms and every recorded artifact hash."""

    root = Path(project_root).resolve()
    manifest_file = _resolve_under(root, manifest_path)
    protocol_file = _resolve_under(root, protocol_path)
    split_file = _resolve_under(root, split_manifest_path)
    manifest = _read_json_object(manifest_file)
    if manifest.get("schema_version") != CONTROLLED_TRAINING_DATA_SCHEMA_VERSION:
        raise ControlledTrainingError("unsupported controlled-training data manifest")
    if manifest.get("status") != "PASS" or manifest.get("test_partition_accessed") is not False:
        raise ControlledTrainingError("controlled-training data manifest is not a safe PASS")
    protocol = load_controlled_training_protocol(protocol_file)
    if manifest.get("protocol", {}).get("sha256") != file_sha256(protocol_file):
        raise ControlledTrainingError("controlled-training protocol hash changed")
    if manifest.get("split_manifest", {}).get("sha256") != file_sha256(split_file):
        raise ControlledTrainingError("controlled-training split hash changed")
    if file_sha256(split_file) != protocol["split_manifest_sha256"]:
        raise ControlledTrainingError("split manifest differs from the protocol")

    artifact_descriptors = [
        manifest["selection"],
        manifest["training_source"],
        manifest["constant_image"],
        *manifest["conditions"].values(),
        *manifest["images"].values(),
    ]
    for descriptor in artifact_descriptors:
        _verify_fingerprint(descriptor, root)
    paired = read_jsonl(_resolve_under(root, manifest["conditions"]["paired_image"]["path"]))
    constant = read_jsonl(
        _resolve_under(root, manifest["conditions"]["constant_image"]["path"])
    )
    _validate_condition_pair(paired, constant)
    expected_sources = int(protocol["data"]["source_images"])
    expected_records = expected_sources * int(protocol["data"]["records_per_source"])
    if len(manifest["images"]) != expected_sources or len(paired) != expected_records:
        raise ControlledTrainingError("prepared pilot dimensions differ from the protocol")
    if manifest.get("ordered_item_ids_sha256") != canonical_json_sha256(
        [stable_item_id(record) for record in paired]
    ):
        raise ControlledTrainingError("prepared pilot item order changed")
    return {
        "status": "PASS",
        "manifest": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "source_images": expected_sources,
        "records_per_condition": expected_records,
        "conditions": list(CONTROLLED_TRAINING_CONDITIONS),
        "ordered_item_ids_sha256": manifest["ordered_item_ids_sha256"],
        "test_partition_accessed": False,
    }


def build_controlled_training_command(
    *,
    protocol: Mapping[str, Any],
    condition: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    resume_from_checkpoint: str | Path | None = None,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build one immutable ms-swift phase for either controlled arm."""

    _validate_protocol(protocol)
    if condition not in CONTROLLED_TRAINING_CONDITIONS:
        raise ControlledTrainingError(f"unsupported controlled condition: {condition}")
    optimisation = protocol["optimisation"]
    phase_one = int(optimisation["phase_one_steps"])
    phase_two = int(optimisation["phase_two_total_steps"])
    if max_steps not in (phase_one, phase_two):
        raise ControlledTrainingError(
            f"max_steps must be the locked phase boundary {phase_one} or {phase_two}"
        )
    if (max_steps == phase_one) != (resume_from_checkpoint is None):
        raise ControlledTrainingError(
            "phase one must start fresh and phase two must explicitly resume"
        )
    model = protocol["model"]
    command = [
        str(python_executable),
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
        str(optimisation["max_length"]),
        "--max_steps",
        str(max_steps),
        "--per_device_train_batch_size",
        str(optimisation["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(optimisation["gradient_accumulation_steps"]),
        "--learning_rate",
        str(optimisation["learning_rate"]),
        "--lr_scheduler_type",
        str(optimisation["lr_scheduler"]),
        "--warmup_ratio",
        str(optimisation["warmup_ratio"]),
        "--weight_decay",
        str(optimisation["weight_decay"]),
        "--optim",
        str(optimisation["optimizer"]),
        "--max_grad_norm",
        str(optimisation["max_grad_norm"]),
        "--lora_rank",
        str(model["lora_rank"]),
        "--lora_alpha",
        str(model["lora_alpha"]),
        "--lora_dropout",
        str(model["lora_dropout"]),
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        _bool_text(model["freeze_vision_tower"]),
        "--freeze_aligner",
        _bool_text(model["freeze_aligner"]),
        "--gradient_checkpointing",
        _bool_text(optimisation["gradient_checkpointing"]),
        "--attn_impl",
        "eager",
        "--split_dataset_ratio",
        "0",
        "--dataset_shuffle",
        "false",
        "--train_dataloader_shuffle",
        "true",
        "--dataset_num_proc",
        "1",
        "--dataloader_num_workers",
        str(optimisation["dataloader_workers"]),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(optimisation["save_steps"]),
        "--save_total_limit",
        "2",
        "--save_safetensors",
        "true",
        "--save_only_model",
        "false",
        "--logging_steps",
        str(optimisation["logging_steps"]),
        "--logging_first_step",
        "true",
        "--report_to",
        "none",
        "--output_dir",
        str(output_dir),
        "--run_name",
        f"{protocol['pilot_id']}-{condition}",
        "--add_version",
        "false",
        "--seed",
        str(optimisation["seed"]),
        "--data_seed",
        str(optimisation["data_seed"]),
    ]
    if resume_from_checkpoint is not None:
        command.extend(["--resume_from_checkpoint", str(resume_from_checkpoint)])
    return command


def adapter_artifact_sha256(checkpoint: str | Path) -> str:
    """Return the portable content identity of a PEFT adapter."""

    root = Path(checkpoint)
    config = root / "adapter_config.json"
    candidates = (root / "adapter_model.safetensors", root / "adapter_model.bin")
    weights = next((path for path in candidates if path.is_file()), candidates[0])
    for path in (config, weights):
        if not path.is_file():
            raise ControlledTrainingError(f"adapter artifact is missing: {path}")
    return canonical_json_sha256(
        {
            "schema_version": "gi-vqa-peft-adapter-artifact-v1",
            "adapter_config": {
                "bytes": config.stat().st_size,
                "sha256": file_sha256(config),
            },
            "adapter_weights": {
                "filename": weights.name,
                "bytes": weights.stat().st_size,
                "sha256": file_sha256(weights),
            },
        }
    )


def _select_training_records(
    train_path: Path,
    *,
    source_count: int,
    records_per_source: int,
    seed: int,
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(train_path):
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("partition") != "train":
            raise ControlledTrainingError("training JSONL contains a non-training record")
        observed = record.get("item_id")
        expected = stable_item_id(record)
        if observed != expected:
            raise ControlledTrainingError(f"training item identity changed: {observed!r}")
        grouped[source_image_id(record)].append(record)
    eligible = [
        source_id
        for source_id, records in grouped.items()
        if len(records) >= records_per_source
    ]
    if len(eligible) < source_count:
        raise ControlledTrainingError(
            f"only {len(eligible)} sources have {records_per_source} records; "
            f"the protocol requires {source_count}"
        )
    ordered_sources = sorted(
        eligible,
        key=lambda source_id: _selection_digest(seed, "source", source_id),
    )[:source_count]
    selected: list[dict[str, Any]] = []
    for source_id in ordered_sources:
        records = sorted(
            grouped[source_id],
            key=lambda record: _selection_digest(
                seed,
                "record",
                source_id,
                stable_item_id(record),
            ),
        )[:records_per_source]
        selected.extend(deepcopy(records))
    item_ids = [stable_item_id(record) for record in selected]
    if len(item_ids) != len(set(item_ids)):
        raise ControlledTrainingError("controlled pilot selection contains duplicate items")
    return {
        "source_img_ids": ordered_sources,
        "item_ids": item_ids,
        "records": selected,
    }


def _selection_digest(seed: int, *parts: str) -> str:
    payload = "\0".join(
        (CONTROLLED_TRAINING_SELECTION_ALGORITHM, str(seed), *parts)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _condition_record(
    record: Mapping[str, Any],
    *,
    condition: str,
    image_path: Path,
    project_root: Path,
    pilot_id: str,
) -> dict[str, Any]:
    value = deepcopy(dict(record))
    value["images"] = [_portable_path(image_path, project_root)]
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ControlledTrainingError("selected training record has no metadata")
    value["metadata"] = {
        **dict(metadata),
        "controlled_training": {
            "pilot_id": pilot_id,
            "condition": condition,
        },
    }
    return value


def _validate_condition_pair(
    paired: Sequence[Mapping[str, Any]],
    constant: Sequence[Mapping[str, Any]],
) -> None:
    if len(paired) != len(constant) or not paired:
        raise ControlledTrainingError("controlled arms must have equal non-zero lengths")
    for index, (left, right) in enumerate(zip(paired, constant)):  # noqa: B905
        if stable_item_id(left) != stable_item_id(right):
            raise ControlledTrainingError(f"controlled arm item mismatch at row {index}")
        if left.get("messages") != right.get("messages"):
            raise ControlledTrainingError(f"controlled arm message mismatch at row {index}")
        left_condition = left.get("metadata", {}).get("controlled_training", {}).get(
            "condition"
        )
        right_condition = right.get("metadata", {}).get("controlled_training", {}).get(
            "condition"
        )
        if (left_condition, right_condition) != CONTROLLED_TRAINING_CONDITIONS:
            raise ControlledTrainingError(f"controlled arm labels differ at row {index}")
        if left.get("images") == right.get("images"):
            raise ControlledTrainingError(f"controlled arm images did not differ at row {index}")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != CONTROLLED_TRAINING_SCHEMA_VERSION:
        raise ControlledTrainingError("unsupported controlled training protocol schema")
    if protocol.get("status") != "LOCKED":
        raise ControlledTrainingError("controlled training protocol is not LOCKED")
    if protocol.get("diagnostic_only") is not True:
        raise ControlledTrainingError("controlled pilot must remain diagnostic_only")
    if protocol.get("excluded_from_research_results") is not True:
        raise ControlledTrainingError("controlled pilot must be excluded from research results")
    data = _required_mapping(protocol.get("data"), "protocol data")
    if data.get("partition") != "train":
        raise ControlledTrainingError("controlled training may access only train")
    if data.get("selection_algorithm") != CONTROLLED_TRAINING_SELECTION_ALGORITHM:
        raise ControlledTrainingError("controlled training selection algorithm changed")
    if data.get("condition_order") != list(CONTROLLED_TRAINING_CONDITIONS):
        raise ControlledTrainingError("controlled condition order changed")
    if data.get("source_images") != 256 or data.get("records_per_source") != 4:
        raise ControlledTrainingError("controlled pilot dimensions changed")
    arms = protocol.get("arms")
    if not isinstance(arms, list) or [arm.get("condition") for arm in arms] != list(
        CONTROLLED_TRAINING_CONDITIONS
    ):
        raise ControlledTrainingError("controlled pilot arms changed")
    model = _required_mapping(protocol.get("model"), "protocol model")
    expected_model = {
        "quantization": "bnb-nf4-4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "freeze_vision_tower": True,
        "freeze_aligner": True,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ControlledTrainingError(f"locked model field changed: {field}")
    optimisation = _required_mapping(protocol.get("optimisation"), "optimisation")
    expected_optimisation = {
        "seed": 42,
        "data_seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "phase_one_steps": 128,
        "phase_two_total_steps": 256,
        "save_steps": 128,
        "max_length": 384,
    }
    for field, expected in expected_optimisation.items():
        if optimisation.get(field) != expected:
            raise ControlledTrainingError(f"locked optimisation field changed: {field}")
    examples = int(data["source_images"]) * int(data["records_per_source"])
    updates = int(optimisation["phase_two_total_steps"])
    effective_batch = int(optimisation["effective_batch_size"])
    if examples != updates * effective_batch:
        raise ControlledTrainingError(
            "controlled pilot must remain exactly one effective training epoch"
        )
    evaluation = _required_mapping(
        protocol.get("evaluation_after_training"),
        "evaluation_after_training",
    )
    if evaluation.get("test_partition_access") is not False:
        raise ControlledTrainingError("controlled pilot evaluation cannot access test")
    receipts = _required_mapping(
        protocol.get("required_pass_receipts"),
        "required_pass_receipts",
    )
    if set(receipts) != {
        "backend_contract_pass.json",
        "training_gate_pass.json",
        "development_smoke_pass.json",
    }:
        raise ControlledTrainingError("controlled pilot prerequisite receipts changed")


def _validate_required_receipts(
    protocol: Mapping[str, Any],
    protocol_directory: Path,
) -> None:
    for filename, expected_hash in protocol["required_pass_receipts"].items():
        path = protocol_directory / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ControlledTrainingError(
                f"required PASS receipt is missing or changed: {filename}"
            )
        payload = _read_json_object(path)
        if payload.get("status") != "PASS":
            raise ControlledTrainingError(f"required receipt is not PASS: {filename}")


def _fetch_huggingface_image(
    dataset_id: str,
    revision: str,
    filename: str,
    token: str | None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional data/GPU dependency
        raise ControlledTrainingError(
            "controlled data preparation requires huggingface-hub"
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            token=token,
        )
    )


def _inspect_rgb_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ControlledTrainingError("Pillow is required for image validation") from exc
    with Image.open(path) as image:
        image.load()
        converted = image.convert("RGB")
        return {
            "format": image.format,
            "mode": converted.mode,
            "width": converted.width,
            "height": converted.height,
            "rgb_sha256": hashlib.sha256(converted.tobytes()).hexdigest(),
        }


def _publish_or_validate_constant_image(
    path: Path,
    *,
    width: int,
    height: int,
    rgb: tuple[int, ...],
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ControlledTrainingError("Pillow is required for constant image") from exc
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ControlledTrainingError("constant image RGB value is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".png",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.new("RGB", (width, height), color=rgb).save(temporary, format="PNG")
        _publish_or_validate_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    with Image.open(path) as observed:
        pixels = observed.convert("RGB")
        if pixels.size != (width, height) or set(pixels.getdata()) != {rgb}:
            raise ControlledTrainingError("constant image pixels changed")


def _publish_or_validate_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    if destination.is_file():
        if file_sha256(destination) != source_hash:
            raise ControlledTrainingError(f"existing artifact differs: {destination}")
        return
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ControlledTrainingError(
                f"refusing to overwrite artifact: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _publish_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    _publish_or_validate_text(path, expected)


def _publish_or_validate_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected = "".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    _publish_or_validate_text(path, expected)


def _publish_or_validate_text(path: Path, expected: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ControlledTrainingError(f"existing artifact differs: {path}")
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
            raise ControlledTrainingError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fingerprint(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": _portable_path(path, project_root),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _verify_fingerprint(descriptor: Mapping[str, Any], project_root: Path) -> None:
    path = _resolve_under(
        project_root,
        _required_string(descriptor.get("path"), "artifact path"),
    )
    if not path.is_file():
        raise FileNotFoundError(f"controlled training artifact is missing: {path}")
    if descriptor.get("bytes") != path.stat().st_size:
        raise ControlledTrainingError(f"controlled training artifact size changed: {path}")
    if descriptor.get("sha256") != file_sha256(path):
        raise ControlledTrainingError(f"controlled training artifact hash changed: {path}")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as exc:
        raise ControlledTrainingError(f"artifact escapes project root: {path}") from exc


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ControlledTrainingError(f"path escapes project root: {path}") from exc
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ControlledTrainingError(f"JSON root must be an object: {path}")
    return value


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledTrainingError(f"{name} must be a mapping")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlledTrainingError(f"{name} must be a non-empty string")
    return value


def _bool_text(value: Any) -> str:
    if not isinstance(value, bool):
        raise ControlledTrainingError("locked Boolean field changed type")
    return "true" if value else "false"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gi-vqa-controlled-training",
        description="Prepare or inspect the locked Study 1 controlled training pilot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", required=True, type=Path)
    prepare.add_argument("--split-manifest", required=True, type=Path)
    prepare.add_argument("--project-root", type=Path, default=Path("."))
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/controlled_training_pilot"),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--protocol", required=True, type=Path)
    verify.add_argument("--split-manifest", required=True, type=Path)
    verify.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_controlled_training_data(
            protocol_path=args.protocol,
            split_manifest_path=args.split_manifest,
            project_root=args.project_root,
            output_dir=args.output_dir,
            token=os.getenv("HF_TOKEN"),
        )
    else:
        result = verify_controlled_training_data(
            manifest_path=args.manifest,
            protocol_path=args.protocol,
            split_manifest_path=args.split_manifest,
            project_root=args.project_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLLED_TRAINING_CONDITIONS",
    "CONTROLLED_TRAINING_DATA_SCHEMA_VERSION",
    "CONTROLLED_TRAINING_SCHEMA_VERSION",
    "ControlledTrainingError",
    "adapter_artifact_sha256",
    "build_controlled_training_command",
    "load_controlled_training_protocol",
    "prepare_controlled_training_data",
    "verify_controlled_training_data",
]
