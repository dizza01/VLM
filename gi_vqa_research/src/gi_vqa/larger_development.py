"""Locked selection and protocol validation for the larger development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .identifiers import source_image_id
from .jsonl import iter_jsonl
from .provenance import canonical_json_sha256, file_sha256

SELECTION_SCHEMA_VERSION = "gi-vqa-larger-development-selection-v1"
PROTOCOL_SCHEMA_VERSION = "gi-vqa-larger-development-protocol-v1"
SELECTION_ALGORITHM = "sha256-development-source-and-item-selection-v1"
CONDITIONS = (
    "paired_correct",
    "constant_control",
    "paired_shuffled",
    "paired_neutral_ablation",
    "base_correct_descriptive",
)


class LargerDevelopmentError(ValueError):
    """Raised when the larger development lock is unsafe or inconsistent."""


def build_selection_manifest(
    *,
    project_root: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
    source_count: int = 256,
    faithfulness_count: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Select one item per development source without resolving test artifacts."""

    root = Path(project_root).resolve()
    split_path = _under(root, split_manifest_path)
    output = _under(root, output_path)
    split = _read_object(split_path)
    if source_count != 256 or faithfulness_count != 64 or seed != 42:
        raise LargerDevelopmentError("the first larger-development lock is fixed")

    artifacts = _mapping(split.get("artifacts"), "split artifacts")
    development_descriptor = _mapping(
        artifacts.get("development"), "development artifact"
    )
    development_path = _under(root, development_descriptor.get("path"))
    if file_sha256(development_path) != development_descriptor.get("sha256"):
        raise LargerDevelopmentError("development artifact differs from split lock")

    smoke = _mapping(split.get("smoke"), "split smoke")
    excluded_sources = set(_string_list(smoke.get("source_img_ids"), "smoke sources"))
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(development_path):
        metadata = _mapping(record.get("metadata"), "record metadata")
        if metadata.get("partition") != "development":
            raise LargerDevelopmentError("non-development record encountered")
        source_id = source_image_id(record)
        if source_id not in excluded_sources:
            candidates[source_id].append(dict(record))
    if len(candidates) < source_count:
        raise LargerDevelopmentError("insufficient development source groups")

    ordered_sources = sorted(
        candidates,
        key=lambda value: (_digest(seed, value), value),
    )[:source_count]
    selected = []
    for source_id in ordered_sources:
        record = min(
            candidates[source_id],
            key=lambda value: (
                _digest(seed, str(value.get("item_id"))),
                str(value.get("item_id")),
            ),
        )
        selected.append(record)

    shuffled_sources = ordered_sources[1:] + ordered_sources[:1]
    if any(left == right for left, right in zip(ordered_sources, shuffled_sources)):
        raise LargerDevelopmentError("shuffled-image permutation has a fixed point")
    records = []
    for rank, (record, shuffled_source) in enumerate(
        zip(selected, shuffled_sources)
    ):
        metadata = _mapping(record.get("metadata"), "record metadata")
        records.append(
            {
                "rank": rank,
                "item_id": _string(record.get("item_id"), "item_id"),
                "source_img_id": source_image_id(record),
                "shuffled_source_img_id": shuffled_source,
                "complexity": metadata.get("complexity"),
                "question_class": metadata.get("question_class", []),
                "record_sha256": canonical_json_sha256(record),
            }
        )

    question_classes = Counter()
    complexities = Counter()
    for record in records:
        complexities[str(record["complexity"])] += 1
        classes = record["question_class"]
        for value in classes if isinstance(classes, list) else [classes]:
            question_classes[str(value)] += 1
    manifest = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "LOCKED",
        "partition": "development",
        "test_partition_accessed": False,
        "algorithm": {
            "id": SELECTION_ALGORITHM,
            "seed": seed,
            "source_order": "ascending sha256(seed + NUL + source_img_id)",
            "item_order": "ascending sha256(seed + NUL + item_id)",
            "shuffled_images": "one-position cyclic rotation of selected sources",
        },
        "split_manifest": {
            "path": _portable(split_path, root),
            "sha256": file_sha256(split_path),
        },
        "development_artifact": {
            "path": _portable(development_path, root),
            "sha256": file_sha256(development_path),
        },
        "pilot_exclusion": {
            "source_count": len(excluded_sources),
            "source_img_ids_sha256": canonical_json_sha256(sorted(excluded_sources)),
        },
        "selection": {
            "source_count": source_count,
            "items_per_source": 1,
            "item_count": len(records),
            "ordered_item_ids_sha256": canonical_json_sha256(
                [record["item_id"] for record in records]
            ),
            "ordered_source_img_ids_sha256": canonical_json_sha256(ordered_sources),
            "faithfulness_item_count": faithfulness_count,
            "faithfulness_item_ids": [
                record["item_id"] for record in records[:faithfulness_count]
            ],
        },
        "balance": {
            "complexity_counts": dict(sorted(complexities.items())),
            "question_class_counts": dict(sorted(question_classes.items())),
        },
        "records": records,
    }
    _publish_or_validate(output, manifest)
    return manifest


def validate_locked_protocol(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Validate every file binding and no-test rule in the locked protocol."""

    root = Path(project_root).resolve()
    path = _under(root, protocol_path)
    protocol = _read_object(path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise LargerDevelopmentError("unexpected larger-development schema")
    if protocol.get("status") != "LOCKED":
        raise LargerDevelopmentError("larger-development protocol is not locked")
    if protocol.get("partition") != "development":
        raise LargerDevelopmentError("larger-development partition must be development")
    seal = _mapping(protocol.get("test_set_seal"), "test-set seal")
    if seal.get("access_allowed") is not False:
        raise LargerDevelopmentError("test-set access is not sealed")
    if tuple(protocol.get("conditions", {}).keys()) != CONDITIONS:
        raise LargerDevelopmentError("condition order differs from the lock")

    for name in (
        "grouped_split_manifest",
        "controlled_training_pass",
        "controlled_evaluation_pass",
        "selection_manifest",
    ):
        descriptor = _mapping(protocol.get("inputs", {}).get(name), name)
        bound = _under(root, descriptor.get("path"))
        if file_sha256(bound) != descriptor.get("sha256"):
            raise LargerDevelopmentError(f"locked input changed: {name}")
    implementation = _mapping(
        protocol.get("implementation"), "implementation bindings"
    )
    for name, descriptor_value in implementation.items():
        descriptor = _mapping(descriptor_value, f"implementation {name}")
        bound = _under(root, descriptor.get("path"))
        if file_sha256(bound) != descriptor.get("sha256"):
            raise LargerDevelopmentError(f"locked implementation changed: {name}")
    selection_descriptor = protocol["inputs"]["selection_manifest"]
    selection = _read_object(_under(root, selection_descriptor["path"]))
    if selection.get("partition") != "development":
        raise LargerDevelopmentError("selection is not development-only")
    if selection.get("test_partition_accessed") is not False:
        raise LargerDevelopmentError("selection accessed test")
    if selection.get("selection", {}).get("source_count") != 256:
        raise LargerDevelopmentError("selection must contain 256 sources")
    if len(selection.get("records", [])) != 256:
        raise LargerDevelopmentError("selection record count changed")
    if len({record["source_img_id"] for record in selection["records"]}) != 256:
        raise LargerDevelopmentError("selection source groups are not unique")
    if any(
        record["source_img_id"] == record["shuffled_source_img_id"]
        for record in selection["records"]
    ):
        raise LargerDevelopmentError("shuffled control contains a fixed point")
    return {
        "status": "PASS",
        "protocol": _portable(path, root),
        "protocol_sha256": file_sha256(path),
        "selection_sha256": file_sha256(_under(root, selection_descriptor["path"])),
        "development_items": 256,
        "conditions": list(CONDITIONS),
        "test_partition_accessed": False,
    }


def _digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LargerDevelopmentError(f"expected JSON object: {path}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LargerDevelopmentError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LargerDevelopmentError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise LargerDevelopmentError(f"{name} must be a string list")
    return value


def _under(root: Path, value: Any) -> Path:
    path = Path(_string(str(value), "path"))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LargerDevelopmentError(f"path escapes project root: {value}") from exc
    return resolved


def _portable(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _publish_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise LargerDevelopmentError(f"existing selection differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--project-root", type=Path, default=Path("."))
    select.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("protocols/study1/grouped_split_manifest.json"),
    )
    select.add_argument(
        "--output",
        type=Path,
        default=Path("protocols/study1/larger_development_selection.json"),
    )
    check = subparsers.add_parser("check")
    check.add_argument("--project-root", type=Path, default=Path("."))
    check.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/study1/larger_development_protocol.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        result = build_selection_manifest(
            project_root=args.project_root,
            split_manifest_path=args.split_manifest,
            output_path=args.output,
        )
    else:
        result = validate_locked_protocol(
            project_root=args.project_root,
            protocol_path=args.protocol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
