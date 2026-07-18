"""Deterministic, leakage-safe Study 1 split construction and verification.

The primary split is built from the union of the pinned public train and test
metadata. Exact duplicates are collapsed, ambiguous image-question annotations
are excluded, and every remaining question for a source image is assigned to
one partition. The fixed backend-contract source image is reserved from all
research partitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import SplitAuditReport, assert_disjoint_source_images, audit_records
from .identifiers import (
    RecordFormatError,
    canonical_text,
    question_text,
    source_image_id,
    stable_item_id,
)
from .jsonl import iter_jsonl, write_jsonl_atomic
from .provenance import canonical_json_sha256, file_sha256

GROUPED_SPLIT_SCHEMA_VERSION = "gi-vqa-grouped-split-manifest-v1"
GROUPED_SPLIT_ALGORITHM = "sha256-source-group-order-v1"
SMOKE_SELECTION_ALGORITHM = "metadata-greedy-balanced-v1"
PINNED_DATASET_REVISION = "61e41148c3214bc5140ad0ab4c28520a512e2a73"
CONTRACT_RESERVED_SOURCE_IDS = ("cl8k2u1pv1e4z08320vbv6jzb",)
PRIMARY_PARTITIONS = ("train", "development", "test")

PINNED_OFFICIAL_AUDIT = {
    "official_train_rows": 143_594,
    "official_test_rows": 15_955,
    "official_train_source_images": 6_212,
    "official_test_source_images": 4_058,
    "overlapping_source_images": 3_821,
    "overlapping_image_question_pairs": 61,
    "overlapping_image_question_answer_triples": 5,
}
PINNED_MERGE_AUDIT = {
    "input_rows": 159_549,
    "exact_duplicates_removed": 18,
    "conflicting_image_question_groups_removed": 324,
    "metadata_conflict_groups_removed": 9,
    "conflicting_rows_removed": 657,
    "primary_rows_after_annotation_exclusions": 158_874,
    "primary_source_images_after_annotation_exclusions": 6_449,
}


class SplitBuildError(ValueError):
    """Raised when split construction cannot satisfy the locked safety rules."""


@dataclass(frozen=True)
class SplitBuildPaths:
    """Filesystem destinations for one grouped split build."""

    project_root: Path
    data_root: Path
    manifest_path: Path
    image_dir: Path

    @classmethod
    def resolve(
        cls,
        *,
        project_root: str | Path,
        data_root: str | Path,
        manifest_path: str | Path,
        image_dir: str | Path,
    ) -> SplitBuildPaths:
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            data_root=_resolve_under(root, data_root),
            manifest_path=_resolve_under(root, manifest_path),
            image_dir=_resolve_under(root, image_dir),
        )


def build_grouped_splits(
    *,
    official_train_records: Iterable[Mapping[str, Any]],
    official_test_records: Iterable[Mapping[str, Any]],
    dataset_id: str,
    dataset_revision: str,
    image_dataset_id: str,
    image_dataset_revision: str,
    seed: int,
    development_fraction: float,
    test_fraction: float,
    smoke_items: int,
    paths: SplitBuildPaths,
    reserved_source_ids: Sequence[str] = CONTRACT_RESERVED_SOURCE_IDS,
) -> dict[str, Any]:
    """Build and publish deterministic grouped splits plus their manifest.

    Data files are assembled in a temporary sibling directory and published by
    one directory rename. The compact manifest is written last and is intended
    to be tracked by Git.
    """

    _validate_build_arguments(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        image_dataset_id=image_dataset_id,
        image_dataset_revision=image_dataset_revision,
        seed=seed,
        development_fraction=development_fraction,
        test_fraction=test_fraction,
        smoke_items=smoke_items,
        reserved_source_ids=reserved_source_ids,
    )
    if paths.data_root.exists():
        raise FileExistsError(
            f"refusing to overwrite split data directory: {paths.data_root}"
        )
    if paths.manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite split manifest: {paths.manifest_path}"
        )

    train = _normalise_official_records(
        official_train_records,
        official_split="train",
        dataset_revision=dataset_revision,
        image_dataset_revision=image_dataset_revision,
        image_dir=paths.image_dir,
        project_root=paths.project_root,
    )
    test = _normalise_official_records(
        official_test_records,
        official_split="test",
        dataset_revision=dataset_revision,
        image_dataset_revision=image_dataset_revision,
        image_dir=paths.image_dir,
        project_root=paths.project_root,
    )
    official_audit = _official_split_audit(train, test)
    _require_pinned_counts(
        dataset_revision,
        official_audit,
        PINNED_OFFICIAL_AUDIT,
        name="official split audit",
    )

    clean_records, excluded_records, merge_audit = _merge_and_exclude(
        (train, test)
    )
    _require_pinned_counts(
        dataset_revision,
        merge_audit,
        PINNED_MERGE_AUDIT,
        name="merge/exclusion audit",
    )

    reserved = tuple(
        sorted({canonical_text(str(value)) for value in reserved_source_ids})
    )
    clean_source_ids = {source_image_id(record) for record in clean_records}
    missing_reserved = sorted(set(reserved) - clean_source_ids)
    if missing_reserved:
        raise SplitBuildError(
            "reserved contract source IDs are absent after annotation "
            f"exclusions: {missing_reserved}"
        )
    reserved_records = [
        record
        for record in clean_records
        if source_image_id(record) in set(reserved)
    ]
    research_records = [
        record
        for record in clean_records
        if source_image_id(record) not in set(reserved)
    ]
    if not reserved_records:
        raise SplitBuildError("the contract reservation removed no records")

    partitions, source_assignments = _assign_source_groups(
        research_records,
        seed=seed,
        development_fraction=development_fraction,
        test_fraction=test_fraction,
    )
    primary_audit = assert_disjoint_source_images(partitions)
    _validate_primary_partitions(
        partitions,
        primary_audit=primary_audit,
        expected_source_ids={
            source_image_id(record) for record in research_records
        },
        reserved_source_ids=set(reserved),
        dataset_revision=dataset_revision,
    )
    smoke_records = select_smoke_records(
        partitions["development"],
        count=smoke_items,
        seed=seed,
    )

    paths.data_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.data_root.name}.",
            suffix=".tmp",
            dir=paths.data_root.parent,
        )
    )
    published = False
    try:
        files = {
            "official_train": staging_root / "official_train.jsonl",
            "official_test": staging_root / "official_test.jsonl",
            "train": staging_root / "train.jsonl",
            "development": staging_root / "development.jsonl",
            "test": staging_root / "test.jsonl",
            "smoke_20": staging_root / "smoke_20.jsonl",
            "excluded_annotation_conflicts": (
                staging_root / "excluded_annotation_conflicts.jsonl"
            ),
            "reserved_contract_records": (
                staging_root / "reserved_contract_records.jsonl"
            ),
            "split_audit": staging_root / "split_audit.json",
        }
        for name, records in (
            ("official_train", train),
            ("official_test", test),
            ("train", partitions["train"]),
            ("development", partitions["development"]),
            ("test", partitions["test"]),
            ("smoke_20", smoke_records),
            ("excluded_annotation_conflicts", excluded_records),
            ("reserved_contract_records", reserved_records),
        ):
            write_jsonl_atomic(files[name], records, sort_keys=True)

        audit_payload = {
            "schema_version": GROUPED_SPLIT_SCHEMA_VERSION,
            "status": "PASS",
            "official": official_audit,
            "merge_and_exclusions": merge_audit,
            "primary": primary_audit.as_dict(),
            "reserved_source_ids": list(reserved),
            "reserved_rows": len(reserved_records),
            "smoke": audit_records(smoke_records).as_dict(),
        }
        _write_json_atomic(files["split_audit"], audit_payload)

        final_files = {
            name: paths.data_root / file_path.name
            for name, file_path in files.items()
        }
        manifest = _build_manifest(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            image_dataset_id=image_dataset_id,
            image_dataset_revision=image_dataset_revision,
            seed=seed,
            development_fraction=development_fraction,
            test_fraction=test_fraction,
            smoke_items=smoke_items,
            paths=paths,
            staging_files=files,
            final_files=final_files,
            official_records={"train": train, "test": test},
            partitions=partitions,
            source_assignments=source_assignments,
            smoke_records=smoke_records,
            excluded_records=excluded_records,
            reserved_records=reserved_records,
            reserved_source_ids=reserved,
            official_audit=official_audit,
            merge_audit=merge_audit,
            primary_audit=primary_audit,
        )

        os.replace(staging_root, paths.data_root)
        published = True
        _fsync_directory(paths.data_root.parent)
        paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(paths.manifest_path, manifest)
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)

    verification = verify_grouped_split_artifacts(
        manifest_path=paths.manifest_path,
        project_root=paths.project_root,
    )
    return {
        "status": "PASS",
        "manifest": str(paths.manifest_path),
        "manifest_sha256": file_sha256(paths.manifest_path),
        "data_root": str(paths.data_root),
        "verification": verification,
    }


def load_official_records_from_hugging_face(
    *,
    dataset_id: str,
    dataset_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the two pinned official metadata splits without caching images."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SplitBuildError(
            "downloading split metadata requires the project data extra: "
            "pip install -e '.[data]'"
        ) from exc

    loaded: list[list[dict[str, Any]]] = []
    for split in ("train", "test"):
        dataset = load_dataset(
            dataset_id,
            split=split,
            revision=dataset_revision,
        )
        loaded.append([dict(row) for row in dataset])
    return loaded[0], loaded[1]


def load_official_records_from_jsonl(
    *,
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load caller-supplied official JSONL metadata."""

    return list(iter_jsonl(train_path)), list(iter_jsonl(test_path))


def materialize_grouped_split_artifacts(
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    image_dir: str | Path = "data/images",
    official_loader: (
        Callable[
            [str, str],
            tuple[list[dict[str, Any]], list[dict[str, Any]]],
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Reconstruct ignored split files against an existing tracked manifest.

    A clean clone contains the compact protocol manifest but not the generated
    JSONL files. Reconstruction builds into the manifest's recorded data
    directory, creates a disposable candidate manifest, and publishes success
    only when that candidate is byte-identical to the tracked lock.
    """

    root = Path(project_root).resolve()
    locked_manifest_path = _resolve_under(root, manifest_path)
    with locked_manifest_path.open("r", encoding="utf-8") as handle:
        locked_manifest = json.load(handle)
    if not isinstance(locked_manifest, dict):
        raise SplitBuildError("grouped split manifest root must be an object")
    if locked_manifest.get("schema_version") != GROUPED_SPLIT_SCHEMA_VERSION:
        raise SplitBuildError("unsupported grouped split manifest schema")
    if locked_manifest.get("status") != "PASS":
        raise SplitBuildError("grouped split manifest status is not PASS")

    artifacts = locked_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise SplitBuildError("grouped split manifest has no artifacts")
    artifact_paths = [
        _resolve_under(root, descriptor["path"])
        for descriptor in artifacts.values()
        if isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("path"), str)
    ]
    if len(artifact_paths) != len(artifacts):
        raise SplitBuildError("grouped split artifact descriptors are invalid")
    data_roots = {path.parent for path in artifact_paths}
    if len(data_roots) != 1:
        raise SplitBuildError(
            "grouped split artifacts do not share one data directory"
        )
    data_root = next(iter(data_roots))
    existing = [path for path in artifact_paths if path.exists()]
    if existing:
        if len(existing) != len(artifact_paths):
            missing = sorted(
                str(path) for path in artifact_paths if not path.exists()
            )
            raise SplitBuildError(
                "split artifact directory is incomplete; refusing to mix "
                f"reconstructed and existing files: {missing}"
            )
        result = verify_grouped_split_artifacts(
            manifest_path=locked_manifest_path,
            project_root=root,
        )
        result["materialized"] = False
        result["reused_artifacts"] = len(artifact_paths)
        return result
    if data_root.exists():
        raise SplitBuildError(
            f"split data directory exists without tracked artifacts: {data_root}"
        )

    dataset = locked_manifest.get("dataset")
    image_dataset = locked_manifest.get("image_dataset")
    algorithm = locked_manifest.get("algorithm")
    smoke = locked_manifest.get("smoke")
    reservation = locked_manifest.get("reservation")
    for name, value in (
        ("dataset", dataset),
        ("image_dataset", image_dataset),
        ("algorithm", algorithm),
        ("smoke", smoke),
        ("reservation", reservation),
    ):
        if not isinstance(value, Mapping):
            raise SplitBuildError(f"manifest {name} descriptor is invalid")

    loader = official_loader or (
        lambda dataset_id, revision: load_official_records_from_hugging_face(
            dataset_id=dataset_id,
            dataset_revision=revision,
        )
    )
    dataset_id = str(dataset["id"])
    dataset_revision = str(dataset["revision"])
    train_records, test_records = loader(dataset_id, dataset_revision)
    temporary_manifest = locked_manifest_path.with_name(
        f".{locked_manifest_path.name}.materializing"
    )
    if temporary_manifest.exists():
        raise FileExistsError(
            f"temporary materialisation manifest already exists: "
            f"{temporary_manifest}"
        )

    published_data = False
    try:
        build_grouped_splits(
            official_train_records=train_records,
            official_test_records=test_records,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            image_dataset_id=str(image_dataset["id"]),
            image_dataset_revision=str(image_dataset["revision"]),
            seed=int(algorithm["seed"]),
            development_fraction=float(algorithm["development_fraction"]),
            test_fraction=float(algorithm["test_fraction"]),
            smoke_items=int(smoke["count"]),
            paths=SplitBuildPaths.resolve(
                project_root=root,
                data_root=data_root,
                manifest_path=temporary_manifest,
                image_dir=image_dir,
            ),
            reserved_source_ids=tuple(reservation["source_img_ids"]),
        )
        published_data = True
        candidate_bytes = temporary_manifest.read_bytes()
        locked_bytes = locked_manifest_path.read_bytes()
        if candidate_bytes != locked_bytes:
            raise SplitBuildError(
                "reconstructed split manifest differs from the tracked lock: "
                f"expected {file_sha256(locked_manifest_path)}, observed "
                f"{file_sha256(temporary_manifest)}"
            )
    except Exception:
        if published_data and data_root.exists():
            shutil.rmtree(data_root)
        raise
    finally:
        temporary_manifest.unlink(missing_ok=True)

    result = verify_grouped_split_artifacts(
        manifest_path=locked_manifest_path,
        project_root=root,
    )
    result["materialized"] = True
    result["reused_artifacts"] = 0
    return result


def select_smoke_records(
    development_records: Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, metadata-balanced development-only smoke set."""

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise SplitBuildError("smoke item count must be a positive integer")
    candidates = [deepcopy(dict(record)) for record in development_records]
    source_ids = {source_image_id(record) for record in candidates}
    if len(source_ids) < count:
        raise SplitBuildError(
            f"smoke selection requires {count} source images; "
            f"development contains {len(source_ids)}"
        )

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    complexity_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    while len(selected) < count:
        available = [
            record
            for record in candidates
            if source_image_id(record) not in used_sources
        ]
        if not available:
            raise SplitBuildError("smoke selection exhausted unique sources")

        def selection_score(record: Mapping[str, Any]) -> tuple[Any, ...]:
            complexity, labels = _record_strata(record)
            class_load = sum(class_counts[label] for label in labels) / len(
                labels
            )
            tie_break = _seeded_digest(seed, stable_item_id(record))
            return (
                complexity_counts[complexity],
                class_load,
                max(class_counts[label] for label in labels),
                tie_break,
            )

        chosen = min(available, key=selection_score)
        source_id = source_image_id(chosen)
        used_sources.add(source_id)
        complexity, labels = _record_strata(chosen)
        complexity_counts[complexity] += 1
        for label in labels:
            class_counts[label] += 1
        metadata = chosen.setdefault("metadata", {})
        metadata["smoke_subset"] = True
        metadata["smoke_selection_rank"] = len(selected)
        chosen["item_id"] = stable_item_id(chosen)
        selected.append(chosen)

    return selected


def verify_grouped_split_artifacts(
    *,
    manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Re-run the grouped-split hard gates and verify all manifest hashes."""

    root = Path(project_root).resolve()
    manifest_file = _resolve_under(root, manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise SplitBuildError("grouped split manifest root must be an object")
    if manifest.get("schema_version") != GROUPED_SPLIT_SCHEMA_VERSION:
        raise SplitBuildError(
            "unsupported grouped split manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "PASS":
        raise SplitBuildError("grouped split manifest status is not PASS")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SplitBuildError("manifest artifacts must be a mapping")
    resolved_artifacts: dict[str, Path] = {}
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise SplitBuildError(f"artifact {name!r} descriptor is invalid")
        relative_path = descriptor.get("path")
        expected_sha256 = descriptor.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise SplitBuildError(f"artifact {name!r} path is invalid")
        path = _resolve_under(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise SplitBuildError(
                f"artifact hash mismatch for {name}: expected "
                f"{expected_sha256}, observed {observed_sha256}"
            )
        resolved_artifacts[str(name)] = path

    missing_primary = set(PRIMARY_PARTITIONS) - set(resolved_artifacts)
    if missing_primary:
        raise SplitBuildError(
            f"manifest is missing primary artifacts: {sorted(missing_primary)}"
        )
    partitions = {
        name: list(iter_jsonl(resolved_artifacts[name]))
        for name in PRIMARY_PARTITIONS
    }
    audit = assert_disjoint_source_images(partitions)
    partition_manifest = manifest.get("partitions")
    if not isinstance(partition_manifest, Mapping):
        raise SplitBuildError("manifest partitions must be a mapping")
    for name in PRIMARY_PARTITIONS:
        descriptor = partition_manifest.get(name)
        if not isinstance(descriptor, Mapping):
            raise SplitBuildError(
                f"manifest partition descriptor is missing: {name}"
            )
        records = partitions[name]
        observed_sources = sorted(
            {source_image_id(record) for record in records}
        )
        expected_sources = descriptor.get("source_img_ids")
        if observed_sources != expected_sources:
            raise SplitBuildError(
                f"{name} source IDs differ from the manifest"
            )
        if descriptor.get("rows") != len(records):
            raise SplitBuildError(f"{name} row count differs from the manifest")
        if descriptor.get("unique_item_ids") != len(
            {stable_item_id(record) for record in records}
        ):
            raise SplitBuildError(
                f"{name} unique item count differs from the manifest"
            )
        for record in records:
            _validate_partition_record(
                record,
                partition=name,
                dataset_revision=str(manifest["dataset"]["revision"]),
            )

    reserved = set(manifest.get("reservation", {}).get("source_img_ids", []))
    observed_primary_sources = {
        source_image_id(record)
        for records in partitions.values()
        for record in records
    }
    overlap_with_reserved = sorted(observed_primary_sources & reserved)
    if overlap_with_reserved:
        raise SplitBuildError(
            f"reserved source IDs entered research splits: {overlap_with_reserved}"
        )

    smoke_path = resolved_artifacts.get("smoke_20")
    if smoke_path is None:
        raise SplitBuildError("manifest is missing smoke_20 artifact")
    smoke_records = list(iter_jsonl(smoke_path))
    expected_smoke = manifest.get("smoke")
    if not isinstance(expected_smoke, Mapping):
        raise SplitBuildError("manifest smoke descriptor is invalid")
    observed_smoke_ids = [stable_item_id(record) for record in smoke_records]
    if observed_smoke_ids != expected_smoke.get("item_ids"):
        raise SplitBuildError("smoke item IDs differ from the manifest")
    development_ids = {
        stable_item_id(record) for record in partitions["development"]
    }
    if not set(observed_smoke_ids) <= development_ids:
        raise SplitBuildError("smoke items are not a subset of development")
    smoke_sources = [source_image_id(record) for record in smoke_records]
    if len(smoke_sources) != len(set(smoke_sources)):
        raise SplitBuildError("smoke items do not have unique source images")
    if len(smoke_records) != expected_smoke.get("count"):
        raise SplitBuildError("smoke item count differs from the manifest")

    return {
        "status": "PASS",
        "manifest": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "artifact_count": len(resolved_artifacts),
        "primary_audit": audit.as_dict(),
        "smoke_items": len(smoke_records),
        "reserved_source_ids": sorted(reserved),
    }


def _normalise_official_records(
    records: Iterable[Mapping[str, Any]],
    *,
    official_split: str,
    dataset_revision: str,
    image_dataset_revision: str,
    image_dir: Path,
    project_root: Path,
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for row_number, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, Mapping):
            raise SplitBuildError(
                f"{official_split} record {row_number} is not a mapping"
            )
        try:
            source_id = source_image_id(raw_record)
            question = question_text(raw_record)
            answer = _answer_text(raw_record)
        except (RecordFormatError, TypeError) as exc:
            raise SplitBuildError(
                f"{official_split} record {row_number}: {exc}"
            ) from exc
        metadata = raw_record.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        observed_revision = metadata.get("dataset_revision")
        if observed_revision not in (None, dataset_revision):
            raise SplitBuildError(
                f"{official_split} record {row_number} has dataset revision "
                f"{observed_revision!r}, expected {dataset_revision!r}"
            )
        complexity = (
            metadata.get("complexity")
            if "complexity" in metadata
            else raw_record.get("complexity")
        )
        question_class = (
            metadata.get("question_class")
            if "question_class" in metadata
            else raw_record.get("question_class")
        )
        original = (
            metadata.get("original")
            if "original" in metadata
            else raw_record.get("original")
        )
        image_path = image_dir / f"{source_id}.jpg"
        record = {
            "item_id": "",
            "messages": [
                {"role": "user", "content": f"<image>{question}"},
                {"role": "assistant", "content": answer},
            ],
            "images": [
                _portable_path(image_path, project_root=project_root)
            ],
            "metadata": {
                "img_id": source_id,
                "source_img_id": source_id,
                "official_split": official_split,
                "dataset_revision": dataset_revision,
                "image_dataset_revision": image_dataset_revision,
                "complexity": complexity,
                "question_class": _normalise_question_classes(
                    question_class
                ),
                "original": original,
                "image_variant": "original",
            },
        }
        record["item_id"] = stable_item_id(record)
        normalised.append(record)
    return normalised


def _merge_and_exclude(
    split_records: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    answers_by_question: defaultdict[tuple[str, str], set[str]] = defaultdict(
        set
    )
    metadata_conflict_keys: set[tuple[str, str, str]] = set()
    input_rows = 0
    exact_duplicates_removed = 0

    for records in split_records:
        for source_record in records:
            input_rows += 1
            record = deepcopy(dict(source_record))
            question_key = canonical_text(
                question_text(record),
                casefold=True,
            )
            answer_key = canonical_text(
                _answer_text(record),
                casefold=True,
            )
            source_id = source_image_id(record)
            image_question_key = (source_id, question_key)
            key = image_question_key + (answer_key,)
            answers_by_question[image_question_key].add(answer_key)
            official_origin = record["metadata"].get("official_split")
            if key in merged:
                exact_duplicates_removed += 1
                if _metadata_signature(
                    merged[key]["metadata"]
                ) != _metadata_signature(record["metadata"]):
                    metadata_conflict_keys.add(key)
                origins = merged[key]["metadata"].setdefault(
                    "official_origins",
                    [],
                )
                if official_origin not in origins:
                    origins.append(official_origin)
                continue
            record["metadata"]["official_origins"] = [official_origin]
            merged[key] = record

    conflicting_questions = {
        key
        for key, answers in answers_by_question.items()
        if len(answers) > 1
    }
    excluded: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for key, record in merged.items():
        reasons: list[str] = []
        if key[:2] in conflicting_questions:
            reasons.append("conflicting_answers_for_image_question")
        if key in metadata_conflict_keys:
            reasons.append("conflicting_duplicate_metadata")
        if reasons:
            record["metadata"]["exclusion_reasons"] = reasons
            excluded.append(record)
        else:
            clean.append(record)

    clean.sort(key=_record_sort_key)
    excluded.sort(key=_record_sort_key)
    merge_audit = {
        "input_rows": input_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_image_question_groups_removed": len(
            conflicting_questions
        ),
        "metadata_conflict_groups_removed": len(metadata_conflict_keys),
        "conflicting_rows_removed": len(excluded),
        "primary_rows_after_annotation_exclusions": len(clean),
        "primary_source_images_after_annotation_exclusions": len(
            {source_image_id(record) for record in clean}
        ),
    }
    return clean, excluded, merge_audit


def _assign_source_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    development_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, ...]]]:
    source_ids = sorted(
        {source_image_id(record) for record in records},
        key=lambda source_id: _seeded_digest(seed, source_id),
    )
    n_test = max(1, round(len(source_ids) * test_fraction))
    n_development = max(1, round(len(source_ids) * development_fraction))
    if n_test + n_development >= len(source_ids):
        raise SplitBuildError(
            "split fractions leave no source images for training"
        )
    assignments = {
        "test": tuple(sorted(source_ids[:n_test])),
        "development": tuple(
            sorted(source_ids[n_test : n_test + n_development])
        ),
        "train": tuple(sorted(source_ids[n_test + n_development :])),
    }
    source_to_partition = {
        source_id: partition
        for partition, ids in assignments.items()
        for source_id in ids
    }
    partitions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PRIMARY_PARTITIONS
    }
    for source_record in records:
        record = deepcopy(dict(source_record))
        partition = source_to_partition[source_image_id(record)]
        record["metadata"]["partition"] = partition
        partitions[partition].append(record)
    for partition_records in partitions.values():
        partition_records.sort(key=_record_sort_key)
    return partitions, assignments


def _validate_primary_partitions(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    primary_audit: SplitAuditReport,
    expected_source_ids: set[str],
    reserved_source_ids: set[str],
    dataset_revision: str,
) -> None:
    if not primary_audit.is_source_disjoint:
        raise SplitBuildError("primary partitions are not source disjoint")
    observed_source_ids: set[str] = set()
    observed_item_ids: set[str] = set()
    for partition, records in partitions.items():
        if not records:
            raise SplitBuildError(f"{partition} partition is empty")
        for record in records:
            _validate_partition_record(
                record,
                partition=partition,
                dataset_revision=dataset_revision,
            )
            source_id = source_image_id(record)
            item_id = stable_item_id(record)
            if item_id in observed_item_ids:
                raise SplitBuildError(
                    f"duplicate stable item ID across primary splits: {item_id}"
                )
            observed_item_ids.add(item_id)
            observed_source_ids.add(source_id)
    if observed_source_ids != expected_source_ids:
        raise SplitBuildError(
            "primary source coverage differs from the cleaned research source "
            "set"
        )
    overlap = observed_source_ids & reserved_source_ids
    if overlap:
        raise SplitBuildError(
            f"reserved sources entered primary partitions: {sorted(overlap)}"
        )


def _validate_partition_record(
    record: Mapping[str, Any],
    *,
    partition: str,
    dataset_revision: str,
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SplitBuildError("partition record metadata is missing")
    if metadata.get("partition") != partition:
        raise SplitBuildError(
            f"record partition metadata differs: expected {partition!r}, "
            f"observed {metadata.get('partition')!r}"
        )
    if metadata.get("dataset_revision") != dataset_revision:
        raise SplitBuildError(
            "record dataset revision differs from the manifest"
        )
    observed_item_id = record.get("item_id")
    expected_item_id = stable_item_id(record)
    if observed_item_id != expected_item_id:
        raise SplitBuildError(
            f"record item_id differs from stable identity: {observed_item_id!r}"
        )


def _build_manifest(
    *,
    dataset_id: str,
    dataset_revision: str,
    image_dataset_id: str,
    image_dataset_revision: str,
    seed: int,
    development_fraction: float,
    test_fraction: float,
    smoke_items: int,
    paths: SplitBuildPaths,
    staging_files: Mapping[str, Path],
    final_files: Mapping[str, Path],
    official_records: Mapping[str, Sequence[Mapping[str, Any]]],
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    source_assignments: Mapping[str, Sequence[str]],
    smoke_records: Sequence[Mapping[str, Any]],
    excluded_records: Sequence[Mapping[str, Any]],
    reserved_records: Sequence[Mapping[str, Any]],
    reserved_source_ids: Sequence[str],
    official_audit: Mapping[str, Any],
    merge_audit: Mapping[str, Any],
    primary_audit: SplitAuditReport,
) -> dict[str, Any]:
    artifacts = {
        name: {
            "path": _portable_path(path, project_root=paths.project_root),
            "sha256": file_sha256(staging_files[name]),
            "bytes": staging_files[name].stat().st_size,
        }
        for name, path in sorted(final_files.items())
    }
    partition_descriptors = {}
    for name in PRIMARY_PARTITIONS:
        records = partitions[name]
        source_ids = list(source_assignments[name])
        audit = audit_records(records)
        partition_descriptors[name] = {
            "rows": len(records),
            "unique_source_images": audit.unique_source_images,
            "unique_item_ids": audit.unique_item_ids,
            "source_img_ids_sha256": canonical_json_sha256(source_ids),
            "source_img_ids": source_ids,
            "complexity_counts": _complexity_counts(records),
            "question_class_counts": _question_class_counts(records),
        }

    smoke_item_ids = [stable_item_id(record) for record in smoke_records]
    smoke_source_ids = [source_image_id(record) for record in smoke_records]
    manifest = {
        "schema_version": GROUPED_SPLIT_SCHEMA_VERSION,
        "status": "PASS",
        "study": "study1",
        "dataset": {
            "id": dataset_id,
            "revision": dataset_revision,
            "official_splits": ["train", "test"],
            "primary_source": "union_after_annotation_exclusions",
        },
        "image_dataset": {
            "id": image_dataset_id,
            "revision": image_dataset_revision,
            "image_paths_verified": False,
        },
        "algorithm": {
            "id": GROUPED_SPLIT_ALGORITHM,
            "group": "source_img_id",
            "seed": seed,
            "source_order": "ascending_sha256(seed + NUL + source_img_id)",
            "test_assignment": "first round(N * test_fraction) source IDs",
            "development_assignment": (
                "next round(N * development_fraction) source IDs"
            ),
            "train_assignment": "remaining source IDs",
            "development_fraction": development_fraction,
            "test_fraction": test_fraction,
            "deduplication": (
                "canonical source-image/question/answer; retain one exact row"
            ),
            "annotation_exclusion": (
                "exclude every image-question group with conflicting "
                "canonical answers and exact duplicates with conflicting "
                "complexity/question_class/original metadata"
            ),
        },
        "official_audit": dict(official_audit),
        "merge_and_exclusions": dict(merge_audit),
        "reservation": {
            "reason": "one-item backend contract fixture",
            "source_img_ids": list(reserved_source_ids),
            "rows": len(reserved_records),
            "item_ids": sorted(
                stable_item_id(record) for record in reserved_records
            ),
        },
        "partitions": partition_descriptors,
        "primary_audit": primary_audit.as_dict(),
        "smoke": {
            "algorithm": SMOKE_SELECTION_ALGORITHM,
            "partition": "development",
            "metadata_only_selection": True,
            "count": smoke_items,
            "unique_source_images": len(set(smoke_source_ids)),
            "item_ids": smoke_item_ids,
            "source_img_ids": smoke_source_ids,
            "complexity_counts": _complexity_counts(smoke_records),
            "question_class_counts": _question_class_counts(smoke_records),
        },
        "record_counts": {
            "official_train": len(official_records["train"]),
            "official_test": len(official_records["test"]),
            "excluded_annotation_conflicts": len(excluded_records),
            "reserved_contract_records": len(reserved_records),
        },
        "artifacts": artifacts,
    }
    return manifest


def _official_split_audit(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_sources = {source_image_id(record) for record in train}
    test_sources = {source_image_id(record) for record in test}
    train_iq = {_image_question_key(record) for record in train}
    test_iq = {_image_question_key(record) for record in test}
    train_iqa = {_image_question_answer_key(record) for record in train}
    test_iqa = {_image_question_answer_key(record) for record in test}
    return {
        "official_train_rows": len(train),
        "official_test_rows": len(test),
        "official_train_source_images": len(train_sources),
        "official_test_source_images": len(test_sources),
        "overlapping_source_images": len(train_sources & test_sources),
        "test_image_overlap_fraction": (
            len(train_sources & test_sources) / len(test_sources)
            if test_sources
            else 0.0
        ),
        "overlapping_image_question_pairs": len(train_iq & test_iq),
        "overlapping_image_question_answer_triples": len(
            train_iqa & test_iqa
        ),
    }


def _answer_text(record: Mapping[str, Any]) -> str:
    top_level = record.get("answer")
    if isinstance(top_level, str) and canonical_text(top_level):
        return canonical_text(top_level)
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
            ):
                answer = canonical_text(message["content"])
                if answer:
                    return answer
    raise RecordFormatError(
        "record has no non-empty answer; expected answer or an assistant message"
    )


def _normalise_question_classes(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted(
        {
            canonical_text(str(item))
            for item in values
            if canonical_text(str(item))
        }
    )


def _record_strata(record: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    complexity = canonical_text(str(metadata.get("complexity", "unlabelled")))
    labels = tuple(
        _normalise_question_classes(metadata.get("question_class"))
        or ["unlabelled"]
    )
    return complexity, labels


def _complexity_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        complexity, _labels = _record_strata(record)
        counts[complexity] += 1
    return dict(sorted(counts.items()))


def _question_class_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        _complexity, labels = _record_strata(record)
        counts.update(labels)
    return dict(sorted(counts.items()))


def _metadata_signature(metadata: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "complexity": metadata.get("complexity"),
            "question_class": metadata.get("question_class"),
            "original": metadata.get("original"),
        }
    )


def _image_question_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        source_image_id(record),
        canonical_text(question_text(record), casefold=True),
    )


def _image_question_answer_key(
    record: Mapping[str, Any],
) -> tuple[str, str, str]:
    return _image_question_key(record) + (
        canonical_text(_answer_text(record), casefold=True),
    )


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        source_image_id(record),
        canonical_text(question_text(record), casefold=True),
        canonical_text(_answer_text(record), casefold=True),
        stable_item_id(record),
    )


def _seeded_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _require_pinned_counts(
    dataset_revision: str,
    observed: Mapping[str, Any],
    expected: Mapping[str, int],
    *,
    name: str,
) -> None:
    if dataset_revision != PINNED_DATASET_REVISION:
        return
    mismatches = {
        field: {"expected": value, "observed": observed.get(field)}
        for field, value in expected.items()
        if observed.get(field) != value
    }
    if mismatches:
        raise SplitBuildError(
            f"pinned {name} counts changed: {mismatches}"
        )


def _validate_build_arguments(
    *,
    dataset_id: str,
    dataset_revision: str,
    image_dataset_id: str,
    image_dataset_revision: str,
    seed: int,
    development_fraction: float,
    test_fraction: float,
    smoke_items: int,
    reserved_source_ids: Sequence[str],
) -> None:
    for name, value in (
        ("dataset_id", dataset_id),
        ("dataset_revision", dataset_revision),
        ("image_dataset_id", image_dataset_id),
        ("image_dataset_revision", image_dataset_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise SplitBuildError(f"{name} must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SplitBuildError("seed must be an integer")
    if (
        not 0 < development_fraction < 1
        or not 0 < test_fraction < 1
        or development_fraction + test_fraction >= 1
    ):
        raise SplitBuildError(
            "development and test fractions must be positive and sum to less "
            "than one"
        )
    if not isinstance(smoke_items, int) or isinstance(smoke_items, bool):
        raise SplitBuildError("smoke_items must be an integer")
    if smoke_items < 1:
        raise SplitBuildError("smoke_items must be positive")
    if not reserved_source_ids:
        raise SplitBuildError("at least one reserved source ID is required")


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _portable_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "CONTRACT_RESERVED_SOURCE_IDS",
    "GROUPED_SPLIT_ALGORITHM",
    "GROUPED_SPLIT_SCHEMA_VERSION",
    "PINNED_DATASET_REVISION",
    "SMOKE_SELECTION_ALGORITHM",
    "SplitBuildError",
    "SplitBuildPaths",
    "build_grouped_splits",
    "load_official_records_from_hugging_face",
    "load_official_records_from_jsonl",
    "select_smoke_records",
    "verify_grouped_split_artifacts",
]
