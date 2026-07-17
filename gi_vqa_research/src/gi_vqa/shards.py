"""Validation and atomic, no-clobber merging for JSONL result shards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Union

from .jsonl import iter_jsonl
from .provenance import file_fingerprint


class ShardValidationError(ValueError):
    """Base class for invalid sharded result data."""


class RecordIdError(ShardValidationError):
    """Raised when a row lacks a valid string record identifier."""


class DuplicateRecordIdError(ShardValidationError):
    """Raised when the same identifier occurs more than once."""

    def __init__(
        self,
        record_id: str,
        first_location: str,
        duplicate_location: str,
    ) -> None:
        self.record_id = record_id
        self.first_location = first_location
        self.duplicate_location = duplicate_location
        super().__init__(
            f"duplicate record ID {record_id!r}: first at {first_location}, "
            f"again at {duplicate_location}"
        )


class ExpectedIdMismatchError(ShardValidationError):
    """Raised when observed IDs do not exactly match the expected ID set."""

    def __init__(
        self,
        *,
        missing_ids: Iterable[str],
        extra_ids: Iterable[str],
    ) -> None:
        self.missing_ids = tuple(sorted(missing_ids))
        self.extra_ids = tuple(sorted(extra_ids))
        parts = []
        if self.missing_ids:
            parts.append(
                f"{len(self.missing_ids)} missing "
                f"({', '.join(self.missing_ids[:5])})"
            )
        if self.extra_ids:
            parts.append(
                f"{len(self.extra_ids)} extra "
                f"({', '.join(self.extra_ids[:5])})"
            )
        super().__init__("expected-ID mismatch: " + "; ".join(parts))


class OutputExistsError(FileExistsError):
    """Raised when a merge destination already exists."""


@dataclass(frozen=True)
class ShardValidation:
    """Successful validation summary for a collection of shards."""

    shard_paths: tuple[str, ...]
    id_field: str
    record_count: int
    expected_id_count: Optional[int]
    observed_ids_sha256: str

    @property
    def shard_count(self) -> int:
        return len(self.shard_paths)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shard_count"] = self.shard_count
        return result


@dataclass(frozen=True)
class ShardMergeReport:
    """Successful merge summary including the published file fingerprint."""

    validation: ShardValidation
    output: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation": self.validation.as_dict(),
            "output": dict(self.output),
        }


def validate_jsonl_shards(
    shard_paths: Sequence[Union[str, Path]],
    *,
    id_field: str,
    expected_ids: Optional[Iterable[str]] = None,
) -> ShardValidation:
    """Validate uniqueness and, when supplied, exact expected-ID coverage."""

    _validate_id_field(id_field)
    paths = _normalise_shard_paths(shard_paths)
    expected = _normalise_expected_ids(expected_ids)
    seen: dict[str, str] = {}
    record_count = 0
    for path in paths:
        for record_number, record in enumerate(iter_jsonl(path), start=1):
            record_count += 1
            _observe_record_id(
                record,
                id_field=id_field,
                location=f"{path}:record-{record_number}",
                seen=seen,
            )
    _assert_expected_ids(set(seen), expected)
    return _validation_summary(paths, id_field, record_count, seen, expected)


def merge_jsonl_shards_atomic(
    shard_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    id_field: str,
    expected_ids: Optional[Iterable[str]] = None,
) -> ShardMergeReport:
    """Stream, validate, and atomically publish shards without overwriting."""

    _validate_id_field(id_field)
    paths = _normalise_shard_paths(shard_paths)
    expected = _normalise_expected_ids(expected_ids)
    destination = Path(output_path)
    if destination.exists():
        raise OutputExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".merge-tmp",
    )
    temporary_path = Path(temporary_name)
    seen: dict[str, str] = {}
    record_count = 0

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for path in paths:
                for record_number, record in enumerate(iter_jsonl(path), start=1):
                    record_count += 1
                    _observe_record_id(
                        record,
                        id_field=id_field,
                        location=f"{path}:record-{record_number}",
                        seen=seen,
                    )
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        _assert_expected_ids(set(seen), expected)
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise OutputExistsError(
                f"refusing to overwrite existing output: {destination}"
            ) from exc
        _fsync_directory(destination.parent)

        validation = _validation_summary(
            paths, id_field, record_count, seen, expected
        )
        return ShardMergeReport(
            validation=validation,
            output=file_fingerprint(destination),
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalise_shard_paths(
    shard_paths: Sequence[Union[str, Path]],
) -> tuple[Path, ...]:
    if isinstance(shard_paths, (str, bytes, Path)):
        raise TypeError("shard_paths must be a sequence of paths, not one path")
    paths = tuple(sorted((Path(path) for path in shard_paths), key=str))
    if not paths:
        raise ValueError("at least one shard path is required")
    if len(set(paths)) != len(paths):
        raise ValueError("shard_paths contains the same path more than once")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"shard is not a regular file: {path}")
    return paths


def _normalise_expected_ids(
    expected_ids: Optional[Iterable[str]],
) -> Optional[set[str]]:
    if expected_ids is None:
        return None
    values = list(expected_ids)
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("expected_ids contains duplicate values")
    return set(values)


def _observe_record_id(
    record: Mapping[str, Any],
    *,
    id_field: str,
    location: str,
    seen: dict[str, str],
) -> None:
    value = record.get(id_field)
    if not isinstance(value, str) or not value.strip():
        raise RecordIdError(
            f"{location}: {id_field!r} must be a non-empty string"
        )
    record_id = value
    if record_id in seen:
        raise DuplicateRecordIdError(record_id, seen[record_id], location)
    seen[record_id] = location


def _validate_id_field(id_field: str) -> None:
    if not isinstance(id_field, str) or not id_field.strip():
        raise ValueError("id_field must be a non-empty string")


def _assert_expected_ids(
    observed: set[str],
    expected: Optional[set[str]],
) -> None:
    if expected is None:
        return
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ExpectedIdMismatchError(missing_ids=missing, extra_ids=extra)


def _validation_summary(
    paths: tuple[Path, ...],
    id_field: str,
    record_count: int,
    seen: Mapping[str, str],
    expected: Optional[set[str]],
) -> ShardValidation:
    digest = hashlib.sha256()
    for record_id in sorted(seen):
        encoded = record_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return ShardValidation(
        shard_paths=tuple(str(path) for path in paths),
        id_field=id_field,
        record_count=record_count,
        expected_id_count=len(expected) if expected is not None else None,
        observed_ids_sha256=digest.hexdigest(),
    )


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
