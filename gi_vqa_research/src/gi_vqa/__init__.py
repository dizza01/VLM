"""Lightweight research infrastructure for leakage-safe GI-VQA experiments."""

from .audit import (
    SplitAudit,
    SplitAuditReport,
    SplitLeakageError,
    assert_disjoint_source_images,
    audit_jsonl_splits,
    audit_records,
)
from .config import ConfigError, config_sha256, load_config, validate_config
from .identifiers import (
    RecordFormatError,
    canonical_text,
    question_text,
    source_image_id,
    stable_item_id,
)
from .jsonl import JsonlDecodeError, iter_jsonl, read_jsonl, write_jsonl_atomic
from .provenance import (
    MANIFEST_SCHEMA_VERSION,
    build_run_manifest,
    canonical_json_sha256,
    file_fingerprint,
    file_sha256,
    load_run_manifest,
    write_run_manifest,
)
from .shards import (
    DuplicateRecordIdError,
    ExpectedIdMismatchError,
    OutputExistsError,
    RecordIdError,
    ShardMergeReport,
    ShardValidation,
    ShardValidationError,
    merge_jsonl_shards_atomic,
    validate_jsonl_shards,
)

__all__ = [
    "JsonlDecodeError",
    "MANIFEST_SCHEMA_VERSION",
    "RecordFormatError",
    "RecordIdError",
    "ConfigError",
    "SplitAudit",
    "SplitAuditReport",
    "SplitLeakageError",
    "ShardMergeReport",
    "ShardValidation",
    "ShardValidationError",
    "DuplicateRecordIdError",
    "ExpectedIdMismatchError",
    "OutputExistsError",
    "assert_disjoint_source_images",
    "audit_jsonl_splits",
    "audit_records",
    "build_run_manifest",
    "canonical_json_sha256",
    "canonical_text",
    "config_sha256",
    "file_fingerprint",
    "file_sha256",
    "iter_jsonl",
    "load_config",
    "load_run_manifest",
    "question_text",
    "read_jsonl",
    "merge_jsonl_shards_atomic",
    "source_image_id",
    "stable_item_id",
    "validate_jsonl_shards",
    "validate_config",
    "write_jsonl_atomic",
    "write_run_manifest",
]
