"""File fingerprints and minimal, immutable run manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Optional, Union


MANIFEST_SCHEMA_VERSION = "gi-vqa-run-manifest-v1"


def file_sha256(
    path: Union[str, Path], *, chunk_size: int = 1024 * 1024
) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Union[str, Path]) -> dict[str, Any]:
    """Return a portable fingerprint for one regular file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a regular file: {file_path}")
    return {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "sha256": file_sha256(file_path),
    }


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a canonical encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_manifest(
    *,
    run_id: str,
    stage: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, Union[str, Path]],
    command: Optional[Sequence[str]] = None,
    code_revision: Optional[str] = None,
    environment: Optional[Mapping[str, Any]] = None,
    created_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    """Build a self-contained manifest before a research stage is executed."""

    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not stage.strip():
        raise ValueError("stage must be non-empty")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input names must be unique")

    config_copy = _json_round_trip(config)
    input_fingerprints = {
        name: file_fingerprint(path) for name, path in sorted(inputs.items())
    }
    created_at = created_at_utc or datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage,
        "created_at_utc": created_at,
        "code_revision": code_revision,
        "command": list(command or []),
        "config": config_copy,
        "config_sha256": canonical_json_sha256(config_copy),
        "inputs": input_fingerprints,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **_json_round_trip(environment or {}),
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def write_run_manifest(
    path: Union[str, Path], manifest: Mapping[str, Any]
) -> Path:
    """Validate and atomically write a run manifest."""

    _validate_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                dict(manifest),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def load_run_manifest(path: Union[str, Path]) -> dict[str, Any]:
    """Load and validate a run manifest, including its content hash."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("run manifest must be a JSON object")
    _validate_manifest(value)
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "stage",
        "created_at_utc",
        "config",
        "config_sha256",
        "inputs",
        "runtime",
        "manifest_content_sha256",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"run manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest['schema_version']!r}")
    if canonical_json_sha256(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("run manifest config hash does not match its config")
    content = dict(manifest)
    saved_hash = content.pop("manifest_content_sha256")
    if canonical_json_sha256(content) != saved_hash:
        raise ValueError("run manifest content hash does not match its contents")


def _json_round_trip(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
