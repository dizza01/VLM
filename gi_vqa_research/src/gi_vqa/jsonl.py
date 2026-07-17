"""Strict, atomic JSON Lines utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Union


class JsonlDecodeError(ValueError):
    """Raised when a JSONL row is malformed or is not a JSON object."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


def iter_jsonl(path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON object rows and report errors with line numbers."""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JsonlDecodeError(jsonl_path, line_number, exc.msg) from exc
            if not isinstance(value, dict):
                raise JsonlDecodeError(
                    jsonl_path,
                    line_number,
                    f"expected a JSON object, received {type(value).__name__}",
                )
            yield value


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    return list(iter_jsonl(path))


def write_jsonl_atomic(
    path: Union[str, Path],
    records: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
) -> Path:
    """Write records atomically, leaving an existing destination intact on failure."""

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
            for row_number, record in enumerate(records, start=1):
                if not isinstance(record, Mapping):
                    raise TypeError(
                        f"record {row_number} must be a mapping, "
                        f"received {type(record).__name__}"
                    )
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=sort_keys,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        _fsync_directory(output_path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync so the rename is durable on POSIX filesystems."""

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
