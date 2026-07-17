"""Canonical record accessors and stable identifiers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any
import unicodedata


ITEM_ID_NAMESPACE = "gi-vqa-item-v1"
_IMAGE_TOKEN_PATTERN = re.compile(r"^\s*<image>\s*", flags=re.IGNORECASE)


class RecordFormatError(ValueError):
    """Raised when a GI-VQA record does not contain a required field."""


def canonical_text(text: str, *, casefold: bool = False) -> str:
    """Apply Unicode normalisation and collapse runs of whitespace."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, received {type(text).__name__}")
    value = unicodedata.normalize("NFKC", text)
    value = " ".join(value.split())
    return value.casefold() if casefold else value


def source_image_id(record: Mapping[str, Any]) -> str:
    """Return the parent source-image ID using an explicit precedence order."""

    metadata = record.get("metadata")
    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        candidates.extend((metadata.get("source_img_id"), metadata.get("img_id")))
    candidates.extend(
        (
            record.get("source_img_id"),
            record.get("img_id"),
            record.get("image_id"),
        )
    )
    for candidate in candidates:
        if candidate is None:
            continue
        value = canonical_text(str(candidate))
        if value:
            return value
    raise RecordFormatError(
        "record has no non-empty source image identifier; expected "
        "metadata.source_img_id, metadata.img_id, source_img_id, img_id, or image_id"
    )


def question_text(record: Mapping[str, Any]) -> str:
    """Return a canonical question from a top-level field or user message."""

    top_level = record.get("question")
    if isinstance(top_level, str) and canonical_text(top_level):
        return canonical_text(_IMAGE_TOKEN_PATTERN.sub("", top_level, count=1))

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                question = canonical_text(
                    _IMAGE_TOKEN_PATTERN.sub("", content, count=1)
                )
                if question:
                    return question
    raise RecordFormatError(
        "record has no non-empty question; expected question or a user message"
    )


def stable_item_id(
    record: Mapping[str, Any],
    *,
    length: int = 20,
    namespace: str = ITEM_ID_NAMESPACE,
) -> str:
    """Hash the source image and canonical question into a stable item ID."""

    if not 8 <= length <= 64:
        raise ValueError("length must be between 8 and 64 hexadecimal characters")
    payload = json.dumps(
        [namespace, source_image_id(record), question_text(record)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
