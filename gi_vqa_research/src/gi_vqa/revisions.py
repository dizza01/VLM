"""Shared validation for immutable external-artifact revisions."""

from __future__ import annotations

import re

MOVING_REVISIONS = frozenset({"main", "master", "latest", "head"})
UNRESOLVED_REVISION_PLACEHOLDERS = frozenset(
    {
        "required",
        "placeholder",
        "tbd",
        "todo",
        "replace-me",
        "replace_me",
    }
)
_COMMIT_OR_CONTENT_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


def validate_immutable_revision(
    value: str | None,
    field: str,
    require_resolved: bool = True,
    require_commit: bool = True,
) -> str:
    """Return a validated immutable revision.

    Moving branch references are never accepted. Placeholder values remain
    available to legacy, non-resolved configuration templates, while executable
    configurations can additionally require a Git SHA-1 or SHA-256/content
    digest represented by exactly 40 or 64 hexadecimal characters.
    """

    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty string")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty immutable revision")

    revision = value.strip()
    normalized = revision.casefold()
    if normalized in MOVING_REVISIONS or normalized.startswith("refs/heads/"):
        raise ValueError(
            f"{field} must identify an immutable revision, not moving revision {value!r}"
        )
    if require_resolved and normalized in UNRESOLVED_REVISION_PLACEHOLDERS:
        raise ValueError(f"{field} is an unresolved revision placeholder")
    if require_commit and (
        revision != value or _COMMIT_OR_CONTENT_REVISION.fullmatch(revision) is None
    ):
        raise ValueError(f"{field} must be a 40- or 64-character hexadecimal immutable revision")
    return revision


__all__ = [
    "MOVING_REVISIONS",
    "UNRESOLVED_REVISION_PLACEHOLDERS",
    "validate_immutable_revision",
]
