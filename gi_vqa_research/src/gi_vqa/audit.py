"""Split summaries and hard leakage gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

from .identifiers import source_image_id, stable_item_id
from .jsonl import iter_jsonl


class SplitLeakageError(RuntimeError):
    """Raised when source images appear in more than one data partition."""

    def __init__(self, overlaps: Mapping[str, tuple[str, ...]]) -> None:
        self.overlaps = dict(overlaps)
        preview = "; ".join(
            f"{pair}: {len(image_ids)} shared ({', '.join(image_ids[:5])})"
            for pair, image_ids in self.overlaps.items()
        )
        super().__init__(f"source-image leakage detected: {preview}")


@dataclass(frozen=True)
class SplitAudit:
    """Compact integrity summary for one split."""

    rows: int
    unique_source_images: int
    unique_item_ids: int
    duplicate_item_rows: int
    max_questions_per_source_image: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAuditReport:
    """Audits and source-ID overlaps for a set of named splits."""

    splits: dict[str, SplitAudit]
    overlaps: dict[str, tuple[str, ...]]

    @property
    def is_source_disjoint(self) -> bool:
        return not self.overlaps

    def assert_source_disjoint(self) -> None:
        if self.overlaps:
            raise SplitLeakageError(self.overlaps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "splits": {
                name: audit.as_dict() for name, audit in sorted(self.splits.items())
            },
            "overlaps": {
                pair: list(image_ids) for pair, image_ids in sorted(self.overlaps.items())
            },
            "is_source_disjoint": self.is_source_disjoint,
        }


def audit_records(records: Iterable[Mapping[str, Any]]) -> SplitAudit:
    """Summarise record and source-image identity integrity."""

    row_count = 0
    source_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for record in records:
        row_count += 1
        source_counts[source_image_id(record)] += 1
        item_counts[stable_item_id(record)] += 1

    return SplitAudit(
        rows=row_count,
        unique_source_images=len(source_counts),
        unique_item_ids=len(item_counts),
        duplicate_item_rows=sum(count - 1 for count in item_counts.values()),
        max_questions_per_source_image=max(source_counts.values(), default=0),
    )


def assert_disjoint_source_images(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> SplitAuditReport:
    """Audit named splits and fail if any source image crosses a boundary."""

    if len(splits) < 2:
        raise ValueError("at least two named splits are required")

    split_audits: dict[str, SplitAudit] = {}
    source_sets: dict[str, set[str]] = {}
    for split_name, records in splits.items():
        if not split_name:
            raise ValueError("split names must be non-empty")
        materialized = list(records)
        split_audits[split_name] = audit_records(materialized)
        source_sets[split_name] = {
            source_image_id(record) for record in materialized
        }

    overlaps = _pairwise_overlaps(source_sets)
    report = SplitAuditReport(splits=split_audits, overlaps=overlaps)
    report.assert_source_disjoint()
    return report


def audit_jsonl_splits(
    split_paths: Mapping[str, Union[str, Path]],
    *,
    hard_gate: bool = True,
) -> SplitAuditReport:
    """Audit JSONL split files, optionally enforcing source-image disjointness."""

    if len(split_paths) < 2:
        raise ValueError("at least two named split paths are required")

    split_audits: dict[str, SplitAudit] = {}
    source_sets: dict[str, set[str]] = {}
    for split_name, path in split_paths.items():
        records = list(iter_jsonl(path))
        split_audits[split_name] = audit_records(records)
        source_sets[split_name] = {source_image_id(record) for record in records}

    report = SplitAuditReport(
        splits=split_audits,
        overlaps=_pairwise_overlaps(source_sets),
    )
    if hard_gate:
        report.assert_source_disjoint()
    return report


def _pairwise_overlaps(
    source_sets: Mapping[str, set[str]],
) -> dict[str, tuple[str, ...]]:
    overlaps: dict[str, tuple[str, ...]] = {}
    split_names = sorted(source_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            shared = tuple(sorted(source_sets[left_name] & source_sets[right_name]))
            if shared:
                overlaps[f"{left_name}<->{right_name}"] = shared
    return overlaps
