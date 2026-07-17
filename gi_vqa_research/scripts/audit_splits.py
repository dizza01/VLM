#!/usr/bin/env python3
"""Convenience wrapper for source-image leakage audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gi_vqa.audit import SplitLeakageError, audit_jsonl_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="+", type=Path)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if len(args.splits) < 2:
        parser.error("provide at least two JSONL split paths")

    names: dict[str, Path] = {}
    for index, path in enumerate(args.splits):
        name = path.stem
        if name in names:
            name = f"{name}-{index}"
        names[name] = path
    try:
        report = audit_jsonl_splits(names, hard_gate=not args.allow_overlap)
    except SplitLeakageError as exc:
        parser.error(str(exc))
    payload = json.dumps(report.as_dict(), indent=2)
    print(payload)
    if args.report:
        if args.report.exists():
            parser.error(f"refusing to overwrite {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.is_source_disjoint else 1


if __name__ == "__main__":
    raise SystemExit(main())

