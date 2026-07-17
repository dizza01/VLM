#!/usr/bin/env python3
"""Capture the code, configuration and environment before a run starts."""

from __future__ import annotations

import argparse

from gi_vqa.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stage")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--record-command", action="append", default=[])
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    forwarded = [
        "manifest",
        "--config",
        args.config,
        "--run-dir",
        args.run_dir,
    ]
    for name in ("run_id", "stage"):
        value = getattr(args, name)
        if value:
            forwarded.extend([f"--{name.replace('_', '-')}", value])
    for value in args.input:
        forwarded.extend(["--input", value])
    for value in args.record_command:
        forwarded.append(f"--record-command={value}")
    if args.require_clean_git:
        forwarded.append("--require-clean-git")
    return cli_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
