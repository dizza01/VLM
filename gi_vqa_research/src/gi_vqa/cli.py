"""Small command-line surface for infrastructure checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .audit import SplitLeakageError, audit_jsonl_splits
from .config import ConfigError, config_sha256, load_config, validate_config
from .jsonl import iter_jsonl
from .provenance import build_run_manifest, canonical_json_sha256, write_run_manifest
from .shards import merge_jsonl_shards_atomic


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ConfigError,
        SplitLeakageError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
    ) as exc:
        parser.error(str(exc))
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gi-vqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config-check", help="validate a study config")
    config_parser.add_argument("--config", required=True, type=Path)
    config_parser.add_argument(
        "--resolved",
        action="store_true",
        help="also reject unresolved confirmatory placeholders",
    )
    config_parser.add_argument(
        "--model-execution",
        action="store_true",
        help="require the complete shared-backend execution contract",
    )
    config_parser.set_defaults(handler=_config_check)

    audit_parser = subparsers.add_parser("audit", help="audit JSONL source-image splits")
    audit_parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each named split",
    )
    audit_parser.add_argument("--report", type=Path)
    audit_parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="report overlaps instead of treating them as a hard failure",
    )
    audit_parser.set_defaults(handler=_audit)

    manifest_parser = subparsers.add_parser(
        "manifest", help="capture an immutable pre-run manifest"
    )
    manifest_parser.add_argument("--config", required=True, type=Path)
    manifest_parser.add_argument("--run-dir", required=True, type=Path)
    manifest_parser.add_argument("--run-id")
    manifest_parser.add_argument("--stage")
    manifest_parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    manifest_parser.add_argument(
        "--record-command",
        action="append",
        default=[],
        metavar="TOKEN",
        help="repeat once per token of the research command being recorded",
    )
    manifest_parser.add_argument("--require-clean-git", action="store_true")
    manifest_parser.set_defaults(handler=_manifest)

    merge_parser = subparsers.add_parser(
        "merge-shards", help="validate and atomically merge JSONL result shards"
    )
    merge_parser.add_argument("--shard", action="append", required=True, type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument("--id-field", required=True)
    merge_parser.add_argument("--expected-jsonl", type=Path)
    merge_parser.add_argument("--expected-id-field")
    merge_parser.add_argument("--report", type=Path)
    merge_parser.set_defaults(handler=_merge_shards)
    return parser


def _config_check(args: argparse.Namespace) -> int:
    config = validate_config(
        load_config(args.config),
        require_resolved=args.resolved,
        require_model_execution=args.model_execution,
    )
    output = {
        "config": str(args.config),
        "profile": config["profile"],
        "sha256": config_sha256(config),
        "resolved_check": bool(args.resolved or args.model_execution),
        "model_execution_check": bool(args.model_execution),
    }
    print(json.dumps(output, indent=2))
    return 0


def _audit(args: argparse.Namespace) -> int:
    split_paths = _parse_name_paths(args.split)
    report = audit_jsonl_splits(split_paths, hard_gate=not args.allow_overlap)
    payload = json.dumps(report.as_dict(), indent=2)
    print(payload)
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.is_source_disjoint else 1


def _manifest(args: argparse.Namespace) -> int:
    config = validate_config(load_config(args.config))
    git = _git_state(Path.cwd())
    require_clean = bool(args.require_clean_git or config["execution"].get("require_clean_git"))
    if require_clean and git["dirty"]:
        raise ConfigError("this run requires a clean Git working tree")

    stage = args.stage or str(config["execution"].get("stage") or config["profile"])
    created = datetime.now(timezone.utc)  # noqa: UP017
    run_id = args.run_id or _default_run_id(
        stage=stage,
        config_digest=config_sha256(config),
        commit=git.get("commit"),
        created=created,
    )
    run_dir = args.run_dir
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {manifest_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty run directory: {run_dir}")

    inputs = {"config_file": args.config, **_parse_name_paths(args.input)}
    environment = {
        "git": git,
        "hostname": platform.node(),
        "executable": sys.executable,
        "packages": _package_versions(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_smi(),
        "container_image_digest": os.getenv("GI_VQA_CONTAINER_DIGEST"),
    }
    command = list(args.record_command) or [
        sys.executable,
        "-m",
        "gi_vqa.cli",
        "manifest",
        "--config",
        str(args.config),
        "--run-dir",
        str(args.run_dir),
    ]
    manifest = build_run_manifest(
        run_id=run_id,
        stage=stage,
        config=config,
        inputs=inputs,
        command=command,
        code_revision=git.get("commit"),
        environment=environment,
        created_at_utc=created.isoformat(),
    )
    write_run_manifest(manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "manifest": str(manifest_path)}, indent=2))
    return 0


def _merge_shards(args: argparse.Namespace) -> int:
    expected_ids = None
    if args.expected_jsonl:
        expected_field = args.expected_id_field or args.id_field
        expected_ids = []
        for row_number, record in enumerate(iter_jsonl(args.expected_jsonl), start=1):
            value = record.get(expected_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{args.expected_jsonl}:record-{row_number}: "
                    f"{expected_field!r} must be a non-empty string"
                )
            expected_ids.append(value)
    report = merge_jsonl_shards_atomic(
        args.shard,
        args.output,
        id_field=args.id_field,
        expected_ids=expected_ids,
    )
    payload = json.dumps(report.as_dict(), indent=2)
    print(payload)
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0


def _parse_name_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, received: {value!r}")
        name, raw_path = value.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"expected non-empty NAME=PATH, received: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate input name: {name}")
        parsed[name] = Path(raw_path)
    return parsed


def _git_state(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists() and not _inside_git_worktree(path):
        return {"commit": None, "dirty": None, "status_sha256": None}
    commit = _run_git(path, "rev-parse", "HEAD").strip()
    status = _run_git(path, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": canonical_json_sha256(status.splitlines()),
    }


def _inside_git_worktree(path: Path) -> bool:
    try:
        return _run_git(path, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except RuntimeError:
        return False


def _run_git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "datasets",
        "ms-swift",
        "bitsandbytes",
        "peft",
        "accelerate",
        "wandb",
        "numpy",
        "Pillow",
        "PyYAML",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _nvidia_smi() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return {"error": completed.stderr.strip()}
    return {"gpus": [line for line in completed.stdout.splitlines() if line.strip()]}


def _default_run_id(
    *,
    stage: str,
    config_digest: str,
    commit: str | None,
    created: datetime,
) -> str:
    commit_part = (commit or "nogit")[:8]
    timestamp = created.strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}-{commit_part}-cfg{config_digest[:8]}-{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
