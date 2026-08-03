"""Fail when tracked repository content is unsafe or non-portable to publish."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

MAX_TRACKED_BYTES = 1_000_000
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".jpg",
    ".jpeg",
    ".npz",
    ".parquet",
    ".pem",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".zip",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "predictions",
}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PRIVATE_PATH_PATTERNS = {
    "macOS user path": re.compile(r"/Users/[^/\s]+/"),
    "old parent repository": re.compile(
        r"https://github\.com/dizza01/VLM(?:\.git)?|/content/VLM"
    ),
}


class PublicationCheckError(RuntimeError):
    pass


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    ]


def check_tree(root: Path) -> list[str]:
    failures: list[str] = []
    files = tracked_files(root)
    for path in files:
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            failures.append(f"generated path is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"binary/artifact suffix is tracked: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(
                f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {relative}"
            )
        if path.suffix.lower() == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("outputs"):
                    failures.append(f"notebook output: {relative} cell {index}")
                if cell.get("execution_count") is not None:
                    failures.append(
                        f"notebook execution count: {relative} cell {index}"
                    )
        if _is_text(path):
            text = path.read_text(encoding="utf-8")
            failures.extend(_scan_text(text, str(relative), SECRET_PATTERNS))
            if relative != Path("scripts/publication_check.py"):
                failures.extend(
                    _scan_text(text, str(relative), PRIVATE_PATH_PATTERNS)
                )
    return failures


def check_history(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--all", "-p", "--no-ext-diff", "--no-textconv"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return _scan_text(result.stdout, "Git history", SECRET_PATTERNS)


def _scan_text(text: str, source: str, patterns) -> list[str]:
    return [
        f"{name} candidate in {source}"
        for name, pattern in patterns.items()
        if pattern.search(text)
    ]


def _is_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    failures = check_tree(root)
    if args.history:
        failures.extend(check_history(root))
    if failures:
        for failure in sorted(set(failures)):
            print(f"FAIL: {failure}")
        return 1
    print(
        f"Publication check PASS: {len(tracked_files(root))} tracked files; "
        f"history_scan={args.history}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
