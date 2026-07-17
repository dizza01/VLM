#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_vm.sh [--apply]

Validate an existing GCP GPU VM. With --apply, create VM-local work
directories. Set INSTALL_DEPS=1 to also install from a committed dependency
lock. This script does not provision cloud resources or install system drivers.
EOF
}

APPLY=0
case "${1:-}" in
  "")
    ;;
  --apply)
    APPLY=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is required." >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is required." >&2
  exit 1
}

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/.git" ]]; then
  echo "ERROR: run this script from a Git checkout or set REPO_ROOT." >&2
  exit 1
fi
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -f "$REPO_ROOT/gi_vqa_research/pyproject.toml" ]]; then
    PROJECT_ROOT="$REPO_ROOT/gi_vqa_research"
  else
    PROJECT_ROOT="$REPO_ROOT"
  fi
fi
[[ -d "$PROJECT_ROOT" ]] || {
  echo "ERROR: project root does not exist: $PROJECT_ROOT" >&2
  exit 1
}

WORK_ROOT="${GI_VQA_WORK_ROOT:-$HOME/gi-vqa-work}"
CACHE_ROOT="$WORK_ROOT/cache"
RUNS_ROOT="$WORK_ROOT/runs"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

echo "Repository: $REPO_ROOT"
echo "Python project: $PROJECT_ROOT"
echo "Git commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$REPO_ROOT" status --short)" ]]; then
  echo "WARNING: the checkout is dirty; do not use it for a confirmatory run."
else
  echo "Git worktree: clean"
fi
echo "Python: $(python3 --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
else
  echo "WARNING: nvidia-smi is unavailable; GPU jobs will not run."
fi

if command -v gcloud >/dev/null 2>&1; then
  echo "gcloud: $(gcloud version 2>/dev/null | head -n 1)"
else
  echo "WARNING: gcloud is unavailable; GCS sync helpers cannot run."
fi

echo "Work root: $WORK_ROOT"
echo "Cache root: $CACHE_ROOT"
echo "Runs root: $RUNS_ROOT"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: pass --apply to create directories."
  if [[ "$INSTALL_DEPS" == "1" ]]; then
    echo "DRY RUN: dependency installation was requested but not performed."
  fi
  exit 0
fi

mkdir -p "$CACHE_ROOT" "$RUNS_ROOT"
chmod 700 "$WORK_ROOT" "$CACHE_ROOT" "$RUNS_ROOT"
echo "Created VM-local work directories."

if [[ "$INSTALL_DEPS" != "1" ]]; then
  echo "Dependency installation skipped. Set INSTALL_DEPS=1 to opt in."
  exit 0
fi

cd "$PROJECT_ROOT"
if [[ -f pyproject.toml && -f uv.lock ]]; then
  command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv.lock exists but uv is not installed." >&2
    exit 1
  }
  echo "Installing the frozen uv environment."
  uv sync --frozen --extra gpu
elif [[ -n "${DEPENDENCY_LOCK_FILE:-}" ]]; then
  LOCK_PATH="$DEPENDENCY_LOCK_FILE"
  if [[ "$LOCK_PATH" != /* ]]; then
    LOCK_PATH="$PROJECT_ROOT/$LOCK_PATH"
  fi
  [[ -f "$LOCK_PATH" ]] || {
    echo "ERROR: dependency lock not found: $LOCK_PATH" >&2
    exit 1
  }
  if [[ "$(basename "$LOCK_PATH")" == "requirements.txt" ]]; then
    echo "ERROR: refusing unpinned requirements.txt as an authoritative lock." >&2
    exit 1
  fi
  python3 -m venv "$PROJECT_ROOT/.venv"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$LOCK_PATH"
else
  echo "ERROR: no uv.lock or DEPENDENCY_LOCK_FILE was supplied." >&2
  exit 1
fi

echo "Bootstrap complete."
