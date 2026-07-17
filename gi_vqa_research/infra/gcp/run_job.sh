#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  RUN_ID=<unique-id> run_job.sh [--apply] --config <path> -- <research command> [args...]

The default is a dry run. With --apply, write a run manifest and launch the
command in a detached tmux session.

Optional:
  GI_VQA_WORK_ROOT   Defaults to $HOME/gi-vqa-work
  GCS_RUN_ROOT       If set, sync the run after completion
  ALLOW_DIRTY=1      Permit a dirty worktree for development only
  AUTO_SHUTDOWN=1    Request VM shutdown only after job and sync succeed

Never pass credentials as command-line arguments.
EOF
}

APPLY=0
CONFIG_PATH=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --config)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --config requires a path." >&2
        exit 2
      }
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      break
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${1:-}" == "--" ]] || {
  usage >&2
  exit 2
}
shift
[[ "$#" -gt 0 ]] || {
  echo "ERROR: a research command is required after --." >&2
  exit 2
}
[[ -n "$CONFIG_PATH" ]] || {
  echo "ERROR: --config is required for every research run." >&2
  exit 2
}

RUN_ID="${RUN_ID:-}"
[[ -n "$RUN_ID" ]] || {
  echo "ERROR: RUN_ID is required." >&2
  exit 1
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 1
}

command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is required." >&2
  exit 1
}
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
[[ -n "$REPO_ROOT" && -d "$REPO_ROOT/.git" ]] || {
  echo "ERROR: run from a Git checkout or set REPO_ROOT." >&2
  exit 1
}
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
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi
[[ -f "$CONFIG_PATH" ]] || {
  echo "ERROR: configuration does not exist: $CONFIG_PATH" >&2
  exit 1
}

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
DIRTY_STATUS="$(git -C "$REPO_ROOT" status --short)"
if [[ -n "$DIRTY_STATUS" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "ERROR: worktree is dirty. Commit the run state or set ALLOW_DIRTY=1 for a development run." >&2
  exit 1
fi

WORK_ROOT="${GI_VQA_WORK_ROOT:-$HOME/gi-vqa-work}"
RUN_DIR="$WORK_ROOT/runs/$RUN_ID"
SYNC_SCRIPT="$REPO_ROOT/gi_vqa_research/infra/gcp/sync_run.sh"
COMMAND=("$@")
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
COMMAND_TEXT="${COMMAND_TEXT% }"

echo "Run ID: $RUN_ID"
echo "Repository: $REPO_ROOT"
echo "Python project: $PROJECT_ROOT"
echo "Git commit: $GIT_COMMIT"
echo "Git short SHA: $GIT_SHORT"
echo "Worktree dirty: $([[ -n "$DIRTY_STATUS" ]] && echo true || echo false)"
echo "Run directory: $RUN_DIR"
echo "Configuration: $CONFIG_PATH"
echo "Command: $COMMAND_TEXT"
echo "GCS run root: ${GCS_RUN_ROOT:-<not set>}"
echo "Automatic shutdown: ${AUTO_SHUTDOWN:-0}"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: pass --apply before -- to launch."
  exit 0
fi

command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is required for session-independent execution." >&2
  exit 1
}
[[ ! -e "$RUN_DIR" ]] || {
  echo "ERROR: refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
}
tmux has-session -t "$RUN_ID" 2>/dev/null && {
  echo "ERROR: tmux session already exists: $RUN_ID" >&2
  exit 1
}

MANIFEST_ARGS=(
  manifest
  --config "$CONFIG_PATH"
  --run-dir "$RUN_DIR"
  --run-id "$RUN_ID"
)
for token in "${COMMAND[@]}"; do
  MANIFEST_ARGS+=("--record-command=$token")
done
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m gi_vqa.cli "${MANIFEST_ARGS[@]}"
chmod 700 "$RUN_DIR"

LAUNCH_SCRIPT="$RUN_DIR/launch.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'RUN_ID=%q\n' "$RUN_ID"
  printf 'RUN_DIR=%q\n' "$RUN_DIR"
  printf 'GCS_RUN_ROOT=%q\n' "${GCS_RUN_ROOT:-}"
  printf 'SYNC_SCRIPT=%q\n' "$SYNC_SCRIPT"
  printf 'AUTO_SHUTDOWN=%q\n' "${AUTO_SHUTDOWN:-0}"
  printf 'JOB_COMMAND=('
  printf '%q ' "${COMMAND[@]}"
  echo ')'
  cat <<'EOF'

set +e
"${JOB_COMMAND[@]}" > >(tee "$RUN_DIR/job.log") 2>&1
job_status=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"run_id":"%s","finished_at_utc":"%s","job_exit_code":%d}\n' \
  "$RUN_ID" "$finished_at" "$job_status" > "$RUN_DIR/job_complete.json"

sync_status=0
if [[ -n "$GCS_RUN_ROOT" ]]; then
  RUN_ID="$RUN_ID" LOCAL_RUN_DIR="$RUN_DIR" GCS_RUN_ROOT="$GCS_RUN_ROOT" \
    bash "$SYNC_SCRIPT" --apply || sync_status=$?
else
  echo "GCS_RUN_ROOT is not set; final artifact sync was skipped." | tee -a "$RUN_DIR/job.log"
  sync_status=1
fi

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  if [[ "$job_status" == "0" && "$sync_status" == "0" ]]; then
    echo "Job and sync succeeded; requesting VM shutdown." | tee -a "$RUN_DIR/job.log"
    sudo shutdown -h now
  else
    echo "Automatic shutdown suppressed because job or sync failed." | tee -a "$RUN_DIR/job.log"
  fi
fi

if [[ "$job_status" != "0" ]]; then
  exit "$job_status"
fi
exit "$sync_status"
EOF
} > "$LAUNCH_SCRIPT"
chmod 700 "$LAUNCH_SCRIPT"

tmux new-session -d -s "$RUN_ID" bash "$LAUNCH_SCRIPT"
echo "Launched tmux session: $RUN_ID"
echo "Attach: tmux attach -t $RUN_ID"
echo "Log: $RUN_DIR/job.log"
