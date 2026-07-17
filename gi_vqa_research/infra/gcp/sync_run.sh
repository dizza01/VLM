#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: sync_run.sh [--apply]

Synchronise one VM-local run directory to GCS. The default is a dry run.

Required:
  RUN_ID             Unique run identifier
  GCS_RUN_ROOT       GCS prefix, for example gs://bucket/runs/study1

Optional:
  GI_VQA_WORK_ROOT   Defaults to $HOME/gi-vqa-work
  LOCAL_RUN_DIR      Overrides the default run directory

This script never requests deletion of cloud-side objects.
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

RUN_ID="${RUN_ID:-}"
GCS_RUN_ROOT="${GCS_RUN_ROOT:-}"
WORK_ROOT="${GI_VQA_WORK_ROOT:-$HOME/gi-vqa-work}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$WORK_ROOT/runs/$RUN_ID}"

[[ -n "$RUN_ID" ]] || {
  echo "ERROR: RUN_ID is required." >&2
  exit 1
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 1
}
[[ "$GCS_RUN_ROOT" == gs://* ]] || {
  echo "ERROR: GCS_RUN_ROOT must begin with gs://." >&2
  exit 1
}
[[ -d "$LOCAL_RUN_DIR" ]] || {
  echo "ERROR: local run directory does not exist: $LOCAL_RUN_DIR" >&2
  exit 1
}

DESTINATION="${GCS_RUN_ROOT%/}/$RUN_ID"
echo "Local run: $LOCAL_RUN_DIR"
echo "Destination: $DESTINATION"
echo "Operation: non-deleting recursive sync"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: pass --apply to perform the sync."
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || {
  echo "ERROR: gcloud is required." >&2
  exit 1
}

gcloud storage rsync --recursive "$LOCAL_RUN_DIR" "$DESTINATION"

SYNCED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"run_id":"%s","destination":"%s","synced_at_utc":"%s"}\n' \
  "$RUN_ID" "$DESTINATION" "$SYNCED_AT" > "$LOCAL_RUN_DIR/sync_complete.json"
gcloud storage cp "$LOCAL_RUN_DIR/sync_complete.json" \
  "$DESTINATION/sync_complete.json"

echo "Sync complete: $DESTINATION"

