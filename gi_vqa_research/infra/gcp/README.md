# GCP GPU Execution Guide

This guide implements the local VS Code to VS Code Remote SSH to GCP GPU VM
workflow described in the project infrastructure README.

The scripts here deliberately do **not** create projects, buckets, service
accounts, firewall rules, VMs, or GPUs. Those are consequential administrative
operations and should be reviewed and performed explicitly. The scripts also
do not delete local or cloud data.

## Files

- `env.example` documents non-secret environment variables.
- `bootstrap_vm.sh` validates an existing VM and optionally creates local work
  directories and installs from a dependency lock.
- `run_job.sh` creates a run manifest and launches a command inside `tmux`.
- `sync_run.sh` synchronises one run directory to one GCS prefix without a
  delete operation.

All three scripts are dry-run by default. Pass `--apply` only after reviewing
the displayed values.

## One-time administrative setup

An administrator should prepare:

1. A GCP project with billing and Compute Engine enabled.
2. A GPU quota in the selected region and zone.
3. A regional GCS bucket for research artifacts.
4. A GPU VM or VM template with a compatible NVIDIA driver.
5. A VM service account with only the required bucket permissions.
6. Budget alerts and, if appropriate, a maximum-spend policy.
7. OS Login or another reviewed SSH access mechanism.

Prefer Application Default Credentials and the VM service account for GCS.
Store Hugging Face and W&B credentials in a secret manager or an interactive
credential store. Never put them in `env.example`, Git, shell history, notebook
cells, startup scripts, or VM images.

## Local VS Code and Remote SSH

1. Commit and push the code that should run.
2. Start the reviewed GPU VM.
3. Connect with VS Code Remote SSH.
4. Clone the repository on the VM, or fetch and check out the exact commit.
5. Confirm that `git status --short` is empty for an authoritative run.
6. Run the bootstrap script from the repository checkout.

The local and remote checkouts should share history through Git. Avoid editing
the same uncommitted file independently on both machines.

The helpers distinguish the Git repository root from the Python project root.
When `gi_vqa_research/pyproject.toml` is present, commands run from
`gi_vqa_research/`. Set `PROJECT_ROOT` only if a different layout is required.

## Configure a shell session

Copy values from `env.example` into your shell profile, a non-versioned local
file, or an approved environment manager. These values are identifiers and
paths, not credentials.

Example:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="your-zone"
export GCP_VM_NAME="your-vm-name"
export GCS_RUN_ROOT="gs://your-bucket/runs/study1"
export GI_VQA_WORK_ROOT="$HOME/gi-vqa-work"
```

Do not `source` a file that contains tokens into a command that prints or logs
the complete environment.

## Validate and bootstrap an existing VM

Preview:

```bash
bash gi_vqa_research/infra/gcp/bootstrap_vm.sh
```

Create the cache/run directories:

```bash
bash gi_vqa_research/infra/gcp/bootstrap_vm.sh --apply
```

Dependency installation is a separate opt-in:

```bash
INSTALL_DEPS=1 bash gi_vqa_research/infra/gcp/bootstrap_vm.sh --apply
```

The bootstrap script accepts a committed `uv.lock`, or a requirements lock
selected through `DEPENDENCY_LOCK_FILE`. It intentionally refuses to treat an
unpinned `requirements.txt` as an authoritative lock.

## Launch a job

Set a unique, non-secret run identifier and pass the actual research command
after `--`.

Preview:

```bash
RUN_ID="study1-paired-<gitsha>-s42-<utc>" \
  bash gi_vqa_research/infra/gcp/run_job.sh \
  --config configs/study1/pilot.yaml -- \
  python -m gi_vqa.train --config configs/study1/pilot.yaml
```

Launch:

```bash
RUN_ID="study1-paired-<gitsha>-s42-<utc>" \
GCS_RUN_ROOT="gs://your-bucket/runs/study1" \
  bash gi_vqa_research/infra/gcp/run_job.sh --apply \
  --config configs/study1/pilot.yaml -- \
  python -m gi_vqa.train --config configs/study1/pilot.yaml
```

The runner:

- refuses a dirty worktree by default;
- refuses to overwrite an existing local run directory;
- writes a small provenance manifest;
- uses the package's tamper-detecting manifest format;
- launches a detached `tmux` session;
- captures standard output and error in the run directory;
- attempts a final non-deleting GCS sync when `GCS_RUN_ROOT` is set;
- leaves automatic shutdown disabled unless explicitly enabled.

Never pass a token as a command-line argument. Use the approved credential
mechanism of the application instead.

Monitor with:

```bash
tmux list-sessions
tmux attach -t "$RUN_ID"
tail -f "$GI_VQA_WORK_ROOT/runs/$RUN_ID/job.log"
```

Detach from `tmux` with `Ctrl-b`, then `d`.

## Synchronise artifacts manually

Preview:

```bash
RUN_ID="<run-id>" \
GCS_RUN_ROOT="gs://your-bucket/runs/study1" \
  bash gi_vqa_research/infra/gcp/sync_run.sh
```

Apply:

```bash
RUN_ID="<run-id>" \
GCS_RUN_ROOT="gs://your-bucket/runs/study1" \
  bash gi_vqa_research/infra/gcp/sync_run.sh --apply
```

The sync script does not use a cloud-side delete flag. GCS object versioning is
still recommended because repeated checkpoint synchronisation may update an
object with the same name.

## Safe shutdown

By default, inspect the completed run, verify its GCS destination, and stop the
VM from the local control plane:

```bash
gcloud compute instances stop "$GCP_VM_NAME" \
  --project "$GCP_PROJECT_ID" \
  --zone "$GCP_ZONE"
```

The runner supports `AUTO_SHUTDOWN=1`, but will request shutdown only when the
research command succeeds and the final sync succeeds. It is still an explicit
opt-in and depends on the VM user's reviewed `sudo` policy. Do not enable it
during initial infrastructure testing.

## Confirmatory Study 1 run

Before a locked evaluation:

- tag the clean commit;
- freeze and hash the protocol, grouped split, faithfulness subset, dependency
  lock or container digest, and adapter revision;
- use an on-demand VM;
- set no sampling override;
- write to a new GCS run prefix;
- retain raw predictions and attribution artifacts;
- verify the artifact manifest before stopping the VM.

The helper scripts enforce only basic operational checks. The Study 1 evaluator
must enforce the scientific protocol.
