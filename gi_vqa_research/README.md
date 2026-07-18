# GI-VQA Research Infrastructure

This directory documents the execution infrastructure for the GI visual
question-answering research programme. It is intentionally separate from the
current notebooks and does not replace the Study 1 protocol.

The recommended workflow is:

```text
Local VS Code
  |-- writing, Git, configuration, tests, small CPU audits
  |
  `-- VS Code Remote SSH --> stoppable GCP GPU VM
                                |-- training and inference
                                |-- attribution and perturbation jobs
                                |-- W&B monitoring plus local logs
                                `-- GCS artifact synchronisation

Final LoRA adapters and model cards --> Hugging Face Hub
Colab                              --> smoke tests only
```

## Responsibilities

### Local VS Code: control plane

Use the local checkout for research design, source control, configuration,
review, lightweight tests, and analysis of already-produced results. Git is the
transfer mechanism between machines; do not maintain independent local and VM
copies by manual file copying.

### GCP GPU VM: authoritative compute plane

Use VS Code Remote SSH to work with a checkout on the VM. Long jobs must run in
`tmux`, a service, or another session-independent runner so that closing VS Code
does not terminate them. Treat the VM's disk as a cache, not as the only copy of
an experiment.

Use development data while iterating. Run a confirmatory test evaluation only
from a clean, tagged commit with a locked protocol, immutable model and dataset
revisions, and a frozen software environment.

### Colab: disposable smoke-test environment

Colab is useful for checking that a small batch can load and run. It is not the
authoritative environment for paper results because runtimes are ephemeral and
their packages and hardware can drift. A Colab smoke test should clone an exact
Git commit and use the same dependency lock as the GCP workflow.

### Notebooks and scripts

Notebooks should display data, inspect failures, and generate report figures.
Training, inference, attribution, perturbation, calibration, and metric
calculation should ultimately be callable as scripts or package entry points.
This avoids hidden notebook state and makes jobs resumable and testable.

## Durable stores

| Store | Purpose | Do not use it for |
|---|---|---|
| Git | Code, tests, small configs, protocol and split manifests | Datasets, checkpoints, saliency arrays, secrets |
| Google Cloud Storage | Checkpoints, predictions, attribution maps, raw logs, run manifests | Live source editing |
| Hugging Face Hub | Final LoRA adapters, immutable revisions, model cards | Intermediate checkpoints |
| Weights & Biases | Live monitoring and run comparison | The sole copy of evidence or artifacts |
| VM disk | Download and processing cache | Long-term retention |

Keep the GCS bucket in the same region as the GPU VM. Enable object versioning
or another appropriate retention mechanism for important result prefixes, and
use lifecycle rules for replaceable intermediate checkpoints.

## Run lifecycle

Every substantive run should follow the same lifecycle:

1. Select a clean Git commit and a named configuration.
2. Record the configuration hash, Git commit, dependency/container identity,
   dataset and model revisions, seeds, and hardware in a run manifest.
3. Launch the job independently of the SSH session.
4. Monitor through W&B and an ordinary log file.
5. Save checkpoints frequently enough to survive a pre-emption or disconnect.
6. Synchronise the run directory to a unique GCS prefix.
7. Verify the upload before stopping the VM.
8. Promote only selected final adapters to the Hugging Face Hub.

A useful run identifier is:

```text
study1-<condition>-<git-sha>-<seed>-<UTC timestamp>
```

Do not reuse a run identifier. Do not put tokens, passwords, signed URLs, or
other credentials in a run name, command line, configuration, log, or manifest.

## Evaluation isolation

Development and confirmatory evaluation should be separate commands and
separate artifact prefixes.

- Model selection and calibration use the grouped development split.
- Training code must not accept the held-out test path.
- The locked evaluator should require a clean Git tree, the expected protocol
  hash, a complete test split, and immutable adapter and dataset revisions.
- A protocol change after test inspection becomes a new disclosed protocol
  version; it must not silently replace the first result.
- Raw predictions are retained even when later summary code changes.

## Cost and shutdown rules

- Start a GPU VM only for GPU work and stop it when the verified upload is
  complete.
- Remember that persistent disks and retained snapshots still cost money while
  a VM is stopped.
- Test checkpoint resume and artifact upload before using Spot capacity.
- Prefer Spot capacity for recoverable pilots and on-demand capacity for the
  final locked evaluation.
- Configure project budget alerts separately from these scripts.
- Automatic shutdown is disabled by default. It may be enabled only when the
  job runner can prove that both the job and final artifact sync succeeded.

## GCP helper scaffolds

The conservative helper scripts and detailed setup guide are in
[`infra/gcp/`](infra/gcp/README.md). They do not provision or delete cloud
resources, contain no secrets, and default to describing work rather than
performing it.

## Implemented foundation

The first migration slice is executable:

- strict, atomic JSONL readers and writers;
- stable source-image and question item identifiers;
- source-image split audits with leakage hard gates;
- versioned smoke, pilot and confirmatory configurations;
- confirmatory configuration safety checks;
- tamper-detecting run manifests with Git and environment provenance;
- restart-safe validation and no-overwrite merging of JSONL shards;
- an immutable PaliGemma model specification and dependency-light backend contract;
- a lazy Transformers/PEFT PaliGemma backend for deterministic generation,
  fixed-answer scoring, decoder answer-to-image attention and answer-conditioned
  vision-layer Grad-CAM;
- exact generated-token reproduction and generation/teacher-forcing score
  parity checks before attribution;
- min-max float32 attribution output plus source-image and processed-tensor
  fingerprints;
- a one-item, package-level Colab T4 contract runner with a thin launch
  notebook, fixed diagnostic fixture, an isolated ms-swift 3.7.0 boundary-bug
  check, project training-template equivalence checks and downloadable
  evidence bundle;
- a guarded `gi-vqa-train`/`python -m gi_vqa.training` entrypoint that forces
  the versioned corrected PaliGemma template instead of raw `swift sft`;
- a deterministic `prepare-splits` builder that unions the pinned official
  metadata, removes ambiguous annotations, reserves the contract fixture,
  assigns complete source-image groups and selects a development-only smoke-20;
- an independent `split-check` hard gate covering source leakage, stable item
  identity, tracked assignments and every generated artifact hash;
- strict shared-backend smoke configuration validation;
- dry-run-first GCP bootstrap, detached-job and GCS-sync helpers;
- standard-library unit tests.

The authoritative project interpreter is Python 3.11.13. The lightweight
foundation tests are also intentionally compatible with the repository
machine's older local interpreter.

Useful local checks from this directory:

```bash
make test
PYTHONPATH=src python3 -m gi_vqa.cli \
  config-check --config configs/study1/smoke.yaml --model-execution
```

Audit prepared splits:

```bash
PYTHONPATH=src python3 -m gi_vqa.cli prepare-splits \
  --config configs/study1/smoke.yaml --project-root .
PYTHONPATH=src python3 -m gi_vqa.cli split-check \
  --manifest protocols/study1/grouped_split_manifest.json --project-root .
```

See [`MIGRATION.md`](MIGRATION.md) for the mapping from the existing Study 1
notebook to the new modules.

## Deliberately not implemented yet

This scaffold does not yet run the experiment end to end. The shared PaliGemma
backend and corrected training-template boundary passed the revised Colab T4
contract. The grouped split builder has now generated the pinned tracked
manifest, and its independent source-leakage and artifact-integrity gate passed.
Complete training orchestration, image caching, per-item restart-safe stage
storage, perturbation generation, metrics and reporting still need to be
extracted. The next gate is image caching followed by a restart-safe 20-item
development run. The base-model contract validates plumbing; an immutable Study
adapter smoke follows after training.
