# RQ1 full-baseline implementation decision

## Scope

This phase addresses only Research Question 1:

> How well does a fine-tuned VLM answer GI endoscopy VQA questions on the full
> Kvasir-VQA-x1 test split?

Clinician validation, causal visual-faithfulness claims, and the dedicated
confidence/failure-mode study are deferred. Earlier attention, Grad-CAM,
calibration and error-audit work remains development infrastructure and does
not answer RQ2 or RQ3.

## Current state

`rq1_full_baseline_protocol_v1.draft.json` is intentionally marked
`DRAFT_REVIEW_REQUIRED`. Validation confirms its pinned data, model and compute
profiles without materialising or reading the test artifact. It does not
authorise test access.

The draft proposes:

- PaliGemma 3B at its immutable revision;
- one full pass over all 143,594 official training records;
- QLoRA with rank 16 over all linear modules;
- an unfrozen vision tower and aligner so visual adaptation is not excluded by
  construction;
- effective batch size 16 and 8,975 optimizer steps;
- full-test comparison of the frozen fine-tuned adapter with the immutable
  unadapted base model;
- 15,955 completions per condition;
- correctness metrics and breakdowns by question class and complexity.

These are reviewable proposed choices, not hidden CLI defaults. Changing one
requires a new versioned profile or protocol and new hashes.

## Compute decision

Use a persistent GCP GPU VM for the authoritative run.

The full workflow includes approximately 143,594 training examples, frequent
checkpoints, checkpoint resume testing, 31,910 test generations across two
conditions, durable prediction storage and report production. Colab runtimes
are too ephemeral for this to be the paper-result execution environment.

Recommended first profile:

- one NVIDIA L4 with at least 22 GB usable GPU memory;
- persistent VM disk for live checkpoints and Hugging Face cache;
- `tmux` execution through `infra/gcp/run_job.sh`;
- periodic, non-deleting synchronization to a unique versioned GCS prefix;
- Spot capacity only after checkpoint resume has passed;
- on-demand capacity for the frozen full-test benchmark.

The L4 choice must pass a 32-record, one-step memory/throughput smoke test.
If it cannot sustain the locked configuration, move to A100 80 GB and record a
new compute-profile hash. Do not silently reduce the scientific batch or model
configuration to fit hardware.

Colab remains useful for the same 32-record non-authoritative smoke test only.

## Larger-model support

Model choices are isolated in versioned model profiles. To test another or
larger model:

1. Copy the model profile and assign a new `profile_id`.
2. Pin immutable model and processor revisions.
3. Specify image size, precision, quantization, sequence length and batch
   parameters.
4. Select and hash a compute profile with adequate memory.
5. Implement and test a backend when the architecture is not PaliGemma.
6. Pass the same smoke, checkpoint, deterministic inference and report
   contracts.

This avoids embedding one model's parameters throughout notebooks and runners.
It does not pretend that an unimplemented architecture is already supported.

## Weights & Biases decision

W&B is useful and should be enabled on GCP when approved credentials are
available. It should record:

- loss, learning rate and gradient norm;
- examples and steps per second;
- GPU utilization and memory;
- checkpoint step and run identity;
- protocol, model-profile and Git identifiers as non-secret metadata.

W&B is not authoritative evidence. The canonical record remains ordinary logs,
checkpoint files, predictions, manifests and SHA-256 receipts stored locally
and synchronized to GCS. Offline and disabled modes remain supported. A W&B
outage must not invalidate or erase an otherwise complete run.

## Reproducibility checkpoints

The authoritative lifecycle is:

1. Review and lock the protocol in a clean commit.
2. Run the 32-record smoke against that commit.
3. Prepare the official training split without resolving the official test
   path.
4. Create a run identity containing Git, protocol, model-profile, data and
   environment hashes.
5. Save complete optimizer, scheduler, trainer and adapter checkpoints every
   359 steps (25 equal intervals, including the locked final step).
6. Prove resume from a complete intermediate checkpoint.
7. Finish at exactly step 8,975 and independently reload the adapter.
8. Hash the final adapter and create a checkpoint-freeze receipt.
9. Freeze evaluator and analysis implementation hashes.
10. Only then authorize one complete official-test benchmark in a new GCS
    prefix.
11. Retain every raw prediction and create a hash-bound analysis and
    visualization bundle.

## Visualization outputs

The report renderer produces deterministic SVG and CSV artifacts for:

- headline metric comparison with bootstrap intervals;
- token F1 by question class;
- token F1 by complexity;
- answer-length error.

SVG and CSV keep figures inspectable, publication-friendly and independent of
notebook display state. Every output receives a SHA-256 entry in the
visualization manifest.

## Implementation state and remaining gate

The restart-safe full-training runner, final checkpoint-freeze receipt,
restart-safe two-condition evaluator, benchmark analysis and deterministic
visualizations are now implementation-bound by SHA-256 in the draft. Focused
tests cover the protocol gate, explicit test authorization, reproducible
source-group bootstrap, metric rendering and restart helpers.

The remaining gate is human review of the scientific choices in this draft.
After review, create the final locked protocol, refresh its implementation
hashes, commit it, tag the clean commit, and run the 32-record smoke. Do not
materialize official test data during review or smoke training.

## Authoritative GCP command sequence

Use the project Make targets so paths and parameters come from versioned
profiles. On the clean, exact GCP checkout:

```bash
cd gi_vqa_research
export EXPECTED_COMMIT="$(git rev-parse HEAD)"
export RQ1_PROTOCOL="protocols/study1/rq1_full_baseline_protocol_v1.json"
export RQ1_WANDB_MODE="online"

make rq1-baseline-check
make rq1-baseline-plan
make rq1-training-smoke
make rq1-training
```

Inspect and synchronize the training report, final adapter and
`checkpoint_freeze_receipt.json`. Only after the receipt says
`test_evaluation_authorized: true`, launch the one frozen benchmark:

```bash
make rq1-full-test
make rq1-analysis
make rq1-visualizations
```

Run these inside `infra/gcp/run_job.sh` when session-independent `tmux`,
manifest capture and non-deleting GCS synchronization are required. Re-running
the training or full-test Make target with the same directories resumes from
complete checkpoints or immutable item files.
