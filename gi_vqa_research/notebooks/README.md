# Notebook policy

Notebooks are for:

- exploratory inspection;
- development-only examples;
- visual quality control;
- regenerating paper tables and figures from immutable run artifacts.

Long-running data preparation, training, inference, attribution, perturbation and scoring belong
in importable modules and command-line stages. A notebook must not be the only place an
experiment can be reproduced.

Recommended notebooks:

- `00_colab_t4_backend_contract.ipynb` — executable one-item CUDA compatibility
  gate;
- `01_colab_t4_training_gate.ipynb` — executable two-step tiny-LoRA checkpoint,
  resume and adapter-reload gate;
- `02_colab_t4_development_smoke.ipynb` — restart-safe 20-item base-model
  inference, attribution and perturbation gate;
- `01_data_audit.ipynb`
- `02_pilot_inspection.ipynb`
- `03_results_report.ipynb`

Keep outputs small or strip them before committing. Never embed credentials or model weights.
The existing Study 1 notebook remains in the repository root during this additive migration.

## Run the Colab T4 contract

Open `00_colab_t4_backend_contract.ipynb` from the pushed GitHub repository in
Colab. Under **Runtime → Change runtime type**, select runtime version
`2025.07` and a T4 GPU. This is the available past runtime matching the
reference Python 3.11.13/PyTorch 2.6.0 environment; the current `2026.04`
runtime is not equivalent. Expose a read-capable Colab secret named
`HF_TOKEN`, paste the full 40-character SHA of the commit containing the
notebook and backend, and run all cells.

The notebook does not contain the scientific implementation. It installs the
package from the same exact commit and invokes:

```bash
python -m gi_vqa.contract
```

The runner always writes `contract_report.json` after it starts, and the
notebook bundles that report, attribution arrays when produced, the diagnostic
image, environment record and `pip freeze`. Preserve the downloaded bundle
with the tested Git commit. Bootstrap and authentication failures are also
converted into a small failure report and evidence bundle before the final
notebook cell raises.

The bootstrap records the full output of `pip check`. Conflicts belonging to
the GI-VQA dependency stack are hard failures. Conflicts caused only by
unrelated packages preinstalled in the shared Colab image are retained as
warnings; the package-level runner still verifies the exact versions it uses.

Contract v1 identified a real off-by-one in the built-in ms-swift 3.7.0
PaliGemma training `token_type_ids`. Contract v2 recorded that raw one-token
difference, required it to match the known pattern, tested the versioned
`gi_vqa_paligemma_v1` template for exact equality with the direct processor,
and passed all 61 checks on the reference T4 environment. This is compatibility
evidence, not a research result.

A PASS establishes only that the shared backend works in the pinned reference
environment. The fixed item is excluded from research results and is reserved
by the tracked grouped split manifest. The grouped split, artifact-integrity,
and bounded image-cache gates have passed. The tiny-LoRA training gate described
below has also passed; the implemented restart-safe 20-item development runner
is the next execution gate.

## Run the Colab T4 training gate

After committing and pushing the training-gate implementation, open
`01_colab_t4_training_gate.ipynb` through Colab's GitHub integration. Use the
same runtime version `2025.07`, T4 GPU and `HF_TOKEN` secret as the backend
contract. Paste the full 40-character commit containing the notebook, tracked
split/cache manifests and runner, then run all cells.

The notebook reconstructs the ignored split files and 40-image cache from
their tracked locks. It invokes:

```bash
python -m gi_vqa.training_gate
```

The gate selects one question from each of the 20 locked training sources,
trains a rank-16 LoRA adapter to checkpoint 1, exits that training process,
resumes the adapter/optimizer/scheduler/trainer state to checkpoint 2, verifies
that the adapter changed, and independently reloads it for a finite-loss
forward pass. It downloads a compact evidence bundle containing reports, logs,
the exact training subset and hashes—not the disposable adapter weights.

That PASS authorised the now-implemented restart-safe 20-item development
inference/explanation runner. It is not a trained research model and must never
be reported as an experimental result. The reference run passed all 15 checks
at commit `da94b251c0f49d4fa74e4351c3487f5ce3286ade`; see
`../protocols/study1/training_gate_pass.json` for the compact evidence receipt.

## Run the restart-safe Colab T4 development smoke

After committing and pushing the implementation, open
`02_colab_t4_development_smoke.ipynb` through Colab's GitHub integration. Use
runtime version `2025.07`, a T4 GPU and the `HF_TOKEN` secret. Paste the exact
40-character commit and run all cells.

The notebook deliberately invokes the package twice against one run directory.
The first process is capped at one new item and must return `INCOMPLETE`. The
second resumes and must report exactly one reused item and 19 new items. The
runner never overwrites completed stage artifacts: each item is complete only
after its prediction, two attribution archives and two perturbation result files
have been hashed into `complete.json`.

A PASS also requires a validated 20-row merge, finite scores for every configured
intervention and an immutable diagnostic report. The bundle contains the run
manifest, compact JSON/JSONL/NPZ artifacts, bootstrap evidence and dependency
lock—not model weights or cached source images. This development-only run checks
the entire execution path; it is explicitly excluded from research results.
