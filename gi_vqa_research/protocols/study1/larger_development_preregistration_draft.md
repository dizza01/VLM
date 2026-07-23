# Study 1 larger development evaluation — preregistration draft

Status: **DRAFT — not authorised for execution**

This document must be reviewed, converted to a machine-validated locked JSON
protocol, committed and hashed before any larger model run. The grouped test
artifact remains sealed throughout development.

## Objective

Determine whether the paired-image adapter improves answer correctness because
it uses the corresponding GI image, rather than only learning question-answer
priors or becoming more confident.

## Frozen inputs proposed for review

- Base model: `google/paligemma-3b-pt-224` at
  `35e4f46485b4d07967e7e9935bc3786aad50687c`.
- Paired and constant adapters: exact hashes in
  `controlled_training_pass.json`.
- Grouped split manifest SHA-256:
  `6dbd368bb9eba0b9fecf9444564e280cbadf4c51915e1ee8b1ac1787c566b885`.
- Development only; grouped test access is forbidden.
- Proposed sample: 256 unique development source images, one deterministically
  selected question per source, excluding the 20 pilot sources.
- Selection seed: 42; selection and ordered item IDs must be published and
  hashed before inference.
- Runtime: Python 3.11, one Tesla T4, PyTorch 2.6.0+cu124 and every package
  version enforced by `training_gate.EXPECTED_PACKAGES`.

## Conditions proposed for review

All conditions use identical ordered questions and deterministic generation.

1. `paired_correct`: paired-image adapter with the corresponding source image.
2. `constant_control`: constant-image adapter with the same neutral 224×224 RGB
   image used during its training.
3. `paired_shuffled`: paired-image adapter with a deterministic no-fixed-point
   permutation of images within the 256 selected development sources.
4. `paired_neutral_ablation`: paired-image adapter with the neutral image.
5. `base_correct_descriptive`: unadapted base with the corresponding image;
   reported as a benchmark, not part of the primary selection comparison.

PaliGemma requires an image input, so a literal question-only condition is not
technically valid. `paired_neutral_ablation` is the prespecified question-only
proxy. It must be described as a neutral-image ablation, not literal text-only
inference.

## Outcomes proposed for review

Primary correctness endpoint:

- Macro mean normalized token F1 against the reference answer.
- Primary contrast: `paired_correct - constant_control`.
- Success requires the two-sided 95% source-cluster bootstrap confidence
  interval for that difference to exclude zero in the positive direction.

Grounding co-primary checks:

- `paired_correct - paired_shuffled` token F1.
- `paired_correct - paired_neutral_ablation` token F1.
- Both lower 95% confidence bounds must be greater than zero; otherwise the
  adapter is not considered demonstrably image-grounded.

Secondary correctness metrics:

- Normalized exact match, ROUGE-L, corpus BLEU and answer-length error.
- Results stratified by complexity and question class.
- Holm correction across non-primary pairwise contrasts.

Calibration metrics:

- Mean token log-probability and sequence confidence are descriptive only.
- Expected calibration error and Brier score require a correctness target fixed
  before execution; the proposed target is item-level token F1 ≥ 0.5.
- Report selective accuracy/coverage curves without selecting a threshold on
  these development results.

Grounding and faithfulness metrics:

- Performance degradation under shuffled and neutral images.
- On a separately locked 64-item subset, run the existing attention/Grad-CAM
  deletion and insertion protocol using fixed generated answers.
- Report most-salient versus random intervention effects with source-cluster
  bootstrap intervals.

Clinical safety metrics:

- Before execution, define a blinded annotation rubric for clinically material
  false presence, false absence, wrong location/count and unsupported certainty.
- Two independent qualified reviewers, condition labels hidden; disagreements
  adjudicated without access to the grouped test set.
- Report per-category rates and paired differences. These are secondary unless
  a minimum clinically important difference is specified before locking.

## Statistical and execution rules

- Ten thousand deterministic source-cluster bootstrap replicates, seed 42.
- Unit of resampling is `source_img_id`.
- Missing or failed items are not silently dropped; any incomplete condition
  makes the gate fail and requires a documented rerun under the same lock.
- No hyperparameter, prompt, metric, threshold or model-selection change after
  predictions are observed.
- Promote the paired adapter only if the primary correctness contrast and both
  grounding checks pass. Otherwise stop and revise using development data under
  a newly versioned protocol.

## Test-set seal

The runner must reject `test` and `grouped_test`, must not resolve the test
artifact path, and must report `test_partition_accessed: false`. The final test
evaluation is allowed once, only after this protocol, the clinical rubric and
the model-selection rule are locked and hashed.
