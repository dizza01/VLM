# Larger-development results

## Decision

The locked development experiment completed successfully, but the paired-image
adapter did **not** satisfy the preregistered progression criteria. It must not
be promoted to the official test evaluation. The official test partition
remains sealed.

This distinguishes execution validity from model success: the inference,
analysis and faithfulness stages all passed their integrity checks, while the
model-selection outcome was negative.

## Completed experiment

- Repository commit:
  `7ad3807915d0e461ed77d437b6df436eab677992`
- Locked protocol SHA-256:
  `6940590751a610d5e62278258ecbef329efc474e1d579dc7a3a6e9515fd694b2`
- Development sources: 256
- Inference conditions: 5
- Completed inference item-conditions: 1,280 of 1,280
- Faithfulness subset: 64 sources across 2 conditions
- Completed faithfulness item-conditions: 128 of 128
- Runtime: Tesla T4, Python 3.11.13, PyTorch 2.6.0+cu124,
  Transformers 4.55.0
- Test partition accessed: false

## Locked outcomes

The primary metric was normalized token F1. A comparison passed only when the
lower bound of its source-level 95% bootstrap confidence interval was greater
than zero.

| Locked comparison | Mean difference | 95% confidence interval | Outcome |
|---|---:|---:|---|
| Paired correct − constant control | -0.0026 | [-0.0163, 0.0111] | Failed |
| Paired correct − paired shuffled | 0.0095 | [-0.0059, 0.0257] | Failed |
| Paired correct − paired neutral | -0.0671 | [-0.1061, -0.0274] | Failed |

All three checks were required. Therefore:

> Do not promote the paired-image adapter. Any further development must use a
> new versioned protocol fixed before another experiment is run.

## Headline condition results

| Condition | Mean normalized token F1 |
|---|---:|
| Paired correct | 0.3091 |
| Constant control | 0.3116 |
| Paired shuffled | 0.2996 |
| Paired neutral ablation | 0.3761 |
| Unadapted base, correct image | 0.1650 |

The adapters improved over the unadapted descriptive baseline, but the
paired-image adapter did not improve over the constant-image control. Supplying
the correct image also did not produce a reliable advantage over a shuffled
image. The paired adapter performed better with the neutral image than with the
correct image, which is inconsistent with the locked image-grounding
hypothesis.

## Development-only diagnostics

These diagnostics explain the observed result; they do not replace or modify
the locked analysis.

- Against the constant control, paired-correct improved on 39 items, tied on
  170 and was worse on 47.
- Against paired-shuffled, paired-correct improved on 41 items, tied on 176 and
  was worse on 39.
- Against paired-neutral, paired-correct improved on 90 items, tied on 43 and
  was worse on 123.
- Paired-correct and paired-shuffled produced the same case-insensitive answer
  on 127 of 256 items (49.6%).
- The paired-correct calibration error was 0.2939, compared with 0.2178 for the
  constant control. Higher confidence from the paired model therefore did not
  translate into better calibrated correctness.
- No complexity stratum showed a meaningful paired-versus-constant advantage:
  the mean differences for complexity levels 1, 2 and 3 were -0.0010, -0.0150
  and 0.0074 respectively.

The question-class breakdown contains small and overlapping subgroups, so it is
exploratory only. Some instrument-related classes showed greater
paired-versus-shuffled differences, while several appearance, location and
count classes favoured the shuffled or constant conditions. This pattern does
not support a general image-grounding claim.

## Faithfulness findings

The faithfulness stage completed and reproduced every saved answer before
scoring image interventions.

- For paired-correct, Grad-CAM salient-region effects were positive for both
  deletion treatments and insertion.
- Decoder attention was mixed: gray deletion was directionally positive, while
  blur deletion and blur insertion were negative.
- Constant-control effects were zero because blurring or replacing regions of
  the uniform neutral image does not change its pixels in a meaningful way.

These results suggest that some paired-model predictions respond to localized
visual features, particularly under Grad-CAM. They do not establish that those
features improve answer correctness, and faithfulness was explicitly
descriptive rather than a model-selection gate.

## Interpretation and next development questions

The current evidence is most consistent with adapter learning that improved
task-format or language behaviour without demonstrating a reliable benefit
from the corresponding image. Plausible development hypotheses include:

1. The controlled training budget may be too small for robust visual
   adaptation.
2. The selected LoRA placement or capacity may favour language-side adaptation
   over visual grounding.
3. The answer distribution may allow strong question or response priors,
   especially for finding-presence questions.
4. Correct-image information may help only a subset of question types, with
   effects diluted or reversed elsewhere.
5. Confidence increased without a corresponding correctness gain, so
   calibration requires attention in any revised model.

These are hypotheses, not conclusions. They should be tested only on training
and development partitions.

## Required next gate

Before any new training run:

1. Review representative development errors by question class without
   accessing test.
2. Choose one scientifically motivated change, such as a larger training
   budget, revised adapter placement or a more visually discriminative
   development design.
3. Define a new versioned protocol with fixed data, training settings,
   hypotheses, metrics and progression rules.
4. Hash and commit that protocol before running the revised experiment.

The current negative result must be retained. The official test set must not be
opened for this adapter.
