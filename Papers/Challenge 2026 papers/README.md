# Medico 2026 Team-Report Synthesis

## Purpose and Scope

This note records what the available Medico challenge reports achieved, what
their evidence does and does not support, and how they inform the GI-VQA PhD
programme. It is intended to prevent repeated literature triage and to preserve
the rationale for the Study 1 and Study 2 designs.

The synthesis was prepared from the six PDFs in this directory on 20 July 2026.
Reported results have not been independently reproduced. A high challenge score
is therefore treated as a reported result rather than proof of clinical utility,
explanation faithfulness, or generalisation.

## Report Inventory

### Challenge definition

- [`paper4.pdf`](paper4.pdf): **Medico 2026: Visual Question Answering for
  Gastrointestinal Imaging**, by Gautam et al.
- This is the challenge overview rather than a participant system report.
- It defines Subtask 1 answer generation and Subtask 2 explainable and safe
  multimodal reasoning.
- It introduces assessment by question class and complexity, behavioural safety
  criteria, and a private challenge set containing images outside
  Kvasir-VQA-x1.
- The private set is important generalisation evidence, but it does not repair
  the overlapping public split for researchers who continue to use the public
  Kvasir-VQA-x1 partitions.

### Boundary-guided multimodal reasoning

- [`paper17.pdf`](paper17.pdf): **Boundary-Guided Multimodal Reasoning for
  Explainable Gastrointestinal VQA**, by Dam, Nguyen, and Le.
- The system uses YOLO segmentation trained on Kvasir-SEG and
  Kvasir-Instrument, renders a thin boundary rather than an opaque mask, and
  lets a fine-tuned Qwen3.5-4B model refine its answer using that boundary.
  Ten question-class-specific self-probes and a separate Qwen3.5-35B-A3B model
  are then used to synthesise an explanation.
- The useful design idea is texture-preserving, question-conditioned visual
  guidance. The comparison with bounding boxes and filled masks is more
  informative than showing one heatmap alone.
- The reported boundary condition does not clearly improve the unmodified
  Qwen3.5-4B baseline: BLEU changes from 0.4964 to 0.4946, ROUGE-L from 0.7005
  to 0.7021, and METEOR remains 0.7040. No uncertainty interval or
  source-clustered test is reported.
- The report does not test whether the boundary or the displayed region
  causally supports the answer. The separate explanation synthesiser can
  produce a plausible post-hoc narrative without representing the answer
  model's reasoning.
- Its qualitative example contains useful warning signs: the primary answer,
  self-probes, size descriptions, and final explanation are not fully
  consistent. These are examples of the cross-modal and within-rationale
  contradictions that Study 2 should measure systematically.

### Medical vision-encoder and PEFT benchmarking

- [`paper31.pdf`](paper31.pdf): **Do Medical Vision Encoders Help? PEFT
  Benchmarking of Vision-Language Models for Gastrointestinal VQA**, by
  Mattioli and Almeida.
- The report compares Granite Vision, SmolVLM2, Qwen2-VL, BLIP-2, and
  MedVLM-R1, and includes LoRA, DoRA, and AdaLoRA conditions.
- Its most useful contribution is a negative result: replacing Granite's
  pretrained SigLIP tower with frozen MedCLIP or PubMedCLIP encoders connected
  through a linear dimensional adapter reduced performance. The native
  SigLIP/LoRA condition reached ROUGE-1 0.4468 on the combined medical corpus,
  compared with 0.3959 for MedCLIP and 0.4200 for PubMedCLIP.
- This does not establish that medical encoders are generally inferior. The
  downstream projector was pretrained for SigLIP features, the replacement
  encoders were frozen, and a simple dimensional adapter may not resolve the
  representation mismatch.
- The competition runs report strong public results, including BLEU 0.4928,
  ROUGE-1 0.7182, and METEOR 0.6968 for Granite Vision. Public scores should
  not be interpreted as unseen-image generalisation because of the known
  Kvasir-VQA-x1 source-image overlap.
- CLAHE and online augmentation were used, but their independent effects were
  not ablated.

### CARES reliability layer

- [`paper9.pdf`](paper9.pdf): **CARES at MediaEval Medico 2026: Calibrated
  Aspect-Reliable Explanations and Safety for Gastrointestinal Visual Question
  Answering**, by Emad, Safwan, and Tahir.
- CARES wraps a frozen Florence-2 model and public adapter with referring
  segmentation, structured explanation templates, empirical-Bayes reliability,
  and selective prediction.
- This report has the strongest evaluation discipline in the collection. It
  uses image-disjoint cross-fitting, compares several confidence signals,
  reports calibration as well as discrimination, applies a
  distribution-free risk-control procedure, and triangulates explanation
  assessment with structured support, NLI, and an LLM judge.
- The reported per-`(aspect, answer)` reliability reaches AUROC 0.87 and ECE
  0.025, compared with approximately 0.60 for maximum softmax probability and
  0.62 for self-consistency. At a target selective risk of 0.05, it reports
  approximately 0.56 coverage and 0.03 verified selective error.
- This "base-rate dominance" result is an important baseline, but it is also
  evidence of dataset and output shortcuts. The score mainly estimates how
  often a particular answer is correct for a question aspect; it is not
  instance-specific visual uncertainty and may fail under prevalence or
  answer-distribution shift.
- A template can be consistent with selected evidence without being faithful
  to the causal computation that produced the answer. CARES therefore
  motivates, but does not replace, a causal visual-faithfulness evaluation.

### Clinical-aware topological adaptation

- [`paper41.pdf`](paper41.pdf): **CATA: Clinical-Aware Topological Adaptation
  for Generative Medical VQA**, by Nguyen and Nguyen.
- CATA combines a frozen ViT/timm encoder, Qwen2.5-3B with QLoRA, patch-level
  topological descriptors, visual fusion, and gated decoder adapters. Task 2
  produces a heatmap, self-probes, evidence JSON, a rationale, and a
  reliability-style score.
- Its architectural ablation is a strength. On the reported full-test
  experiment, BLEU rises from 0.354 for the baseline to 0.373 with visual TDA,
  0.447 with the decoder TDA adapter, and 0.451 with both.
- On the private challenge set, the clean model receives an LLM-judge overall
  score of 8.47. For Task 2, the reported correctness score is 9.21, but
  faithfulness is 7.16 and clinical relevance 7.26. The authors also identify
  contradictions among self-probes.
- The TDA heatmap is evidence derived from the method, but the report does not
  establish that highlighted regions causally determine the generated answer.
  Explanation properties are primarily assessed by an automated judge.
- The reported ECE of 0.131 is calculated on 300 examples whose correctness is
  thresholded from another LLM judge. It is useful as a diagnostic, but not
  equivalent to calibration against independently verified medical labels.

### Public-split leakage audit

- [`paper29.pdf`](paper29.pdf): **Data Leakage in Kvasir-VQA-x1: Overlapping
  Images Threaten Evaluation Validity**, by Azmoodeh-Kalati et al.
- Despite its location in this folder, this is a MediaEval 2025 report.
- It identifies 3,821 image IDs shared by the official training and test
  partitions: approximately 94% of official-test image IDs have appeared in
  training. It recommends assigning all questions for a source image to one
  partition.
- It establishes the structural leakage problem but does not train controlled
  models to estimate the amount of metric inflation. Quantifying that effect
  remains a useful research contribution.

## Cross-Report Findings

### What was done well

1. CARES demonstrates good practice for image-grouped calibration,
   error-selective prediction, and explicit comparison with simple baselines.
2. CATA provides a real component ablation instead of attributing all gains to
   a multi-component final system.
3. The boundary report tests a clinically motivated display choice that
   preserves internal lesion texture.
4. The encoder report publishes a useful negative result and explains the
   architectural mismatch that may have caused it.
5. The leakage audit makes the public benchmark's intended generalisation claim
   testable rather than assumed.

### Recurring limitations

1. **Causal faithfulness is not established.** A relevant-looking heatmap,
   boundary, mask, template, or rationale is not shown to reflect evidence that
   changes model support for the answer.
2. **Explanation provenance is often detached from answer generation.**
   Separate synthesis models, templates, self-probes, and external
   segmentation systems can create persuasive explanations after prediction.
3. **Public scores are affected by source-image overlap.** Most reports do not
   separate familiar-image performance from genuinely unseen-image
   performance.
4. **Automated judges dominate Task 2 evaluation.** LLM ratings are useful
   secondary measures but cannot establish clinician usefulness, trust, or
   safety, and may reward fluency and verbosity.
5. **Statistical uncertainty is rarely reported.** Aggregate point estimates
   generally lack training-seed sensitivity, paired source-clustered
   confidence intervals, and negative explanation controls.
6. **Question and answer shortcuts are insufficiently controlled.** CARES
   exposes strong output-conditioned base rates, but the reports generally lack
   trained constant-image/question-prior controls and shuffled-image tests.
7. **Cross-modal consistency is not systematically tested.** Contradictory
   probe answers, visual evidence, numeric attributes, and final rationales can
   survive explanation synthesis.
8. **Calibration under distribution shift is unresolved.** Reliability
   measured on an image-disjoint sample from the same dataset may not transfer
   to new prevalence, devices, institutions, or answer distributions.

## Implications for Study 1

The reports support retaining the current Study 1 focus rather than pivoting to
a new architecture. The proposed paper occupies a gap left by the challenge
systems:

> On leakage-safe source-image partitions, do GI-VQA attribution maps identify
> evidence that causally supports generated answers, and does causal visual
> faithfulness add information beyond question/answer priors and model
> confidence?

The core contribution should combine:

1. a source-image-disjoint primary benchmark;
2. a controlled secondary official-split replication that estimates leakage
   inflation;
3. trained constant-image, test-time constant-image, and shuffled-image
   controls;
4. answer-token-specific attention, gradient attribution, and a
   model-agnostic occlusion method;
5. deletion and insertion tests against random-map, random-region, and
   least-salient controls;
6. fixed-answer teacher-forced log probability so a perturbation measures
   change in support for the same prediction;
7. source-image-clustered uncertainty intervals and predefined endpoints.

CARES also motivates a focused extension to the existing Study 1 confidence
question. Compare:

- length-normalised generated-answer confidence;
- per-question-class empirical reliability;
- empirical-Bayes per-`(question class, normalised answer)` reliability;
- self-consistency if compute permits;
- the causal attribution effect;
- a combined prior, confidence, and faithfulness model.

Suitable evaluation measures include AUROC, AUPRC, Brier score, ECE, and
risk-coverage curves. The central test is not merely whether an attribution
effect correlates with correctness, but whether it adds out-of-fold predictive
information beyond the strong class/answer base-rate baseline.

### Candidate confirmatory hypotheses

- **H1:** An otherwise controlled model evaluated under the overlapping
  official protocol will receive more favourable answer metrics than under
  unseen-source evaluation.
- **H2:** Targeted deletion based on at least one attribution method will reduce
  fixed-answer log probability more than matched random deletion, but the
  effect will vary by question class and complexity.
- **H3:** Visual attribution faithfulness, answer correctness, and generated
  sequence confidence will be related but non-equivalent.
- **H4:** Per-`(question class, answer)` reliability will be a strong
  correctness baseline; causal visual-faithfulness measurements may or may not
  add incremental error-prediction value.

H4 should be retained even if the result is negative. A well-powered finding
that visual explanation scores add no error-detection value beyond dataset
priors would itself be important evidence against using explanation
plausibility as a confidence signal.

## Deferred Opportunities for Study 2

Boundary guidance is a useful Study 2 condition, especially with:

- correct boundaries;
- boundaries shifted away from the target;
- boundaries copied from another image;
- irrelevant entity boundaries;
- bounding boxes and filled masks;
- the original unmodified image.

These controls would distinguish useful localisation from prompt-like visual
cueing. Study 2 should also score contradictions across answer, self-probes,
visual localisation, numeric attributes, and final rationale.

CATA-style TDA explanations are a possible later comparator, but implementing a
new topological architecture is lower priority than completing the
model-agnostic causal evaluation. Medical-encoder replacement is likewise a
secondary modelling study unless it is framed around a controlled hypothesis
about representation alignment.

## Claims This Programme Should Avoid

- Do not describe a heatmap, boundary, mask, or high overlap score as faithful
  without an intervention-based test.
- Do not describe an LLM-judge score as clinician validation.
- Do not describe a confidence value as calibrated without calibration
  outcomes on data excluded from fitting and threshold selection.
- Do not describe public Kvasir-VQA-x1 performance as unseen-image
  generalisation.
- Do not infer trust or appropriate reliance from explanation fluency,
  completeness, or stated user preference.

