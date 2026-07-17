# Provisional Study 1 PaliGemma backend contract

Status: implementation and executable contract notebook ready; Colab T4
execution still pending. This is not yet a locked confirmatory protocol.

## Model identity

- Backend: direct Transformers PaliGemma with an optional PEFT LoRA adapter.
- Base checkpoint, processor and adapter revisions must be immutable.
- The processor uses its saved slow image processor (`use_fast: false`).
- Generation, target scoring and attribution use the same loaded model object.
- The initial attribution smoke uses FP16, batch size one, no quantisation and
  eager attention.
- Model parameters are frozen during inference and attribution. Grad-CAM
  retains gradients only through the processed image and selected vision
  activation.

## Prompt and saved answer

The source record uses `<image>{question}`. The pinned PaliGemma processor must
expand that placeholder into the model's image-token prefix, insert BOS, and
append the prompt newline. The Colab contract must show exact input-ID,
training-label and token-type equivalence with the ms-swift 3.7 `paligemma`
template, plus equality of its processed FP16 image tensor. The equivalence
check loads only a second processor, not a second model.

Greedy generation uses no sampling, one beam and at most 64 new tokens. The
saved answer identity is its generated token-ID sequence after removing terminal
EOS and padding. Decoded text is an accompanying representation, not the
primary identity of the fixed target.

## Fixed-answer score

The processor receives the saved decoded answer as `suffix`, producing
`labels` and `token_type_ids`. The backend must assert that suffix
tokenisation, excluding EOS, reproduces the saved generated token IDs.

For target token at input position `t`, use logits at position `t - 1`.
Question, image, BOS, padding and EOS tokens are excluded. The primary scalar
target is the arithmetic mean of the selected tokens' natural log
probabilities:

```text
mean_t log P(saved_answer_token_t | image, question, preceding saved tokens)
```

Generation-time and teacher-forced token log probabilities must agree within
the tolerance recorded in the resolved configuration.

Attribution accepts the complete saved generation result, not free text. It
performs the token-identity and score-parity check before attribution and
repeats the parity check on the attribution forward pass.

## Decoder answer-to-image attention

Use decoder self-attention from the configured layer. For each scored answer
token at input position `t`, use the query at `t - 1`, because that position
predicts the token. Retain only keys corresponding to image-token positions,
then apply the configured head and answer-token aggregation. Reshape the 256
image-token values to the validated 16 by 16 patch grid.

## Answer-conditioned Grad-CAM

Hook the configured SigLIP vision encoder layer. Backpropagate the fixed
answer's mean token log probability, average activation gradients over patch
positions, form the channel-weighted activation sum and apply ReLU when
configured. Reshape the result to the same validated 16 by 16 patch grid.

Both attribution methods are min-max normalised to float32. The result records
the raw minimum and maximum; a constant map is a hard failure in the initial
smoke protocol.

## Reproduction identity

The backend fingerprint covers loading, decoding, scoring and attribution
settings. Each prepared input records SHA-256 digests for source file bytes
when a path is available, canonical RGB pixels and the processed pixel tensor.
The restart-safe stage runner must persist and compare these values before
reusing an item.

## Contract-only fixture

The executable CUDA contract pins official training row `143500`, source image
`cl8k2u1pv1e4z08320vbv6jzb`, and the exact source-JPEG SHA-256 recorded in the
runner. It verifies that this source ID is absent from the pinned official test
split. This reduces accidental test contact, but it does not turn the fixture
into a research evaluation item: it is excluded from every reported result and
must be reserved when the grouped split manifest is built.

## Hard failures

The backend must fail rather than silently continue when:

- the exact repository, dependency, data or model identities do not resolve;
- direct Transformers and ms-swift template encodings differ;
- processor/model geometry does not produce 256 image tokens;
- a saved answer cannot be reproduced as the same token IDs;
- generation-time and teacher-forced token scores exceed the configured
  absolute tolerance;
- target scores are missing or non-finite;
- attention tensors or vision gradients are unavailable;
- an attribution is non-finite, constant or has the wrong shape; or
- any prepared input, score or attribution belongs to different model
  provenance.
