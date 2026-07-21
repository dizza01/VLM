# Study 1 migration map

The existing root notebook remains the research specification while execution is migrated in
small, testable stages. This prevents a large rewrite from changing the scientific protocol.

| Existing notebook responsibility | New destination | Current state |
| --- | --- | --- |
| Environment and study configuration | `configs/study1/`, `gi_vqa.config`, run manifest | Foundation implemented |
| JSONL creation and image caching | `gi_vqa.splits`, `gi_vqa.image_cache` | Grouped metadata complete; locked smoke/training cache implemented |
| Split audit and leakage hard gates | `gi_vqa.splits`, `gi_vqa.audit` | Built from pinned metadata; independent gate passed |
| Protocol lock | `protocols/study1/` | Tracked destination created |
| Shared PaliGemma model contract | `gi_vqa.model_spec`, `gi_vqa.backends`, `gi_vqa.contract` | Implemented; Colab T4 contract v2 passed |
| Training | `gi_vqa.training`, `gi_vqa.training_gate` | Corrected template plus tiny-LoRA save/resume/reload gate implemented; T4 gate passed |
| Deterministic inference and controls | `gi_vqa.smoke_runner` | Restart-safe 20-item development runner implemented; T4 run pending |
| Answer metrics and stratification | `gi_vqa.metrics` | Planned extraction |
| Calibration | `gi_vqa.calibration` | Planned extraction |
| Attention and Grad-CAM | `gi_vqa.backends`, `gi_vqa.smoke_runner` | Per-item atomic attribution archives implemented; 20-item CUDA validation pending |
| Deletion/insertion interventions | `gi_vqa.perturbations`, `gi_vqa.smoke_runner` | Deterministic controls and fixed-answer scoring implemented; 20-item CUDA validation pending |
| Bootstrap intervals and reporting | `gi_vqa.statistics`, results notebook | Planned extraction |
| GCP job execution | `infra/gcp/` | Conservative scaffold implemented |

## Migration rules

1. Preserve the existing notebook until a replacement stage passes a fixed-fixture equivalence
   test.
2. Move functions, not outputs, into `src/gi_vqa/`.
3. Give every long stage a CLI, resolved configuration, manifest, restart behaviour and atomic
   output.
4. Keep development and confirmatory commands separate.
5. Do not add a test-data path to training code.
6. Do not run the full attribution study until the same model backend reproduces saved answers
   during both generation and teacher-forced scoring.

## Next implementation slice

The Colab T4 package-level backend contract has passed. It established:

1. verify the exact reference software/GPU environment and immutable inputs;
2. load the pinned processor and base checkpoint;
3. isolate the known built-in ms-swift 3.7.0 PaliGemma boundary defect and
   assert exact direct-processor equivalence for the project-owned training
   template;
4. assert 256 image tokens and the 16 by 16 patch grid;
5. generate the same answer twice;
6. reproduce its saved token IDs through suffix tokenisation;
7. compare generation and teacher-forced token log probabilities;
8. produce finite, nonconstant attention and Grad-CAM maps; and
9. record peak CUDA memory, hashes and resolved provenance.

This is compatibility evidence, not a research result. The grouped split hard
gate has also passed:

```bash
python -m gi_vqa.cli prepare-splits \
  --config configs/study1/smoke.yaml \
  --project-root .
python -m gi_vqa.cli split-check \
  --manifest protocols/study1/grouped_split_manifest.json \
  --project-root .
```

The tracked manifest records 126,064 training, 16,477 development and 16,297
test items across mutually disjoint source-image groups. The builder reserved
the contract fixture and selected the fixed 20-item development smoke set.
The locked 20-development/20-training image cache and offline audit passed.
The two-step tiny-LoRA checkpoint/resume/reload gate passed all 15 checks on
the reference Colab T4 at commit
`da94b251c0f49d4fa74e4351c3487f5ce3286ade`; the tracked compact receipt is
`protocols/study1/training_gate_pass.json`. The per-item restart-safe stages are
implemented and await the locked T4 development smoke.

All extracted PaliGemma training must run through:

```bash
python -m gi_vqa.training ...
```

This wrapper forces `gi_vqa_paligemma_v1` through ms-swift's external-plugin
mechanism. Do not substitute raw `swift sft`, whose pinned built-in PaliGemma
template produces the token-type boundary defect detected by contract v1.

The next useful milestone is executing the complete 20-item development run:

```text
prepare data
  -> audit source-image separation
  -> load one immutable adapter
  -> deterministic predictions
  -> one attention and one Grad-CAM map
  -> every deletion/insertion mode
  -> fixed-answer scores
  -> merged shard validation
  -> metrics and a compact report
```

Only after that path is restart-safe should the pilot increase beyond 20 items.
