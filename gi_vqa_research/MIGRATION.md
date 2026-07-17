# Study 1 migration map

The existing root notebook remains the research specification while execution is migrated in
small, testable stages. This prevents a large rewrite from changing the scientific protocol.

| Existing notebook responsibility | New destination | Current state |
| --- | --- | --- |
| Environment and study configuration | `configs/study1/`, `gi_vqa.config`, run manifest | Foundation implemented |
| JSONL creation and image caching | `gi_vqa.data` | Planned extraction |
| Split audit and leakage hard gates | `gi_vqa.audit` | Implemented and tested |
| Protocol lock | `protocols/study1/` | Tracked destination created |
| Training | `gi_vqa.train` | Planned extraction |
| Deterministic inference and controls | `gi_vqa.infer` | Planned extraction |
| Answer metrics and stratification | `gi_vqa.metrics` | Planned extraction |
| Calibration | `gi_vqa.calibration` | Planned extraction |
| Attention and Grad-CAM | `gi_vqa.attribution` | Blocked on model-backend implementation |
| Deletion/insertion interventions | `gi_vqa.perturbation` | Planned as resumable shards |
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

The next useful milestone is a complete 20-item development run:

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

