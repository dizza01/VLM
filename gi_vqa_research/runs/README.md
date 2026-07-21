# Generated runs

Each run receives its own immutable directory:

```text
runs/<run_id>/
  manifest.json
  backend.json
  status.json
  items/<rank>-<item_id>/
    prediction.json
    attributions/<method>.npz
    perturbations/<method>.json
    complete.json
  shards/
  predictions/
  metrics/
```

Run contents are ignored by Git. Sync durable artifacts to a run-specific GCS prefix and never
reuse a confirmatory run ID.

`complete.json` is the commit marker for one smoke item. On resume, the runner
verifies every referenced artifact hash before reusing it. A partial item may
recompute its unfinished stage, but a completed stage is never overwritten.
