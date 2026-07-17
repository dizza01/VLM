# Generated runs

Each run receives its own immutable directory:

```text
runs/<run_id>/
  manifest.json
  resolved_config.yaml
  status.json
  logs/
  checkpoints/
  predictions/
  attributions/
  perturbations/
  metrics/
```

Run contents are ignored by Git. Sync durable artifacts to a run-specific GCS prefix and never
reuse a confirmatory run ID.

