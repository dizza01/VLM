# Artifact ownership

| Artifact | Authoritative store |
| --- | --- |
| Source, configs, tests and protocol manifests | Git |
| Raw and intermediate run artifacts | GCS |
| Final LoRA adapters and model cards | Hugging Face Hub |
| Live metrics and comparison dashboards | Weights & Biases |
| Active cache | VM persistent disk |

Every external artifact referenced by a paper result must have an immutable identifier: Git SHA,
GCS object generation/hash, or Hugging Face commit revision.

