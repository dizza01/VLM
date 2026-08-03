# Artifact and Data Policy

Git contains source, tests, dependency declarations, small configuration,
protocols, compact receipts and hash manifests. It must not contain datasets,
endoscopy images, model weights, adapters, checkpoints, predictions,
attribution arrays, raw logs, credentials or complete run bundles.

| Store | Permitted content |
| --- | --- |
| Git | Code, documentation, protocols, compact receipts and hashes |
| GCS | Checkpoints, predictions, raw logs, analysis inputs and run bundles |
| Hugging Face Hub | Referenced datasets and selected final adapters/model cards |
| W&B | Non-authoritative monitoring metrics and run metadata |
| Local/VM cache | Reconstructable downloads and active job state |

Every durable run directory should have a unique identifier and a manifest
binding the Git commit, protocol, model/data revisions, environment and
artifact hashes. GCS synchronization must be non-deleting. W&B is never the
sole evidence store.

The Kvasir-VQA-x1 data and base models are obtained from their named upstream
repositories at the immutable revisions recorded in `protocols/` and
`configs/`. This repository does not redistribute those files. Users are
responsible for accepting upstream terms and supplying a read-capable
`HF_TOKEN` through an environment or secret store.

Ignored artifacts can be reconstructed by the relevant materialization or run
command. Historical local bundles should be archived outside the checkout with
a relative-path SHA-256 manifest before deletion.
