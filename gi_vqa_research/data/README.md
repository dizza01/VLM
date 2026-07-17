# Local data cache

This directory is ignored except for this file. Reconstruct data from immutable dataset
revisions and the tracked split manifests. Do not commit Kvasir images or generated JSONL data.

On GCP, point the cache to a persistent disk. Treat that disk as replaceable: durable manifests,
predictions and released artifacts must also be uploaded to GCS.

