# Local data cache

This directory is ignored except for this file. Reconstruct data from immutable dataset
revisions and the tracked split manifests. Do not commit Kvasir images or generated JSONL data.

On GCP, point the cache to a persistent disk. Treat that disk as replaceable: durable manifests,
predictions and released artifacts must also be uploaded to GCS.

## Build the Study 1 grouped splits

Install the metadata dependency without the GPU stack:

```bash
python -m pip install -e ".[data]"
```

From the `gi_vqa_research` directory, download the pinned public metadata and
build the source-image-disjoint partitions:

```bash
python -m gi_vqa.cli prepare-splits \
  --config configs/study1/smoke.yaml \
  --project-root .
```

This does not download images or inspect model outputs. It writes ignored,
reconstructable files under `data/processed/study1/` and the compact tracked
assignment manifest at `protocols/study1/grouped_split_manifest.json`.

The builder:

- unions the official train and test metadata;
- checks the pinned official overlap counts;
- collapses exact duplicates;
- removes image-question groups with conflicting answers or duplicate metadata;
- reserves the backend-contract source image;
- assigns complete source-image groups by an interpreter-independent SHA-256
  ordering;
- writes train, development and test JSONL files;
- selects 20 metadata-balanced, unique-source items from development only; and
- runs the hard source-leakage and artifact-hash gates before returning PASS.

Re-run the independent gate at any time:

```bash
python -m gi_vqa.cli split-check \
  --manifest protocols/study1/grouped_split_manifest.json \
  --project-root .
```

On a clean clone, the tracked manifest exists but the ignored JSONLs do not.
Reconstruct them without replacing the lock:

```bash
python -m gi_vqa.cli materialize-splits \
  --manifest protocols/study1/grouped_split_manifest.json \
  --project-root .
```

This command rebuilds from the pinned public metadata and returns PASS only
when its candidate manifest is byte-identical to the tracked manifest. It
reuses an already complete verified data directory and rejects partial mixes.

Both commands refuse to overwrite existing outputs. The grouped test file is
created and hashed for protocol completeness, but development code must not
read it; only a locked confirmatory runner may use the `test` artifact.

The pinned build completed with 126,064 training, 16,477 development and
16,297 test items. The tracked manifest is the protocol record; the much larger
JSONL artifacts remain ignored because `split-check` can reconstruct and verify
them from the pinned metadata and manifest.

## Cache the bounded execution-gate images

The first training/evaluation gate needs 20 development images and 20
deterministically selected training images. It deliberately does not access or
cache grouped-test images:

```bash
python -m gi_vqa.cli prepare-image-cache \
  --config configs/study1/smoke.yaml \
  --project-root . \
  --training-source-images 20
```

The source JPEGs are individually fetched from the exact pinned
`SimulaMet/Kvasir-VQA-x1` revision. The canonical
`SimulaMet-HOST/Kvasir-VQA` revision is also resolved and recorded. This avoids
streaming all embedded-image parquet shards for a 40-image infrastructure gate.

JPEGs remain ignored under `data/images/`. The compact tracked manifest records
their encoded-byte SHA-256, decoded-RGB SHA-256, dimensions, mode, source
filename, selection scope and grouped-split hash.

The pinned gate contains 40 JPEGs (20 development and 20 training), totalling
2,769,492 encoded bytes. Its offline audit passed with no grouped-test or
reserved-contract source IDs.

On a clean Colab or GCP checkout, reconstruct missing images from the tracked
manifest:

```bash
python -m gi_vqa.cli materialize-image-cache \
  --manifest protocols/study1/smoke_training_image_cache_manifest.json \
  --project-root .
```

Run the offline hard gate before model execution:

```bash
python -m gi_vqa.cli image-cache-check \
  --manifest protocols/study1/smoke_training_image_cache_manifest.json \
  --project-root .
```
