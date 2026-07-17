# Locked Study 1 protocol artifacts

This directory is intentionally tracked by Git. It is the destination for the compact,
reviewable artifacts that define a paper experiment:

- `grouped_split_manifest.json`
- `faithfulness_subset_manifest.json`
- `locked_protocol.json`
- `backend_contract.md` (provisional until its CUDA contract test passes)

Do not place images, predictions, checkpoints, saliency arrays or secrets here. Those belong in
the ignored local `runs/` tree and the durable GCS run prefix.

The confirmatory runner must verify these files by hash and refuse to continue when they are
missing or have changed.

The one-item backend contract uses a fixed official-training fixture only for
infrastructure validation. Its source image is absent from the official test
split, but it is still excluded from every research result and must be reserved
when the future grouped split manifest is created.
