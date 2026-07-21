# Locked Study 1 protocol artifacts

This directory is intentionally tracked by Git. It is the destination for the compact,
reviewable artifacts that define a paper experiment:

- `grouped_split_manifest.json`
- `faithfulness_subset_manifest.json`
- `locked_protocol.json`
- `backend_contract.md` (backend boundary and CUDA acceptance criteria)
- `backend_contract_pass.json` (compact receipt for the retained evidence bundle)
- `smoke_training_image_cache_manifest.json` (locked image bytes for the next gate)
- `training_gate_pass.json` (compact receipt for the retained tiny-LoRA evidence bundle)

Do not place images, predictions, checkpoints, saliency arrays or secrets here. Those belong in
the ignored local `runs/` tree and the durable GCS run prefix.

The confirmatory runner must verify these files by hash and refuse to continue when they are
missing or have changed.

The one-item backend contract used a fixed official-training fixture only for
infrastructure validation. Its source image is absent from the official test
split, but it is still excluded from every research result and must be reserved
by every grouped split.

Contract v2 passed on the T4 reference environment. The package-level
`prepare-splits` command subsequently produced `grouped_split_manifest.json`
from the pinned dataset revision. Its independent `split-check` gate passed:
all three primary partitions are source-image disjoint, the contract fixture is
reserved, and all nine reconstructable artifacts match their recorded hashes.

The image cache gate materialises only the 20 development smoke source images
and 20 deterministic training source images. Its tracked manifest locks encoded
and decoded-RGB hashes without committing the JPEGs themselves. The real pinned
40-image cache and its offline integrity/no-test-contact audit passed.

The two-process tiny-LoRA gate passed all 15 checks on the reference Colab T4
at Git commit `da94b251c0f49d4fa74e4351c3487f5ce3286ade`. It proved complete
step-1 and step-2 checkpoints, explicit resume from step 1, changed adapter
weights after the resumed step, and independent PEFT reload with finite loss.
The compact tracked receipt preserves the evidence hashes and the observed
diagnostic caveats. The resulting restart-safe 20-item development runner is
now implemented. Its next T4 run must be retained as a separate evidence bundle;
it is not a research model or research result.
