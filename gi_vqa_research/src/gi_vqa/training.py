"""Study 1 training compatibility and the project-owned ms-swift entrypoint.

The pinned ms-swift 3.7.0 PaliGemma template has a one-token off-by-one in
training ``token_type_ids``.  This module keeps the correction dependency-light
so it can be unit-tested locally.  The optional Swift plugin applies it during
real training and the executable backend contract validates that same plugin.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STUDY1_SWIFT_TEMPLATE_TYPE = "gi_vqa_paligemma_v1"
SWIFT_TOKEN_TYPE_CORRECTION_ID = (
    "ms-swift-3.7.0-paligemma-prefix-boundary-v1"
)
EXPECTED_MS_SWIFT_VERSION = "3.7.0"
IGNORE_LABEL_ID = -100


class TrainingCompatibilityError(RuntimeError):
    """Raised when training preprocessing is not the validated Study 1 path."""


@dataclass(frozen=True)
class SwiftTokenTypeCorrection:
    """Auditable description of one Swift PaliGemma token-type correction."""

    correction_id: str
    package_version: str
    action: str
    sequence_length: int
    prefix_length: int
    target_length: int
    mismatch_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "package_version": self.package_version,
            "action": self.action,
            "sequence_length": self.sequence_length,
            "prefix_length": self.prefix_length,
            "target_length": self.target_length,
            "mismatch_count": len(self.mismatch_indices),
            "mismatch_indices": list(self.mismatch_indices),
        }


def canonical_paligemma_training_token_type_ids(
    labels: Sequence[int],
) -> tuple[int, ...]:
    """Return canonical PaliGemma prefix/suffix types from training labels.

    A valid sequence has one non-empty ignored prompt prefix followed by one
    non-empty, contiguous supervised answer suffix.  PaliGemma requires type 0
    for the entire prompt prefix and type 1 for the answer suffix.
    """

    label_values = tuple(int(value) for value in labels)
    if not label_values:
        raise TrainingCompatibilityError("training labels must not be empty")

    try:
        target_start = next(
            index
            for index, value in enumerate(label_values)
            if value != IGNORE_LABEL_ID
        )
    except StopIteration as exc:
        raise TrainingCompatibilityError(
            "training labels contain no supervised answer suffix"
        ) from exc

    if target_start == 0:
        raise TrainingCompatibilityError(
            "training labels contain no ignored prompt prefix"
        )
    if any(
        value == IGNORE_LABEL_ID for value in label_values[target_start:]
    ):
        raise TrainingCompatibilityError(
            "training labels are not one contiguous ignored prefix followed "
            "by one contiguous supervised suffix"
        )

    return (0,) * target_start + (1,) * (len(label_values) - target_start)


def correct_ms_swift_paligemma_training_encoding(
    encoded: Mapping[str, Any],
    *,
    package_version: str,
) -> tuple[dict[str, Any], SwiftTokenTypeCorrection]:
    """Correct only the pinned, known ms-swift PaliGemma boundary defect.

    Arbitrary differences are rejected.  This prevents a future upstream or
    configuration change from being silently hidden by the compatibility
    layer.
    """

    required = {"input_ids", "labels", "token_type_ids"}
    missing = required - set(encoded)
    if missing:
        raise TrainingCompatibilityError(
            f"Swift training encoding is missing fields: {sorted(missing)}"
        )

    input_ids = tuple(int(value) for value in encoded["input_ids"])
    labels = tuple(int(value) for value in encoded["labels"])
    raw_token_types = tuple(
        int(value) for value in encoded["token_type_ids"]
    )
    lengths = {
        "input_ids": len(input_ids),
        "labels": len(labels),
        "token_type_ids": len(raw_token_types),
    }
    if len(set(lengths.values())) != 1:
        raise TrainingCompatibilityError(
            f"Swift training sequence lengths differ: {lengths}"
        )
    if not all(value in (0, 1) for value in raw_token_types):
        raise TrainingCompatibilityError(
            "Swift PaliGemma token_type_ids must contain only 0 and 1"
        )

    canonical = canonical_paligemma_training_token_type_ids(labels)
    mismatch_indices = tuple(
        index
        for index, (raw, expected) in enumerate(
            zip(raw_token_types, canonical)  # noqa: B905 - lengths checked above
        )
        if raw != expected
    )
    prefix_length = canonical.count(0)

    if not mismatch_indices:
        action = "already_canonical"
    else:
        known_boundary_index = prefix_length - 1
        is_known_defect = (
            package_version == EXPECTED_MS_SWIFT_VERSION
            and mismatch_indices == (known_boundary_index,)
            and raw_token_types[known_boundary_index] == 1
            and canonical[known_boundary_index] == 0
            and labels[known_boundary_index] == IGNORE_LABEL_ID
        )
        if not is_known_defect:
            raise TrainingCompatibilityError(
                "Swift PaliGemma token types differ outside the validated "
                f"{SWIFT_TOKEN_TYPE_CORRECTION_ID} pattern: "
                f"package_version={package_version!r}, "
                f"mismatch_indices={list(mismatch_indices)}, "
                f"prefix_length={prefix_length}"
            )
        action = "corrected_known_boundary_off_by_one"

    corrected = dict(encoded)
    corrected["token_type_ids"] = list(canonical)
    correction = SwiftTokenTypeCorrection(
        correction_id=SWIFT_TOKEN_TYPE_CORRECTION_ID,
        package_version=package_version,
        action=action,
        sequence_length=len(labels),
        prefix_length=prefix_length,
        target_length=len(labels) - prefix_length,
        mismatch_indices=mismatch_indices,
    )
    return corrected, correction


def _option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise TrainingCompatibilityError(
                    f"{option} requires an explicit value"
                )
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        index += 1
    return values


def _prepare_sft_argv(argv: Sequence[str]) -> list[str]:
    """Force training through the versioned project template plugin."""

    prepared = [str(value) for value in argv]
    if prepared and prepared[0].endswith(".json"):
        raise TrainingCompatibilityError(
            "JSON-only ms-swift argument files are not supported by the Study "
            "1 wrapper; pass explicit CLI arguments so the template gate can "
            "be verified"
        )

    template_values = _option_values(prepared, "--template")
    if any(
        value != STUDY1_SWIFT_TEMPLATE_TYPE for value in template_values
    ):
        raise TrainingCompatibilityError(
            "Study 1 training must use "
            f"--template {STUDY1_SWIFT_TEMPLATE_TYPE}"
        )
    if len(template_values) > 1:
        raise TrainingCompatibilityError(
            "--template must be supplied at most once"
        )
    if not template_values:
        prepared.extend(["--template", STUDY1_SWIFT_TEMPLATE_TYPE])

    plugin_path = str(
        Path(__file__).with_name("swift_paligemma_plugin.py").resolve()
    )
    external_plugins = _option_values(prepared, "--external_plugins")
    if plugin_path not in external_plugins:
        prepared.extend(["--external_plugins", plugin_path])
    return prepared


def run_sft(argv: Sequence[str] | None = None) -> Any:
    """Run pinned ms-swift SFT with the Study 1 template hard-wired."""

    try:
        from swift.llm import sft_main
    except ImportError as exc:
        raise TrainingCompatibilityError(
            "Study 1 training requires the pinned GPU extra, including "
            f"ms-swift {EXPECTED_MS_SWIFT_VERSION}"
        ) from exc

    arguments = sys.argv[1:] if argv is None else list(argv)
    return sft_main(_prepare_sft_argv(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    run_sft(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
