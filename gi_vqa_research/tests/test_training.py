from __future__ import annotations

import hashlib
import json
import unittest

from gi_vqa.training import (
    EXPECTED_MS_SWIFT_VERSION,
    STUDY1_SWIFT_TEMPLATE_TYPE,
    SWIFT_TOKEN_TYPE_CORRECTION_ID,
    TrainingCompatibilityError,
    _prepare_sft_argv,
    canonical_paligemma_training_token_type_ids,
    correct_ms_swift_paligemma_training_encoding,
)


def sequence_sha256(values: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(
        [int(value) for value in values],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TrainingCompatibilityTests(unittest.TestCase):
    def reported_failure_encoding(self) -> dict:
        return {
            "input_ids": list(range(276)),
            "labels": [-100] * 274 + [41, 1],
            "token_type_ids": [0] * 273 + [1] * 3,
        }

    def test_reported_swift_boundary_failure_is_corrected_exactly(self) -> None:
        encoded = self.reported_failure_encoding()
        corrected, report = correct_ms_swift_paligemma_training_encoding(
            encoded,
            package_version=EXPECTED_MS_SWIFT_VERSION,
        )

        self.assertEqual(
            sequence_sha256(encoded["token_type_ids"]),
            "fc928d13e87812712d8958612575f5e778ddb44055489bcab0a6d3891074460a",
        )
        self.assertEqual(
            sequence_sha256(corrected["token_type_ids"]),
            "560c56d52576cc68a0e7c8586d5bd1e42e4ae6274fc75ee6717dca0a65e3f81e",
        )
        self.assertEqual(report.correction_id, SWIFT_TOKEN_TYPE_CORRECTION_ID)
        self.assertEqual(report.action, "corrected_known_boundary_off_by_one")
        self.assertEqual(report.mismatch_indices, (273,))
        self.assertEqual(report.prefix_length, 274)
        self.assertEqual(report.target_length, 2)
        self.assertEqual(encoded["token_type_ids"][273], 1)
        self.assertEqual(corrected["token_type_ids"][273], 0)
        self.assertEqual(
            [
                index
                for index, (before, after) in enumerate(
                    zip(  # noqa: B905 - equal lengths are under test
                        encoded["token_type_ids"],
                        corrected["token_type_ids"],
                    )
                )
                if before != after
            ],
            [273],
        )

    def test_already_canonical_encoding_is_a_no_op(self) -> None:
        encoded = {
            "input_ids": [10, 11, 12, 13],
            "labels": [-100, -100, 12, 13],
            "token_type_ids": [0, 0, 1, 1],
        }
        corrected, report = correct_ms_swift_paligemma_training_encoding(
            encoded,
            package_version=EXPECTED_MS_SWIFT_VERSION,
        )
        self.assertEqual(corrected["token_type_ids"], [0, 0, 1, 1])
        self.assertEqual(report.action, "already_canonical")
        self.assertEqual(report.mismatch_indices, ())

    def test_canonical_types_require_one_contiguous_prefix_and_suffix(self) -> None:
        self.assertEqual(
            canonical_paligemma_training_token_type_ids(
                [-100, -100, 7, 8]
            ),
            (0, 0, 1, 1),
        )
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "no supervised answer",
        ):
            canonical_paligemma_training_token_type_ids([-100, -100])
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "no ignored prompt",
        ):
            canonical_paligemma_training_token_type_ids([7, 8])
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "not one contiguous",
        ):
            canonical_paligemma_training_token_type_ids(
                [-100, 7, -100, 8]
            )

    def test_unexpected_token_type_difference_is_rejected(self) -> None:
        encoded = self.reported_failure_encoding()
        encoded["token_type_ids"][10] = 1
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "outside the validated",
        ):
            correct_ms_swift_paligemma_training_encoding(
                encoded,
                package_version=EXPECTED_MS_SWIFT_VERSION,
            )

        encoded = self.reported_failure_encoding()
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "outside the validated",
        ):
            correct_ms_swift_paligemma_training_encoding(
                encoded,
                package_version="3.7.1",
            )

    def test_sequence_length_and_required_field_errors_are_rejected(self) -> None:
        encoded = self.reported_failure_encoding()
        encoded["labels"] = encoded["labels"][:-1]
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "sequence lengths differ",
        ):
            correct_ms_swift_paligemma_training_encoding(
                encoded,
                package_version=EXPECTED_MS_SWIFT_VERSION,
            )

        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "missing fields",
        ):
            correct_ms_swift_paligemma_training_encoding(
                {"input_ids": [1]},
                package_version=EXPECTED_MS_SWIFT_VERSION,
            )

    def test_training_argv_forces_project_template_and_plugin(self) -> None:
        prepared = _prepare_sft_argv(["--model", "model-id"])
        self.assertIn("--template", prepared)
        template_index = prepared.index("--template")
        self.assertEqual(
            prepared[template_index + 1],
            STUDY1_SWIFT_TEMPLATE_TYPE,
        )
        self.assertIn("--external_plugins", prepared)
        plugin_index = prepared.index("--external_plugins")
        self.assertTrue(
            prepared[plugin_index + 1].endswith(
                "gi_vqa/swift_paligemma_plugin.py"
            )
        )

        explicit = _prepare_sft_argv(
            [
                "--model",
                "model-id",
                "--template",
                STUDY1_SWIFT_TEMPLATE_TYPE,
            ]
        )
        self.assertEqual(explicit.count("--template"), 1)

    def test_training_argv_rejects_template_bypass(self) -> None:
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "must use --template",
        ):
            _prepare_sft_argv(
                ["--model", "model-id", "--template", "paligemma"]
            )
        with self.assertRaisesRegex(
            TrainingCompatibilityError,
            "JSON-only",
        ):
            _prepare_sft_argv(["train_args.json"])


if __name__ == "__main__":
    unittest.main()
