from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gi_vqa.jsonl import read_jsonl, write_jsonl_atomic
from gi_vqa.shards import (
    DuplicateRecordIdError,
    ExpectedIdMismatchError,
    OutputExistsError,
    RecordIdError,
    merge_jsonl_shards_atomic,
    validate_jsonl_shards,
)


class ShardTests(unittest.TestCase):
    def test_validation_and_atomic_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_jsonl_atomic(
                root / "shard-00.jsonl",
                [{"item_id": "item-1", "prediction": "yes"}],
            )
            second = write_jsonl_atomic(
                root / "shard-01.jsonl",
                [{"item_id": "item-2", "prediction": "no"}],
            )

            validation = validate_jsonl_shards(
                [second, first],
                id_field="item_id",
                expected_ids={"item-1", "item-2"},
            )
            self.assertEqual(validation.record_count, 2)
            self.assertEqual(validation.shard_count, 2)
            self.assertEqual(validation.expected_id_count, 2)

            output = root / "predictions.jsonl"
            report = merge_jsonl_shards_atomic(
                [second, first],
                output,
                id_field="item_id",
                expected_ids={"item-1", "item-2"},
            )
            self.assertEqual(
                read_jsonl(output),
                [
                    {"item_id": "item-1", "prediction": "yes"},
                    {"item_id": "item-2", "prediction": "no"},
                ],
            )
            self.assertEqual(report.validation.record_count, 2)
            self.assertEqual(report.output["path"], str(output))
            self.assertEqual(report.output["bytes"], output.stat().st_size)

    def test_duplicate_id_across_shards_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_jsonl_atomic(
                root / "one.jsonl", [{"intervention_id": "duplicate"}]
            )
            second = write_jsonl_atomic(
                root / "two.jsonl", [{"intervention_id": "duplicate"}]
            )
            output = root / "merged.jsonl"
            with self.assertRaises(DuplicateRecordIdError) as caught:
                merge_jsonl_shards_atomic(
                    [first, second],
                    output,
                    id_field="intervention_id",
                )
            self.assertEqual(caught.exception.record_id, "duplicate")
            self.assertFalse(output.exists())

    def test_missing_and_extra_expected_ids_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_jsonl_atomic(
                root / "one.jsonl",
                [{"item_id": "expected-1"}, {"item_id": "unexpected"}],
            )
            output = root / "merged.jsonl"
            with self.assertRaises(ExpectedIdMismatchError) as caught:
                merge_jsonl_shards_atomic(
                    [shard],
                    output,
                    id_field="item_id",
                    expected_ids={"expected-1", "expected-2"},
                )
            self.assertEqual(caught.exception.missing_ids, ("expected-2",))
            self.assertEqual(caught.exception.extra_ids, ("unexpected",))
            self.assertFalse(output.exists())

    def test_missing_or_non_string_record_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_jsonl_atomic(
                root / "one.jsonl", [{"item_id": 1}]
            )
            with self.assertRaisesRegex(RecordIdError, "non-empty string"):
                validate_jsonl_shards([shard], id_field="item_id")

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_jsonl_atomic(
                root / "one.jsonl", [{"item_id": "item-1"}]
            )
            output = root / "merged.jsonl"
            output.write_text("sentinel\n", encoding="utf-8")

            with self.assertRaises(OutputExistsError):
                merge_jsonl_shards_atomic(
                    [shard],
                    output,
                    id_field="item_id",
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_duplicate_expected_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_jsonl_atomic(
                root / "one.jsonl", [{"item_id": "item-1"}]
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_jsonl_shards(
                    [shard],
                    id_field="item_id",
                    expected_ids=["item-1", "item-1"],
                )

    def test_empty_id_field_is_rejected_even_for_empty_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_jsonl_atomic(root / "empty.jsonl", [])
            with self.assertRaisesRegex(ValueError, "id_field"):
                validate_jsonl_shards([shard], id_field="")


if __name__ == "__main__":
    unittest.main()
