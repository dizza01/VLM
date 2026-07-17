from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gi_vqa.audit import (
    SplitLeakageError,
    assert_disjoint_source_images,
    audit_jsonl_splits,
    audit_records,
)
from gi_vqa.jsonl import write_jsonl_atomic


def record(image_id: str, question: str) -> dict:
    return {
        "metadata": {"source_img_id": image_id},
        "messages": [{"role": "user", "content": f"<image>{question}"}],
    }


class AuditTests(unittest.TestCase):
    def test_audit_counts_duplicate_items_and_questions_per_image(self) -> None:
        rows = [
            record("image-1", "Question A"),
            record("image-1", "Question B"),
            record("image-1", "Question A"),
            record("image-2", "Question C"),
        ]
        audit = audit_records(rows)
        self.assertEqual(audit.rows, 4)
        self.assertEqual(audit.unique_source_images, 2)
        self.assertEqual(audit.unique_item_ids, 3)
        self.assertEqual(audit.duplicate_item_rows, 1)
        self.assertEqual(audit.max_questions_per_source_image, 3)

    def test_disjoint_hard_gate_passes(self) -> None:
        report = assert_disjoint_source_images(
            {
                "train": [record("train-1", "Q")],
                "development": [record("dev-1", "Q")],
                "test": [record("test-1", "Q")],
            }
        )
        self.assertTrue(report.is_source_disjoint)
        self.assertEqual(report.overlaps, {})

    def test_disjoint_hard_gate_reports_all_pairwise_leakage(self) -> None:
        with self.assertRaises(SplitLeakageError) as caught:
            assert_disjoint_source_images(
                {
                    "train": [record("shared", "Train")],
                    "development": [record("shared", "Dev")],
                    "test": [record("shared", "Test")],
                }
            )
        self.assertEqual(
            set(caught.exception.overlaps),
            {
                "development<->test",
                "development<->train",
                "test<->train",
            },
        )
        self.assertIn("shared", str(caught.exception))

    def test_jsonl_audit_can_report_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            test_path = Path(directory) / "test.jsonl"
            write_jsonl_atomic(train_path, [record("shared", "Train")])
            write_jsonl_atomic(test_path, [record("shared", "Test")])
            report = audit_jsonl_splits(
                {"train": train_path, "test": test_path}, hard_gate=False
            )
            self.assertFalse(report.is_source_disjoint)
            self.assertEqual(report.as_dict()["overlaps"]["test<->train"], ["shared"])


if __name__ == "__main__":
    unittest.main()
