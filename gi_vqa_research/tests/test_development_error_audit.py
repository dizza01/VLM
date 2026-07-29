from __future__ import annotations

import unittest

from gi_vqa.development_error_audit import (
    CONDITIONS,
    REASONS,
    _blind_items,
    _select_items,
)


def _condition_rows(count: int = 96):
    rows = {condition: {} for condition in CONDITIONS}
    for index in range(count):
        item_id = f"item-{index:03d}"
        reference = "polyp present"
        paired_prediction = "no finding" if index % 3 else reference
        predictions = {
            "paired_correct": paired_prediction,
            "constant_control": "no finding",
            "paired_shuffled": (
                paired_prediction if index % 2 else "no finding"
            ),
            "paired_neutral_ablation": (
                reference if index % 4 else "no finding"
            ),
        }
        for condition in CONDITIONS:
            rows[condition][item_id] = {
                "item_id": item_id,
                "rank": index,
                "source_img_id": f"source-{index:03d}",
                "input_source_img_id": f"input-{condition}-{index:03d}",
                "record_sha256": str(index),
                "reference_answer": reference,
                "prediction": predictions[condition],
                "sequence_confidence": 0.9 - index / 1000,
                "mean_generated_token_logprob": -0.2,
                "complexity": index % 3 + 1,
                "question_class": ["finding_presence"],
            }
    return rows


class DevelopmentErrorAuditTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_unique(self) -> None:
        rows = _condition_rows()
        first = _select_items(rows, total=64, quota=16)
        second = _select_items(rows, total=64, quota=16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(len({row["item_id"] for row in first}), 64)
        counts = {
            reason: sum(row["primary_reason"] == reason for row in first)
            for reason in REASONS
        }
        self.assertEqual(counts, {reason: 16 for reason in REASONS})

    def test_reviewer_rows_are_blinded_and_key_is_separate(self) -> None:
        rows = _condition_rows()
        selected = _select_items(rows, total=64, quota=16)
        questions = {
            item_id: f"Question {item_id}?" for item_id in rows["paired_correct"]
        }
        reviewer, key, items = _blind_items(
            selected, rows, questions=questions, seed=42
        )
        self.assertEqual(len(reviewer), 256)
        self.assertEqual(len(key), 256)
        self.assertEqual(len(items), 64)
        self.assertNotIn("condition", reviewer[0])
        self.assertNotIn("sequence_confidence", reviewer[0])
        self.assertIn("condition", key[0])
        self.assertIn("sequence_confidence", key[0])
        for item in items:
            item_rows = [
                row
                for row in reviewer
                if row["audit_item_id"] == item["audit_item_id"]
            ]
            self.assertEqual(
                {row["output_code"] for row in item_rows},
                {"A", "B", "C", "D"},
            )


if __name__ == "__main__":
    unittest.main()
