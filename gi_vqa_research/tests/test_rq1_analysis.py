from __future__ import annotations

import unittest

from gi_vqa.rq1_analysis import (
    _condition_metrics,
    normalized_token_f1,
    rouge_l_f1,
)


class RQ1AnalysisTests(unittest.TestCase):
    def test_text_metrics_are_normalized_and_bounded(self) -> None:
        self.assertEqual(normalized_token_f1("Polyp present", "polyp present"), 1.0)
        self.assertEqual(normalized_token_f1("absent", "present"), 0.0)
        self.assertEqual(rouge_l_f1("one red polyp", "red polyp"), 0.8)

    def test_source_group_bootstrap_is_reproducible(self) -> None:
        rows = [
            {
                "source_img_id": f"image-{index // 2}",
                "normalized_token_f1": float(index % 2),
                "normalized_exact_match": float(index % 2),
                "rouge_l_f1": float(index % 2),
                "prediction_tokens": index + 1,
                "absolute_answer_token_count_error": index,
                "inference_seconds": 0.1 + index,
                "bleu_correct": [index % 2] * 4,
                "bleu_total": [1] * 4,
                "bleu_sys_len": 2,
                "bleu_ref_len": 2,
            }
            for index in range(6)
        ]
        first = _condition_metrics(rows, replicates=100, seed=42)
        second = _condition_metrics(rows, replicates=100, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["completed_items"], 6)
        self.assertIn("corpus_bleu_0_to_1_confidence_interval_95", first)


if __name__ == "__main__":
    unittest.main()
