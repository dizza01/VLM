from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gi_vqa.provenance import file_sha256
from gi_vqa.rq1_visualizations import render_rq1_visualizations


class RQ1VisualizationTests(unittest.TestCase):
    def _report(self):
        metrics = {}
        for condition, offset in (
            ("fine_tuned_baseline", 0.1),
            ("unadapted_base", 0.0),
        ):
            metrics[condition] = {
                "normalized_token_f1": 0.5 + offset,
                "normalized_exact_match": 0.3 + offset,
                "rouge_l_f1": 0.48 + offset,
                "corpus_bleu_0_to_1": 0.25 + offset,
                "mean_absolute_answer_token_count_error": 1.5 - offset,
            }
        strata = {
            "question_class": [
                {
                    "question_class": "finding_presence",
                    "condition": condition,
                    "items": 40,
                    "normalized_token_f1": 0.5 + offset,
                }
                for condition, offset in (
                    ("fine_tuned_baseline", 0.1),
                    ("unadapted_base", 0.0),
                )
            ],
            "complexity": [
                {
                    "complexity": str(level),
                    "condition": condition,
                    "items": 20,
                    "normalized_token_f1": 0.4 + offset + level / 100,
                }
                for level in (1, 2, 3)
                for condition, offset in (
                    ("fine_tuned_baseline", 0.1),
                    ("unadapted_base", 0.0),
                )
            ],
        }
        return {
            "schema_version": "gi-vqa-rq1-benchmark-report-v1",
            "status": "PASS",
            "test_partition_accessed": True,
            "condition_metrics": metrics,
            "strata": strata,
        }

    def test_visualizations_are_deterministic_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(
                json.dumps(self._report(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = root / "figures"
            first = render_rq1_visualizations(
                report_path=report, output_dir=output
            )
            second = render_rq1_visualizations(
                report_path=report, output_dir=output
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first["artifacts"]), 8)
            for relative, descriptor in first["artifacts"].items():
                path = output / relative
                self.assertEqual(file_sha256(path), descriptor["sha256"])
            svg = (output / "headline_metrics.svg").read_text(encoding="utf-8")
            self.assertIn("RQ1 full-test benchmark", svg)
            self.assertIn("fine_tuned_baseline", svg)


if __name__ == "__main__":
    unittest.main()
