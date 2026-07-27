from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from gi_vqa.larger_development_analysis import (
    LargerDevelopmentAnalysisError,
    _contrast,
    _ece,
    analyze_larger_development,
    normalized_token_f1,
)
from gi_vqa.provenance import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LargerDevelopmentAnalysisTests(unittest.TestCase):
    def test_normalized_token_f1(self) -> None:
        self.assertEqual(normalized_token_f1("Polyp present", "polyp present"), 1.0)
        self.assertEqual(normalized_token_f1("absent", "present"), 0.0)
        self.assertAlmostEqual(
            normalized_token_f1("one red polyp", "red polyp"),
            0.8,
        )

    def test_calibration_error(self) -> None:
        self.assertAlmostEqual(_ece([0.9, 0.1], [1.0, 0.0]), 0.1)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        better = [
            {"prediction": "polyp present", "reference_answer": "polyp present"}
            for _ in range(8)
        ]
        worse = [
            {"prediction": "no finding", "reference_answer": "polyp present"}
            for _ in range(8)
        ]
        first = _contrast(better, worse, replicates=1000, seed=42)
        second = _contrast(better, worse, replicates=1000, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(first["confidence_interval_95"][0], 0)

    def test_complete_synthetic_analysis_and_incomplete_rejection(self) -> None:
        protocol = (
            PROJECT_ROOT / "protocols/study1/larger_development_protocol.json"
        )
        selection = json.loads(
            (
                PROJECT_ROOT
                / "protocols/study1/larger_development_selection.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "identity.json").write_text(
                json.dumps({"protocol_sha256": file_sha256(protocol)}) + "\n",
                encoding="utf-8",
            )
            conditions = (
                "paired_correct",
                "constant_control",
                "paired_shuffled",
                "paired_neutral_ablation",
                "base_correct_descriptive",
            )
            for condition in conditions:
                item_dir = run / condition / "items"
                item_dir.mkdir(parents=True)
                for rank, selected in enumerate(selection["records"]):
                    prediction = (
                        "polyp present"
                        if condition == "paired_correct"
                        else "no finding"
                    )
                    row = {
                        "condition": condition,
                        "item_id": selected["item_id"],
                        "prediction": prediction,
                        "reference_answer": "polyp present",
                        "sequence_confidence": 0.8,
                        "mean_generated_token_logprob": -0.2,
                    }
                    (item_dir / f"{rank:03d}-{selected['item_id']}.json").write_text(
                        json.dumps(row) + "\n",
                        encoding="utf-8",
                    )
            with patch(
                "gi_vqa.larger_development_analysis._condition_metrics",
                return_value={"normalized_token_f1": 1.0},
            ):
                report = analyze_larger_development(
                    project_root=PROJECT_ROOT,
                    protocol_path=protocol,
                    run_dir=run,
                )
            self.assertTrue(
                report["promotion_decision"]["promote_paired_image_adapter"]
            )
            missing = (
                run
                / "paired_correct/items"
                / f"000-{selection['records'][0]['item_id']}.json"
            )
            missing.unlink()
            (run / "larger_development_analysis.json").unlink()
            with self.assertRaisesRegex(
                LargerDevelopmentAnalysisError, "missing completion"
            ):
                with patch(
                    "gi_vqa.larger_development_analysis._condition_metrics",
                    return_value={"normalized_token_f1": 1.0},
                ):
                    analyze_larger_development(
                        project_root=PROJECT_ROOT,
                        protocol_path=protocol,
                        run_dir=run,
                    )


if __name__ == "__main__":
    unittest.main()
