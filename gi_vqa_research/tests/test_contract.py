from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gi_vqa.backends import GenerationResult
from gi_vqa.contract import (
    DIAGNOSTIC_COMPLEXITY,
    DIAGNOSTIC_IMAGE_SHA256,
    DIAGNOSTIC_IMG_ID,
    DIAGNOSTIC_QUESTION,
    DIAGNOSTIC_REFERENCE_ANSWER,
    ContractFailure,
    _array_summary,
    _atomic_write_json,
    _integer_sequence_sha256,
    _repeat_logprob_difference,
    _sanitise_text,
    _validate_diagnostic_row,
    run_contract,
)
from gi_vqa.model_spec import PaliGemmaModelSpec


class ContractTests(unittest.TestCase):
    def diagnostic_row(self) -> dict:
        return {
            "img_id": DIAGNOSTIC_IMG_ID,
            "question": DIAGNOSTIC_QUESTION,
            "answer": DIAGNOSTIC_REFERENCE_ANSWER,
            "complexity": DIAGNOSTIC_COMPLEXITY,
            "image": f"https://example.invalid/images/{DIAGNOSTIC_IMG_ID}.jpg",
            "question_class": ["location", "size"],
        }

    def generation(self, logprobs: tuple[float, ...]) -> GenerationResult:
        spec = PaliGemmaModelSpec(
            base_model_id="google/paligemma-3b-pt-224",
            base_model_revision="a" * 40,
        )
        provenance = spec.to_provenance(backend_version="1", device="cuda:0")
        return GenerationResult(
            text="answer",
            token_ids=(11, 12),
            token_logprobs=logprobs,
            provenance=provenance,
        )

    def test_fixed_diagnostic_row_is_exact(self) -> None:
        normalized = _validate_diagnostic_row(self.diagnostic_row())
        self.assertEqual(normalized["img_id"], DIAGNOSTIC_IMG_ID)
        self.assertEqual(normalized["complexity"], DIAGNOSTIC_COMPLEXITY)
        self.assertEqual(len(DIAGNOSTIC_IMAGE_SHA256), 64)

        changed = self.diagnostic_row()
        changed["answer"] = "changed"
        with self.assertRaisesRegex(ContractFailure, "content changed"):
            _validate_diagnostic_row(changed)

        missing = self.diagnostic_row()
        del missing["image"]
        with self.assertRaisesRegex(ContractFailure, "missing required fields"):
            _validate_diagnostic_row(missing)

    def test_sequence_hash_is_stable_and_order_sensitive(self) -> None:
        first = _integer_sequence_sha256((1, 2, 3))
        self.assertEqual(first, _integer_sequence_sha256([1, 2, 3]))
        self.assertNotEqual(first, _integer_sequence_sha256((3, 2, 1)))
        self.assertEqual(len(first), 64)

    def test_array_summary_requires_normalized_float32_map(self) -> None:
        report: dict = {}
        values = np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        array, summary = _array_summary(
            report,
            np,
            name="map",
            values=values,
            expected_shape=(2, 2),
        )
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(summary["shape"], [2, 2])
        self.assertEqual(len(summary["sha256"]), 64)
        self.assertTrue(all(check["passed"] for check in report["checks"].values()))

        with self.assertRaisesRegex(ContractFailure, "map_dtype"):
            _array_summary(
                {},
                np,
                name="map",
                values=values.astype(np.float64),
                expected_shape=(2, 2),
            )

    def test_repeat_logprob_difference_handles_mismatch(self) -> None:
        first = self.generation((-0.2, -0.4))
        second = self.generation((-0.2005, -0.399))
        self.assertAlmostEqual(_repeat_logprob_difference(first, second), 0.001)

        shorter = GenerationResult(
            text="answer",
            token_ids=(11,),
            token_logprobs=(-0.2,),
            provenance=first.provenance,
        )
        self.assertTrue(math.isinf(_repeat_logprob_difference(first, shorter)))

    def test_report_write_is_atomic_json_and_tokens_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _atomic_write_json(path, {"status": "PASS", "value": 1})
            self.assertEqual(json.loads(path.read_text()), {"status": "PASS", "value": 1})
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())

        secret = "hf_abcdefghijklmnopqrstuvwxyz"
        self.assertNotIn(secret, _sanitise_text(f"failure for {secret}"))
        self.assertIn("<redacted-token>", _sanitise_text(f"failure for {secret}"))
        signed_url = "https://example.invalid/file?X-Amz-Signature=secret&part=1"
        self.assertNotIn("secret", _sanitise_text(signed_url))
        bearer = "Authorization: Bearer opaque-value"
        self.assertNotIn("opaque-value", _sanitise_text(bearer))

    def test_early_failure_still_writes_machine_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "artifacts"
            with self.assertRaisesRegex(
                ContractFailure,
                "required_gpu_substring",
            ):
                run_contract(
                    config_path=root / "missing.yaml",
                    repository_root=root,
                    expected_commit="a" * 40,
                    artifact_dir=artifact_dir,
                    required_gpu_substring="",
                )
            report = json.loads(
                (artifact_dir / "contract_report.json").read_text()
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertNotIn("attribution_maps", report["artifact_paths"])
            self.assertEqual(
                report["error"]["type"],
                "gi_vqa.contract.ContractFailure",
            )


if __name__ == "__main__":
    unittest.main()
