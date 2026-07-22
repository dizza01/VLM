from __future__ import annotations

import unittest
from pathlib import Path

from gi_vqa.config import load_config, validate_config
from gi_vqa.controlled_evaluation_runner import (
    CONDITIONS,
    _condition_config,
    _read_object,
    _validate_protocol,
    _validate_training_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ControlledEvaluationTests(unittest.TestCase):
    def test_passed_bundle_matches_compact_receipt(self) -> None:
        receipt_path = PROJECT_ROOT / "protocols/study1/controlled_training_pass.json"
        protocol = _read_object(
            PROJECT_ROOT / "protocols/study1/controlled_evaluation_pilot.json"
        )
        receipt = _read_object(receipt_path)
        _validate_protocol(protocol, receipt_path)
        _validate_training_bundle(PROJECT_ROOT / "controlled_training_bundle-4", receipt)

    def test_condition_configs_bind_each_adapter(self) -> None:
        base = validate_config(
            load_config(PROJECT_ROOT / "configs/study1/smoke.yaml"),
            require_resolved=True,
            require_model_execution=True,
        )
        receipt = _read_object(
            PROJECT_ROOT / "protocols/study1/controlled_training_pass.json"
        )
        bundle = PROJECT_ROOT / "controlled_training_bundle-4"
        configs = {
            condition: _condition_config(
                base, condition=condition, bundle=bundle, receipt=receipt
            )
            for condition in CONDITIONS
        }
        self.assertEqual(configs["unadapted_paired_image"]["model"]["condition"], "base")
        self.assertIsNone(configs["unadapted_paired_image"]["model"]["adapter"])
        self.assertEqual(
            configs["paired_image_adapter"]["model"]["adapter"],
            str(bundle / "adapters/paired_image"),
        )
        self.assertEqual(
            configs["constant_image_adapter"]["model"]["adapter"],
            str(bundle / "adapters/constant_image"),
        )


if __name__ == "__main__":
    unittest.main()
