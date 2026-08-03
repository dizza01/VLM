from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gi_vqa.rq1_test_runner import RQ1TestError, run_rq1_full_test
from gi_vqa.rq1_training_runner import (
    RQ1TrainingError,
    _latest_complete_checkpoint,
    prepare_rq1_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "protocols/study1/rq1_full_baseline_protocol_v1.draft.json"


class RQ1RunnerTests(unittest.TestCase):
    def test_test_runner_requires_explicit_authorization_before_loading_data(self) -> None:
        def forbidden_loader(dataset_id, revision):
            raise AssertionError("test loader must not be called")

        with self.assertRaisesRegex(RQ1TestError, "explicit --authorize-test"):
            run_rq1_full_test(
                project_root=PROJECT_ROOT,
                protocol_path=PROTOCOL,
                freeze_receipt_path="missing.json",
                adapter_dir="missing-adapter",
                run_dir="missing-run",
                expected_commit="deadbeef",
                authorize_test=False,
                official_test_loader=forbidden_loader,
            )

    def test_training_prepare_rejects_non_smoke_partial_limits(self) -> None:
        with self.assertRaisesRegex(RQ1TrainingError, "between 1 and 32"):
            prepare_rq1_training_data(
                project_root=PROJECT_ROOT,
                protocol_path=PROTOCOL,
                output_dir="runs/rq1-invalid-test",
                maximum_records=33,
            )

    def test_checkpoint_scan_tolerates_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                _latest_complete_checkpoint(Path(directory) / "missing", 8975)
            )


if __name__ == "__main__":
    unittest.main()
