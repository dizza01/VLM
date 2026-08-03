from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gi_vqa.rq1_baseline import (
    RQ1BaselineError,
    build_execution_plan,
    validate_rq1_protocol,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PROJECT_ROOT
    / "protocols/study1/rq1_full_baseline_protocol_v1.draft.json"
)


class RQ1BaselineTests(unittest.TestCase):
    def test_draft_validates_without_authorizing_or_accessing_test(self) -> None:
        result = validate_rq1_protocol(
            project_root=PROJECT_ROOT,
            protocol_path=PROTOCOL,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["training_records"], 143594)
        self.assertEqual(result["test_records"], 15955)
        self.assertEqual(result["optimizer_steps"], 8975)
        self.assertFalse(result["test_access_authorized"])
        self.assertFalse(result["test_partition_accessed"])
        with self.assertRaisesRegex(RQ1BaselineError, "requires status=LOCKED"):
            validate_rq1_protocol(
                project_root=PROJECT_ROOT,
                protocol_path=PROTOCOL,
                require_locked=True,
            )

    def test_plan_is_gcp_first_and_wandb_is_non_authoritative(self) -> None:
        plan = build_execution_plan(
            project_root=PROJECT_ROOT,
            protocol_path=PROTOCOL,
            wandb_mode="online",
        )
        self.assertEqual(plan["status"], "DRAFT")
        self.assertEqual(
            plan["decision"]["authoritative_platform"], "GCP GPU VM"
        )
        self.assertFalse(plan["wandb"]["authoritative"])
        self.assertFalse(plan["test_gate"]["authorized_now"])
        self.assertFalse(plan["test_partition_accessed"])
        command = plan["training"]["command"]
        self.assertEqual(command[command.index("--max_steps") + 1], "8975")
        self.assertEqual(
            command[command.index("--template") + 1],
            "gi_vqa_paligemma_v1",
        )
        self.assertEqual(command[command.index("--report_to") + 1], "wandb")
        smoke = plan["smoke"]["command"]
        self.assertEqual(smoke[smoke.index("--max_steps") + 1], "1")

    def test_bound_profile_tamper_is_rejected(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            temporary = Path(directory)
            altered_profile = temporary / "profile.json"
            altered_profile.write_text("{}\n", encoding="utf-8")
            protocol["model_profile"]["path"] = str(
                altered_profile.relative_to(PROJECT_ROOT)
            )
            altered_protocol = temporary / "protocol.json"
            altered_protocol.write_text(
                json.dumps(protocol) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RQ1BaselineError, "missing or changed"):
                validate_rq1_protocol(
                    project_root=PROJECT_ROOT,
                    protocol_path=altered_protocol,
                )


if __name__ == "__main__":
    unittest.main()
