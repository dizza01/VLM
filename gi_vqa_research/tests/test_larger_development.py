from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gi_vqa.larger_development import (
    CONDITIONS,
    LargerDevelopmentError,
    build_selection_manifest,
    validate_locked_protocol,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LargerDevelopmentTests(unittest.TestCase):
    def test_locked_protocol_and_selection_pass(self) -> None:
        result = validate_locked_protocol(
            project_root=PROJECT_ROOT,
            protocol_path="protocols/study1/larger_development_protocol.json",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["development_items"], 256)
        self.assertEqual(tuple(result["conditions"]), CONDITIONS)
        self.assertFalse(result["test_partition_accessed"])

    def test_selection_is_reconstructable_and_has_no_shuffled_fixed_points(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            output = Path(directory) / "selection.json"
            result = build_selection_manifest(
                project_root=PROJECT_ROOT,
                split_manifest_path="protocols/study1/grouped_split_manifest.json",
                output_path=output,
            )
        tracked = json.loads(
            (
                PROJECT_ROOT
                / "protocols/study1/larger_development_selection.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(result, tracked)
        self.assertEqual(len(result["records"]), 256)
        self.assertTrue(
            all(
                row["source_img_id"] != row["shuffled_source_img_id"]
                for row in result["records"]
            )
        )

    def test_protocol_rejects_test_access(self) -> None:
        protocol_path = (
            PROJECT_ROOT / "protocols/study1/larger_development_protocol.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["test_set_seal"]["access_allowed"] = True
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            temporary = Path(directory) / "protocol.json"
            temporary.write_text(
                json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LargerDevelopmentError, "test-set"):
                validate_locked_protocol(
                    project_root=PROJECT_ROOT,
                    protocol_path=temporary,
                )


if __name__ == "__main__":
    unittest.main()
