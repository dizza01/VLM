from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from gi_vqa.cli import main
from gi_vqa.jsonl import read_jsonl, write_jsonl_atomic


class CliTests(unittest.TestCase):
    def test_config_check_accepts_json(self) -> None:
        config = {
            "schema_version": 1,
            "study": "study1",
            "profile": "smoke",
            "data": {
                "dataset_revision": "data-sha",
                "image_dataset_revision": "image-sha",
            },
            "model": {"base_model_revision": "model-sha"},
            "execution": {
                "evaluation_partition": "development",
                "shard_count": 1,
            },
            "monitoring": {},
            "storage": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["config-check", "--config", str(path)])
        self.assertEqual(status, 0)
        self.assertIn('"profile": "smoke"', output.getvalue())

    def test_merge_shards_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_jsonl_atomic(
                root / "part-1.jsonl",
                [{"item_id": "a", "prediction": "one"}],
            )
            second = write_jsonl_atomic(
                root / "part-2.jsonl",
                [{"item_id": "b", "prediction": "two"}],
            )
            destination = root / "merged.jsonl"
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "merge-shards",
                        "--shard",
                        str(second),
                        "--shard",
                        str(first),
                        "--output",
                        str(destination),
                        "--id-field",
                        "item_id",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                [row["item_id"] for row in read_jsonl(destination)],
                ["a", "b"],
            )
            self.assertIn('"record_count": 2', output.getvalue())


if __name__ == "__main__":
    unittest.main()

