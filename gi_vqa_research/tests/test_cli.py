from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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

    def test_prepare_and_check_grouped_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "schema_version": 1,
                "study": "study1",
                "profile": "smoke",
                "seed": 42,
                "data": {
                    "dataset": "example/dataset",
                    "dataset_revision": "d" * 40,
                    "image_dataset": "example/images",
                    "image_dataset_revision": "i" * 40,
                    "split_manifest": (
                        "protocols/study1/grouped_split_manifest.json"
                    ),
                },
                "model": {
                    "base_model_revision": "m" * 40,
                    "adapter": None,
                    "adapter_revision": None,
                },
                "execution": {
                    "evaluation_partition": "development",
                    "max_items": 2,
                    "shard_count": 1,
                },
                "monitoring": {},
                "storage": {},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def raw_record(image_id: str, index: int) -> dict:
                return {
                    "img_id": image_id,
                    "question": f"Question {index}",
                    "answer": f"Answer {index}",
                    "complexity": index % 2,
                    "question_class": ["location"],
                    "original": True,
                }

            train_path = write_jsonl_atomic(
                root / "official_train.jsonl",
                [
                    raw_record("cl8k2u1pv1e4z08320vbv6jzb", 100),
                    *[
                        raw_record(f"train-{index:02d}", index)
                        for index in range(20)
                    ],
                ],
            )
            test_path = write_jsonl_atomic(
                root / "official_test.jsonl",
                [
                    raw_record(f"test-{index:02d}", 20 + index)
                    for index in range(10)
                ],
            )
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "prepare-splits",
                        "--config",
                        "config.json",
                        "--project-root",
                        str(root),
                        "--official-train",
                        str(train_path),
                        "--official-test",
                        str(test_path),
                        "--development-fraction",
                        "0.2",
                        "--test-fraction",
                        "0.2",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn('"status": "PASS"', output.getvalue())

            manifest_path = (
                root / "protocols/study1/grouped_split_manifest.json"
            )
            check_output = io.StringIO()
            with redirect_stdout(check_output):
                check_status = main(
                    [
                        "split-check",
                        "--manifest",
                        str(manifest_path),
                        "--project-root",
                        str(root),
                    ]
                )
            self.assertEqual(check_status, 0)
            self.assertIn('"smoke_items": 2', check_output.getvalue())

    def test_image_cache_commands_delegate_with_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "schema_version": 1,
                "study": "study1",
                "profile": "smoke",
                "seed": 42,
                "data": {
                    "dataset": "example/dataset",
                    "dataset_revision": "d" * 40,
                    "image_dataset": "example/images",
                    "image_dataset_revision": "i" * 40,
                    "split_manifest": "protocols/split.json",
                },
                "model": {"base_model_revision": "m" * 40},
                "execution": {
                    "evaluation_partition": "development",
                    "shard_count": 1,
                },
                "monitoring": {},
                "storage": {},
            }
            (root / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            with patch(
                "gi_vqa.cli.prepare_image_cache",
                return_value={"status": "PASS", "image_count": 4},
            ) as prepare:
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main(
                        [
                            "prepare-image-cache",
                            "--config",
                            "config.json",
                            "--project-root",
                            str(root),
                            "--training-source-images",
                            "2",
                        ]
                    )
            self.assertEqual(status, 0)
            self.assertIn('"image_count": 4', output.getvalue())
            self.assertEqual(
                prepare.call_args.kwargs["split_manifest_path"],
                Path("protocols/split.json"),
            )
            self.assertEqual(
                prepare.call_args.kwargs["training_source_images"],
                2,
            )

            with patch(
                "gi_vqa.cli.verify_image_cache",
                return_value={"status": "PASS", "image_count": 4},
            ) as verify:
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main(
                        [
                            "image-cache-check",
                            "--manifest",
                            "protocols/cache.json",
                            "--project-root",
                            str(root),
                        ]
                    )
            self.assertEqual(status, 0)
            self.assertIn('"status": "PASS"', output.getvalue())
            self.assertEqual(
                verify.call_args.kwargs["manifest_path"],
                Path("protocols/cache.json"),
            )


if __name__ == "__main__":
    unittest.main()
