from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from gi_vqa.controlled_training import (
    CONTROLLED_TRAINING_CONDITIONS,
    ControlledTrainingError,
    build_controlled_training_command,
    load_controlled_training_protocol,
    prepare_controlled_training_data,
    verify_controlled_training_data,
)
from gi_vqa.identifiers import stable_item_id
from gi_vqa.jsonl import read_jsonl, write_jsonl_atomic
from gi_vqa.provenance import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "protocols/study1/controlled_training_pilot.json"


class ControlledTrainingTests(unittest.TestCase):
    def test_locked_protocol_builds_matched_restart_commands(self) -> None:
        protocol = load_controlled_training_protocol(PROTOCOL_PATH)
        first = build_controlled_training_command(
            protocol=protocol,
            condition="paired_image",
            dataset_path="paired.jsonl",
            output_dir="phase-one",
            max_steps=128,
            python_executable="python",
        )
        second = build_controlled_training_command(
            protocol=protocol,
            condition="constant_image",
            dataset_path="constant.jsonl",
            output_dir="phase-two",
            max_steps=256,
            resume_from_checkpoint="checkpoint-128",
            python_executable="python",
        )

        self.assertEqual(first[first.index("--max_steps") + 1], "128")
        self.assertNotIn("--resume_from_checkpoint", first)
        self.assertEqual(second[second.index("--max_steps") + 1], "256")
        self.assertEqual(
            second[second.index("--resume_from_checkpoint") + 1],
            "checkpoint-128",
        )
        for option, expected in (
            ("--gradient_accumulation_steps", "4"),
            ("--lora_rank", "16"),
            ("--lora_alpha", "32"),
            ("--lora_dropout", "0.05"),
            ("--freeze_vit", "true"),
            ("--freeze_aligner", "true"),
            ("--target_modules", "all-linear"),
        ):
            self.assertEqual(first[first.index(option) + 1], expected)
            self.assertEqual(second[second.index(option) + 1], expected)

    def test_phase_boundary_requires_explicit_resume(self) -> None:
        protocol = load_controlled_training_protocol(PROTOCOL_PATH)
        with self.assertRaisesRegex(ControlledTrainingError, "explicitly resume"):
            build_controlled_training_command(
                protocol=protocol,
                condition="paired_image",
                dataset_path="paired.jsonl",
                output_dir="output",
                max_steps=256,
            )

    def test_prepare_creates_identical_train_only_arms_and_is_restart_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_dir = root / "protocols/study1"
            protocol_dir.mkdir(parents=True)
            data_dir = root / "data/processed/study1"
            data_dir.mkdir(parents=True)
            train_path = data_dir / "train.jsonl"
            records = []
            for source_index in range(256):
                source_id = f"source-{source_index:03d}"
                for question_index in range(4):
                    record = {
                        "images": [f"data/images/{source_id}.jpg"],
                        "messages": [
                            {
                                "role": "user",
                                "content": f"<image>Question {question_index}?",
                            },
                            {
                                "role": "assistant",
                                "content": f"Answer {source_index}-{question_index}",
                            },
                        ],
                        "metadata": {
                            "partition": "train",
                            "source_img_id": source_id,
                        },
                    }
                    record["item_id"] = stable_item_id(record)
                    records.append(record)
            write_jsonl_atomic(train_path, records)

            split_path = protocol_dir / "grouped_split_manifest.json"
            split_payload = {
                "artifacts": {
                    "train": {
                        "path": str(train_path.relative_to(root)),
                        "sha256": file_sha256(train_path),
                    }
                },
                "dataset": {"id": "metadata/repo", "revision": "a" * 40},
                "image_dataset": {"id": "image/repo", "revision": "b" * 40},
            }
            split_path.write_text(
                json.dumps(split_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
            protocol["split_manifest_sha256"] = file_sha256(split_path)
            receipt_hashes = {}
            for filename in protocol["required_pass_receipts"]:
                receipt = protocol_dir / filename
                receipt.write_text('{"status":"PASS"}\n', encoding="utf-8")
                receipt_hashes[filename] = file_sha256(receipt)
            protocol["required_pass_receipts"] = receipt_hashes
            protocol_path = protocol_dir / "controlled_training_pilot.json"
            protocol_path.write_text(
                json.dumps(protocol, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            fixture_image = root / "fixture.jpg"
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(fixture_image)
            fetch_calls = []

            def fetch_image(
                dataset_id: str,
                revision: str,
                filename: str,
                token: str | None,
            ) -> Path:
                fetch_calls.append((dataset_id, revision, filename, token))
                return fixture_image

            output = root / "data/controlled_training_pilot"
            with patch(
                "gi_vqa.controlled_training.verify_grouped_split_artifacts",
                return_value={"status": "PASS", "test_partition_accessed": False},
            ):
                result = prepare_controlled_training_data(
                    protocol_path=protocol_path,
                    split_manifest_path=split_path,
                    project_root=root,
                    output_dir=output,
                    fetch_image=fetch_image,
                    materialize_splits=False,
                )
                repeated = prepare_controlled_training_data(
                    protocol_path=protocol_path,
                    split_manifest_path=split_path,
                    project_root=root,
                    output_dir=output,
                    fetch_image=fetch_image,
                    materialize_splits=False,
                )

            self.assertEqual(result, repeated)
            self.assertEqual(result["source_images"], 256)
            self.assertEqual(result["records_per_condition"], 1024)
            self.assertEqual(len(fetch_calls), 256)
            self.assertTrue(
                all(call[:2] == ("image/repo", "b" * 40) for call in fetch_calls)
            )
            paired = read_jsonl(output / "paired_image_train.jsonl")
            constant = read_jsonl(output / "constant_image_train.jsonl")
            self.assertEqual(
                [record["item_id"] for record in paired],
                [record["item_id"] for record in constant],
            )
            self.assertEqual(
                [record["messages"] for record in paired],
                [record["messages"] for record in constant],
            )
            self.assertEqual(
                {record["images"][0] for record in constant},
                {"data/controlled_training_pilot/constant_image.png"},
            )
            self.assertEqual(
                tuple(result["conditions"]),
                CONTROLLED_TRAINING_CONDITIONS,
            )
            verified = verify_controlled_training_data(
                manifest_path=output / "controlled_training_data_manifest.json",
                protocol_path=protocol_path,
                split_manifest_path=split_path,
                project_root=root,
            )
            self.assertFalse(verified["test_partition_accessed"])


if __name__ == "__main__":
    unittest.main()
