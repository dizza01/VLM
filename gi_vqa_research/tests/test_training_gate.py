from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gi_vqa.jsonl import write_jsonl_atomic
from gi_vqa.training import STUDY1_SWIFT_TEMPLATE_TYPE
from gi_vqa.training_gate import (
    TrainingGateFailure,
    build_training_gate_subset,
    build_training_phase_command,
    inspect_training_checkpoint,
    verify_checkpoint_resume,
)


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


class TrainingGateTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "seed": 42,
            "model": {
                "base_model": "example/model",
                "base_model_revision": "m" * 40,
            },
        }

    def test_phase_commands_are_fixed_and_resume_is_explicit(self) -> None:
        phase_one = build_training_phase_command(
            config=self._config(),
            dataset_path="/tmp/train.jsonl",
            output_dir="/tmp/output",
            max_steps=1,
            python_executable="/usr/bin/python3",
        )
        self.assertEqual(
            phase_one[:3],
            ["/usr/bin/python3", "-m", "gi_vqa.training"],
        )
        self.assertEqual(
            _option_value(phase_one, "--template"),
            STUDY1_SWIFT_TEMPLATE_TYPE,
        )
        self.assertEqual(_option_value(phase_one, "--max_steps"), "1")
        self.assertEqual(
            _option_value(phase_one, "--per_device_train_batch_size"),
            "1",
        )
        self.assertEqual(_option_value(phase_one, "--save_steps"), "1")
        self.assertNotIn("--resume_from_checkpoint", phase_one)

        phase_two = build_training_phase_command(
            config=self._config(),
            dataset_path="/tmp/train.jsonl",
            output_dir="/tmp/output",
            max_steps=2,
            resume_from_checkpoint="/tmp/output/checkpoint-1",
        )
        self.assertEqual(_option_value(phase_two, "--max_steps"), "2")
        self.assertEqual(
            _option_value(phase_two, "--resume_from_checkpoint"),
            "/tmp/output/checkpoint-1",
        )
        with self.assertRaisesRegex(
            TrainingGateFailure,
            "phase 2 must resume",
        ):
            build_training_phase_command(
                config=self._config(),
                dataset_path="/tmp/train.jsonl",
                output_dir="/tmp/output",
                max_steps=2,
            )

    def test_subset_selects_one_stable_item_per_locked_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "data/images"
            image_dir.mkdir(parents=True)
            sources = ["source-a", "source-b"]
            images = {}
            records = []
            for source_index, source_id in enumerate(sources):
                image_path = image_dir / f"{source_id}.jpg"
                Image.new(
                    "RGB",
                    (16, 16),
                    color=(source_index * 20, 10, 30),
                ).save(image_path, format="JPEG")
                images[source_id] = {
                    "path": f"data/images/{source_id}.jpg",
                    "scope": "training_loader_smoke",
                }
                for question_index in range(2):
                    records.append(
                        {
                            "item_id": "",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"<image>Question {source_index}-"
                                        f"{question_index}"
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": "Answer",
                                },
                            ],
                            "images": [
                                f"data/images/{source_id}.jpg"
                            ],
                            "metadata": {
                                "source_img_id": source_id,
                                "partition": "train",
                            },
                        }
                    )
            from gi_vqa.identifiers import stable_item_id

            for record in records:
                record["item_id"] = stable_item_id(record)
            train_path = write_jsonl_atomic(root / "train.jsonl", records)
            cache_manifest = root / "cache.json"
            cache_manifest.write_text(
                json.dumps(
                    {
                        "selection": {
                            "training_source_img_ids": sources,
                        },
                        "images": images,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "subset.jsonl"
            result = build_training_gate_subset(
                train_jsonl=train_path,
                image_cache_manifest=cache_manifest,
                output_jsonl=output,
                project_root=root,
                seed=42,
            )
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["unique_source_images"], 2)
            self.assertEqual(result["source_img_ids"], sources)
            self.assertEqual(len(set(result["item_ids"])), 2)

    def _checkpoint(
        self,
        root: Path,
        *,
        step: int,
        adapter_bytes: bytes,
    ) -> Path:
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "target_modules": ["q_proj"],
                    "base_model_name_or_path": "example/model",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(adapter_bytes)
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer" + bytes([step]))
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler" + bytes([step]))
        (checkpoint / "training_args.bin").write_bytes(b"arguments")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "log_history": [
                        {"step": index, "loss": 1.0 / index}
                        for index in range(1, step + 1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def test_checkpoint_resume_requires_state_and_changed_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = inspect_training_checkpoint(
                self._checkpoint(root, step=1, adapter_bytes=b"first"),
                expected_step=1,
            )
            second = inspect_training_checkpoint(
                self._checkpoint(root, step=2, adapter_bytes=b"second"),
                expected_step=2,
            )
            result = verify_checkpoint_resume(first, second)
            self.assertTrue(result["adapter_changed_after_resume"])
            self.assertEqual(result["finished_global_step"], 2)

            second["files"]["adapter_weights"]["sha256"] = first["files"][
                "adapter_weights"
            ]["sha256"]
            with self.assertRaisesRegex(
                TrainingGateFailure,
                "did not change",
            ):
                verify_checkpoint_resume(first, second)


if __name__ == "__main__":
    unittest.main()
