from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gi_vqa.backends import (
    BackendProvenance,
    GenerationResult,
    PreparedInput,
)
from gi_vqa.larger_development_runner import (
    _condition_image,
    _publish_or_match,
    _run_condition,
)


class FakeBackend:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.provenance = BackendProvenance(
            backend_name="fake",
            backend_version="1",
            model_id="fake/model",
            model_revision="a" * 40,
            model_spec_sha256="b" * 64,
            processor_id="fake/model",
            processor_revision="a" * 40,
            torch_dtype="float16",
            attention_implementation="eager",
            device="cuda:0",
            software_versions=(("fake", "1"),),
        )

    def prepare(self, *, item_id, image, question):
        return PreparedInput(
            item_id=item_id,
            question=question,
            prompt=question,
            payload={},
            provenance=self.provenance,
            input_token_ids=(1,),
            image_token_indices=(0,),
            preprocessing={},
        )

    def generate(self, prepared):
        self.generate_calls += 1
        return GenerationResult(
            text="polyp present",
            token_ids=(2, 3),
            token_logprobs=(-0.1, -0.2),
            provenance=self.provenance,
        )

    def close(self):
        return None


class LargerDevelopmentRunnerTests(unittest.TestCase):
    def test_locked_condition_image_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = root / "correct.jpg"
            shuffled = root / "shuffled.jpg"
            neutral = root / "neutral.png"
            images = {"correct": correct, "shuffled": shuffled}
            self.assertEqual(
                _condition_image(
                    "paired_correct",
                    source_id="correct",
                    shuffled_source_id="shuffled",
                    images=images,
                    constant_image=neutral,
                ),
                correct,
            )
            self.assertEqual(
                _condition_image(
                    "paired_shuffled",
                    source_id="correct",
                    shuffled_source_id="shuffled",
                    images=images,
                    constant_image=neutral,
                ),
                shuffled,
            )
            for condition in ("constant_control", "paired_neutral_ablation"):
                self.assertEqual(
                    _condition_image(
                        condition,
                        source_id="correct",
                        shuffled_source_id="shuffled",
                        images=images,
                        constant_image=neutral,
                    ),
                    neutral,
                )

    def test_completion_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item.json"
            _publish_or_match(path, {"item_id": "one", "prediction": "yes"})
            _publish_or_match(path, {"item_id": "one", "prediction": "yes"})
            with self.assertRaisesRegex(Exception, "existing artifact differs"):
                _publish_or_match(path, {"item_id": "one", "prediction": "no"})

    def test_condition_interrupts_resumes_and_rejects_stale_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = root / "correct.jpg"
            shuffled = root / "shuffled.jpg"
            neutral = root / "neutral.png"
            for path in (correct, shuffled, neutral):
                path.write_bytes(path.name.encode())
            records = [
                {
                    "item_id": f"item-{index}",
                    "messages": [
                        {"role": "user", "content": "<image>Finding?"},
                        {"role": "assistant", "content": "polyp present"},
                    ],
                    "metadata": {
                        "complexity": 1,
                        "question_class": ["finding_presence"],
                    },
                }
                for index in range(3)
            ]
            selection = {
                "records": [
                    {
                        "item_id": record["item_id"],
                        "source_img_id": "correct",
                        "shuffled_source_img_id": "shuffled",
                        "record_sha256": f"hash-{index}",
                    }
                    for index, record in enumerate(records)
                ]
            }
            backend = FakeBackend()
            output = root / "run"
            first = _run_condition(
                condition="paired_correct",
                records=records,
                selection=selection,
                images={"correct": correct, "shuffled": shuffled},
                constant_image=neutral,
                output=output,
                backend=backend,
                max_new_items=1,
            )
            self.assertEqual(first["status"], "INCOMPLETE")
            self.assertEqual(first["new_items"], 1)
            second = _run_condition(
                condition="paired_correct",
                records=records,
                selection=selection,
                images={"correct": correct, "shuffled": shuffled},
                constant_image=neutral,
                output=output,
                backend=backend,
                max_new_items=None,
            )
            self.assertEqual(second["status"], "COMPLETE")
            self.assertEqual(second["reused_items"], 1)
            completion = output / "items/000-item-0.json"
            saved = completion.read_text(encoding="utf-8")
            completion.write_text(
                saved.replace('"condition": "paired_correct"', '"condition": "wrong"'),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "stale completion"):
                _run_condition(
                    condition="paired_correct",
                    records=records,
                    selection=selection,
                    images={"correct": correct, "shuffled": shuffled},
                    constant_image=neutral,
                    output=output,
                    backend=backend,
                    max_new_items=None,
                )


if __name__ == "__main__":
    unittest.main()
