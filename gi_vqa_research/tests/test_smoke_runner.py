from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from gi_vqa.backends import (
    AttributionResult,
    BackendProvenance,
    GenerationResult,
    PreparedInput,
    ScoreVerification,
    TargetScore,
)
from gi_vqa.config import load_config
from gi_vqa.jsonl import write_jsonl_atomic
from gi_vqa.provenance import file_sha256
from gi_vqa.smoke_runner import run_development_smoke


class FakeBackend:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.generate_calls = 0
        self.attribute_calls = 0
        self.score_calls = 0
        self.provenance = BackendProvenance(
            backend_name="fake-paligemma",
            backend_version="1",
            model_id="fake/model",
            model_revision="a" * 40,
            model_spec_sha256="b" * 64,
            processor_id="fake/model",
            processor_revision="a" * 40,
            torch_dtype="float16",
            attention_implementation="eager",
            device="cuda",
            software_versions=(("fake", "1"),),
        )

    def prepare(self, *, item_id, image, question):
        self.prepare_calls += 1
        perturbed = not isinstance(image, (str, Path))
        if perturbed:
            pixels = image.convert("RGB").tobytes()
        else:
            with Image.open(image) as opened:
                pixels = opened.convert("RGB").tobytes()
        digest = hashlib.sha256(pixels).hexdigest()
        return PreparedInput(
            item_id=item_id,
            question=question,
            prompt=f"<image>{question}\n",
            payload={"perturbed": perturbed},
            provenance=self.provenance,
            input_token_ids=(10, 11, 12),
            image_token_indices=(0,),
            preprocessing={
                "rgb_pixels_sha256": digest,
                "processed_pixel_values_sha256": digest,
            },
        )

    def generate(self, prepared):
        self.generate_calls += 1
        return GenerationResult(
            text="answer",
            token_ids=(20, 21),
            token_logprobs=(-0.1, -0.2),
            provenance=self.provenance,
            finish_reason="eos",
        )

    def score_target(
        self,
        prepared,
        target_text,
        *,
        include_eos=None,
        expected_token_ids=None,
    ):
        self.score_calls += 1
        token_ids = tuple(expected_token_ids or (20, 21))
        logprobs = (-0.3, -0.4) if prepared.payload["perturbed"] else (-0.1, -0.2)
        return TargetScore(
            target_text=target_text,
            token_ids=token_ids,
            token_logprobs=logprobs,
            provenance=self.provenance,
            includes_eos=False,
        )

    def verify_generation_score(self, generation, target_score, *, absolute_tolerance=None):
        differences = [
            abs(left - right)
            for left, right in zip(  # noqa: B905 - local runner uses Python 3.9
                generation.token_logprobs,
                target_score.token_logprobs,
            )
        ]
        return ScoreVerification(
            token_count=len(differences),
            absolute_tolerance=1e-4,
            maximum_absolute_difference=max(differences),
            mean_absolute_difference=sum(differences) / len(differences),
        )

    def attribute(self, prepared, generation, *, method):
        self.attribute_calls += 1
        offset = 0.05 if method == "answer_conditioned_grad_cam" else 0.0
        target = self.score_target(
            prepared,
            generation.text,
            expected_token_ids=generation.token_ids,
        )
        return AttributionResult(
            method=method,
            values=np.asarray(
                [[0.1 + offset, 0.8], [0.4, 0.2]],
                dtype=np.float32,
            ),
            patch_grid_shape=(2, 2),
            target_score=target,
            image_token_indices=(0,),
            aggregation="fake",
            provenance=self.provenance,
        )

    def close(self):
        return None


class SmokeRunnerTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Path, Path, Path]:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_config(repository_root / "configs/study1/smoke.yaml")
        config_path = root / "smoke.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        image_dir = root / "data/images"
        image_dir.mkdir(parents=True)
        records = []
        images = {}
        item_ids = []
        source_ids = []
        for index in range(20):
            item_id = f"item-{index:02d}"
            source_id = f"source-{index:02d}"
            relative_image = Path("data/images") / f"{source_id}.jpg"
            image_path = root / relative_image
            Image.new(
                "RGB",
                (8, 8),
                color=(index, 20, 30),
            ).save(image_path, format="JPEG")
            records.append(
                {
                    "item_id": item_id,
                    "messages": [
                        {"role": "user", "content": f"<image>Question {index}?"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "metadata": {
                        "source_img_id": source_id,
                        "partition": "development",
                        "complexity": 1,
                        "question_class": ["finding_presence"],
                    },
                }
            )
            images[source_id] = {
                "path": str(relative_image),
                "scope": "development_smoke",
                "sha256": file_sha256(image_path),
            }
            item_ids.append(item_id)
            source_ids.append(source_id)

        smoke_path = write_jsonl_atomic(root / "data/smoke_20.jsonl", records)
        split_manifest_path = root / "split_manifest.json"
        split_manifest_path.write_text(
            json.dumps(
                {
                    "smoke": {
                        "item_ids": item_ids,
                        "source_img_ids": source_ids,
                    },
                    "artifacts": {
                        "smoke_20": {
                            "path": str(smoke_path.relative_to(root)),
                            "sha256": file_sha256(smoke_path),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        cache_manifest_path = root / "cache_manifest.json"
        cache_manifest_path.write_text(
            json.dumps({"images": images}),
            encoding="utf-8",
        )
        return config_path, split_manifest_path, cache_manifest_path

    @patch(
        "gi_vqa.smoke_runner.verify_grouped_split_artifacts",
        return_value={"status": "PASS", "records": 20},
    )
    @patch(
        "gi_vqa.smoke_runner.verify_image_cache",
        return_value={"status": "PASS", "images": 20},
    )
    def test_interrupted_run_resumes_without_reexecuting_completed_items(
        self,
        _cache_check,
        _split_check,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, split_manifest, cache_manifest = self._workspace(root)
            run_dir = root / "runs/development-smoke"

            first_backend = FakeBackend()
            first = run_development_smoke(
                config_path=config,
                project_root=root,
                split_manifest_path=split_manifest,
                image_cache_manifest_path=cache_manifest,
                run_dir=run_dir,
                run_id="development-smoke",
                materialize_inputs=False,
                max_new_items=1,
                backend=first_backend,
            )
            self.assertEqual(first["status"], "INCOMPLETE")
            self.assertEqual(first["completed_items"], 1)
            self.assertEqual(first_backend.generate_calls, 1)
            first_completion = next((run_dir / "items").glob("*/complete.json"))
            first_completion_hash = file_sha256(first_completion)

            second_backend = FakeBackend()
            second = run_development_smoke(
                config_path=config,
                project_root=root,
                split_manifest_path=split_manifest,
                image_cache_manifest_path=cache_manifest,
                run_dir=run_dir,
                run_id="development-smoke",
                materialize_inputs=False,
                backend=second_backend,
            )
            self.assertEqual(second["status"], "PASS")
            self.assertEqual(second["invocation"], {"new_items": 19, "reused_items": 1})
            self.assertEqual(second_backend.generate_calls, 19)
            self.assertEqual(file_sha256(first_completion), first_completion_hash)

            third_backend = FakeBackend()
            third = run_development_smoke(
                config_path=config,
                project_root=root,
                split_manifest_path=split_manifest,
                image_cache_manifest_path=cache_manifest,
                run_dir=run_dir,
                run_id="development-smoke",
                materialize_inputs=False,
                backend=third_backend,
            )
            self.assertEqual(third["status"], "PASS")
            self.assertEqual(third["invocation"], {"new_items": 0, "reused_items": 20})
            self.assertEqual(third_backend.generate_calls, 0)
            self.assertEqual(third_backend.attribute_calls, 0)
            self.assertEqual(third_backend.score_calls, 0)
            self.assertEqual(file_sha256(first_completion), first_completion_hash)
            self.assertTrue((run_dir / "metrics/smoke_report.json").is_file())
            self.assertTrue((run_dir / "predictions/smoke_results.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
