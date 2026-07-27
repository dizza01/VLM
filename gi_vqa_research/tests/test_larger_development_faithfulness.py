from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
from gi_vqa.larger_development_faithfulness import (
    LargerDevelopmentFaithfulnessError,
    _require_reproduction,
    _run_condition,
    _summarize,
)


class FakeFaithfulnessBackend:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.score_calls = 0
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
            payload={"item_id": item_id},
            provenance=self.provenance,
            input_token_ids=(1,),
            image_token_indices=(0,),
        )

    def generate(self, prepared):
        self.generate_calls += 1
        return GenerationResult(
            text="polyp present",
            token_ids=(2, 3),
            token_logprobs=(-0.1, -0.2),
            provenance=self.provenance,
        )

    def score_target(self, prepared, target_text, *, expected_token_ids=None, **_):
        self.score_calls += 1
        offset = -0.01 if ":" in prepared.item_id else 0.0
        return TargetScore(
            target_text=target_text,
            token_ids=tuple(expected_token_ids or (2, 3)),
            token_logprobs=(-0.1 + offset, -0.2 + offset),
            provenance=self.provenance,
        )

    def verify_generation_score(self, generation, target_score, **_):
        return ScoreVerification(
            token_count=2,
            absolute_tolerance=0.02,
            maximum_absolute_difference=0.0,
            mean_absolute_difference=0.0,
        )

    def attribute(self, prepared, generation, *, method):
        return AttributionResult(
            method=method,
            values=np.asarray([[0.1, 0.8], [0.4, 0.2]], dtype=np.float32),
            patch_grid_shape=(2, 2),
            target_score=self.score_target(
                prepared,
                generation.text,
                expected_token_ids=generation.token_ids,
            ),
            image_token_indices=(0,),
            aggregation="fake",
            provenance=self.provenance,
        )

    def close(self):
        return None


class LargerDevelopmentFaithfulnessTests(unittest.TestCase):
    def test_saved_answer_reproduction_is_strict(self) -> None:
        backend = FakeFaithfulnessBackend()
        generated = backend.generate(
            backend.prepare(item_id="one", image="unused", question="Finding?")
        )
        saved = {
            "prediction": "polyp present",
            "generated_token_count": 2,
            "mean_generated_token_logprob": -0.15,
            "sequence_confidence": generated.sequence_confidence,
        }
        _require_reproduction(saved, generated)
        saved["prediction"] = "no polyp"
        with self.assertRaisesRegex(
            LargerDevelopmentFaithfulnessError, "did not reproduce"
        ):
            _require_reproduction(saved, generated)

    def test_condition_is_restart_safe_and_rejects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            constant = root / "constant.png"
            Image.new("RGB", (8, 8), (20, 30, 40)).save(image)
            Image.new("RGB", (8, 8), (128, 128, 128)).save(constant)
            inference = root / "inference"
            item_id = "item-0"
            prediction_path = (
                inference / "paired_correct/items" / f"000-{item_id}.json"
            )
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_text(
                json.dumps(
                    {
                        "prediction": "polyp present",
                        "generated_token_count": 2,
                        "mean_generated_token_logprob": -0.15,
                        "sequence_confidence": float(np.exp(-0.15)),
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "item_id": item_id,
                    "messages": [
                        {"role": "user", "content": "<image>Finding?"},
                        {"role": "assistant", "content": "polyp present"},
                    ],
                }
            ]
            selection = {
                "records": [
                    {
                        "item_id": item_id,
                        "source_img_id": "source",
                        "record_sha256": "record-hash",
                    }
                ]
            }
            config = {
                "seed": 42,
                "attribution": {"methods": ["decoder_answer_to_image_attention"]},
                "perturbation": {
                    "patch_fractions": [0.25],
                    "deletion_treatments": ["gray"],
                    "insertion_treatments": ["blur"],
                    "selection_modes": ["most_salient", "random"],
                    "random_repeats": 1,
                    "gray_value": 128,
                    "blur_radius": 2.0,
                },
            }
            output = root / "faithfulness"
            first_backend = FakeFaithfulnessBackend()
            first = _run_condition(
                condition="paired_correct",
                records=records,
                selection=selection,
                images={"source": image},
                constant_image=constant,
                inference=inference,
                output=output,
                backend=first_backend,
                config=config,
                max_new_items=None,
            )
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["new_items"], 1)
            self.assertEqual(first_backend.generate_calls, 1)

            second_backend = FakeFaithfulnessBackend()
            second = _run_condition(
                condition="paired_correct",
                records=records,
                selection=selection,
                images={"source": image},
                constant_image=constant,
                inference=inference,
                output=output,
                backend=second_backend,
                config=config,
                max_new_items=None,
            )
            self.assertEqual(second["reused_items"], 1)
            self.assertEqual(second_backend.generate_calls, 0)

            completion = output / "paired_correct/items/000-item-0.json"
            value = json.loads(completion.read_text(encoding="utf-8"))
            value["condition"] = "wrong"
            completion.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                LargerDevelopmentFaithfulnessError, "stale"
            ):
                _run_condition(
                    condition="paired_correct",
                    records=records,
                    selection=selection,
                    images={"source": image},
                    constant_image=constant,
                    inference=inference,
                    output=output,
                    backend=FakeFaithfulnessBackend(),
                    config=config,
                    max_new_items=None,
                )

    def test_summary_reports_positive_direction_for_both_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in ("paired_correct", "constant_control"):
                item_dir = root / condition / "items"
                item_dir.mkdir(parents=True)
                for rank in range(64):
                    interventions = []
                    for operation, salient, random in (
                        ("deletion", -0.4, -0.1),
                        ("insertion", -0.1, -0.3),
                    ):
                        for selection, value in (
                            ("most_salient", salient),
                            ("random", random),
                        ):
                            interventions.append(
                                {
                                    "operation": operation,
                                    "treatment": "blur",
                                    "fraction": 0.25,
                                    "selection": selection,
                                    "score_minus_original": value,
                                }
                            )
                    (item_dir / f"{rank:03d}-item-{rank}.json").write_text(
                        json.dumps(
                            {
                                "methods": {
                                    "decoder_answer_to_image_attention": {
                                        "interventions": interventions
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
            report = _summarize(root)
            self.assertEqual(len(report["rows"]), 4)
            self.assertTrue(
                all(
                    row["faithfulness_effect_positive_is_expected"] > 0
                    for row in report["rows"]
                )
            )


if __name__ == "__main__":
    unittest.main()
