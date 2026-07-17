from __future__ import annotations

import math
import unittest

from gi_vqa.backends import (
    AttributionResult,
    BackendProvenance,
    GenerationResult,
    PreparedInput,
    ScoreVerification,
    TargetScore,
    VisionLanguageBackend,
)


def provenance(revision: str = "a" * 40) -> BackendProvenance:
    return BackendProvenance(
        backend_name="fake",
        backend_version="1",
        model_id="example/model",
        model_revision=revision,
        model_spec_sha256="b" * 64,
        processor_id="example/model",
        processor_revision=revision,
        software_versions=(("library", "1.0"),),
    )


class FakeBackend:
    def __init__(self) -> None:
        self._provenance = provenance()
        self.closed = False

    @property
    def provenance(self) -> BackendProvenance:
        return self._provenance

    def prepare(self, *, item_id, image, question):
        return PreparedInput(
            item_id=item_id,
            question=question,
            prompt=f"<image>{question}",
            payload={"image": image},
            provenance=self.provenance,
            input_token_ids=(1, 2, 3),
            image_token_indices=(0,),
        )

    def generate(self, prepared):
        return GenerationResult(
            text="yes",
            token_ids=(10,),
            token_logprobs=(-0.25,),
            provenance=self.provenance,
            finish_reason="stop",
        )

    def score_target(
        self,
        prepared,
        target_text,
        *,
        include_eos=None,
        expected_token_ids=None,
    ):
        return TargetScore(
            target_text=target_text,
            token_ids=(10,),
            token_logprobs=(-0.25,),
            provenance=self.provenance,
            includes_eos=False if include_eos is None else include_eos,
        )

    def verify_generation_score(
        self,
        generation,
        target_score,
        *,
        absolute_tolerance=None,
    ):
        return ScoreVerification(
            token_count=1,
            absolute_tolerance=0.001,
            maximum_absolute_difference=0.0,
            mean_absolute_difference=0.0,
        )

    def attribute(self, prepared, generation, *, method):
        target_score = self.score_target(prepared, generation.text)
        return AttributionResult(
            method=method,
            values=((0.1, 0.2), (0.3, 0.4)),
            patch_grid_shape=(2, 2),
            target_score=target_score,
            image_token_indices=(0,),
            aggregation="mean",
            provenance=self.provenance,
        )

    def close(self):
        self.closed = True


class BackendContractTests(unittest.TestCase):
    def test_runtime_protocol_accepts_structural_backend(self) -> None:
        backend = FakeBackend()
        self.assertIsInstance(backend, VisionLanguageBackend)
        prepared = backend.prepare(item_id="item-1", image=object(), question="Visible?")
        generation = backend.generate(prepared)
        self.assertEqual(generation.text, "yes")
        self.assertEqual(
            backend.attribute(
                prepared,
                generation,
                method="grad_cam",
            ).patch_grid_shape,
            (2, 2),
        )

    def test_score_verification_is_json_compatible(self) -> None:
        verification = ScoreVerification(
            token_count=2,
            absolute_tolerance=0.001,
            maximum_absolute_difference=0.0005,
            mean_absolute_difference=0.00025,
        )
        self.assertEqual(verification.as_dict()["token_count"], 2)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            ScoreVerification(
                token_count=1,
                absolute_tolerance=0.001,
                maximum_absolute_difference=0.01,
                mean_absolute_difference=0.01,
            )

    def test_generation_exposes_length_normalised_confidence(self) -> None:
        result = GenerationResult(
            text="two tokens",
            token_ids=(10, 11),
            token_logprobs=(-0.2, -0.4),
            provenance=provenance(),
        )
        self.assertAlmostEqual(result.mean_token_logprob or 0.0, -0.3)
        self.assertAlmostEqual(result.sequence_confidence or 0.0, math.exp(-0.3))

    def test_generation_rejects_unaligned_token_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            GenerationResult(
                text="bad",
                token_ids=(10, 11),
                token_logprobs=(-0.2,),
                provenance=provenance(),
            )

    def test_target_score_requires_finite_log_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            TargetScore(
                target_text="answer",
                token_ids=(10,),
                token_logprobs=(float("nan"),),
                provenance=provenance(),
            )

    def test_prepared_input_validates_image_token_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "positions"):
            PreparedInput(
                item_id="item",
                question="question",
                prompt="<image>question",
                payload={},
                provenance=provenance(),
                input_token_ids=(1,),
                image_token_indices=(1,),
            )

    def test_attribution_and_target_must_share_provenance(self) -> None:
        target = TargetScore(
            target_text="yes",
            token_ids=(10,),
            token_logprobs=(-0.2,),
            provenance=provenance(),
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            AttributionResult(
                method="attention",
                values=((1.0,),),
                patch_grid_shape=(1, 1),
                target_score=target,
                image_token_indices=(0,),
                aggregation="mean",
                provenance=provenance("c" * 40),
            )

    def test_provenance_requires_complete_adapter_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            BackendProvenance(
                backend_name="fake",
                backend_version="1",
                model_id="example/model",
                model_revision="a" * 40,
                model_spec_sha256="b" * 64,
                adapter_id="example/adapter",
            )


if __name__ == "__main__":
    unittest.main()
