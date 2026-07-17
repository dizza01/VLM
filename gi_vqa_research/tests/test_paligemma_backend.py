from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_config import model_execution_config

from gi_vqa.backends.base import GenerationResult, TargetScore
from gi_vqa.backends.paligemma import (
    AttributionError,
    BackendCompatibilityError,
    BackendDependencyError,
    PaliGemmaBackend,
    _file_sha256,
    _import_gpu_stack,
    _normalise_attribution_values,
    _resolve_layer_index,
    _rgb_image_sha256,
)
from gi_vqa.model_spec import PaliGemmaModelSpec


class PaliGemmaBackendTests(unittest.TestCase):
    def test_from_config_builds_backend_without_loading_gpu_dependencies(self) -> None:
        config = model_execution_config()
        backend = PaliGemmaBackend.from_config(config, load=False)
        self.assertFalse(backend.is_loaded)
        self.assertEqual(backend.spec.base_model_id, config["model"]["base_model"])
        self.assertEqual(backend.spec.device, "cuda")
        self.assertEqual(backend.spec.quantization, "none")
        self.assertEqual(backend.attention_layer, -1)
        self.assertEqual(backend.grad_cam_vision_layer, -1)
        with self.assertRaisesRegex(RuntimeError, "not been loaded"):
            _ = backend.provenance
        backend._model = object()
        backend._processor = object()
        backend._input_device = "cuda:0"
        self.assertFalse(backend.is_loaded)

    def test_runtime_settings_cannot_diverge_from_immutable_spec(self) -> None:
        spec = PaliGemmaModelSpec(
            base_model_id="google/paligemma-3b-pt-224",
            base_model_revision="a" * 40,
            device="cuda",
            quantization="none",
        )
        with self.assertRaisesRegex(ValueError, "device"):
            PaliGemmaBackend(spec, device="cpu")
        with self.assertRaisesRegex(ValueError, "quantization"):
            PaliGemmaBackend(spec, quantization="bnb-nf4-4bit")

    def test_prepare_requires_loaded_model(self) -> None:
        backend = PaliGemmaBackend.from_config(
            model_execution_config(),
            load=False,
        )
        with self.assertRaisesRegex(RuntimeError, "load"):
            backend.prepare(item_id="item", image=object(), question="Question?")
        with self.assertRaisesRegex(ValueError, "immutable model specification"):
            backend.score_target(
                object(),
                "yes",
                include_eos=True,
            )
        with self.assertRaisesRegex(TypeError, "saved GenerationResult"):
            backend.attribute(
                object(),
                "yes",
                method="decoder_answer_to_image_attention",
            )

    def test_layer_indices_are_resolved_and_checked(self) -> None:
        self.assertEqual(_resolve_layer_index(-1, 18, "decoder"), 17)
        self.assertEqual(_resolve_layer_index(0, 18, "decoder"), 0)
        with self.assertRaisesRegex(AttributionError, "out of range"):
            _resolve_layer_index(-19, 18, "decoder")
        with self.assertRaisesRegex(AttributionError, "out of range"):
            _resolve_layer_index(18, 18, "decoder")

    def test_missing_optional_gpu_stack_has_actionable_error(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaisesRegex(BackendDependencyError, "gpu.*extra"):
                _import_gpu_stack()

    def test_generation_and_teacher_forced_scores_must_agree(self) -> None:
        backend = PaliGemmaBackend.from_config(
            model_execution_config(),
            load=False,
        )
        provenance = backend.spec.to_provenance(
            backend_version="1",
            device="cuda:0",
        )
        backend._provenance = provenance
        generation = GenerationResult(
            text="yes",
            token_ids=(10, 11),
            token_logprobs=(-0.2, -0.4),
            provenance=provenance,
        )
        target = TargetScore(
            target_text="yes",
            token_ids=(10, 11),
            token_logprobs=(-0.2005, -0.3995),
            provenance=provenance,
        )
        verification = backend.verify_generation_score(generation, target)
        self.assertEqual(verification.token_count, 2)
        self.assertAlmostEqual(verification.maximum_absolute_difference, 0.0005)
        with self.assertRaisesRegex(ValueError, "immutable model specification"):
            backend.verify_generation_score(
                generation,
                target,
                absolute_tolerance=0.1,
            )

        divergent = TargetScore(
            target_text="yes",
            token_ids=(10, 11),
            token_logprobs=(-0.2, -0.5),
            provenance=provenance,
        )
        with self.assertRaisesRegex(BackendCompatibilityError, "exceeding tolerance"):
            backend.verify_generation_score(generation, divergent)

        different_tokens = TargetScore(
            target_text="yes",
            token_ids=(10, 12),
            token_logprobs=(-0.2, -0.4),
            provenance=provenance,
        )
        with self.assertRaisesRegex(BackendCompatibilityError, "token IDs differ"):
            backend.verify_generation_score(generation, different_tokens)

    def test_attribution_maps_are_minmax_normalised(self) -> None:
        values, metadata = _normalise_attribution_values(
            [[2.0, 4.0], [6.0, 10.0]],
            zero_range_policy="error",
        )
        self.assertEqual(str(values.dtype), "float32")
        self.assertAlmostEqual(float(values.min()), 0.0)
        self.assertAlmostEqual(float(values.max()), 1.0)
        self.assertEqual(metadata["raw_minimum"], 2.0)
        with self.assertRaisesRegex(AttributionError, "constant"):
            _normalise_attribution_values(
                [[1.0, 1.0]],
                zero_range_policy="error",
            )

    def test_source_file_and_rgb_pixel_hashes_are_content_sensitive(self) -> None:
        class FakeRgbImage:
            mode = "RGB"
            size = (2, 1)

            def __init__(self, pixels: bytes) -> None:
                self._pixels = pixels

            def tobytes(self) -> bytes:
                return self._pixels

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.bin"
            path.write_bytes(b"first")
            first = _file_sha256(path)
            path.write_bytes(b"second")
            self.assertNotEqual(first, _file_sha256(path))

        first_image = FakeRgbImage(b"\x00\x01\x02\x03\x04\x05")
        same_image = FakeRgbImage(b"\x00\x01\x02\x03\x04\x05")
        changed_image = FakeRgbImage(b"\x00\x01\x02\x03\x04\x06")
        self.assertEqual(_rgb_image_sha256(first_image), _rgb_image_sha256(same_image))
        self.assertNotEqual(
            _rgb_image_sha256(first_image),
            _rgb_image_sha256(changed_image),
        )


if __name__ == "__main__":
    unittest.main()
