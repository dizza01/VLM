from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from gi_vqa.model_spec import (
    PALIGEMMA_BACKEND,
    PALIGEMMA_MODEL_SPEC_SCHEMA,
    PaliGemmaModelSpec,
)


class PaliGemmaModelSpecTests(unittest.TestCase):
    def make_spec(self, **overrides) -> PaliGemmaModelSpec:
        values = {
            "base_model_id": "google/paligemma-3b-pt-224",
            "base_model_revision": "a" * 40,
        }
        values.update(overrides)
        return PaliGemmaModelSpec(**values)

    def test_defaults_resolve_processor_and_format_prompt(self) -> None:
        spec = self.make_spec()
        self.assertEqual(spec.schema_version, PALIGEMMA_MODEL_SPEC_SCHEMA)
        self.assertEqual(spec.backend, PALIGEMMA_BACKEND)
        self.assertEqual(spec.resolved_processor_id, spec.base_model_id)
        self.assertEqual(spec.resolved_processor_revision, spec.base_model_revision)
        self.assertEqual(spec.format_prompt("  Is there a polyp?  "), "<image>Is there a polyp?")
        self.assertEqual(spec.device, "cuda")
        self.assertIsNone(spec.temperature)
        self.assertFalse(spec.do_sample)
        self.assertTrue(spec.return_token_logprobs)
        self.assertFalse(spec.is_adapted)
        with self.assertRaisesRegex(ValueError, "reserved image token"):
            spec.format_prompt("Is <image> visible?")

    def test_spec_is_frozen(self) -> None:
        spec = self.make_spec()
        with self.assertRaises(FrozenInstanceError):
            spec.seed = 7

    def test_moving_revision_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            self.make_spec(base_model_revision="main")

    def test_adapter_identity_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.make_spec(adapter_id="example/adapter")

    def test_fingerprint_is_stable_and_setting_sensitive(self) -> None:
        first = self.make_spec()
        same = self.make_spec()
        changed = self.make_spec(quantization="bnb-nf4-4bit")
        changed_attribution = self.make_spec(attention_layer=-2)
        self.assertEqual(first.fingerprint(), same.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed_attribution.fingerprint())
        self.assertEqual(len(first.fingerprint()), 64)
        json.dumps(first.as_dict(), allow_nan=False)

    def test_from_config_maps_smoke_model_and_generation_fields(self) -> None:
        spec = PaliGemmaModelSpec.from_config(
            {
                "seed": 17,
                "model": {
                    "base_model": "google/paligemma-3b-pt-224",
                    "base_model_revision": "a" * 40,
                    "backend": "transformers-paligemma",
                    "condition": "base",
                    "adapter": None,
                    "adapter_revision": None,
                    "device": "auto",
                    "precision": "bfloat16",
                    "quantization": "bnb-nf4-4bit",
                    "attn_implementation": "sdpa",
                    "trust_remote_code": False,
                    "processor_use_fast": True,
                    "prompt_template": "paligemma",
                },
                "generation": {
                    "max_new_tokens": 32,
                    "do_sample": True,
                    "temperature": 0.7,
                    "num_beams": 2,
                    "batch_size": 4,
                    "return_token_logprobs": False,
                },
                "target_scoring": {"include_eos": True},
            }
        )
        self.assertEqual(spec.device, "auto")
        self.assertEqual(spec.torch_dtype, "bfloat16")
        self.assertEqual(spec.quantization, "bnb-nf4-4bit")
        self.assertEqual(spec.attention_implementation, "sdpa")
        self.assertTrue(spec.processor_use_fast)
        self.assertFalse(spec.trust_remote_code)
        self.assertEqual(spec.prompt_template, "paligemma")
        self.assertEqual(spec.max_new_tokens, 32)
        self.assertTrue(spec.do_sample)
        self.assertEqual(spec.temperature, 0.7)
        self.assertEqual(spec.num_beams, 2)
        self.assertFalse(spec.return_token_logprobs)
        self.assertEqual(spec.seed, 17)
        self.assertTrue(spec.include_eos_in_target_score)
        self.assertEqual(spec.score_absolute_tolerance, 0.001)
        self.assertEqual(spec.attention_layer, -1)
        self.assertTrue(spec.attention_image_tokens_only)

    def test_from_config_requires_supported_backend_and_required_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "model.backend"):
            PaliGemmaModelSpec.from_config(
                {
                    "model": {
                        "backend": "other",
                        "base_model": "example/model",
                        "base_model_revision": "a" * 40,
                    },
                    "generation": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "base_model_revision"):
            PaliGemmaModelSpec.from_config(
                {
                    "model": {"base_model": "example/model"},
                    "generation": {},
                }
            )

    def test_adapted_spec_builds_complete_provenance(self) -> None:
        spec = self.make_spec(
            adapter_id="example/adapter",
            adapter_revision="c" * 40,
            processor_id="example/processor",
            processor_revision="d" * 40,
        )
        provenance = spec.to_provenance(
            backend_version="1",
            device="cuda:0",
            software_versions={"transformers": "4.55.0", "peft": "0.16.0"},
        )
        self.assertTrue(spec.is_adapted)
        self.assertEqual(provenance.model_spec_sha256, spec.fingerprint())
        self.assertEqual(provenance.adapter_revision, "c" * 40)
        self.assertEqual(
            provenance.software_versions,
            (("peft", "0.16.0"), ("transformers", "4.55.0")),
        )

    def test_provenance_uses_fixed_backend_and_configured_device(self) -> None:
        spec = self.make_spec(device="auto")
        provenance = spec.to_provenance(backend_version="1")
        self.assertEqual(provenance.backend_name, PALIGEMMA_BACKEND)
        self.assertEqual(provenance.device, "auto")
        with self.assertRaisesRegex(ValueError, "backend_name"):
            spec.to_provenance(backend_name="other", backend_version="1")

    def test_invalid_decoding_and_image_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_size"):
            self.make_spec(image_size=(224, 0))
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            self.make_spec(max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.make_spec(temperature=float("nan"))
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.make_spec(do_sample=True)
        with self.assertRaisesRegex(ValueError, "num_beams"):
            self.make_spec(num_beams=0)
        with self.assertRaisesRegex(ValueError, "attention_image_tokens_only"):
            self.make_spec(attention_image_tokens_only=False)

    def test_invalid_backend_settings_are_rejected(self) -> None:
        for field_name, value in (
            ("device", "tpu"),
            ("torch_dtype", "float64"),
            ("quantization", "int8"),
            ("attention_implementation", "unknown"),
            ("prompt_template", "chatml"),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    self.make_spec(**{field_name: value})
        for field_name in (
            "processor_use_fast",
            "trust_remote_code",
            "do_sample",
            "return_token_logprobs",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError):
                    self.make_spec(**{field_name: 1})


if __name__ == "__main__":
    unittest.main()
