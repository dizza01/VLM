from __future__ import annotations

import unittest

from gi_vqa.config import ConfigError, config_sha256, validate_config


def base_config(profile: str = "smoke") -> dict:
    return {
        "schema_version": 1,
        "study": "study1",
        "profile": profile,
        "data": {
            "dataset_revision": "dataset-sha",
            "image_dataset_revision": "image-sha",
        },
        "model": {
            "base_model_revision": "model-sha",
            "adapter": None,
            "adapter_revision": None,
        },
        "execution": {
            "evaluation_partition": "development",
            "shard_count": 1,
        },
        "monitoring": {},
        "storage": {},
    }


def model_execution_config() -> dict:
    config = base_config()
    config["seed"] = 42
    config["data"].update(
        {
            "dataset_revision": "a" * 40,
            "image_dataset_revision": "b" * 40,
        }
    )
    config["model"].update(
        {
            "base_model": "google/paligemma-3b-pt-224",
            "base_model_revision": "c" * 40,
            "backend": "transformers-paligemma",
            "condition": "base",
            "device": "cuda",
            "precision": "float16",
            "quantization": "none",
            "attn_implementation": "eager",
            "trust_remote_code": False,
            "processor_use_fast": False,
            "prompt_template": "paligemma",
        }
    )
    config["generation"] = {
        "max_new_tokens": 64,
        "do_sample": False,
        "temperature": None,
        "num_beams": 1,
        "batch_size": 1,
        "return_token_logprobs": True,
    }
    config["target_scoring"] = {
        "target_source": "saved_prediction",
        "reduction": "mean_logprob",
        "include_eos": False,
        "batch_size": 1,
        "verify_generation_score": True,
        "absolute_tolerance": 0.001,
    }
    config["attribution"] = {
        "methods": [
            "decoder_answer_to_image_attention",
            "answer_conditioned_grad_cam",
        ],
        "target_source": "saved_prediction",
        "require_prediction_reproduction": True,
        "normalization": "minmax",
        "output_dtype": "float32",
        "zero_range_policy": "error",
        "attention": {
            "layer": -1,
            "head_aggregation": "mean",
            "answer_token_aggregation": "mean",
            "image_tokens_only": True,
        },
        "grad_cam": {
            "vision_layer": -1,
            "gradient_pooling": "spatial_mean",
            "activation_combination": "weighted_sum",
            "relu": True,
        },
    }
    config["perturbation"] = {
        "patch_fractions": [0.25],
        "deletion_treatments": ["gray", "blur"],
        "insertion_treatments": ["blur"],
        "selection_modes": ["most_salient", "least_salient", "random"],
        "random_repeats": 1,
        "gray_value": 128,
        "blur_radius": 12.0,
    }
    return config


class ConfigTests(unittest.TestCase):
    def test_valid_smoke_config(self) -> None:
        config = base_config()
        self.assertEqual(validate_config(config)["profile"], "smoke")
        self.assertEqual(len(config_sha256(config)), 64)

    def test_moving_revision_is_rejected(self) -> None:
        for revision in ("main", "HEAD", "refs/heads/research"):
            with self.subTest(revision=revision):
                config = base_config()
                config["data"]["dataset_revision"] = revision
                with self.assertRaisesRegex(ConfigError, "moving revision"):
                    validate_config(config)

    def test_legacy_non_strict_revision_labels_remain_valid(self) -> None:
        validated = validate_config(base_config())
        self.assertEqual(validated["data"]["dataset_revision"], "dataset-sha")

    def test_resolved_validation_rejects_revision_placeholders(self) -> None:
        for section, field in (
            ("data", "dataset_revision"),
            ("data", "image_dataset_revision"),
            ("model", "base_model_revision"),
        ):
            with self.subTest(field=f"{section}.{field}"):
                config = base_config()
                config[section][field] = "REQUIRED"
                validate_config(config)
                with self.assertRaisesRegex(ConfigError, "unresolved"):
                    validate_config(config, require_resolved=True)

    def test_nonconfirmatory_test_access_is_rejected(self) -> None:
        config = base_config()
        config["execution"]["evaluation_partition"] = "grouped_test"
        with self.assertRaisesRegex(ConfigError, "only the confirmatory"):
            validate_config(config)

    def test_confirmatory_placeholders_fail_resolved_check(self) -> None:
        config = base_config("confirmatory")
        config["data"]["locked_protocol"] = "protocol.json"
        config["model"]["adapter"] = "REQUIRED"
        config["model"]["adapter_revision"] = "REQUIRED"
        config["storage"]["gcs_uri"] = "REQUIRED"
        config["execution"].update(
            {
                "evaluation_partition": "grouped_test",
                "max_items": None,
                "require_clean_git": True,
                "require_locked_protocol": True,
                "forbid_overwrite": True,
            }
        )
        validate_config(config)
        with self.assertRaisesRegex(ConfigError, "unresolved"):
            validate_config(config, require_resolved=True)

    def test_complete_shared_backend_config_passes_strict_validation(self) -> None:
        config = model_execution_config()
        validated = validate_config(config, require_model_execution=True)
        self.assertEqual(validated["model"]["backend"], "transformers-paligemma")
        self.assertEqual(validated["attribution"]["methods"][1], "answer_conditioned_grad_cam")

    def test_strict_execution_requires_commit_or_content_revisions(self) -> None:
        for section, field, revision in (
            ("data", "dataset_revision", "immutable-but-not-a-digest"),
            ("data", "image_dataset_revision", f" {'f' * 40} "),
            ("model", "base_model_revision", "release-tag"),
        ):
            with self.subTest(field=f"{section}.{field}"):
                config = model_execution_config()
                config[section][field] = revision
                with self.assertRaisesRegex(ConfigError, "40- or 64-character"):
                    validate_config(config, require_model_execution=True)

        config = model_execution_config()
        config["data"]["dataset_revision"] = "d" * 64
        validate_config(config, require_model_execution=True)

    def test_strict_execution_implies_resolved_revision_validation(self) -> None:
        config = model_execution_config()
        config["data"]["dataset_revision"] = "REQUIRED"
        with self.assertRaisesRegex(ConfigError, "unresolved"):
            validate_config(config, require_model_execution=True)

    def test_strict_model_execution_requires_all_sections(self) -> None:
        config = model_execution_config()
        del config["target_scoring"]
        with self.assertRaisesRegex(ConfigError, "missing sections"):
            validate_config(config, require_model_execution=True)

    def test_adapter_condition_requires_complete_immutable_identity(self) -> None:
        config = model_execution_config()
        config["model"]["condition"] = "adapter"
        config["model"]["adapter"] = "example/adapter"
        with self.assertRaisesRegex(ConfigError, "adapter_revision"):
            validate_config(config)
        config["model"]["adapter_revision"] = "main"
        with self.assertRaisesRegex(ConfigError, "moving revision"):
            validate_config(config)

        config = model_execution_config()
        config["model"].update(
            {
                "condition": "adapter",
                "adapter": "REQUIRED",
                "adapter_revision": "REQUIRED",
            }
        )
        with self.assertRaisesRegex(ConfigError, "unresolved"):
            validate_config(config, require_model_execution=True)

        config["model"].update(
            {
                "adapter": "example/adapter",
                "adapter_revision": "not-a-commit",
            }
        )
        with self.assertRaisesRegex(ConfigError, "40- or 64-character"):
            validate_config(config, require_model_execution=True)

        config["model"]["adapter_revision"] = "d" * 40
        validate_config(config, require_model_execution=True)

    def test_optional_processor_revision_is_strictly_validated(self) -> None:
        config = model_execution_config()
        config["model"].update(
            {
                "processor": "example/processor",
                "processor_revision": "processor-tag",
            }
        )
        with self.assertRaisesRegex(ConfigError, "40- or 64-character"):
            validate_config(config, require_model_execution=True)

        config["model"]["processor_revision"] = "e" * 64
        validate_config(config, require_model_execution=True)

        del config["model"]["processor_revision"]
        with self.assertRaisesRegex(ConfigError, "provided together"):
            validate_config(config)

    def test_base_condition_rejects_adapter_values(self) -> None:
        config = model_execution_config()
        config["model"]["adapter"] = "example/adapter"
        config["model"]["adapter_revision"] = "a" * 40
        with self.assertRaisesRegex(ConfigError, "condition base"):
            validate_config(config)

    def test_attention_requires_eager_and_first_smoke_requires_full_precision(self) -> None:
        config = model_execution_config()
        config["model"]["attn_implementation"] = "sdpa"
        with self.assertRaisesRegex(ConfigError, "requires.*eager"):
            validate_config(config)

        config = model_execution_config()
        config["model"]["quantization"] = "bnb-nf4-4bit"
        with self.assertRaisesRegex(ConfigError, "quantization none"):
            validate_config(config, require_model_execution=True)

    def test_attention_must_select_image_tokens_only(self) -> None:
        config = model_execution_config()
        config["attribution"]["attention"]["image_tokens_only"] = False
        with self.assertRaisesRegex(ConfigError, "image_tokens_only must be true"):
            validate_config(config)

    def test_generation_typos_and_boolean_integers_are_rejected(self) -> None:
        config = model_execution_config()
        config["generation"]["max_token"] = 64
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            validate_config(config)

        config = model_execution_config()
        config["model"]["attn_implementaton"] = "eager"
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            validate_config(config)

        config = model_execution_config()
        config["generation"]["batch_size"] = True
        with self.assertRaisesRegex(ConfigError, "integer"):
            validate_config(config)

    def test_backend_setting_changes_config_fingerprint(self) -> None:
        config = model_execution_config()
        original = config_sha256(config)
        config["attribution"]["attention"]["layer"] = -2
        self.assertNotEqual(original, config_sha256(config))

    def test_perturbation_contract_rejects_unsafe_or_incomplete_plans(self) -> None:
        config = model_execution_config()
        config["perturbation"]["patch_fractions"] = [0.0]
        with self.assertRaisesRegex(ConfigError, "greater than zero"):
            validate_config(config)

        config = model_execution_config()
        config["perturbation"]["selection_modes"] = ["most_salient"]
        with self.assertRaisesRegex(ConfigError, "most_salient and random"):
            validate_config(config)

        config = model_execution_config()
        config["perturbation"]["deletion_treatments"] = ["inpaint"]
        with self.assertRaisesRegex(ConfigError, "unsupported perturbation"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
