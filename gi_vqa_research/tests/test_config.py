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


class ConfigTests(unittest.TestCase):
    def test_valid_smoke_config(self) -> None:
        config = base_config()
        self.assertEqual(validate_config(config)["profile"], "smoke")
        self.assertEqual(len(config_sha256(config)), 64)

    def test_moving_revision_is_rejected(self) -> None:
        config = base_config()
        config["data"]["dataset_revision"] = "main"
        with self.assertRaisesRegex(ConfigError, "moving revision"):
            validate_config(config)

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


if __name__ == "__main__":
    unittest.main()

