"""Versioned study configuration loading and safety validation."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256


class ConfigError(ValueError):
    """Raised when a study configuration is incomplete or unsafe."""


REQUIRED_SECTIONS = ("data", "model", "execution", "monitoring", "storage")
IMMUTABLE_REVISION_FIELDS = (
    ("data", "dataset_revision"),
    ("data", "image_dataset_revision"),
    ("model", "base_model_revision"),
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping without executing arbitrary YAML objects."""

    config_path = Path(path)
    suffix = config_path.suffix.casefold()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            value = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required for YAML configurations; install the project first"
                ) from exc
            value = yaml.safe_load(handle)
        else:
            raise ConfigError(f"unsupported configuration extension: {config_path.suffix}")
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def validate_config(
    config: Mapping[str, Any],
    *,
    require_resolved: bool = False,
) -> dict[str, Any]:
    """Validate structure and confirmatory-run safety gates."""

    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for name in ("study", "profile"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ConfigError(f"{name} must be a non-empty string")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), Mapping):
            raise ConfigError(f"{section} must be a mapping")

    for section, field in IMMUTABLE_REVISION_FIELDS:
        value = config[section].get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{section}.{field} must be a non-empty immutable revision")
        if value.casefold() in {"main", "master", "latest"}:
            raise ConfigError(f"{section}.{field} must not use a moving revision")

    execution = config["execution"]
    if execution.get("evaluation_partition") not in {"development", "grouped_test"}:
        raise ConfigError(
            "execution.evaluation_partition must be development or grouped_test"
        )
    shard_count = execution.get("shard_count")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        raise ConfigError("execution.shard_count must be a positive integer")

    profile = config["profile"]
    if profile == "confirmatory":
        _require_confirmatory_gates(config, require_resolved=require_resolved)
    elif execution.get("evaluation_partition") == "grouped_test":
        raise ConfigError("only the confirmatory profile may use grouped_test")

    return _json_copy(config)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Return a canonical digest for a validated configuration."""

    return canonical_json_sha256(validate_config(config))


def _require_confirmatory_gates(
    config: Mapping[str, Any],
    *,
    require_resolved: bool,
) -> None:
    execution = config["execution"]
    required_true = (
        "require_clean_git",
        "require_locked_protocol",
        "forbid_overwrite",
    )
    for field in required_true:
        if execution.get(field) is not True:
            raise ConfigError(f"confirmatory execution.{field} must be true")
    if execution.get("max_items") is not None:
        raise ConfigError("confirmatory execution.max_items must be null")
    if execution.get("evaluation_partition") != "grouped_test":
        raise ConfigError("confirmatory evaluation must use grouped_test")

    locked_protocol = config["data"].get("locked_protocol")
    if not isinstance(locked_protocol, str) or not locked_protocol:
        raise ConfigError("confirmatory data.locked_protocol is required")

    if require_resolved:
        required_values = {
            "model.adapter": config["model"].get("adapter"),
            "model.adapter_revision": config["model"].get("adapter_revision"),
            "storage.gcs_uri": config["storage"].get("gcs_uri"),
        }
        unresolved = [
            name
            for name, value in required_values.items()
            if not isinstance(value, str)
            or not value.strip()
            or value.strip().casefold() == "required"
        ]
        if unresolved:
            raise ConfigError(
                f"confirmatory configuration has unresolved values: {sorted(unresolved)}"
            )


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )

