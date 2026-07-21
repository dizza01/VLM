"""Versioned study configuration loading and safety validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256
from .revisions import validate_immutable_revision


class ConfigError(ValueError):
    """Raised when a study configuration is incomplete or unsafe."""


REQUIRED_SECTIONS = ("data", "model", "execution", "monitoring", "storage")
IMMUTABLE_REVISION_FIELDS = (
    ("data", "dataset_revision"),
    ("data", "image_dataset_revision"),
    ("model", "base_model_revision"),
)
MODEL_BACKENDS = {"transformers-paligemma"}
MODEL_CONDITIONS = {"base", "adapter"}
MODEL_DEVICES = {"cpu", "cuda", "auto"}
MODEL_PRECISIONS = {"float16", "bfloat16", "float32"}
MODEL_QUANTIZATIONS = {"none", "bnb-nf4-4bit"}
ATTENTION_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2"}
MODEL_FIELDS = {
    "base_model",
    "base_model_revision",
    "backend",
    "condition",
    "adapter",
    "adapter_revision",
    "processor",
    "processor_revision",
    "device",
    "precision",
    "quantization",
    "attn_implementation",
    "trust_remote_code",
    "processor_use_fast",
    "prompt_template",
}

GENERATION_FIELDS = {
    "max_new_tokens",
    "do_sample",
    "temperature",
    "num_beams",
    "batch_size",
    "return_token_logprobs",
}
TARGET_SCORING_FIELDS = {
    "target_source",
    "reduction",
    "include_eos",
    "batch_size",
    "verify_generation_score",
    "absolute_tolerance",
}
ATTRIBUTION_FIELDS = {
    "methods",
    "target_source",
    "require_prediction_reproduction",
    "normalization",
    "output_dtype",
    "zero_range_policy",
    "attention",
    "grad_cam",
}
ATTENTION_ATTRIBUTION_FIELDS = {
    "layer",
    "head_aggregation",
    "answer_token_aggregation",
    "image_tokens_only",
}
GRAD_CAM_FIELDS = {
    "vision_layer",
    "gradient_pooling",
    "activation_combination",
    "relu",
}
PERTURBATION_FIELDS = {
    "patch_fractions",
    "deletion_treatments",
    "insertion_treatments",
    "selection_modes",
    "random_repeats",
    "gray_value",
    "blur_radius",
}
PERTURBATION_TREATMENTS = {"gray", "blur"}
PERTURBATION_SELECTION_MODES = {
    "most_salient",
    "least_salient",
    "random",
}
ATTRIBUTION_METHODS = {
    "decoder_answer_to_image_attention",
    "answer_conditioned_grad_cam",
}


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
    require_model_execution: bool = False,
) -> dict[str, Any]:
    """Validate structure, execution settings and confirmatory-run safety gates.

    The model-execution sections are additive in schema version 1. Legacy
    infrastructure-only configurations remain valid, while GPU entry points can
    request the stricter shared-backend contract with
    ``require_model_execution=True``.
    """

    resolved_revisions_required = require_resolved or require_model_execution
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for name in ("study", "profile"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ConfigError(f"{name} must be a non-empty string")
    if "seed" in config:
        _require_int(config["seed"], "seed")
    elif require_model_execution:
        raise ConfigError("model execution configuration requires an integer seed")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), Mapping):
            raise ConfigError(f"{section} must be a mapping")

    for section, field in IMMUTABLE_REVISION_FIELDS:
        value = config[section].get(field)
        _validate_config_revision(
            value,
            f"{section}.{field}",
            require_resolved=resolved_revisions_required,
            require_commit=require_model_execution,
        )

    _validate_model(
        config["model"],
        require_resolved=resolved_revisions_required,
        require_commit=require_model_execution,
    )
    _validate_optional_execution_sections(config)
    if require_model_execution:
        _require_model_execution_contract(config)

    execution = config["execution"]
    if execution.get("evaluation_partition") not in {"development", "grouped_test"}:
        raise ConfigError("execution.evaluation_partition must be development or grouped_test")
    shard_count = execution.get("shard_count")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        raise ConfigError("execution.shard_count must be a positive integer")

    profile = config["profile"]
    if profile == "confirmatory":
        _require_confirmatory_gates(
            config,
            require_resolved=resolved_revisions_required,
        )
    elif execution.get("evaluation_partition") == "grouped_test":
        raise ConfigError("only the confirmatory profile may use grouped_test")

    return _json_copy(config)


def _validate_model(
    model: Mapping[str, Any],
    *,
    require_resolved: bool,
    require_commit: bool,
) -> None:
    _reject_unknown_fields(model, MODEL_FIELDS, "model")
    enum_fields = {
        "backend": MODEL_BACKENDS,
        "condition": MODEL_CONDITIONS,
        "device": MODEL_DEVICES,
        "precision": MODEL_PRECISIONS,
        "quantization": MODEL_QUANTIZATIONS,
        "attn_implementation": ATTENTION_IMPLEMENTATIONS,
        "prompt_template": {"paligemma"},
    }
    for field, allowed in enum_fields.items():
        if field in model and model[field] not in allowed:
            raise ConfigError(f"model.{field} must be one of {sorted(allowed)}")
    if "trust_remote_code" in model:
        _require_bool(model["trust_remote_code"], "model.trust_remote_code")
    if "processor_use_fast" in model:
        _require_bool(model["processor_use_fast"], "model.processor_use_fast")

    condition = model.get("condition")
    adapter = model.get("adapter")
    adapter_revision = model.get("adapter_revision")
    if condition == "base" and (adapter is not None or adapter_revision is not None):
        raise ConfigError(
            "model.condition base requires model.adapter and model.adapter_revision to be null"
        )
    if condition == "adapter":
        for field, value in (
            ("adapter", adapter),
            ("adapter_revision", adapter_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"model.condition adapter requires model.{field}")
    if adapter_revision is not None:
        _validate_config_revision(
            adapter_revision,
            "model.adapter_revision",
            require_resolved=require_resolved,
            require_commit=require_commit,
        )

    processor = model.get("processor")
    processor_revision = model.get("processor_revision")
    if (processor is None) != (processor_revision is None):
        raise ConfigError("model.processor and model.processor_revision must be provided together")
    if processor is not None:
        if not isinstance(processor, str) or not processor.strip():
            raise ConfigError("model.processor must be a non-empty string")
        _validate_config_revision(
            processor_revision,
            "model.processor_revision",
            require_resolved=require_resolved,
            require_commit=require_commit,
        )


def _validate_optional_execution_sections(config: Mapping[str, Any]) -> None:
    if "generation" in config:
        generation = _require_mapping(config["generation"], "generation")
        _reject_unknown_fields(generation, GENERATION_FIELDS, "generation")
        _require_int_range(
            generation.get("max_new_tokens"),
            "generation.max_new_tokens",
            minimum=1,
            maximum=512,
        )
        do_sample = _require_bool(generation.get("do_sample"), "generation.do_sample")
        temperature = generation.get("temperature")
        if do_sample:
            _require_nonnegative_number(
                temperature,
                "generation.temperature",
                strictly_positive=True,
            )
        elif temperature is not None:
            raise ConfigError(
                "generation.temperature must be null when generation.do_sample is false"
            )
        _require_int_range(generation.get("num_beams"), "generation.num_beams", minimum=1)
        _require_int_range(generation.get("batch_size"), "generation.batch_size", minimum=1)
        _require_bool(
            generation.get("return_token_logprobs"),
            "generation.return_token_logprobs",
        )

    if "target_scoring" in config:
        scoring = _require_mapping(config["target_scoring"], "target_scoring")
        _reject_unknown_fields(scoring, TARGET_SCORING_FIELDS, "target_scoring")
        if scoring.get("target_source") != "saved_prediction":
            raise ConfigError("target_scoring.target_source must be saved_prediction")
        if scoring.get("reduction") not in {"mean_logprob", "sum_logprob"}:
            raise ConfigError("target_scoring.reduction must be mean_logprob or sum_logprob")
        _require_bool(scoring.get("include_eos"), "target_scoring.include_eos")
        _require_int_range(
            scoring.get("batch_size"),
            "target_scoring.batch_size",
            minimum=1,
        )
        _require_bool(
            scoring.get("verify_generation_score"),
            "target_scoring.verify_generation_score",
        )
        _require_nonnegative_number(
            scoring.get("absolute_tolerance"),
            "target_scoring.absolute_tolerance",
        )

    if "attribution" in config:
        attribution = _require_mapping(config["attribution"], "attribution")
        _reject_unknown_fields(attribution, ATTRIBUTION_FIELDS, "attribution")
        methods = attribution.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ConfigError("attribution.methods must be a non-empty list")
        if any(not isinstance(method, str) for method in methods):
            raise ConfigError("attribution.methods must contain strings")
        if len(set(methods)) != len(methods):
            raise ConfigError("attribution.methods must not contain duplicates")
        unknown_methods = set(methods) - ATTRIBUTION_METHODS
        if unknown_methods:
            raise ConfigError(f"unsupported attribution methods: {sorted(unknown_methods)}")
        if attribution.get("target_source") != "saved_prediction":
            raise ConfigError("attribution.target_source must be saved_prediction")
        _require_bool(
            attribution.get("require_prediction_reproduction"),
            "attribution.require_prediction_reproduction",
        )
        if attribution.get("normalization") != "minmax":
            raise ConfigError("attribution.normalization must be minmax")
        if attribution.get("output_dtype") != "float32":
            raise ConfigError("attribution.output_dtype must be float32")
        if attribution.get("zero_range_policy") not in {"error", "zeros"}:
            raise ConfigError("attribution.zero_range_policy must be error or zeros")

        if "decoder_answer_to_image_attention" in methods:
            attention = _require_mapping(
                attribution.get("attention"),
                "attribution.attention",
            )
            _reject_unknown_fields(
                attention,
                ATTENTION_ATTRIBUTION_FIELDS,
                "attribution.attention",
            )
            _require_int(attention.get("layer"), "attribution.attention.layer")
            if attention.get("head_aggregation") not in {"mean", "max"}:
                raise ConfigError("attribution.attention.head_aggregation must be mean or max")
            if attention.get("answer_token_aggregation") not in {"mean", "max"}:
                raise ConfigError(
                    "attribution.attention.answer_token_aggregation must be mean or max"
                )
            image_tokens_only = _require_bool(
                attention.get("image_tokens_only"),
                "attribution.attention.image_tokens_only",
            )
            if image_tokens_only is not True:
                raise ConfigError("attribution.attention.image_tokens_only must be true")

        if "answer_conditioned_grad_cam" in methods:
            grad_cam = _require_mapping(
                attribution.get("grad_cam"),
                "attribution.grad_cam",
            )
            _reject_unknown_fields(
                grad_cam,
                GRAD_CAM_FIELDS,
                "attribution.grad_cam",
            )
            _require_int(
                grad_cam.get("vision_layer"),
                "attribution.grad_cam.vision_layer",
            )
            if grad_cam.get("gradient_pooling") != "spatial_mean":
                raise ConfigError("attribution.grad_cam.gradient_pooling must be spatial_mean")
            if grad_cam.get("activation_combination") != "weighted_sum":
                raise ConfigError(
                    "attribution.grad_cam.activation_combination must be weighted_sum"
                )
            _require_bool(grad_cam.get("relu"), "attribution.grad_cam.relu")

        if "target_scoring" in config:
            if attribution["target_source"] != config["target_scoring"].get("target_source"):
                raise ConfigError("attribution and target_scoring must use the same target_source")
        if (
            "decoder_answer_to_image_attention" in methods
            and config["model"].get("attn_implementation") != "eager"
        ):
            raise ConfigError(
                "decoder answer-to-image attention requires model.attn_implementation eager"
            )

    if "perturbation" in config:
        perturbation = _require_mapping(
            config["perturbation"],
            "perturbation",
        )
        _reject_unknown_fields(
            perturbation,
            PERTURBATION_FIELDS,
            "perturbation",
        )
        fractions = perturbation.get("patch_fractions")
        if not isinstance(fractions, list) or not fractions:
            raise ConfigError(
                "perturbation.patch_fractions must be a non-empty list"
            )
        for index, fraction in enumerate(fractions):
            value = _require_nonnegative_number(
                fraction,
                f"perturbation.patch_fractions[{index}]",
                strictly_positive=True,
            )
            if value > 1:
                raise ConfigError(
                    "perturbation.patch_fractions values must not exceed 1"
                )
        if len({float(value) for value in fractions}) != len(fractions):
            raise ConfigError(
                "perturbation.patch_fractions must not contain duplicates"
            )
        for field in ("deletion_treatments", "insertion_treatments"):
            values = perturbation.get(field)
            if not isinstance(values, list) or not values:
                raise ConfigError(f"perturbation.{field} must be a non-empty list")
            if any(not isinstance(value, str) for value in values):
                raise ConfigError(f"perturbation.{field} must contain strings")
            if len(set(values)) != len(values):
                raise ConfigError(
                    f"perturbation.{field} must not contain duplicates"
                )
            unknown = set(values) - PERTURBATION_TREATMENTS
            if unknown:
                raise ConfigError(
                    f"unsupported perturbation treatments: {sorted(unknown)}"
                )
        modes = perturbation.get("selection_modes")
        if not isinstance(modes, list) or not modes:
            raise ConfigError(
                "perturbation.selection_modes must be a non-empty list"
            )
        if any(not isinstance(mode, str) for mode in modes):
            raise ConfigError(
                "perturbation.selection_modes must contain strings"
            )
        if len(set(modes)) != len(modes):
            raise ConfigError(
                "perturbation.selection_modes must not contain duplicates"
            )
        unknown_modes = set(modes) - PERTURBATION_SELECTION_MODES
        if unknown_modes:
            raise ConfigError(
                "unsupported perturbation selection modes: "
                f"{sorted(unknown_modes)}"
            )
        if "most_salient" not in modes or "random" not in modes:
            raise ConfigError(
                "perturbation.selection_modes must include most_salient and random"
            )
        _require_int_range(
            perturbation.get("random_repeats"),
            "perturbation.random_repeats",
            minimum=1,
        )
        _require_int_range(
            perturbation.get("gray_value"),
            "perturbation.gray_value",
            minimum=0,
            maximum=255,
        )
        _require_nonnegative_number(
            perturbation.get("blur_radius"),
            "perturbation.blur_radius",
            strictly_positive=True,
        )


def _require_model_execution_contract(config: Mapping[str, Any]) -> None:
    missing_sections = [
        name
        for name in (
            "generation",
            "target_scoring",
            "attribution",
            "perturbation",
        )
        if name not in config
    ]
    if missing_sections:
        raise ConfigError(f"model execution configuration is missing sections: {missing_sections}")

    model = config["model"]
    required_model_fields = (
        "base_model",
        "base_model_revision",
        "backend",
        "condition",
        "device",
        "precision",
        "quantization",
        "attn_implementation",
        "trust_remote_code",
        "processor_use_fast",
        "prompt_template",
    )
    missing_model_fields = [field for field in required_model_fields if field not in model]
    if missing_model_fields:
        raise ConfigError(
            f"model execution configuration is missing model fields: {missing_model_fields}"
        )
    if not isinstance(model["base_model"], str) or not model["base_model"].strip():
        raise ConfigError("model.base_model must be a non-empty string")
    if model["trust_remote_code"] is not False:
        raise ConfigError("model execution requires model.trust_remote_code false")
    if model["processor_use_fast"] is not False:
        raise ConfigError(
            "the pinned PaliGemma compatibility environment requires model.processor_use_fast false"
        )
    if model["condition"] == "adapter":
        unresolved_adapter_fields = [
            field
            for field in ("adapter", "adapter_revision")
            if str(model.get(field, "")).strip().casefold() == "required"
        ]
        if unresolved_adapter_fields:
            raise ConfigError(
                f"model execution has unresolved adapter fields: {unresolved_adapter_fields}"
            )

    generation = config["generation"]
    if (
        generation["do_sample"] is not False
        or generation["temperature"] is not None
        or generation["num_beams"] != 1
        or generation["batch_size"] != 1
        or generation["return_token_logprobs"] is not True
    ):
        raise ConfigError(
            "model execution requires greedy batch-one generation with token logprobs"
        )

    scoring = config["target_scoring"]
    if (
        scoring["target_source"] != "saved_prediction"
        or scoring["reduction"] != "mean_logprob"
        or scoring["include_eos"] is not False
        or scoring["batch_size"] != 1
        or scoring["verify_generation_score"] is not True
    ):
        raise ConfigError(
            "model execution requires batch-one mean logprob scoring of the "
            "saved prediction without EOS and with score verification"
        )

    attribution = config["attribution"]
    if (
        attribution["require_prediction_reproduction"] is not True
        or attribution["zero_range_policy"] != "error"
    ):
        raise ConfigError(
            "model execution requires prediction reproduction and rejection "
            "of degenerate attribution maps"
        )
    if model["quantization"] != "none":
        raise ConfigError("the first attribution smoke requires model.quantization none")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _validate_config_revision(
    value: Any,
    field: str,
    *,
    require_resolved: bool,
    require_commit: bool,
) -> str:
    try:
        return validate_immutable_revision(
            value,
            field,
            require_resolved=require_resolved,
            require_commit=require_commit,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    field: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{field} contains unknown fields: {sorted(unknown)}")


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a Boolean")
    return value


def _require_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    return value


def _require_int_range(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    number = _require_int(value, field)
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is None:
            raise ConfigError(f"{field} must be at least {minimum}")
        raise ConfigError(f"{field} must be between {minimum} and {maximum}")
    return number


def _require_nonnegative_number(
    value: Any,
    field: str,
    *,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field} must be a finite number")
    if strictly_positive and number <= 0:
        raise ConfigError(f"{field} must be greater than zero")
    if not strictly_positive and number < 0:
        raise ConfigError(f"{field} must be non-negative")
    return number


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
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
