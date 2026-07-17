"""Immutable, dependency-light specification for the pinned PaliGemma stack."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .backends.base import BackendProvenance
from .revisions import validate_immutable_revision

PALIGEMMA_MODEL_SPEC_SCHEMA = "gi-vqa-paligemma-model-spec-v1"
PALIGEMMA_BACKEND = "transformers-paligemma"
_DEVICES = frozenset({"auto", "cpu", "cuda"})
_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_QUANTIZATIONS = frozenset({"none", "bnb-nf4-4bit"})
_ATTENTION_IMPLEMENTATIONS = frozenset({"eager", "sdpa", "flash_attention_2"})
_ATTRIBUTION_METHODS = frozenset(
    {
        "decoder_answer_to_image_attention",
        "answer_conditioned_grad_cam",
    }
)


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _section(config: Mapping[str, Any], name: str, *, required: bool = True) -> Mapping[str, Any]:
    value = config.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        qualifier = "required and " if required else ""
        raise TypeError(f"config.{name} is {qualifier}must be a mapping")
    return value


def _required(section: Mapping[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ValueError(f"config.{section_name}.{key} is required")
    return section[key]


@dataclass(frozen=True)
class PaliGemmaModelSpec:
    """All choices required to reproduce one PaliGemma backend's outputs."""

    base_model_id: str
    base_model_revision: str
    adapter_id: str | None = None
    adapter_revision: str | None = None
    processor_id: str | None = None
    processor_revision: str | None = None
    device: str = "cuda"
    torch_dtype: str = "float16"
    quantization: str = "none"
    attention_implementation: str = "eager"
    processor_use_fast: bool = False
    trust_remote_code: bool = False
    prompt_template: str = "paligemma"
    image_size: tuple[int, int] = (224, 224)
    image_token: str = "<image>"
    prompt_separator: str = ""
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None
    num_beams: int = 1
    return_token_logprobs: bool = True
    seed: int = 42
    target_score_reduction: str = "mean_logprob"
    include_eos_in_target_score: bool = False
    score_absolute_tolerance: float = 0.001
    attribution_methods: tuple[str, ...] = (
        "decoder_answer_to_image_attention",
        "answer_conditioned_grad_cam",
    )
    attribution_normalization: str = "minmax"
    attribution_output_dtype: str = "float32"
    attribution_zero_range_policy: str = "error"
    attention_layer: int = -1
    attention_head_aggregation: str = "mean"
    attention_token_aggregation: str = "mean"
    attention_image_tokens_only: bool = True
    grad_cam_vision_layer: int = -1
    grad_cam_gradient_pooling: str = "spatial_mean"
    grad_cam_activation_combination: str = "weighted_sum"
    grad_cam_relu: bool = True
    schema_version: str = field(default=PALIGEMMA_MODEL_SPEC_SCHEMA, init=False)
    model_type: str = field(default="paligemma", init=False)
    backend: str = field(default=PALIGEMMA_BACKEND, init=False)

    def __post_init__(self) -> None:
        _non_empty("base_model_id", self.base_model_id)
        validate_immutable_revision(
            self.base_model_revision,
            "base_model_revision",
            require_resolved=True,
            require_commit=True,
        )
        if (self.adapter_id is None) != (self.adapter_revision is None):
            raise ValueError("adapter_id and adapter_revision must be provided together")
        if self.adapter_id is not None:
            _non_empty("adapter_id", self.adapter_id)
            validate_immutable_revision(
                self.adapter_revision,
                "adapter_revision",
                require_resolved=True,
                require_commit=True,
            )
        if (self.processor_id is None) != (self.processor_revision is None):
            raise ValueError("processor_id and processor_revision must be provided together")
        if self.processor_id is not None:
            _non_empty("processor_id", self.processor_id)
            validate_immutable_revision(
                self.processor_revision,
                "processor_revision",
                require_resolved=True,
                require_commit=True,
            )
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {sorted(_DEVICES)}")
        if self.torch_dtype not in _DTYPES:
            raise ValueError(f"torch_dtype must be one of {sorted(_DTYPES)}")
        if self.quantization not in _QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {sorted(_QUANTIZATIONS)}")
        if self.attention_implementation not in _ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                f"attention_implementation must be one of {sorted(_ATTENTION_IMPLEMENTATIONS)}"
            )
        if not isinstance(self.processor_use_fast, bool):
            raise TypeError("processor_use_fast must be a boolean")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a boolean")
        if self.prompt_template != "paligemma":
            raise ValueError("prompt_template must be 'paligemma'")
        _non_empty("image_token", self.image_token)
        if not isinstance(self.prompt_separator, str):
            raise TypeError("prompt_separator must be a string")

        image_size = tuple(self.image_size)
        if len(image_size) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in image_size
        ):
            raise ValueError("image_size must contain two positive integers")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if self.do_sample:
            if (
                not isinstance(self.temperature, (int, float))
                or isinstance(self.temperature, bool)
                or not math.isfinite(float(self.temperature))
                or self.temperature <= 0
            ):
                raise ValueError(
                    "temperature must be a finite, positive number when do_sample is true"
                )
        elif self.temperature is not None:
            raise ValueError("temperature must be null when do_sample is false")
        if (
            not isinstance(self.num_beams, int)
            or isinstance(self.num_beams, bool)
            or self.num_beams < 1
        ):
            raise ValueError("num_beams must be a positive integer")
        if not isinstance(self.return_token_logprobs, bool):
            raise TypeError("return_token_logprobs must be a boolean")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.target_score_reduction != "mean_logprob":
            raise ValueError("target_score_reduction must be 'mean_logprob'")
        if not isinstance(self.include_eos_in_target_score, bool):
            raise TypeError("include_eos_in_target_score must be a boolean")
        if (
            not isinstance(self.score_absolute_tolerance, (int, float))
            or isinstance(self.score_absolute_tolerance, bool)
            or not math.isfinite(float(self.score_absolute_tolerance))
            or self.score_absolute_tolerance < 0
        ):
            raise ValueError("score_absolute_tolerance must be a finite non-negative number")

        attribution_methods = tuple(self.attribution_methods)
        if not attribution_methods:
            raise ValueError("attribution_methods must not be empty")
        if len(set(attribution_methods)) != len(attribution_methods):
            raise ValueError("attribution_methods must not contain duplicates")
        unknown_methods = set(attribution_methods) - _ATTRIBUTION_METHODS
        if unknown_methods:
            raise ValueError(f"unsupported attribution methods: {sorted(unknown_methods)}")
        if self.attribution_normalization != "minmax":
            raise ValueError("attribution_normalization must be 'minmax'")
        if self.attribution_output_dtype != "float32":
            raise ValueError("attribution_output_dtype must be 'float32'")
        if self.attribution_zero_range_policy not in {"error", "zeros"}:
            raise ValueError("attribution_zero_range_policy must be error or zeros")
        for name, value in (
            ("attention_layer", self.attention_layer),
            ("grad_cam_vision_layer", self.grad_cam_vision_layer),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.attention_head_aggregation not in {"mean", "max"}:
            raise ValueError("attention_head_aggregation must be mean or max")
        if self.attention_token_aggregation not in {"mean", "max"}:
            raise ValueError("attention_token_aggregation must be mean or max")
        if self.attention_image_tokens_only is not True:
            raise ValueError("attention_image_tokens_only must be true")
        if self.grad_cam_gradient_pooling != "spatial_mean":
            raise ValueError("grad_cam_gradient_pooling must be 'spatial_mean'")
        if self.grad_cam_activation_combination != "weighted_sum":
            raise ValueError("grad_cam_activation_combination must be 'weighted_sum'")
        if not isinstance(self.grad_cam_relu, bool):
            raise TypeError("grad_cam_relu must be a boolean")

        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "score_absolute_tolerance", float(self.score_absolute_tolerance))
        object.__setattr__(self, "attribution_methods", attribution_methods)
        if self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PaliGemmaModelSpec:
        """Map the validated study configuration into an immutable model spec."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        model = _section(config, "model")
        generation = _section(config, "generation")
        target_scoring = _section(config, "target_scoring", required=False)
        attribution = _section(config, "attribution", required=False)
        attention = attribution.get("attention", {})
        grad_cam = attribution.get("grad_cam", {})

        backend = model.get("backend", PALIGEMMA_BACKEND)
        if backend != PALIGEMMA_BACKEND:
            raise ValueError(f"model.backend must be {PALIGEMMA_BACKEND!r}")

        return cls(
            base_model_id=_required(model, "model", "base_model"),
            base_model_revision=_required(model, "model", "base_model_revision"),
            adapter_id=model.get("adapter"),
            adapter_revision=model.get("adapter_revision"),
            processor_id=model.get("processor"),
            processor_revision=model.get("processor_revision"),
            device=model.get("device", "cuda"),
            torch_dtype=model.get("precision", "float16"),
            quantization=model.get("quantization", "none"),
            attention_implementation=model.get("attn_implementation", "eager"),
            processor_use_fast=model.get("processor_use_fast", False),
            trust_remote_code=model.get("trust_remote_code", False),
            prompt_template=model.get("prompt_template", "paligemma"),
            max_new_tokens=generation.get("max_new_tokens", 64),
            do_sample=generation.get("do_sample", False),
            temperature=generation.get("temperature"),
            num_beams=generation.get("num_beams", 1),
            return_token_logprobs=generation.get("return_token_logprobs", True),
            seed=config.get("seed", 42),
            target_score_reduction=target_scoring.get("reduction", "mean_logprob"),
            include_eos_in_target_score=target_scoring.get("include_eos", False),
            score_absolute_tolerance=target_scoring.get("absolute_tolerance", 0.001),
            attribution_methods=tuple(
                attribution.get(
                    "methods",
                    (
                        "decoder_answer_to_image_attention",
                        "answer_conditioned_grad_cam",
                    ),
                )
            ),
            attribution_normalization=attribution.get("normalization", "minmax"),
            attribution_output_dtype=attribution.get("output_dtype", "float32"),
            attribution_zero_range_policy=attribution.get("zero_range_policy", "error"),
            attention_layer=attention.get("layer", -1),
            attention_head_aggregation=attention.get("head_aggregation", "mean"),
            attention_token_aggregation=attention.get(
                "answer_token_aggregation",
                "mean",
            ),
            attention_image_tokens_only=attention.get("image_tokens_only", True),
            grad_cam_vision_layer=grad_cam.get("vision_layer", -1),
            grad_cam_gradient_pooling=grad_cam.get(
                "gradient_pooling",
                "spatial_mean",
            ),
            grad_cam_activation_combination=grad_cam.get(
                "activation_combination",
                "weighted_sum",
            ),
            grad_cam_relu=grad_cam.get("relu", True),
        )

    @property
    def resolved_processor_id(self) -> str:
        return self.processor_id or self.base_model_id

    @property
    def resolved_processor_revision(self) -> str:
        return self.processor_revision or self.base_model_revision

    @property
    def is_adapted(self) -> bool:
        return self.adapter_id is not None

    def format_prompt(self, question: str) -> str:
        """Format the explicit image placeholder used by training and inference.

        The pinned native processor expands one literal ``<image>`` placeholder
        to its configured image sequence; it does not prepend a second sequence.
        """

        _non_empty("question", question)
        if self.image_token in question:
            raise ValueError(
                f"question must not contain the reserved image token {self.image_token!r}"
            )
        return f"{self.image_token}{self.prompt_separator}{question.strip()}"

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical, JSON-compatible specification."""

        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "backend": self.backend,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "processor_id": self.resolved_processor_id,
            "processor_revision": self.resolved_processor_revision,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "quantization": self.quantization,
            "attention_implementation": self.attention_implementation,
            "processor_use_fast": self.processor_use_fast,
            "trust_remote_code": self.trust_remote_code,
            "prompt_template": self.prompt_template,
            "image_size": list(self.image_size),
            "image_token": self.image_token,
            "prompt_separator": self.prompt_separator,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "num_beams": self.num_beams,
            "return_token_logprobs": self.return_token_logprobs,
            "seed": self.seed,
            "target_score_reduction": self.target_score_reduction,
            "include_eos_in_target_score": self.include_eos_in_target_score,
            "score_absolute_tolerance": self.score_absolute_tolerance,
            "attribution_methods": list(self.attribution_methods),
            "attribution_normalization": self.attribution_normalization,
            "attribution_output_dtype": self.attribution_output_dtype,
            "attribution_zero_range_policy": self.attribution_zero_range_policy,
            "attention_layer": self.attention_layer,
            "attention_head_aggregation": self.attention_head_aggregation,
            "attention_token_aggregation": self.attention_token_aggregation,
            "attention_image_tokens_only": self.attention_image_tokens_only,
            "grad_cam_vision_layer": self.grad_cam_vision_layer,
            "grad_cam_gradient_pooling": self.grad_cam_gradient_pooling,
            "grad_cam_activation_combination": self.grad_cam_activation_combination,
            "grad_cam_relu": self.grad_cam_relu,
        }

    def fingerprint(self) -> str:
        """Return a stable SHA-256 identity for all backend-behaviour settings."""

        payload = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_provenance(
        self,
        *,
        backend_version: str,
        backend_name: str | None = None,
        device: str | None = None,
        software_versions: Mapping[str, str] | None = None,
    ) -> BackendProvenance:
        """Build resolved backend provenance from this immutable specification."""

        resolved_backend_name = self.backend if backend_name is None else backend_name
        if resolved_backend_name != self.backend:
            raise ValueError(f"backend_name must be {self.backend!r}")
        versions = tuple(sorted((software_versions or {}).items()))
        return BackendProvenance(
            backend_name=resolved_backend_name,
            backend_version=backend_version,
            model_id=self.base_model_id,
            model_revision=self.base_model_revision,
            model_spec_sha256=self.fingerprint(),
            adapter_id=self.adapter_id,
            adapter_revision=self.adapter_revision,
            processor_id=self.resolved_processor_id,
            processor_revision=self.resolved_processor_revision,
            torch_dtype=self.torch_dtype,
            attention_implementation=self.attention_implementation,
            device=self.device if device is None else device,
            software_versions=versions,
        )
