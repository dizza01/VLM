"""Dependency-light contracts for vision-language model backends.

This module intentionally contains no imports from torch, transformers, PEFT, or
any concrete inference framework. Backend implementations may keep framework
objects in :class:`PreparedInput.payload` while exposing portable result types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_non_empty(name: str, value: str | None) -> str | None:
    if value is not None:
        _non_empty(name, value)
    return value


def _integer_tuple(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative integers")
    return result


def _logprob_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("token_logprobs must contain only finite values")
    if any(value > 1e-9 for value in result):
        raise ValueError("token_logprobs must be natural-log probabilities no greater than zero")
    return result


@dataclass(frozen=True)
class BackendProvenance:
    """Resolved identity of the backend and every model artifact it loaded."""

    backend_name: str
    backend_version: str
    model_id: str
    model_revision: str
    model_spec_sha256: str
    adapter_id: str | None = None
    adapter_revision: str | None = None
    processor_id: str | None = None
    processor_revision: str | None = None
    torch_dtype: str | None = None
    attention_implementation: str | None = None
    device: str | None = None
    software_versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "backend_name",
            "backend_version",
            "model_id",
            "model_revision",
        ):
            _non_empty(name, getattr(self, name))
        if len(self.model_spec_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_spec_sha256
        ):
            raise ValueError("model_spec_sha256 must be a lowercase SHA-256 digest")
        for name in (
            "adapter_id",
            "adapter_revision",
            "processor_id",
            "processor_revision",
            "torch_dtype",
            "attention_implementation",
            "device",
        ):
            _optional_non_empty(name, getattr(self, name))
        if (self.adapter_id is None) != (self.adapter_revision is None):
            raise ValueError("adapter_id and adapter_revision must be provided together")
        if (self.processor_id is None) != (self.processor_revision is None):
            raise ValueError("processor_id and processor_revision must be provided together")

        versions = tuple(sorted(tuple(pair) for pair in self.software_versions))
        if any(
            len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0].strip()
            or not isinstance(pair[1], str)
            or not pair[1].strip()
            for pair in versions
        ):
            raise ValueError("software_versions must contain non-empty name/version pairs")
        if len({name for name, _ in versions}) != len(versions):
            raise ValueError("software_versions must not contain duplicate package names")
        object.__setattr__(self, "software_versions", versions)

    def as_dict(self) -> dict:
        """Return a JSON-compatible provenance mapping."""

        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_spec_sha256": self.model_spec_sha256,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "processor_id": self.processor_id,
            "processor_revision": self.processor_revision,
            "torch_dtype": self.torch_dtype,
            "attention_implementation": self.attention_implementation,
            "device": self.device,
            "software_versions": dict(self.software_versions),
        }


@dataclass(frozen=True)
class PreparedInput:
    """A backend-prepared image/question pair and its opaque native payload."""

    item_id: str
    question: str
    prompt: str
    payload: Any = field(repr=False, compare=False)
    provenance: BackendProvenance
    image_reference: str | None = None
    input_token_ids: tuple[int, ...] = ()
    image_token_indices: tuple[int, ...] = ()
    preprocessing: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _non_empty("item_id", self.item_id)
        _non_empty("question", self.question)
        _non_empty("prompt", self.prompt)
        _optional_non_empty("image_reference", self.image_reference)
        input_token_ids = _integer_tuple("input_token_ids", self.input_token_ids)
        image_token_indices = _integer_tuple("image_token_indices", self.image_token_indices)
        if input_token_ids and any(index >= len(input_token_ids) for index in image_token_indices):
            raise ValueError("image_token_indices must refer to positions in input_token_ids")
        if len(set(image_token_indices)) != len(image_token_indices):
            raise ValueError("image_token_indices must not contain duplicates")
        if not isinstance(self.preprocessing, Mapping):
            raise TypeError("preprocessing must be a mapping")
        object.__setattr__(self, "input_token_ids", input_token_ids)
        object.__setattr__(self, "image_token_indices", image_token_indices)
        object.__setattr__(self, "preprocessing", dict(self.preprocessing))


@dataclass(frozen=True)
class GenerationResult:
    """Generated text with token-aligned natural-log probabilities."""

    text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    provenance: BackendProvenance
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        token_ids = _integer_tuple("token_ids", self.token_ids)
        token_logprobs = _logprob_tuple(self.token_logprobs)
        if len(token_ids) != len(token_logprobs):
            raise ValueError("token_ids and token_logprobs must have equal lengths")
        _optional_non_empty("finish_reason", self.finish_reason)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_logprobs", token_logprobs)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def mean_token_logprob(self) -> float | None:
        if not self.token_logprobs:
            return None
        return sum(self.token_logprobs) / len(self.token_logprobs)

    @property
    def sequence_confidence(self) -> float | None:
        """Return the geometric mean generated-token probability."""

        mean_logprob = self.mean_token_logprob
        return math.exp(mean_logprob) if mean_logprob is not None else None


@dataclass(frozen=True)
class TargetScore:
    """Teacher-forced score for one fixed answer under a prepared input."""

    target_text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    provenance: BackendProvenance
    includes_eos: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _non_empty("target_text", self.target_text)
        token_ids = _integer_tuple("token_ids", self.token_ids)
        token_logprobs = _logprob_tuple(self.token_logprobs)
        if not token_ids:
            raise ValueError("a target score must contain at least one target token")
        if len(token_ids) != len(token_logprobs):
            raise ValueError("token_ids and token_logprobs must have equal lengths")
        if not isinstance(self.includes_eos, bool):
            raise TypeError("includes_eos must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_logprobs", token_logprobs)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def mean_token_logprob(self) -> float:
        return sum(self.token_logprobs) / len(self.token_logprobs)

    @property
    def total_logprob(self) -> float:
        return sum(self.token_logprobs)


@dataclass(frozen=True)
class ScoreVerification:
    """Agreement between generation-time and teacher-forced token scores."""

    token_count: int
    absolute_tolerance: float
    maximum_absolute_difference: float
    mean_absolute_difference: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 1
        ):
            raise ValueError("token_count must be a positive integer")
        for name in (
            "absolute_tolerance",
            "maximum_absolute_difference",
            "mean_absolute_difference",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, float(value))
        if self.maximum_absolute_difference > self.absolute_tolerance:
            raise ValueError("maximum_absolute_difference must not exceed absolute_tolerance")

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible verification record."""

        return {
            "token_count": self.token_count,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "mean_absolute_difference": self.mean_absolute_difference,
        }


@dataclass(frozen=True)
class AttributionResult:
    """Answer-conditioned patch attribution returned by a concrete backend."""

    method: str
    values: Any = field(repr=False, compare=False)
    patch_grid_shape: tuple[int, int]
    target_score: TargetScore
    image_token_indices: tuple[int, ...]
    aggregation: str
    provenance: BackendProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _non_empty("method", self.method)
        _non_empty("aggregation", self.aggregation)
        if self.values is None:
            raise ValueError("values must contain a backend-native attribution array")
        shape = tuple(self.patch_grid_shape)
        if len(shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape
        ):
            raise ValueError("patch_grid_shape must contain two positive integers")
        image_token_indices = _integer_tuple("image_token_indices", self.image_token_indices)
        if self.target_score.provenance != self.provenance:
            raise ValueError("target_score and attribution provenance must match")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "patch_grid_shape", shape)
        object.__setattr__(self, "image_token_indices", image_token_indices)
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class VisionLanguageBackend(Protocol):
    """Shared contract implemented by concrete PaliGemma or other VLM backends."""

    @property
    def provenance(self) -> BackendProvenance:
        """Return the resolved identity of the currently loaded backend."""

    def prepare(self, *, item_id: str, image: Any, question: str) -> PreparedInput:
        """Preprocess one image/question pair without generating an answer."""

    def generate(self, prepared: PreparedInput) -> GenerationResult:
        """Generate an answer and token-aligned log probabilities."""

    def score_target(
        self,
        prepared: PreparedInput,
        target_text: str,
        *,
        include_eos: bool | None = None,
        expected_token_ids: Sequence[int] | None = None,
    ) -> TargetScore:
        """Teacher-force one fixed answer and optionally verify its saved token IDs."""

    def verify_generation_score(
        self,
        generation: GenerationResult,
        target_score: TargetScore,
        *,
        absolute_tolerance: float | None = None,
    ) -> ScoreVerification:
        """Verify fixed-target identity and generation/teacher-forcing score parity."""

    def attribute(
        self,
        prepared: PreparedInput,
        generation: GenerationResult,
        *,
        method: str,
    ) -> AttributionResult:
        """Return an answer-conditioned patch attribution."""

    def close(self) -> None:
        """Release backend-owned accelerator or file resources."""
