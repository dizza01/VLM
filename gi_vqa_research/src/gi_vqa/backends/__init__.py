"""Framework-independent contracts shared by concrete VLM backends."""

from .base import (
    AttributionResult,
    BackendProvenance,
    GenerationResult,
    PreparedInput,
    ScoreVerification,
    TargetScore,
    VisionLanguageBackend,
)
from .paligemma import (
    AttributionError,
    BackendCompatibilityError,
    BackendDependencyError,
    PaliGemmaBackend,
)

__all__ = [
    "AttributionError",
    "AttributionResult",
    "BackendCompatibilityError",
    "BackendDependencyError",
    "BackendProvenance",
    "GenerationResult",
    "PaliGemmaBackend",
    "PreparedInput",
    "ScoreVerification",
    "TargetScore",
    "VisionLanguageBackend",
]
