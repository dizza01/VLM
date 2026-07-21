"""Deterministic patch interventions for Study 1 visual-faithfulness tests."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .provenance import canonical_json_sha256

PERTURBATION_PLAN_SCHEMA_VERSION = "gi-vqa-perturbation-plan-v1"


@dataclass(frozen=True)
class PatchIntervention:
    """One completely specified deletion or insertion intervention."""

    intervention_id: str
    operation: str
    treatment: str
    selection: str
    fraction: float
    repeat: int
    seed: int | None
    gray_value: int
    blur_radius: float
    patch_grid_shape: tuple[int, int]
    patch_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["patch_grid_shape"] = list(self.patch_grid_shape)
        value["patch_indices"] = list(self.patch_indices)
        return value


def build_intervention_plan(
    attribution_values: Any,
    *,
    item_id: str,
    method: str,
    seed: int,
    config: Mapping[str, Any],
) -> list[PatchIntervention]:
    """Build a stable intervention plan from one normalised patch map."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("NumPy is required for perturbation planning") from exc

    values = np.asarray(attribution_values, dtype=np.float32)
    if values.ndim != 2 or any(size < 1 for size in values.shape):
        raise ValueError("attribution_values must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("attribution_values must contain only finite values")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("item_id must be a non-empty string")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    fractions = tuple(float(value) for value in config["patch_fractions"])
    deletion = tuple(str(value) for value in config["deletion_treatments"])
    insertion = tuple(str(value) for value in config["insertion_treatments"])
    modes = tuple(str(value) for value in config["selection_modes"])
    random_repeats = int(config["random_repeats"])
    gray_value = int(config["gray_value"])
    blur_radius = float(config["blur_radius"])
    flat = [float(value) for value in values.reshape(-1)]
    total = len(flat)
    grid = (int(values.shape[0]), int(values.shape[1]))
    descending = tuple(sorted(range(total), key=lambda index: (-flat[index], index)))
    ascending = tuple(sorted(range(total), key=lambda index: (flat[index], index)))

    plan: list[PatchIntervention] = []
    for operation, treatments in (
        ("deletion", deletion),
        ("insertion", insertion),
    ):
        for treatment in treatments:
            for fraction in fractions:
                patch_count = min(total, max(1, math.ceil(total * fraction)))
                for selection in modes:
                    repeats = range(random_repeats) if selection == "random" else range(1)
                    for repeat in repeats:
                        intervention_seed = None
                        if selection == "most_salient":
                            indices = descending[:patch_count]
                        elif selection == "least_salient":
                            indices = ascending[:patch_count]
                        elif selection == "random":
                            intervention_seed = _intervention_seed(
                                seed=seed,
                                item_id=item_id,
                                method=method,
                                operation=operation,
                                treatment=treatment,
                                fraction=fraction,
                                repeat=repeat,
                            )
                            population = list(range(total))
                            random.Random(intervention_seed).shuffle(population)
                            indices = tuple(population[:patch_count])
                        else:  # validated configuration should make this unreachable
                            raise ValueError(f"unsupported selection mode: {selection}")
                        identity = {
                            "schema_version": PERTURBATION_PLAN_SCHEMA_VERSION,
                            "item_id": item_id,
                            "method": method,
                            "operation": operation,
                            "treatment": treatment,
                            "selection": selection,
                            "fraction": fraction,
                            "repeat": repeat,
                            "seed": intervention_seed,
                            "gray_value": gray_value,
                            "blur_radius": blur_radius,
                            "patch_grid_shape": list(grid),
                            "patch_indices": list(indices),
                        }
                        plan.append(
                            PatchIntervention(
                                intervention_id=canonical_json_sha256(identity)[:24],
                                operation=operation,
                                treatment=treatment,
                                selection=selection,
                                fraction=fraction,
                                repeat=repeat,
                                seed=intervention_seed,
                                gray_value=gray_value,
                                blur_radius=blur_radius,
                                patch_grid_shape=grid,
                                patch_indices=tuple(indices),
                            )
                        )
    if len({item.intervention_id for item in plan}) != len(plan):
        raise RuntimeError("perturbation plan contains duplicate intervention IDs")
    return plan


def apply_patch_intervention(image: Any, intervention: PatchIntervention) -> Any:
    """Apply one patch intervention and return a new RGB PIL image."""

    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("Pillow is required for patch interventions") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    source = image.convert("RGB")
    rows, columns = intervention.patch_grid_shape
    if intervention.treatment == "gray":
        baseline = Image.new(
            "RGB",
            source.size,
            color=(intervention.gray_value,) * 3,
        )
    elif intervention.treatment == "blur":
        baseline = source.filter(
            ImageFilter.GaussianBlur(radius=intervention.blur_radius)
        )
    else:
        raise ValueError(f"unsupported intervention treatment: {intervention.treatment}")

    if intervention.operation == "deletion":
        output = source.copy()
        replacement = baseline
    elif intervention.operation == "insertion":
        output = baseline.copy()
        replacement = source
    else:
        raise ValueError(f"unsupported intervention operation: {intervention.operation}")

    width, height = source.size
    for index in intervention.patch_indices:
        if index < 0 or index >= rows * columns:
            raise ValueError("patch index is outside the intervention grid")
        row, column = divmod(index, columns)
        left = round(column * width / columns)
        right = round((column + 1) * width / columns)
        top = round(row * height / rows)
        bottom = round((row + 1) * height / rows)
        box = (left, top, right, bottom)
        output.paste(replacement.crop(box), box)
    return output


def intervention_plan_sha256(plan: Sequence[PatchIntervention]) -> str:
    """Fingerprint an ordered intervention plan."""

    return canonical_json_sha256([item.as_dict() for item in plan])


def _intervention_seed(
    *,
    seed: int,
    item_id: str,
    method: str,
    operation: str,
    treatment: str,
    fraction: float,
    repeat: int,
) -> int:
    payload = "\0".join(
        (
            PERTURBATION_PLAN_SCHEMA_VERSION,
            str(seed),
            item_id,
            method,
            operation,
            treatment,
            format(fraction, ".17g"),
            str(repeat),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "PERTURBATION_PLAN_SCHEMA_VERSION",
    "PatchIntervention",
    "apply_patch_intervention",
    "build_intervention_plan",
    "intervention_plan_sha256",
]
