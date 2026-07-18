"""ms-swift plugin for the canonical Study 1 PaliGemma training template.

This module is loaded through ms-swift's ``--external_plugins`` mechanism.
Importing it registers a new template ID; the upstream ``paligemma`` template
is deliberately left unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from typing import Any

from swift.llm import (
    TEMPLATE_MAPPING,
    TemplateType,
    get_template,
    get_template_meta,
    register_template,
)
from swift.llm.template.template.gemma import PaliGemmaTemplate

from gi_vqa.training import (
    EXPECTED_MS_SWIFT_VERSION,
    STUDY1_SWIFT_TEMPLATE_TYPE,
    SWIFT_TOKEN_TYPE_CORRECTION_ID,
    TrainingCompatibilityError,
    correct_ms_swift_paligemma_training_encoding,
)

OBSERVED_MS_SWIFT_VERSION = version("ms-swift")


class Study1PaliGemmaTemplate(PaliGemmaTemplate):
    """PaliGemma template with canonical training prefix/suffix token types."""

    placeholder_tokens = ["<image>"]
    gi_vqa_correction_id = SWIFT_TOKEN_TYPE_CORRECTION_ID

    def _encode(self, inputs: Any) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        if encoded.get("labels") is None:
            return encoded
        corrected, _correction = (
            correct_ms_swift_paligemma_training_encoding(
                encoded,
                package_version=OBSERVED_MS_SWIFT_VERSION,
            )
        )
        return corrected


def register_study1_paligemma_template() -> None:
    """Register the version-pinned template idempotently."""

    if OBSERVED_MS_SWIFT_VERSION != EXPECTED_MS_SWIFT_VERSION:
        raise TrainingCompatibilityError(
            "the Study 1 Swift plugin requires "
            f"ms-swift {EXPECTED_MS_SWIFT_VERSION}; observed "
            f"{OBSERVED_MS_SWIFT_VERSION}"
        )

    existing = TEMPLATE_MAPPING.get(STUDY1_SWIFT_TEMPLATE_TYPE)
    if existing is not None:
        existing_correction = getattr(
            existing.template_cls,
            "gi_vqa_correction_id",
            None,
        )
        if existing_correction != SWIFT_TOKEN_TYPE_CORRECTION_ID:
            raise TrainingCompatibilityError(
                f"template id {STUDY1_SWIFT_TEMPLATE_TYPE!r} is already "
                "registered by a different implementation"
            )
        return

    template_meta = deepcopy(get_template_meta(TemplateType.paligemma))
    template_meta.template_type = STUDY1_SWIFT_TEMPLATE_TYPE
    template_meta.template_cls = Study1PaliGemmaTemplate
    register_template(template_meta)


def get_study1_paligemma_template(
    processor: Any,
    *,
    max_length: int = 512,
) -> Any:
    """Build the same corrected template used by production training."""

    register_study1_paligemma_template()
    return get_template(
        STUDY1_SWIFT_TEMPLATE_TYPE,
        processor,
        max_length=max_length,
        template_backend="swift",
    )


register_study1_paligemma_template()
