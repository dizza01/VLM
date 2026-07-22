"""Shared Transformers/PEFT backend for the pinned PaliGemma Study 1 stack.

Heavy GPU dependencies are imported only when :meth:`PaliGemmaBackend.load` is
called. Importing this module therefore remains safe in the lightweight local
test environment.

The backend deliberately uses one processor/model/adapter instance for
generation, teacher-forced target scoring, decoder attention and Grad-CAM. This
is a scientific control: the model being explained must be the model that
generated the saved answer.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import validate_config
from ..model_spec import PaliGemmaModelSpec
from .base import (
    AttributionResult,
    BackendProvenance,
    GenerationResult,
    PreparedInput,
    ScoreVerification,
    TargetScore,
)

BACKEND_NAME = "transformers-paligemma"
BACKEND_VERSION = "1"
SUPPORTED_ATTRIBUTIONS = {
    "decoder_answer_to_image_attention",
    "answer_conditioned_grad_cam",
}


class BackendDependencyError(RuntimeError):
    """Raised when the pinned optional GPU stack is not installed."""


class BackendCompatibilityError(RuntimeError):
    """Raised when a loaded artifact violates the PaliGemma contract."""


class AttributionError(RuntimeError):
    """Raised when an attribution cannot produce a valid patch map."""


@dataclass
class _PreparedPayload:
    """Backend-private tensors and source image retained for later stages."""

    image: Any
    model_inputs: dict[str, Any]
    patch_grid_shape: tuple[int, int]


class PaliGemmaBackend:
    """One loaded PaliGemma processor/model/adapter used by every model stage."""

    def __init__(
        self,
        spec: PaliGemmaModelSpec,
        *,
        device: str | None = None,
        quantization: str | None = None,
        processor_use_fast: bool | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        device = spec.device if device is None else device
        quantization = spec.quantization if quantization is None else quantization
        processor_use_fast = (
            spec.processor_use_fast if processor_use_fast is None else processor_use_fast
        )
        trust_remote_code = (
            spec.trust_remote_code if trust_remote_code is None else trust_remote_code
        )
        if device != spec.device:
            raise ValueError("backend device must match the immutable model specification")
        if quantization != spec.quantization:
            raise ValueError("backend quantization must match the immutable model specification")
        if processor_use_fast != spec.processor_use_fast:
            raise ValueError("backend processor mode must match the immutable model specification")
        if trust_remote_code != spec.trust_remote_code:
            raise ValueError(
                "backend trust_remote_code must match the immutable model specification"
            )
        if spec.do_sample or spec.num_beams != 1 or not spec.return_token_logprobs:
            raise ValueError(
                "the Study 1 backend requires greedy one-beam generation with "
                "token log probabilities"
            )
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if quantization not in {"none", "bnb-nf4-4bit"}:
            raise ValueError("quantization must be none or bnb-nf4-4bit")
        if not isinstance(processor_use_fast, bool):
            raise TypeError("processor_use_fast must be a Boolean")
        if trust_remote_code is not False:
            raise ValueError("Study 1 does not permit trust_remote_code")

        self.spec = spec
        self.device_request = device
        self.quantization = quantization
        self.processor_use_fast = processor_use_fast
        self.trust_remote_code = trust_remote_code
        self.attention_layer = spec.attention_layer
        self.attention_head_aggregation = spec.attention_head_aggregation
        self.attention_token_aggregation = spec.attention_token_aggregation
        self.grad_cam_vision_layer = spec.grad_cam_vision_layer
        self.grad_cam_relu = spec.grad_cam_relu

        self._torch = None
        self._processor = None
        self._model = None
        self._torch_dtype = None
        self._input_device = None
        self._provenance: BackendProvenance | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        load: bool = True,
    ) -> PaliGemmaBackend:
        """Build the backend from a fully resolved model-execution config."""

        resolved = validate_config(
            config,
            require_resolved=True,
            require_model_execution=True,
        )
        spec = PaliGemmaModelSpec.from_config(resolved)
        backend = cls(spec)
        return backend.load() if load else backend

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._processor is not None
            and self._provenance is not None
            and self._input_device is not None
        )

    @property
    def provenance(self) -> BackendProvenance:
        if self._provenance is None:
            raise RuntimeError("PaliGemma backend has not been loaded")
        return self._provenance

    def load(self) -> PaliGemmaBackend:
        """Load the exact processor, base model and optional PEFT adapter once."""

        if self.is_loaded:
            return self
        torch, transformers, peft = _import_gpu_stack()
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, self.spec.torch_dtype)
        self._torch_dtype = dtype

        if self.device_request == "cuda" and not torch.cuda.is_available():
            raise BackendCompatibilityError(
                "configuration requires CUDA, but torch.cuda.is_available() is false"
            )

        processor = transformers.AutoProcessor.from_pretrained(
            self.spec.resolved_processor_id,
            revision=self.spec.resolved_processor_revision,
            use_fast=self.processor_use_fast,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "revision": self.spec.base_model_revision,
            "torch_dtype": dtype,
            "attn_implementation": self.spec.attention_implementation,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_request == "auto":
            model_kwargs["device_map"] = "auto"
        if self.quantization == "bnb-nf4-4bit":
            if self.device_request == "cuda":
                model_kwargs["device_map"] = "cuda:0"
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=[
                    "model.vision_tower",
                    "model.multi_modal_projector",
                    "lm_head",
                ],
            )

        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            self.spec.base_model_id,
            **model_kwargs,
        )
        if self.spec.is_adapted:
            adapter_kwargs: dict[str, Any] = {"is_trainable": False}
            if not Path(self.spec.adapter_id).is_dir():
                adapter_kwargs["revision"] = self.spec.adapter_revision
            model = peft.PeftModel.from_pretrained(
                model,
                self.spec.adapter_id,
                **adapter_kwargs,
            )
        if self.device_request != "auto" and self.quantization == "none":
            model = model.to(torch.device(self.device_request))
        model.eval()
        model.requires_grad_(False)

        self._processor = processor
        self._model = model
        self._input_device = _model_input_device(torch, model)
        try:
            self._validate_loaded_architecture()
            self._provenance = self.spec.to_provenance(
                backend_name=BACKEND_NAME,
                backend_version=BACKEND_VERSION,
                device=str(self._input_device),
                software_versions={
                    name: installed
                    for name in ("torch", "transformers", "peft", "bitsandbytes")
                    if (installed := _installed_version(name)) is not None
                },
            )
        except Exception:
            self._model = None
            self._processor = None
            self._input_device = None
            self._provenance = None
            self._torch_dtype = None
            raise
        return self

    def prepare(self, *, item_id: str, image: Any, question: str) -> PreparedInput:
        """Prepare one image/question pair with the pinned processor."""

        self._require_loaded()
        pil_image, image_reference = _coerce_rgb_image(image)
        prompt = self.spec.format_prompt(question)
        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        processor_inputs = dict(inputs)
        pixel_values = processor_inputs["pixel_values"]
        model_inputs = self._move_inputs(processor_inputs)
        input_ids = model_inputs["input_ids"][0]
        image_token_id = _image_token_id(self._processor, self._base_model())
        image_indices_tensor = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
        image_indices = tuple(int(value) for value in image_indices_tensor.detach().cpu().tolist())
        patch_grid_shape = self._patch_grid_shape(len(image_indices))
        preprocessing = self._preprocessing_metadata(
            image=pil_image,
            image_reference=image_reference,
            pixel_values=pixel_values,
            patch_grid_shape=patch_grid_shape,
        )
        payload = _PreparedPayload(
            image=pil_image,
            model_inputs=model_inputs,
            patch_grid_shape=patch_grid_shape,
        )
        return PreparedInput(
            item_id=item_id,
            question=question,
            prompt=prompt,
            payload=payload,
            provenance=self.provenance,
            image_reference=image_reference,
            input_token_ids=tuple(int(value) for value in input_ids.detach().cpu().tolist()),
            image_token_indices=image_indices,
            preprocessing=preprocessing,
        )

    def generate(self, prepared: PreparedInput) -> GenerationResult:
        """Greedily generate one answer and save token-aligned log probabilities."""

        payload = self._validate_prepared(prepared)
        torch = self._torch
        self._set_deterministic_seed()
        inputs = _clone_tensor_mapping(payload.model_inputs)
        generation_kwargs = {
            "max_new_tokens": self.spec.max_new_tokens,
            "do_sample": self.spec.do_sample,
            "num_beams": self.spec.num_beams,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }
        if self.spec.temperature is not None:
            generation_kwargs["temperature"] = self.spec.temperature
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation_kwargs)

        if int(inputs["input_ids"].shape[0]) != 1:
            raise BackendCompatibilityError("shared backend requires generation batch size one")
        prompt_length = int(inputs["input_ids"].shape[-1])
        step_count = len(output.scores)
        if int(output.sequences.shape[0]) != 1:
            raise BackendCompatibilityError("generation returned more than one sequence")
        if int(output.sequences.shape[-1]) < prompt_length + step_count:
            raise BackendCompatibilityError(
                "generation sequence is shorter than its prompt and score tensors"
            )
        if not bool(output.sequences[0, :prompt_length].equal(inputs["input_ids"][0])):
            raise BackendCompatibilityError(
                "generation output does not preserve the prepared prompt prefix"
            )
        raw_ids_tensor = output.sequences[0, prompt_length : prompt_length + step_count]
        raw_ids = [int(value) for value in raw_ids_tensor.detach().cpu().tolist()]
        if len(raw_ids) != step_count:
            raise BackendCompatibilityError(
                "generation returned a different number of tokens and score tensors"
            )
        raw_logprobs = []
        for step, token_id in zip(output.scores, raw_ids):  # noqa: B905
            value = torch.log_softmax(step[0].float(), dim=-1)[token_id]
            raw_logprobs.append(float(value.detach().cpu().item()))

        eos_id = self._processor.tokenizer.eos_token_id
        pad_id = self._processor.tokenizer.pad_token_id
        token_ids: list[int] = []
        token_logprobs: list[float] = []
        finish_reason = "length" if step_count >= self.spec.max_new_tokens else "stop"
        for token_id, logprob in zip(raw_ids, raw_logprobs):  # noqa: B905
            if eos_id is not None and token_id == int(eos_id):
                finish_reason = "eos"
                break
            if pad_id is not None and token_id == int(pad_id):
                break
            token_ids.append(token_id)
            token_logprobs.append(logprob)
        if not token_ids:
            raise BackendCompatibilityError(
                "generation produced no scoreable answer tokens before EOS or padding"
            )
        text = self._processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not text.strip():
            raise BackendCompatibilityError("generated answer tokens decode to empty text")
        return GenerationResult(
            text=text,
            token_ids=tuple(token_ids),
            token_logprobs=tuple(token_logprobs),
            provenance=self.provenance,
            finish_reason=finish_reason,
            metadata={
                "raw_generated_token_ids": raw_ids,
                "raw_includes_eos": bool(eos_id is not None and int(eos_id) in raw_ids),
                "prompt_token_count": prompt_length,
                "decoding": generation_kwargs,
                "display_text": text.strip(),
            },
        )

    def score_target(
        self,
        prepared: PreparedInput,
        target_text: str,
        *,
        include_eos: bool | None = None,
        expected_token_ids: Sequence[int] | None = None,
    ) -> TargetScore:
        """Teacher-force and score only the fixed saved answer tokens."""

        resolved_include_eos = self.spec.include_eos_in_target_score
        if include_eos is not None and include_eos != resolved_include_eos:
            raise ValueError(
                f"include_eos must match the immutable model specification ({resolved_include_eos})"
            )
        payload = self._validate_prepared(prepared)
        inputs = self._prepare_target_inputs(prepared, payload, target_text)
        torch = self._torch
        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        score = self._target_score_from_logits(
            target_text,
            inputs["labels"],
            outputs.logits,
            include_eos=resolved_include_eos,
        )
        if expected_token_ids is not None:
            expected = tuple(int(value) for value in expected_token_ids)
            if score.token_ids != expected:
                raise BackendCompatibilityError(
                    "saved generated token IDs do not match processor suffix "
                    f"tokenisation: expected {expected}, observed {score.token_ids}"
                )
        return score

    def verify_generation_score(
        self,
        generation: GenerationResult,
        target_score: TargetScore,
        *,
        absolute_tolerance: float | None = None,
    ) -> ScoreVerification:
        """Require exact target identity and near-identical per-token scores."""

        if not isinstance(generation, GenerationResult):
            raise TypeError("generation must be a GenerationResult")
        if not isinstance(target_score, TargetScore):
            raise TypeError("target_score must be a TargetScore")
        if generation.provenance != self.provenance:
            raise BackendCompatibilityError("generation belongs to a different model backend")
        if target_score.provenance != self.provenance:
            raise BackendCompatibilityError("target score belongs to a different model backend")
        if target_score.includes_eos != self.spec.include_eos_in_target_score:
            raise BackendCompatibilityError(
                "target score EOS policy differs from the immutable model specification"
            )
        if generation.token_ids != target_score.token_ids:
            raise BackendCompatibilityError("generation and teacher-forced target token IDs differ")
        if len(generation.token_logprobs) != len(target_score.token_logprobs):
            raise BackendCompatibilityError(
                "generation and teacher-forced score sequences have different lengths"
            )

        tolerance = self.spec.score_absolute_tolerance
        if absolute_tolerance is not None and absolute_tolerance != tolerance:
            raise ValueError(
                f"absolute_tolerance must match the immutable model specification ({tolerance})"
            )
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("absolute_tolerance must be a finite non-negative number")
        differences = [
            abs(generated - teacher_forced)
            for generated, teacher_forced in zip(  # noqa: B905
                generation.token_logprobs,
                target_score.token_logprobs,
            )
        ]
        maximum = max(differences)
        mean = sum(differences) / len(differences)
        if maximum > float(tolerance):
            raise BackendCompatibilityError(
                "generation and teacher-forced token log probabilities differ "
                f"by up to {maximum:.8g}, exceeding tolerance {float(tolerance):.8g}"
            )
        return ScoreVerification(
            token_count=len(differences),
            absolute_tolerance=float(tolerance),
            maximum_absolute_difference=maximum,
            mean_absolute_difference=mean,
        )

    def attribute(
        self,
        prepared: PreparedInput,
        generation: GenerationResult,
        *,
        method: str,
    ) -> AttributionResult:
        """Compute one answer-conditioned attribution using the shared model."""

        if not isinstance(generation, GenerationResult):
            raise TypeError("attribute requires the saved GenerationResult, not decoded text")
        if generation.provenance != self.provenance:
            raise BackendCompatibilityError("generation belongs to a different model backend")
        if method not in self.spec.attribution_methods:
            raise ValueError(
                f"attribution method {method!r} is not enabled by the model specification"
            )
        if method not in SUPPORTED_ATTRIBUTIONS:
            raise ValueError(f"unsupported PaliGemma attribution method: {method}")

        preflight_score = self.score_target(
            prepared,
            generation.text,
            expected_token_ids=generation.token_ids,
        )
        preflight = self.verify_generation_score(generation, preflight_score)
        if method == "decoder_answer_to_image_attention":
            result = self._answer_to_image_attention(prepared, generation.text)
        else:
            result = self._answer_conditioned_grad_cam(prepared, generation.text)
        repeated = self.verify_generation_score(generation, result.target_score)
        return replace(
            result,
            metadata={
                **result.metadata,
                "preflight_generation_score_verification": preflight.as_dict(),
                "attribution_generation_score_verification": repeated.as_dict(),
            },
        )

    def _answer_to_image_attention(
        self,
        prepared: PreparedInput,
        target_text: str,
    ) -> AttributionResult:
        if self.spec.attention_implementation != "eager":
            raise AttributionError("decoder attention extraction requires eager attention")
        payload = self._validate_prepared(prepared)
        inputs = self._prepare_target_inputs(prepared, payload, target_text)
        torch = self._torch
        with torch.inference_mode():
            outputs = self._model(
                **inputs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if not outputs.attentions:
            raise AttributionError("model returned no decoder attention tensors")
        layer_index = _resolve_layer_index(
            self.attention_layer,
            len(outputs.attentions),
            "decoder attention",
        )
        labels = inputs["labels"]
        target_positions = self._target_positions(
            labels,
            include_eos=self.spec.include_eos_in_target_score,
        )
        query_positions = target_positions - 1
        if bool((query_positions < 0).any()):
            raise AttributionError("target token has no preceding prediction position")

        attention = outputs.attentions[layer_index][0]
        if attention.ndim != 3:
            raise AttributionError("decoder attention must have [heads, query, key] dimensions")
        sequence_length = int(labels.shape[-1])
        if int(attention.shape[1]) != sequence_length or int(attention.shape[2]) != sequence_length:
            raise AttributionError(
                "decoder attention sequence dimensions do not match target inputs"
            )
        image_positions = torch.tensor(
            prepared.image_token_indices,
            dtype=torch.long,
            device=attention.device,
        )
        selected = attention.index_select(1, query_positions).index_select(2, image_positions)
        selected = _aggregate_tensor(
            selected,
            dimension=0,
            mode=self.attention_head_aggregation,
        )
        selected = _aggregate_tensor(
            selected,
            dimension=0,
            mode=self.attention_token_aggregation,
        )
        values = selected.detach().float().cpu().numpy().reshape(payload.patch_grid_shape)
        values, normalization = _normalise_attribution_values(
            values,
            zero_range_policy=self.spec.attribution_zero_range_policy,
        )
        target_score = self._target_score_from_logits(
            target_text,
            labels,
            outputs.logits,
            include_eos=self.spec.include_eos_in_target_score,
        )
        return AttributionResult(
            method="decoder_answer_to_image_attention",
            values=values,
            patch_grid_shape=payload.patch_grid_shape,
            target_score=target_score,
            image_token_indices=prepared.image_token_indices,
            aggregation=(
                f"layer={layer_index};heads={self.attention_head_aggregation};"
                f"answer_tokens={self.attention_token_aggregation}"
            ),
            provenance=self.provenance,
            metadata={
                "requested_layer": self.attention_layer,
                "resolved_layer": layer_index,
                "target_token_positions": [
                    int(value) for value in target_positions.detach().cpu().tolist()
                ],
                "query_positions": [
                    int(value) for value in query_positions.detach().cpu().tolist()
                ],
                "normalization": normalization,
            },
        )

    def _answer_conditioned_grad_cam(
        self,
        prepared: PreparedInput,
        target_text: str,
    ) -> AttributionResult:
        payload = self._validate_prepared(prepared)
        inputs = self._prepare_target_inputs(prepared, payload, target_text)
        torch = self._torch
        inputs["pixel_values"] = inputs["pixel_values"].detach().requires_grad_(True)

        vision_layers, vision_path = self._vision_layers()
        layer_index = _resolve_layer_index(
            self.grad_cam_vision_layer,
            len(vision_layers),
            "vision",
        )
        captured: dict[str, Any] = {}

        def capture_activation(_module, _arguments, output):
            activation = output[0] if isinstance(output, tuple) else output
            if not getattr(activation, "requires_grad", False):
                raise AttributionError("vision activation is not connected to a gradient graph")
            expected_patches = payload.patch_grid_shape[0] * payload.patch_grid_shape[1]
            if (
                activation.ndim != 3
                or int(activation.shape[0]) != 1
                or int(activation.shape[1]) != expected_patches
            ):
                raise AttributionError(
                    "vision activation must have batch-one patch-token dimensions "
                    f"[1, {expected_patches}, channels]"
                )
            activation.retain_grad()
            captured["activation"] = activation

        hook = vision_layers[layer_index].register_forward_hook(capture_activation)
        self._model.zero_grad(set_to_none=True)
        try:
            outputs = self._model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
            labels = inputs["labels"]
            token_ids, token_logprobs, _positions = self._target_tensor_values(
                labels,
                outputs.logits,
                include_eos=self.spec.include_eos_in_target_score,
            )
            scalar_target = token_logprobs.mean()
            scalar_target.backward()
            if any(parameter.grad is not None for parameter in self._model.parameters()):
                raise AttributionError(
                    "model parameter gradients were populated during frozen Grad-CAM"
                )
            activation = captured.get("activation")
            if activation is None or activation.grad is None:
                raise AttributionError("vision-layer gradient was not captured")
            gradients = activation.grad
            weights = gradients.mean(dim=1, keepdim=True)
            cam = (weights * activation).sum(dim=-1)[0]
            if self.grad_cam_relu:
                cam = torch.relu(cam)
            values = cam.detach().float().cpu().numpy().reshape(payload.patch_grid_shape)
            values, normalization = _normalise_attribution_values(
                values,
                zero_range_policy=self.spec.attribution_zero_range_policy,
            )
            target_score = TargetScore(
                target_text=target_text,
                token_ids=tuple(int(value) for value in token_ids.detach().cpu().tolist()),
                token_logprobs=tuple(
                    float(value) for value in token_logprobs.detach().float().cpu().tolist()
                ),
                provenance=self.provenance,
                includes_eos=self.spec.include_eos_in_target_score,
                metadata={"reduction": "mean_logprob"},
            )
        finally:
            hook.remove()
            self._model.zero_grad(set_to_none=True)

        resolved_path = f"{vision_path}.layers[{layer_index}]"
        return AttributionResult(
            method="answer_conditioned_grad_cam",
            values=values,
            patch_grid_shape=payload.patch_grid_shape,
            target_score=target_score,
            image_token_indices=prepared.image_token_indices,
            aggregation="spatial_mean_gradients;weighted_sum_channels",
            provenance=self.provenance,
            metadata={
                "requested_vision_layer": self.grad_cam_vision_layer,
                "resolved_vision_layer": layer_index,
                "resolved_module_path": resolved_path,
                "relu": self.grad_cam_relu,
                "normalization": normalization,
            },
        )

    def _prepare_target_inputs(
        self,
        prepared: PreparedInput,
        payload: _PreparedPayload,
        target_text: str,
    ) -> dict[str, Any]:
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("target_text must be a non-empty string")
        inputs = self._processor(
            images=payload.image,
            text=prepared.prompt,
            suffix=target_text,
            return_tensors="pt",
        )
        moved = self._move_inputs(dict(inputs))
        required = {"input_ids", "attention_mask", "pixel_values", "labels", "token_type_ids"}
        missing = required - set(moved)
        if missing:
            raise BackendCompatibilityError(
                f"PaliGemma suffix processing omitted fields: {sorted(missing)}"
            )
        prefix_ids = tuple(
            int(value)
            for value in moved["input_ids"][0, : len(prepared.input_token_ids)]
            .detach()
            .cpu()
            .tolist()
        )
        if prefix_ids != prepared.input_token_ids:
            raise BackendCompatibilityError(
                "processor produced a different image/question prefix while "
                "adding the fixed answer suffix"
            )
        return moved

    def _target_score_from_logits(
        self,
        target_text: str,
        labels: Any,
        logits: Any,
        *,
        include_eos: bool,
    ) -> TargetScore:
        token_ids, token_logprobs, target_positions = self._target_tensor_values(
            labels,
            logits,
            include_eos=include_eos,
        )
        return TargetScore(
            target_text=target_text,
            token_ids=tuple(int(value) for value in token_ids.detach().cpu().tolist()),
            token_logprobs=tuple(
                float(value) for value in token_logprobs.detach().float().cpu().tolist()
            ),
            provenance=self.provenance,
            includes_eos=include_eos,
            metadata={
                "reduction": "mean_logprob",
                "target_token_positions": [
                    int(value) for value in target_positions.detach().cpu().tolist()
                ],
            },
        )

    def _target_tensor_values(
        self,
        labels: Any,
        logits: Any,
        *,
        include_eos: bool,
    ) -> tuple[Any, Any, Any]:
        torch = self._torch
        if labels.ndim != 2 or labels.shape[0] != 1:
            raise BackendCompatibilityError(
                "shared backend currently requires target-scoring batch size one"
            )
        target_positions = self._target_positions(labels, include_eos=include_eos)
        prediction_positions = target_positions - 1
        if bool((prediction_positions < 0).any()):
            raise BackendCompatibilityError(
                "a target token has no preceding causal prediction position"
            )
        token_ids = labels[0].index_select(0, target_positions)
        selected_logits = logits[0].index_select(0, prediction_positions).float()
        token_logprobs = torch.log_softmax(selected_logits, dim=-1).gather(
            1, token_ids.unsqueeze(1)
        )[:, 0]
        if not bool(torch.isfinite(token_logprobs).all()):
            raise BackendCompatibilityError(
                "teacher-forced target score contains non-finite values"
            )
        return token_ids, token_logprobs, target_positions

    def _target_positions(self, labels: Any, *, include_eos: bool) -> Any:
        positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
        if not include_eos and len(positions):
            eos_id = self._processor.tokenizer.eos_token_id
            if eos_id is not None and int(labels[0, positions[-1]].item()) == int(eos_id):
                positions = positions[:-1]
        if not len(positions):
            raise BackendCompatibilityError("fixed answer produced no scoreable target tokens")
        return positions

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        moved = {}
        for name, value in inputs.items():
            if not hasattr(value, "to"):
                moved[name] = value
            elif name == "pixel_values":
                moved[name] = value.to(
                    device=self._input_device,
                    dtype=self._torch_dtype,
                )
            else:
                moved[name] = value.to(device=self._input_device)
        return moved

    def _patch_grid_shape(self, image_token_count: int) -> tuple[int, int]:
        base = self._base_model()
        vision_config = base.config.vision_config
        image_size = int(vision_config.image_size)
        patch_size = int(vision_config.patch_size)
        if image_size % patch_size:
            raise BackendCompatibilityError("vision image size is not divisible by its patch size")
        side = image_size // patch_size
        expected = side * side
        if image_token_count != expected:
            raise BackendCompatibilityError(
                f"expected {expected} image tokens for a {side}x{side} patch grid, "
                f"observed {image_token_count}"
            )
        return (side, side)

    def _preprocessing_metadata(
        self,
        *,
        image: Any,
        image_reference: str | None,
        pixel_values: Any,
        patch_grid_shape: tuple[int, int],
    ) -> dict[str, Any]:
        image_processor = self._processor.image_processor
        return {
            "processor_class": type(self._processor).__name__,
            "image_processor_class": type(image_processor).__name__,
            "processor_use_fast": self.processor_use_fast,
            "pixel_values_shape": [int(value) for value in pixel_values.shape],
            "image_size": list(self.spec.image_size),
            "patch_grid_shape": list(patch_grid_shape),
            "image_seq_length": int(self._processor.image_seq_length),
            "resample": str(getattr(image_processor, "resample", None)),
            "image_mean": _json_sequence(getattr(image_processor, "image_mean", None)),
            "image_std": _json_sequence(getattr(image_processor, "image_std", None)),
            "source_file_sha256": (
                _file_sha256(Path(image_reference)) if image_reference is not None else None
            ),
            "rgb_pixels_sha256": _rgb_image_sha256(image),
            "processed_pixel_values_sha256": _tensor_sha256(pixel_values),
        }

    def _validate_loaded_architecture(self) -> None:
        base = self._base_model()
        model_type = getattr(base.config, "model_type", None)
        if model_type != "paligemma":
            raise BackendCompatibilityError(
                f"expected a PaliGemma model, received model_type={model_type!r}"
            )
        vision = base.config.vision_config
        observed_size = (int(vision.image_size), int(vision.image_size))
        if observed_size != self.spec.image_size:
            raise BackendCompatibilityError(
                f"model image size {observed_size} differs from spec {self.spec.image_size}"
            )
        expected_tokens = (int(vision.image_size) // int(vision.patch_size)) ** 2
        processor_tokens = int(self._processor.image_seq_length)
        if processor_tokens != expected_tokens:
            raise BackendCompatibilityError(
                f"processor exposes {processor_tokens} image tokens, "
                f"model geometry requires {expected_tokens}"
            )
        if self.spec.is_adapted:
            self._validate_adapter_compatibility()

    def _validate_adapter_compatibility(self) -> None:
        configurations = getattr(self._model, "peft_config", None)
        if not isinstance(configurations, Mapping) or not configurations:
            raise BackendCompatibilityError("adapted model does not expose a PEFT configuration")
        active_name = getattr(self._model, "active_adapter", None)
        adapter_config = configurations.get(active_name) if isinstance(active_name, str) else None
        if adapter_config is None and len(configurations) == 1:
            adapter_config = next(iter(configurations.values()))
        if adapter_config is None:
            raise BackendCompatibilityError(
                "could not resolve the active PEFT adapter configuration"
            )
        declared_base = getattr(adapter_config, "base_model_name_or_path", None)
        declared_base_matches = not declared_base or (
            str(declared_base).rstrip("/") == self.spec.base_model_id.rstrip("/")
            or (
                Path(str(declared_base)).name == self.spec.base_model_revision
                and "models--google--paligemma-3b-pt-224"
                in Path(str(declared_base)).parts
            )
        )
        if not declared_base_matches:
            raise BackendCompatibilityError(
                "adapter declares a different base model: "
                f"{declared_base!r} != {self.spec.base_model_id!r}"
            )
        declared_revision = getattr(adapter_config, "revision", None)
        if declared_revision and declared_revision != self.spec.base_model_revision:
            raise BackendCompatibilityError(
                "adapter declares a different base model revision: "
                f"{declared_revision!r} != {self.spec.base_model_revision!r}"
            )

    def _vision_layers(self) -> tuple[Any, str]:
        base = self._base_model()
        try:
            layers = base.model.vision_tower.vision_model.encoder.layers
        except AttributeError as exc:
            raise BackendCompatibilityError(
                "pinned PaliGemma vision-layer path is unavailable"
            ) from exc
        return layers, "model.vision_tower.vision_model.encoder"

    def _base_model(self) -> Any:
        model = self._model
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            return get_base_model()
        return model

    def _validate_prepared(self, prepared: PreparedInput) -> _PreparedPayload:
        self._require_loaded()
        if not isinstance(prepared, PreparedInput):
            raise TypeError("prepared must be a PreparedInput")
        if prepared.provenance != self.provenance:
            raise BackendCompatibilityError("prepared input belongs to a different model backend")
        if not isinstance(prepared.payload, _PreparedPayload):
            raise BackendCompatibilityError("prepared input does not contain a PaliGemma payload")
        return prepared.payload

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("call PaliGemmaBackend.load() before model execution")

    def _set_deterministic_seed(self) -> None:
        torch = self._torch
        torch.manual_seed(self.spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.spec.seed)

    def close(self) -> None:
        """Release model references and cached CUDA allocations."""

        torch = self._torch
        self._model = None
        self._processor = None
        self._provenance = None
        self._input_device = None
        self._torch_dtype = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> PaliGemmaBackend:
        return self.load()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _import_gpu_stack() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise BackendDependencyError(
            "PaliGemma execution requires a compatible PyTorch runtime plus "
            "the optional GPU dependencies; use docker/Dockerfile.gpu or "
            "install the project with the 'gpu' extra in a GPU runtime"
        ) from exc
    return torch, transformers, peft


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _resolve_torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc


def _model_input_device(torch: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _image_token_id(processor: Any, model: Any) -> int:
    value = getattr(processor, "image_token_id", None)
    if value is None:
        value = getattr(model.config, "image_token_id", None)
    if value is None:
        value = getattr(model.config, "image_token_index", None)
    if value is None:
        raise BackendCompatibilityError("could not resolve PaliGemma image token ID")
    return int(value)


def _coerce_rgb_image(image: Any) -> tuple[Any, str | None]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise BackendDependencyError("Pillow is required for image preparation") from exc

    image_reference = None
    if isinstance(image, (str, Path)):
        path = Path(image)
        image_reference = str(path)
        with Image.open(path) as opened:
            return opened.convert("RGB").copy(), image_reference
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy(), image_reference
    raise TypeError("image must be a PIL image or filesystem path")


def _clone_tensor_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if hasattr(value, "clone") else value for name, value in values.items()
    }


def _resolve_layer_index(requested: int, count: int, label: str) -> int:
    resolved = requested if requested >= 0 else count + requested
    if resolved < 0 or resolved >= count:
        raise AttributionError(f"{label} layer {requested} is out of range for {count} layers")
    return resolved


def _aggregate_tensor(value: Any, *, dimension: int, mode: str) -> Any:
    if mode == "mean":
        return value.mean(dim=dimension)
    return value.max(dim=dimension).values


def _normalise_attribution_values(
    values: Any,
    *,
    zero_range_policy: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise BackendDependencyError("NumPy is required for attribution") from exc
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise AttributionError(f"expected a two-dimensional patch map, got {array.shape}")
    if not np.isfinite(array).all():
        raise AttributionError("attribution contains non-finite values")
    raw_minimum = float(array.min())
    raw_maximum = float(array.max())
    raw_range = raw_maximum - raw_minimum
    if raw_range <= 1e-12:
        if zero_range_policy == "error":
            raise AttributionError("attribution is constant")
        if zero_range_policy != "zeros":
            raise ValueError("zero_range_policy must be error or zeros")
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = ((array - raw_minimum) / raw_range).astype(
            np.float32,
            copy=False,
        )
    return normalized, {
        "method": "minmax",
        "output_dtype": "float32",
        "zero_range_policy": zero_range_policy,
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_image_sha256(image: Any) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {"mode": image.mode, "size": list(image.size)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sequence(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return str(value)


__all__ = [
    "AttributionError",
    "BACKEND_NAME",
    "BACKEND_VERSION",
    "BackendCompatibilityError",
    "BackendDependencyError",
    "PaliGemmaBackend",
    "SUPPORTED_ATTRIBUTIONS",
]
