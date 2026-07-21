"""Restart-safe 20-item development inference and faithfulness smoke runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import (
    BackendProvenance,
    GenerationResult,
    PaliGemmaBackend,
    ScoreVerification,
    TargetScore,
    VisionLanguageBackend,
)
from .config import config_sha256, load_config, validate_config
from .identifiers import canonical_text, question_text, source_image_id
from .image_cache import materialize_image_cache, verify_image_cache
from .jsonl import iter_jsonl, read_jsonl, write_jsonl_atomic
from .perturbations import (
    apply_patch_intervention,
    build_intervention_plan,
    intervention_plan_sha256,
)
from .provenance import (
    build_run_manifest,
    canonical_json_sha256,
    file_fingerprint,
    file_sha256,
    load_run_manifest,
    write_run_manifest,
)
from .shards import merge_jsonl_shards_atomic, validate_jsonl_shards
from .splits import (
    materialize_grouped_split_artifacts,
    verify_grouped_split_artifacts,
)

SMOKE_RUN_SCHEMA_VERSION = "gi-vqa-development-smoke-run-v1"
PREDICTION_SCHEMA_VERSION = "gi-vqa-smoke-prediction-v1"
ATTRIBUTION_SCHEMA_VERSION = "gi-vqa-smoke-attribution-v1"
PERTURBATION_RESULT_SCHEMA_VERSION = "gi-vqa-smoke-perturbations-v1"
ITEM_COMPLETION_SCHEMA_VERSION = "gi-vqa-smoke-item-completion-v1"
SMOKE_REPORT_SCHEMA_VERSION = "gi-vqa-development-smoke-report-v1"


class SmokeRunError(RuntimeError):
    """Raised when the development smoke contract is incomplete or inconsistent."""


def run_development_smoke(
    *,
    config_path: str | Path,
    project_root: str | Path,
    split_manifest_path: str | Path,
    image_cache_manifest_path: str | Path,
    run_dir: str | Path,
    run_id: str | None = None,
    expected_commit: str | None = None,
    require_clean_git: bool = False,
    materialize_inputs: bool = True,
    max_new_items: int | None = None,
    backend: VisionLanguageBackend | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run or resume the locked development smoke workflow."""

    root = Path(project_root).resolve()
    config_file = _resolve_under(root, config_path)
    split_manifest_file = _resolve_under(root, split_manifest_path)
    cache_manifest_file = _resolve_under(root, image_cache_manifest_path)
    output = Path(run_dir).resolve()
    config = validate_config(
        load_config(config_file),
        require_resolved=True,
        require_model_execution=True,
    )
    _require_smoke_config(config)
    if max_new_items is not None and (
        not isinstance(max_new_items, int)
        or isinstance(max_new_items, bool)
        or max_new_items < 1
    ):
        raise ValueError("max_new_items must be a positive integer")

    git = _git_state(root)
    if expected_commit is not None and git["commit"] != expected_commit:
        raise SmokeRunError(
            f"repository commit {git['commit']!r} does not match "
            f"expected {expected_commit!r}"
        )
    if require_clean_git and git["dirty"]:
        raise SmokeRunError("development smoke requires a clean Git checkout")

    if materialize_inputs:
        materialize_grouped_split_artifacts(
            manifest_path=split_manifest_file,
            project_root=root,
        )
        materialize_image_cache(
            manifest_path=cache_manifest_file,
            project_root=root,
            token=os.getenv("HF_TOKEN"),
        )
    split_check = verify_grouped_split_artifacts(
        manifest_path=split_manifest_file,
        project_root=root,
    )
    cache_check = verify_image_cache(
        manifest_path=cache_manifest_file,
        project_root=root,
    )
    records, smoke_jsonl, cache_manifest = _load_locked_smoke_inputs(
        project_root=root,
        config=config,
        split_manifest_path=split_manifest_file,
        image_cache_manifest_path=cache_manifest_file,
    )

    resolved_run_id = run_id or output.name
    if not resolved_run_id.strip():
        raise ValueError("run_id must be non-empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_inputs = {
        "config_file": config_file,
        "split_manifest": split_manifest_file,
        "image_cache_manifest": cache_manifest_file,
        "smoke_records": smoke_jsonl,
    }
    if manifest_path.exists():
        manifest = load_run_manifest(manifest_path)
        _validate_resumed_manifest(
            manifest,
            run_id=resolved_run_id,
            config=config,
            inputs=manifest_inputs,
            code_revision=git["commit"],
        )
    else:
        unexpected = [path for path in output.iterdir() if path.name != "manifest.json"]
        if unexpected:
            raise SmokeRunError(
                "run directory contains artifacts without a manifest: "
                f"{sorted(path.name for path in unexpected)}"
            )
        manifest = build_run_manifest(
            run_id=resolved_run_id,
            stage="development-smoke-20",
            config=config,
            inputs=manifest_inputs,
            command=command or sys.argv,
            code_revision=git["commit"],
            environment={
                "git": git,
                "hostname": platform.node(),
                "executable": sys.executable,
                "diagnostic_only": True,
                "excluded_from_research_results": True,
            },
        )
        write_run_manifest(manifest_path, manifest)

    owns_backend = backend is None
    model_backend = backend or PaliGemmaBackend.from_config(config)
    try:
        provenance = model_backend.provenance
        _write_or_validate_backend_identity(
            output / "backend.json",
            run_id=resolved_run_id,
            provenance=provenance,
        )
        _write_status(
            output / "status.json",
            status="RUNNING",
            run_id=resolved_run_id,
            completed_items=_count_completed_items(output),
            expected_items=len(records),
        )
        report = _execute_items(
            records=records,
            cache_manifest=cache_manifest,
            project_root=root,
            run_dir=output,
            run_id=resolved_run_id,
            config=config,
            backend=model_backend,
            max_new_items=max_new_items,
            split_check=split_check,
            cache_check=cache_check,
        )
        return report
    except Exception as exc:
        _write_status(
            output / "status.json",
            status="FAIL",
            run_id=resolved_run_id,
            completed_items=_count_completed_items(output),
            expected_items=len(records),
            error={
                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if owns_backend:
            model_backend.close()


def _execute_items(
    *,
    records: Sequence[Mapping[str, Any]],
    cache_manifest: Mapping[str, Any],
    project_root: Path,
    run_dir: Path,
    run_id: str,
    config: Mapping[str, Any],
    backend: VisionLanguageBackend,
    max_new_items: int | None,
    split_check: Mapping[str, Any],
    cache_check: Mapping[str, Any],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    reused_items = 0
    new_items = 0
    for rank, record in enumerate(records):
        item_id = _required_string(record.get("item_id"), "record.item_id")
        item_dir = run_dir / "items" / f"{rank:03d}-{item_id}"
        completion_path = item_dir / "complete.json"
        record_digest = canonical_json_sha256(record)
        if completion_path.is_file():
            completion = _validate_completion(
                completion_path,
                run_id=run_id,
                item_id=item_id,
                record_sha256=record_digest,
                provenance=backend.provenance,
            )
            summaries.append(dict(completion["summary"]))
            reused_items += 1
            continue
        if max_new_items is not None and new_items >= max_new_items:
            continue
        item_dir.mkdir(parents=True, exist_ok=True)
        summary = _execute_item(
            record=record,
            item_dir=item_dir,
            run_id=run_id,
            record_sha256=record_digest,
            cache_manifest=cache_manifest,
            project_root=project_root,
            config=config,
            backend=backend,
        )
        summaries.append(summary)
        new_items += 1
        _write_status(
            run_dir / "status.json",
            status="RUNNING",
            run_id=run_id,
            completed_items=len(summaries),
            expected_items=len(records),
        )

    completed = _count_completed_items(run_dir)
    if completed != len(records):
        result = {
            "schema_version": SMOKE_RUN_SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "run_id": run_id,
            "expected_items": len(records),
            "completed_items": completed,
            "new_items": new_items,
            "reused_items": reused_items,
            "remaining_items": len(records) - completed,
            "diagnostic_only": True,
            "excluded_from_research_results": True,
        }
        _write_status(run_dir / "status.json", **result)
        return result

    ordered_summaries = [
        dict(
            _validate_completion(
                run_dir / "items" / f"{rank:03d}-{record['item_id']}" / "complete.json",
                run_id=run_id,
                item_id=str(record["item_id"]),
                record_sha256=canonical_json_sha256(record),
                provenance=backend.provenance,
            )["summary"]
        )
        for rank, record in enumerate(records)
    ]
    shard_paths = _publish_shards(
        run_dir=run_dir,
        summaries=ordered_summaries,
        shard_count=int(config["execution"]["shard_count"]),
    )
    merged_path = run_dir / "predictions" / "smoke_results.jsonl"
    _publish_or_validate_merged(
        shard_paths=shard_paths,
        destination=merged_path,
        expected=ordered_summaries,
    )
    report_path = run_dir / "metrics" / "smoke_report.json"
    report = {
        "schema_version": SMOKE_REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "run_id": run_id,
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "expected_items": len(records),
        "completed_items": len(ordered_summaries),
        "backend": backend.provenance.as_dict(),
        "input_gates": {
            "split": dict(split_check),
            "image_cache": dict(cache_check),
        },
        "merge": {
            "shards": [file_fingerprint(path) for path in shard_paths],
            "output": file_fingerprint(merged_path),
        },
        "metrics": _aggregate_smoke_metrics(ordered_summaries),
    }
    if report_path.is_file():
        saved_report = _read_json_object(report_path)
        if saved_report != report:
            raise SmokeRunError("existing smoke report differs from reconstructed report")
    else:
        _publish_json_once(report_path, report)
    _write_status(
        run_dir / "status.json",
        status="PASS",
        run_id=run_id,
        completed_items=len(records),
        expected_items=len(records),
        invocation={
            "new_items": new_items,
            "reused_items": reused_items,
        },
        report=file_fingerprint(report_path),
    )
    return {
        **report,
        "invocation": {
            "new_items": new_items,
            "reused_items": reused_items,
        },
    }


def _execute_item(
    *,
    record: Mapping[str, Any],
    item_dir: Path,
    run_id: str,
    record_sha256: str,
    cache_manifest: Mapping[str, Any],
    project_root: Path,
    config: Mapping[str, Any],
    backend: VisionLanguageBackend,
) -> dict[str, Any]:
    item_id = _required_string(record.get("item_id"), "record.item_id")
    source_id = source_image_id(record)
    question = question_text(record)
    reference = _answer_text(record)
    image_descriptor = _image_descriptor(cache_manifest, source_id)
    if image_descriptor.get("scope") != "development_smoke":
        raise SmokeRunError(f"{source_id} is not locked for development smoke")
    image_path = _resolve_under(project_root, image_descriptor["path"])
    if file_sha256(image_path) != image_descriptor["sha256"]:
        raise SmokeRunError(f"cached image hash changed for {source_id}")
    prepared = backend.prepare(
        item_id=item_id,
        image=image_path,
        question=question,
    )
    prediction_path = item_dir / "prediction.json"
    if prediction_path.is_file():
        prediction = _validate_prediction(
            prediction_path,
            run_id=run_id,
            item_id=item_id,
            record_sha256=record_sha256,
            provenance=backend.provenance,
            image_sha256=str(image_descriptor["sha256"]),
        )
        generation = _generation_from_dict(
            prediction["generation"],
            backend.provenance,
        )
    else:
        generation = backend.generate(prepared)
        target_score = backend.score_target(
            prepared,
            generation.text,
            expected_token_ids=generation.token_ids,
        )
        verification = backend.verify_generation_score(generation, target_score)
        prediction = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "run_id": run_id,
            "item_id": item_id,
            "record_sha256": record_sha256,
            "source_img_id": source_id,
            "question": question,
            "reference_answer": reference,
            "image": file_fingerprint(image_path),
            "prepared": {
                "prompt": prepared.prompt,
                "input_token_ids": list(prepared.input_token_ids),
                "image_token_indices": list(prepared.image_token_indices),
                "preprocessing": dict(prepared.preprocessing),
            },
            "generation": _generation_to_dict(generation),
            "target_score": _target_score_to_dict(target_score),
            "score_verification": verification.as_dict(),
            "backend": backend.provenance.as_dict(),
        }
        _publish_json_once(prediction_path, prediction)
    original_score = float(prediction["target_score"]["mean_token_logprob"])

    attribution_files: dict[str, dict[str, Any]] = {}
    perturbation_files: dict[str, dict[str, Any]] = {}
    all_interventions: list[dict[str, Any]] = []
    for method in config["attribution"]["methods"]:
        attribution_path = item_dir / "attributions" / f"{method}.npz"
        if attribution_path.is_file():
            values, attribution_metadata = _load_attribution_npz(
                attribution_path,
                run_id=run_id,
                item_id=item_id,
                method=method,
                prediction_sha256=file_sha256(prediction_path),
                provenance=backend.provenance,
            )
        else:
            attribution = backend.attribute(
                prepared,
                generation,
                method=method,
            )
            if attribution.method != method:
                raise SmokeRunError(
                    f"backend returned attribution method {attribution.method!r}; "
                    f"expected {method!r}"
                )
            if attribution.provenance != backend.provenance:
                raise SmokeRunError("attribution belongs to a different backend")
            attribution_metadata = {
                "schema_version": ATTRIBUTION_SCHEMA_VERSION,
                "run_id": run_id,
                "item_id": item_id,
                "method": method,
                "prediction_sha256": file_sha256(prediction_path),
                "patch_grid_shape": list(attribution.patch_grid_shape),
                "image_token_indices": list(attribution.image_token_indices),
                "aggregation": attribution.aggregation,
                "target_score": _target_score_to_dict(attribution.target_score),
                "metadata": dict(attribution.metadata),
                "backend": backend.provenance.as_dict(),
            }
            values = attribution.values
            _publish_attribution_npz(
                attribution_path,
                values=values,
                metadata=attribution_metadata,
            )
        attribution_files[method] = file_fingerprint(attribution_path)

        plan = build_intervention_plan(
            values,
            item_id=item_id,
            method=method,
            seed=int(config["seed"]),
            config=config["perturbation"],
        )
        perturbation_path = item_dir / "perturbations" / f"{method}.json"
        if perturbation_path.is_file():
            perturbation = _validate_perturbation_result(
                perturbation_path,
                run_id=run_id,
                item_id=item_id,
                method=method,
                attribution_sha256=file_sha256(attribution_path),
                plan=plan,
                provenance=backend.provenance,
            )
        else:
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                raise SmokeRunError("Pillow is required for perturbations") from exc
            with Image.open(image_path) as opened:
                source_image = opened.convert("RGB").copy()
            results = []
            for intervention in plan:
                perturbed = apply_patch_intervention(source_image, intervention)
                perturbed_input = backend.prepare(
                    item_id=f"{item_id}:{intervention.intervention_id}",
                    image=perturbed,
                    question=question,
                )
                score = backend.score_target(
                    perturbed_input,
                    generation.text,
                    expected_token_ids=generation.token_ids,
                )
                score_value = float(score.mean_token_logprob)
                if not math.isfinite(score_value):
                    raise SmokeRunError(
                        f"non-finite intervention score: {intervention.intervention_id}"
                    )
                results.append(
                    {
                        **intervention.as_dict(),
                        "target_score": _target_score_to_dict(score),
                        "score_minus_original": score_value - original_score,
                        "processed_pixel_values_sha256": perturbed_input.preprocessing.get(
                            "processed_pixel_values_sha256"
                        ),
                        "rgb_pixels_sha256": perturbed_input.preprocessing.get(
                            "rgb_pixels_sha256"
                        ),
                    }
                )
            perturbation = {
                "schema_version": PERTURBATION_RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "item_id": item_id,
                "method": method,
                "attribution_sha256": file_sha256(attribution_path),
                "plan_sha256": intervention_plan_sha256(plan),
                "original_mean_token_logprob": original_score,
                "fixed_target_text": generation.text,
                "fixed_target_token_ids": list(generation.token_ids),
                "interventions": results,
                "backend": backend.provenance.as_dict(),
            }
            _publish_json_once(perturbation_path, perturbation)
        perturbation_files[method] = file_fingerprint(perturbation_path)
        all_interventions.extend(
            {"method": method, **dict(value)}
            for value in perturbation["interventions"]
        )

    metadata = record.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    score_verification = _required_mapping(
        prediction.get("score_verification"),
        "prediction score verification",
    )
    summary = {
        "item_id": item_id,
        "source_img_id": source_id,
        "partition": metadata.get("partition"),
        "complexity": metadata.get("complexity"),
        "question_class": metadata.get("question_class", []),
        "question": question,
        "reference_answer": reference,
        "prediction": generation.text,
        "normalized_exact_match": (
            canonical_text(generation.text, casefold=True)
            == canonical_text(reference, casefold=True)
        ),
        "generated_token_count": len(generation.token_ids),
        "mean_token_logprob": generation.mean_token_logprob,
        "sequence_confidence": generation.sequence_confidence,
        "fixed_answer_mean_token_logprob": original_score,
        "generation_teacher_forcing_score_parity": {
            "absolute_tolerance": float(score_verification["absolute_tolerance"]),
            "maximum_absolute_difference": float(
                score_verification["maximum_absolute_difference"]
            ),
            "mean_absolute_difference": float(
                score_verification["mean_absolute_difference"]
            ),
        },
        "interventions": all_interventions,
    }
    completion = {
        "schema_version": ITEM_COMPLETION_SCHEMA_VERSION,
        "run_id": run_id,
        "item_id": item_id,
        "record_sha256": record_sha256,
        "backend": backend.provenance.as_dict(),
        "artifacts": {
            "prediction": file_fingerprint(prediction_path),
            "attributions": attribution_files,
            "perturbations": perturbation_files,
        },
        "summary": summary,
    }
    _publish_json_once(item_dir / "complete.json", completion)
    return summary


def _load_locked_smoke_inputs(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    split_manifest_path: Path,
    image_cache_manifest_path: Path,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    split_manifest = _read_json_object(split_manifest_path)
    smoke = _required_mapping(split_manifest.get("smoke"), "split manifest smoke")
    artifact = _required_mapping(
        _required_mapping(split_manifest.get("artifacts"), "split artifacts").get(
            "smoke_20"
        ),
        "smoke_20 artifact",
    )
    smoke_jsonl = _resolve_under(project_root, artifact["path"])
    if file_sha256(smoke_jsonl) != artifact["sha256"]:
        raise SmokeRunError("smoke JSONL differs from the grouped split manifest")
    records = list(iter_jsonl(smoke_jsonl))
    expected_count = int(config["execution"]["max_items"])
    if expected_count != 20 or len(records) != expected_count:
        raise SmokeRunError("the first development smoke must contain exactly 20 items")
    item_ids = [_required_string(record.get("item_id"), "record.item_id") for record in records]
    source_ids = [source_image_id(record) for record in records]
    if item_ids != list(smoke["item_ids"]):
        raise SmokeRunError("smoke record order differs from locked item IDs")
    if source_ids != list(smoke["source_img_ids"]):
        raise SmokeRunError("smoke record order differs from locked source IDs")
    if len(set(item_ids)) != len(item_ids) or len(set(source_ids)) != len(source_ids):
        raise SmokeRunError("smoke items and sources must be unique")
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("partition") != "development":
            raise SmokeRunError("smoke record is not in the development partition")
    cache_manifest = _read_json_object(image_cache_manifest_path)
    cached_sources = set(
        _required_mapping(cache_manifest.get("images"), "cache images")
    )
    if not set(source_ids) <= cached_sources:
        raise SmokeRunError("image cache does not contain every smoke source")
    return records, smoke_jsonl, cache_manifest


def _require_smoke_config(config: Mapping[str, Any]) -> None:
    if config.get("profile") != "smoke":
        raise SmokeRunError("development smoke runner requires profile=smoke")
    if config["execution"].get("evaluation_partition") != "development":
        raise SmokeRunError("development smoke runner cannot access another partition")
    if config["execution"].get("max_items") != 20:
        raise SmokeRunError("development smoke runner requires max_items=20")
    if config["execution"].get("shard_count") != 1:
        raise SmokeRunError("the first development smoke requires shard_count=1")
    if config["model"].get("condition") != "base":
        raise SmokeRunError("first development smoke requires the base condition")
    if config["model"].get("adapter") is not None:
        raise SmokeRunError("first development smoke must not load an adapter")
    expected_perturbation = {
        "patch_fractions": [0.25],
        "deletion_treatments": ["gray", "blur"],
        "insertion_treatments": ["blur"],
        "selection_modes": ["most_salient", "least_salient", "random"],
        "random_repeats": 1,
        "gray_value": 128,
        "blur_radius": 12.0,
    }
    if config.get("perturbation") != expected_perturbation:
        raise SmokeRunError(
            "first development smoke requires the locked perturbation plan"
        )


def _write_or_validate_backend_identity(
    path: Path,
    *,
    run_id: str,
    provenance: BackendProvenance,
) -> None:
    expected = {
        "schema_version": "gi-vqa-smoke-backend-identity-v1",
        "run_id": run_id,
        "backend": provenance.as_dict(),
    }
    if path.is_file():
        if _read_json_object(path) != expected:
            raise SmokeRunError("loaded backend differs from the saved run identity")
    else:
        _publish_json_once(path, expected)


def _validate_resumed_manifest(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, Path],
    code_revision: str | None,
) -> None:
    if manifest.get("run_id") != run_id:
        raise SmokeRunError("run ID differs from the existing manifest")
    if manifest.get("stage") != "development-smoke-20":
        raise SmokeRunError("existing manifest belongs to another stage")
    if manifest.get("config_sha256") != config_sha256(config):
        raise SmokeRunError("configuration differs from the existing run manifest")
    if manifest.get("code_revision") != code_revision:
        raise SmokeRunError("Git commit differs from the existing run manifest")
    saved_inputs = _required_mapping(manifest.get("inputs"), "manifest inputs")
    for name, path in inputs.items():
        saved = _required_mapping(saved_inputs.get(name), f"manifest input {name}")
        if saved.get("sha256") != file_sha256(path):
            raise SmokeRunError(f"run input changed since manifest creation: {name}")


def _validate_prediction(
    path: Path,
    *,
    run_id: str,
    item_id: str,
    record_sha256: str,
    provenance: BackendProvenance,
    image_sha256: str,
) -> dict[str, Any]:
    value = _read_json_object(path)
    checks = (
        (value.get("schema_version"), PREDICTION_SCHEMA_VERSION, "schema"),
        (value.get("run_id"), run_id, "run ID"),
        (value.get("item_id"), item_id, "item ID"),
        (value.get("record_sha256"), record_sha256, "record hash"),
        (value.get("backend"), provenance.as_dict(), "backend"),
        (value.get("image", {}).get("sha256"), image_sha256, "image hash"),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise SmokeRunError(f"saved prediction {label} changed for {item_id}")
    generation = value.get("generation")
    target = value.get("target_score")
    if not isinstance(generation, Mapping) or not isinstance(target, Mapping):
        raise SmokeRunError(f"saved prediction payload is incomplete for {item_id}")
    generated = _generation_from_dict(generation, provenance)
    scored = _target_score_from_dict(target, provenance)
    if generated.text != scored.target_text or generated.token_ids != scored.token_ids:
        raise SmokeRunError(
            f"saved prediction target differs from generated answer for {item_id}"
        )
    verification = value.get("score_verification")
    if not isinstance(verification, Mapping):
        raise SmokeRunError(f"saved score verification is missing for {item_id}")
    try:
        ScoreVerification(
            token_count=int(verification["token_count"]),
            absolute_tolerance=float(verification["absolute_tolerance"]),
            maximum_absolute_difference=float(
                verification["maximum_absolute_difference"]
            ),
            mean_absolute_difference=float(verification["mean_absolute_difference"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SmokeRunError(
            f"saved score verification is invalid for {item_id}"
        ) from exc
    return value


def _load_attribution_npz(
    path: Path,
    *,
    run_id: str,
    item_id: str,
    method: str,
    prediction_sha256: str,
    provenance: BackendProvenance,
) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SmokeRunError("NumPy is required for attribution storage") from exc
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float32)
            metadata = json.loads(str(archive["metadata_json"].item()))
    except Exception as exc:
        raise SmokeRunError(f"invalid attribution archive: {path}") from exc
    expected = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "run_id": run_id,
        "item_id": item_id,
        "method": method,
        "prediction_sha256": prediction_sha256,
        "backend": provenance.as_dict(),
    }
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise SmokeRunError(f"saved attribution {field} changed for {item_id}")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SmokeRunError(f"saved attribution values are invalid for {item_id}")
    if list(values.shape) != metadata.get("patch_grid_shape"):
        raise SmokeRunError(f"saved attribution shape changed for {item_id}")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6 or maximum - minimum <= 1e-8:
        raise SmokeRunError(
            f"saved attribution is not a non-degenerate min-max map for {item_id}"
        )
    target = metadata.get("target_score")
    if not isinstance(target, Mapping):
        raise SmokeRunError(f"saved attribution target score is missing for {item_id}")
    _target_score_from_dict(target, provenance)
    return values, metadata


def _validate_perturbation_result(
    path: Path,
    *,
    run_id: str,
    item_id: str,
    method: str,
    attribution_sha256: str,
    plan: Sequence[Any],
    provenance: BackendProvenance,
) -> dict[str, Any]:
    value = _read_json_object(path)
    expected = {
        "schema_version": PERTURBATION_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "item_id": item_id,
        "method": method,
        "attribution_sha256": attribution_sha256,
        "plan_sha256": intervention_plan_sha256(plan),
        "backend": provenance.as_dict(),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise SmokeRunError(f"saved perturbation {field} changed for {item_id}")
    interventions = value.get("interventions")
    if not isinstance(interventions, list) or len(interventions) != len(plan):
        raise SmokeRunError(f"saved perturbations are empty for {item_id}")
    for result, expected_intervention in zip(interventions, plan):  # noqa: B905
        expected = expected_intervention.as_dict()
        for field, expected_value in expected.items():
            if result.get(field) != expected_value:
                raise SmokeRunError(
                    f"saved perturbation plan field {field} changed for {item_id}"
                )
        score = result.get("target_score", {}).get("mean_token_logprob")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise SmokeRunError(f"saved perturbation score is invalid for {item_id}")
        target = _target_score_from_dict(result["target_score"], provenance)
        if list(target.token_ids) != value.get("fixed_target_token_ids"):
            raise SmokeRunError(
                f"saved perturbation target token IDs changed for {item_id}"
            )
    return value


def _validate_completion(
    path: Path,
    *,
    run_id: str,
    item_id: str,
    record_sha256: str,
    provenance: BackendProvenance,
) -> dict[str, Any]:
    value = _read_json_object(path)
    expected = {
        "schema_version": ITEM_COMPLETION_SCHEMA_VERSION,
        "run_id": run_id,
        "item_id": item_id,
        "record_sha256": record_sha256,
        "backend": provenance.as_dict(),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise SmokeRunError(f"completion {field} changed for {item_id}")
    artifacts = _required_mapping(value.get("artifacts"), "completion artifacts")
    fingerprints: list[Mapping[str, Any]] = [
        _required_mapping(artifacts.get("prediction"), "prediction fingerprint")
    ]
    for group in ("attributions", "perturbations"):
        for name, fingerprint in _required_mapping(
            artifacts.get(group), group
        ).items():
            fingerprints.append(
                _required_mapping(fingerprint, f"{group} fingerprint {name}")
            )
    for fingerprint in fingerprints:
        artifact_path = Path(_required_string(fingerprint.get("path"), "artifact path"))
        if not artifact_path.is_file() or file_sha256(artifact_path) != fingerprint.get(
            "sha256"
        ):
            raise SmokeRunError(f"completed item artifact changed: {artifact_path}")
    if not isinstance(value.get("summary"), Mapping):
        raise SmokeRunError(f"completion summary is missing for {item_id}")
    return value


def _publish_shards(
    *,
    run_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    shard_count: int,
) -> list[Path]:
    if shard_count < 1 or shard_count > len(summaries):
        raise SmokeRunError("shard_count must be between 1 and the item count")
    paths = []
    for shard_index in range(shard_count):
        start = len(summaries) * shard_index // shard_count
        end = len(summaries) * (shard_index + 1) // shard_count
        rows = [dict(row) for row in summaries[start:end]]
        path = run_dir / "shards" / f"shard-{shard_index:03d}.jsonl"
        if path.is_file():
            if read_jsonl(path) != rows:
                raise SmokeRunError(f"existing shard differs from completed items: {path}")
        else:
            write_jsonl_atomic(path, rows)
        validate_jsonl_shards(
            [path],
            id_field="item_id",
            expected_ids=[str(row["item_id"]) for row in rows],
        )
        paths.append(path)
    return paths


def _publish_or_validate_merged(
    *,
    shard_paths: Sequence[Path],
    destination: Path,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    expected_ids = [str(row["item_id"]) for row in expected]
    if destination.is_file():
        validate_jsonl_shards(
            [destination],
            id_field="item_id",
            expected_ids=expected_ids,
        )
        if read_jsonl(destination) != [dict(row) for row in expected]:
            raise SmokeRunError("merged smoke results differ from item completions")
        return
    merge_jsonl_shards_atomic(
        shard_paths,
        destination,
        id_field="item_id",
        expected_ids=expected_ids,
    )
    if read_jsonl(destination) != [dict(row) for row in expected]:
        raise SmokeRunError("merged smoke result order differs from locked items")


def _aggregate_smoke_metrics(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [bool(row["normalized_exact_match"]) for row in summaries]
    confidences = [float(row["sequence_confidence"]) for row in summaries]
    logprobs = [float(row["fixed_answer_mean_token_logprob"]) for row in summaries]
    parity = [
        _required_mapping(
            row["generation_teacher_forcing_score_parity"],
            "summary generation/teacher-forcing parity",
        )
        for row in summaries
    ]
    parity_tolerances = {
        float(value["absolute_tolerance"])
        for value in parity
    }
    if len(parity_tolerances) != 1:
        raise SmokeRunError("items used different generation-score tolerances")
    parity_maxima = [
        float(value["maximum_absolute_difference"])
        for value in parity
    ]
    interventions = [
        intervention
        for row in summaries
        for intervention in row["interventions"]
    ]
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for value in interventions:
        key = (
            value["method"],
            value["operation"],
            value["treatment"],
            value["selection"],
            float(value["fraction"]),
        )
        grouped.setdefault(key, []).append(float(value["score_minus_original"]))
    effects = [
        {
            "method": key[0],
            "operation": key[1],
            "treatment": key[2],
            "selection": key[3],
            "fraction": key[4],
            "items": len(values),
            "mean_score_minus_original": sum(values) / len(values),
        }
        for key, values in sorted(grouped.items())
    ]
    return {
        "items": len(summaries),
        "normalized_exact_match": sum(exact) / len(exact),
        "mean_sequence_confidence": sum(confidences) / len(confidences),
        "mean_fixed_answer_logprob": sum(logprobs) / len(logprobs),
        "generation_teacher_forcing_score_parity": {
            "absolute_tolerance": next(iter(parity_tolerances)),
            "maximum_absolute_difference": max(parity_maxima),
            "mean_item_maximum_absolute_difference": (
                sum(parity_maxima) / len(parity_maxima)
            ),
        },
        "intervention_scores": len(interventions),
        "finite_intervention_scores": all(
            math.isfinite(float(value["target_score"]["mean_token_logprob"]))
            for value in interventions
        ),
        "mean_intervention_effects": effects,
    }


def _generation_to_dict(value: GenerationResult) -> dict[str, Any]:
    return {
        "text": value.text,
        "token_ids": list(value.token_ids),
        "token_logprobs": list(value.token_logprobs),
        "mean_token_logprob": value.mean_token_logprob,
        "sequence_confidence": value.sequence_confidence,
        "finish_reason": value.finish_reason,
        "metadata": dict(value.metadata),
    }


def _generation_from_dict(
    value: Mapping[str, Any],
    provenance: BackendProvenance,
) -> GenerationResult:
    return GenerationResult(
        text=str(value["text"]),
        token_ids=tuple(int(item) for item in value["token_ids"]),
        token_logprobs=tuple(float(item) for item in value["token_logprobs"]),
        provenance=provenance,
        finish_reason=value.get("finish_reason"),
        metadata=value.get("metadata", {}),
    )


def _target_score_to_dict(value: TargetScore) -> dict[str, Any]:
    return {
        "target_text": value.target_text,
        "token_ids": list(value.token_ids),
        "token_logprobs": list(value.token_logprobs),
        "mean_token_logprob": value.mean_token_logprob,
        "total_logprob": value.total_logprob,
        "includes_eos": value.includes_eos,
        "metadata": dict(value.metadata),
    }


def _target_score_from_dict(
    value: Mapping[str, Any],
    provenance: BackendProvenance,
) -> TargetScore:
    return TargetScore(
        target_text=str(value["target_text"]),
        token_ids=tuple(int(item) for item in value["token_ids"]),
        token_logprobs=tuple(float(item) for item in value["token_logprobs"]),
        provenance=provenance,
        includes_eos=bool(value["includes_eos"]),
        metadata=value.get("metadata", {}),
    )


def _publish_attribution_npz(
    path: Path,
    *,
    values: Any,
    metadata: Mapping[str, Any],
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SmokeRunError("NumPy is required for attribution storage") from exc
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise SmokeRunError("attribution values must be a finite 2D array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                values=array,
                metadata_json=np.asarray(
                    json.dumps(
                        dict(metadata),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        _link_no_overwrite(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _link_no_overwrite(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _link_no_overwrite(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise SmokeRunError(f"refusing to overwrite stage artifact: {destination}") from exc
    _fsync_directory(destination.parent)


def _write_status(path: Path, **value: Any) -> None:
    payload = {
        **value,
        "schema_version": "gi-vqa-smoke-status-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _count_completed_items(run_dir: Path) -> int:
    items = run_dir / "items"
    return len(list(items.glob("*/complete.json"))) if items.is_dir() else 0


def _answer_text(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, str) and canonical_text(answer):
        return canonical_text(answer)
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
                and canonical_text(message["content"])
            ):
                return canonical_text(message["content"])
    raise SmokeRunError("smoke record has no reference answer")


def _image_descriptor(
    cache_manifest: Mapping[str, Any],
    source_id: str,
) -> Mapping[str, Any]:
    images = _required_mapping(cache_manifest.get("images"), "cache images")
    return _required_mapping(images.get(source_id), f"cache image {source_id}")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SmokeRunError(f"JSON root must be an object: {path}")
    return value


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeRunError(f"{name} must be a mapping")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeRunError(f"{name} must be a non-empty string")
    return value


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SmokeRunError(f"path escapes project root: {value}") from exc
    return resolved


def _git_state(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return {"commit": None, "dirty": None, "status_sha256": None}
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": completed.stdout.strip(),
        "dirty": bool(status.strip()),
        "status_sha256": canonical_json_sha256(status.splitlines()),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gi-vqa-smoke",
        description="Run or resume the locked 20-item development smoke gate.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("protocols/study1/grouped_split_manifest.json"),
    )
    parser.add_argument(
        "--image-cache-manifest",
        type=Path,
        default=Path(
            "protocols/study1/smoke_training_image_cache_manifest.json"
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--max-new-items", type=int)
    parser.add_argument("--no-materialize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_development_smoke(
        config_path=args.config,
        project_root=args.project_root,
        split_manifest_path=args.split_manifest,
        image_cache_manifest_path=args.image_cache_manifest,
        run_dir=args.run_dir,
        run_id=args.run_id,
        expected_commit=args.expected_commit,
        require_clean_git=args.require_clean_git,
        materialize_inputs=not args.no_materialize,
        max_new_items=args.max_new_items,
        command=[sys.executable, "-m", "gi_vqa.smoke_runner", *(argv or sys.argv[1:])],
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SMOKE_REPORT_SCHEMA_VERSION",
    "SMOKE_RUN_SCHEMA_VERSION",
    "SmokeRunError",
    "run_development_smoke",
]
