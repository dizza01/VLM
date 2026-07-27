"""Restart-safe faithfulness evaluation for the locked 64-item subset."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import PaliGemmaBackend, VisionLanguageBackend
from .config import load_config, validate_config
from .controlled_evaluation_runner import _validate_training_bundle
from .larger_development import validate_locked_protocol
from .larger_development_runner import (
    _config_for_condition,
    _git,
    _object,
    _publish_or_match,
    _selected_records,
    _under,
)
from .perturbations import apply_patch_intervention, build_intervention_plan
from .provenance import canonical_json_sha256, file_sha256
from .training_gate import _validate_runtime

SCHEMA_VERSION = "gi-vqa-larger-development-faithfulness-v1"
ITEM_SCHEMA_VERSION = "gi-vqa-larger-development-faithfulness-item-v1"
CONDITIONS = ("paired_correct", "constant_control")
BackendFactory = Callable[[Mapping[str, Any]], VisionLanguageBackend]


class LargerDevelopmentFaithfulnessError(RuntimeError):
    """Raised when the locked faithfulness evaluation cannot safely continue."""


def run_larger_development_faithfulness(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    training_bundle_dir: str | Path,
    inference_run_dir: str | Path,
    run_dir: str | Path | None = None,
    config_path: str | Path = "configs/study1/smoke.yaml",
    expected_commit: str | None = None,
    require_clean_git: bool = False,
    required_gpu_substring: str = "T4",
    backend_factory: BackendFactory | None = None,
    max_new_items: int | None = None,
) -> dict[str, Any]:
    """Run or resume the preregistered two-condition, 64-item stage."""

    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    bundle = _under(root, training_bundle_dir)
    inference = Path(inference_run_dir).resolve()
    output = (
        Path(run_dir).resolve()
        if run_dir is not None
        else inference / "faithfulness"
    )
    lock = validate_locked_protocol(project_root=root, protocol_path=protocol_file)
    protocol = _object(protocol_file)
    _validate_faithfulness_lock(protocol)
    git = _git(root)
    if expected_commit is not None and git["commit"] != expected_commit:
        raise LargerDevelopmentFaithfulnessError(
            "repository commit differs from expected"
        )
    if require_clean_git and git["dirty"]:
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness evaluation requires a clean checkout"
        )
    if max_new_items is not None and (
        isinstance(max_new_items, bool) or max_new_items < 1
    ):
        raise ValueError("max_new_items must be a positive integer")

    runtime: dict[str, Any] = {"checks": {}}
    if backend_factory is None:
        _validate_runtime(runtime, required_gpu_substring=required_gpu_substring)
    else:
        runtime["runtime"] = {"test_backend": True}

    receipt_path = _under(
        root, protocol["inputs"]["controlled_training_pass"]["path"]
    )
    receipt = _object(receipt_path)
    _validate_training_bundle(bundle, receipt)
    inference_status_path = inference / "inference_status.json"
    inference_status = _object(inference_status_path)
    if inference_status.get("status") != "INFERENCE_COMPLETE":
        raise LargerDevelopmentFaithfulnessError(
            "larger-development inference is not complete"
        )
    if inference_status.get("test_partition_accessed") is not False:
        raise LargerDevelopmentFaithfulnessError(
            "inference status does not preserve the test-set seal"
        )
    identity = _object(inference / "identity.json")
    if identity.get("protocol_sha256") != file_sha256(protocol_file):
        raise LargerDevelopmentFaithfulnessError(
            "inference was produced under a different protocol"
        )
    if identity.get("selection_sha256") != lock["selection_sha256"]:
        raise LargerDevelopmentFaithfulnessError(
            "inference selection differs from the locked selection"
        )

    selection = _object(
        _under(root, protocol["inputs"]["selection_manifest"]["path"])
    )
    records = _selected_records(root, protocol, selection)
    faithfulness_ids = selection["selection"]["faithfulness_item_ids"]
    if len(faithfulness_ids) != 64:
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness subset must contain exactly 64 items"
        )
    if faithfulness_ids != [record["item_id"] for record in records[:64]]:
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness subset is not the first 64 locked ranks"
        )

    constant_image = bundle / "prepared_data/constant_image.png"
    manifest = _object(bundle / "bundle_manifest.json")
    expected_constant_hash = manifest.get("members_sha256", {}).get(
        "prepared_data/constant_image.png"
    )
    if (
        not constant_image.is_file()
        or file_sha256(constant_image) != expected_constant_hash
    ):
        raise LargerDevelopmentFaithfulnessError(
            "training neutral image is missing or changed"
        )
    image_cache = root / "data/larger_development_images"
    images = {
        row["source_img_id"]: image_cache / f"{row['source_img_id']}.jpg"
        for row in selection["records"][:64]
    }
    if any(not path.is_file() for path in images.values()):
        raise LargerDevelopmentFaithfulnessError(
            "selected development images are not materialized"
        )

    config = validate_config(
        load_config(_under(root, config_path)),
        require_resolved=True,
        require_model_execution=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": file_sha256(protocol_file),
        "selection_sha256": lock["selection_sha256"],
        "inference_identity_sha256": file_sha256(inference / "identity.json"),
        "inference_status_sha256": file_sha256(inference_status_path),
        "training_receipt_sha256": file_sha256(receipt_path),
        "repository_commit": git["commit"],
        "conditions": list(CONDITIONS),
        "ordered_item_ids_sha256": canonical_json_sha256(faithfulness_ids),
    }
    _publish_or_match(output / "identity.json", run_identity)

    factory = backend_factory or (lambda value: PaliGemmaBackend.from_config(value))
    condition_reports: dict[str, Any] = {}
    remaining = max_new_items
    for condition in CONDITIONS:
        condition_config = _config_for_condition(
            config, condition=condition, bundle=bundle, receipt=receipt
        )
        backend = factory(condition_config)
        try:
            report = _run_condition(
                condition=condition,
                records=records[:64],
                selection=selection,
                images=images,
                constant_image=constant_image,
                inference=inference,
                output=output,
                backend=backend,
                config=config,
                max_new_items=remaining,
            )
        finally:
            backend.close()
        condition_reports[condition] = report
        if remaining is not None:
            remaining = max(0, remaining - report["new_items"])

    completed = sum(value["completed_items"] for value in condition_reports.values())
    expected = 64 * len(CONDITIONS)
    status = "PASS" if completed == expected else "INCOMPLETE"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "test_partition_accessed": False,
        "expected_item_conditions": expected,
        "completed_item_conditions": completed,
        "identity": run_identity,
        "repository": git,
        "runtime": runtime["runtime"],
        "conditions": condition_reports,
        "comparison": (
            _summarize(output) if status == "PASS" else None
        ),
    }
    _write_replace(output / "faithfulness_report.json", report)
    return report


def _run_condition(
    *,
    condition: str,
    records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    images: Mapping[str, Path],
    constant_image: Path,
    inference: Path,
    output: Path,
    backend: VisionLanguageBackend,
    config: Mapping[str, Any],
    max_new_items: int | None,
) -> dict[str, Any]:
    rows = {row["item_id"]: row for row in selection["records"]}
    new_items = reused_items = 0
    for rank, record in enumerate(records):
        item_id = str(record["item_id"])
        row = rows[item_id]
        image = (
            constant_image
            if condition == "constant_control"
            else images[row["source_img_id"]]
        )
        inference_path = inference / condition / "items" / f"{rank:03d}-{item_id}.json"
        saved_prediction = _object(inference_path)
        expected = {
            "schema_version": ITEM_SCHEMA_VERSION,
            "condition": condition,
            "rank": rank,
            "item_id": item_id,
            "record_sha256": row["record_sha256"],
            "image_sha256": file_sha256(image),
            "inference_item_sha256": file_sha256(inference_path),
            "backend": backend.provenance.as_dict(),
        }
        destination = output / condition / "items" / f"{rank:03d}-{item_id}.json"
        if destination.is_file():
            completed = _object(destination)
            if any(completed.get(key) != value for key, value in expected.items()):
                raise LargerDevelopmentFaithfulnessError(
                    f"stale faithfulness completion: {destination}"
                )
            reused_items += 1
            continue
        if max_new_items is not None and new_items >= max_new_items:
            continue

        question = _question(record)
        prepared = backend.prepare(item_id=item_id, image=image, question=question)
        generation = backend.generate(prepared)
        _require_reproduction(saved_prediction, generation)
        original = backend.score_target(
            prepared,
            generation.text,
            expected_token_ids=generation.token_ids,
        )
        parity = backend.verify_generation_score(generation, original)
        methods: dict[str, Any] = {}
        for method in config["attribution"]["methods"]:
            attribution = backend.attribute(prepared, generation, method=method)
            if attribution.method != method:
                raise LargerDevelopmentFaithfulnessError(
                    f"backend returned the wrong attribution method: {method}"
                )
            values = _finite_matrix(attribution.values)
            plan = build_intervention_plan(
                values,
                item_id=f"{condition}:{item_id}",
                method=method,
                seed=int(config["seed"]),
                config=config["perturbation"],
            )
            interventions = []
            from PIL import Image

            with Image.open(image) as opened:
                source = opened.convert("RGB").copy()
            for intervention in plan:
                perturbed = apply_patch_intervention(source, intervention)
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
                interventions.append(
                    {
                        **intervention.as_dict(),
                        "mean_token_logprob": score.mean_token_logprob,
                        "score_minus_original": (
                            score.mean_token_logprob - original.mean_token_logprob
                        ),
                    }
                )
            methods[method] = {
                "patch_grid_shape": list(attribution.patch_grid_shape),
                "values": values,
                "aggregation": attribution.aggregation,
                "metadata": dict(attribution.metadata),
                "interventions": interventions,
            }
        result = {
            **expected,
            "source_img_id": row["source_img_id"],
            "fixed_target_text": generation.text,
            "fixed_target_token_ids": list(generation.token_ids),
            "original_mean_token_logprob": original.mean_token_logprob,
            "score_verification": parity.as_dict(),
            "methods": methods,
        }
        _publish_or_match(destination, result)
        new_items += 1
    item_dir = output / condition / "items"
    completed = len(list(item_dir.glob("*.json"))) if item_dir.exists() else 0
    return {
        "status": "COMPLETE" if completed == len(records) else "INCOMPLETE",
        "expected_items": len(records),
        "completed_items": completed,
        "new_items": new_items,
        "reused_items": reused_items,
        "backend": backend.provenance.as_dict(),
    }


def _require_reproduction(saved: Mapping[str, Any], generation: Any) -> None:
    if generation.text != saved.get("prediction"):
        raise LargerDevelopmentFaithfulnessError(
            "deterministic generation did not reproduce the saved answer"
        )
    if len(generation.token_ids) != saved.get("generated_token_count"):
        raise LargerDevelopmentFaithfulnessError(
            "reproduced answer token count differs from inference"
        )
    for key, actual in (
        ("mean_generated_token_logprob", generation.mean_token_logprob),
        ("sequence_confidence", generation.sequence_confidence),
    ):
        expected = saved.get(key)
        if expected is None or actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-7
        ):
            raise LargerDevelopmentFaithfulnessError(
                f"reproduced answer {key} differs from inference"
            )


def _summarize(output: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, float], dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for condition in CONDITIONS:
        for path in sorted((output / condition / "items").glob("*.json")):
            item = _object(path)
            for method, result in item["methods"].items():
                for value in result["interventions"]:
                    key = (
                        condition,
                        method,
                        value["operation"],
                        value["treatment"],
                        float(value["fraction"]),
                    )
                    grouped[key][value["selection"]].append(
                        float(value["score_minus_original"])
                    )
    rows = []
    for key in sorted(grouped):
        condition, method, operation, treatment, fraction = key
        selections = grouped[key]
        salient = selections["most_salient"]
        random = selections["random"]
        if len(salient) != 64 or len(random) != 64:
            raise LargerDevelopmentFaithfulnessError(
                "faithfulness comparison is missing salient or random results"
            )
        salient_mean = sum(salient) / len(salient)
        random_mean = sum(random) / len(random)
        effect = (
            random_mean - salient_mean
            if operation == "deletion"
            else salient_mean - random_mean
        )
        rows.append(
            {
                "condition": condition,
                "method": method,
                "operation": operation,
                "treatment": treatment,
                "fraction": fraction,
                "items": len(salient),
                "most_salient_mean_score_minus_original": salient_mean,
                "random_mean_score_minus_original": random_mean,
                "faithfulness_effect_positive_is_expected": effect,
            }
        )
    return {
        "role": "descriptive faithfulness evidence; not a model-selection gate",
        "comparison": "most_salient versus deterministic random patches",
        "rows": rows,
    }


def _finite_matrix(values: Any) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise LargerDevelopmentFaithfulnessError("NumPy is required") from exc
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise LargerDevelopmentFaithfulnessError(
            "attribution must be a finite two-dimensional array"
        )
    return [[float(value) for value in row] for row in array.tolist()]


def _validate_faithfulness_lock(protocol: Mapping[str, Any]) -> None:
    faithfulness = protocol.get("metrics", {}).get("faithfulness", {})
    if faithfulness.get("conditions") != list(CONDITIONS):
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness conditions differ from the locked pair"
        )
    if faithfulness.get("subset") != "first 64 locked selection ranks":
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness subset differs from the lock"
        )
    if faithfulness.get("target") != "fixed generated answer":
        raise LargerDevelopmentFaithfulnessError(
            "faithfulness target differs from the lock"
        )


def _question(record: Mapping[str, Any]) -> str:
    values = [
        message.get("content")
        for message in record.get("messages", [])
        if message.get("role") == "user"
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise LargerDevelopmentFaithfulnessError(
            "record must contain exactly one user question"
        )
    return values[0].replace("<image>", "").strip()


def _write_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/study1/larger_development_protocol.json"),
    )
    parser.add_argument("--training-bundle", required=True, type=Path)
    parser.add_argument("--inference-run-dir", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/study1/smoke.yaml"))
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--required-gpu-substring", default="T4")
    parser.add_argument("--max-new-items", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_larger_development_faithfulness(
            project_root=args.project_root,
            protocol_path=args.protocol,
            training_bundle_dir=args.training_bundle,
            inference_run_dir=args.inference_run_dir,
            run_dir=args.run_dir,
            config_path=args.config,
            expected_commit=args.expected_commit,
            require_clean_git=args.require_clean_git,
            required_gpu_substring=args.required_gpu_substring,
            max_new_items=args.max_new_items,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
