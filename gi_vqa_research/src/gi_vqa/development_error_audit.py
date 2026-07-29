"""Build a deterministic blinded error-audit pack from development evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .identifiers import canonical_text, question_text
from .jsonl import iter_jsonl
from .larger_development_analysis import normalized_token_f1
from .provenance import canonical_json_sha256, file_sha256

SCHEMA_VERSION = "gi-vqa-development-error-audit-v1"
CONDITIONS = (
    "paired_correct",
    "constant_control",
    "paired_shuffled",
    "paired_neutral_ablation",
)
REASONS = (
    "paired_worse_than_neutral",
    "paired_better_than_shuffled",
    "correct_equals_shuffled_high_confidence_wrong",
    "high_confidence_paired_error",
)


class DevelopmentErrorAuditError(RuntimeError):
    """Raised when an audit pack cannot be built without violating its lock."""


def build_development_error_audit(
    *,
    project_root: str | Path,
    evidence_dir: str | Path,
    output_dir: str | Path,
    specification_path: str | Path = (
        "protocols/study1/development_error_audit_v1.json"
    ),
    materialize_images: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Validate evidence and build a blinded, deterministic reviewer pack."""

    root = Path(project_root).resolve()
    evidence = Path(evidence_dir).resolve()
    output = Path(output_dir).resolve()
    specification_file = _under(root, specification_path)
    specification = _object(specification_file)
    _validate_specification(specification)
    implementation = specification["implementation"]
    implementation_file = _under(root, implementation["path"])
    if file_sha256(implementation_file) != implementation["sha256"]:
        raise DevelopmentErrorAuditError("error-audit implementation changed")
    receipt_file = _under(root, specification["input"]["result_receipt"]["path"])
    receipt = _object(receipt_file)
    if file_sha256(receipt_file) != specification["input"]["result_receipt"]["sha256"]:
        raise DevelopmentErrorAuditError("larger-development result receipt changed")
    _validate_receipt_and_evidence(receipt, evidence)

    rows = _load_condition_rows(evidence)
    questions = _development_questions(root, specification)
    selected = _select_items(
        rows,
        total=int(specification["selection"]["total_items"]),
        quota=int(specification["selection"]["quota_per_reason"]),
    )
    reviewer_rows, key_rows, item_rows = _blind_items(
        selected,
        rows,
        questions=questions,
        seed=int(specification["selection"]["seed"]),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        _write_csv(temporary / "reviewer_1.csv", reviewer_rows)
        _write_csv(temporary / "reviewer_2.csv", reviewer_rows)
        _write_csv(temporary / "adjudication.csv", _adjudication_rows(item_rows))
        _write_csv(temporary / "unblinding_key.csv", key_rows)
        _write_json(
            temporary / "selected_items.json",
            {
                "schema_version": SCHEMA_VERSION,
                "test_partition_accessed": False,
                "items": item_rows,
            },
        )
        (temporary / "README.md").write_text(
            _instructions(len(item_rows)), encoding="utf-8"
        )
        image_result = _images(
            temporary=temporary,
            item_rows=item_rows,
            specification=specification,
            materialize=materialize_images,
            token=token,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "diagnostic_only": True,
            "inferential_use_forbidden": True,
            "test_partition_accessed": False,
            "repository_commit": receipt["repository_commit"],
            "protocol_sha256": receipt["protocol_sha256"],
            "result_receipt": {
                "path": str(receipt_file.relative_to(root)),
                "sha256": file_sha256(receipt_file),
            },
            "specification": {
                "path": str(specification_file.relative_to(root)),
                "sha256": file_sha256(specification_file),
            },
            "selection": {
                "algorithm": specification["selection"]["algorithm"],
                "seed": specification["selection"]["seed"],
                "items": len(item_rows),
                "ordered_item_ids_sha256": canonical_json_sha256(
                    [row["audit_item_id"] for row in item_rows]
                ),
                "reason_counts": _reason_counts(item_rows),
            },
            "blinding": {
                "method": "per-item deterministic permutation",
                "condition_codes": ["A", "B", "C", "D"],
                "reviewer_sheets_include_condition": False,
                "reviewer_sheets_include_confidence": False,
                "key_separate": True,
            },
            "images": image_result,
        }
        artifact_paths = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "audit_manifest.json"
        )
        manifest["artifacts"] = {
            str(path.relative_to(temporary)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        }
        _write_json(temporary / "audit_manifest.json", manifest)
        _publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _object(output / "audit_manifest.json")


def _validate_receipt_and_evidence(
    receipt: Mapping[str, Any], evidence: Path
) -> None:
    if receipt.get("schema_version") != "gi-vqa-larger-development-result-v1":
        raise DevelopmentErrorAuditError("unexpected result receipt schema")
    if receipt.get("test_partition_accessed") is not False:
        raise DevelopmentErrorAuditError("result receipt does not preserve test seal")
    if receipt.get("outcome", {}).get("promote_paired_image_adapter") is not False:
        raise DevelopmentErrorAuditError("error audit requires a non-promotion result")
    required = {
        "inference_status.json": receipt["artifacts"]["inference_status_sha256"],
        "larger_development_analysis.json": receipt["artifacts"][
            "larger_development_analysis_sha256"
        ],
        "faithfulness/faithfulness_report.json": receipt["artifacts"][
            "faithfulness_report_sha256"
        ],
        "identity.json": receipt["artifacts"]["inference_identity_sha256"],
        "final_evidence_manifest.json": receipt["artifacts"][
            "final_evidence_manifest_sha256"
        ],
    }
    for relative, expected in required.items():
        path = evidence / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise DevelopmentErrorAuditError(
                f"evidence artifact missing or changed: {relative}"
            )
    status = _object(evidence / "inference_status.json")
    analysis = _object(evidence / "larger_development_analysis.json")
    faithfulness = _object(evidence / "faithfulness/faithfulness_report.json")
    if status.get("status") != "INFERENCE_COMPLETE":
        raise DevelopmentErrorAuditError("inference evidence is incomplete")
    if analysis.get("status") != "PASS" or faithfulness.get("status") != "PASS":
        raise DevelopmentErrorAuditError("analysis evidence did not pass")
    if any(
        value.get("test_partition_accessed") is not False
        for value in (status, analysis, faithfulness)
    ):
        raise DevelopmentErrorAuditError("evidence does not preserve test seal")


def _load_condition_rows(evidence: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        directory = evidence / condition / "items"
        values: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            row = _object(path)
            if row.get("condition") != condition:
                raise DevelopmentErrorAuditError(f"condition mismatch: {path}")
            item_id = _string(row.get("item_id"), "item_id")
            if item_id in values:
                raise DevelopmentErrorAuditError(f"duplicate item: {item_id}")
            values[item_id] = row
        if len(values) != 256:
            raise DevelopmentErrorAuditError(
                f"{condition} must contain 256 development items"
            )
        result[condition] = values
    ids = set(result[CONDITIONS[0]])
    if any(set(result[condition]) != ids for condition in CONDITIONS[1:]):
        raise DevelopmentErrorAuditError("condition item sets differ")
    for item_id in ids:
        reference = result["paired_correct"][item_id]
        for condition in CONDITIONS[1:]:
            candidate = result[condition][item_id]
            for field in (
                "item_id",
                "record_sha256",
                "source_img_id",
                "reference_answer",
                "complexity",
                "question_class",
            ):
                if candidate.get(field) != reference.get(field):
                    raise DevelopmentErrorAuditError(
                        f"matched item field differs: {item_id}.{field}"
                    )
    return result


def _development_questions(
    root: Path, specification: Mapping[str, Any]
) -> dict[str, str]:
    descriptor = specification["input"]["grouped_split_manifest"]
    manifest_path = _under(root, descriptor["path"])
    if file_sha256(manifest_path) != descriptor["sha256"]:
        raise DevelopmentErrorAuditError("grouped split manifest changed")
    manifest = _object(manifest_path)
    if manifest.get("dataset", {}).get("id") != specification["images"]["dataset_id"]:
        raise DevelopmentErrorAuditError("audit image dataset differs from split lock")
    if (
        manifest.get("dataset", {}).get("revision")
        != specification["images"]["dataset_revision"]
    ):
        raise DevelopmentErrorAuditError(
            "audit image dataset revision differs from split lock"
        )
    development = manifest.get("artifacts", {}).get("development", {})
    development_path = _under(root, development.get("path"))
    if (
        not development_path.is_file()
        or file_sha256(development_path) != development.get("sha256")
    ):
        raise DevelopmentErrorAuditError(
            "materialized development artifact is missing or changed"
        )
    questions: dict[str, str] = {}
    for record in iter_jsonl(development_path):
        item_id = record.get("item_id")
        if isinstance(item_id, str):
            questions[item_id] = question_text(record)
    return questions


def _select_items(
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    total: int,
    quota: int,
) -> list[dict[str, Any]]:
    paired = rows["paired_correct"]
    candidates = []
    for item_id, row in paired.items():
        values = {
            condition: normalized_token_f1(
                str(rows[condition][item_id]["prediction"]),
                str(row["reference_answer"]),
            )
            for condition in CONDITIONS
        }
        same_shuffled = (
            canonical_text(str(row["prediction"]), casefold=True)
            == canonical_text(
                str(rows["paired_shuffled"][item_id]["prediction"]),
                casefold=True,
            )
        )
        reasons = {
            "paired_worse_than_neutral": (
                values["paired_correct"] < values["paired_neutral_ablation"]
            ),
            "paired_better_than_shuffled": (
                values["paired_correct"] > values["paired_shuffled"]
            ),
            "correct_equals_shuffled_high_confidence_wrong": (
                same_shuffled
                and values["paired_correct"] < 0.5
                and float(row["sequence_confidence"]) >= 0.5
            ),
            "high_confidence_paired_error": (
                values["paired_correct"] < 0.5
                and float(row["sequence_confidence"]) >= 0.5
            ),
        }
        candidates.append(
            {
                "item_id": item_id,
                "rank": int(row["rank"]),
                "f1": values,
                "confidence": float(row["sequence_confidence"]),
                "reasons": reasons,
            }
        )
    orderings = {
        "paired_worse_than_neutral": lambda value: (
            value["f1"]["paired_correct"]
            - value["f1"]["paired_neutral_ablation"],
            value["rank"],
        ),
        "paired_better_than_shuffled": lambda value: (
            -(
                value["f1"]["paired_correct"]
                - value["f1"]["paired_shuffled"]
            ),
            value["rank"],
        ),
        "correct_equals_shuffled_high_confidence_wrong": lambda value: (
            -value["confidence"],
            value["rank"],
        ),
        "high_confidence_paired_error": lambda value: (
            -value["confidence"],
            value["rank"],
        ),
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for reason in REASONS:
        eligible = sorted(
            (value for value in candidates if value["reasons"][reason]),
            key=orderings[reason],
        )
        added = 0
        for value in eligible:
            if value["item_id"] in selected_ids:
                continue
            chosen = dict(value)
            chosen["primary_reason"] = reason
            selected.append(chosen)
            selected_ids.add(value["item_id"])
            added += 1
            if added == quota:
                break
    if len(selected) < total:
        remaining = sorted(
            (value for value in candidates if value["item_id"] not in selected_ids),
            key=lambda value: (
                -sum(bool(flag) for flag in value["reasons"].values()),
                value["f1"]["paired_correct"],
                -value["confidence"],
                value["rank"],
            ),
        )
        for value in remaining:
            chosen = dict(value)
            chosen["primary_reason"] = "diagnostic_fill"
            selected.append(chosen)
            if len(selected) == total:
                break
    if len(selected) != total:
        raise DevelopmentErrorAuditError("could not fill the locked audit size")
    return selected


def _blind_items(selected, rows, *, questions, seed):
    reviewer_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for audit_rank, selected_row in enumerate(selected):
        item_id = selected_row["item_id"]
        paired = rows["paired_correct"][item_id]
        if item_id not in questions:
            raise DevelopmentErrorAuditError(
                f"selected item is absent from development artifact: {item_id}"
            )
        question = questions[item_id]
        conditions = list(CONDITIONS)
        conditions.sort(
            key=lambda condition: hashlib.sha256(
                f"{seed}\0{item_id}\0{condition}".encode()
            ).hexdigest()
        )
        audit_id = f"DEV-AUDIT-{audit_rank + 1:03d}"
        codes = dict(zip(conditions, ("A", "B", "C", "D")))
        item_rows.append(
            {
                "audit_item_id": audit_id,
                "source_img_id": paired["source_img_id"],
                "image_path": f"images/{paired['source_img_id']}.jpg",
                "item_id_sha256": hashlib.sha256(item_id.encode()).hexdigest(),
                "complexity": paired.get("complexity"),
                "question_class": paired.get("question_class", []),
                "question": question,
                "reference_answer": paired["reference_answer"],
                "primary_reason": selected_row["primary_reason"],
                "reason_flags": selected_row["reasons"],
            }
        )
        for condition in CONDITIONS:
            row = rows[condition][item_id]
            code = codes[condition]
            reviewer_rows.append(
                {
                    "audit_item_id": audit_id,
                    "image_path": f"images/{paired['source_img_id']}.jpg",
                    "complexity": paired.get("complexity"),
                    "question_class": "|".join(paired.get("question_class", [])),
                    "question": question,
                    "reference_answer": paired["reference_answer"],
                    "output_code": code,
                    "model_answer": row["prediction"],
                    "answer_correct_0_1_U": "",
                    "image_supported_0_1_U_NA": "",
                    "clinically_material_false_presence_0_1_NA": "",
                    "clinically_material_false_absence_0_1_NA": "",
                    "wrong_anatomical_location_0_1_NA": "",
                    "wrong_count_0_1_NA": "",
                    "unsupported_certainty_0_1_NA": "",
                    "reviewer_notes": "",
                }
            )
            key_rows.append(
                {
                    "audit_item_id": audit_id,
                    "output_code": code,
                    "condition": condition,
                    "source_img_id": paired["source_img_id"],
                    "input_source_img_id": row["input_source_img_id"],
                    "normalized_token_f1": selected_row["f1"][condition],
                    "sequence_confidence": row["sequence_confidence"],
                    "mean_generated_token_logprob": row[
                        "mean_generated_token_logprob"
                    ],
                    "primary_selection_reason": selected_row["primary_reason"],
                }
            )
    return reviewer_rows, key_rows, item_rows


def _adjudication_rows(item_rows):
    fields = {
        "audit_item_id": "",
        "output_code": "",
        "reviewer_1_answer_correct": "",
        "reviewer_2_answer_correct": "",
        "reviewer_1_image_supported": "",
        "reviewer_2_image_supported": "",
        "disagreement_0_1": "",
        "adjudicated_answer_correct_0_1_U": "",
        "adjudicated_image_supported_0_1_U_NA": "",
        "adjudicator_notes": "",
    }
    return [
        {**fields, "audit_item_id": item["audit_item_id"], "output_code": code}
        for item in item_rows
        for code in ("A", "B", "C", "D")
    ]


def _images(*, temporary, item_rows, specification, materialize, token):
    image_dir = temporary / "images"
    image_dir.mkdir()
    if not materialize:
        return {
            "materialized": False,
            "expected": len(item_rows),
            "materialized_count": 0,
            "source": dict(specification["images"]),
        }
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise DevelopmentErrorAuditError(
            "install the data dependencies to materialize images"
        ) from exc
    source = specification["images"]
    for row in item_rows:
        source_id = row["source_img_id"]
        downloaded = Path(
            hf_hub_download(
                repo_id=source["dataset_id"],
                repo_type="dataset",
                revision=source["dataset_revision"],
                filename=f"images/{source_id}.jpg",
                token=token,
            )
        )
        shutil.copyfile(downloaded, image_dir / f"{source_id}.jpg")
    return {
        "materialized": True,
        "expected": len(item_rows),
        "materialized_count": len(list(image_dir.glob("*.jpg"))),
        "source": dict(source),
    }


def _instructions(items):
    return f"""# Blinded development error audit

This pack contains {items} development-only cases. It is a post-hoc diagnostic
exercise and must not be used for inferential claims or to alter the completed
larger-development result.

## Reviewer workflow

1. Keep `unblinding_key.csv` inaccessible to reviewers.
2. Assign `reviewer_1.csv` and `reviewer_2.csv` independently.
3. Review the corresponding correct development image, reference answer and
   each blinded output A-D.
4. Complete every applicable coded field. Use `U` for uncertain and `NA` only
   where the category cannot apply.
5. Merge disagreements into `adjudication.csv`; a third qualified reviewer
   resolves clinically material disagreements.
6. Unblind only after both independent reviews and adjudication are frozen.

Condition identities and model confidence are intentionally absent from the
reviewer sheets. The official test partition must remain inaccessible.
"""


def _validate_specification(value):
    if value.get("schema_version") != SCHEMA_VERSION:
        raise DevelopmentErrorAuditError("unexpected audit specification schema")
    if value.get("status") != "LOCKED_POST_HOC_DIAGNOSTIC":
        raise DevelopmentErrorAuditError("error-audit specification is not locked")
    if value.get("partition") != "development":
        raise DevelopmentErrorAuditError("audit specification is not development-only")
    if value.get("test_partition_access_allowed") is not False:
        raise DevelopmentErrorAuditError("audit specification does not seal test")
    selection = value.get("selection", {})
    if (
        selection.get("algorithm") != "four-reason-priority-v1"
        or selection.get("seed") != 42
        or selection.get("total_items") != 64
        or selection.get("quota_per_reason") != 16
        or selection.get("reasons") != list(REASONS)
    ):
        raise DevelopmentErrorAuditError("audit selection differs from v1 lock")


def _reason_counts(items):
    counts: dict[str, int] = {}
    for row in items:
        reason = row["primary_reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _publish_directory(temporary, output):
    if output.exists():
        existing = _tree_hashes(output)
        proposed = _tree_hashes(temporary)
        if existing != proposed:
            raise DevelopmentErrorAuditError(
                "existing audit pack differs from deterministic reconstruction"
            )
        return
    os.replace(temporary, output)


def _tree_hashes(root):
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise DevelopmentErrorAuditError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DevelopmentErrorAuditError(f"expected JSON object: {path}")
    return value


def _under(root, value):
    path = Path(value)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise DevelopmentErrorAuditError(f"path escapes project root: {value}") from exc
    return result


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise DevelopmentErrorAuditError(f"{name} must be a non-empty string")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("protocols/study1/development_error_audit_v1.json"),
    )
    parser.add_argument("--materialize-images", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_development_error_audit(
        project_root=args.project_root,
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        specification_path=args.specification,
        materialize_images=args.materialize_images,
        token=os.getenv("HF_TOKEN"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
