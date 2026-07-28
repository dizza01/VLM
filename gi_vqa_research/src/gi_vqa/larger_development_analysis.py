"""Locked correctness, grounding and calibration analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .identifiers import canonical_text
from .larger_development import CONDITIONS, validate_locked_protocol
from .provenance import file_sha256

ANALYSIS_SCHEMA_VERSION = "gi-vqa-larger-development-analysis-v1"


class LargerDevelopmentAnalysisError(RuntimeError):
    """Raised when inference evidence is incomplete or violates the lock."""


def analyze_larger_development(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    lock = validate_locked_protocol(project_root=root, protocol_path=protocol_file)
    protocol = _object(protocol_file)
    run = Path(run_dir).resolve()
    identity = _object(run / "identity.json")
    if identity.get("protocol_sha256") != lock["protocol_sha256"]:
        raise LargerDevelopmentAnalysisError("run belongs to another protocol")
    selection = _object(
        _under(root, protocol["inputs"]["selection_manifest"]["path"])
    )
    item_ids = [row["item_id"] for row in selection["records"]]
    rows = {
        condition: _condition_rows(run, condition, item_ids)
        for condition in CONDITIONS
    }
    condition_metrics = {
        condition: _condition_metrics(values)
        for condition, values in rows.items()
    }
    analysis = protocol["analysis"]
    primary = _contrast(
        rows["paired_correct"],
        rows["constant_control"],
        replicates=analysis["bootstrap"]["replicates"],
        seed=analysis["bootstrap"]["seed"],
    )
    shuffled = _contrast(
        rows["paired_correct"],
        rows["paired_shuffled"],
        replicates=analysis["bootstrap"]["replicates"],
        seed=analysis["bootstrap"]["seed"] + 1,
    )
    neutral = _contrast(
        rows["paired_correct"],
        rows["paired_neutral_ablation"],
        replicates=analysis["bootstrap"]["replicates"],
        seed=analysis["bootstrap"]["seed"] + 2,
    )
    primary_pass = primary["confidence_interval_95"][0] > 0
    shuffled_pass = shuffled["confidence_interval_95"][0] > 0
    neutral_pass = neutral["confidence_interval_95"][0] > 0
    promote = primary_pass and shuffled_pass and neutral_pass
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "PASS",
        "analysis_complete": True,
        "diagnostic_only": True,
        "excluded_from_research_results": True,
        "test_partition_accessed": False,
        "protocol_sha256": file_sha256(protocol_file),
        "items_per_condition": len(item_ids),
        "condition_metrics": condition_metrics,
        "contrasts": {
            "primary_paired_minus_constant": primary,
            "grounding_paired_minus_shuffled": shuffled,
            "grounding_paired_minus_neutral": neutral,
        },
        "locked_success_checks": {
            "primary_correctness": primary_pass,
            "shuffled_grounding": shuffled_pass,
            "neutral_grounding": neutral_pass,
        },
        "promotion_decision": {
            "promote_paired_image_adapter": promote,
            "action": (
                "promote_to_next_locked_stage"
                if promote
                else "do_not_promote; revise only under a new development protocol"
            ),
        },
    }
    _publish_or_match(run / "larger_development_analysis.json", report)
    return report


def normalized_token_f1(prediction: str, reference: str) -> float:
    predicted = canonical_text(prediction, casefold=True).split()
    expected = canonical_text(reference, casefold=True).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from rouge_score import rouge_scorer
        import sacrebleu
    except ImportError as exc:
        raise LargerDevelopmentAnalysisError(
            "rouge-score and sacrebleu are required"
        ) from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f1 = [
        normalized_token_f1(row["prediction"], row["reference_answer"])
        for row in rows
    ]
    exact = [
        canonical_text(row["prediction"], casefold=True)
        == canonical_text(row["reference_answer"], casefold=True)
        for row in rows
    ]
    rouge = [
        scorer.score(row["reference_answer"], row["prediction"])["rougeL"].fmeasure
        for row in rows
    ]
    length_error = [
        abs(
            len(canonical_text(row["prediction"], casefold=True).split())
            - len(canonical_text(row["reference_answer"], casefold=True).split())
        )
        for row in rows
    ]
    confidence = [float(row["sequence_confidence"]) for row in rows]
    correctness = [1.0 if value >= 0.5 else 0.0 for value in f1]
    return {
        "normalized_token_f1": sum(f1) / len(f1),
        "normalized_exact_match": sum(exact) / len(exact),
        "rouge_l_f1": sum(rouge) / len(rouge),
        "corpus_bleu": sacrebleu.corpus_bleu(
            [row["prediction"] for row in rows],
            [[row["reference_answer"] for row in rows]],
        ).score,
        "mean_absolute_answer_token_count_error": sum(length_error) / len(length_error),
        "mean_sequence_confidence": sum(confidence) / len(confidence),
        "mean_generated_token_logprob": sum(
            float(row["mean_generated_token_logprob"]) for row in rows
        )
        / len(rows),
        "calibration": {
            "correctness_event": "normalized_token_f1>=0.5",
            "brier_score": sum(
                (probability - target) ** 2
                for probability, target in zip(confidence, correctness)
            )
            / len(rows),
            "expected_calibration_error_10_bins": _ece(confidence, correctness),
        },
    }


def _contrast(left, right, *, replicates, seed):
    import numpy as np

    differences = np.asarray(
        [
            normalized_token_f1(a["prediction"], a["reference_answer"])
            - normalized_token_f1(b["prediction"], b["reference_answer"])
            for a, b in zip(left, right)
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(differences), size=(replicates, len(differences))
    )
    bootstrap = differences[indices].mean(axis=1)
    return {
        "metric": "normalized_token_f1",
        "items": len(differences),
        "mean_difference": float(differences.mean()),
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_replicates": replicates,
        "resampling_unit": "source_img_id",
        "seed": seed,
    }


def _ece(confidence, correctness):
    total = len(confidence)
    value = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected = [
            position
            for position, probability in enumerate(confidence)
            if lower <= probability < upper or (index == 9 and probability == 1)
        ]
        if selected:
            mean_confidence = sum(confidence[i] for i in selected) / len(selected)
            mean_accuracy = sum(correctness[i] for i in selected) / len(selected)
            value += len(selected) / total * abs(mean_confidence - mean_accuracy)
    return value


def _condition_rows(run, condition, item_ids):
    directory = run / condition / "items"
    rows = []
    for rank, item_id in enumerate(item_ids):
        path = directory / f"{rank:03d}-{item_id}.json"
        if not path.is_file():
            raise LargerDevelopmentAnalysisError(f"missing completion: {path}")
        row = _object(path)
        if row.get("condition") != condition or row.get("item_id") != item_id:
            raise LargerDevelopmentAnalysisError(f"invalid completion: {path}")
        rows.append(row)
    if len(list(directory.glob("*.json"))) != len(item_ids):
        raise LargerDevelopmentAnalysisError(f"unexpected completions: {condition}")
    return rows


def _under(root, value):
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LargerDevelopmentAnalysisError("path escapes project root") from exc
    return resolved


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LargerDevelopmentAnalysisError(f"expected object: {path}")
    return value


def _publish_or_match(path, value):
    if path.exists():
        if _object(path) != value:
            raise LargerDevelopmentAnalysisError("existing analysis differs")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Mounted Google Drive does not implement hard links. Publish the fully
    # flushed temporary file with an atomic rename so Colab runs remain
    # restart-safe.
    try:
        if path.exists():
            if _object(path) != value:
                raise LargerDevelopmentAnalysisError(
                    "existing analysis differs"
                )
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/study1/larger_development_protocol.json"),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_larger_development(
        project_root=args.project_root,
        protocol_path=args.protocol,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
