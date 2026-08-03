"""Frozen correctness analysis for the RQ1 full-test benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .identifiers import canonical_text
from .provenance import file_sha256
from .rq1_baseline import _object, _under, validate_rq1_protocol
from .rq1_test_runner import CONDITIONS, ITEM_SCHEMA_VERSION, SCHEMA_VERSION

REPORT_SCHEMA_VERSION = "gi-vqa-rq1-benchmark-report-v1"


class RQ1AnalysisError(RuntimeError):
    """Raised when immutable full-test outputs cannot be analysed."""


def analyze_rq1_benchmark(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    inference_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol_file = _under(root, protocol_path)
    protocol = _object(protocol_file)
    validate_rq1_protocol(
        project_root=root, protocol_path=protocol_file, require_locked=True
    )
    run = Path(inference_dir).resolve()
    status_path = run / "inference_status.json"
    status = _object(status_path)
    if status.get("schema_version") != SCHEMA_VERSION:
        raise RQ1AnalysisError("unexpected inference status schema")
    if status.get("status") != "INFERENCE_COMPLETE":
        raise RQ1AnalysisError("full-test inference is incomplete")
    if status.get("test_partition_accessed") is not True:
        raise RQ1AnalysisError("inference status does not disclose test access")

    rows_by_condition = {
        condition: _load_condition(run / condition, condition)
        for condition in CONDITIONS
    }
    expected = protocol["evaluation"]["expected_items_per_condition"]
    if any(len(rows) != expected for rows in rows_by_condition.values()):
        raise RQ1AnalysisError("condition item count differs from protocol")
    item_orders = {
        condition: [row["item_id"] for row in rows]
        for condition, rows in rows_by_condition.items()
    }
    if len({tuple(value) for value in item_orders.values()}) != 1:
        raise RQ1AnalysisError("condition item order differs")

    bootstrap = protocol["analysis"]["bootstrap"]
    scored = {
        condition: [_score_item(row) for row in rows]
        for condition, rows in rows_by_condition.items()
    }
    condition_metrics = {
        condition: _condition_metrics(
            rows,
            replicates=bootstrap["replicates"],
            seed=bootstrap["seed"] + index,
        )
        for index, (condition, rows) in enumerate(scored.items())
    }
    strata = {
        "question_class": _stratify(scored, "question_class"),
        "complexity": _stratify(scored, "complexity"),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "baseline_id": protocol["baseline_id"],
        "protocol_sha256": file_sha256(protocol_file),
        "inference_status_sha256": file_sha256(status_path),
        "test_partition_accessed": True,
        "conditions": list(CONDITIONS),
        "condition_metrics": condition_metrics,
        "strata": strata,
        "bootstrap": {
            **bootstrap,
            "implementation": (
                "source-group resampling of additive metric sufficient statistics"
            ),
        },
        "artifacts": {
            condition: {
                "items": len(rows_by_condition[condition]),
                "items_sha256": _directory_sha256(run / condition / "items"),
            }
            for condition in CONDITIONS
        },
    }
    _publish_or_match(Path(output_path).resolve(), report)
    return report


def normalized_token_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
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


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, target in enumerate(expected, start=1):
            current.append(
                previous[index - 1] + 1
                if token == target
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall)


def _score_item(row: Mapping[str, Any]) -> dict[str, Any]:
    prediction = str(row["prediction"])
    reference = str(row["reference_answer"])
    predicted_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    correct, total, sys_len, ref_len = _bleu_statistics(
        predicted_tokens, reference_tokens
    )
    classes = row.get("question_class") or ["unclassified"]
    return {
        "item_id": row["item_id"],
        "source_img_id": row["source_img_id"],
        "question_class": [str(value) for value in classes],
        "complexity": str(row.get("complexity") or "unclassified"),
        "normalized_token_f1": normalized_token_f1(prediction, reference),
        "normalized_exact_match": float(
            canonical_text(prediction, casefold=True)
            == canonical_text(reference, casefold=True)
        ),
        "rouge_l_f1": rouge_l_f1(prediction, reference),
        "prediction_tokens": len(predicted_tokens),
        "reference_tokens": len(reference_tokens),
        "absolute_answer_token_count_error": abs(
            len(predicted_tokens) - len(reference_tokens)
        ),
        "inference_seconds": float(row["inference_seconds"]),
        "bleu_correct": correct,
        "bleu_total": total,
        "bleu_sys_len": sys_len,
        "bleu_ref_len": ref_len,
    }


def _condition_metrics(rows, *, replicates, seed):
    if not rows:
        raise RQ1AnalysisError("cannot analyse an empty condition")
    metrics = ("normalized_token_f1", "normalized_exact_match", "rouge_l_f1")
    result = {
        metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics
    }
    result.update(
        {
            "corpus_bleu_0_to_1": _bleu_from_rows(rows),
            "completed_items": len(rows),
            "failed_items": 0,
            "mean_answer_token_count": sum(
                row["prediction_tokens"] for row in rows
            )
            / len(rows),
            "mean_absolute_answer_token_count_error": sum(
                row["absolute_answer_token_count_error"] for row in rows
            )
            / len(rows),
            "inference_seconds_per_item": sum(
                row["inference_seconds"] for row in rows
            )
            / len(rows),
        }
    )
    intervals = _group_bootstrap(rows, replicates=replicates, seed=seed)
    for metric, interval in intervals.items():
        result[f"{metric}_confidence_interval_95"] = interval
    return result


def _group_bootstrap(rows, *, replicates, seed):
    import numpy as np

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_img_id"]].append(row)
    groups = list(grouped.values())
    metric_names = (
        "normalized_token_f1",
        "normalized_exact_match",
        "rouge_l_f1",
    )
    # Each source image becomes one row of additive sufficient statistics.
    matrix = np.asarray(
        [
            [
                len(group),
                *(sum(row[name] for row in group) for name in metric_names),
                *(sum(row["bleu_correct"][i] for row in group) for i in range(4)),
                *(sum(row["bleu_total"][i] for row in group) for i in range(4)),
                sum(row["bleu_sys_len"] for row in group),
                sum(row["bleu_ref_len"] for row in group),
            ]
            for group in groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    distributions = {
        name: np.empty(replicates, dtype=np.float64)
        for name in (*metric_names, "corpus_bleu_0_to_1")
    }
    chunk_size = 100
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        totals = matrix[indices].sum(axis=1)
        for offset, name in enumerate(metric_names, start=1):
            distributions[name][start:stop] = totals[:, offset] / totals[:, 0]
        distributions["corpus_bleu_0_to_1"][start:stop] = _bleu_from_totals(
            totals[:, 4:8],
            totals[:, 8:12],
            totals[:, 12],
            totals[:, 13],
        )
    return {
        metric: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for metric, values in distributions.items()
    }


def _bleu_from_totals(correct, total, sys_len, ref_len):
    import numpy as np

    precisions = (correct + 1.0) / (total + 1.0)
    brevity = np.where(
        sys_len >= ref_len,
        1.0,
        np.exp(1.0 - np.divide(ref_len, sys_len, where=sys_len != 0)),
    )
    scores = brevity * np.exp(np.log(precisions).mean(axis=1))
    return np.where(sys_len == 0, 0.0, scores)


def _stratify(scored, field):
    values = []
    for condition, rows in scored.items():
        groups = defaultdict(list)
        for row in rows:
            labels = row[field] if field == "question_class" else [row[field]]
            for label in labels:
                groups[str(label)].append(row)
        for label in sorted(groups):
            selected = groups[label]
            values.append(
                {
                    field: label,
                    "condition": condition,
                    "items": len(selected),
                    "normalized_token_f1": sum(
                        row["normalized_token_f1"] for row in selected
                    )
                    / len(selected),
                }
            )
    return values


def _bleu_statistics(predicted, reference):
    correct, total = [], []
    for order in range(1, 5):
        predicted_ngrams = Counter(_ngrams(predicted, order))
        reference_ngrams = Counter(_ngrams(reference, order))
        correct.append(sum((predicted_ngrams & reference_ngrams).values()))
        total.append(sum(predicted_ngrams.values()))
    return correct, total, len(predicted), len(reference)


def _bleu_from_rows(rows):
    correct = [sum(row["bleu_correct"][i] for row in rows) for i in range(4)]
    total = [sum(row["bleu_total"][i] for row in rows) for i in range(4)]
    sys_len = sum(row["bleu_sys_len"] for row in rows)
    ref_len = sum(row["bleu_ref_len"] for row in rows)
    if sys_len == 0:
        return 0.0
    precisions = [
        (value + 1.0) / (denominator + 1.0)
        for value, denominator in zip(correct, total)
    ]
    brevity = 1.0 if sys_len >= ref_len else math.exp(1.0 - ref_len / sys_len)
    return brevity * math.exp(sum(math.log(value) for value in precisions) / 4)


def _ngrams(tokens, order):
    return (tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _tokens(value):
    normalized = canonical_text(value, casefold=True)
    return normalized.split() if normalized else []


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _load_condition(directory, condition):
    paths = sorted((directory / "items").glob("*.json"))
    rows = []
    for path in paths:
        row = _object(path)
        if row.get("schema_version") != ITEM_SCHEMA_VERSION:
            raise RQ1AnalysisError(f"unexpected item schema: {path}")
        if row.get("condition") != condition:
            raise RQ1AnalysisError(f"condition mismatch: {path}")
        rows.append(row)
    return rows


def _directory_sha256(directory):
    entries = [
        {"path": path.name, "sha256": file_sha256(path)}
        for path in sorted(directory.glob("*.json"))
    ]
    from .provenance import canonical_json_sha256

    return canonical_json_sha256(entries)


def _publish_or_match(path, value):
    if path.exists():
        if _object(path) != value:
            raise RQ1AnalysisError(f"existing report differs: {path}")
        return
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze_rq1_benchmark(
        project_root=args.project_root,
        protocol_path=args.protocol,
        inference_dir=args.inference_dir,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
