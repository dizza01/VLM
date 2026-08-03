"""Dependency-light, reproducible SVG and CSV figures for the RQ1 benchmark."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import file_sha256

SCHEMA_VERSION = "gi-vqa-rq1-visualization-bundle-v1"
REPORT_SCHEMA_VERSION = "gi-vqa-rq1-benchmark-report-v1"
COLORS = {
    "fine_tuned_baseline": "#2563eb",
    "unadapted_base": "#94a3b8",
}


class RQ1VisualizationError(RuntimeError):
    """Raised when a benchmark report cannot produce the locked figures."""


def render_rq1_visualizations(
    *, report_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Render exact SVG/CSV artifacts without mutating the benchmark report."""

    report_file = Path(report_path).resolve()
    output = Path(output_dir).resolve()
    report = _object(report_file)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise RQ1VisualizationError("unexpected benchmark report schema")
    if report.get("status") != "PASS":
        raise RQ1VisualizationError("benchmark report has not passed")
    if report.get("test_partition_accessed") is not True:
        raise RQ1VisualizationError("RQ1 benchmark must disclose test access")
    metrics = report.get("condition_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(COLORS):
        raise RQ1VisualizationError("benchmark conditions are incomplete")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        headline_rows = _headline_rows(metrics)
        _write_csv(temporary / "headline_metrics.csv", headline_rows)
        (temporary / "headline_metrics.svg").write_text(
            _grouped_bar_svg(
                headline_rows,
                title="RQ1 full-test benchmark",
                category_field="metric",
                value_field="value",
            ),
            encoding="utf-8",
        )
        strata = report.get("strata", {})
        question_rows = _rows(strata.get("question_class"), "question_class")
        complexity_rows = _rows(strata.get("complexity"), "complexity")
        _write_csv(temporary / "question_class_token_f1.csv", question_rows)
        (temporary / "question_class_token_f1.svg").write_text(
            _heatmap_svg(
                question_rows,
                title="Normalized token F1 by question class",
                stratum_field="question_class",
            ),
            encoding="utf-8",
        )
        _write_csv(temporary / "complexity_token_f1.csv", complexity_rows)
        (temporary / "complexity_token_f1.svg").write_text(
            _grouped_bar_svg(
                complexity_rows,
                title="Normalized token F1 by complexity",
                category_field="complexity",
                value_field="normalized_token_f1",
            ),
            encoding="utf-8",
        )
        length_rows = [
            {
                "condition": condition,
                "mean_absolute_answer_token_count_error": value[
                    "mean_absolute_answer_token_count_error"
                ],
            }
            for condition, value in metrics.items()
        ]
        _write_csv(temporary / "answer_length_error.csv", length_rows)
        (temporary / "answer_length_error.svg").write_text(
            _single_bar_svg(
                length_rows,
                title="Mean absolute answer-length error",
                value_field="mean_absolute_answer_token_count_error",
            ),
            encoding="utf-8",
        )
        artifacts = {
            str(path.relative_to(temporary)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "report_sha256": file_sha256(report_file),
            "renderer": "deterministic dependency-light SVG v1",
            "artifacts": artifacts,
        }
        _write_json(temporary / "visualization_manifest.json", manifest)
        _publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _object(output / "visualization_manifest.json")


def _headline_rows(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in (
        "normalized_token_f1",
        "normalized_exact_match",
        "rouge_l_f1",
        "corpus_bleu_0_to_1",
    ):
        for condition, values in metrics.items():
            value = _unit_float(values.get(metric), f"{condition}.{metric}")
            interval = values.get(f"{metric}_confidence_interval_95", [value, value])
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(not isinstance(item, (int, float)) for item in interval)
            ):
                raise RQ1VisualizationError(f"invalid interval: {condition}.{metric}")
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "value": value,
                    "lower_95": float(interval[0]),
                    "upper_95": float(interval[1]),
                }
            )
    return rows


def _rows(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RQ1VisualizationError(f"missing benchmark stratum: {field}")
    rows = []
    for row in value:
        if not isinstance(row, Mapping):
            raise RQ1VisualizationError(f"invalid benchmark stratum: {field}")
        condition = row.get("condition")
        if condition not in COLORS:
            raise RQ1VisualizationError(f"invalid condition in {field}")
        stratum = str(row.get(field, "")).strip()
        if not stratum:
            raise RQ1VisualizationError(f"empty {field}")
        rows.append(
            {
                field: stratum,
                "condition": condition,
                "items": int(row["items"]),
                "normalized_token_f1": _unit_float(
                    row.get("normalized_token_f1"),
                    f"{field}.normalized_token_f1",
                ),
            }
        )
    return rows


def _grouped_bar_svg(rows, *, title, category_field, value_field):
    categories = list(dict.fromkeys(str(row[category_field]) for row in rows))
    conditions = list(COLORS)
    width = max(760, 160 + len(categories) * 145)
    height = 500
    left, top, plot_height = 90, 75, 320
    plot_width = width - left - 40
    group_width = plot_width / len(categories)
    bar_width = min(42, group_width / (len(conditions) + 1))
    parts = [_svg_open(width, height), _text(width / 2, 32, title, 22, "middle")]
    parts.extend(_axes(left, top, plot_width, plot_height))
    for category_index, category in enumerate(categories):
        centre = left + group_width * (category_index + 0.5)
        parts.append(_text(centre, top + plot_height + 28, category, 11, "middle"))
        for condition_index, condition in enumerate(conditions):
            matching = next(
                (
                    row
                    for row in rows
                    if str(row[category_field]) == category
                    and row["condition"] == condition
                ),
                None,
            )
            if matching is None:
                continue
            value = _unit_float(matching[value_field], value_field)
            x = centre + (condition_index - 0.5) * bar_width - bar_width / 2
            bar_height = value * plot_height
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 3:.2f}" '
                f'height="{bar_height:.2f}" fill="{COLORS[condition]}"/>'
            )
            parts.append(_text(x + bar_width / 2, y - 6, f"{value:.3f}", 10, "middle"))
    parts.extend(_legend(width - 265, 45))
    parts.append("</svg>")
    return "".join(parts)


def _single_bar_svg(rows, *, title, value_field):
    maximum = max(float(row[value_field]) for row in rows)
    scaled = [
        {
            "metric": "answer_length_error",
            "condition": row["condition"],
            "value": float(row[value_field]) / maximum if maximum else 0.0,
        }
        for row in rows
    ]
    svg = _grouped_bar_svg(
        scaled,
        title=title + f" (maximum={maximum:.3f} tokens)",
        category_field="metric",
        value_field="value",
    )
    return svg


def _heatmap_svg(rows, *, title, stratum_field):
    strata = list(dict.fromkeys(str(row[stratum_field]) for row in rows))
    width = 760
    cell_height = 30
    height = 100 + len(strata) * cell_height
    label_width, cell_width = 260, 200
    parts = [_svg_open(width, height), _text(width / 2, 30, title, 21, "middle")]
    for column, condition in enumerate(COLORS):
        parts.append(
            _text(
                label_width + column * cell_width + cell_width / 2,
                62,
                condition,
                12,
                "middle",
            )
        )
    for row_index, stratum in enumerate(strata):
        y = 75 + row_index * cell_height
        parts.append(_text(label_width - 10, y + 20, stratum, 11, "end"))
        for column, condition in enumerate(COLORS):
            matching = next(
                (
                    row
                    for row in rows
                    if str(row[stratum_field]) == stratum
                    and row["condition"] == condition
                ),
                None,
            )
            if matching is None:
                continue
            value = _unit_float(
                matching["normalized_token_f1"], "normalized_token_f1"
            )
            x = label_width + column * cell_width
            blue = int(245 - 145 * value)
            color = f"rgb({blue},{blue + 5},{255})"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 2}" '
                f'height="{cell_height - 2}" fill="{color}"/>'
            )
            parts.append(_text(x + cell_width / 2, y + 20, f"{value:.3f}", 12, "middle"))
    parts.append("</svg>")
    return "".join(parts)


def _axes(left, top, width, height):
    values = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#334155"/>',
    ]
    for index in range(6):
        value = index / 5
        y = top + height - value * height
        values.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" '
            'stroke="#e2e8f0"/>'
        )
        values.append(_text(left - 10, y + 4, f"{value:.1f}", 10, "end"))
    return values


def _legend(x, y):
    parts = []
    for index, (condition, color) in enumerate(COLORS.items()):
        offset = index * 20
        parts.append(
            f'<rect x="{x}" y="{y + offset}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(_text(x + 18, y + offset + 11, condition, 11, "start"))
    return parts


def _svg_open(width, height):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
    )


def _text(x, y, value, size, anchor):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#0f172a">'
        f"{html.escape(str(value))}</text>"
    )


def _unit_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RQ1VisualizationError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise RQ1VisualizationError(f"{name} must be between zero and one")
    return result


def _write_csv(path, rows):
    if not rows:
        raise RQ1VisualizationError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_directory(temporary, output):
    if output.exists():
        if _tree_hashes(output) != _tree_hashes(temporary):
            raise RQ1VisualizationError("existing visualizations differ")
        return
    os.replace(temporary, output)


def _tree_hashes(root):
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RQ1VisualizationError(f"expected JSON object: {path}")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = render_rq1_visualizations(
        report_path=args.report, output_dir=args.output_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
