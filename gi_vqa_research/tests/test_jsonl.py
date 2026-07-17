from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gi_vqa.jsonl import JsonlDecodeError, iter_jsonl, read_jsonl, write_jsonl_atomic


class JsonlTests(unittest.TestCase):
    def test_unicode_round_trip_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [{"question": "What is visible?"}, {"answer": "œsophagitis"}]
            write_jsonl_atomic(path, rows)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertEqual(read_jsonl(path), rows)

    def test_decode_error_reports_path_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"ok":1}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(JsonlDecodeError) as caught:
                list(iter_jsonl(path))
            self.assertEqual(caught.exception.line_number, 2)
            self.assertIn(str(path), str(caught.exception))

    def test_atomic_write_preserves_existing_file_on_generator_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"original":true}\n', encoding="utf-8")

            def failing_records():
                yield {"replacement": 1}
                raise RuntimeError("synthetic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                write_jsonl_atomic(path, failing_records())
            self.assertEqual(path.read_text(encoding="utf-8"), '{"original":true}\n')

    def test_non_object_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.jsonl"
            path.write_text("[1,2,3]\n", encoding="utf-8")
            with self.assertRaisesRegex(JsonlDecodeError, "expected a JSON object"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
