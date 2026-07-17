from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from gi_vqa.provenance import (
    build_run_manifest,
    canonical_json_sha256,
    file_fingerprint,
    file_sha256,
    load_run_manifest,
    write_run_manifest,
)


class ProvenanceTests(unittest.TestCase):
    def test_streaming_hash_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"abc")
            expected = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(file_sha256(path, chunk_size=1), expected)
            self.assertEqual(
                file_fingerprint(path),
                {"path": str(path), "bytes": 3, "sha256": expected},
            )

    def test_canonical_json_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )

    def test_manifest_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "split.jsonl"
            input_path.write_text('{"item":1}\n', encoding="utf-8")
            manifest = build_run_manifest(
                run_id="study1-seed42",
                stage="audit",
                config={"seed": 42, "model": "example/model"},
                inputs={"split": input_path},
                command=["python", "-m", "gi_vqa.cli", "audit"],
                code_revision="abc123",
                environment={"cuda": "12.4"},
                created_at_utc="2026-07-17T12:00:00+00:00",
            )
            path = write_run_manifest(root / "manifest.json", manifest)
            self.assertEqual(load_run_manifest(path), manifest)

            tampered = dict(manifest)
            tampered["stage"] = "train"
            with self.assertRaisesRegex(ValueError, "content hash"):
                write_run_manifest(root / "tampered.json", tampered)

    def test_input_mutation_changes_future_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("first", encoding="utf-8")
            first = build_run_manifest(
                run_id="run",
                stage="stage",
                config={},
                inputs={"input": path},
                created_at_utc="2026-01-01T00:00:00+00:00",
            )
            path.write_text("second", encoding="utf-8")
            second = build_run_manifest(
                run_id="run",
                stage="stage",
                config={},
                inputs={"input": path},
                created_at_utc="2026-01-01T00:00:00+00:00",
            )
            self.assertNotEqual(
                first["inputs"]["input"]["sha256"],
                second["inputs"]["input"]["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
