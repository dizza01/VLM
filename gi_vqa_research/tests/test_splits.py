from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from gi_vqa.identifiers import source_image_id, stable_item_id
from gi_vqa.jsonl import read_jsonl
from gi_vqa.splits import (
    GROUPED_SPLIT_SCHEMA_VERSION,
    SplitBuildError,
    SplitBuildPaths,
    build_grouped_splits,
    materialize_grouped_split_artifacts,
    select_smoke_records,
    verify_grouped_split_artifacts,
)


def raw_record(
    image_id: str,
    question: str,
    answer: str,
    *,
    complexity: int = 1,
    question_class: list[str] | None = None,
) -> dict:
    return {
        "img_id": image_id,
        "question": question,
        "answer": answer,
        "complexity": complexity,
        "question_class": question_class or ["location"],
        "original": True,
    }


class GroupedSplitTests(unittest.TestCase):
    def source_records(self) -> tuple[list[dict], list[dict]]:
        train = [
            raw_record(
                f"source-{index:02d}",
                f"Question {index}",
                f"Answer {index}",
                complexity=(index % 3) + 1,
                question_class=[
                    ("location", "colour", "count")[index % 3]
                ],
            )
            for index in range(40)
        ]
        test = [
            # Exact cross-split duplicate: retain one row and both origins.
            raw_record("source-00", "Question 0", "Answer 0"),
            # Same image/question with a different answer: exclude both rows.
            raw_record("source-01", "Question 1", "Different answer"),
            # Exact QA duplicate with incompatible metadata: exclude the group.
            raw_record(
                "source-02",
                "Question 2",
                "Answer 2",
                complexity=99,
                question_class=["count"],
            ),
            *[
                raw_record(
                    f"source-{index:02d}",
                    f"Question {index}",
                    f"Answer {index}",
                    complexity=(index % 3) + 1,
                    question_class=[
                        ("location", "colour", "count")[index % 3]
                    ],
                )
                for index in range(40, 45)
            ],
        ]
        return train, test

    def build(self, root: Path) -> dict:
        train, test = self.source_records()
        return build_grouped_splits(
            official_train_records=train,
            official_test_records=test,
            dataset_id="example/dataset",
            dataset_revision="d" * 40,
            image_dataset_id="example/images",
            image_dataset_revision="i" * 40,
            seed=42,
            development_fraction=0.20,
            test_fraction=0.20,
            smoke_items=5,
            paths=SplitBuildPaths.resolve(
                project_root=root,
                data_root="data/processed/study1",
                manifest_path="protocols/study1/grouped_split_manifest.json",
                image_dir="data/images",
            ),
            reserved_source_ids=("source-03",),
        )

    def test_build_is_disjoint_reserved_and_fully_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.build(root)
            self.assertEqual(result["status"], "PASS")

            manifest_path = (
                root / "protocols/study1/grouped_split_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["schema_version"],
                GROUPED_SPLIT_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest["merge_and_exclusions"],
                {
                    "input_rows": 48,
                    "exact_duplicates_removed": 2,
                    "conflicting_image_question_groups_removed": 1,
                    "metadata_conflict_groups_removed": 1,
                    "conflicting_rows_removed": 3,
                    "primary_rows_after_annotation_exclusions": 43,
                    "primary_source_images_after_annotation_exclusions": 43,
                },
            )
            self.assertEqual(manifest["reservation"]["source_img_ids"], ["source-03"])
            self.assertEqual(manifest["reservation"]["rows"], 1)

            data_root = root / "data/processed/study1"
            partitions = {
                name: read_jsonl(data_root / f"{name}.jsonl")
                for name in ("train", "development", "test")
            }
            source_sets = {
                name: {source_image_id(record) for record in records}
                for name, records in partitions.items()
            }
            self.assertFalse(source_sets["train"] & source_sets["development"])
            self.assertFalse(source_sets["train"] & source_sets["test"])
            self.assertFalse(source_sets["development"] & source_sets["test"])
            self.assertNotIn(
                "source-03",
                set().union(*source_sets.values()),
            )

            smoke = read_jsonl(data_root / "smoke_20.jsonl")
            self.assertEqual(len(smoke), 5)
            self.assertEqual(
                len({source_image_id(record) for record in smoke}),
                5,
            )
            development_ids = {
                stable_item_id(record)
                for record in partitions["development"]
            }
            self.assertTrue(
                {stable_item_id(record) for record in smoke}
                <= development_ids
            )
            verification = verify_grouped_split_artifacts(
                manifest_path=manifest_path,
                project_root=root,
            )
            self.assertEqual(verification["status"], "PASS")

    def test_build_is_reproducible_across_clean_roots(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                first_root = Path(first_directory)
                second_root = Path(second_directory)
                self.build(first_root)
                self.build(second_root)
                first_manifest = json.loads(
                    (
                        first_root
                        / "protocols/study1/grouped_split_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                second_manifest = json.loads(
                    (
                        second_root
                        / "protocols/study1/grouped_split_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(first_manifest, second_manifest)

    def test_verification_rejects_tampered_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root)
            train_path = root / "data/processed/study1/train.jsonl"
            train_path.write_text(
                train_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SplitBuildError,
                "artifact hash mismatch",
            ):
                verify_grouped_split_artifacts(
                    manifest_path=(
                        root
                        / "protocols/study1/grouped_split_manifest.json"
                    ),
                    project_root=root,
                )

    def test_builder_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root)
            with self.assertRaisesRegex(
                FileExistsError,
                "refusing to overwrite",
            ):
                self.build(root)

    def test_materialize_reconstructs_and_reuses_tracked_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root)
            manifest_path = (
                root / "protocols/study1/grouped_split_manifest.json"
            )
            locked_bytes = manifest_path.read_bytes()
            shutil.rmtree(root / "data/processed/study1")
            train, test = self.source_records()

            def loader(
                dataset_id: str,
                revision: str,
            ) -> tuple[list[dict], list[dict]]:
                self.assertEqual(dataset_id, "example/dataset")
                self.assertEqual(revision, "d" * 40)
                return train, test

            result = materialize_grouped_split_artifacts(
                manifest_path=manifest_path,
                project_root=root,
                official_loader=loader,
            )
            self.assertTrue(result["materialized"])
            self.assertEqual(result["reused_artifacts"], 0)
            self.assertEqual(manifest_path.read_bytes(), locked_bytes)

            reused = materialize_grouped_split_artifacts(
                manifest_path=manifest_path,
                project_root=root,
                official_loader=lambda *_args: self.fail(
                    "reused artifacts must not reload the dataset"
                ),
            )
            self.assertFalse(reused["materialized"])
            self.assertEqual(reused["reused_artifacts"], 9)

    def test_smoke_selection_requires_unique_sources(self) -> None:
        records = [
            {
                "messages": [
                    {"role": "user", "content": f"<image>Question {index}"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "metadata": {
                    "source_img_id": "one-source",
                    "complexity": 1,
                    "question_class": ["location"],
                },
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(
            SplitBuildError,
            "requires 2 source images",
        ):
            select_smoke_records(records, count=2, seed=42)


if __name__ == "__main__":
    unittest.main()
