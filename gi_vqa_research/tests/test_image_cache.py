from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gi_vqa.image_cache import (
    ImageCacheError,
    materialize_image_cache,
    prepare_image_cache,
    verify_image_cache,
)
from gi_vqa.splits import SplitBuildPaths, build_grouped_splits


class ImageCacheTests(unittest.TestCase):
    def _build_fixture(
        self,
        root: Path,
    ) -> tuple[Path, Path, dict[str, Path]]:
        records = []
        remote_images = root / "remote-images"
        remote_images.mkdir(parents=True)
        image_paths: dict[str, Path] = {}
        for index in range(12):
            source_id = f"source-{index:02d}"
            records.append(
                {
                    "img_id": source_id,
                    "question": f"Question {index}?",
                    "answer": f"Answer {index}",
                    "complexity": (index % 3) + 1,
                    "question_class": [f"class-{index % 4}"],
                    "original": True,
                }
            )
            image_path = remote_images / f"{source_id}.jpg"
            Image.new(
                "RGB",
                (24 + index, 20 + index),
                color=(index * 10, 20, 30),
            ).save(image_path, format="JPEG")
            image_paths[source_id] = image_path

        split_manifest = root / "protocols" / "grouped_split_manifest.json"
        build_grouped_splits(
            official_train_records=records,
            official_test_records=[],
            dataset_id="example/qa",
            dataset_revision="d" * 40,
            image_dataset_id="example/canonical-images",
            image_dataset_revision="i" * 40,
            seed=42,
            development_fraction=0.2,
            test_fraction=0.2,
            smoke_items=2,
            paths=SplitBuildPaths.resolve(
                project_root=root,
                data_root="data/processed/study1",
                manifest_path=split_manifest,
                image_dir="data/images",
            ),
            reserved_source_ids=("source-00",),
        )
        return (
            split_manifest,
            root / "protocols" / "image_cache_manifest.json",
            image_paths,
        )

    @staticmethod
    def _resolver(_repo_id: str, revision: str) -> str:
        return revision

    def test_prepare_verify_and_materialize_are_restart_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_manifest, cache_manifest, image_paths = self._build_fixture(
                root
            )
            fetch_calls: list[str] = []

            def fetch(
                _repo_id: str,
                _revision: str,
                filename: str,
            ) -> Path:
                source_id = Path(filename).stem
                fetch_calls.append(source_id)
                return image_paths[source_id]

            result = prepare_image_cache(
                project_root=root,
                split_manifest_path=split_manifest,
                cache_manifest_path=cache_manifest,
                image_dir="data/images",
                training_source_images=2,
                resolve_revision=self._resolver,
                fetch_file=fetch,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["image_count"], 4)
            self.assertEqual(result["development_smoke_images"], 2)
            self.assertEqual(result["training_loader_smoke_images"], 2)
            self.assertEqual(len(fetch_calls), 4)

            manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifact_source"]["requested_revision"],
                "d" * 40,
            )
            self.assertEqual(
                manifest["canonical_image_dataset"]["requested_revision"],
                "i" * 40,
            )
            self.assertEqual(
                set(manifest["images"]),
                set(
                    manifest["selection"]["training_source_img_ids"]
                    + manifest["selection"][
                        "development_smoke_source_img_ids"
                    ]
                ),
            )
            for descriptor in manifest["images"].values():
                self.assertEqual(descriptor["format"], "JPEG")
                self.assertEqual(len(descriptor["sha256"]), 64)
                self.assertEqual(len(descriptor["rgb_sha256"]), 64)

            with self.assertRaises(FileExistsError):
                prepare_image_cache(
                    project_root=root,
                    split_manifest_path=split_manifest,
                    cache_manifest_path=cache_manifest,
                    image_dir="data/images",
                    training_source_images=2,
                    resolve_revision=self._resolver,
                    fetch_file=fetch,
                )

            corrupt_source = next(iter(manifest["images"]))
            corrupt_path = root / manifest["images"][corrupt_source]["path"]
            corrupt_path.write_bytes(b"not an image")
            with self.assertRaises(ImageCacheError):
                verify_image_cache(
                    manifest_path=cache_manifest,
                    project_root=root,
                )

            fetch_calls.clear()
            repaired = materialize_image_cache(
                manifest_path=cache_manifest,
                project_root=root,
                resolve_revision=self._resolver,
                fetch_file=fetch,
            )
            self.assertEqual(repaired["status"], "PASS")
            self.assertEqual(repaired["materialized_images"], 1)
            self.assertEqual(repaired["reused_images"], 3)
            self.assertEqual(fetch_calls, [corrupt_source])

    def test_split_manifest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_manifest, cache_manifest, image_paths = self._build_fixture(
                root
            )

            def fetch(
                _repo_id: str,
                _revision: str,
                filename: str,
            ) -> Path:
                return image_paths[Path(filename).stem]

            prepare_image_cache(
                project_root=root,
                split_manifest_path=split_manifest,
                cache_manifest_path=cache_manifest,
                image_dir="data/images",
                training_source_images=2,
                resolve_revision=self._resolver,
                fetch_file=fetch,
            )
            split_manifest.write_text(
                split_manifest.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ImageCacheError,
                "grouped split manifest hash differs",
            ):
                verify_image_cache(
                    manifest_path=cache_manifest,
                    project_root=root,
                )

    def test_unsafe_source_identifier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_manifest, cache_manifest, image_paths = self._build_fixture(
                root
            )
            split_payload = json.loads(
                split_manifest.read_text(encoding="utf-8")
            )
            split_payload["smoke"]["source_img_ids"][0] = "../escape"
            split_manifest.write_text(
                json.dumps(split_payload),
                encoding="utf-8",
            )

            def fetch(
                _repo_id: str,
                _revision: str,
                filename: str,
            ) -> Path:
                return image_paths[Path(filename).stem]

            with self.assertRaises(ValueError):
                prepare_image_cache(
                    project_root=root,
                    split_manifest_path=split_manifest,
                    cache_manifest_path=cache_manifest,
                    image_dir="data/images",
                    training_source_images=2,
                    resolve_revision=self._resolver,
                    fetch_file=fetch,
                )


if __name__ == "__main__":
    unittest.main()
