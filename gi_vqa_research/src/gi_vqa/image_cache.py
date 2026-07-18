"""Pinned, auditable image caching for Study 1 execution gates.

The grouped split artifacts contain portable paths but intentionally do not
materialise image bytes. This module downloads selected JPEGs from the exact
Kvasir-VQA-x1 dataset revision, publishes them atomically, and records both
encoded-file and canonical RGB fingerprints in a compact tracked manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256, file_sha256
from .splits import GROUPED_SPLIT_SCHEMA_VERSION, verify_grouped_split_artifacts

IMAGE_CACHE_SCHEMA_VERSION = "gi-vqa-image-cache-manifest-v1"
IMAGE_CACHE_SELECTION_ALGORITHM = "sha256-training-source-selection-v1"
IMAGE_FILENAME_TEMPLATE = "images/{source_img_id}.jpg"
_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

RevisionResolver = Callable[[str, str], str]
FileFetcher = Callable[[str, str, str], Path]


class ImageCacheError(ValueError):
    """Raised when image materialisation or verification is unsafe."""


def prepare_image_cache(
    *,
    project_root: str | Path,
    split_manifest_path: str | Path,
    cache_manifest_path: str | Path,
    image_dir: str | Path,
    training_source_images: int,
    token: str | None = None,
    resolve_revision: RevisionResolver | None = None,
    fetch_file: FileFetcher | None = None,
) -> dict[str, Any]:
    """Create a locked smoke/training image cache manifest.

    The split hard gate is re-run first. Partial image downloads are safe to
    reuse, but an existing manifest is never overwritten.
    """

    root = Path(project_root).resolve()
    split_path = _resolve_under(root, split_manifest_path)
    cache_manifest = _resolve_under(root, cache_manifest_path)
    cache_dir = _resolve_under(root, image_dir)
    if cache_manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite image cache manifest: {cache_manifest}"
        )
    if (
        not isinstance(training_source_images, int)
        or isinstance(training_source_images, bool)
        or training_source_images < 1
    ):
        raise ImageCacheError("training_source_images must be a positive integer")

    verify_grouped_split_artifacts(
        manifest_path=split_path,
        project_root=root,
    )
    split_manifest = _read_json_object(split_path)
    _validate_split_manifest(split_manifest)

    train_sources = _select_training_sources(
        split_manifest,
        count=training_source_images,
    )
    smoke_sources = _required_string_list(
        split_manifest.get("smoke", {}).get("source_img_ids"),
        name="split manifest smoke.source_img_ids",
    )
    if len(smoke_sources) != len(set(smoke_sources)):
        raise ImageCacheError("split manifest smoke sources are not unique")
    overlap = sorted(set(train_sources) & set(smoke_sources))
    if overlap:
        raise ImageCacheError(
            f"training and development cache selections overlap: {overlap}"
        )

    dataset = _required_mapping(split_manifest.get("dataset"), name="dataset")
    image_dataset = _required_mapping(
        split_manifest.get("image_dataset"),
        name="image_dataset",
    )
    artifact_repo_id = _required_string(dataset.get("id"), name="dataset.id")
    artifact_revision = _required_string(
        dataset.get("revision"),
        name="dataset.revision",
    )
    canonical_repo_id = _required_string(
        image_dataset.get("id"),
        name="image_dataset.id",
    )
    canonical_revision = _required_string(
        image_dataset.get("revision"),
        name="image_dataset.revision",
    )

    resolver = resolve_revision or _hugging_face_revision_resolver(token=token)
    fetcher = fetch_file or _hugging_face_file_fetcher(token=token)
    resolved_artifact_revision = resolver(artifact_repo_id, artifact_revision)
    resolved_canonical_revision = resolver(canonical_repo_id, canonical_revision)
    if resolved_artifact_revision != artifact_revision:
        raise ImageCacheError(
            "artifact dataset revision did not resolve exactly: "
            f"expected {artifact_revision}, observed {resolved_artifact_revision}"
        )
    if resolved_canonical_revision != canonical_revision:
        raise ImageCacheError(
            "canonical image dataset revision did not resolve exactly: "
            f"expected {canonical_revision}, observed {resolved_canonical_revision}"
        )

    scopes = {
        **{source_id: "development_smoke" for source_id in smoke_sources},
        **{source_id: "training_loader_smoke" for source_id in train_sources},
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict[str, Any]] = {}
    for source_id in sorted(scopes):
        _validate_source_id(source_id)
        repo_filename = IMAGE_FILENAME_TEMPLATE.format(source_img_id=source_id)
        fetched_path = Path(
            fetcher(artifact_repo_id, artifact_revision, repo_filename)
        )
        if not fetched_path.is_file():
            raise FileNotFoundError(
                f"image source fetch returned no file for {source_id}: {fetched_path}"
            )
        source_descriptor = _inspect_image(fetched_path)
        if source_descriptor["format"] != "JPEG":
            raise ImageCacheError(
                f"expected JPEG for {source_id}, observed "
                f"{source_descriptor['format']!r}"
            )
        destination = cache_dir / f"{source_id}.jpg"
        _publish_verified_file(
            source=fetched_path,
            destination=destination,
            expected_sha256=source_descriptor["sha256"],
        )
        local_descriptor = _inspect_image(destination)
        if local_descriptor != source_descriptor:
            raise ImageCacheError(
                f"published image differs from pinned source for {source_id}"
            )
        images[source_id] = {
            "scope": scopes[source_id],
            "path": _portable_path(destination, project_root=root),
            "repo_filename": repo_filename,
            **source_descriptor,
        }

    manifest = {
        "schema_version": IMAGE_CACHE_SCHEMA_VERSION,
        "status": "PASS",
        "study": "study1",
        "split_manifest": {
            "path": _portable_path(split_path, project_root=root),
            "sha256": file_sha256(split_path),
            "schema_version": split_manifest["schema_version"],
        },
        "artifact_source": {
            "id": artifact_repo_id,
            "requested_revision": artifact_revision,
            "resolved_revision": resolved_artifact_revision,
            "repo_type": "dataset",
            "filename_template": IMAGE_FILENAME_TEMPLATE,
            "reason": (
                "Kvasir-VQA-x1 exposes the pinned source JPEGs as individually "
                "addressable files; the canonical HOST dataset remains recorded below"
            ),
        },
        "canonical_image_dataset": {
            "id": canonical_repo_id,
            "requested_revision": canonical_revision,
            "resolved_revision": resolved_canonical_revision,
        },
        "selection": {
            "algorithm": IMAGE_CACHE_SELECTION_ALGORITHM,
            "seed": split_manifest["algorithm"]["seed"],
            "training_source_images": len(train_sources),
            "development_smoke_source_images": len(smoke_sources),
            "training_source_img_ids": train_sources,
            "development_smoke_source_img_ids": smoke_sources,
        },
        "cache": {
            "root": _portable_path(cache_dir, project_root=root),
            "image_count": len(images),
            "encoded_bytes": sum(
                int(descriptor["bytes"]) for descriptor in images.values()
            ),
            "source_img_ids_sha256": canonical_json_sha256(sorted(images)),
        },
        "images": images,
    }
    _validate_cache_manifest_payload(
        manifest,
        project_root=root,
        require_files=True,
    )
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cache_manifest, manifest)
    return verify_image_cache(
        manifest_path=cache_manifest,
        project_root=root,
    )


def materialize_image_cache(
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    token: str | None = None,
    resolve_revision: RevisionResolver | None = None,
    fetch_file: FileFetcher | None = None,
) -> dict[str, Any]:
    """Populate missing/corrupt local files from an existing locked manifest."""

    root = Path(project_root).resolve()
    manifest_file = _resolve_under(root, manifest_path)
    manifest = _read_json_object(manifest_file)
    _validate_cache_manifest_payload(
        manifest,
        project_root=root,
        require_files=False,
    )

    artifact_source = manifest["artifact_source"]
    canonical_source = manifest["canonical_image_dataset"]
    resolver = resolve_revision or _hugging_face_revision_resolver(token=token)
    fetcher = fetch_file or _hugging_face_file_fetcher(token=token)
    for descriptor in (artifact_source, canonical_source):
        observed = resolver(
            str(descriptor["id"]),
            str(descriptor["requested_revision"]),
        )
        if observed != descriptor["resolved_revision"]:
            raise ImageCacheError(
                f"dataset revision resolution changed for {descriptor['id']}: "
                f"expected {descriptor['resolved_revision']}, observed {observed}"
            )

    repaired = 0
    reused = 0
    for source_id, descriptor in sorted(manifest["images"].items()):
        destination = _resolve_under(root, descriptor["path"])
        if _file_matches_descriptor(destination, descriptor):
            reused += 1
            continue
        fetched_path = Path(
            fetcher(
                str(artifact_source["id"]),
                str(artifact_source["requested_revision"]),
                str(descriptor["repo_filename"]),
            )
        )
        fetched_descriptor = _inspect_image(fetched_path)
        _require_image_descriptor_match(
            source_id,
            observed=fetched_descriptor,
            expected=descriptor,
        )
        _publish_verified_file(
            source=fetched_path,
            destination=destination,
            expected_sha256=str(descriptor["sha256"]),
        )
        repaired += 1

    result = verify_image_cache(
        manifest_path=manifest_file,
        project_root=root,
    )
    result["reused_images"] = reused
    result["materialized_images"] = repaired
    return result


def verify_image_cache(
    *,
    manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verify the locked selection, source partitions and every local image."""

    root = Path(project_root).resolve()
    manifest_file = _resolve_under(root, manifest_path)
    manifest = _read_json_object(manifest_file)
    summary = _validate_cache_manifest_payload(
        manifest,
        project_root=root,
        require_files=True,
    )
    return {
        "status": "PASS",
        "manifest": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        **summary,
    }


def _validate_cache_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
    require_files: bool,
) -> dict[str, Any]:
    if manifest.get("schema_version") != IMAGE_CACHE_SCHEMA_VERSION:
        raise ImageCacheError(
            f"unsupported image cache schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "PASS":
        raise ImageCacheError("image cache manifest status is not PASS")

    split_descriptor = _required_mapping(
        manifest.get("split_manifest"),
        name="split_manifest",
    )
    split_path = _resolve_under(
        project_root,
        _required_string(split_descriptor.get("path"), name="split_manifest.path"),
    )
    if not split_path.is_file():
        raise FileNotFoundError(f"grouped split manifest is missing: {split_path}")
    observed_split_sha = file_sha256(split_path)
    if observed_split_sha != split_descriptor.get("sha256"):
        raise ImageCacheError(
            "grouped split manifest hash differs from the image cache lock"
        )
    split_manifest = _read_json_object(split_path)
    _validate_split_manifest(split_manifest)
    if split_descriptor.get("schema_version") != split_manifest["schema_version"]:
        raise ImageCacheError("grouped split schema differs from the image cache lock")

    artifact_source = _required_mapping(
        manifest.get("artifact_source"),
        name="artifact_source",
    )
    canonical_source = _required_mapping(
        manifest.get("canonical_image_dataset"),
        name="canonical_image_dataset",
    )
    dataset = split_manifest["dataset"]
    image_dataset = split_manifest["image_dataset"]
    expected_artifact_identity = (dataset["id"], dataset["revision"])
    observed_artifact_identity = (
        artifact_source.get("id"),
        artifact_source.get("requested_revision"),
    )
    if observed_artifact_identity != expected_artifact_identity:
        raise ImageCacheError(
            "image artifact identity differs from the grouped split dataset"
        )
    expected_canonical_identity = (
        image_dataset["id"],
        image_dataset["revision"],
    )
    observed_canonical_identity = (
        canonical_source.get("id"),
        canonical_source.get("requested_revision"),
    )
    if observed_canonical_identity != expected_canonical_identity:
        raise ImageCacheError(
            "canonical image identity differs from the grouped split manifest"
        )
    for name, descriptor in (
        ("artifact_source", artifact_source),
        ("canonical_image_dataset", canonical_source),
    ):
        if descriptor.get("resolved_revision") != descriptor.get(
            "requested_revision"
        ):
            raise ImageCacheError(f"{name} did not resolve to its requested revision")

    selection = _required_mapping(manifest.get("selection"), name="selection")
    if selection.get("algorithm") != IMAGE_CACHE_SELECTION_ALGORITHM:
        raise ImageCacheError("unsupported image cache selection algorithm")
    smoke_sources = _required_string_list(
        selection.get("development_smoke_source_img_ids"),
        name="development smoke source IDs",
    )
    training_sources = _required_string_list(
        selection.get("training_source_img_ids"),
        name="training source IDs",
    )
    expected_smoke = _required_string_list(
        split_manifest["smoke"].get("source_img_ids"),
        name="grouped split smoke source IDs",
    )
    if smoke_sources != expected_smoke:
        raise ImageCacheError("image cache smoke selection differs from grouped split")
    expected_training = _select_training_sources(
        split_manifest,
        count=len(training_sources),
    )
    if training_sources != expected_training:
        raise ImageCacheError("image cache training selection is not deterministic")
    if selection.get("training_source_images") != len(training_sources):
        raise ImageCacheError("training image count differs from its selection")
    if selection.get("development_smoke_source_images") != len(smoke_sources):
        raise ImageCacheError("development image count differs from its selection")
    if set(training_sources) & set(smoke_sources):
        raise ImageCacheError("training and development image selections overlap")

    images = _required_mapping(manifest.get("images"), name="images")
    cache = _required_mapping(manifest.get("cache"), name="cache")
    cache_root = _resolve_under(
        project_root,
        _required_string(cache.get("root"), name="cache.root"),
    )
    expected_sources = set(training_sources) | set(smoke_sources)
    if set(images) != expected_sources:
        raise ImageCacheError("image entries differ from the locked source selection")
    expected_scopes = {
        **{source_id: "development_smoke" for source_id in smoke_sources},
        **{source_id: "training_loader_smoke" for source_id in training_sources},
    }
    encoded_bytes = 0
    for source_id, raw_descriptor in images.items():
        _validate_source_id(str(source_id))
        descriptor = _required_mapping(
            raw_descriptor,
            name=f"images.{source_id}",
        )
        if descriptor.get("scope") != expected_scopes[source_id]:
            raise ImageCacheError(f"incorrect cache scope for {source_id}")
        expected_repo_filename = IMAGE_FILENAME_TEMPLATE.format(
            source_img_id=source_id
        )
        if descriptor.get("repo_filename") != expected_repo_filename:
            raise ImageCacheError(f"incorrect repository filename for {source_id}")
        expected_path = cache_root / f"{source_id}.jpg"
        observed_path = _resolve_under(
            project_root,
            _required_string(
                descriptor.get("path"),
                name=f"images.{source_id}.path",
            ),
        )
        if observed_path != expected_path:
            raise ImageCacheError(f"incorrect local cache path for {source_id}")
        encoded_bytes += _required_positive_int(
            descriptor.get("bytes"),
            name=f"images.{source_id}.bytes",
        )
        if require_files:
            if not observed_path.is_file():
                raise FileNotFoundError(
                    f"cached image is missing for {source_id}: {observed_path}"
                )
            observed = _inspect_image(observed_path)
            _require_image_descriptor_match(
                source_id,
                observed=observed,
                expected=descriptor,
            )

    if cache.get("image_count") != len(images):
        raise ImageCacheError("cache image count differs from image entries")
    if cache.get("encoded_bytes") != encoded_bytes:
        raise ImageCacheError("cache encoded byte count differs from image entries")
    if cache.get("source_img_ids_sha256") != canonical_json_sha256(
        sorted(images)
    ):
        raise ImageCacheError("cache source ID fingerprint differs")
    return {
        "image_count": len(images),
        "encoded_bytes": encoded_bytes,
        "development_smoke_images": len(smoke_sources),
        "training_loader_smoke_images": len(training_sources),
        "split_manifest_sha256": observed_split_sha,
    }


def _select_training_sources(
    split_manifest: Mapping[str, Any],
    *,
    count: int,
) -> list[str]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ImageCacheError("training source count must be a positive integer")
    partitions = _required_mapping(
        split_manifest.get("partitions"),
        name="partitions",
    )
    train = _required_mapping(partitions.get("train"), name="partitions.train")
    candidates = _required_string_list(
        train.get("source_img_ids"),
        name="training source IDs",
    )
    if count > len(candidates):
        raise ImageCacheError(
            f"requested {count} training source images but only "
            f"{len(candidates)} are available"
        )
    algorithm = _required_mapping(
        split_manifest.get("algorithm"),
        name="algorithm",
    )
    seed = algorithm.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ImageCacheError("grouped split algorithm seed must be an integer")

    def selection_key(source_id: str) -> tuple[str, str]:
        payload = (
            f"{IMAGE_CACHE_SELECTION_ALGORITHM}\0{seed}\0{source_id}"
        ).encode()
        return hashlib.sha256(payload).hexdigest(), source_id

    return sorted(candidates, key=selection_key)[:count]


def _publish_verified_file(
    *,
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        return
    temporary_path: Path | None = None
    try:
        with source.open("rb") as source_handle:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if file_sha256(temporary_path) != expected_sha256:
            raise ImageCacheError(
                f"atomic image copy hash mismatch for {destination.name}"
            )
        _inspect_image(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _inspect_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImageCacheError("Pillow is required for image verification") from exc
    try:
        with Image.open(path) as opened:
            opened.load()
            image_format = opened.format
            source_mode = opened.mode
            width, height = opened.size
            rgb = opened.convert("RGB")
            rgb_digest = hashlib.sha256()
            rgb_digest.update(f"{width}x{height}\0RGB\0".encode("ascii"))
            rgb_digest.update(rgb.tobytes())
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageCacheError(f"image cannot be decoded: {path}") from exc
    if width < 1 or height < 1:
        raise ImageCacheError(f"image has invalid dimensions: {path}")
    return {
        "sha256": file_sha256(path),
        "rgb_sha256": rgb_digest.hexdigest(),
        "bytes": path.stat().st_size,
        "format": image_format,
        "mode": source_mode,
        "width": int(width),
        "height": int(height),
    }


def _file_matches_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        observed = _inspect_image(path)
        _require_image_descriptor_match(
            path.stem,
            observed=observed,
            expected=descriptor,
        )
    except (ImageCacheError, OSError):
        return False
    return True


def _require_image_descriptor_match(
    source_id: str,
    *,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "sha256",
        "rgb_sha256",
        "bytes",
        "format",
        "mode",
        "width",
        "height",
    )
    differences = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in fields
        if expected.get(field) != observed.get(field)
    }
    if differences:
        raise ImageCacheError(
            f"cached image descriptor mismatch for {source_id}: {differences}"
        )


def _validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != GROUPED_SPLIT_SCHEMA_VERSION:
        raise ImageCacheError("image caching requires the grouped split v1 manifest")
    if manifest.get("status") != "PASS":
        raise ImageCacheError("grouped split manifest status is not PASS")
    _required_mapping(manifest.get("dataset"), name="dataset")
    _required_mapping(manifest.get("image_dataset"), name="image_dataset")
    _required_mapping(manifest.get("algorithm"), name="algorithm")
    _required_mapping(manifest.get("partitions"), name="partitions")
    _required_mapping(manifest.get("smoke"), name="smoke")


def _hugging_face_revision_resolver(
    *,
    token: str | None,
) -> RevisionResolver:
    def resolve(repo_id: str, revision: str) -> str:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImageCacheError(
                "image caching requires the data extra: pip install -e '.[data]'"
            ) from exc
        resolved = HfApi().dataset_info(
            repo_id=repo_id,
            revision=revision,
            token=token,
        ).sha
        if not isinstance(resolved, str) or not resolved:
            raise ImageCacheError(f"dataset revision did not resolve for {repo_id}")
        return resolved

    return resolve


def _hugging_face_file_fetcher(
    *,
    token: str | None,
) -> FileFetcher:
    def fetch(repo_id: str, revision: str, filename: str) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImageCacheError(
                "image caching requires the data extra: pip install -e '.[data]'"
            ) from exc
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                token=token,
            )
        )

    return fetch


def _validate_source_id(source_id: str) -> None:
    if not source_id or not _SAFE_SOURCE_ID.fullmatch(source_id):
        raise ImageCacheError(f"unsafe source image ID: {source_id!r}")


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImageCacheError(f"{name} must be a mapping")
    return value


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageCacheError(f"{name} must be a non-empty string")
    return value


def _required_string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ImageCacheError(f"{name} must be a list")
    result = []
    for index, item in enumerate(value):
        result.append(_required_string(item, name=f"{name}[{index}]"))
    if len(result) != len(set(result)):
        raise ImageCacheError(f"{name} contains duplicates")
    return result


def _required_positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ImageCacheError(f"{name} must be a positive integer")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ImageCacheError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ImageCacheError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _portable_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ImageCacheError(
            f"path must be inside the project root: {resolved}"
        ) from exc


def _resolve_under(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ImageCacheError(
            f"path escapes the project root: {path}"
        ) from exc
    return resolved


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
