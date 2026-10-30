"""ZIP export and extraction tests for packages (offline)."""

from __future__ import annotations

import json
import os
import shlex
import stat
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from kbforge.cli import build_parser, main
from kbforge.package import (
    BUILD_MARKER_FILENAME,
    BUILD_STATS_FILENAME,
    MANIFEST_FILENAME,
    README_FILENAME,
    IncompleteBuildError,
    KnowledgeBase,
    MissingArtifactError,
    PackageError,
    build_package,
    export_package,
)

REQUIRED_ARCHIVE_FILES = {
    "manifest.json",
    "build_config.json",
    "build_stats.json",
    "chunks.jsonl",
    "index/postings.json",
    "index/doc_lengths.json",
    "index/chunk_info.json",
    "index/df.json",
    "index/index_stats.json",
}

PRESENT_OPTIONAL_ARCHIVE_FILES = {
    "documents.jsonl",
    "documents_cleaned.jsonl",
    "crawl_report.json",
    "phase2_stats.json",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _archive_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _assert_safe_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            assert not relative.is_absolute()
            assert relative.parts
            assert ".." not in relative.parts
            assert "\\" not in name
            assert not stat.S_ISLNK(info.external_attr >> 16)


def _results(kb: KnowledgeBase, mode: str):
    return [
        (result.chunk_id, result.score, result.rank, result.source, result.title, result.text)
        for result in kb.search("python", mode=mode, limit=5)
    ]


def test_export_includes_manifest_required_and_present_optional_artifacts(
    sample_kb, tmp_path
):
    config, output_dir = sample_kb
    build_package(config)
    pages = output_dir / "pages"
    pages.mkdir()
    (pages / "page.html").write_text("<html>not packaged</html>", encoding="utf-8")
    (output_dir / "unlisted.txt").write_text("not packaged", encoding="utf-8")

    archive_path = tmp_path / "delivery" / "client.zip"
    summary = export_package(output_dir, archive_path)

    expected = REQUIRED_ARCHIVE_FILES | PRESENT_OPTIONAL_ARCHIVE_FILES | {
        README_FILENAME
    }
    assert archive_path.is_file()
    assert summary == {
        "archive_path": str(archive_path),
        "archive_size": archive_path.stat().st_size,
        "file_count": len(expected),
    }
    assert _archive_names(archive_path) == expected
    assert "pages/page.html" not in _archive_names(archive_path)
    assert "unlisted.txt" not in _archive_names(archive_path)
    assert not (output_dir / README_FILENAME).exists()
    _assert_safe_archive(archive_path)


def test_exported_readme_contains_metadata_and_valid_commands(
    sample_kb, tmp_path, monkeypatch
):
    config, output_dir = sample_kb
    build_package(config)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_json(manifest_path)
    build_stats = _read_json(output_dir / BUILD_STATS_FILENAME)
    archive_path = tmp_path / "client.zip"
    export_package(output_dir, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        readme = archive.read(README_FILENAME).decode("utf-8")

    assert readme.startswith("# KBForge Knowledge Base\n")
    assert f"- Documents: {build_stats['documents_cleaned']}" in readme
    assert f"- Chunks: {manifest['statistics']['chunks']}" in readme
    assert (
        "- Supported ranking modes: " + ", ".join(manifest["ranking_modes"])
    ) in readme
    assert "- `documents.jsonl`" in readme
    assert "- `index`\n" not in readme
    assert str(output_dir) not in readme
    assert manifest_path.read_bytes() == manifest_bytes
    assert README_FILENAME not in manifest["artifacts"]

    query_commands = [
        line.strip()
        for line in readme.splitlines()
        if line.strip().startswith("kbforge query ")
    ]
    assert len(query_commands) == 1 + len(manifest["ranking_modes"])
    for command in query_commands:
        parsed = build_parser().parse_args(shlex.split(command)[1:])
        assert parsed.command == "query"
        assert parsed.package == "."
        assert parsed.mode in manifest["ranking_modes"]

    serve_commands = [
        line.strip()
        for line in readme.splitlines()
        if line.strip().startswith("kbforge serve ")
    ]
    assert serve_commands
    for command in serve_commands:
        parsed = build_parser().parse_args(shlex.split(command)[1:])
        assert parsed.command == "serve"
        assert parsed.package == "."

    assert "GET /health" in readme
    assert "POST /search" in readme
    assert "HTTP 503" in readme

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    monkeypatch.chdir(extract_dir)
    for mode in manifest["ranking_modes"]:
        assert main(["query", "python", "--package", ".", "--mode", mode, "--k", "1"]) == 0


def test_extracted_archive_loads_and_matches_original_searches(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    original = KnowledgeBase.load(output_dir)
    archive_path = tmp_path / "client.zip"
    export_package(output_dir, archive_path)

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    extracted = KnowledgeBase.load(extract_dir)
    assert extracted.chunk_texts == original.chunk_texts
    assert all(result[5] for result in _results(extracted, "bm25"))
    for mode in ("bm25", "tfidf"):
        assert _results(extracted, mode) == _results(original, mode)


def test_package_with_only_required_artifacts_can_be_exported_and_searched(
    sample_kb, tmp_path
):
    config, output_dir = sample_kb
    for name in PRESENT_OPTIONAL_ARCHIVE_FILES:
        (output_dir / name).unlink()
    build_package(config)

    archive_path = tmp_path / "required-only.zip"
    export_package(output_dir, archive_path)

    assert _archive_names(archive_path) == REQUIRED_ARCHIVE_FILES | {README_FILENAME}
    extract_dir = tmp_path / "required-only"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    kb = KnowledgeBase.load(extract_dir)
    assert kb.search("python", mode="bm25")
    assert kb.search("python", mode="tfidf")


def test_export_missing_artifact_fails_without_partial_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "chunks.jsonl").unlink()
    archive_path = tmp_path / "client.zip"

    with pytest.raises(MissingArtifactError, match="chunks"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_corrupt_artifact_fails_without_partial_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "index" / "postings.json").write_text("{bad json", encoding="utf-8")
    archive_path = tmp_path / "client.zip"

    with pytest.raises(PackageError, match="postings.json"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_corrupt_optional_artifact_fails_without_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "crawl_report.json").write_text("{bad json", encoding="utf-8")
    archive_path = tmp_path / "client.zip"

    with pytest.raises(PackageError, match="optional artifact"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_incomplete_build_fails_without_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / BUILD_MARKER_FILENAME).write_text("incomplete\n", encoding="utf-8")
    archive_path = tmp_path / "client.zip"

    with pytest.raises(IncompleteBuildError, match="incomplete"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_failure_preserves_existing_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "index" / "df.json").write_text("{bad json", encoding="utf-8")
    archive_path = tmp_path / "client.zip"
    archive_path.write_bytes(b"existing archive")

    with pytest.raises(PackageError):
        export_package(output_dir, archive_path, overwrite=True)

    assert archive_path.read_bytes() == b"existing archive"
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_write_failure_cleans_temp_and_preserves_destination(
    sample_kb, tmp_path, monkeypatch
):
    config, output_dir = sample_kb
    build_package(config)
    archive_path = tmp_path / "client.zip"
    archive_path.write_bytes(b"existing archive")
    original_write = zipfile.ZipFile.write
    calls = 0

    def failing_write(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", failing_write)

    with pytest.raises(PackageError, match="could not write archive"):
        export_package(output_dir, archive_path, overwrite=True)

    assert archive_path.read_bytes() == b"existing archive"
    assert not list(tmp_path.glob(".client.zip.*.tmp"))


def test_export_requires_overwrite_for_existing_archive(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    archive_path = tmp_path / "client.zip"
    archive_path.write_bytes(b"existing archive")

    with pytest.raises(PackageError, match="overwrite"):
        export_package(output_dir, archive_path)
    assert archive_path.read_bytes() == b"existing archive"

    summary = export_package(output_dir, archive_path, overwrite=True)
    assert summary["file_count"] == len(
        REQUIRED_ARCHIVE_FILES | PRESENT_OPTIONAL_ARCHIVE_FILES | {README_FILENAME}
    )
    assert zipfile.is_zipfile(archive_path)
    _assert_safe_archive(archive_path)


@pytest.mark.parametrize("unsafe_path", ["traversal", "absolute"])
def test_export_rejects_unsafe_artifact_paths(sample_kb, tmp_path, unsafe_path):
    config, output_dir = sample_kb
    build_package(config)
    outside = tmp_path / "outside-chunks.jsonl"
    outside.write_bytes((output_dir / "chunks.jsonl").read_bytes())

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    if unsafe_path == "traversal":
        manifest["artifacts"]["chunks"]["path"] = "../outside-chunks.jsonl"
    else:
        manifest["artifacts"]["chunks"]["path"] = str(outside)
    _write_json(manifest_path, manifest)
    archive_path = tmp_path / "client.zip"

    with pytest.raises(PackageError, match="unsafe artifact path"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()


def test_export_rejects_symlink_artifact(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    outside = tmp_path / "outside-chunks.jsonl"
    outside.write_bytes((output_dir / "chunks.jsonl").read_bytes())
    chunks_path = output_dir / "chunks.jsonl"
    chunks_path.unlink()
    os.symlink(outside, chunks_path)
    archive_path = tmp_path / "client.zip"

    with pytest.raises(PackageError, match="symlink"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()


def test_export_rejects_symlink_inside_directory_artifact(sample_kb, tmp_path):
    config, output_dir = sample_kb
    build_package(config)
    link_path = output_dir / "index" / "linked.json"
    os.symlink(output_dir / "documents.jsonl", link_path)
    archive_path = tmp_path / "client.zip"

    with pytest.raises(PackageError, match="symlink"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()


def test_export_rejects_archive_inside_package(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    archive_path = output_dir / "client.zip"

    with pytest.raises(PackageError, match="outside the source package"):
        export_package(output_dir, archive_path)

    assert not archive_path.exists()
