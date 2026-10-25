"""Phase 5 package creation, validation and saved-index search tests."""

from __future__ import annotations

import json

import pytest

from kbforge.package import (
    BUILD_CONFIG_FILENAME,
    BUILD_STATS_FILENAME,
    MANIFEST_FILENAME,
    PACKAGE_FORMAT,
    PACKAGE_FORMAT_VERSION,
    IncompatiblePackageError,
    KnowledgeBase,
    MissingArtifactError,
    PackageError,
    build_package,
    load_package,
)


def _read(path):
    return path.read_text(encoding="utf-8")


def test_build_package_creates_manifest_and_stats(sample_kb):
    config, output_dir = sample_kb

    summary = build_package(config)

    manifest_path = output_dir / MANIFEST_FILENAME
    config_path = output_dir / BUILD_CONFIG_FILENAME
    stats_path = output_dir / BUILD_STATS_FILENAME
    assert manifest_path.is_file()
    assert config_path.is_file()
    assert stats_path.is_file()
    assert summary["manifest_path"] == str(manifest_path)

    manifest = json.loads(_read(manifest_path))
    assert manifest["format"] == PACKAGE_FORMAT
    assert manifest["format_version"] == PACKAGE_FORMAT_VERSION
    assert manifest["ranking_modes"] == ["bm25", "tfidf"]
    assert manifest["statistics"]["chunks"] > 0
    assert manifest["artifacts"]["chunks"]["path"] == "chunks.jsonl"
    assert manifest["artifacts"]["index/postings"]["path"] == "index/postings.json"
    assert all(not artifact["path"].startswith("/") for artifact in manifest["artifacts"].values())

    build_config = json.loads(_read(config_path))
    assert build_config["client_name"] == "test"
    assert build_config["sources"]["url_count"] == 1
    assert "urls" not in build_config["sources"]
    assert "output_dir" not in build_config

    build_stats = json.loads(_read(stats_path))
    assert build_stats["chunks"] == manifest["statistics"]["chunks"]
    assert build_stats["index"]["chunks_indexed"] == manifest["statistics"]["chunks_indexed"]
    assert build_stats["ingestion"]["pages_fetched"] == 3


def test_package_is_deterministic_and_does_not_touch_artifacts(sample_kb):
    config, output_dir = sample_kb
    build_package(config)

    tracked = [output_dir / "chunks.jsonl", output_dir / "index" / "postings.json"]
    before = {path: path.read_bytes() for path in tracked}
    first = {name: (output_dir / name).read_bytes() for name in (
        MANIFEST_FILENAME,
        BUILD_CONFIG_FILENAME,
        BUILD_STATS_FILENAME,
    )}

    build_package(config)

    after = {path: path.read_bytes() for path in tracked}
    second = {name: (output_dir / name).read_bytes() for name in first}
    assert before == after
    assert first == second


def test_load_package_without_rebuilding(sample_kb, monkeypatch):
    config, output_dir = sample_kb
    build_package(config)

    def fail(*args, **kwargs):
        raise AssertionError("ingestion/processing must not run while querying")

    monkeypatch.setattr("kbforge.ingest.crawler.run_ingest", fail)
    monkeypatch.setattr("kbforge.process.run_process", fail)
    monkeypatch.setattr("kbforge.process.run_chunk", fail)
    monkeypatch.setattr("kbforge.process.run_index", fail)

    kb = load_package(output_dir)
    assert isinstance(kb, KnowledgeBase)
    assert kb.index.N == 3

    kb_from_index = KnowledgeBase.load(output_dir / "index")
    assert kb_from_index.index.N == 3


@pytest.mark.parametrize("mode", ["bm25", "tfidf"])
def test_saved_index_search_preserves_metadata(sample_kb, mode):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)

    results = kb.search("python", mode=mode, limit=5)

    assert results
    top = results[0]
    assert top.source == "https://kb.test/python"
    assert top.title == "Python guide"
    assert top.chunk_id
    assert top.doc_id.startswith("web-")
    assert top.score > 0
    assert top.rank == 0
    assert top.text


def test_results_are_deterministic(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)

    first = kb.search("search index", mode="bm25", limit=3)
    second = kb.search("search index", mode="bm25", limit=3)
    assert [(r.chunk_id, r.score, r.rank) for r in first] == [
        (r.chunk_id, r.score, r.rank) for r in second
    ]


def test_missing_artifact_is_reported(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "chunks.jsonl").unlink()

    with pytest.raises(MissingArtifactError, match="chunks"):
        load_package(output_dir)


def test_corrupt_index_artifact_is_reported(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "index" / "postings.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(PackageError, match="postings.json"):
        load_package(output_dir)


def test_corrupt_chunks_artifact_is_reported(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "chunks.jsonl").write_text("not-json\n", encoding="utf-8")

    with pytest.raises(PackageError, match="chunks"):
        load_package(output_dir)


def test_unsupported_package_version_is_rejected(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(_read(manifest_path))
    manifest["format_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IncompatiblePackageError, match="99.0"):
        load_package(output_dir)


def test_missing_package_manifest_is_reported(tmp_path):
    with pytest.raises(MissingArtifactError, match="manifest"):
        load_package(tmp_path)


def test_invalid_search_arguments(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)

    with pytest.raises(ValueError, match="ranking mode"):
        kb.search("python", mode="dense")
    with pytest.raises(ValueError, match="greater than 0"):
        kb.search("python", limit=0)
    with pytest.raises(ValueError, match="integer"):
        kb.search("python", limit="5")
    assert kb.search("   ") == []
    assert kb.search("nonexistentterm") == []


def test_package_refuses_to_overwrite_unrelated_metadata(sample_kb):
    config, output_dir = sample_kb
    user_manifest = output_dir / MANIFEST_FILENAME
    user_manifest.write_text('{"custom": "keep me"}', encoding="utf-8")

    with pytest.raises(PackageError, match="refusing to overwrite"):
        build_package(config)

    assert json.loads(user_manifest.read_text(encoding="utf-8")) == {"custom": "keep me"}
