"""Regression tests for build failure handling and Phase 2 input selection."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from kbforge.cli import main
from kbforge.config import CleaningConfig, DeduplicationConfig
from kbforge.documents import Document, write_documents_jsonl
from kbforge.package import (
    BUILD_MARKER_FILENAME,
    IncompleteBuildError,
    build_package,
    load_package,
)
from kbforge.process import run_chunk
from tests.conftest import make_config


def _write_config(path, output_dir):
    path.write_text(
        yaml.safe_dump(
            {
                "client_name": "build-test",
                "sources": {"urls": ["https://kb.test/"], "allowed_domains": ["kb.test"]},
                "chunking": {"strategy": "fixed", "size": 50, "overlap": 0},
                "index": {"bm25": True, "dense": False},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("phase", "target"),
    [
        (2, "kbforge.process.run_process"),
        (3, "kbforge.process.run_chunk"),
        (4, "kbforge.process.run_index"),
        (5, "kbforge.package.build_package"),
    ],
)
def test_build_returns_nonzero_when_phase_fails(
    tmp_path, monkeypatch, capsys, phase, target
):
    output_dir = tmp_path / "kb"
    config_path = tmp_path / "client.yaml"
    _write_config(config_path, output_dir)

    monkeypatch.setattr(
        "kbforge.ingest.crawler.run_ingest",
        lambda config: {
            "pages_fetched": 1,
            "pages_failed": 0,
            "files_loaded": 0,
            "files_failed": 0,
            "documents_written": 1,
            "edges": 0,
        },
    )
    monkeypatch.setattr(
        "kbforge.process.run_process",
        lambda config: {
            "processed_documents": 1,
            "removals": 0,
            "cleaning_warnings": 0,
            "cleaning_exclusions": 0,
            "cleaned_path": "cleaned",
            "removal_log_path": "removals",
        },
    )
    monkeypatch.setattr(
        "kbforge.process.run_chunk",
        lambda config: {"chunks_written": 1, "chunks_path": "chunks"},
    )
    monkeypatch.setattr(
        "kbforge.process.run_index",
        lambda config: {"chunks_indexed": 1, "index_dir": "index"},
    )
    monkeypatch.setattr(
        "kbforge.package.build_package",
        lambda config: {"manifest_path": "manifest", "ranking_modes": ["bm25"]},
    )

    def fail(*args, **kwargs):
        raise RuntimeError(f"phase {phase} exploded")

    monkeypatch.setattr(target, fail)

    exit_code = main(["build", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"phase {phase} error: phase {phase} exploded" in captured.err


def test_failed_rebuild_leaves_marker_and_blocks_stale_package(
    sample_kb, tmp_path, monkeypatch, capsys
):
    config, output_dir = sample_kb
    build_package(config)
    config_path = tmp_path / "client.yaml"
    _write_config(config_path, output_dir)

    monkeypatch.setattr(
        "kbforge.ingest.crawler.run_ingest",
        lambda config: {
            "pages_fetched": 1,
            "pages_failed": 0,
            "files_loaded": 0,
            "files_failed": 0,
            "documents_written": 1,
            "edges": 0,
        },
    )
    monkeypatch.setattr(
        "kbforge.process.run_process",
        lambda config: {
            "processed_documents": 1,
            "removals": 0,
            "cleaning_warnings": 0,
            "cleaning_exclusions": 0,
            "cleaned_path": "cleaned",
            "removal_log_path": "removals",
        },
    )

    def fail(*args, **kwargs):
        raise RuntimeError("phase 3 exploded")

    monkeypatch.setattr("kbforge.process.run_chunk", fail)

    exit_code = main(["build", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "phase 3 error: phase 3 exploded" in captured.err
    assert (output_dir / BUILD_MARKER_FILENAME).is_file()
    with pytest.raises(IncompleteBuildError, match="build is incomplete"):
        load_package(output_dir)


def test_build_reports_phase2_removals_and_cleaning_diagnostics_separately(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "kb"
    config_path = tmp_path / "client.yaml"
    _write_config(config_path, output_dir)

    monkeypatch.setattr(
        "kbforge.ingest.crawler.run_ingest",
        lambda config: {
            "pages_fetched": 1,
            "pages_failed": 0,
            "files_loaded": 0,
            "files_failed": 0,
            "documents_written": 1,
            "edges": 0,
        },
    )
    monkeypatch.setattr(
        "kbforge.process.run_process",
        lambda config: {
            "processed_documents": 3,
            "removals": 2,
            "cleaning_warnings": 4,
            "cleaning_exclusions": 5,
            "cleaned_path": "cleaned",
            "removal_log_path": "removals",
        },
    )
    monkeypatch.setattr(
        "kbforge.process.run_chunk",
        lambda config: {"chunks_written": 1, "chunks_path": "chunks"},
    )
    monkeypatch.setattr(
        "kbforge.process.run_index",
        lambda config: {"chunks_indexed": 1, "index_dir": "index"},
    )
    monkeypatch.setattr(
        "kbforge.package.build_package",
        lambda config: {"manifest_path": "manifest", "ranking_modes": ["bm25"]},
    )

    exit_code = main(["build", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "duplicate removals  : 2" in captured.out
    assert "cleaning warnings   : 4" in captured.out
    assert "cleaning exclusions : 5" in captured.out


def test_run_chunk_uses_raw_documents_when_phase2_is_disabled(tmp_path):
    output_dir = tmp_path / "kb"
    raw_doc = Document.from_parts(
        source="https://kb.test/page",
        title="Page",
        raw_text="fresh raw documents",
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="txt",
    )
    stale_doc = Document.from_parts(
        source="https://kb.test/page",
        title="Page",
        raw_text="stale cleaned documents",
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="txt",
    )
    write_documents_jsonl([raw_doc], output_dir / "documents.jsonl")
    write_documents_jsonl([stale_doc], output_dir / "documents_cleaned.jsonl")

    config = replace(
        make_config(output_dir=str(output_dir)),
        cleaning=CleaningConfig(enabled=False),
        deduplication=DeduplicationConfig(enabled=False),
    )

    run_chunk(config)

    chunks = [
        json.loads(line)
        for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(chunks) == 1
    assert chunks[0]["text"] == "fresh raw documents"
    assert (output_dir / "documents_cleaned.jsonl").exists()
