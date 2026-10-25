"""Phase 5 CLI packaging/query tests (offline)."""

from __future__ import annotations

import json

import yaml

from kbforge.cli import main
from kbforge.package import MANIFEST_FILENAME, build_package


def _write_config(path, output_dir):
    path.write_text(
        yaml.safe_dump(
            {
                "client_name": "cli-test",
                "sources": {"urls": ["https://kb.test/"], "allowed_domains": ["kb.test"]},
                "chunking": {"strategy": "fixed", "size": 50, "overlap": 0},
                "index": {"bm25": True, "dense": False},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )


def test_query_saved_package_does_not_run_pipeline(sample_kb, monkeypatch, capsys):
    config, output_dir = sample_kb
    build_package(config)

    def fail(*args, **kwargs):
        raise AssertionError("the query command must not rebuild or crawl")

    monkeypatch.setattr("kbforge.ingest.crawler.run_ingest", fail)
    monkeypatch.setattr("kbforge.process.run_process", fail)
    monkeypatch.setattr("kbforge.process.run_chunk", fail)
    monkeypatch.setattr("kbforge.process.run_index", fail)

    exit_code = main(
        ["query", "python", "--package", str(output_dir), "--mode", "bm25", "--k", "2"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Python guide" in captured.out
    assert "https://kb.test/python" in captured.out
    assert "doc web-" in captured.out
    assert "chunk chunk-" in captured.out


def test_query_uses_config_output_dir(sample_kb, tmp_path, capsys):
    config, output_dir = sample_kb
    build_package(config)
    config_path = tmp_path / "client.yaml"
    _write_config(config_path, output_dir)

    exit_code = main(["query", "inverted", "--config", str(config_path), "--mode", "tfidf"])

    assert exit_code == 0
    assert "Search notes" in capsys.readouterr().out


def test_query_validates_inputs(sample_kb, capsys):
    config, output_dir = sample_kb
    build_package(config)

    assert main(["query", "   ", "--package", str(output_dir)]) == 2
    assert "empty" in capsys.readouterr().err

    assert main(["query", "python", "--package", str(output_dir), "--k", "0"]) == 2
    assert "--k" in capsys.readouterr().err


def test_query_missing_package_reports_error(tmp_path, capsys):
    exit_code = main(["query", "python", "--package", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "error loading knowledge base" in capsys.readouterr().err


def test_query_no_results_is_graceful(sample_kb, capsys):
    config, output_dir = sample_kb
    build_package(config)

    assert main(["query", "nonexistentterm", "--package", str(output_dir)]) == 0
    assert "no results" in capsys.readouterr().out


def test_package_command_creates_manifest(sample_kb, tmp_path, capsys):
    config, output_dir = sample_kb
    config_path = tmp_path / "client.yaml"
    _write_config(config_path, output_dir)

    exit_code = main(["package", "--config", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert (output_dir / MANIFEST_FILENAME).is_file()
    assert "package manifest" in out
    assert "bm25, tfidf" in out


def test_cli_help_lists_phase5_commands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    for command in ("build", "query", "package", "serve"):
        assert command in out


def test_package_manifest_is_valid_json(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    manifest = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "kbforge-knowledge-base"
