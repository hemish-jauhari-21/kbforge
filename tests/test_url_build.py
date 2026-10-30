"""Offline tests for the interactive URL build workflow."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from kbforge.cli import main
from kbforge.config import load_config
from kbforge.documents import read_documents_jsonl, write_documents_jsonl
from kbforge.package import (
    BUILD_CONFIG_FILENAME,
    BUILD_MARKER_FILENAME,
    MANIFEST_FILENAME,
    README_FILENAME,
    PackageError,
)
from kbforge.url_build import (
    GENERATED_CONFIG_MARKER,
    UrlBuildError,
    collect_url_build_options,
    output_slug,
    resolve_build_paths,
    validate_website_url,
)
from kbforge.urls import host_matches


def _install_fake_ingest(monkeypatch, source_output: Path):
    documents = read_documents_jsonl(source_output / "documents.jsonl")

    def fake_run_ingest(config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = write_documents_jsonl(documents, output_dir / "documents.jsonl")
        return {
            "page_attempts": written,
            "pages_fetched": written,
            "pages_failed": 0,
            "files_loaded": 0,
            "files_failed": 0,
            "documents_written": written,
            "edges": 0,
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("kbforge.ingest.crawler.run_ingest", fake_run_ingest)


def _input_feeder(monkeypatch, values):
    prompts: list[str] = []
    iterator = iter(values)

    def fake_input(prompt=""):
        prompts.append(prompt)
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError(f"unexpected extra prompt: {prompt!r}") from exc

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def _no_input(monkeypatch):
    def fail(prompt=""):
        raise AssertionError(f"non-interactive build prompted: {prompt!r}")

    monkeypatch.setattr("builtins.input", fail)


def test_validate_website_url_and_output_slug():
    url, hostname = validate_website_url("https://www.Example.com/docs?utm_source=x")
    assert url == "https://www.example.com/docs"
    assert hostname == "www.example.com"
    assert output_slug(hostname) == "example-com"
    assert output_slug("127.0.0.1") == "127-0-0-1"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "must not be empty"),
        ("example.com", "http"),
        ("ftp://example.com", "http"),
        ("https://", "http"),
        ("https://exa mple.com", "hostname"),
        ("https://user:pass@example.com", "credentials"),
        ("https://example.com:bad/", "invalid website URL"),
    ],
)
def test_validate_website_url_rejects_invalid_input(value, match):
    with pytest.raises(UrlBuildError, match=match):
        validate_website_url(value)


@pytest.mark.parametrize(
    "output",
    [
        "output",
        "../escape",
        "output/../escape",
        "output/.kbforge-configs/inside",
    ],
)
def test_resolve_build_paths_prevents_escape(tmp_path, output):
    with pytest.raises(UrlBuildError):
        resolve_build_paths(output, "example.com", tmp_path)


def test_host_matches_can_require_exact_host():
    assert host_matches("https://docs.example.com/", ["example.com"])
    assert not host_matches(
        "https://docs.example.com/",
        ["example.com"],
        include_subdomains=False,
    )
    assert host_matches(
        "https://example.com/",
        ["example.com"],
        include_subdomains=False,
    )


def test_interactive_url_prompts_allow_correction(tmp_path):
    answers = iter(
        ["not a URL", "ftp://example.com", "https://example.com", "y"]
    )
    result = collect_url_build_options(
        url=None,
        max_pages=1,
        max_depth=0,
        include_subdomains=True,
        chunking_strategy="fixed",
        chunk_size=10,
        chunk_overlap=0,
        ranking_modes="bm25",
        output="output/name",
        yes=False,
        input_fn=lambda prompt="": next(answers),
        print_fn=lambda *args, **kwargs: None,
        cwd=tmp_path,
    )
    assert result is not None
    options, paths = result
    assert options.url == "https://example.com/"
    assert paths.output_dir.name == "name"


def test_interactive_defaults_build_and_export(tmp_path, sample_kb, monkeypatch, capsys):
    source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    prompts = _input_feeder(monkeypatch, ["", "", "", "", "", "", "", "y"])

    exit_code = main(["build", "--url", "https://docs.example.com/guide"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert not any("Enter website URL" in prompt for prompt in prompts)
    assert prompts[0].startswith("Maximum pages to crawl")
    assert "KBForge - Website Knowledge Base Builder" in captured.out
    assert "Review configuration:" in captured.out
    assert "Start build?" in prompts[-1]
    assert "Build completed successfully." in captured.out
    assert "Documents: 3" in captured.out
    assert "Chunking strategy: fixed" in captured.out
    assert "README: included in ZIP" in captured.out

    output_dir = tmp_path / "output" / "docs-example-com"
    config_path = tmp_path / "output" / ".kbforge-configs" / "docs-example-com.yaml"
    archive_path = tmp_path / "output" / "docs-example-com.zip"
    assert output_dir.is_dir()
    assert config_path.is_file()
    assert config_path.read_text(encoding="utf-8").startswith(GENERATED_CONFIG_MARKER)
    assert archive_path.is_file()

    config = load_config(config_path)
    assert config.sources.urls == ["https://docs.example.com/guide"]
    assert config.sources.allowed_domains == ["docs.example.com"]
    assert config.crawl.max_pages == 300
    assert config.crawl.max_depth == 2
    assert config.crawl.include_subdomains is False
    assert config.chunking.strategy == "fixed"
    assert config.chunking.size == 300
    assert config.chunking.overlap == 50
    assert config.index.ranking_modes == ("bm25", "tfidf")

    build_config = json.loads(
        (output_dir / BUILD_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert build_config["chunking"] == {
        "strategy": "fixed",
        "size": 300,
        "overlap": 50,
    }
    manifest = json.loads(
        (output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["ranking_modes"] == ["bm25", "tfidf"]

    with zipfile.ZipFile(archive_path) as archive:
        assert README_FILENAME in archive.namelist()


def test_interactive_cancellation_stops_before_pipeline(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    _input_feeder(monkeypatch, ["", "", "", "", "", "", "", "n"])

    def fail(*args, **kwargs):
        raise AssertionError("cancelled build must not run the pipeline")

    monkeypatch.setattr("kbforge.cli._run_build_pipeline", fail)

    exit_code = main(["build", "--url", "https://example.com/"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build cancelled." in captured.out
    assert not (tmp_path / "output" / "example-com").exists()
    assert not (tmp_path / "output" / ".kbforge-configs").exists()
    assert not (tmp_path / "output" / "example-com.zip").exists()


def test_interactive_invalid_values_can_be_corrected(
    tmp_path, sample_kb, monkeypatch, capsys
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _input_feeder(
        monkeypatch,
        [
            "abc",
            "0",
            "5",
            "-2",
            "1",
            "maybe",
            "yes",
            "9",
            "2",
            "0",
            "40",
            "bogus",
            "bm25, bm25",
            "../escape",
            "",
            "y",
        ],
    )

    exit_code = main(["build", "--url", "https://example.com/"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Please enter a whole number." in captured.out
    assert "greater than or equal to 1" in captured.out
    assert "Please answer yes or no." in captured.out
    assert "invalid chunking strategy" in captured.out
    assert "unsupported ranking mode" in captured.out
    assert "output directory must be inside output/" in captured.out

    config = load_config(
        tmp_path / "output" / ".kbforge-configs" / "example-com.yaml"
    )
    assert config.crawl.max_pages == 5
    assert config.crawl.max_depth == 1
    assert config.crawl.include_subdomains is True
    assert config.chunking.strategy == "sentence"
    assert config.chunking.size == 40
    assert config.chunking.overlap == 0
    assert config.index.ranking_modes == ("bm25",)

    chunks = [
        json.loads(line)
        for line in (tmp_path / "output" / "example-com" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert chunks
    assert {chunk["strategy"] for chunk in chunks} == {"sentence"}


def test_noninteractive_mode_does_not_prompt(
    tmp_path, sample_kb, monkeypatch, capsys
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    exit_code = main(
        [
            "build",
            "--url",
            "https://example.com/",
            "--yes",
        ]
    )
    assert exit_code == 0
    assert "Build completed successfully." in capsys.readouterr().out
    assert (tmp_path / "output" / "example-com.zip").is_file()


@pytest.mark.parametrize(
    ("strategy", "expected_overlap"),
    [("fixed", 24), ("sentence", 0), ("paragraph", 0)],
)
def test_all_chunking_strategies_map_to_existing_pipeline(
    tmp_path,
    sample_kb,
    monkeypatch,
    strategy,
    expected_overlap,
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    exit_code = main(
        [
            "build",
            "--url",
            "https://example.com/",
            "--max-pages",
            "2",
            "--max-depth",
            "0",
            "--chunking-strategy",
            strategy,
            "--chunk-size",
            "25",
            "--ranking-modes",
            "bm25",
            "--output",
            f"output/{strategy}-kb",
            "--yes",
        ]
    )

    assert exit_code == 0
    output_dir = tmp_path / "output" / f"{strategy}-kb"
    config = load_config(
        tmp_path / "output" / ".kbforge-configs" / f"{strategy}-kb.yaml"
    )
    assert config.chunking.strategy == strategy
    assert config.chunking.size == 25
    assert config.chunking.overlap == expected_overlap
    assert config.index.ranking_modes == ("bm25",)
    manifest = json.loads(
        (output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["ranking_modes"] == ["bm25"]
    chunks = [
        json.loads(line)
        for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks
    assert {chunk["strategy"] for chunk in chunks} == {strategy}


@pytest.mark.parametrize(
    ("extra_args", "match"),
    [
        (["--max-pages", "0"], "maximum pages"),
        (["--max-depth", "-1"], "maximum crawl depth"),
        (["--chunk-size", "0"], "maximum words per chunk"),
        (["--chunk-overlap", "300"], "chunk overlap"),
        (["--ranking-modes", "magic"], "unsupported ranking mode"),
    ],
)
def test_noninteractive_invalid_options_exit_without_building(
    tmp_path, monkeypatch, capsys, extra_args, match
):
    monkeypatch.chdir(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("invalid input must not start a crawl")

    monkeypatch.setattr("kbforge.cli._run_build_pipeline", fail)

    exit_code = main(
        ["build", "--url", "https://example.com/", "--yes", *extra_args]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "input error" in captured.err
    assert match in captured.err
    assert not (tmp_path / "output").exists()


def test_invalid_url_exits_without_building(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("invalid URL must not start a crawl")

    monkeypatch.setattr("kbforge.cli._run_build_pipeline", fail)

    exit_code = main(["build", "--url", "ftp://example.com", "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "website URL must start with http" in captured.err


def test_noninteractive_requires_url(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["build", "--yes"]) == 2
    assert "require --url" in capsys.readouterr().err


def test_invalid_strategy_is_rejected_by_parser(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "build",
                "--url",
                "https://example.com/",
                "--chunking-strategy",
                "magic",
                "--yes",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("collision", ["output", "zip", "config"])
def test_output_collision_protection_and_overwrite(
    tmp_path, sample_kb, monkeypatch, capsys, collision
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    output_root = tmp_path / "output"
    output_dir = output_root / "example-com"
    archive_path = output_root / "example-com.zip"
    config_path = output_root / ".kbforge-configs" / "example-com.yaml"
    output_root.mkdir()

    if collision == "output":
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old build", encoding="utf-8")
    elif collision == "zip":
        archive_path.write_bytes(b"old archive")
    else:
        config_path.parent.mkdir()
        config_path.write_text(
            GENERATED_CONFIG_MARKER + "\nold generated config\n",
            encoding="utf-8",
        )

    exit_code = main(["build", "--url", "https://example.com/", "--yes"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "already exists" in captured.err
    assert not (output_root / "example-com" / "documents.jsonl").exists()

    exit_code = main(
        ["build", "--url", "https://example.com/", "--yes", "--overwrite"]
    )
    assert exit_code == 0
    assert (output_dir / MANIFEST_FILENAME).is_file()
    assert archive_path.is_file()
    assert config_path.read_text(encoding="utf-8").startswith(GENERATED_CONFIG_MARKER)
    assert not (output_dir / "old.txt").exists()


def test_export_failure_reports_failure(
    tmp_path, sample_kb, monkeypatch, capsys
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    def fail_export(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("kbforge.package.export_package", fail_export)

    exit_code = main(["build", "--url", "https://example.com/", "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "export error: disk full" in captured.err
    assert "Build completed successfully" not in captured.out
    assert (tmp_path / "output" / "example-com").is_dir()
    assert not (tmp_path / "output" / "example-com.zip").exists()


def test_package_validation_happens_before_export(
    tmp_path, sample_kb, monkeypatch, capsys
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    def fail_validation(cls, path):
        raise PackageError("invalid generated package")

    def fail_export(*args, **kwargs):
        raise AssertionError("export must not run after validation failure")

    monkeypatch.setattr(
        "kbforge.package.KnowledgeBase.from_package",
        classmethod(fail_validation),
    )
    monkeypatch.setattr("kbforge.package.export_package", fail_export)

    exit_code = main(["build", "--url", "https://example.com/", "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "package validation error" in captured.err
    assert "invalid generated package" in captured.err
    assert "Build completed successfully" not in captured.out
    output_dir = tmp_path / "output" / "example-com"
    assert (output_dir / BUILD_MARKER_FILENAME).is_file()
    assert not (tmp_path / "output" / "example-com.zip").exists()


def test_yaml_build_remains_available(sample_kb, tmp_path, monkeypatch, capsys):
    _config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "client_name": "yaml-client",
                "sources": {
                    "urls": ["https://example.com/"],
                    "allowed_domains": ["example.com"],
                },
                "chunking": {"strategy": "fixed", "size": 50, "overlap": 0},
                "output_dir": "output/yaml-client",
            }
        ),
        encoding="utf-8",
    )
    _no_input(monkeypatch)

    exit_code = main(["build", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "phase 5 complete" in captured.out
    assert (tmp_path / "output" / "yaml-client" / MANIFEST_FILENAME).is_file()
    assert not (tmp_path / "output" / ".kbforge-configs").exists()


def test_url_and_config_cannot_be_combined(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "build",
            "--url",
            "https://example.com/",
            "--config",
            "config.yaml",
            "--yes",
        ]
    )
    assert exit_code == 2
    assert "cannot be used together" in capsys.readouterr().err


def test_url_options_without_url_are_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["build", "--output", "output/test"]) == 2
    assert "require --url" in capsys.readouterr().err


def test_single_ranking_mode_package_can_be_queried_and_evaluated(
    tmp_path, sample_kb, monkeypatch, capsys
):
    _source_config, source_output = sample_kb
    monkeypatch.chdir(tmp_path)
    _install_fake_ingest(monkeypatch, source_output)
    _no_input(monkeypatch)

    assert (
        main(
            [
                "build",
                "--url",
                "https://example.com/",
                "--ranking-modes",
                "bm25",
                "--yes",
            ]
        )
        == 0
    )
    output_dir = tmp_path / "output" / "example-com"
    capsys.readouterr()

    assert (
        main(["query", "python", "--package", str(output_dir), "--k", "1"]) == 0
    )
    assert "doc " in capsys.readouterr().out

    assert (
        main(
            [
                "query",
                "python",
                "--package",
                str(output_dir),
                "--mode",
                "tfidf",
            ]
        )
        == 2
    )
    assert "not available in this package" in capsys.readouterr().err

    documents = read_documents_jsonl(output_dir / "documents_cleaned.jsonl")
    relevant = next(document.doc_id for document in documents if "Python" in document.title)
    dataset_path = tmp_path / "judgments.json"
    dataset_path.write_text(
        json.dumps(
            {
                "format": "kbforge-evaluation",
                "format_version": "1.0",
                "name": "single-mode",
                "synthetic": True,
                "unit": "document",
                "queries": [
                    {
                        "query_id": "q-python",
                        "query": "python",
                        "relevant": [relevant],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    assert (
        main(
            [
                "evaluate",
                "--package",
                str(output_dir),
                "--queries",
                str(dataset_path),
                "--mode",
                "both",
                "--k",
                "1",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "bm25:" in captured.out
    assert "tfidf:" not in captured.out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["configuration"]["modes"] == ["bm25"]
