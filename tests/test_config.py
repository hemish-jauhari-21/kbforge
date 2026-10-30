import textwrap
from pathlib import Path

import pytest

from kbforge.config import ConfigError, load_config


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


VALID = """
client_name: demo
sources:
  urls: ["https://example.com/docs"]
  allowed_domains: ["example.com"]
  files_dir: "./files"
crawl:
  max_depth: 1
  max_pages: 10
  delay_seconds: 0.0
  timeout_seconds: 15.0
  retries: 3
chunking:
  strategy: sentence
  size: 200
  overlap: 20
index:
  bm25: true
  dense: false
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
output_dir: "./out"
"""


def test_load_valid_config(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.client_name == "demo"
    assert config.sources.urls == ["https://example.com/docs"]
    assert config.crawl.max_depth == 1
    assert config.chunking.strategy == "sentence"
    assert config.index.dense is False
    assert config.output_dir == "./out"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_strategy_raises(tmp_path):
    path = write_config(tmp_path, VALID.replace("strategy: sentence", "strategy: magic"))
    with pytest.raises(ConfigError, match="chunking.strategy"):
        load_config(path)


def test_overlap_must_be_smaller_than_size(tmp_path):
    path = write_config(tmp_path, VALID.replace("overlap: 20", "overlap: 200"))
    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


def test_non_http_seed_raises(tmp_path):
    path = write_config(tmp_path, VALID.replace("https://example.com/docs", "ftp://example.com"))
    with pytest.raises(ConfigError, match="http"):
        load_config(path)


def test_sources_needs_url_or_files_dir(tmp_path):
    body = VALID.replace('urls: ["https://example.com/docs"]', "urls: []").replace(
        'files_dir: "./files"', "files_dir: null"
    )
    path = write_config(tmp_path, body)
    with pytest.raises(ConfigError, match="at least one"):
        load_config(path)


def test_derived_domains_when_empty(tmp_path):
    body = VALID.replace("allowed_domains: [\"example.com\"]", "allowed_domains: []")
    config = load_config(write_config(tmp_path, body))
    assert config.sources.allowed_domains == []


def test_defaults_fill_missing_sections(tmp_path):
    body = """
client_name: minimal
sources:
  urls: ["https://example.com"]
output_dir: "./out"
"""
    config = load_config(write_config(tmp_path, body))
    assert config.crawl.max_pages == 300
    assert config.chunking.strategy == "fixed"
    assert config.index.bm25 is True
    assert config.index.dense is False


def test_dense_index_is_rejected_as_unsupported(tmp_path):
    path = write_config(tmp_path, VALID.replace("dense: false", "dense: true"))
    with pytest.raises(ConfigError, match="index.dense.*not supported"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("max_depth: 1", "max_depth: -1", "crawl.max_depth"),
        ("max_depth: 1", "max_depth: true", "crawl.max_depth"),
        ("max_depth: 1", "max_depth: '1'", "crawl.max_depth"),
        ("max_pages: 10", "max_pages: 0", "crawl.max_pages"),
        ("max_pages: 10", "max_pages: -1", "crawl.max_pages"),
        ("max_pages: 10", "max_pages: true", "crawl.max_pages"),
        ("delay_seconds: 0.0", "delay_seconds: -1", "crawl.delay_seconds"),
        ("delay_seconds: 0.0", "delay_seconds: true", "crawl.delay_seconds"),
        ("delay_seconds: 0.0", "delay_seconds: .nan", "crawl.delay_seconds"),
        ("timeout_seconds: 15.0", "timeout_seconds: 0", "crawl.timeout_seconds"),
        ("timeout_seconds: 15.0", "timeout_seconds: -1", "crawl.timeout_seconds"),
        ("timeout_seconds: 15.0", "timeout_seconds: false", "crawl.timeout_seconds"),
        ("timeout_seconds: 15.0", "timeout_seconds: .inf", "crawl.timeout_seconds"),
        ("retries: 3", "retries: -1", "crawl.retries"),
        ("retries: 3", "retries: true", "crawl.retries"),
        ("retries: 3", "retries: 1.5", "crawl.retries"),
        ("size: 200", "size: true", "chunking.size"),
        ("size: 200", "size: 0", "chunking.size"),
        ("overlap: 20", "overlap: true", "chunking.overlap"),
        ("overlap: 20", "overlap: -1", "chunking.overlap"),
        ("bm25: true", "bm25: 'yes'", "index.bm25"),
        ("dense: false", "dense: 'no'", "index.dense"),
        ('embedding_model: "sentence-transformers/all-MiniLM-L6-v2"', "embedding_model: 123", "index.embedding_model"),
        ('output_dir: "./out"', "output_dir: 123", "output_dir"),
        ('client_name: demo', "client_name: 123", "client_name"),
    ],
)
def test_invalid_scalar_config_values_raise_config_error(tmp_path, old, new, match):
    body = VALID.replace(old, new)
    assert body != VALID, f"test fixture replacement failed for {old!r}"
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path, body))


def test_sources_section_must_be_mapping(tmp_path):
    body = """
client_name: bad
sources: ["https://example.com"]
output_dir: "./out"
"""
    with pytest.raises(ConfigError, match="sources"):
        load_config(write_config(tmp_path, body))


def test_crawl_section_must_be_mapping(tmp_path):
    body = """
client_name: bad
sources:
  urls: ["https://example.com"]
crawl: []
output_dir: "./out"
"""
    with pytest.raises(ConfigError, match="crawl"):
        load_config(write_config(tmp_path, body))


def test_sources_urls_must_be_list(tmp_path):
    body = VALID.replace('urls: ["https://example.com/docs"]', 'urls: "https://example.com/docs"')
    with pytest.raises(ConfigError, match="sources.urls"):
        load_config(write_config(tmp_path, body))


def test_sources_url_entries_must_be_strings(tmp_path):
    body = VALID.replace('urls: ["https://example.com/docs"]', "urls: [123]")
    with pytest.raises(ConfigError, match="sources.urls"):
        load_config(write_config(tmp_path, body))


def test_seed_url_with_invalid_port_is_config_error(tmp_path):
    body = VALID.replace('urls: ["https://example.com/docs"]', 'urls: ["https://example.com:bad/docs"]')
    with pytest.raises(ConfigError, match="invalid seed URL"):
        load_config(write_config(tmp_path, body))


def test_allowed_domains_must_be_list(tmp_path):
    body = VALID.replace('allowed_domains: ["example.com"]', 'allowed_domains: "example.com"')
    with pytest.raises(ConfigError, match="sources.allowed_domains"):
        load_config(write_config(tmp_path, body))


def test_allowed_domain_entries_must_be_non_empty_strings(tmp_path):
    for replacement in ("[123]", '[""]'):
        body = VALID.replace('allowed_domains: ["example.com"]', f"allowed_domains: {replacement}")
        with pytest.raises(ConfigError, match="sources.allowed_domains"):
            load_config(write_config(tmp_path, body))


def test_files_dir_must_be_string(tmp_path):
    body = VALID.replace('files_dir: "./files"', "files_dir: 123")
    with pytest.raises(ConfigError, match="sources.files_dir"):
        load_config(write_config(tmp_path, body))


def test_chunking_strategy_must_be_string(tmp_path):
    body = VALID.replace("strategy: sentence", "strategy: 123")
    with pytest.raises(ConfigError, match="chunking.strategy"):
        load_config(write_config(tmp_path, body))
