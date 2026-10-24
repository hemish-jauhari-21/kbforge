"""Configuration loading and validation for kbforge.

The configuration lives in a single YAML file (see ``config.yaml`` for a
commented example). Loading is strict: unknown chunking strategies,
non-HTTP seed URLs, negative limits etc. raise :class:`ConfigError`
immediately so the client sees the problem before a crawl starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

VALID_CHUNKING_STRATEGIES = ("fixed", "sentence", "paragraph")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class SourcesConfig:
    urls: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    files_dir: str | None = None


@dataclass(frozen=True)
class CrawlConfig:
    """Crawl settings.

    ``max_pages`` limits page-fetch attempts, including failed requests and
    responses rejected for their content type. robots.txt requests are not
    counted as page fetches.
    """

    max_depth: int = 2
    max_pages: int = 300
    delay_seconds: float = 1.0
    timeout_seconds: float = 15.0
    retries: int = 3
    user_agent: str = "kbforge/0.1 (academic IR project; polite crawler)"


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "fixed"
    size: int = 300
    overlap: int = 50


@dataclass(frozen=True)
class IndexConfig:
    bm25: bool = True
    dense: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class CleaningConfig:
    enabled: bool = True


@dataclass(frozen=True)
class DeduplicationConfig:
    enabled: bool = True
    exact: bool = True
    near_duplicate: bool = True
    similarity_threshold: float = 0.8
    shingle_size: int = 3
    num_perm: int = 128
@dataclass(frozen=True)
class Config:
    client_name: str
    sources: SourcesConfig
    crawl: CrawlConfig
    chunking: ChunkingConfig
    index: IndexConfig
    output_dir: str
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def _require(data: dict, key: str, context: str):
    if key not in data or data[key] is None:
        raise ConfigError(f"missing required key '{key}' in {context}")
    return data[key]


def _expect_type(value, expected, name: str):
    if not isinstance(value, expected):
        raise ConfigError(f"'{name}' must be of type {expected.__name__}, got {type(value).__name__}")
    return value


def _int_value(value, name: str) -> int:
    """Return an integer, rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {type(value).__name__}")
    return value


def _positive_int(value, name: str) -> int:
    value = _int_value(value, name)
    if value <= 0:
        raise ConfigError(f"'{name}' must be > 0, got {value}")
    return value


def _non_negative_int(value, name: str) -> int:
    value = _int_value(value, name)
    if value < 0:
        raise ConfigError(f"'{name}' must be >= 0, got {value}")
    return value


def _number(value, name: str, *, allow_zero: bool) -> float:
    """Return a finite number, rejecting booleans, NaN, and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a finite number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfigError(f"'{name}' must be a finite number, got {value!r}")
    if allow_zero and value < 0:
        raise ConfigError(f"'{name}' must be >= 0, got {value!r}")
    if not allow_zero and value <= 0:
        raise ConfigError(f"'{name}' must be > 0, got {value!r}")
    return float(value)


def _parse_sources(raw: dict) -> SourcesConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"'sources' must be a mapping, got {type(raw).__name__}")

    urls = _expect_type(raw.get("urls", []), list, "sources.urls")
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            raise ConfigError(f"'sources.urls' entries must be non-empty strings, got {url!r}")
        try:
            parsed = urlparse(url)
            _ = parsed.port  # validate malformed ports eagerly
            host = parsed.hostname
        except ValueError as exc:
            raise ConfigError(f"invalid seed URL {url!r}: {exc}") from exc
        if parsed.scheme not in ("http", "https") or not host:
            raise ConfigError(f"seed URL must be http(s): {url!r}")

    raw_domains = _expect_type(raw.get("allowed_domains", []), list, "sources.allowed_domains")
    domains: list[str] = []
    for domain in raw_domains:
        if not isinstance(domain, str):
            raise ConfigError(
                "'sources.allowed_domains' entries must be strings, "
                f"got {type(domain).__name__}"
            )
        normalized = domain.lower().strip().strip(".")
        if not normalized:
            raise ConfigError("'sources.allowed_domains' entries must be non-empty strings")
        domains.append(normalized)

    files_dir = raw.get("files_dir")
    if files_dir is not None:
        files_dir = _expect_type(files_dir, str, "sources.files_dir").strip()
        if not files_dir:
            raise ConfigError("'sources.files_dir' must be a non-empty string when set")

    if not urls and not files_dir:
        raise ConfigError("sources must define at least one seed URL or a files_dir")
    return SourcesConfig(urls=list(urls), allowed_domains=domains, files_dir=files_dir)


def _parse_crawl(raw: dict) -> CrawlConfig:
    max_depth = _non_negative_int(raw.get("max_depth", 2), "crawl.max_depth")
    max_pages = _positive_int(raw.get("max_pages", 300), "crawl.max_pages")
    delay = _number(raw.get("delay_seconds", 1.0), "crawl.delay_seconds", allow_zero=True)
    timeout = _number(raw.get("timeout_seconds", 15.0), "crawl.timeout_seconds", allow_zero=False)
    retries = _non_negative_int(raw.get("retries", 3), "crawl.retries")
    user_agent = raw.get("user_agent", "kbforge/0.1 (academic IR project; polite crawler)")
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ConfigError("'crawl.user_agent' must be a non-empty string")
    return CrawlConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        delay_seconds=delay,
        timeout_seconds=timeout,
        retries=retries,
        user_agent=user_agent.strip(),
    )


def _parse_chunking(raw: dict) -> ChunkingConfig:
    strategy = raw.get("strategy", "fixed")
    if not isinstance(strategy, str) or strategy not in VALID_CHUNKING_STRATEGIES:
        raise ConfigError(
            f"'chunking.strategy' must be one of {VALID_CHUNKING_STRATEGIES}, got {strategy!r}"
        )
    size = _positive_int(raw.get("size", 300), "chunking.size")
    overlap = _non_negative_int(raw.get("overlap", 50), "chunking.overlap")
    if overlap >= size:
        raise ConfigError(f"'chunking.overlap' ({overlap}) must be smaller than 'chunking.size' ({size})")
    return ChunkingConfig(strategy=strategy, size=size, overlap=overlap)


def _parse_index(raw: dict) -> IndexConfig:
    bm25 = raw.get("bm25", True)
    dense = raw.get("dense", True)
    if not isinstance(bm25, bool) or not isinstance(dense, bool):
        raise ConfigError("'index.bm25' and 'index.dense' must be booleans")
    model = raw.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("'index.embedding_model' must be a non-empty string")
    return IndexConfig(bm25=bm25, dense=dense, embedding_model=model.strip())


def _parse_cleaning(raw: dict) -> CleaningConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"'cleaning' must be a mapping, got {type(raw).__name__}")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("'cleaning.enabled' must be a boolean")
    return CleaningConfig(enabled=enabled)


def _parse_deduplication(raw: dict) -> DeduplicationConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"'deduplication' must be a mapping, got {type(raw).__name__}")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("'deduplication.enabled' must be a boolean")
    exact = raw.get("exact", True)
    if not isinstance(exact, bool):
        raise ConfigError("'deduplication.exact' must be a boolean")
    near_duplicate = raw.get("near_duplicate", True)
    if not isinstance(near_duplicate, bool):
        raise ConfigError("'deduplication.near_duplicate' must be a boolean")
    similarity_threshold = raw.get("similarity_threshold", 0.8)
    if isinstance(similarity_threshold, bool) or not isinstance(similarity_threshold, (int, float)):
        raise ConfigError("'deduplication.similarity_threshold' must be a number")
    if not (0.0 <= float(similarity_threshold) <= 1.0):
        raise ConfigError("'deduplication.similarity_threshold' must be between 0.0 and 1.0")
    shingle_size = raw.get("shingle_size", 3)
    shingle_size = _int_value(shingle_size, "deduplication.shingle_size")
    if shingle_size < 1:
        raise ConfigError("'deduplication.shingle_size' must be >= 1")
    num_perm = raw.get("num_perm", 128)
    num_perm = _int_value(num_perm, "deduplication.num_perm")
    if num_perm < 1:
        raise ConfigError("'deduplication.num_perm' must be >= 1")
    return DeduplicationConfig(
        enabled=enabled,
        exact=exact,
        near_duplicate=near_duplicate,
        similarity_threshold=float(similarity_threshold),
        shingle_size=shingle_size,
        num_perm=num_perm,
    )



def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    client_name = _require(data, "client_name", "config root")
    if not isinstance(client_name, str) or not client_name.strip():
        raise ConfigError("'client_name' must be a non-empty string")

    sources = _parse_sources(_expect_type(_require(data, "sources", "config root"), dict, "sources"))
    crawl = _parse_crawl(_expect_type(data.get("crawl", {}), dict, "crawl"))
    chunking = _parse_chunking(_expect_type(data.get("chunking", {}), dict, "chunking"))
    index = _parse_index(_expect_type(data.get("index", {}), dict, "index"))
    cleaning = _parse_cleaning(_expect_type(data.get("cleaning", {}), dict, "cleaning"))
    deduplication = _parse_deduplication(_expect_type(data.get("deduplication", {}), dict, "deduplication"))

    output_dir = _require(data, "output_dir", "config root")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'output_dir' must be a non-empty string")

    return Config(
        client_name=client_name.strip(),
        sources=sources,
        crawl=crawl,
        chunking=chunking,
        index=index,
        cleaning=cleaning,
        deduplication=deduplication,
        output_dir=output_dir.strip(),
    )
