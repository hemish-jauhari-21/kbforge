"""Knowledge-base packaging and loading (Phase 5).

A *package* is an existing ``output_dir`` plus a small set of metadata
files.  The package deliberately references the artifacts written by the
ingestion, cleaning, chunking and indexing phases instead of copying them:

    output_dir/
      manifest.json          package manifest (format + artifact paths)
      build_config.json      safe, credential-free configuration summary
      build_stats.json       aggregated build/index statistics
      documents.jsonl        raw ingestion output (optional for querying)
      documents_cleaned.jsonl
      chunks.jsonl           chunk text + metadata for result reconstruction
      index/                 inverted index, BM25/TF-IDF source data
        postings.json
        doc_lengths.json
        chunk_info.json
        df.json
        index_stats.json

Only ``manifest.json``, ``build_config.json`` and ``build_stats.json`` are
new.  All artifact paths in the manifest are relative to the manifest so
the directory can be moved as a unit.  The format is versioned with
``format_version``; loaders reject versions they do not understand.

Ranking is *not* reimplemented here: :class:`KnowledgeBase` delegates to
the existing :class:`kbforge.index.BM25` and :class:`kbforge.index.TFIDF`
implementations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kbforge import __version__
from kbforge.index import BM25, InvertedIndex, SearchResult, TFIDF

logger = logging.getLogger("kbforge.package")

PACKAGE_FORMAT = "kbforge-knowledge-base"
PACKAGE_FORMAT_VERSION = "1.0"
SUPPORTED_FORMAT_VERSIONS = (PACKAGE_FORMAT_VERSION,)
SUPPORTED_RANKING_MODES = ("bm25", "tfidf")

MANIFEST_FILENAME = "manifest.json"
BUILD_CONFIG_FILENAME = "build_config.json"
BUILD_STATS_FILENAME = "build_stats.json"
BUILD_CONFIG_FORMAT = "kbforge-build-config"
BUILD_STATS_FORMAT = "kbforge-build-stats"

INDEX_FILENAMES = (
    "postings.json",
    "doc_lengths.json",
    "chunk_info.json",
    "df.json",
    "index_stats.json",
)

REQUIRED_ARTIFACT_KEYS = (
    "chunks",
    "index/postings",
    "index/doc_lengths",
    "index/chunk_info",
    "index/df",
    "index/index_stats",
    "build_config",
    "build_stats",
)


class PackageError(Exception):
    """Base class for package problems (missing/corrupt/incompatible)."""


class MissingArtifactError(PackageError):
    """A required package artifact is missing."""


class IncompatiblePackageError(PackageError):
    """The package format or version is not supported."""


@dataclass(frozen=True)
class Manifest:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    def artifact_path(self, key: str) -> Path:
        entry = self.data.get("artifacts", {}).get(key)
        if not isinstance(entry, dict) or not entry.get("path"):
            raise MissingArtifactError(f"manifest has no artifact entry for '{key}'")
        candidate = Path(entry["path"])
        if candidate.is_absolute():
            return candidate
        return self.root / candidate


def _guard_metadata_path(path: Path, expected_format: str, description: str) -> None:
    """Refuse to clobber a same-named file that is not ours."""
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(
            f"refusing to overwrite existing {description} '{path.name}': not a kbforge file"
        ) from exc
    if not isinstance(existing, dict) or existing.get("format") != expected_format:
        raise PackageError(
            f"refusing to overwrite existing {description} '{path.name}': "
            "unrecognized format"
        )


def _write_json(path: Path, data: Any) -> None:
    """Write deterministic JSON atomically, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MissingArtifactError(f"missing {description}: {path.name}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid {description} '{path.name}': {exc}") from exc


def _relpath(target: Path, base: Path) -> str:
    try:
        value = os.path.relpath(target, base)
    except ValueError:  # different drives on Windows
        value = str(target)
    return Path(value).as_posix()


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _artifact(path: Path, base: Path, fmt: str, *, required: bool = True, **extra) -> dict:
    entry = {"path": _relpath(path, base), "format": fmt, "required": required}
    entry.update(extra)
    return entry


def _config_summary(config) -> dict:
    """Return a safe, deterministic configuration summary.

    Seed URLs and local filesystem paths are intentionally not copied: URLs
    can contain tokens and absolute paths are host-specific.  Counts, limits
    and ranking settings are enough to understand how the package was built.
    """
    if config is None:
        return {
            "available": False,
            "note": "no configuration was supplied when packaging",
        }

    sources = getattr(config, "sources", None)
    crawl = getattr(config, "crawl", None)
    chunking = getattr(config, "chunking", None)
    index = getattr(config, "index", None)
    cleaning = getattr(config, "cleaning", None)
    dedup = getattr(config, "deduplication", None)

    summary: dict[str, Any] = {
        "available": True,
        "client_name": getattr(config, "client_name", None),
        "sources": {
            "url_count": len(getattr(sources, "urls", []) or []),
            "allowed_domains": sorted(getattr(sources, "allowed_domains", []) or []),
            "files_dir_configured": bool(getattr(sources, "files_dir", None)),
        },
        "crawl": {},
        "chunking": {},
        "index": {},
        "cleaning": {},
        "deduplication": {},
    }
    if crawl is not None:
        summary["crawl"] = {
            "max_depth": getattr(crawl, "max_depth", None),
            "max_pages": getattr(crawl, "max_pages", None),
            "delay_seconds": getattr(crawl, "delay_seconds", None),
            "timeout_seconds": getattr(crawl, "timeout_seconds", None),
            "retries": getattr(crawl, "retries", None),
        }
    if chunking is not None:
        summary["chunking"] = {
            "strategy": getattr(chunking, "strategy", None),
            "size": getattr(chunking, "size", None),
            "overlap": getattr(chunking, "overlap", None),
        }
    if index is not None:
        summary["index"] = {
            "bm25": getattr(index, "bm25", None),
            "dense": getattr(index, "dense", None),
            "embedding_model": getattr(index, "embedding_model", None),
        }
    if cleaning is not None:
        summary["cleaning"] = {"enabled": getattr(cleaning, "enabled", None)}
    if dedup is not None:
        summary["deduplication"] = {
            "enabled": getattr(dedup, "enabled", None),
            "exact": getattr(dedup, "exact", None),
            "near_duplicate": getattr(dedup, "near_duplicate", None),
            "similarity_threshold": getattr(dedup, "similarity_threshold", None),
            "shingle_size": getattr(dedup, "shingle_size", None),
            "num_perm": getattr(dedup, "num_perm", None),
        }
    return summary


def _build_statistics(output_dir: Path, chunks_count: int, index_stats: dict) -> dict:
    """Aggregate counts from earlier phases without copying volatile data."""
    stats: dict[str, Any] = {
        "chunks": chunks_count,
        "index": {
            "chunks_indexed": index_stats.get("N"),
            "average_document_length": index_stats.get("avgdl"),
        },
    }

    def optional_json(name: str) -> dict | None:
        path = output_dir / name
        if not path.exists():
            return None
        try:
            data = _read_json(path, name)
            return data if isinstance(data, dict) else None
        except PackageError:
            logger.warning("could not read optional build statistics from %s", name)
            return None

    crawl = optional_json("crawl_report.json")
    if crawl:
        stats["ingestion"] = {
            key: crawl.get(key)
            for key in (
                "pages_fetched",
                "pages_failed",
                "files_loaded",
                "files_failed",
                "documents_written",
            )
            if key in crawl
        }

    phase2 = optional_json("phase2_stats.json")
    if phase2:
        stats["processing"] = phase2

    for key, filename in (
        ("documents_raw", "documents.jsonl"),
        ("documents_cleaned", "documents_cleaned.jsonl"),
    ):
        path = output_dir / filename
        if path.exists():
            try:
                stats[key] = _count_lines(path)
            except (OSError, UnicodeDecodeError):
                logger.warning("could not count lines in %s", filename)

    return stats


def build_package(
    config=None,
    output_dir: str | Path | None = None,
    package_dir: str | Path | None = None,
) -> dict:
    """Create/refresh package metadata for an already-built knowledge base.

    Existing documents, chunks and index files are never modified or copied.
    Re-running this function only rewrites the three metadata files and is
    deterministic for unchanged inputs.
    """
    if output_dir is None:
        if config is None or not getattr(config, "output_dir", None):
            raise ValueError("output_dir is required when config is not supplied")
        output_dir = config.output_dir
    output_dir = Path(output_dir)
    package_root = Path(package_dir) if package_dir is not None else output_dir
    package_root.mkdir(parents=True, exist_ok=True)

    index_dir = output_dir / "index"
    chunks_path = output_dir / "chunks.jsonl"

    missing: list[str] = []
    if not chunks_path.is_file():
        missing.append("chunks.jsonl")
    if not index_dir.is_dir():
        missing.append("index/")
    for name in INDEX_FILENAMES:
        if not (index_dir / name).is_file():
            missing.append(f"index/{name}")
    if missing:
        raise MissingArtifactError(
            "cannot package knowledge base; missing required artifacts: "
            + ", ".join(sorted(missing))
        )

    chunks_count = _validate_chunks(chunks_path)

    index_stats = _read_json(index_dir / "index_stats.json", "index statistics")
    if not isinstance(index_stats, dict):
        raise PackageError("invalid index statistics: expected a JSON object")

    config_summary = _config_summary(config)
    config_summary["format"] = BUILD_CONFIG_FORMAT
    config_summary["format_version"] = PACKAGE_FORMAT_VERSION
    config_path = package_root / BUILD_CONFIG_FILENAME
    _guard_metadata_path(config_path, BUILD_CONFIG_FORMAT, "build configuration")

    build_stats = _build_statistics(output_dir, chunks_count, index_stats)
    build_stats["format"] = BUILD_STATS_FORMAT
    build_stats["format_version"] = PACKAGE_FORMAT_VERSION
    stats_path = package_root / BUILD_STATS_FILENAME
    _guard_metadata_path(stats_path, BUILD_STATS_FORMAT, "build statistics")

    artifacts: dict[str, dict] = {
        "chunks": _artifact(
            chunks_path, package_root, "jsonl", required=True, records=chunks_count
        ),
        "index": _artifact(index_dir, package_root, "directory", required=True),
        "index/postings": _artifact(
            index_dir / "postings.json", package_root, "json", required=True
        ),
        "index/doc_lengths": _artifact(
            index_dir / "doc_lengths.json", package_root, "json", required=True
        ),
        "index/chunk_info": _artifact(
            index_dir / "chunk_info.json", package_root, "json", required=True
        ),
        "index/df": _artifact(index_dir / "df.json", package_root, "json", required=True),
        "index/index_stats": _artifact(
            index_dir / "index_stats.json", package_root, "json", required=True
        ),
        "build_config": _artifact(config_path, package_root, "json", required=True),
        "build_stats": _artifact(stats_path, package_root, "json", required=True),
    }

    optional_files = {
        "documents": (output_dir / "documents.jsonl", "jsonl"),
        "documents_cleaned": (output_dir / "documents_cleaned.jsonl", "jsonl"),
        "crawl_report": (output_dir / "crawl_report.json", "json"),
        "link_graph": (output_dir / "link_graph.json", "json"),
        "phase2_stats": (output_dir / "phase2_stats.json", "json"),
        "removal_log": (output_dir / "removal_log.json", "json"),
    }
    for key, (path, fmt) in optional_files.items():
        if path.is_file():
            artifacts[key] = _artifact(path, package_root, fmt, required=False)

    manifest = {
        "format": PACKAGE_FORMAT,
        "format_version": PACKAGE_FORMAT_VERSION,
        "kbforge_version": __version__,
        "client_name": getattr(config, "client_name", None),
        "ranking_modes": list(SUPPORTED_RANKING_MODES),
        "statistics": {
            "chunks": chunks_count,
            "chunks_indexed": index_stats.get("N"),
            "average_document_length": index_stats.get("avgdl"),
        },
        "artifacts": artifacts,
    }
    manifest_path = package_root / MANIFEST_FILENAME
    _guard_metadata_path(manifest_path, PACKAGE_FORMAT, "package manifest")
    _write_json(config_path, config_summary)
    _write_json(stats_path, build_stats)
    _write_json(manifest_path, manifest)

    return {
        "package_dir": str(package_root),
        "manifest_path": str(manifest_path),
        "build_config_path": str(config_path),
        "build_stats_path": str(stats_path),
        "chunks": chunks_count,
        "ranking_modes": list(SUPPORTED_RANKING_MODES),
    }


def find_manifest(path: str | Path) -> Path | None:
    """Return the manifest path for a package/output directory, if present."""
    path = Path(path)
    if path.is_file():
        return path if path.name == MANIFEST_FILENAME else None
    for candidate in (path / MANIFEST_FILENAME, path / "package" / MANIFEST_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def _load_manifest(path: str | Path) -> Manifest:
    manifest_path = find_manifest(path)
    if manifest_path is None:
        raise MissingArtifactError(
            f"no {MANIFEST_FILENAME} found in {path}"
        )
    data = _read_json(manifest_path, "package manifest")
    if not isinstance(data, dict):
        raise IncompatiblePackageError("invalid package manifest: expected a JSON object")
    if data.get("format") != PACKAGE_FORMAT:
        raise IncompatiblePackageError(
            f"not a kbforge package (expected format '{PACKAGE_FORMAT}')"
        )
    version = data.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise IncompatiblePackageError(
            "unsupported package format version "
            f"{version!r} (supported: {', '.join(SUPPORTED_FORMAT_VERSIONS)})"
        )
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise IncompatiblePackageError("invalid package manifest: missing 'artifacts'")

    manifest = Manifest(path=manifest_path, data=data)
    missing: list[str] = []
    for key in REQUIRED_ARTIFACT_KEYS:
        entry = artifacts.get(key)
        if not isinstance(entry, dict) or not entry.get("path"):
            missing.append(key)
            continue
        resolved = manifest.artifact_path(key)
        if not resolved.exists():
            missing.append(key)
    if missing:
        raise MissingArtifactError(
            "package is missing required artifacts: " + ", ".join(sorted(missing))
        )
    return manifest


def _validate_json_artifact(path: Path, description: str) -> Any:
    data = _read_json(path, description)
    return data


def _validate_chunks(path: Path, *, collect_text: bool = False) -> int | dict[str, str]:
    count = 0
    chunk_text: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PackageError(
                        f"invalid chunks artifact: line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise PackageError(
                        f"invalid chunks artifact: line {line_number} is not an object"
                    )
                chunk_id = record.get("chunk_id") or record.get("id")
                if not chunk_id:
                    raise PackageError(
                        f"invalid chunks artifact: line {line_number} has no chunk_id"
                    )
                if collect_text:
                    text = record.get("text", "")
                    chunk_text[str(chunk_id)] = text if isinstance(text, str) else ""
                count += 1
    except FileNotFoundError as exc:
        raise MissingArtifactError(f"missing chunks artifact: {path.name}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise PackageError(f"invalid chunks artifact: {exc}") from exc
    return chunk_text if collect_text else count


def load_package(path: str | Path) -> "KnowledgeBase":
    """Load and validate a package without crawling or rebuilding anything."""
    return KnowledgeBase.from_package(path)


class KnowledgeBase:
    """A loaded index plus the package metadata needed to serve queries."""

    def __init__(
        self,
        index: InvertedIndex,
        *,
        root: Path | None = None,
        manifest: dict | None = None,
        supported_modes: tuple[str, ...] = SUPPORTED_RANKING_MODES,
        chunks_path: Path | None = None,
        chunk_texts: dict[str, str] | None = None,
    ):
        self.index = index
        self.root = root
        self.manifest = manifest or {}
        self.supported_modes = tuple(supported_modes)
        self.chunks_path = chunks_path
        self.chunk_texts = chunk_texts or {}
        self._rankers: dict[str, BM25 | TFIDF] = {}

    @classmethod
    def from_index(cls, index_dir: str | Path) -> "KnowledgeBase":
        index_dir = Path(index_dir)
        if not (index_dir / "postings.json").is_file():
            raise MissingArtifactError(f"index not found in {index_dir}")
        try:
            for name in INDEX_FILENAMES:
                _validate_json_artifact(index_dir / name, f"index artifact '{name}'")
            index = InvertedIndex.load(index_dir)
        except PackageError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PackageError(f"could not load index: {exc}") from exc

        chunk_texts: dict[str, str] = {}
        chunks_path = index_dir.parent / "chunks.jsonl"
        if chunks_path.is_file():
            try:
                loaded = _validate_chunks(chunks_path, collect_text=True)
                if isinstance(loaded, dict):
                    chunk_texts = loaded
            except PackageError as exc:
                logger.warning("could not load chunk text from %s: %s", chunks_path, exc)
        return cls(index, root=index_dir.parent, chunks_path=chunks_path, chunk_texts=chunk_texts)

    @classmethod
    def from_package(cls, path: str | Path) -> "KnowledgeBase":
        manifest = _load_manifest(path)
        artifacts = manifest.data["artifacts"]

        chunks_path = manifest.artifact_path("chunks")
        chunk_texts = _validate_chunks(chunks_path, collect_text=True)
        if not isinstance(chunk_texts, dict):
            chunk_texts = {}

        # Validate the small metadata artifacts eagerly so corrupt files are
        # reported with a useful name rather than during the first query.
        _validate_json_artifact(manifest.artifact_path("build_config"), "build configuration")
        _validate_json_artifact(manifest.artifact_path("build_stats"), "build statistics")

        index_stats_path = manifest.artifact_path("index/index_stats")
        index_stats = _validate_json_artifact(index_stats_path, "index statistics")
        if not isinstance(index_stats, dict) or "N" not in index_stats or "avgdl" not in index_stats:
            raise PackageError("invalid index statistics: 'N' and 'avgdl' are required")

        for key in (
            "index/postings",
            "index/doc_lengths",
            "index/chunk_info",
            "index/df",
        ):
            _validate_json_artifact(manifest.artifact_path(key), f"index artifact '{key}'")

        index_dir_entry = artifacts.get("index")
        if isinstance(index_dir_entry, dict) and index_dir_entry.get("path"):
            index_dir = manifest.artifact_path("index")
        else:
            index_dir = index_stats_path.parent
        try:
            index = InvertedIndex.load(index_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PackageError(f"could not load package index: {exc}") from exc

        modes = manifest.data.get("ranking_modes", list(SUPPORTED_RANKING_MODES))
        if not isinstance(modes, list) or not modes:
            raise IncompatiblePackageError("manifest has no supported ranking modes")
        unknown = [mode for mode in modes if mode not in SUPPORTED_RANKING_MODES]
        if unknown:
            raise IncompatiblePackageError(
                "manifest lists unsupported ranking modes: " + ", ".join(map(str, unknown))
            )

        return cls(
            index,
            root=manifest.root,
            manifest=manifest.data,
            supported_modes=tuple(modes),
            chunks_path=chunks_path,
            chunk_texts=chunk_texts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        """Load a package directory, or a bare index directory, read-only."""
        if find_manifest(path) is not None:
            return cls.from_package(path)
        path = Path(path)
        if (path / "postings.json").is_file():
            return cls.from_index(path)
        index_dir = path / "index"
        if (index_dir / "postings.json").is_file():
            return cls.from_index(index_dir)
        raise MissingArtifactError(
            f"no package manifest or index found in {path}. "
            "Run 'kbforge build' and 'kbforge package' first."
        )

    def _ranker(self, mode: str):
        if mode not in self.supported_modes:
            raise ValueError(
                f"unsupported ranking mode '{mode}'; "
                f"available: {', '.join(self.supported_modes)}"
            )
        if mode not in self._rankers:
            if mode == "bm25":
                self._rankers[mode] = BM25(self.index)
            elif mode == "tfidf":
                self._rankers[mode] = TFIDF(self.index)
            else:  # defensive; supported_modes is validated on load
                raise ValueError(f"unsupported ranking mode '{mode}'")
        return self._rankers[mode]

    def search(
        self,
        query: str,
        mode: str = "bm25",
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search the loaded index using the existing BM25/TF-IDF rankers."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("result limit must be an integer")
        if limit <= 0:
            raise ValueError("result limit must be greater than 0")
        ranker = self._ranker(mode)
        if not isinstance(query, str) or not query.strip():
            return []
        results = ranker.score(query.strip())[:limit]
        # The Phase 4 index stores metadata but not chunk text; restore it
        # from chunks.jsonl so callers can reconstruct result snippets.
        for result in results:
            if not result.text:
                result.text = self.chunk_texts.get(result.chunk_id, "")
        return results
