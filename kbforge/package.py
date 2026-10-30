from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kbforge import __version__
from kbforge.config import VALID_RANKING_MODES
from kbforge.index import BM25, InvertedIndex, SearchResult, TFIDF

logger = logging.getLogger("kbforge.package")

PACKAGE_FORMAT = "kbforge-knowledge-base"
PACKAGE_FORMAT_VERSION = "1.0"
SUPPORTED_FORMAT_VERSIONS = (PACKAGE_FORMAT_VERSION,)
SUPPORTED_RANKING_MODES = VALID_RANKING_MODES

MANIFEST_FILENAME = "manifest.json"
README_FILENAME = "README.md"
BUILD_CONFIG_FILENAME = "build_config.json"
BUILD_STATS_FILENAME = "build_stats.json"
BUILD_MARKER_FILENAME = ".kbforge-build-incomplete"
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

REQUIRED_ARTIFACT_FORMATS = {
    "chunks": "jsonl",
    "index": "directory",
    "index/postings": "json",
    "index/doc_lengths": "json",
    "index/chunk_info": "json",
    "index/df": "json",
    "index/index_stats": "json",
    "build_config": "json",
    "build_stats": "json",
}


class PackageError(Exception):
    """Base class for package problems (missing/corrupt/incompatible)."""


class MissingArtifactError(PackageError):
    """A required package artifact is missing."""


class IncompatiblePackageError(PackageError):
    """The package format or version is not supported."""


class IncompleteBuildError(PackageError):
    """The package was left in an incomplete build state."""


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


def mark_build_in_progress(output_dir: str | Path) -> Path:
    """Mark an output directory as containing a build that is not complete."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / BUILD_MARKER_FILENAME
    marker.write_text("kbforge build did not complete\n", encoding="utf-8")
    return marker


def _clear_build_marker(output_dir: str | Path) -> None:
    marker = Path(output_dir) / BUILD_MARKER_FILENAME
    try:
        marker.unlink()
    except FileNotFoundError:
        pass


def _raise_if_build_incomplete(directory: str | Path) -> None:
    marker = Path(directory) / BUILD_MARKER_FILENAME
    if marker.is_file():
        raise IncompleteBuildError(
            "package build is incomplete "
            f"(found {BUILD_MARKER_FILENAME}); "
            "rerun 'kbforge build' or 'kbforge package' successfully before loading"
        )


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
            "include_subdomains": getattr(crawl, "include_subdomains", True),
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
            "ranking_modes": list(
                getattr(index, "ranking_modes", SUPPORTED_RANKING_MODES)
            ),
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


def _configured_ranking_modes(config) -> list[str]:
    index = getattr(config, "index", None)
    modes = getattr(index, "ranking_modes", None)
    if not modes:
        return list(SUPPORTED_RANKING_MODES)
    return [str(mode) for mode in modes]


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
    mark_build_in_progress(package_root)
    if package_root != output_dir:
        mark_build_in_progress(output_dir)

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

    chunk_texts = _validate_chunks(chunks_path, collect_text=True)
    if not isinstance(chunk_texts, dict):
        chunk_texts = {}
    chunks_count = len(chunk_texts)
    chunk_ids = set(chunk_texts)

    index_stats = _read_index_stats(index_dir / "index_stats.json")
    chunk_info = _read_json_object(index_dir / "chunk_info.json", "index chunk metadata")
    doc_lengths = _read_json_object(index_dir / "doc_lengths.json", "index document lengths")
    _read_json_object(index_dir / "postings.json", "index postings")
    _read_json_object(index_dir / "df.json", "index document frequencies")
    _validate_index_inventory(
        chunks_count=chunks_count,
        chunk_ids=chunk_ids,
        index_stats=index_stats,
        chunk_info=chunk_info,
        doc_lengths=doc_lengths,
    )

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

    ranking_modes = _configured_ranking_modes(config)
    manifest = {
        "format": PACKAGE_FORMAT,
        "format_version": PACKAGE_FORMAT_VERSION,
        "kbforge_version": __version__,
        "client_name": getattr(config, "client_name", None),
        "ranking_modes": ranking_modes,
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
    _clear_build_marker(package_root)
    if package_root != output_dir:
        _clear_build_marker(output_dir)

    return {
        "package_dir": str(package_root),
        "manifest_path": str(manifest_path),
        "build_config_path": str(config_path),
        "build_stats_path": str(stats_path),
        "chunks": chunks_count,
        "ranking_modes": ranking_modes,
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
    _raise_if_build_incomplete(manifest_path.parent)
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
        if entry.get("format") != REQUIRED_ARTIFACT_FORMATS[key]:
            raise IncompatiblePackageError(
                f"manifest artifact '{key}' has format {entry.get('format')!r}; "
                f"expected {REQUIRED_ARTIFACT_FORMATS[key]!r}"
            )
        if entry.get("required") is not True:
            raise IncompatiblePackageError(
                f"manifest artifact '{key}' must be marked as required"
            )
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


def _read_json_object(path: Path, description: str) -> dict:
    data = _read_json(path, description)
    if not isinstance(data, dict):
        raise PackageError(f"invalid {description}: expected a JSON object")
    return data


def _read_index_stats(path: Path) -> dict:
    stats = _read_json_object(path, "index statistics")
    if "N" not in stats or "avgdl" not in stats:
        raise PackageError("invalid index statistics: 'N' and 'avgdl' are required")
    count = stats["N"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise PackageError("invalid index statistics: 'N' must be a non-negative integer")
    avgdl = stats["avgdl"]
    if (
        isinstance(avgdl, bool)
        or not isinstance(avgdl, (int, float))
        or not math.isfinite(avgdl)
        or avgdl < 0
    ):
        raise PackageError("invalid index statistics: 'avgdl' must be a finite non-negative number")
    return stats


def _summarize_ids(ids: set[str]) -> str:
    ordered = sorted(ids)
    shown = ", ".join(ordered[:5])
    if len(ordered) > 5:
        shown += f", ... ({len(ordered)} total)"
    return shown


def _validate_index_inventory(
    *,
    chunks_count: int,
    chunk_ids: set[str],
    index_stats: dict,
    chunk_info: dict,
    doc_lengths: dict,
) -> None:
    """Ensure chunk and index artifacts describe the same corpus."""
    info_count = len(chunk_info)
    if info_count != chunks_count:
        raise PackageError(
            "inconsistent package: chunks.jsonl contains "
            f"{chunks_count} chunk(s) but index/chunk_info.json contains {info_count}"
        )
    lengths_count = len(doc_lengths)
    if lengths_count != chunks_count:
        raise PackageError(
            "inconsistent package: chunks.jsonl contains "
            f"{chunks_count} chunk(s) but index/doc_lengths.json contains {lengths_count}"
        )
    index_count = index_stats["N"]
    if index_count != chunks_count:
        raise PackageError(
            "inconsistent package: index_stats.json has "
            f"N={index_count} but chunks.jsonl contains {chunks_count} chunk(s)"
        )

    info_ids = set(chunk_info)
    if info_ids != chunk_ids:
        missing = chunk_ids - info_ids
        extra = info_ids - chunk_ids
        details = []
        if missing:
            details.append(f"missing from chunk_info: {_summarize_ids(missing)}")
        if extra:
            details.append(f"unknown in chunk_info: {_summarize_ids(extra)}")
        raise PackageError(
            "inconsistent package: chunks.jsonl and index/chunk_info.json "
            "have different chunk IDs (" + "; ".join(details) + ")"
        )

    length_ids = set(doc_lengths)
    if length_ids != chunk_ids:
        missing = chunk_ids - length_ids
        extra = length_ids - chunk_ids
        details = []
        if missing:
            details.append(f"missing from doc_lengths: {_summarize_ids(missing)}")
        if extra:
            details.append(f"unknown in doc_lengths: {_summarize_ids(extra)}")
        raise PackageError(
            "inconsistent package: chunks.jsonl and index/doc_lengths.json "
            "have different chunk IDs (" + "; ".join(details) + ")"
        )


def _validate_manifest_counts(
    manifest_data: dict,
    *,
    chunks_count: int,
    index_count: int,
) -> None:
    chunk_entry = manifest_data.get("artifacts", {}).get("chunks", {})
    records = chunk_entry.get("records") if isinstance(chunk_entry, dict) else None
    if records is not None and records != chunks_count:
        raise PackageError(
            "inconsistent package manifest: chunks artifact records "
            f"{records!r} does not match chunks.jsonl count {chunks_count}"
        )

    statistics = manifest_data.get("statistics")
    if statistics is None:
        return
    if not isinstance(statistics, dict):
        raise PackageError("invalid package manifest: 'statistics' must be an object")
    expected = {
        "chunks": chunks_count,
        "chunks_indexed": index_count,
    }
    for field, actual in expected.items():
        if field in statistics and statistics[field] != actual:
            raise PackageError(
                "inconsistent package manifest: "
                f"statistics.{field}={statistics[field]!r} does not match {actual}"
            )


def _validate_build_stats_counts(
    build_stats: dict,
    *,
    chunks_count: int,
    index_count: int,
) -> None:
    if "chunks" in build_stats and build_stats["chunks"] != chunks_count:
        raise PackageError(
            "inconsistent build statistics: "
            f"chunks={build_stats['chunks']!r} does not match chunks.jsonl count {chunks_count}"
        )
    index_stats = build_stats.get("index")
    if index_stats is None:
        return
    if not isinstance(index_stats, dict):
        raise PackageError("invalid build statistics: 'index' must be an object")
    if (
        "chunks_indexed" in index_stats
        and index_stats["chunks_indexed"] != index_count
    ):
        raise PackageError(
            "inconsistent build statistics: "
            f"index.chunks_indexed={index_stats['chunks_indexed']!r} "
            f"does not match index_stats.json N={index_count}"
        )


def _validate_chunks(path: Path, *, collect_text: bool = False) -> int | dict[str, str]:
    count = 0
    chunk_text: dict[str, str] = {}
    seen: dict[str, int] = {}
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
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise PackageError(
                        f"invalid chunks artifact: line {line_number} has no chunk_id"
                    )
                if chunk_id in seen:
                    raise PackageError(
                        "invalid chunks artifact: duplicate chunk_id "
                        f"{chunk_id!r} at line {line_number} "
                        f"(first seen at line {seen[chunk_id]})"
                    )
                seen[chunk_id] = line_number
                if collect_text:
                    text = record.get("text", "")
                    chunk_text[chunk_id] = text if isinstance(text, str) else ""
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
        _raise_if_build_incomplete(index_dir.parent)
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
        chunks_count = len(chunk_texts)
        chunk_ids = set(chunk_texts)

        # Validate the small metadata artifacts eagerly so corrupt files are
        # reported with a useful name rather than during the first query.
        _validate_json_artifact(manifest.artifact_path("build_config"), "build configuration")
        build_stats = _read_json_object(
            manifest.artifact_path("build_stats"), "build statistics"
        )

        index_stats_path = manifest.artifact_path("index/index_stats")
        index_stats = _read_index_stats(index_stats_path)
        chunk_info = _read_json_object(
            manifest.artifact_path("index/chunk_info"), "index chunk metadata"
        )
        doc_lengths = _read_json_object(
            manifest.artifact_path("index/doc_lengths"), "index document lengths"
        )
        _validate_index_inventory(
            chunks_count=chunks_count,
            chunk_ids=chunk_ids,
            index_stats=index_stats,
            chunk_info=chunk_info,
            doc_lengths=doc_lengths,
        )
        _validate_manifest_counts(
            manifest.data,
            chunks_count=chunks_count,
            index_count=index_stats["N"],
        )
        _validate_build_stats_counts(
            build_stats,
            chunks_count=chunks_count,
            index_count=index_stats["N"],
        )

        for key in (
            "index/postings",
            "index/df",
        ):
            _read_json_object(manifest.artifact_path(key), f"index artifact '{key}'")

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


def _artifact_relative_path(
    manifest: Manifest,
    key: str,
    entry: Any,
    *,
    required: bool,
) -> PurePosixPath | None:
    """Return a safe, normalized artifact path relative to the manifest root."""
    if not isinstance(entry, dict):
        if required:
            raise IncompatiblePackageError(f"invalid manifest artifact '{key}'")
        return None
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        if required:
            raise MissingArtifactError(f"manifest has no artifact path for '{key}'")
        return None
    if "\\" in raw_path:
        raise PackageError(
            f"unsafe artifact path for '{key}': backslashes are not allowed"
        )

    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PackageError(
            f"unsafe artifact path for '{key}': absolute paths and '..' are not allowed"
        )
    if len(relative.parts[0]) == 2 and relative.parts[0][1] == ":":
        raise PackageError(
            f"unsafe artifact path for '{key}': drive-qualified paths are not allowed"
        )

    candidate = manifest.root / relative
    if candidate.is_symlink():
        raise PackageError(f"artifact '{key}' must not be a symlink: {raw_path}")
    resolved_root = manifest.root.resolve()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise PackageError(f"could not resolve artifact '{key}': {exc}") from exc
    if not resolved.is_relative_to(resolved_root):
        raise PackageError(
            f"artifact '{key}' resolves outside the package root: {raw_path}"
        )
    return relative


def _add_export_file(
    files: dict[str, Path],
    arcname: str,
    source: Path,
    description: str,
) -> None:
    """Add one validated file to the export set, rejecting unsafe entry names."""
    relative = PurePosixPath(arcname)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "\\" in arcname
    ):
        raise PackageError(f"unsafe archive entry name for {description}: {arcname!r}")
    if len(relative.parts[0]) == 2 and relative.parts[0][1] == ":":
        raise PackageError(
            f"unsafe archive entry name for {description}: {arcname!r}"
        )

    existing = files.get(arcname)
    if existing is not None:
        if existing.resolve() != source.resolve():
            raise PackageError(
                f"duplicate archive entry {arcname!r} for {description}"
            )
        return
    files[arcname] = source


def _validate_optional_export_file(
    path: Path,
    fmt: str,
    description: str,
) -> None:
    """Ensure a present optional file is valid before including it."""
    if fmt == "json":
        _read_json(path, description)
        return
    if fmt == "jsonl":
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise PackageError(
                            f"invalid {description} '{path.name}': "
                            f"line {line_number}: {exc.msg}"
                        ) from exc
        except FileNotFoundError as exc:
            raise MissingArtifactError(
                f"missing {description}: {path.name}"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise PackageError(f"invalid {description} '{path.name}': {exc}") from exc
        return
    raise IncompatiblePackageError(
        f"invalid {description}: unsupported format {fmt!r}"
    )


def _add_directory_export_files(
    files: dict[str, Path],
    directory: Path,
    prefix: PurePosixPath,
    description: str,
    resolved_root: Path,
) -> None:
    """Add regular files below a manifest-listed directory artifact."""
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            child = current_path / dirname
            if child.is_symlink():
                raise PackageError(
                    f"artifact '{description}' contains a symlink: "
                    f"{child.relative_to(directory).as_posix()}"
                )
        for filename in filenames:
            source = current_path / filename
            if source.is_symlink():
                raise PackageError(
                    f"artifact '{description}' contains a symlink: "
                    f"{source.relative_to(directory).as_posix()}"
                )
            try:
                resolved = source.resolve()
            except OSError as exc:
                raise PackageError(
                    f"could not resolve artifact file '{source.name}': {exc}"
                ) from exc
            if not resolved.is_relative_to(resolved_root):
                raise PackageError(
                    f"artifact '{description}' contains a path outside the package root: "
                    f"{source.relative_to(directory).as_posix()}"
                )
            if not source.is_file():
                raise PackageError(
                    f"artifact '{description}' contains a non-regular file: "
                    f"{source.relative_to(directory).as_posix()}"
                )
            relative = source.relative_to(directory)
            arcname = PurePosixPath(prefix, *relative.parts).as_posix()
            _add_export_file(files, arcname, source, f"artifact '{description}'")


def _collect_export_files(manifest: Manifest) -> dict[str, Path]:
    """Build the manifest allowlist of files to include in an archive."""
    if manifest.path.is_symlink():
        raise PackageError("package manifest must not be a symlink")
    resolved_root = manifest.root.resolve()
    try:
        resolved_manifest = manifest.path.resolve()
    except OSError as exc:
        raise PackageError(f"could not resolve package manifest: {exc}") from exc
    if not resolved_manifest.is_relative_to(resolved_root):
        raise PackageError("package manifest resolves outside the package root")

    files: dict[str, Path] = {}
    _add_export_file(files, MANIFEST_FILENAME, manifest.path, "package manifest")

    artifacts = manifest.data["artifacts"]
    for key in sorted(artifacts):
        entry = artifacts[key]
        required = key in REQUIRED_ARTIFACT_KEYS
        relative = _artifact_relative_path(manifest, key, entry, required=required)
        if relative is None:
            continue

        candidate = manifest.root / relative
        if not candidate.exists():
            if required:
                raise MissingArtifactError(f"missing artifact '{key}': {relative.as_posix()}")
            logger.warning("skipping missing optional artifact '%s': %s", key, relative)
            continue
        if candidate.is_symlink():
            raise PackageError(
                f"artifact '{key}' must not be a symlink: {relative.as_posix()}"
            )

        fmt = entry.get("format")
        if not isinstance(fmt, str) or fmt not in {"json", "jsonl", "directory"}:
            raise IncompatiblePackageError(
                f"invalid manifest artifact '{key}': unsupported format {fmt!r}"
            )
        if candidate.is_dir():
            if fmt != "directory":
                raise IncompatiblePackageError(
                    f"invalid manifest artifact '{key}': expected a file, found a directory"
                )
            _add_directory_export_files(
                files, candidate, relative, key, resolved_root
            )
        elif candidate.is_file():
            if fmt == "directory":
                raise IncompatiblePackageError(
                    f"invalid manifest artifact '{key}': expected a directory, found a file"
                )
            if not required:
                _validate_optional_export_file(
                    candidate, fmt, f"optional artifact '{key}'"
                )
            _add_export_file(
                files, relative.as_posix(), candidate, f"artifact '{key}'"
            )
        else:
            raise PackageError(
                f"artifact '{key}' is not a regular file or directory: "
                f"{relative.as_posix()}"
            )

    return files


def _first_nonnegative_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _included_optional_artifact_paths(
    manifest: Manifest,
    file_names: set[str],
) -> list[str]:
    paths: list[str] = []
    for key, entry in sorted(manifest.data["artifacts"].items()):
        if not isinstance(entry, dict) or entry.get("required") is True:
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        arcname = PurePosixPath(raw_path).as_posix()
        if arcname in file_names or any(
            name.startswith(arcname + "/") for name in file_names
        ):
            paths.append(arcname)
    return paths


def _render_package_readme(
    manifest: Manifest,
    *,
    file_names: set[str],
    kb: KnowledgeBase,
) -> str:
    data = manifest.data
    build_stats = _read_json_object(
        manifest.artifact_path("build_stats"), "build statistics"
    )
    statistics = data.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
    index_stats = build_stats.get("index")
    if not isinstance(index_stats, dict):
        index_stats = {}
    processing = build_stats.get("processing")
    if not isinstance(processing, dict):
        processing = {}
    ingestion = build_stats.get("ingestion")
    if not isinstance(ingestion, dict):
        ingestion = {}

    chunks = _first_nonnegative_int(
        statistics.get("chunks"),
        build_stats.get("chunks"),
        kb.index.N,
    )
    chunks_indexed = _first_nonnegative_int(
        statistics.get("chunks_indexed"),
        index_stats.get("chunks_indexed"),
        kb.index.N,
    )
    documents = _first_nonnegative_int(
        build_stats.get("documents_cleaned"),
        build_stats.get("documents_raw"),
        processing.get("output_documents"),
        processing.get("documents_cleaned"),
        ingestion.get("documents_written"),
    )
    modes = list(kb.supported_modes)
    example_mode = modes[0] if modes else "bm25"
    mode_labels = {"bm25": "BM25", "tfidf": "TF-IDF"}

    lines = [
        "# KBForge Knowledge Base",
        "",
        "This ZIP contains a prebuilt KBForge lexical search package. The index "
        "can be loaded and searched locally without crawling, cleaning, chunking, "
        "or rebuilding the source data.",
        "",
        "## Package details",
        "",
    ]
    client_name = data.get("client_name")
    if isinstance(client_name, str) and client_name.strip():
        lines.append(f"- Client: {client_name.strip()}")
    kbforge_version = data.get("kbforge_version")
    if isinstance(kbforge_version, str) and kbforge_version.strip():
        lines.append(f"- KBForge version: {kbforge_version.strip()}")
    lines.append(
        f"- Package format version: {data.get('format_version', 'unknown')}"
    )
    if documents is not None:
        lines.append(f"- Documents: {documents}")
    if chunks is not None:
        lines.append(f"- Chunks: {chunks}")
    if chunks_indexed is not None and chunks_indexed != chunks:
        lines.append(f"- Indexed chunks: {chunks_indexed}")
    lines.append(f"- Supported ranking modes: {', '.join(modes)}")

    optional_paths = _included_optional_artifact_paths(manifest, file_names)
    if optional_paths:
        lines.extend(["", "Optional artifacts included in this archive:", ""])
        lines.extend(f"- `{path}`" for path in optional_paths)

    lines.extend(
        [
            "",
            "## What is included",
            "",
            "- `manifest.json` - package manifest and artifact inventory.",
            "- `build_config.json` - safe summary of the build settings.",
            "- `build_stats.json` - build and indexing counts.",
            "- `chunks.jsonl` - chunk text and metadata used to reconstruct results.",
            "- `index/` - prebuilt lexical index:",
            "  - `postings.json`",
            "  - `doc_lengths.json`",
            "  - `chunk_info.json`",
            "  - `df.json`",
            "  - `index_stats.json`",
            "- `README.md` - this guide.",
            "",
            "Raw pages and files that are not listed in `manifest.json` are not part "
            "of the package.",
            "",
            "## Prerequisites",
            "",
            "- Python 3.10 or newer.",
            "- A working KBForge installation and its runtime dependencies. This ZIP "
            "contains package data only; it does not bundle the `kbforge` Python "
            "package or a standalone executable.",
            "- The examples below assume `kbforge` is available on your `PATH`. "
            "Activate your virtual environment first if needed.",
            "",
            "## Extract the archive",
            "",
            "```bash",
            "unzip YOUR-ARCHIVE.zip -d kb-package",
            "cd kb-package",
            "```",
            "",
            "Replace `YOUR-ARCHIVE.zip` with the delivered archive name. On systems "
            "without `unzip`, use the operating system archive tool. Keep the "
            "extracted directory structure intact: `manifest.json` must remain at "
            "the package root next to the `index/` directory.",
            "",
            "## Validate the package",
            "",
            "There is no separate validation subcommand. Loading the package through "
            "an existing KBForge command performs validation. From the extracted "
            "package root:",
            "",
            "```bash",
            f'kbforge query "test query" --package . --mode {example_mode} --k 1',
            "```",
            "",
            "This loads and checks `manifest.json`, chunks, index files, artifact "
            "counts, and ranking modes before running the search. A broken package "
            "exits non-zero with an error message.",
            "",
            "To validate from Python:",
            "",
            "```bash",
            "python3 -c \"from kbforge.package import KnowledgeBase; "
            "kb = KnowledgeBase.load('.'); print('Indexed chunks:', kb.index.N)\"",
            "```",
            "",
            "## Search from the command line",
            "",
            "The `--package .` option loads the package from the current directory. "
            "The `--k` option controls the number of results (default 5).",
            "",
        ]
    )
    for mode in modes:
        label = mode_labels.get(mode, mode.upper())
        lines.extend(
            [
                f"### {label}",
                "",
                "```bash",
                f'kbforge query "your search terms" --package . --mode {mode} --k 5',
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Search with the HTTP API",
            "",
            "Start the local search API from the extracted package root:",
            "",
            "```bash",
            "kbforge serve --package . --host 127.0.0.1 --port 8000",
            "```",
            "",
            "Check service health:",
            "",
            "```bash",
            "curl http://127.0.0.1:8000/health",
            "```",
            "",
            "Search using `POST /search`:",
            "",
            "```bash",
            "curl -X POST http://127.0.0.1:8000/search \\",
            '  -H "Content-Type: application/json" \\',
            f'  -d \'{{"query": "your search terms", "mode": "{example_mode}", "limit": 5}}\'',
            "```",
            "",
            "The API accepts the supported ranking modes listed above and a `limit` "
            "from 1 to 100. `GET /health` reports the package version, chunk count, "
            "and ranking modes. If the package cannot be loaded, the API returns "
            "HTTP 503.",
            "",
            "## Limitations",
            "",
            "- This package supports lexical retrieval only, using the ranking modes "
            "listed above. It does not include dense or vector search.",
            "- The index is static. The package does not crawl, rebuild, or update "
            "the source data.",
            "- Search quality depends on the source content and the original chunking "
            "configuration.",
            "- The archive does not include a Python runtime or the KBForge "
            "application; those must be installed separately.",
            "",
            "## Troubleshooting",
            "",
            "- `kbforge: command not found`: activate the environment where KBForge "
            "is installed, or add it to `PATH`.",
            "- Package load errors: confirm that you are in the extracted package root "
            "and that `manifest.json` and `index/` are present. Do not rename or move "
            "files inside the package.",
            "- HTTP 503 from `/health` or `/search`: start `kbforge serve` from the "
            "extracted root, or pass an absolute path to `--package`.",
            "- No search results: try different or broader terms, or another "
            "supported ranking mode.",
            "",
        ]
    )
    return "\n".join(lines)


def export_package(
    path: str | Path,
    archive_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Export a validated package as a portable ZIP archive.

    The manifest is the archive allowlist: the package manifest itself and
    every manifest-listed artifact are included, while unrelated files such
    as ``pages/`` are excluded.  The archive is written atomically and is
    never placed inside the source package.
    """
    manifest = _load_manifest(path)
    package_root = manifest.root
    destination = Path(archive_path)

    package_root_abs = package_root.absolute()
    destination_abs = destination.absolute()
    if (
        destination_abs == package_root_abs
        or destination_abs.is_relative_to(package_root_abs)
    ):
        raise PackageError(
            "archive destination must be outside the source package directory"
        )

    resolved_root = package_root.resolve()
    try:
        resolved_destination = destination.resolve()
    except OSError as exc:
        raise PackageError(f"could not resolve archive destination: {exc}") from exc
    if (
        resolved_destination == resolved_root
        or resolved_destination.is_relative_to(resolved_root)
    ):
        raise PackageError(
            "archive destination must be outside the source package directory"
        )

    if os.path.lexists(destination):
        if destination.is_dir():
            raise PackageError(f"archive destination is a directory: {destination}")
        if not overwrite:
            raise PackageError(
                f"refusing to overwrite existing archive: {destination}; "
                "use overwrite=True to replace it"
            )

    files = _collect_export_files(manifest)
    kb = KnowledgeBase.from_package(manifest.path)
    if README_FILENAME in files:
        raise PackageError(
            f"package already contains a manifest-listed {README_FILENAME}; "
            "cannot add the generated client README"
        )
    readme = _render_package_readme(
        manifest,
        file_names=set(files),
        kb=kb,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(fd)
    except OSError as exc:
        raise PackageError(f"could not prepare archive destination: {exc}") from exc

    tmp_path: Path | None = Path(tmp_name)
    try:
        with zipfile.ZipFile(
            tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(README_FILENAME, readme)
            for arcname in sorted(files):
                archive.write(files[arcname], arcname=arcname)
        os.replace(tmp_path, destination)
        tmp_path = None
    except Exception as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError(f"could not write archive '{destination}': {exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    return {
        "archive_path": str(destination),
        "archive_size": destination.stat().st_size,
        "file_count": len(files) + 1,
    }
