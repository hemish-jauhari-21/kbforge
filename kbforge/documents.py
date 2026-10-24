"""Shared document schema used by every ingestion source.

Every document that enters the pipeline (crawled page or uploaded file)
is represented as a :class:`Document`. Later phases (clean, chunk, index)
consume the JSONL serialization of these objects.

Design notes:
- ``raw_text`` holds the raw HTML for crawled pages and the extracted
  plain text for files. Phase 2 extracts main content from HTML.
- ``content_hash`` is the SHA-256 of ``raw_text`` encoded as UTF-8.
  It is used for exact-duplicate detection.
- ``doc_id`` is a short stable identifier derived from the source URL or
  path, so re-builds produce the same ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

KIND_WEB = "web"
KIND_FILE = "file"


def compute_content_hash(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_doc_id(kind: str, source: str) -> str:
    """Short deterministic id: ``web-<sha1[:12]>`` or ``file-<sha1[:12]>``."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}"


@dataclass
class Document:
    doc_id: str
    source: str                 # URL for web pages, filesystem path for files
    title: str
    raw_text: str               # raw HTML (web) or extracted text (files)
    fetched_at: str             # ISO 8601 UTC timestamp
    content_hash: str
    kind: str                   # "web" or "file"
    content_type: str           # "html", "pdf", "docx", "txt"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        source: str,
        title: str,
        raw_text: str,
        fetched_at: str,
        kind: str,
        content_type: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> "Document":
        return cls(
            doc_id=doc_id or stable_doc_id(kind, source),
            source=source,
            title=title,
            raw_text=raw_text,
            fetched_at=fetched_at,
            content_hash=compute_content_hash(raw_text),
            kind=kind,
            content_type=content_type,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            title=data.get("title", ""),
            raw_text=data["raw_text"],
            fetched_at=data["fetched_at"],
            content_hash=data["content_hash"],
            kind=data["kind"],
            content_type=data["content_type"],
            metadata=data.get("metadata", {}),
        )


def write_documents_jsonl(documents: Iterable[Document], path: str | Path) -> int:
    """Write documents to a JSONL file, one object per line. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_documents_jsonl(path: str | Path) -> list[Document]:
    """Read a JSONL file of documents back into Document objects."""
    documents: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                documents.append(Document.from_dict(json.loads(line)))
    return documents
