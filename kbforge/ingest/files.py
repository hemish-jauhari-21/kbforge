from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from kbforge.documents import KIND_FILE, Document

logger = logging.getLogger("kbforge.files")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


def _file_fetched_at(path: Path) -> str:
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds")


def load_pdf(path: Path) -> str:
    parts: list[str] = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            parts.append(page.get_text())
    return "\n".join(parts)


def load_docx(path: Path) -> tuple[str, str]:
    """Return (title, text). Title comes from core properties or the filename."""
    doc = DocxDocument(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    title = (doc.core_properties.title or "").strip() or path.stem
    return title, "\n".join(paragraphs)


def load_html_file(path: Path) -> tuple[str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return title, text


def load_file(path: Path) -> Document:
    """Load a single file into a Document. Raises ValueError for unknown types."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type: {path}")

    if suffix == ".pdf":
        text = load_pdf(path)
        content_type = "pdf"
        title = path.stem
    elif suffix == ".docx":
        title, text = load_docx(path)
        content_type = "docx"
    elif suffix in (".html", ".htm"):
        title, text = load_html_file(path)
        content_type = "html"
    else:  # .txt / .md
        text = path.read_text(encoding="utf-8", errors="replace")
        content_type = "txt"
        title = path.stem

    return Document.from_parts(
        source=str(path.resolve()),
        title=title,
        raw_text=text,
        fetched_at=_file_fetched_at(path),
        kind=KIND_FILE,
        content_type=content_type,
        metadata={"filename": path.name, "extension": suffix},
    )


def load_files(files_dir: str | Path, errors: list[dict] | None = None) -> list[Document]:
    """Recursively load every supported file under ``files_dir``.

    When ``errors`` is provided, failed paths and human-readable reasons are
    appended to it so the caller can include them in the ingestion report.
    Unsupported extensions are intentionally skipped, not treated as failures.
    """
    root = Path(files_dir)
    if not root.exists():
        message = f"files_dir does not exist: {root}"
        logger.warning(message)
        if errors is not None:
            errors.append({"path": str(root), "error": message})
        return []
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.append(load_file(path))
        except Exception as exc:  # one broken file must not kill the build
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("failed to load %s: %s", path, message)
            if errors is not None:
                errors.append({"path": str(path.resolve()), "error": message})
    return documents
