"""Shared test helpers: a fake HTTP fetcher so crawler tests never hit the network."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when running pytest without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import pytest

from kbforge.chunk import chunk_documents, chunk_to_dict
from kbforge.config import (
    ChunkingConfig,
    Config,
    CrawlConfig,
    IndexConfig,
    SourcesConfig,
)
from kbforge.documents import Document, write_documents_jsonl
from kbforge.index import InvertedIndex
from kbforge.ingest.crawler import FetchResult


@dataclass
class FakeRoute:
    status: int = 200
    body: bytes = b""
    content_type: str = "text/html; charset=utf-8"
    final_url: str | None = None

    def to_result(self, url: str) -> FetchResult:
        return FetchResult(
            url=url,
            status_code=self.status,
            headers={"Content-Type": self.content_type},
            content=self.body,
            final_url=self.final_url or url,
        )


class FakeFetcher:
    """Maps exact URLs to canned responses and records every call."""

    def __init__(self, routes: dict[str, FakeRoute]):
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float | None = None) -> FetchResult:
        self.calls.append(url)
        route = self.routes.get(url)
        if route is None:
            return FetchResult(url=url, status_code=404, headers={}, content=b"", final_url=url,
                               error="connection refused")
        return route.to_result(url)


def html_page(title: str, body: str, links: list[str] | None = None) -> bytes:
    anchors = "".join(f'<a href="{href}">link</a>' for href in (links or []))
    page = f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{body}</p>{anchors}</body></html>"
    return page.encode("utf-8")


def make_config(
    urls: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    max_depth: int = 2,
    max_pages: int = 50,
    delay_seconds: float = 0.0,
    files_dir: str | None = None,
    output_dir: str = "output/test",
) -> Config:
    return Config(
        client_name="test",
        sources=SourcesConfig(
            urls=urls or ["http://kb.test/"],
            allowed_domains=allowed_domains if allowed_domains is not None else ["kb.test"],
            files_dir=files_dir,
        ),
        crawl=CrawlConfig(max_depth=max_depth, max_pages=max_pages, delay_seconds=delay_seconds),
        chunking=ChunkingConfig(),
        index=IndexConfig(),
        output_dir=output_dir,
    )


@pytest.fixture
def sample_kb(tmp_path):
    """Build a tiny, completely offline knowledge base on disk.

    Returns ``(config, output_dir)``.  Tests can then package/load it without
    invoking any ingestion code or network access.
    """
    import json

    output_dir = tmp_path / "kb"
    output_dir.mkdir()

    specs = [
        (
            "https://kb.test/python",
            "Python guide",
            "Python is a programming language used for automation and data work.",
        ),
        (
            "https://kb.test/search",
            "Search notes",
            "An inverted index maps terms to chunks and supports BM25 ranking.",
        ),
        (
            "https://kb.test/other",
            "Other page",
            "Cooking recipes and gardening tips have nothing in common with search.",
        ),
    ]
    documents = [
        Document.from_parts(
            source=source,
            title=title,
            raw_text=text,
            fetched_at="2026-01-01T00:00:00+00:00",
            kind="web",
            content_type="txt",
        )
        for source, title, text in specs
    ]
    write_documents_jsonl(documents, output_dir / "documents.jsonl")
    write_documents_jsonl(documents, output_dir / "documents_cleaned.jsonl")

    chunks = chunk_documents(documents, strategy="fixed", size=50, overlap=0)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")

    index = InvertedIndex()
    index.build([chunk_to_dict(chunk) for chunk in chunks])
    index.save(output_dir / "index")

    (output_dir / "phase2_stats.json").write_text(
        json.dumps({"input_documents": 3, "output_documents": 3}),
        encoding="utf-8",
    )
    (output_dir / "crawl_report.json").write_text(
        json.dumps(
            {
                "pages_fetched": 3,
                "pages_failed": 0,
                "documents_written": 3,
                "files_loaded": 0,
                "files_failed": 0,
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    config = make_config(urls=["https://kb.test/"], output_dir=str(output_dir))
    return config, output_dir
