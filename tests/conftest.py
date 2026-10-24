"""Shared test helpers: a fake HTTP fetcher so crawler tests never hit the network."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when running pytest without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

from kbforge.config import (
    ChunkingConfig,
    Config,
    CrawlConfig,
    IndexConfig,
    SourcesConfig,
)
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
