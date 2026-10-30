from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib import robotparser
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kbforge.config import Config
from kbforge.documents import KIND_WEB, Document, write_documents_jsonl
from kbforge.urls import host_matches, normalize_url

logger = logging.getLogger("kbforge.crawler")

GRAPH_SAVE_EVERY = 25  # re-write link_graph.json every N pages (crash safety)

HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([^;\"'\s]+)", re.IGNORECASE)


def _declared_charset(headers: dict) -> str | None:
    """Return the charset declared in a Content-Type header, if any."""
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    match = _CHARSET_RE.search(str(content_type))
    return match.group(1) if match else None


def _decode_html(content: bytes, headers: dict) -> str:
    """Decode HTML using the declared charset, falling back to UTF-8."""
    charset = _declared_charset(headers) or "utf-8"
    try:
        return content.decode(charset, errors="replace")
    except LookupError:
        logger.warning("unknown charset %r; falling back to UTF-8", charset)
        return content.decode("utf-8", errors="replace")


@dataclass
class FetchResult:
    url: str
    status_code: int
    headers: dict
    content: bytes
    final_url: str
    error: str | None = None


Fetcher = Callable[[str, float | None], FetchResult]


def default_fetcher(user_agent: str, retries: int) -> Fetcher:
    """requests-based fetcher with retry and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})

    def fetch(url: str, timeout: float | None = None) -> FetchResult:
        try:
            response = session.get(url, timeout=timeout)
            return FetchResult(
                url=url,
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                final_url=str(response.url),
            )
        except requests.RequestException as exc:
            return FetchResult(
                url=url, status_code=0, headers={}, content=b"", final_url=url, error=str(exc)
            )

    return fetch


@dataclass
class LinkGraph:
    """Directed graph of hyperlinks seen during the crawl."""

    nodes: dict[str, dict] = field(default_factory=dict)  # url -> node info
    edges: list[dict] = field(default_factory=list)

    def add_node(self, url: str, external: bool, fetched: bool = False) -> None:
        node = self.nodes.setdefault(url, {"id": url, "external": external, "fetched": False})
        if fetched:
            node["fetched"] = True

    def add_edge(self, source: str, target: str, target_external: bool) -> None:
        self.edges.append({"source": source, "target": target, "external_target": target_external})

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": {
                "nodes": len(self.nodes),
                "internal_nodes": sum(1 for n in self.nodes.values() if not n["external"]),
                "external_nodes": sum(1 for n in self.nodes.values() if n["external"]),
                "edges": len(self.edges),
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["created_at"] = _utc_now()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class CrawlOutcome:
    documents: list[Document]
    link_graph: LinkGraph
    report: dict


@dataclass
class _PageState:
    url: str
    depth: int


class Crawler:
    """BFS crawler with robots.txt, politeness delay and domain restriction.

    ``crawl.max_pages`` caps page-fetch attempts, including failed requests and
    responses skipped because of their content type. robots.txt requests are
    not counted against this limit.
    """

    def __init__(self, config: Config, fetcher: Fetcher | None = None):
        self.config = config
        self.crawl_cfg = config.crawl
        self.allowed_domains = list(config.sources.allowed_domains)
        if not self.allowed_domains and config.sources.urls:
            self.allowed_domains = sorted({
                (urlsplit(u).hostname or "").lower() for u in config.sources.urls if urlsplit(u).hostname
            })
        self.fetcher = fetcher or default_fetcher(self.crawl_cfg.user_agent, self.crawl_cfg.retries)
        self.graph = LinkGraph()
        self.visited: set[str] = set()
        self.enqueued: set[str] = set()
        self._documents: dict[str, Document] = {}
        self._robots_cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._fetched_at: dict[str, str] = {}
        self.stats = {
            "page_attempts": 0,
            "pages_fetched": 0,
            "pages_failed": 0,
            "skipped_robots": 0,
            "skipped_domain": 0,
            "skipped_visited": 0,
            "skipped_depth": 0,
            "skipped_content_type": 0,
            "skipped_limit": 0,
        }
        self.failures: list[dict] = []

    # -- robots.txt ---------------------------------------------------------

    def _robots_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots_cache:
            self._robots_cache[origin] = self._load_robots(origin)
        parser = self._robots_cache[origin]
        if parser is None:
            return True
        return parser.can_fetch(self.crawl_cfg.user_agent, url)

    def _load_robots(self, origin: str) -> robotparser.RobotFileParser | None:
        """Fetch and parse robots.txt. A missing/unreachable file means allow."""
        robots_url = f"{origin}/robots.txt"
        result = self.fetcher(robots_url, self.crawl_cfg.timeout_seconds)
        if result.error or result.status_code != 200:
            logger.warning("robots.txt not available at %s (%s); proceeding", robots_url,
                           result.error or f"HTTP {result.status_code}")
            return None
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        text = result.content.decode("utf-8", errors="replace")
        parser.parse(text.splitlines())
        return parser

    # -- link extraction ----------------------------------------------------

    def _extract_links(self, page_url: str, html: str) -> list[tuple[str, bool]]:
        """Return normalized outbound links; record every one as a graph edge."""
        soup = BeautifulSoup(html, "lxml")
        targets: list[str] = []
        seen_on_page: set[str] = set()
        for tag in soup.find_all("a", href=True):
            normalized = normalize_url(tag["href"], base=page_url)
            if normalized is None or normalized in seen_on_page:
                continue
            seen_on_page.add(normalized)
            external = not host_matches(
                normalized,
                self.allowed_domains,
                include_subdomains=self.crawl_cfg.include_subdomains,
            )
            if external:
                self.stats["skipped_domain"] += 1
            self.graph.add_node(normalized, external=external)
            self.graph.add_edge(page_url, normalized, external)
            targets.append((normalized, external))
        return targets

    # -- main loop ----------------------------------------------------------

    def crawl(self) -> CrawlOutcome:
        started = time.monotonic()
        started_at = _utc_now()
        output_pages = Path(self.config.output_dir) / "pages"
        output_pages.mkdir(parents=True, exist_ok=True)

        queue: deque[_PageState] = deque()
        for seed in self.config.sources.urls:
            normalized = normalize_url(seed)
            if normalized is None:
                logger.warning("skipping invalid seed URL: %s", seed)
                continue
            if not host_matches(
                normalized,
                self.allowed_domains,
                include_subdomains=self.crawl_cfg.include_subdomains,
            ):
                self.stats["skipped_domain"] += 1
                logger.warning("seed URL outside allowed domains, skipping: %s", normalized)
                continue
            self.graph.add_node(normalized, external=False)
            self.enqueued.add(normalized)
            queue.append(_PageState(url=normalized, depth=0))

        while queue:
            # max_pages counts page-fetch attempts, not only successful pages.
            # robots.txt requests are handled separately and are not counted.
            if self.stats["page_attempts"] >= self.crawl_cfg.max_pages:
                self.stats["skipped_limit"] += len(queue)
                break

            page = queue.popleft()
            if page.url in self.visited:
                self.stats["skipped_visited"] += 1
                continue

            if not self._robots_allowed(page.url):
                self.stats["skipped_robots"] += 1
                logger.info("robots.txt disallows %s", page.url)
                continue

            self.visited.add(page.url)
            if self.crawl_cfg.delay_seconds > 0:
                time.sleep(self.crawl_cfg.delay_seconds)

            self.stats["page_attempts"] += 1
            result = self.fetcher(page.url, self.crawl_cfg.timeout_seconds)
            if result.error or result.status_code >= 400:
                reason = result.error or f"HTTP {result.status_code}"
                self.stats["pages_failed"] += 1
                self.failures.append({"url": page.url, "depth": page.depth, "reason": reason})
                logger.warning("failed to fetch %s: %s", page.url, reason)
                continue

            content_type = result.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and not any(content_type.startswith(t) for t in HTML_CONTENT_TYPES):
                self.stats["skipped_content_type"] += 1
                continue

            # A redirect may have moved us to a new canonical URL. Revalidate
            # the final URL before accepting any content from it.
            raw_final_url = result.final_url or page.url
            final_url = normalize_url(raw_final_url)
            if final_url is None:
                reason = f"invalid redirect URL: {raw_final_url!r}"
                self.stats["pages_failed"] += 1
                self.failures.append({"url": page.url, "depth": page.depth, "reason": reason})
                logger.warning("failed to accept %s: %s", page.url, reason)
                continue
            if final_url != page.url:
                if not host_matches(
                    final_url,
                    self.allowed_domains,
                    include_subdomains=self.crawl_cfg.include_subdomains,
                ):
                    self.stats["skipped_domain"] += 1
                    self.graph.add_node(final_url, external=True)
                    self.graph.add_edge(page.url, final_url, True)
                    logger.warning(
                        "redirect target outside allowed domains; skipping %s -> %s",
                        page.url,
                        final_url,
                    )
                    continue
                if final_url in self.visited:
                    self.stats["skipped_visited"] += 1
                    continue
                if not self._robots_allowed(final_url):
                    self.stats["skipped_robots"] += 1
                    self.graph.add_node(final_url, external=False)
                    self.graph.add_edge(page.url, final_url, False)
                    logger.info("robots.txt disallows redirect target %s", final_url)
                    continue
                self.visited.add(final_url)
                self.graph.add_node(final_url, external=False)
                self.graph.add_edge(page.url, final_url, False)

            html = _decode_html(result.content, result.headers)
            fetched_at = _utc_now()
            doc = self._store_page(final_url, html, result.content, fetched_at, page.depth, output_pages)
            self._fetched_at[final_url] = fetched_at
            self.graph.add_node(final_url, external=False, fetched=True)
            self.stats["pages_fetched"] += 1

            if self.stats["pages_fetched"] % GRAPH_SAVE_EVERY == 0:
                self.graph.save(Path(self.config.output_dir) / "link_graph.json")

            # Enqueue outgoing links (depth-limited, domain-restricted).
            if page.depth < self.crawl_cfg.max_depth:
                for target, external in self._extract_links(final_url, html):
                    if external or target in self.visited or target in self.enqueued:
                        continue
                    self.enqueued.add(target)
                    queue.append(_PageState(url=target, depth=page.depth + 1))
            else:
                # Still record the edges, but count internal links that were
                # not followed because the crawl reached its depth limit.
                for target, external in self._extract_links(final_url, html):
                    if external or target in self.visited or target in self.enqueued:
                        continue
                    self.stats["skipped_depth"] += 1

        report = {
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "seed_urls": list(self.config.sources.urls),
            "allowed_domains": self.allowed_domains,
            "max_depth": self.crawl_cfg.max_depth,
            "max_pages": self.crawl_cfg.max_pages,
            **self.stats,
            "failures": self.failures,
        }
        return CrawlOutcome(documents=list(self._documents.values()), link_graph=self.graph, report=report)

    def _store_page(
        self, url: str, html: str, raw_bytes: bytes, fetched_at: str, depth: int, pages_dir: Path
    ) -> Document:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""

        doc = Document.from_parts(
            source=url,
            title=title,
            raw_text=html,
            fetched_at=fetched_at,
            kind=KIND_WEB,
            content_type="html",
            metadata={
                "depth": depth,
                "content_hash_sha256_of_bytes": hashlib.sha256(raw_bytes).hexdigest(),
            },
        )
        page_path = pages_dir / f"{doc.doc_id}.html"
        page_path.write_bytes(raw_bytes)
        doc.metadata["html_path"] = str(page_path)
        self._documents[doc.doc_id] = doc
        return doc

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_ingest(config: Config, fetcher: Fetcher | None = None) -> dict:
    """Run the full Phase 1 ingestion: crawl seeds, load files, write outputs."""
    from kbforge.ingest.files import load_files  # local import avoids a cycle

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crawler = Crawler(config, fetcher=fetcher)
    outcome = crawler.crawl()

    documents = list(outcome.documents)
    file_documents: list[Document] = []
    file_failures: list[dict] = []
    if config.sources.files_dir:
        file_documents = load_files(config.sources.files_dir, errors=file_failures)
        documents.extend(file_documents)

    written = write_documents_jsonl(documents, output_dir / "documents.jsonl")
    outcome.link_graph.save(output_dir / "link_graph.json")

    report = outcome.report
    report["files_loaded"] = len(file_documents)
    report["files_failed"] = len(file_failures)
    report["file_failures"] = file_failures
    report["documents_written"] = written
    report["output_files"] = {
        "documents": str(output_dir / "documents.jsonl"),
        "link_graph": str(output_dir / "link_graph.json"),
        "pages_dir": str(output_dir / "pages"),
    }
    (output_dir / "crawl_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "page_attempts": report["page_attempts"],
        "pages_fetched": report["pages_fetched"],
        "pages_failed": report["pages_failed"],
        "files_loaded": len(file_documents),
        "files_failed": len(file_failures),
        "documents_written": written,
        "edges": outcome.link_graph.to_dict()["stats"]["edges"],
        "output_dir": str(output_dir),
    }


# Make sure imports work
