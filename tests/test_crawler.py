"""Crawler tests. All HTTP is served by FakeFetcher; nothing touches the network."""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import FakeFetcher, FakeRoute, html_page, make_config

from kbforge.documents import read_documents_jsonl
from kbforge.ingest.crawler import Crawler, run_ingest

ROBOTS = b"User-agent: *\nDisallow: /private/\n"


def standard_site() -> FakeFetcher:
    return FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(
                body=html_page(
                    "Home",
                    "welcome",
                    links=[
                        "/page1",
                        "/page2",
                        "/page1?utm_source=newsletter",  # same page after normalization
                        "/private/secret",                # disallowed by robots
                        "http://other.test/x",            # outside allowed domains
                        "mailto:a@b.c",                   # not crawlable
                        "#top",                           # fragment only
                    ],
                )
            ),
            "http://kb.test/page1": FakeRoute(body=html_page("Page 1", "first", links=["/deep1"])),
            "http://kb.test/page2": FakeRoute(body=html_page("Page 2", "second")),
            "http://kb.test/deep1": FakeRoute(body=html_page("Deep 1", "deep")),
            "http://kb.test/private/secret": FakeRoute(body=html_page("Secret", "hidden")),
        }
    )


def test_bfs_fetches_expected_pages(tmp_path):
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    fetched = sorted(d.source for d in outcome.documents)
    assert fetched == [
        "http://kb.test/",
        "http://kb.test/deep1",
        "http://kb.test/page1",
        "http://kb.test/page2",
    ]
    assert outcome.report["pages_fetched"] == 4
    assert outcome.report["pages_failed"] == 0
    assert outcome.report["page_attempts"] == 4


def test_robots_disallow_is_respected(tmp_path):
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    sources = {d.source for d in outcome.documents}
    assert "http://kb.test/private/secret" not in sources
    assert outcome.report["skipped_robots"] == 1


def test_depth_limit_stops_expansion(tmp_path):
    config = make_config(max_depth=1, output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    sources = {d.source for d in outcome.documents}
    # deep1 sits at depth 2 (linked from page1 which is depth 1)
    assert "http://kb.test/deep1" not in sources
    assert "http://kb.test/page1" in sources
    assert outcome.report["skipped_depth"] == 1


def test_max_pages_caps_fetches(tmp_path):
    config = make_config(max_pages=2, output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    assert outcome.report["pages_fetched"] == 2
    assert outcome.report["page_attempts"] == 2


def test_domain_restriction_blocks_other_hosts(tmp_path):
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    sources = {d.source for d in outcome.documents}
    assert not any(s.startswith("http://other.test") for s in sources)
    assert outcome.report["skipped_domain"] == 1


def test_link_graph_records_edges_and_external_flags(tmp_path):
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    edges = {(e["source"], e["target"], e["external_target"]) for e in outcome.link_graph.edges}
    assert ("http://kb.test/", "http://kb.test/page1", False) in edges
    assert ("http://kb.test/page1", "http://kb.test/deep1", False) in edges
    assert ("http://kb.test/", "http://other.test/x", True) in edges
    # tracking-param duplicate collapses into the canonical edge target
    targets_from_home = {t for s, t, _ in edges if s == "http://kb.test/"}
    assert "http://kb.test/page1" in targets_from_home


def test_failed_fetch_is_recorded_and_crawl_continues(tmp_path):
    fetcher = standard_site()
    fetcher.routes["http://kb.test/page1"] = FakeRoute(status=500)
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=fetcher).crawl()
    assert outcome.report["pages_failed"] == 1
    assert outcome.report["page_attempts"] == 3
    assert outcome.report["failures"][0]["url"] == "http://kb.test/page1"
    sources = {d.source for d in outcome.documents}
    assert "http://kb.test/page2" in sources  # siblings still crawled


def test_non_html_content_type_is_skipped(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(body=html_page("Home", "hi")),
            "http://kb.test/file.pdf": FakeRoute(body=b"%PDF", content_type="application/pdf"),
        }
    )
    # home page links to the pdf so the crawler attempts it
    fetcher.routes["http://kb.test/"] = FakeRoute(
        body=html_page("Home", "hi", links=["/file.pdf"])
    )
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=fetcher).crawl()
    sources = {d.source for d in outcome.documents}
    assert "http://kb.test/file.pdf" not in sources
    assert outcome.report["skipped_content_type"] == 1
    assert outcome.report["page_attempts"] == 2
    assert outcome.report["pages_fetched"] == 1


def test_run_ingest_writes_all_outputs(tmp_path):
    files_dir = tmp_path / "client_files"
    files_dir.mkdir()
    (files_dir / "notes.txt").write_text("hello notes", encoding="utf-8")

    out = tmp_path / "out"
    config = make_config(files_dir=str(files_dir), output_dir=str(out))
    summary = run_ingest(config, fetcher=standard_site())

    assert summary["pages_fetched"] == 4
    assert summary["files_loaded"] == 1
    assert summary["files_failed"] == 0
    assert summary["documents_written"] == 5

    docs = read_documents_jsonl(out / "documents.jsonl")
    assert len(docs) == 5
    assert all(d.content_hash and d.doc_id and d.fetched_at for d in docs)

    graph = json.loads((out / "link_graph.json").read_text(encoding="utf-8"))
    assert graph["stats"]["edges"] > 0
    assert any(n["fetched"] for n in graph["nodes"])

    report = json.loads((out / "crawl_report.json").read_text(encoding="utf-8"))
    assert report["pages_fetched"] == 4
    assert report["files_loaded"] == 1
    assert report["files_failed"] == 0
    assert report["file_failures"] == []

    # raw HTML saved per page under pages/
    pages = list((out / "pages").glob("*.html"))
    assert len(pages) == 4


def test_crawled_documents_carry_expected_schema(tmp_path):
    config = make_config(output_dir=str(tmp_path / "out"))
    outcome = Crawler(config, fetcher=standard_site()).crawl()
    doc = next(d for d in outcome.documents if d.source == "http://kb.test/page1")
    assert doc.kind == "web"
    assert doc.content_type == "html"
    assert doc.title == "Page 1"
    assert "<title>Page 1</title>" in doc.raw_text
    assert doc.metadata["depth"] == 1


def test_failed_requests_count_toward_max_pages(tmp_path):
    urls = [f"http://kb.test/fail{i}" for i in range(5)]
    routes = {"http://kb.test/robots.txt": FakeRoute(body=ROBOTS)}
    routes.update({url: FakeRoute(status=500) for url in urls})
    config = make_config(urls=urls, max_pages=1, output_dir=str(tmp_path / "out"))

    outcome = Crawler(config, fetcher=FakeFetcher(routes)).crawl()

    assert outcome.report["page_attempts"] == 1
    assert outcome.report["pages_failed"] == 1
    assert outcome.report["pages_fetched"] == 0
    assert outcome.report["skipped_limit"] == 4


def test_external_redirect_is_not_ingested(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(
                body=html_page("Home", "welcome"),
                final_url="http://other.test/landing",
            ),
        }
    )
    config = make_config(output_dir=str(tmp_path / "out"))

    outcome = Crawler(config, fetcher=fetcher).crawl()

    assert outcome.documents == []
    assert outcome.report["page_attempts"] == 1
    assert outcome.report["pages_fetched"] == 0
    assert outcome.report["skipped_domain"] == 1
    graph = outcome.link_graph.to_dict()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["http://other.test/landing"]["external"] is True
    assert any(
        edge["target"] == "http://other.test/landing" and edge["external_target"] is True
        for edge in graph["edges"]
    )


def test_robots_disallowed_redirect_is_not_ingested(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(
                body=html_page("Home", "welcome"),
                final_url="http://kb.test/private/secret",
            ),
        }
    )
    config = make_config(output_dir=str(tmp_path / "out"))

    outcome = Crawler(config, fetcher=fetcher).crawl()

    assert outcome.documents == []
    assert outcome.report["page_attempts"] == 1
    assert outcome.report["pages_fetched"] == 0
    assert outcome.report["skipped_robots"] == 1
    graph = outcome.link_graph.to_dict()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["http://kb.test/private/secret"]["external"] is False
    assert any(
        edge["target"] == "http://kb.test/private/secret" and edge["external_target"] is False
        for edge in graph["edges"]
    )


def test_utf8_html_is_decoded_without_replacement(tmp_path):
    body = '<html><head><title>Café</title></head><body><p>Café</p></body></html>'.encode("utf-8")
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(body=body, content_type="text/html; charset=utf-8"),
        }
    )
    outcome = Crawler(make_config(output_dir=str(tmp_path / "out")), fetcher=fetcher).crawl()

    doc = outcome.documents[0]
    assert doc.title == "Café"
    assert "Café" in doc.raw_text
    assert "\ufffd" not in doc.raw_text


def test_iso_8859_1_html_uses_declared_charset(tmp_path):
    body = '<html><head><title>Café</title></head><body><p>Café</p></body></html>'.encode("latin-1")
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(body=body, content_type="text/html; charset=iso-8859-1"),
        }
    )
    outcome = Crawler(make_config(output_dir=str(tmp_path / "out")), fetcher=fetcher).crawl()

    doc = outcome.documents[0]
    assert doc.title == "Café"
    assert "Café" in doc.raw_text
    assert "\ufffd" not in doc.raw_text


def test_run_ingest_reports_file_failures(tmp_path):
    files_dir = tmp_path / "client_files"
    files_dir.mkdir()
    (files_dir / "good.txt").write_text("hello notes", encoding="utf-8")
    (files_dir / "broken.pdf").write_bytes(b"this is not a PDF")

    out = tmp_path / "out"
    config = make_config(files_dir=str(files_dir), output_dir=str(out))
    summary = run_ingest(config, fetcher=standard_site())

    assert summary["files_loaded"] == 1
    assert summary["files_failed"] == 1

    report = json.loads((out / "crawl_report.json").read_text(encoding="utf-8"))
    assert report["files_loaded"] == 1
    assert report["files_failed"] == 1
    assert len(report["file_failures"]) == 1
    failure = report["file_failures"][0]
    assert failure["path"].endswith("broken.pdf")
    assert failure["error"]


def test_malformed_hyperlink_is_skipped_and_crawl_continues(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(
                body=html_page(
                    "Home",
                    "welcome",
                    links=["http://kb.test:bad/broken", "/page1"],
                )
            ),
            "http://kb.test/page1": FakeRoute(body=html_page("Page 1", "first")),
        }
    )
    config = make_config(output_dir=str(tmp_path / "out"))

    outcome = Crawler(config, fetcher=fetcher).crawl()

    sources = {doc.source for doc in outcome.documents}
    assert sources == {"http://kb.test/", "http://kb.test/page1"}
    node_ids = {node["id"] for node in outcome.link_graph.to_dict()["nodes"]}
    assert "http://kb.test:bad/broken" not in node_ids


def test_allowed_internal_redirect_is_ingested(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/old": FakeRoute(
                body=html_page("New Home", "moved"),
                final_url="http://kb.test/new",
            ),
        }
    )
    config = make_config(urls=["http://kb.test/old"], output_dir=str(tmp_path / "out"))

    outcome = Crawler(config, fetcher=fetcher).crawl()

    assert [doc.source for doc in outcome.documents] == ["http://kb.test/new"]
    graph = outcome.link_graph.to_dict()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["http://kb.test/new"]["external"] is False
    assert nodes["http://kb.test/new"]["fetched"] is True


def test_seed_outside_allowed_domains_is_counted(tmp_path):
    config = make_config(
        urls=["http://other.test/"],
        allowed_domains=["kb.test"],
        output_dir=str(tmp_path / "out"),
    )

    outcome = Crawler(config, fetcher=FakeFetcher({})).crawl()

    assert outcome.documents == []
    assert outcome.report["page_attempts"] == 0
    assert outcome.report["skipped_domain"] == 1


def test_malformed_redirect_target_is_rejected(tmp_path):
    fetcher = FakeFetcher(
        {
            "http://kb.test/robots.txt": FakeRoute(body=ROBOTS),
            "http://kb.test/": FakeRoute(
                body=html_page("Home", "welcome"),
                final_url="http://kb.test:bad/destination",
            ),
        }
    )
    outcome = Crawler(make_config(output_dir=str(tmp_path / "out")), fetcher=fetcher).crawl()

    assert outcome.documents == []
    assert outcome.report["page_attempts"] == 1
    assert outcome.report["pages_failed"] == 1
    assert "invalid redirect URL" in outcome.report["failures"][0]["reason"]
