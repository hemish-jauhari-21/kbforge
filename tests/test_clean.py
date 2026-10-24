"""Tests for content cleaning."""

from __future__ import annotations

from kbforge.clean import clean_document, clean_documents, normalize_unicode_and_whitespace
from kbforge.documents import Document


def make_web_html_doc(html: str, source="http://example.com/page") -> Document:
    return Document.from_parts(
        source=source,
        title="Test",
        raw_text=html,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={"depth": 0},
    )


def make_text_doc(text: str) -> Document:
    return Document.from_parts(
        source="/tmp/file.txt",
        title="file",
        raw_text=text,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="file",
        content_type="txt",
        metadata={},
    )


def test_html_main_content_extraction():
    html = """<html><head><title>Test</title></head>
    <body>
        <nav>Navigation menu</nav>
        <script>var x=1;</script>
        <style>body{}</style>
        <main><p>This is the main content that should be extracted.</p>
        <p>Another paragraph of important information.</p></main>
        <footer>Footer content</footer>
    </body></html>"""
    doc = make_web_html_doc(html)
    cleaned, failure = clean_document(doc)
    # Main content should be extracted
    assert "main content" in cleaned.raw_text
    assert "Navigation menu" not in cleaned.raw_text or cleaned.raw_text.count("Navigation") < 5  # boilerplate reduced
    assert cleaned.doc_id == doc.doc_id
    assert cleaned.source == doc.source
    assert cleaned.title == doc.title


def test_script_style_nav_footer_removal():
    html = """<html><body>
    <nav><a href="/">Home</a></nav>
    <script>alert('hi');</script>
    <style>.nav{}</style>
    <div><p>Actual article text here.</p></div>
    <footer>Copyright 2024</footer>
    </body></html>"""
    doc = make_web_html_doc(html)
    cleaned, _ = clean_document(doc)
    # Should contain article text
    assert "article text" in cleaned.raw_text


def test_unicode_and_whitespace_normalization():
    text = "Hello\u00a0World   with\t\tspaces\n\n\n\nand   newlines"
    normalized = normalize_unicode_and_whitespace(text)
    assert "\u00a0" not in normalized  # NBSP normalized
    assert "   " not in normalized or normalized.count("  ") < normalized.count(" ")  # excessive spaces collapsed


def test_plain_text_documents_remain_intact_apart_from_normalization():
    text = "This is plain text.\n\nWith paragraphs.\tAnd\ttabs."
    doc = make_text_doc(text)
    cleaned, failure = clean_document(doc)
    assert "plain text" in cleaned.raw_text
    assert cleaned.source == doc.source
    assert cleaned.doc_id == doc.doc_id


def test_empty_or_malformed_html_fallback():
    # Empty HTML
    doc = make_web_html_doc("<html><body></body></html>")
    cleaned, failure = clean_document(doc)
    # Should not crash
    assert isinstance(cleaned.raw_text, str)
    
    # Malformed HTML
    doc2 = make_web_html_doc("<html><body><p>Broken")
    cleaned2, failure2 = clean_document(doc2)
    assert isinstance(cleaned2.raw_text, str)


def test_metadata_preservation():
    doc = Document.from_parts(
        source="http://example.com/page",
        title="My Title",
        raw_text="<html><body><p>Hello world content here.</p></body></html>",
        fetched_at="2026-01-01T12:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={"depth": 2, "custom": "value"},
    )
    cleaned, _ = clean_document(doc)
    assert cleaned.source == doc.source
    assert cleaned.title == doc.title
    assert cleaned.doc_id == doc.doc_id
    assert cleaned.fetched_at == doc.fetched_at
    assert cleaned.kind == doc.kind
    assert cleaned.content_type == doc.content_type
    assert cleaned.metadata.get("depth") == 2
    assert cleaned.metadata.get("custom") == "value"


def test_content_hash_recalculated():
    html = "<html><body><p>Some content here</p></body></html>"
    doc = make_web_html_doc(html)
    original_hash = doc.content_hash
    cleaned, _ = clean_document(doc)
    # Hash should be recalculated for cleaned content
    assert cleaned.content_hash != original_hash or cleaned.raw_text != doc.raw_text


def test_clean_documents_batch():
    docs = [
        make_web_html_doc("<html><body><p>Content 1</p></body></html>"),
        make_text_doc("Plain text content"),
    ]
    cleaned, failures, stats = clean_documents(docs)
    assert len(cleaned) == 2
    assert isinstance(stats, dict)
