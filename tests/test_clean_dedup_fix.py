"""General-purpose tests for extraction and dedup fix."""

from __future__ import annotations

from kbforge.clean import clean_document
from kbforge.dedup import deduplicate_documents
from kbforge.documents import Document


def make_web_html(html: str, source: str = "http://example.com/page") -> Document:
    return Document.from_parts(
        source=source,
        title="Test",
        raw_text=html,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={},
    )


def test_shared_nav_footer_different_article_bodies_survive():
    html1 = """<html><head><title>Page1</title></head>
    <body>
        <nav>Home About Contact</nav>
        <header>Site Header</header>
        <article>
            <h1>Article One</h1>
            <p>Unique content about quantum computing algorithms appears here with specific details.</p>
        </article>
        <footer>Copyright 2024</footer>
    </body></html>"""
    html2 = """<html><head><title>Page2</title></head>
    <body>
        <nav>Home About Contact</nav>
        <header>Site Header</header>
        <article>
            <h1>Article Two</h1>
            <p>Different unique content about baking sourdough bread recipes and techniques.</p>
        </article>
        <footer>Copyright 2024</footer>
    </body></html>"""
    c1, _ = clean_document(make_web_html(html1, "http://example.com/a"))
    c2, _ = clean_document(make_web_html(html2, "http://example.com/b"))
    assert "quantum computing" in c1.raw_text
    assert "sourdough" in c2.raw_text
    assert c1.raw_text != c2.raw_text
    kept, log, stats = deduplicate_documents([c1, c2])
    assert len(kept) == 2


def test_department_pages_with_shared_template_different_main_content():
    html1 = """<html><body><nav>Menu</nav><main><h1>Cardiology</h1><p>Cardiology treats heart conditions with angioplasty and pacemakers.</p></main><footer>Foot</footer></body></html>"""
    html2 = """<html><body><nav>Menu</nav><main><h1>Urology</h1><p>Urology treats kidney stones and urinary tract conditions with lithotripsy.</p></main><footer>Foot</footer></body></html>"""
    c1, _ = clean_document(make_web_html(html1, "http://site.com/depts/cardio"))
    c2, _ = clean_document(make_web_html(html2, "http://site.com/depts/uro"))
    assert "angioplasty" in c1.raw_text
    assert "lithotripsy" in c2.raw_text
    assert c1.raw_text != c2.raw_text


def test_documentation_page_with_headings_lists_code():
    html = """<html><body><nav>Docs</nav><main><h1>API Guide</h1><p>Overview</p><ul><li>Install</li><li>Configure</li></ul><pre><code>pip install pkg</code></pre></main><footer></footer></body></html>"""
    c, _ = clean_document(make_web_html(html, "http://site.com/docs"))
    assert "API Guide" in c.raw_text
    assert "pip install" in c.raw_text or "Install" in c.raw_text


def test_landing_page_with_multiple_sections():
    html = """<html><body><header>Brand</header><article><section><h2>Features</h2><p>Fast performance and scalability.</p></section><section><h2>Pricing</h2><p>Affordable plans for all users.</p></section></article><footer>Info</footer></body></html>"""
    c, _ = clean_document(make_web_html(html, "http://site.com/landing"))
    assert "Features" in c.raw_text or "performance" in c.raw_text
    assert "Pricing" in c.raw_text or "Affordable" in c.raw_text


def test_legitimately_short_page():
    html = """<html><body><main><p>Contact us at hello@example.com</p></main></body></html>"""
    c, fail = clean_document(make_web_html(html, "http://site.com/contact"))
    assert "hello@example.com" in c.raw_text


def test_empty_or_malformed_html():
    c1, _ = clean_document(make_web_html("<html></html>", "http://site.com/e1"))
    c2, _ = clean_document(make_web_html("<p>broken", "http://site.com/e2"))
    assert isinstance(c1.raw_text, str)
    assert isinstance(c2.raw_text, str)


def test_page_with_no_obvious_semantic_main():
    html = """<html><body><div id="content"><h1>Title</h1><p>This is the actual content of the page without main tag.</p></div></body></html>"""
    c, _ = clean_document(make_web_html(html, "http://site.com/p"))
    assert "actual content" in c.raw_text


def test_genuinely_identical_pages_deduplicated():
    html = """<html><body><nav>x</nav><main><p>Same exact content here</p></main><footer>y</footer></body></html>"""
    c1, _ = clean_document(make_web_html(html, "http://site.com/1"))
    c2, _ = clean_document(make_web_html(html, "http://site.com/2"))
    kept, log, stats = deduplicate_documents([c1, c2])
    assert len(kept) == 1
    assert stats["exact_duplicates_removed"] == 1


def test_page_with_little_content_extensive_boilerplate():
    html = """<html><body><nav>Lots of menu items here repeated many times</nav><header>Header content</header><main><p>Real info</p></main><aside>Ads and links</aside><footer>Footer links</footer></body></html>"""
    c, _ = clean_document(make_web_html(html, "http://site.com/q"))
    assert "Real info" in c.raw_text
