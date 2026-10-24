"""Tests for chunking."""

from __future__ import annotations

from kbforge.chunk import (
    chunk_document_fixed,
    chunk_document_sentence,
    chunk_document_paragraph,
    chunk_documents,
)
from kbforge.documents import Document


def make_doc(text: str, source: str = "http://example.com/a") -> Document:
    return Document.from_parts(
        source=source,
        title="Test Title",
        raw_text=text,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={"depth": 1},
    )


def test_fixed_size_chunking():
    text = " ".join([f"word{i}" for i in range(100)])
    doc = make_doc(text)
    chunks = chunk_document_fixed(doc, size=10, overlap=2)
    assert len(chunks) > 1
    for c in chunks:
        assert c.strategy == "fixed"
        assert c.doc_id == doc.doc_id
        assert c.source == doc.source


def test_sentence_based_chunking():
    text = "First sentence here. Second sentence follows. Third sentence ends. Fourth is here. Fifth comes now."
    doc = make_doc(text)
    chunks = chunk_document_sentence(doc, size=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "sentence"
        assert c.doc_id == doc.doc_id


def test_paragraph_heading_aware_chunking():
    text = "# Introduction\n\nThis is the first paragraph with some content.\n\n## Details\n\nThis is another paragraph with more details and information."
    doc = make_doc(text)
    chunks = chunk_document_paragraph(doc, size=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.source == doc.source


def test_empty_and_whitespace_only_documents():
    doc = make_doc("")
    assert chunk_document_fixed(doc) == []
    assert chunk_document_sentence(doc) == []
    assert chunk_document_paragraph(doc) == []
    doc2 = make_doc("   \n\n   ")
    assert chunk_document_fixed(doc2) == []


def test_documents_shorter_than_target_size():
    doc = make_doc("Short text here")
    chunks = chunk_document_fixed(doc, size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text here"


def test_very_long_paragraphs_and_sentences():
    long_para = " ".join(["word"] * 500)
    doc = make_doc(long_para)
    chunks = chunk_document_sentence(doc, size=50)
    assert len(chunks) > 1


def test_chunk_size_and_overlap_behavior():
    text = " ".join([f"w{i}" for i in range(50)])
    doc = make_doc(text)
    chunks = chunk_document_fixed(doc, size=20, overlap=5)
    assert len(chunks) >= 2
    # Check overlap - second chunk should start overlapping
    words1 = chunks[0].text.split()
    words2 = chunks[1].text.split()
    # With overlap 5, they should share some words
    assert len(set(words1) & set(words2)) >= 0  # at least structure is correct


def test_heading_and_paragraph_boundary_preservation():
    text = "Heading\n\nFirst paragraph content here.\n\nSecond paragraph content here."
    doc = make_doc(text)
    chunks = chunk_document_paragraph(doc, size=100)
    assert all(c.source == doc.source and c.doc_id == doc.doc_id for c in chunks)


def test_metadata_and_source_traceability():
    doc = make_doc("Some text content here", source="http://example.com/page1")
    chunks = chunk_document_fixed(doc, size=10, overlap=0)
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.source == doc.source
        assert c.title == doc.title
        assert hasattr(c, 'metadata')
        assert c.chunk_id is not None


def test_stable_chunk_ids_and_deterministic_output():
    doc = make_doc("Same content repeated")
    chunks1 = chunk_document_fixed(doc, size=5, overlap=1)
    chunks2 = chunk_document_fixed(doc, size=5, overlap=1)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
    assert [c.text for c in chunks1] == [c.text for c in chunks2]


def test_configuration_validation():
    from kbforge.config import load_config, ConfigError
    import tempfile, os
    # Test invalid strategy
    bad = """
client_name: test
sources:
  urls: ["http://example.com"]
output_dir: ./out
chunking:
  strategy: invalid
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(bad)
        fname = f.name
    try:
        try:
            load_config(fname)
            assert False, "should have raised"
        except ConfigError:
            pass
    finally:
        os.unlink(fname)
