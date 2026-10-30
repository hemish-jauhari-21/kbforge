"""Tests for chunking."""

from __future__ import annotations

import pytest

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
    assert chunk_document_sentence(doc2) == []
    assert chunk_document_paragraph(doc2) == []


def test_documents_shorter_than_target_size():
    doc = make_doc("Short text here")
    chunks = chunk_document_fixed(doc, size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text here"


def test_very_long_paragraphs_and_sentences():
    long_para = " ".join(["word"] * 500) + "."
    doc = make_doc(long_para)
    chunks = chunk_document_sentence(doc, size=50)
    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)


def test_chunk_size_and_overlap_behavior():
    text = " ".join([f"w{i}" for i in range(50)])
    doc = make_doc(text)
    chunks = chunk_document_fixed(doc, size=20, overlap=5)
    assert len(chunks) >= 2
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks)
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


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


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph"])
def test_offsets_and_metadata_match_all_strategies(strategy):
    text = (
        "Alpha beta gamma.\n\n"
        "Delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
        "sigma tau.\n\n"
        "Upsilon phi chi psi omega."
    )
    doc = make_doc(text)
    chunks = chunk_documents([doc], strategy=strategy, size=6, overlap=2)

    assert chunks
    for chunk in chunks:
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert len(chunk.text.split()) <= 6
        assert chunk.metadata["depth"] == 1
        assert chunk.metadata["chunking_strategy"] == strategy


def test_paragraph_chunks_keep_whole_paragraphs_when_they_fit():
    first_paragraph = "one two three four five"
    second_paragraph = "six seven eight nine ten"
    text = first_paragraph + "\n\n" + second_paragraph
    doc = make_doc(text)

    chunks = chunk_document_paragraph(doc, size=12)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert text[chunks[0].start_char : chunks[0].end_char] == chunks[0].text


def test_oversized_paragraph_is_split_without_exceeding_size():
    first_paragraph = "Short paragraph stays together."
    oversized_paragraph = " ".join(f"word{i}" for i in range(35)) + "."
    text = first_paragraph + "\n\n" + oversized_paragraph
    doc = make_doc(text)

    chunks = chunk_document_paragraph(doc, size=10)

    assert chunks[0].text == first_paragraph
    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)
    assert all(chunk.strategy == "paragraph" for chunk in chunks)
    assert sum(len(chunk.text.split()) for chunk in chunks) == len(text.split())


def test_oversized_paragraph_prefers_sentence_boundaries():
    sentences = [
        "First sentence has six words now.",
        "Second sentence also has six words.",
        "Third sentence finishes the paragraph.",
    ]
    text = " ".join(sentences)
    doc = make_doc(text)

    chunks = chunk_document_paragraph(doc, size=12)

    assert [chunk.text for chunk in chunks] == [
        f"{sentences[0]} {sentences[1]}",
        sentences[2],
    ]
    assert all(len(chunk.text.split()) <= 12 for chunk in chunks)


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
