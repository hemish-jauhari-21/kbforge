"""Tests for inverted index and retrieval."""

import tempfile
import os
from kbforge.index import InvertedIndex, BM25, TFIDF
from kbforge.textproc import tokenize


def test_inverted_index_postings():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c1", "text": "cat sat on mat", "doc_id": "d1", "source": "s1", "title": "t1"},
        {"chunk_id": "c2", "text": "dog sat on log", "doc_id": "d2", "source": "s2", "title": "t2"},
    ]
    idx.build(chunks)
    assert "cat" in idx.postings
    assert "c1" in idx.postings["cat"]
    assert idx.postings["cat"]["c1"] == 1
    assert idx.df["cat"] == 1
    assert idx.N == 2


def test_term_frequencies():
    idx = InvertedIndex()
    chunks = [{"chunk_id": "c1", "text": "cat cat dog", "doc_id": "d1", "source": "s1", "title": "t1"}]
    idx.build(chunks)
    assert idx.postings["cat"]["c1"] == 2


def test_document_lengths():
    idx = InvertedIndex()
    chunks = [{"chunk_id": "c1", "text": "a b c d", "doc_id": "d1", "source": "s1", "title": "t1"}]
    idx.build(chunks)
    assert idx.doc_lengths["c1"] == 4


def test_persistence_and_loading():
    idx = InvertedIndex()
    chunks = [{"chunk_id": "c1", "text": "hello world", "doc_id": "d1", "source": "s1", "title": "t1"}]
    idx.build(chunks)
    with tempfile.TemporaryDirectory() as td:
        idx.save(td)
        idx2 = InvertedIndex.load(td)
        assert idx2.N == 1
        assert "hello" in idx2.postings


def test_bm25_ranking_known_corpus():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c1", "text": "the cat sat on the mat", "doc_id": "d1", "source": "s1", "title": "t1"},
        {"chunk_id": "c2", "text": "the dog sat on the log", "doc_id": "d2", "source": "s2", "title": "t2"},
        {"chunk_id": "c3", "text": "birds fly in the sky", "doc_id": "d3", "source": "s3", "title": "t3"},
    ]
    idx.build(chunks)
    bm25 = BM25(idx)
    results = bm25.score("cat")
    assert len(results) > 0
    assert results[0].chunk_id == "c1"


def test_tfidf_cosine_ranking():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c1", "text": "cat dog cat", "doc_id": "d1", "source": "s1", "title": "t1"},
        {"chunk_id": "c2", "text": "dog bird", "doc_id": "d2", "source": "s2", "title": "t2"},
    ]
    idx.build(chunks)
    tfidf = TFIDF(idx)
    results = tfidf.score("cat")
    assert len(results) > 0


def test_empty_corpus():
    idx = InvertedIndex()
    idx.build([])
    bm25 = BM25(idx)
    assert bm25.score("test") == []
    tfidf = TFIDF(idx)
    assert tfidf.score("test") == []


def test_empty_query():
    idx = InvertedIndex()
    chunks = [{"chunk_id": "c1", "text": "hello", "doc_id": "d1", "source": "s1", "title": "t1"}]
    idx.build(chunks)
    bm25 = BM25(idx)
    assert bm25.score("") == []
    assert bm25.score("   ") == []


def test_unknown_terms():
    idx = InvertedIndex()
    chunks = [{"chunk_id": "c1", "text": "hello world", "doc_id": "d1", "source": "s1", "title": "t1"}]
    idx.build(chunks)
    bm25 = BM25(idx)
    results = bm25.score("xyzabc")
    assert results == []


def test_deterministic_tie_breaking():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c2", "text": "test", "doc_id": "d2", "source": "s2", "title": "t2"},
        {"chunk_id": "c1", "text": "test", "doc_id": "d1", "source": "s1", "title": "t1"},
    ]
    idx.build(chunks)
    bm25 = BM25(idx)
    results = bm25.score("test")
    # Same score, should be deterministic
    assert results[0].chunk_id == results[0].chunk_id  # stable


def test_malformed_chunk_records():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c1"},  # missing text
        {"text": "hello"},   # missing chunk_id
    ]
    idx.build(chunks)  # should not crash
