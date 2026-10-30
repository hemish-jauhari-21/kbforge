"""Tests for inverted index and retrieval."""

import math
import tempfile
import os

import pytest

from kbforge.index import InvertedIndex, BM25, TFIDF
from kbforge.textproc import tokenize


def _tfidf_corpus():
    idx = InvertedIndex()
    idx.build(
        [
            {"chunk_id": "c1", "text": "alpha alpha beta"},
            {"chunk_id": "c2", "text": "beta gamma"},
            {"chunk_id": "c3", "text": "gamma delta"},
        ]
    )
    return idx


def _reference_tfidf_scores(index, query):
    query_tf = {}
    for token in tokenize(query):
        query_tf[token] = query_tf.get(token, 0) + 1

    idf = {
        term: math.log(index.N / df) if df > 0 else 0.0
        for term, df in index.df.items()
    }
    query_vec = {
        term: (1.0 + math.log(tf)) * idf[term]
        for term, tf in query_tf.items()
        if term in idf
    }
    query_norm = math.sqrt(sum(value * value for value in query_vec.values()))
    if query_norm == 0:
        return {}

    scores = {}
    for chunk_id in index.chunk_info:
        dot = 0.0
        for term, query_weight in query_vec.items():
            posting = index.postings.get(term)
            if posting is None or chunk_id not in posting:
                continue
            tf = posting[chunk_id]
            doc_weight = (1.0 + math.log(tf)) * idf.get(term, 0.0)
            dot += doc_weight * query_weight
        if dot <= 0:
            continue
        if index.doc_lengths.get(chunk_id, 0) > 0:
            doc_norm = math.sqrt(
                sum(
                    (
                        (1.0 + math.log(posting[chunk_id]))
                        * idf.get(term, 0.0)
                    )
                    ** 2
                    for term, posting in index.postings.items()
                    if chunk_id in posting
                )
            )
        else:
            doc_norm = 1.0
        if doc_norm > 0:
            scores[chunk_id] = dot / (query_norm * doc_norm)
    return scores


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


def test_tie_breaking_uses_ascending_chunk_id_for_bm25_and_tfidf():
    idx = InvertedIndex()
    idx.build(
        [
            {"chunk_id": "c2", "text": "alpha"},
            {"chunk_id": "c1", "text": "alpha"},
            {"chunk_id": "c4", "text": "beta"},
        ]
    )

    for ranker in (BM25(idx), TFIDF(idx)):
        results = ranker.score("alpha")
        assert [result.chunk_id for result in results] == ["c1", "c2"]
        assert [result.rank for result in results] == [0, 1]
        assert results[0].score == pytest.approx(results[1].score)


def test_tfidf_scores_match_reference_cosine():
    idx = _tfidf_corpus()
    expected = _reference_tfidf_scores(idx, "alpha beta")

    results = TFIDF(idx).score("alpha beta")

    assert [result.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].chunk_id == "c1"
    assert set(result.chunk_id for result in results) == set(expected)
    for result in results:
        assert result.score == pytest.approx(expected[result.chunk_id])


@pytest.mark.parametrize(
    "query", ["alpha", "alpha alpha", "alpha beta", "beta gamma", "delta", "delta alpha"]
)
def test_tfidf_candidate_scoring_matches_full_scan(query):
    idx = _tfidf_corpus()
    expected = _reference_tfidf_scores(idx, query)

    results = TFIDF(idx).score(query)

    assert set(result.chunk_id for result in results) == set(expected)
    for result in results:
        assert result.score == pytest.approx(expected[result.chunk_id])
        assert result.score > 0


def test_tfidf_repeated_queries_return_identical_results():
    ranker = TFIDF(_tfidf_corpus())

    first = ranker.score("alpha beta")
    second = ranker.score("alpha beta")

    assert [(r.chunk_id, r.score, r.rank) for r in first] == [
        (r.chunk_id, r.score, r.rank) for r in second
    ]


def test_tfidf_unknown_terms_and_zero_norm_return_no_results():
    idx = _tfidf_corpus()
    ranker = TFIDF(idx)

    assert ranker.score("missingterm") == []

    common_idx = InvertedIndex()
    common_idx.build(
        [
            {"chunk_id": "c1", "text": "common shared"},
            {"chunk_id": "c2", "text": "common other"},
        ]
    )
    assert TFIDF(common_idx).score("common") == []


def test_tfidf_zero_length_document_uses_unit_norm():
    idx = InvertedIndex()
    idx.build(
        [
            {"chunk_id": "empty", "text": ""},
            {"chunk_id": "c1", "text": "alpha beta"},
            {"chunk_id": "c2", "text": "beta gamma"},
        ]
    )
    ranker = TFIDF(idx)

    assert idx.doc_lengths["empty"] == 0
    assert ranker.doc_norms["empty"] == 1.0
    assert all(result.chunk_id != "empty" for result in ranker.score("alpha"))


def test_malformed_chunk_records():
    idx = InvertedIndex()
    chunks = [
        {"chunk_id": "c1"},  # missing text
        {"text": "hello"},   # missing chunk_id
    ]
    idx.build(chunks)  # should not crash
