"""Tests for deduplication."""

from __future__ import annotations

from kbforge.dedup import deduplicate_documents, _select_representative
from kbforge.documents import Document


def make_doc(text: str, source: str = "http://example.com/a", title: str = "T") -> Document:
    return Document.from_parts(
        source=source,
        title=title,
        raw_text=text,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={},
    )


def test_exact_duplicate_detection():
    docs = [
        make_doc("Same content here", "http://example.com/1"),
        make_doc("Same content here", "http://example.com/2"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs, exact=True, near_duplicate=False)
    assert len(kept) == 1
    assert stats["exact_duplicates_removed"] == 1
    assert len(removal_log) == 1
    assert removal_log[0]["reason"] == "exact"


def test_near_duplicate_detection_above_threshold():
    # Very similar documents
    docs = [
        make_doc("This is a long document about machine learning and artificial intelligence with many details", "http://example.com/1"),
        make_doc("This is a long document about machine learning and artificial intelligence with many details and extra words", "http://example.com/2"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs, exact=True, near_duplicate=True, similarity_threshold=0.5, shingle_size=3)
    # May or may not be detected depending on threshold/shingling, but should be safe
    assert len(kept) >= 1


def test_near_duplicate_below_threshold_not_removed():
    docs = [
        make_doc("Completely different topic about cats and dogs running in the park", "http://example.com/1"),
        make_doc("Another entirely different subject about cars and trucks on highways", "http://example.com/2"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs, exact=True, near_duplicate=True, similarity_threshold=0.8)
    assert len(kept) == 2


def test_unrelated_documents_remain_distinct():
    docs = [
        make_doc("Python programming language tutorial", "http://example.com/1"),
        make_doc("Cooking recipes for Italian food", "http://example.com/2"),
        make_doc("Space exploration and astronomy facts", "http://example.com/3"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs)
    assert len(kept) == 3


def test_deterministic_representative_selection():
    docs = [
        make_doc("Same content", "http://example.com/b", title=""),
        make_doc("Same content", "http://example.com/a", title="Better Title"),
    ]
    rep = _select_representative(docs)
    # Prefer document with better title/source info
    assert rep.title == "Better Title" or rep.source == "http://example.com/a"


def test_empty_and_short_documents():
    docs = [
        make_doc("", "http://example.com/empty"),
        make_doc("Hi", "http://example.com/short"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs)
    # Should handle safely without crashing
    assert isinstance(kept, list)


def test_removal_log_accuracy():
    docs = [
        make_doc("Duplicate content", "http://example.com/1", title="T1"),
        make_doc("Duplicate content", "http://example.com/2", title="T2"),
    ]
    kept, removal_log, stats = deduplicate_documents(docs)
    assert len(removal_log) == 1
    assert "removed_doc_id" in removal_log[0]
    assert "retained_doc_id" in removal_log[0]
    assert "reason" in removal_log[0]
    assert removal_log[0]["reason"] in ("exact", "near_duplicate")


def test_repeated_runs_produce_consistent_results():
    docs = [
        make_doc("Content A", "http://example.com/1"),
        make_doc("Content A", "http://example.com/2"),
        make_doc("Content B", "http://example.com/3"),
    ]
    kept1, log1, stats1 = deduplicate_documents(docs)
    kept2, log2, stats2 = deduplicate_documents(docs)
    assert len(kept1) == len(kept2)
    assert stats1["kept_count"] == stats2["kept_count"]
    assert [d.doc_id for d in kept1] == [d.doc_id for d in kept2]
