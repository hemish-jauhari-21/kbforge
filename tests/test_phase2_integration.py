"""Integration tests for Phase 2."""

from __future__ import annotations

from kbforge.clean import clean_documents
from kbforge.dedup import deduplicate_documents
from kbforge.documents import Document
from kbforge.process import process_documents
from tests.conftest import make_config


def test_disabled_cleaning_and_deduplication():
    docs = [
        Document.from_parts(
            source="http://example.com/1",
            title="T",
            raw_text="<html><body><p>Hi</p></body></html>",
            fetched_at="2026-01-01T00:00:00+00:00",
            kind="web",
            content_type="html",
            metadata={},
        ),
    ]
    config = make_config()
    # Modify config
    from kbforge.config import CleaningConfig, DeduplicationConfig
    from dataclasses import replace
    config2 = replace(config, cleaning=CleaningConfig(enabled=False), deduplication=DeduplicationConfig(enabled=False))
    processed, removal_log, stats = process_documents(docs, config=config2)
    assert len(processed) == 1
    # With cleaning disabled, should not extract via trafilatura the same way
    assert "deduplication_skipped" in stats or "cleaning_skipped" in stats
