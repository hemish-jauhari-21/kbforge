"""Integration tests for Phase 2."""

from __future__ import annotations

import json
from dataclasses import replace

from kbforge.clean import clean_document as clean_document_original
from kbforge.config import CleaningConfig, DeduplicationConfig
from kbforge.documents import Document, read_documents_jsonl
from kbforge.process import process_documents, run_process
from tests.conftest import make_config


def _text_doc(text: str, source: str = "https://kb.test/page") -> Document:
    return Document.from_parts(
        source=source,
        title="Page",
        raw_text=text,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="txt",
    )


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
    config2 = replace(
        config,
        cleaning=CleaningConfig(enabled=False),
        deduplication=DeduplicationConfig(enabled=False),
    )
    processed, removal_log, cleaning_diagnostics, stats = process_documents(
        docs, config=config2
    )
    assert len(processed) == 1
    assert removal_log == []
    assert cleaning_diagnostics == []
    assert "deduplication_skipped" in stats
    assert "cleaning_skipped" in stats
    assert stats["duplicate_removals"] == 0
    assert stats["cleaning_warnings"] == 0


def test_cleaning_warning_is_retained_without_removal(monkeypatch):
    doc = _text_doc("short warning content", "https://kb.test/warning")

    def clean_with_warning(document):
        return document, {
            "doc_id": document.doc_id,
            "reason": "suspiciously_short_extraction",
            "source": document.source,
        }

    monkeypatch.setattr("kbforge.clean.clean_document", clean_with_warning)
    config = replace(make_config(), deduplication=DeduplicationConfig(enabled=False))

    processed, removal_log, cleaning_diagnostics, stats = process_documents(
        [doc], config=config
    )

    assert [result.doc_id for result in processed] == [doc.doc_id]
    assert removal_log == []
    assert len(cleaning_diagnostics) == 1
    assert cleaning_diagnostics[0]["outcome"] == "retained"
    assert cleaning_diagnostics[0]["reason"] == "suspiciously_short_extraction"
    assert stats["cleaning_warnings"] == 1
    assert stats["documents_excluded_by_cleaning"] == 0
    assert stats["duplicate_removals"] == 0
    assert stats["output_documents"] == 1


def test_exact_duplicate_is_counted_as_removal():
    first = _text_doc("identical content", "https://kb.test/duplicate-a")
    second = _text_doc("identical content", "https://kb.test/duplicate-b")
    config = replace(
        make_config(),
        deduplication=DeduplicationConfig(exact=True, near_duplicate=False),
    )

    processed, removal_log, cleaning_diagnostics, stats = process_documents(
        [first, second], config=config
    )

    assert len(processed) == 1
    assert len(removal_log) == 1
    assert removal_log[0]["reason"] == "exact"
    assert cleaning_diagnostics == []
    assert stats["duplicate_removals"] == 1
    assert stats["cleaning_warnings"] == 0
    assert stats["output_documents"] == 1


def test_unrecoverable_cleaning_error_is_reported_as_exclusion(monkeypatch):
    doc = _text_doc("content", "https://kb.test/broken")

    def fail_cleaning(document):
        raise RuntimeError("cleaning failed")

    def fail_normalization(text):
        raise RuntimeError("fallback failed")

    monkeypatch.setattr("kbforge.clean.clean_document", fail_cleaning)
    monkeypatch.setattr(
        "kbforge.clean.normalize_unicode_and_whitespace", fail_normalization
    )
    config = replace(make_config(), deduplication=DeduplicationConfig(enabled=False))

    processed, removal_log, cleaning_diagnostics, stats = process_documents(
        [doc], config=config
    )

    assert processed == []
    assert removal_log == []
    assert len(cleaning_diagnostics) == 1
    assert cleaning_diagnostics[0]["outcome"] == "excluded"
    assert stats["cleaning_warnings"] == 0
    assert stats["documents_excluded_by_cleaning"] == 1
    assert stats["duplicate_removals"] == 0
    assert stats["output_documents"] == 0


def test_run_process_separates_warnings_from_duplicate_removals(
    tmp_path, monkeypatch
):
    duplicate_a = _text_doc("identical content", "https://kb.test/duplicate-a")
    duplicate_b = _text_doc("identical content", "https://kb.test/duplicate-b")
    warning_doc = _text_doc("unique warning content", "https://kb.test/warning")

    def clean_with_warning(document):
        cleaned, info = clean_document_original(document)
        if document.source == warning_doc.source:
            info = {
                "doc_id": document.doc_id,
                "reason": "suspiciously_short_extraction",
                "source": document.source,
            }
        return cleaned, info

    monkeypatch.setattr("kbforge.clean.clean_document", clean_with_warning)
    output_dir = tmp_path / "kb"
    config = replace(
        make_config(output_dir=str(output_dir)),
        deduplication=DeduplicationConfig(exact=True, near_duplicate=False),
    )

    summary = run_process(config, documents=[duplicate_a, duplicate_b, warning_doc])

    removal_log = json.loads(
        (output_dir / "removal_log.json").read_text(encoding="utf-8")
    )
    assert [entry["reason"] for entry in removal_log["removals"]] == ["exact"]
    assert len(removal_log["cleaning_warnings"]) == 1
    assert removal_log["cleaning_warnings"][0]["outcome"] == "retained"
    assert removal_log["cleaning_exclusions"] == []
    assert summary["removals"] == 1
    assert summary["cleaning_warnings"] == 1
    assert summary["cleaning_exclusions"] == 0
    assert summary["stats"]["duplicate_removals"] == 1
    assert summary["stats"]["cleaning_warnings"] == 1
    assert summary["stats"]["documents_excluded_by_cleaning"] == 0
    assert summary["stats"]["output_documents"] == 2

    cleaned_docs = read_documents_jsonl(output_dir / "documents_cleaned.jsonl")
    assert {doc.source for doc in cleaned_docs} == {
        warning_doc.source,
        removal_log["removals"][0]["retained_source"],
    }
