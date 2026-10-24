"""Content cleaning for kbforge Phase 2.

Cleans documents using trafilatura for crawled HTML, normalizes text,
and prepares cleaned documents for deduplication.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import unicodedata
from dataclasses import replace
from typing import Iterable

import trafilatura
from bs4 import BeautifulSoup

from kbforge.documents import Document, compute_content_hash

logger = logging.getLogger("kbforge.clean")


def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize Unicode (NFKC) and collapse excessive whitespace while
    preserving paragraph boundaries where possible."""
    if text is None:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Replace various whitespace chars
    # Try to preserve paragraph breaks (double newlines)
    # First, normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs within lines but keep newlines
    # Split by newlines and normalize each line
    lines = []
    for line in text.split("\n"):
        # Collapse whitespace
        line = " ".join(line.split())
        lines.append(line)
    # Join back - but we need to be smart about preserving paragraph structure
    # Multiple consecutive empty lines become paragraph breaks
    # Collapse runs of empty lines to at most 2 newlines (paragraph break)
    result_lines = []
    for line in lines:
        result_lines.append(line)
    # Rejoin
    normalized = "\n".join(result_lines)
    # Collapse 3+ consecutive newlines to 2 (paragraph break)
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    # Also collapse spaces around newlines
    normalized = normalized.strip()
    return normalized


def _is_html_content(doc: Document) -> bool:
    """Determine if document contains crawled HTML that needs trafilatura extraction."""
    # Crawled HTML documents are typically web kind with html content_type
    if doc.kind == "web" and doc.content_type == "html":
        return True
    # Also if it's HTML content type from files? The requirement says
    # "For crawled HTML documents..." - be careful to follow spec
    # But better to be reasonable: if it's HTML and has HTML tags, try to extract
    # For files loaded as HTML, they were already processed by BeautifulSoup in files.py
    # But cleaner to avoid double-processing inappropriately
    return False


def _extract_main_content_html(html: str) -> str | None:
    """Extract main content from HTML using trafilatura."""
    try:
        # Try with default settings first
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True, 
                                         include_formatting=False, favor_precision=False)
        if extracted and extracted.strip():
            return extracted
        # Try bare extraction as fallback
        try:
            extracted = trafilatura.extract(html, output_format='txt', include_comments=False)
            if extracted and extracted.strip():
                return extracted
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"trafilatura extraction failed: {e}")
    return None


def _fallback_html_extraction(html: str) -> str:
    """Deterministic fallback for HTML extraction failures."""
    try:
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        # Get text
        text = soup.get_text(separator="\n")
        return text
    except Exception:
        # Last resort - return as-is
        return html


def clean_document(doc: Document) -> tuple[Document, dict | None]:
    """
    Clean a single document.
    Returns (cleaned_document, failure_info_or_none)
    """
    failure_info = None
    original = doc
    
    if _is_html_content(doc):
        # Try trafilatura extraction
        extracted = _extract_main_content_html(doc.raw_text)
        if not extracted or not extracted.strip():
            # Deterministic fallback - do not silently discard meaningful content
            extracted = _fallback_html_extraction(doc.raw_text)
            if not extracted or not extracted.strip():
                failure_info = {
                    "doc_id": doc.doc_id,
                    "reason": "empty_extraction",
                    "source": doc.source
                }
                logger.warning(f"Empty extraction for {doc.doc_id} from {doc.source}")
            else:
                failure_info = {
                    "doc_id": doc.doc_id,
                    "reason": "extraction_fallback_used",
                    "source": doc.source
                }
                logger.debug(f"Used fallback extraction for {doc.doc_id}")
        cleaned_text = normalize_unicode_and_whitespace(extracted)
        if not cleaned_text.strip() and doc.raw_text.strip():
            # Still nothing extracted but original had content - use fallback normalized
            fallback = _fallback_html_extraction(doc.raw_text)
            cleaned_text = normalize_unicode_and_whitespace(fallback)
            failure_info = failure_info or {
                "doc_id": doc.doc_id,
                "reason": "extraction_failed_fallback",
                "source": doc.source
            }
    else:
        # For plain text documents (files like txt/pdf/docx/html files already processed)
        # or other content - normalize without parsing as HTML
        cleaned_text = normalize_unicode_and_whitespace(doc.raw_text)
    
    # Create new document instance (don't mutate original)
    cleaned_doc = Document(
        doc_id=doc.doc_id,
        source=doc.source,
        title=doc.title,
        raw_text=cleaned_text,
        fetched_at=doc.fetched_at,
        content_hash=compute_content_hash(cleaned_text),  # Recalculate after cleaning
        kind=doc.kind,
        content_type=doc.content_type,
        metadata=copy.deepcopy(doc.metadata),
    )
    
    return cleaned_doc, failure_info


def clean_documents(documents: Iterable[Document]) -> tuple[list[Document], list[dict], dict]:
    """
    Clean multiple documents.
    Returns (cleaned_documents, failures, stats)
    """
    cleaned = []
    failures = []
    
    for doc in documents:
        try:
            cleaned_doc, failure_info = clean_document(doc)
            cleaned.append(cleaned_doc)
            if failure_info:
                failures.append(failure_info)
        except Exception as e:
            logger.error(f"Failed to clean document {doc.doc_id}: {e}")
            failures.append({
                "doc_id": doc.doc_id,
                "reason": f"cleaning_error: {e}",
                "source": doc.source
            })
            # Still try to add normalized version as fallback
            try:
                cleaned_text = normalize_unicode_and_whitespace(doc.raw_text)
                cleaned_doc = Document(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    raw_text=cleaned_text,
                    fetched_at=doc.fetched_at,
                    content_hash=compute_content_hash(cleaned_text),
                    kind=doc.kind,
                    content_type=doc.content_type,
                    metadata=copy.deepcopy(doc.metadata),
                )
                cleaned.append(cleaned_doc)
            except Exception:
                # Skip only if completely broken
                continue
    
    stats = {
        "total_input": len(list(documents)) if False else sum(1 for _ in []) + len(cleaned) + len(failures),  # rough
        "cleaned": len(cleaned),
        "failures": len(failures),
    }
    # Better stats
    stats = {
        "documents_cleaned": len(cleaned),
        "cleaning_failures": len(failures),
    }
    return cleaned, failures, stats


# Alias for flexibility
clean = clean_documents
