from __future__ import annotations

import copy
import logging
import unicodedata
from typing import Iterable

import trafilatura
from bs4 import BeautifulSoup

from kbforge.documents import Document, compute_content_hash

logger = logging.getLogger("kbforge.clean")


def normalize_unicode_and_whitespace(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        lines.append(line)
    normalized = "\n".join(lines)
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    normalized = normalized.strip()
    return normalized


def _is_html_content(doc: Document) -> bool:
    if doc.kind == "web" and doc.content_type == "html":
        return True
    return False


def _is_likely_boilerplate_text(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    boilerplate_indicators = [
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "subscribe to",
        "sign up",
    ]
    count = sum(1 for b in boilerplate_indicators if b in t)
    if count >= 2:
        return True
    return False


def _score_extracted_text(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    norm = normalize_unicode_and_whitespace(text)
    words = norm.split()
    if len(words) == 0:
        return 0.0
    score = float(len(words))
    score += norm.count("\n\n") * 2
    if _is_likely_boilerplate_text(norm):
        score *= 0.5
    return score


def _try_trafilatura_configs(html: str):
    configs = [
        dict(include_comments=False, include_tables=True, include_formatting=False, favor_precision=False),
        dict(include_comments=False, include_tables=True, include_formatting=False, favor_precision=True),
        dict(include_comments=False, include_tables=True, favor_recall=True),
        dict(output_format="txt", include_comments=False, include_tables=True),
    ]
    results = []
    for cfg in configs:
        try:
            extracted = trafilatura.extract(html, **cfg)
            if extracted and extracted.strip():
                results.append(extracted.strip())
        except Exception as e:
            logger.debug(f"trafilatura config failed: {e}")
            continue
    return results


def _extract_with_semantic_fallback(html: str) -> tuple[str | None, dict]:
    diagnostics = {"strategy": "semantic_fallback"}
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None, diagnostics

    # Remove clearly non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas", "form"]):
        try:
            tag.decompose()
        except Exception:
            pass

    # Try to find main content in semantic containers FIRST (high priority)
    candidates_main = []
    for elements in [soup.find_all("main"), soup.find_all("article"), soup.find_all(attrs={"role": "main"})]:
        for el in elements:
            try:
                # Don't decompose nav/header/footer inside main if they're the main structure? Usually not
                text = el.get_text(separator="\n")
                if text and text.strip():
                    candidates_main.append(text)
            except Exception:
                continue

    if candidates_main:
        # Prefer main/article content even if short - it's likely the actual page content
        best = max(candidates_main, key=_score_extracted_text)
        return best, diagnostics

    # Remove common boilerplate
    for tag in soup.find_all(["nav", "header", "footer", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass
    for tag in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo", "complementary", "search"]}):
        try:
            tag.decompose()
        except Exception:
            pass

    candidates = []
    body = soup.find("body") or soup
    for tag in body.find_all(["nav", "header", "footer", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass
    for tag in body.find_all(["section", "div"]):
        try:
            text = tag.get_text(separator="\n")
            if text and text.strip():
                words = text.strip().split()
                if len(words) > 20:  # be more lenient
                    candidates.append(text)
        except Exception:
            continue

    if candidates:
        best = max(candidates, key=_score_extracted_text)
        return best, diagnostics

    text = body.get_text(separator="\n")
    if text and text.strip():
        return text, diagnostics
    return None, diagnostics


def _extract_main_content_html(html: str) -> tuple[str | None, dict]:
    diagnostics = {"trafilatura_attempted": True}
    
    # First check semantic main/article - if they contain distinct content, prefer them
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None
    
    semantic_main_content = None
    if soup:
        for elements in [soup.find_all("main"), soup.find_all("article"), soup.find_all(attrs={"role": "main"})]:
            for el in elements:
                try:
                    # Get a copy to analyze
                    text = el.get_text(separator="\n")
                    if text and text.strip():
                        semantic_main_content = text
                        break
                except Exception:
                    continue
            if semantic_main_content:
                break
    
    trafilatura_results = _try_trafilatura_configs(html)
    if trafilatura_results:
        diagnostics["trafilatura_succeeded"] = True
        best_traf = max(trafilatura_results, key=_score_extracted_text)
        
        if semantic_main_content:
            best_sem_main = max([semantic_main_content], key=_score_extracted_text)
            # If main/article contains page-specific content different from generic boilerplate
            # or is very different in nature, prefer it
            sem_norm = normalize_unicode_and_whitespace(best_sem_main)
            traf_norm = normalize_unicode_and_whitespace(best_traf)
            # If semantic main has "not found" or specific error messages, or is much shorter
            # but contains specific terms that suggest it's the actual content, prefer it
            if len(sem_norm) < 200 and ("not found" in sem_norm.lower() or "404" in sem_norm.lower() or len(sem_norm.split()) < 30):
                # Could be a short page - but also could be important content
                # Prefer semantic main if it looks like actual page content vs boilerplate
                return best_sem_main, {**diagnostics, "strategy": "semantic_main_preferred"}
            # If they are completely different and semantic main has page-unique content
            if sem_norm != traf_norm and len(sem_norm) > 10:
                # Check if semantic main contains content not typical of site chrome
                # Prefer semantic main content when it exists - it's more likely to be page-specific
                return best_sem_main, {**diagnostics, "strategy": "semantic_main_preferred"}
            # If trafilatura gives much longer content that might be boilerplate
            if len(traf_norm) > len(sem_norm) * 10 and len(sem_norm) < 500:
                return best_sem_main, {**diagnostics, "strategy": "semantic_main_preferred_short"}
        
        semantic_result, sem_diag = _extract_with_semantic_fallback(html)
        diagnostics.update(sem_diag)
        if semantic_result:
            if _score_extracted_text(semantic_result) > _score_extracted_text(best_traf) * 1.5:
                return semantic_result, diagnostics
            return best_traf, diagnostics
        return best_traf, diagnostics

    diagnostics["trafilatura_succeeded"] = False
    if semantic_main_content:
        return semantic_main_content, {**diagnostics, "strategy": "semantic_main_preferred"}
    semantic_result, sem_diag = _extract_with_semantic_fallback(html)
    diagnostics.update(sem_diag)
    if semantic_result:
        return semantic_result, diagnostics

    try:
        soup2 = BeautifulSoup(html, "lxml")
        for tag in soup2(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        text = soup2.get_text(separator="\n")
        if text and text.strip():
            diagnostics["strategy"] = "bs4_get_text"
            return text, diagnostics
    except Exception:
        pass

    diagnostics["strategy"] = "failed"
    return None, diagnostics


def _fallback_html_extraction(html: str) -> tuple[str, dict]:
    text, diag = _extract_main_content_html(html)
    if text and text.strip():
        return text, diag
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return normalize_unicode_and_whitespace(text), {"strategy": "bs4_last_resort"}
    except Exception:
        return normalize_unicode_and_whitespace(html) if html else "", {"strategy": "raw"}


def clean_document(doc: Document) -> tuple[Document, dict | None]:
    failure_info = None

    if _is_html_content(doc):
        extracted, diag = _extract_main_content_html(doc.raw_text)
        if not extracted or not extracted.strip():
            extracted, diag = _fallback_html_extraction(doc.raw_text)
            if not extracted or not extracted.strip():
                failure_info = {
                    "doc_id": doc.doc_id,
                    "reason": "empty_extraction",
                    "source": doc.source,
                    "diagnostics": diag,
                }
                logger.warning(f"Empty extraction for {doc.doc_id} from {doc.source}")
            else:
                failure_info = {
                    "doc_id": doc.doc_id,
                    "reason": "extraction_fallback_used",
                    "source": doc.source,
                    "diagnostics": diag,
                }
                logger.debug(f"Used fallback extraction for {doc.doc_id}")
        cleaned_text = normalize_unicode_and_whitespace(extracted)
        if not cleaned_text.strip() and doc.raw_text and doc.raw_text.strip():
            fallback_text, diag = _fallback_html_extraction(doc.raw_text)
            cleaned_text = normalize_unicode_and_whitespace(fallback_text)
            failure_info = failure_info or {
                "doc_id": doc.doc_id,
                "reason": "extraction_failed_fallback",
                "source": doc.source,
                "diagnostics": diag,
            }
        if cleaned_text and len(cleaned_text.strip()) < 50 and doc.raw_text and len(doc.raw_text) > 10000:
            failure_info = failure_info or {
                "doc_id": doc.doc_id,
                "reason": "suspiciously_short_extraction",
                "source": doc.source,
                "diagnostics": diag,
            }
    else:
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

    return cleaned_doc, failure_info


def clean_documents(documents: Iterable[Document]) -> tuple[list[Document], list[dict], dict]:
    """Clean documents and classify cleaning diagnostics.

    Documents are retained when cleaning succeeds or when a fallback produces
    usable text. A diagnostic with ``outcome="retained"`` is therefore a
    warning about a document that remains in the corpus. Only an unrecoverable
    cleaning error produces ``outcome="excluded"``.
    """
    cleaned = []
    warnings = []
    exclusions = []

    for doc in documents:
        try:
            cleaned_doc, failure_info = clean_document(doc)
            cleaned.append(cleaned_doc)
            if failure_info:
                warnings.append({**failure_info, "outcome": "retained"})
        except Exception as e:
            logger.error(f"Failed to clean document {doc.doc_id}: {e}")
            try:
                cleaned_text = normalize_unicode_and_whitespace(doc.raw_text) if doc.raw_text else ""
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
                warnings.append(
                    {
                        "doc_id": doc.doc_id,
                        "reason": f"cleaning_error: {e}",
                        "source": doc.source,
                        "outcome": "retained",
                    }
                )
            except Exception:
                exclusions.append(
                    {
                        "doc_id": doc.doc_id,
                        "reason": f"cleaning_error: {e}",
                        "source": doc.source,
                        "outcome": "excluded",
                    }
                )

    stats = {
        "documents_cleaned": len(cleaned),
        "cleaning_warnings": len(warnings),
        "documents_excluded_by_cleaning": len(exclusions),
    }
    return cleaned, warnings + exclusions, stats


clean = clean_documents
