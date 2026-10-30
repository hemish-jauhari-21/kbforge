from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from kbforge.documents import Document

logger = logging.getLogger("kbforge.chunk")

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])')


def _stable_chunk_id(doc_id: str, index: int, strategy: str, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{strategy}:{index}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"chunk-{h}"


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _sentence_spans(
    text: str, start: int = 0, end: int | None = None
) -> list[tuple[int, int]]:
    if end is None:
        end = len(text)

    spans = []
    cursor = start
    for match in _SENTENCE_BOUNDARY.finditer(text, start, end):
        span = _trimmed_span(text, cursor, match.start())
        if span:
            spans.append(span)
        cursor = match.end()

    span = _trimmed_span(text, cursor, end)
    if span:
        spans.append(span)
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return paragraph spans separated by blank lines."""
    if not text:
        return []

    spans = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        span = _trimmed_span(text, cursor, match.start())
        if span:
            spans.append(span)
        cursor = match.end()

    span = _trimmed_span(text, cursor, len(text))
    if span:
        spans.append(span)
    return spans


def _word_spans(
    text: str, start: int = 0, end: int | None = None
) -> list[tuple[int, int]]:
    if end is None:
        end = len(text)
    return [
        (start + match.start(), start + match.end())
        for match in re.finditer(r"\S+", text[start:end])
    ]


def _split_span_by_words(
    text: str, start: int, end: int, size: int
) -> list[tuple[int, int]]:
    words = _word_spans(text, start, end)
    spans = []
    for index in range(0, len(words), size):
        group = words[index : index + size]
        spans.append((group[0][0], group[-1][1]))
    return spans


def _pack_spans(
    text: str, spans: list[tuple[int, int]], size: int
) -> list[tuple[int, int]]:
    packed = []
    current_start = None
    current_end = None
    current_words = 0

    for start, end in spans:
        span_words = _word_count(text[start:end])
        if span_words > size:
            if current_start is not None:
                packed.append((current_start, current_end))
                current_start = None
                current_end = None
                current_words = 0
            packed.extend(_split_span_by_words(text, start, end, size))
            continue

        if current_start is not None and current_words + span_words > size:
            packed.append((current_start, current_end))
            current_start = None
            current_end = None
            current_words = 0

        if current_start is None:
            current_start = start
        current_end = end
        current_words += span_words

    if current_start is not None:
        packed.append((current_start, current_end))
    return packed


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    strategy: str
    metadata: dict


def _make_chunk(
    doc: Document,
    chunk_index: int,
    strategy: str,
    text: str,
    start_char: int,
    end_char: int,
) -> Chunk:
    chunk_text = text[start_char:end_char]
    return Chunk(
        chunk_id=_stable_chunk_id(doc.doc_id, chunk_index, strategy, chunk_text),
        doc_id=doc.doc_id,
        source=doc.source,
        title=doc.title,
        text=chunk_text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        strategy=strategy,
        metadata={**doc.metadata, "chunking_strategy": strategy},
    )


def chunk_document_fixed(doc: Document, size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Fixed-size chunking by words with overlap."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")

    text = doc.raw_text if doc.raw_text else ""
    words = _word_spans(text)
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        end = min(i + size, len(words))
        chunks.append(
            _make_chunk(
                doc,
                len(chunks),
                "fixed",
                text,
                words[i][0],
                words[end - 1][1],
            )
        )
        if end == len(words):
            break
        i = end - overlap
    return chunks


def chunk_document_sentence(doc: Document, size: int = 300) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    text = doc.raw_text if doc.raw_text else ""
    if not text.strip():
        return []

    spans = _pack_spans(text, _sentence_spans(text), size)
    return [
        _make_chunk(doc, index, "sentence", text, start, end)
        for index, (start, end) in enumerate(spans)
    ]


def chunk_document_paragraph(doc: Document, size: int = 300) -> list[Chunk]:
    """Paragraph-aware chunking with safe splitting for oversized paragraphs."""
    if size <= 0:
        raise ValueError("size must be > 0")
    text = doc.raw_text if doc.raw_text else ""
    if not text.strip():
        return []

    blocks = _paragraph_spans(text)
    if not blocks:
        return []

    packed = []
    current_start = None
    current_end = None
    current_words = 0

    for start, end in blocks:
        block_words = _word_count(text[start:end])

        if block_words > size:
            if current_start is not None:
                packed.append((current_start, current_end))
                current_start = None
                current_end = None
                current_words = 0
            packed.extend(
                _pack_spans(text, _sentence_spans(text, start, end), size)
            )
            continue

        if current_start is not None and current_words + block_words > size:
            packed.append((current_start, current_end))
            current_start = None
            current_end = None
            current_words = 0

        if current_start is None:
            current_start = start
        current_end = end
        current_words += block_words

    if current_start is not None:
        packed.append((current_start, current_end))

    return [
        _make_chunk(doc, index, "paragraph", text, start, end)
        for index, (start, end) in enumerate(packed)
    ]


def chunk_documents(documents: Iterable[Document], strategy: str = "fixed", size: int = 300, overlap: int = 50) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        if strategy == "fixed":
            chunks = chunk_document_fixed(doc, size=size, overlap=overlap)
        elif strategy == "sentence":
            chunks = chunk_document_sentence(doc, size=size)
        elif strategy == "paragraph":
            chunks = chunk_document_paragraph(doc, size=size)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        all_chunks.extend(chunks)
    return all_chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "title": chunk.title,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "strategy": chunk.strategy,
        "metadata": chunk.metadata,
    }
