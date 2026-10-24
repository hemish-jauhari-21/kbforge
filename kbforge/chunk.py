"""Document chunking for kbforge Phase 3.

Supports fixed-size, sentence-based, and paragraph/heading-aware chunking.
Size is measured in words by default for readability and to match
existing config comments; overlap applies to fixed-size strategy.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from kbforge.documents import Document

logger = logging.getLogger("kbforge.chunk")


def _stable_chunk_id(doc_id: str, index: int, strategy: str, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{strategy}:{index}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"chunk-{h}"


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    # Simple sentence splitter - general purpose, no heavy deps
    if not text:
        return []
    # Avoid splitting on abbreviations naively; use a reasonable heuristic
    # Split on . ! ? followed by whitespace and capital letter or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])', text)
    # Also handle end of string
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    if not sentences:
        # Fallback: split by newlines
        sentences = [s.strip() for s in text.replace('\n\n', '. ').split('. ') if s.strip()]
    return sentences


def _split_paragraphs(text: str) -> list[tuple[str, str]]:
    """Split into paragraphs/heading-like blocks. Returns (type, text)."""
    if not text:
        return []
    blocks = []
    # Split by double newlines
    paras = re.split(r'\n\s*\n', text)
    for para in paras:
        para = para.strip()
        if not para:
            continue
        # Heuristic: short line followed by content might be heading
        lines = para.split('\n')
        if len(lines) == 1 and len(lines[0].split()) <= 10 and len(lines[0]) < 80:
            # Could be heading
            blocks.append(('heading', lines[0]))
        else:
            blocks.append(('paragraph', para))
    return blocks


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


def chunk_document_fixed(doc: Document, size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Fixed-size chunking by words with overlap."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")

    text = doc.raw_text if doc.raw_text else ""
    words = text.split()
    if not words:
        return []

    chunks = []
    chunk_index = 0
    i = 0
    while i < len(words):
        end = min(i + size, len(words))
        chunk_words = words[i:end]
        chunk_text = " ".join(chunk_words)
        # Calculate approximate char positions
        start_char = len(" ".join(words[:i])) if i > 0 else 0
        end_char = start_char + len(chunk_text)
        chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "fixed", chunk_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                strategy="fixed",
                metadata=dict(doc.metadata),
            )
        )
        chunk_index += 1
        if end == len(words):
            break
        i = end - overlap  # move back by overlap
    return chunks


def chunk_document_sentence(doc: Document, size: int = 300) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    text = doc.raw_text if doc.raw_text else ""
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    chunk_index = 0
    current_sentences = []
    current_words = 0

    for sent in sentences:
        sent_words = _word_count(sent)
        # If a single sentence is longer than size, split it by words
        if sent_words > size:
            # Flush current
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "sentence", chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(chunk_text),
                        strategy="sentence",
                        metadata=dict(doc.metadata),
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_words = 0
            # Split long sentence by words
            words = sent.split()
            for i in range(0, len(words), size):
                chunk_words = words[i : i + size]
                chunk_text = " ".join(chunk_words)
                chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "sentence", chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=0,
                        end_char=len(chunk_text),
                        strategy="sentence",
                        metadata=dict(doc.metadata),
                    )
                )
                chunk_index += 1
            continue

        if current_sentences and current_words + sent_words > size:
            chunk_text = " ".join(current_sentences)
            chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "sentence", chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_text),
                    strategy="sentence",
                    metadata=dict(doc.metadata),
                )
            )
            chunk_index += 1
            current_sentences = []
            current_words = 0
        current_sentences.append(sent)
        current_words += sent_words

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "sentence", chunk_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                strategy="sentence",
                metadata=dict(doc.metadata),
            )
        )

    return chunks


def chunk_document_paragraph(doc: Document, size: int = 300) -> list[Chunk]:
    """Paragraph/heading-aware chunking."""
    if size <= 0:
        raise ValueError("size must be > 0")
    text = doc.raw_text if doc.raw_text else ""
    if not text.strip():
        return []

    blocks = _split_paragraphs(text)
    if not blocks:
        return chunk_document_sentence(doc, size)  # fallback

    chunks = []
    chunk_index = 0
    current_blocks = []
    current_words = 0

    for btype, btext in blocks:
        bwords = _word_count(btext)
        if current_blocks and current_words + bwords > size and btype != 'heading':  # prefer keeping headings with content
            chunk_text = "\n\n".join(current_blocks)
            chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "paragraph", chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_text),
                    strategy="paragraph",
                    metadata=dict(doc.metadata) | {"chunking_strategy": "paragraph"},
                )
            )
            chunk_index += 1
            current_blocks = []
            current_words = 0
        elif bwords > size and not current_blocks:
            # Split oversized paragraph
            # Try sentence split
            subchunks = chunk_document_sentence(doc.__class__(**{**doc.__dict__, 'raw_text': btext}), size)
            for sc in subchunks:
                sc.chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "paragraph", sc.text)
                sc.chunk_index = chunk_index
                sc.strategy = "paragraph"
                sc.metadata = dict(doc.metadata) | {"chunking_strategy": "paragraph"}
                chunks.append(sc)
                chunk_index += 1
            continue
        current_blocks.append(btext)
        current_words += bwords

    if current_blocks:
        chunk_text = "\n\n".join(current_blocks)
        chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, "paragraph", chunk_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                strategy="paragraph",
                metadata=dict(doc.metadata) | {"chunking_strategy": "paragraph"},
            )
        )

    return chunks


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
