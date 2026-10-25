"""Inverted index and lexical retrieval from scratch."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kbforge.textproc import tokenize

logger = logging.getLogger("kbforge.index")


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    score: float
    rank: int
    metadata: dict


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(dict)  # term -> {chunk_id: term_freq}
        self.doc_lengths = {}  # chunk_id -> length in tokens
        self.chunk_info = {}  # chunk_id -> metadata
        self.df = defaultdict(int)  # document frequency
        self.N = 0  # number of documents/chunks
        self.avgdl = 0.0

    def add_chunk(self, chunk_id: str, text: str, metadata: dict | None = None):
        tokens = tokenize(text)
        # Term frequencies
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        self.doc_lengths[chunk_id] = len(tokens)
        self.chunk_info[chunk_id] = metadata or {}
        for term, freq in tf.items():
            self.postings[term][chunk_id] = freq
        self.N += 1
        if self.N > 0:
            self.avgdl = sum(self.doc_lengths.values()) / self.N

    def build(self, chunks):
        self.postings.clear()
        self.doc_lengths.clear()
        self.chunk_info.clear()
        self.df.clear()
        self.N = 0
        self.avgdl = 0.0
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
            text = chunk.get('text', '')
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            if chunk_id:
                self.add_chunk(chunk_id, text, metadata)
        # Compute document frequencies
        self.df = defaultdict(int)
        for term, posting in self.postings.items():
            self.df[term] = len(posting)
        if self.N > 0:
            self.avgdl = sum(self.doc_lengths.values()) / self.N

    def save(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        # Save postings
        postings_data = {term: posting for term, posting in self.postings.items()}
        with open(index_dir / 'postings.json', 'w', encoding='utf-8') as f:
            json.dump(postings_data, f)
        with open(index_dir / 'doc_lengths.json', 'w', encoding='utf-8') as f:
            json.dump(self.doc_lengths, f)
        with open(index_dir / 'chunk_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.chunk_info, f)
        with open(index_dir / 'df.json', 'w', encoding='utf-8') as f:
            json.dump(dict(self.df), f)
        stats = {'N': self.N, 'avgdl': self.avgdl}
        with open(index_dir / 'index_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f)

    @classmethod
    def load(cls, index_dir: str | Path):
        index_dir = Path(index_dir)
        idx = cls()
        with open(index_dir / 'postings.json', 'r', encoding='utf-8') as f:
            postings_data = json.load(f)
            idx.postings = defaultdict(dict, postings_data)
        with open(index_dir / 'doc_lengths.json', 'r', encoding='utf-8') as f:
            idx.doc_lengths = json.load(f)
        with open(index_dir / 'chunk_info.json', 'r', encoding='utf-8') as f:
            idx.chunk_info = json.load(f)
        with open(index_dir / 'df.json', 'r', encoding='utf-8') as f:
            idx.df = defaultdict(int, json.load(f))
        with open(index_dir / 'index_stats.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
            idx.N = stats['N']
            idx.avgdl = stats['avgdl']
        return idx


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or b < 0 or b > 1:
            raise ValueError("Invalid BM25 parameters")
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, query: str) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if self.index.N == 0:
            return []
        query_tokens = tokenize(query)
        scores = defaultdict(float)
        for term in query_tokens:
            if term not in self.index.postings:
                continue
            df = self.index.df.get(term, 0)
            # IDF: BM25 IDF term (Robertson–Walker) - common form
            idf = math.log((self.index.N - df + 0.5) / (df + 0.5) + 1.0)
            for chunk_id, tf in self.index.postings[term].items():
                doc_len = self.index.doc_lengths.get(chunk_id, 0)
                # BM25 scoring formula
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.index.avgdl if self.index.avgdl > 0 else 1.0))
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / denom if denom > 0 else 0.0
        # Sort by score desc, then by chunk_id for deterministic tie-breaking
        sorted_results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_results):
            info = self.index.chunk_info.get(chunk_id, {})
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=info.get('doc_id', ''),
                    source=info.get('source', ''),
                    title=info.get('title', ''),
                    text=info.get('text', ''),
                    score=float(score),
                    rank=rank,
                    metadata=info,
                )
            )
        return results


class TFIDF:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.N = index.N
        # Precompute IDF
        self.idf = {}
        for term, df in index.df.items():
            # IDF = log(N/df)
            self.idf[term] = math.log(self.N / df) if df > 0 else 0.0

    def score(self, query: str) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if self.index.N == 0:
            return []
        query_tokens = tokenize(query)
        # Query vector (sparse)
        query_tf = defaultdict(int)
        for t in query_tokens:
            query_tf[t] += 1
        query_vec = {}
        for t, tf in query_tf.items():
            if t in self.idf:
                query_vec[t] = (1.0 + math.log(tf)) * self.idf[t] if tf > 0 else 0.0
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if query_norm == 0:
            return []

        scores = {}
        # For each term in query, look up postings
        for term in query_vec:
            if term not in self.index.postings:
                continue
            for chunk_id, tf in self.index.postings[term].items():
                if chunk_id not in scores:
                    # Compute doc vector component on-the-fly contribution
                    scores[chunk_id] = 0.0
        # Compute cosine similarity
        result_scores = {}
        for chunk_id in self.index.chunk_info:
            # Compute dot product
            dot = 0.0
            doc_len = self.index.doc_lengths.get(chunk_id, 0)
            for term, qv in query_vec.items():
                if term in self.index.postings and chunk_id in self.index.postings[term]:
                    tf = self.index.postings[term][chunk_id]
                    # TF-IDF with log normalization
                    tfidf_doc = (1.0 + math.log(tf)) * self.idf.get(term, 0.0)
                    dot += tfidf_doc * qv
            if dot > 0:
                # Compute doc norm (approximate from stored info)
                doc_norm = math.sqrt(sum(
                    ((1.0 + math.log(self.index.postings[t][chunk_id])) * self.idf.get(t, 0.0)) ** 2
                    for t in self.index.postings
                    if chunk_id in self.index.postings[t]
                )) if doc_len > 0 else 1.0
                if doc_norm > 0:
                    result_scores[chunk_id] = dot / (query_norm * doc_norm)

        sorted_results = sorted(result_scores.items(), key=lambda x: (-x[1], x[0]))
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_results):
            info = self.index.chunk_info.get(chunk_id, {})
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=info.get('doc_id', ''),
                    source=info.get('source', ''),
                    title=info.get('title', ''),
                    text=info.get('text', ''),
                    score=float(score),
                    rank=rank,
                    metadata=info,
                )
            )
        return results
