"""Deduplication for kbforge Phase 2.

Exact and near-duplicate detection using normalized content hashes
and MinHash with datasketch.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from kbforge.documents import Document, compute_content_hash

logger = logging.getLogger("kbforge.dedup")


def get_normalized_text_for_dedup(doc: Document) -> str:
    """Get normalized text for deduplication comparison."""
    # Use raw_text as it's already cleaned/normalized
    text = doc.raw_text if doc.raw_text is not None else ""
    return text


def compute_exact_hash(text: str) -> str:
    """Compute stable hash of normalized text for exact duplicate detection."""
    return compute_content_hash(text)


def _word_shingles(text: str, shingle_size: int = 3) -> set[str]:
    """Generate word shingles from text."""
    if not text or not text.strip():
        return set()
    # Tokenize into words
    # Split on whitespace
    words = text.split()
    if len(words) < shingle_size:
        # For short documents, use the whole text as shingles or handle specially
        # Don't treat short docs as duplicates just because of little content
        if len(words) == 0:
            return set()
        # Return the document as a single shingle or small set
        return {" ".join(words)}
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingle = " ".join(words[i : i + shingle_size])
        shingles.add(shingle)
    return shingles


def _create_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    """Create MinHash from shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


class DeduplicationResult:
    def __init__(self):
        self.kept: list[Document] = []
        self.removed: list[tuple[Document, str, str, float | None]] = []  # (removed_doc, reason, representative_id, similarity)
        self.removal_log: list[dict] = []
        self.stats = {}


def _select_representative(documents: list[Document], group_key: str = None) -> Document:
    """
    Deterministic representative selection rule.
    Prefer documents with more useful source information:
    1. Prefer non-empty titles
    2. Prefer longer content (more informative)
    3. Prefer earlier in some stable ordering? Or prefer certain sources
    4. Fall back to lexicographically smallest source/doc_id for determinism
    """
    if not documents:
        raise ValueError("Cannot select representative from empty list")
    
    def score(doc: Document):
        title_score = 1 if doc.title and doc.title.strip() else 0
        content_score = len(doc.raw_text) if doc.raw_text else 0
        # Prefer more metadata
        meta_score = len(doc.metadata)
        # Deterministic tie-breaker
        return (title_score, content_score, meta_score, doc.source, doc.doc_id)
    
    # Return document with highest score
    return sorted(documents, key=score, reverse=True)[0]


def deduplicate_documents(
    documents: Iterable[Document],
    exact: bool = True,
    near_duplicate: bool = True,
    similarity_threshold: float = 0.8,
    shingle_size: int = 3,
    num_perm: int = 128,
) -> tuple[list[Document], list[dict], dict]:
    """
    Deduplicate documents using exact and near-duplicate detection.
    Returns (kept_documents, removal_log, stats)
    """
    docs_list = list(documents)
    removal_log = []
    stats = {
        "input_count": len(docs_list),
        "exact_duplicates_removed": 0,
        "near_duplicates_removed": 0,
        "kept_count": 0,
    }
    
    if not docs_list:
        stats["kept_count"] = 0
        return [], removal_log, stats
    
    kept_docs = []
    removed_map = {}  # doc_id -> removal info
    
    # Step 1: Exact duplicate detection
    if exact:
        # Group by normalized content hash
        hash_groups = defaultdict(list)
        for doc in docs_list:
            text = get_normalized_text_for_dedup(doc)
            h = compute_exact_hash(text)
            hash_groups[h].append(doc)
        
        # Process each group
        for h, group in hash_groups.items():
            if len(group) == 0:
                continue
            if len(group) == 1:
                doc = group[0]
                if doc.doc_id not in removed_map:
                    kept_docs.append(doc)
            else:
                # Multiple exact duplicates - select representative
                representative = _select_representative(group)
                kept_docs.append(representative)
                # Remove others
                for doc in group:
                    if doc.doc_id == representative.doc_id:
                        continue
                    removed_map[doc.doc_id] = {
                        "removed_doc_id": doc.doc_id,
                        "retained_doc_id": representative.doc_id,
                        "reason": "exact",
                        "similarity": 1.0,
                        "removed_source": doc.source,
                        "retained_source": representative.source,
                    }
                    removal_log.append(removed_map[doc.doc_id])
                    stats["exact_duplicates_removed"] += 1
    else:
        kept_docs = list(docs_list)
    
    # Step 2: Near-duplicate detection
    if near_duplicate and len(kept_docs) > 1:
        # Filter out empty and very short documents - don't treat them as duplicates
        # based on too little content
        candidates = []
        for doc in kept_docs:
            text = get_normalized_text_for_dedup(doc)
            words = text.split() if text else []
            if len(words) == 0:
                # Keep empty docs? Or be careful - handle safely
                candidates.append((doc, None, set()))
            elif len(words) < shingle_size:
                # Very short documents - be careful not to treat unrelated short docs as duplicates
                candidates.append((doc, None, _word_shingles(text, shingle_size)))
            else:
                shingles = _word_shingles(text, shingle_size)
                if len(shingles) == 0:
                    candidates.append((doc, None, set()))
                else:
                    candidates.append((doc, _create_minhash(shingles, num_perm), shingles))
        
        # Use LSH for efficiency
        lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
        # Insert all candidates
        for idx, (doc, minhash, shingles) in enumerate(candidates):
            if minhash is not None and len(shingles) > 0:
                try:
                    lsh.insert(f"doc_{idx}", minhash)
                except Exception as e:
                    logger.debug(f"Failed to insert into LSH: {e}")
        
        # Find duplicates
        to_remove = set()
        near_dup_pairs = []
        for idx, (doc, minhash, shingles) in enumerate(candidates):
            if minhash is None or len(shingles) == 0:
                continue  # Don't consider empty/very short with no shingles for near-dup
            try:
                result = lsh.query(minhash)
            except Exception:
                continue
            for other_key in result:
                other_idx = int(other_key.split("_")[1])
                if other_idx <= idx:
                    continue
                other_doc = candidates[other_idx][0]
                if other_doc.doc_id in removed_map or doc.doc_id in removed_map:
                    continue
                if other_doc.doc_id in to_remove or doc.doc_id in to_remove:
                    continue
                # Compute similarity estimate
                try:
                    sim = minhash.jaccard(candidates[other_idx][1])
                except Exception:
                    sim = 0.0
                if sim >= similarity_threshold:
                    # Determine which to keep (representative)
                    group = [doc, other_doc]
                    rep = _select_representative(group)
                    if rep.doc_id == doc.doc_id:
                        # Remove other
                        to_remove.add(other_doc.doc_id)
                        removed_map[other_doc.doc_id] = {
                            "removed_doc_id": other_doc.doc_id,
                            "retained_doc_id": rep.doc_id,
                            "reason": "near_duplicate",
                            "similarity": float(sim),
                            "removed_source": other_doc.source,
                            "retained_source": rep.source,
                        }
                        removal_log.append(removed_map[other_doc.doc_id])
                        stats["near_duplicates_removed"] += 1
                    else:
                        to_remove.add(doc.doc_id)
                        removed_map[doc.doc_id] = {
                            "removed_doc_id": doc.doc_id,
                            "retained_doc_id": rep.doc_id,
                            "reason": "near_duplicate",
                            "similarity": float(sim),
                            "removed_source": doc.source,
                            "retained_source": rep.source,
                        }
                        removal_log.append(removed_map[doc.doc_id])
                        stats["near_duplicates_removed"] += 1
        
        # Rebuild kept list excluding removed
        final_kept = []
        for doc in kept_docs:
            if doc.doc_id in to_remove or doc.doc_id in removed_map and removed_map[doc.doc_id]["removed_doc_id"] == doc.doc_id and any(x["removed_doc_id"] == doc.doc_id and x["reason"] == "near_duplicate" for x in removal_log if x["removed_doc_id"] == doc.doc_id and x.get("reason") == "near_duplicate"):
                # Check if it was removed as near-dup
                was_removed_near = False
                for entry in removal_log:
                    if entry["removed_doc_id"] == doc.doc_id and entry["reason"] == "near_duplicate":
                        was_removed_near = True
                        break
                if was_removed_near:
                    continue
            final_kept.append(doc)
        kept_docs = final_kept
    
    stats["kept_count"] = len(kept_docs)
    return kept_docs, removal_log, stats


# Alias
deduplicate = deduplicate_documents
