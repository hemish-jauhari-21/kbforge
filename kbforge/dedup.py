from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from kbforge.documents import Document, compute_content_hash

logger = logging.getLogger("kbforge.dedup")


def get_normalized_text_for_dedup(doc: Document) -> str:
    text = doc.raw_text if doc.raw_text is not None else ""
    return text


def compute_exact_hash(text: str) -> str:
    return compute_content_hash(text)


def _word_shingles(text: str, shingle_size: int = 3) -> set[str]:
    if not text or not text.strip():
        return set()
    words = text.split()
    if len(words) < shingle_size:
        if len(words) == 0:
            return set()
        return {" ".join(words)}
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingle = " ".join(words[i : i + shingle_size])
        shingles.add(shingle)
    return shingles


def _create_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _select_representative(documents: list[Document], group_key: str = None) -> Document:
    if not documents:
        raise ValueError("Cannot select representative from empty list")

    def score(doc: Document):
        title_score = 1 if doc.title and doc.title.strip() else 0
        content_score = len(doc.raw_text) if doc.raw_text else 0
        meta_score = len(doc.metadata)
        return (title_score, content_score, meta_score, doc.source, doc.doc_id)

    return sorted(documents, key=score, reverse=True)[0]


def _exact_shingle_jaccard(text1: str, text2: str, shingle_size: int = 3) -> float:
    s1 = _word_shingles(text1, shingle_size)
    s2 = _word_shingles(text2, shingle_size)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return 0.0
    return inter / union


def deduplicate_documents(
    documents: Iterable[Document],
    exact: bool = True,
    near_duplicate: bool = True,
    similarity_threshold: float = 0.8,
    shingle_size: int = 3,
    num_perm: int = 128,
) -> tuple[list[Document], list[dict], dict]:
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
    removed_map = {}

    # Step 1: Exact duplicate detection
    if exact:
        hash_groups = defaultdict(list)
        for doc in docs_list:
            text = get_normalized_text_for_dedup(doc)
            h = compute_exact_hash(text)
            hash_groups[h].append(doc)

        for h, group in hash_groups.items():
            if len(group) == 0:
                continue
            if len(group) == 1:
                doc = group[0]
                if doc.doc_id not in removed_map:
                    kept_docs.append(doc)
            else:
                representative = _select_representative(group)
                kept_docs.append(representative)
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
        candidates = []
        for doc in kept_docs:
            text = get_normalized_text_for_dedup(doc)
            words = text.split() if text else []
            if len(words) == 0:
                candidates.append((doc, None, set()))
            elif len(words) < shingle_size:
                candidates.append((doc, None, _word_shingles(text, shingle_size)))
            else:
                shingles = _word_shingles(text, shingle_size)
                if len(shingles) == 0:
                    candidates.append((doc, None, set()))
                else:
                    candidates.append((doc, _create_minhash(shingles, num_perm), shingles))

        lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
        for idx, (doc, minhash, shingles) in enumerate(candidates):
            if minhash is not None and len(shingles) > 0:
                try:
                    lsh.insert(f"doc_{idx}", minhash)
                except Exception as e:
                    logger.debug(f"Failed to insert into LSH: {e}")

        to_remove = set()
        for idx, (doc, minhash, shingles) in enumerate(candidates):
            if minhash is None or len(shingles) == 0:
                continue
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
                try:
                    sim_estimate = minhash.jaccard(candidates[other_idx][1])  # MinHash estimate
                except Exception:
                    sim_estimate = 0.0
                if sim_estimate >= similarity_threshold:
                    group = [doc, other_doc]
                    rep = _select_representative(group)
                    if rep.doc_id == doc.doc_id:
                        to_remove.add(other_doc.doc_id)
                        removal_log.append(
                            {
                                "removed_doc_id": other_doc.doc_id,
                                "retained_doc_id": rep.doc_id,
                                "reason": "near_duplicate",
                                "similarity": float(sim_estimate),
                                "similarity_estimate_minhash": float(sim_estimate),
                                "similarity_exact_jaccard": float(_exact_shingle_jaccard(doc.raw_text, other_doc.raw_text, shingle_size)),
                                "removed_source": other_doc.source,
                                "retained_source": rep.source,
                            }
                        )
                        stats["near_duplicates_removed"] += 1
                    else:
                        to_remove.add(doc.doc_id)
                        removal_log.append(
                            {
                                "removed_doc_id": doc.doc_id,
                                "retained_doc_id": rep.doc_id,
                                "reason": "near_duplicate",
                                "similarity": float(sim_estimate),
                                "similarity_estimate_minhash": float(sim_estimate),
                                "similarity_exact_jaccard": float(_exact_shingle_jaccard(doc.raw_text, other_doc.raw_text, shingle_size)),
                                "removed_source": doc.source,
                                "retained_source": rep.source,
                            }
                        )
                        stats["near_duplicates_removed"] += 1

        final_kept = []
        for doc in kept_docs:
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


deduplicate = deduplicate_documents
