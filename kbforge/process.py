"""Phase 2 processing pipeline: cleaning and deduplication."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from kbforge.clean import clean_documents
from kbforge.config import Config
from kbforge.dedup import deduplicate_documents
from kbforge.documents import Document, write_documents_jsonl

logger = logging.getLogger("kbforge.process")


def process_documents(
    documents: Iterable[Document],
    config: Config | None = None,
    enable_cleaning: bool | None = None,
    enable_deduplication: bool | None = None,
):
    """
    Process documents through Phase 2: cleaning and deduplication.
    Returns (processed_documents, removal_log, stats)
    """
    docs_list = list(documents)
    
    cleaning_enabled = enable_cleaning if enable_cleaning is not None else (config.cleaning.enabled if config and hasattr(config, 'cleaning') else True)
    dedup_enabled = enable_deduplication if enable_deduplication is not None else (config.deduplication.enabled if config and hasattr(config, 'deduplication') else True)
    
    all_stats = {
        "input_documents": len(docs_list),
    }
    removal_log = []
    cleaning_failures = []
    
    current_docs = docs_list
    
    # Stage 1: Cleaning
    if cleaning_enabled:
        current_docs, cleaning_failures, clean_stats = clean_documents(current_docs)
        all_stats.update(clean_stats)
        if cleaning_failures:
            for cf in cleaning_failures:
                removal_log.append({
                    "type": "cleaning_failure",
                    **cf
                })
    else:
        all_stats["cleaning_skipped"] = True
    
    # Stage 2: Deduplication
    if dedup_enabled and config and hasattr(config, 'deduplication'):
        dedup_cfg = config.deduplication
        current_docs, dedup_removals, dedup_stats = deduplicate_documents(
            current_docs,
            exact=dedup_cfg.exact,
            near_duplicate=dedup_cfg.near_duplicate,
            similarity_threshold=dedup_cfg.similarity_threshold,
            shingle_size=dedup_cfg.shingle_size,
            num_perm=dedup_cfg.num_perm,
        )
        all_stats.update(dedup_stats)
        removal_log.extend(dedup_removals)
    elif dedup_enabled:
        current_docs, dedup_removals, dedup_stats = deduplicate_documents(current_docs)
        all_stats.update(dedup_stats)
        removal_log.extend(dedup_removals)
    else:
        all_stats["deduplication_skipped"] = True
    
    all_stats["output_documents"] = len(current_docs)
    
    return current_docs, removal_log, all_stats


def run_process(config: Config, documents: Iterable[Document] | None = None, input_path: str | Path | None = None) -> dict:
    """Run Phase 2 processing and write outputs."""
    from kbforge.documents import read_documents_jsonl
    
    if documents is None and input_path is not None:
        documents = read_documents_jsonl(input_path)
    elif documents is None:
        # Try default input
        input_path = Path(config.output_dir) / "documents.jsonl"
        documents = read_documents_jsonl(input_path)
    
    processed_docs, removal_log, stats = process_documents(documents, config=config)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write cleaned/deduplicated documents
    cleaned_path = output_dir / "documents_cleaned.jsonl"
    write_documents_jsonl(processed_docs, cleaned_path)
    
    # Write removal log
    removal_log_path = output_dir / "removal_log.json"
    removal_log_data = {
        "removals": removal_log,
        "stats": stats,
    }
    removal_log_path.write_text(json.dumps(removal_log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Write stats
    stats_path = output_dir / "phase2_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return {
        "processed_documents": len(processed_docs),
        "removals": len(removal_log),
        "cleaned_path": str(cleaned_path),
        "removal_log_path": str(removal_log_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }
