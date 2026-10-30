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


def _partition_cleaning_diagnostics(
    diagnostics: Iterable[dict],
) -> tuple[list[dict], list[dict]]:
    warnings = []
    exclusions = []
    for diagnostic in diagnostics:
        if diagnostic.get("outcome") == "excluded":
            exclusions.append(diagnostic)
        else:
            warnings.append(diagnostic)
    return warnings, exclusions


def process_documents(
    documents: Iterable[Document],
    config: Config | None = None,
    enable_cleaning: bool | None = None,
    enable_deduplication: bool | None = None,
):
    """
    Process documents through Phase 2: cleaning and deduplication.
    Returns (processed_documents, duplicate_removals, cleaning_diagnostics, stats)
    """
    docs_list = list(documents)

    cleaning_enabled = enable_cleaning if enable_cleaning is not None else (config.cleaning.enabled if config and hasattr(config, 'cleaning') else True)
    dedup_enabled = enable_deduplication if enable_deduplication is not None else (config.deduplication.enabled if config and hasattr(config, 'deduplication') else True)

    all_stats = {
        "input_documents": len(docs_list),
        "duplicate_removals": 0,
        "cleaning_warnings": 0,
        "documents_excluded_by_cleaning": 0,
    }
    removal_log = []
    cleaning_diagnostics = []

    current_docs = docs_list

    # Stage 1: Cleaning
    if cleaning_enabled:
        current_docs, cleaning_diagnostics, clean_stats = clean_documents(current_docs)
        all_stats.update(clean_stats)
        cleaning_warnings, cleaning_exclusions = _partition_cleaning_diagnostics(
            cleaning_diagnostics
        )
        all_stats["cleaning_warnings"] = len(cleaning_warnings)
        all_stats["documents_excluded_by_cleaning"] = len(cleaning_exclusions)
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

    all_stats["duplicate_removals"] = len(removal_log)
    all_stats["output_documents"] = len(current_docs)

    return current_docs, removal_log, cleaning_diagnostics, all_stats


def run_process(config: Config, documents: Iterable[Document] | None = None, input_path: str | Path | None = None) -> dict:
    """Run Phase 2 processing and write outputs."""
    from kbforge.documents import read_documents_jsonl
    
    if documents is None and input_path is not None:
        documents = read_documents_jsonl(input_path)
    elif documents is None:
        # Try default input
        input_path = Path(config.output_dir) / "documents.jsonl"
        documents = read_documents_jsonl(input_path)
    
    processed_docs, removal_log, cleaning_diagnostics, stats = process_documents(
        documents, config=config
    )
    cleaning_warnings, cleaning_exclusions = _partition_cleaning_diagnostics(
        cleaning_diagnostics
    )
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write cleaned/deduplicated documents
    cleaned_path = output_dir / "documents_cleaned.jsonl"
    write_documents_jsonl(processed_docs, cleaned_path)
    
    # Write removal log
    removal_log_path = output_dir / "removal_log.json"
    removal_log_data = {
        "removals": removal_log,
        "cleaning_warnings": cleaning_warnings,
        "cleaning_exclusions": cleaning_exclusions,
        "stats": stats,
    }
    removal_log_path.write_text(json.dumps(removal_log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Write stats
    stats_path = output_dir / "phase2_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return {
        "processed_documents": len(processed_docs),
        "removals": len(removal_log),
        "cleaning_warnings": len(cleaning_warnings),
        "cleaning_exclusions": len(cleaning_exclusions),
        "cleaned_path": str(cleaned_path),
        "removal_log_path": str(removal_log_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def run_chunk(config, documents=None, input_path=None):
    from kbforge.documents import read_documents_jsonl, write_documents_jsonl
    from kbforge.chunk import chunk_documents, chunk_to_dict
    import json
    from pathlib import Path

    if documents is None and input_path is not None:
        documents = read_documents_jsonl(input_path)
    elif documents is None:
        cleaning_enabled = getattr(getattr(config, "cleaning", None), "enabled", True)
        dedup_enabled = getattr(getattr(config, "deduplication", None), "enabled", True)
        phase2_enabled = cleaning_enabled or dedup_enabled
        # Only use cleaned documents when Phase 2 ran for this build. Otherwise
        # a cleaned file left by an earlier build would be silently reused.
        cleaned_path = Path(config.output_dir) / "documents_cleaned.jsonl"
        if phase2_enabled and cleaned_path.exists():
            documents = read_documents_jsonl(cleaned_path)
        else:
            documents = read_documents_jsonl(Path(config.output_dir) / "documents.jsonl")

    docs_list = list(documents)
    strategy = getattr(config.chunking, 'strategy', 'fixed')
    size = getattr(config.chunking, 'size', 300)
    overlap = getattr(config.chunking, 'overlap', 50)
    chunks = chunk_documents(docs_list, strategy=strategy, size=size, overlap=overlap)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(chunk_to_dict(c), ensure_ascii=False) + '\n')

    return {"chunks_written": len(chunks), "chunks_path": str(chunks_path)}


def run_index(config, chunks=None, chunks_path=None):
    from kbforge.index import InvertedIndex
    from pathlib import Path
    import json

    if chunks is None:
        if chunks_path is None:
            chunks_path = Path(config.output_dir) / "chunks.jsonl"
        chunks = []
        if Path(chunks_path).exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
    
    index = InvertedIndex()
    index.build(chunks)
    index_dir = Path(config.output_dir) / "index"
    index.save(index_dir)
    return {"chunks_indexed": index.N, "index_dir": str(index_dir)}
