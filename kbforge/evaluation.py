"""Retrieval evaluation for kbforge (Phase 6).

This module implements the classical IR metrics used by the evaluation CLI
and a small, versionable JSON/JSONL relevance-judgment format.  Retrieval is
delegated to the existing :class:`kbforge.package.KnowledgeBase`, which in
turn uses the Phase 4 BM25 and TF-IDF rankers unchanged.

Metric conventions
------------------
All metrics operate on a ranked list of *unique* IDs (chunk IDs or document
IDs).  If a retrieved list contains duplicates, only the first occurrence is
kept.  This matters for document-level evaluation because several chunks can
belong to the same document.

Relevance judgments are binary: an ID is either relevant or not.  Empty
retrieved lists and queries with no relevant IDs produce metric value 0.0
instead of undefined/NaN values.  Precision@k uses ``k`` as the denominator,
so missing results in a short list count as non-relevant.  Recall@k divides
by the total number of judged-relevant IDs.  AP and MAP use the full ranked
list (not truncated at k); P@k, R@k and nDCG use the cutoff k.  MRR uses the
full ranked list.

Evaluation units
----------------
``chunk``
    Relevant IDs are chunk IDs and ranked results are chunk IDs.
``document``
    Relevant IDs are document IDs.  Retrieved chunks are collapsed to a
    ranked list of unique document IDs, keeping the first (best-ranked)
    occurrence of each document.  A relevant document is therefore credited
    once even if many of its chunks are retrieved.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from kbforge import __version__
from kbforge.package import KnowledgeBase, SUPPORTED_RANKING_MODES

EVALUATION_FORMAT = "kbforge-evaluation"
EVALUATION_FORMAT_VERSION = "1.0"
SUPPORTED_EVALUATION_VERSIONS = (EVALUATION_FORMAT_VERSION,)
REPORT_FORMAT = "kbforge-evaluation-report"
REPORT_FORMAT_VERSION = "1.0"

SUPPORTED_UNITS = ("chunk", "document")
METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
)


class EvaluationError(ValueError):
    """Raised for invalid datasets, judgments, IDs, or evaluation settings."""


@dataclass(frozen=True)
class QueryJudgment:
    query_id: str
    query: str
    relevant_ids: tuple[str, ...]

    @property
    def relevant_set(self) -> set[str]:
        return set(self.relevant_ids)


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    description: str
    synthetic: bool | None
    unit: str
    queries: tuple[QueryJudgment, ...]
    source_path: Path | None = None
    format_version: str = EVALUATION_FORMAT_VERSION


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than 0")


def _unique_in_order(ids: Iterable[str]) -> list[str]:
    """Keep the first occurrence of each ID; drop later duplicates."""
    if isinstance(ids, str):
        raise ValueError("IDs must be a sequence, not a single string")
    seen: set[str] = set()
    unique: list[str] = []
    for item in ids:
        if not isinstance(item, str):
            raise ValueError("IDs must be strings")
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def precision_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """``|relevant ∩ top-k| / k`` (missing results count as non-relevant)."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _unique_in_order(retrieved)[:k]
    return len(set(ranked) & relevant_set) / float(k)


def recall_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """``|relevant ∩ top-k| / |relevant|``; zero when no judgments exist."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _unique_in_order(retrieved)[:k]
    return len(set(ranked) & relevant_set) / float(len(relevant_set))


def reciprocal_rank(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant result; zero if none is retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, item in enumerate(_unique_in_order(retrieved), start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: Sequence[Iterable[str]],
    relevant_sets: Sequence[Iterable[str]],
) -> float:
    """Average :func:`reciprocal_rank` over paired query result lists."""
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have the same length")
    if not ranked_lists:
        return 0.0
    total = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(ranked_lists, relevant_sets)
    )
    return total / len(ranked_lists)


def dcg_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance DCG@k using ``gain / log2(rank + 1)``."""
    _validate_k(k)
    relevant_set = set(relevant)
    ranked = _unique_in_order(retrieved)[:k]
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in relevant_set
    )


def ndcg_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k; zero if there are no relevant judgments."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant_set, k) / ideal_dcg


def average_precision(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    k: int | None = None,
) -> float:
    """Average Precision over the full ranking, or over the top ``k``.

    ``AP = (1 / |relevant|) * sum(P@i for each relevant result at rank i)``.
    Relevant IDs that are never retrieved therefore reduce the score.
    """
    if k is not None:
        _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _unique_in_order(retrieved)
    if k is not None:
        ranked = ranked[:k]
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hits += 1
            total += hits / float(rank)
    return total / float(len(relevant_set))


def mean_average_precision(
    ranked_lists: Sequence[Iterable[str]],
    relevant_sets: Sequence[Iterable[str]],
) -> float:
    """Average :func:`average_precision` over paired query judgment sets."""
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have the same length")
    if not ranked_lists:
        return 0.0
    total = sum(
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(ranked_lists, relevant_sets)
    )
    return total / len(ranked_lists)


def _parse_unit(value: object, description: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_UNITS:
        raise EvaluationError(
            f"{description} must be one of {SUPPORTED_UNITS}, got {value!r}"
        )
    return value


def _dataset_queries(raw_queries: object, path: Path) -> tuple[QueryJudgment, ...]:
    if not isinstance(raw_queries, list):
        raise EvaluationError("evaluation dataset 'queries' must be a list")
    if not raw_queries:
        raise EvaluationError("evaluation dataset contains no queries")

    judgments: list[QueryJudgment] = []
    seen_query_ids: set[str] = set()
    for index, raw in enumerate(raw_queries, start=1):
        where = f"{path.name}: query #{index}"
        if not isinstance(raw, dict):
            raise EvaluationError(f"{where} must be a JSON object")

        query_id = raw.get("query_id", raw.get("id"))
        if not isinstance(query_id, str) or not query_id.strip():
            raise EvaluationError(f"{where} needs a non-empty string 'query_id'")
        query_id = query_id.strip()
        if query_id in seen_query_ids:
            raise EvaluationError(f"duplicate query_id '{query_id}' in {path.name}")
        seen_query_ids.add(query_id)

        query_text = raw.get("query", raw.get("text"))
        if not isinstance(query_text, str) or not query_text.strip():
            raise EvaluationError(f"{where} needs a non-empty string 'query'")

        if "relevant" in raw:
            raw_relevant = raw["relevant"]
        else:
            raw_relevant = raw.get("relevant_ids")
        if not isinstance(raw_relevant, list):
            raise EvaluationError(
                f"{where} needs a list of relevant IDs in 'relevant'"
            )

        relevant: list[str] = []
        seen_relevant: set[str] = set()
        for item in raw_relevant:
            if not isinstance(item, str) or not item.strip():
                raise EvaluationError(
                    f"{where} has an invalid relevant ID: {item!r}"
                )
            normalized = item.strip()
            if normalized not in seen_relevant:
                seen_relevant.add(normalized)
                relevant.append(normalized)

        judgments.append(
            QueryJudgment(
                query_id=query_id,
                query=query_text.strip(),
                relevant_ids=tuple(relevant),
            )
        )
    return tuple(judgments)


def load_dataset(
    path: str | Path,
    *,
    requested_unit: str | None = None,
    default_unit: str = "chunk",
) -> EvaluationDataset:
    """Load and validate a JSON or JSONL relevance-judgment dataset.

    JSON can be either an object with format metadata and a ``queries`` list,
    or a bare list of query objects.  JSONL files contain one query object per
    non-empty line and use the requested/default unit and filename as name.
    """
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"evaluation dataset not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"could not read evaluation dataset: {exc}") from exc
    if not text.strip():
        raise EvaluationError(f"evaluation dataset is empty: {path}")

    def parse_jsonl() -> list[object]:
        records: list[object] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"invalid JSON/JSONL in {path.name} at line {line_number}: {exc.msg}"
                ) from exc
        return records

    if path.suffix.lower() in (".jsonl", ".ndjson"):
        data = parse_jsonl()
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = parse_jsonl()

    if isinstance(data, dict) and "queries" not in data and (
        "query" in data or "text" in data
    ):
        data = [data]

    if isinstance(data, list):
        root: dict = {}
        raw_queries = data
    elif isinstance(data, dict):
        root = data
        raw_queries = data.get("queries")
    else:
        raise EvaluationError("evaluation dataset root must be an object or list")

    if root.get("format") is not None and root["format"] != EVALUATION_FORMAT:
        raise EvaluationError(
            f"unsupported evaluation format {root['format']!r}; "
            f"expected {EVALUATION_FORMAT!r}"
        )
    version = root.get("format_version", EVALUATION_FORMAT_VERSION)
    if version not in SUPPORTED_EVALUATION_VERSIONS:
        raise EvaluationError(
            f"unsupported evaluation dataset version {version!r}; "
            f"supported: {', '.join(SUPPORTED_EVALUATION_VERSIONS)}"
        )

    if requested_unit is not None:
        _parse_unit(requested_unit, "requested unit")
    dataset_unit = root.get("unit")
    if dataset_unit is not None:
        _parse_unit(dataset_unit, "dataset unit")
    if requested_unit is not None and dataset_unit is not None and requested_unit != dataset_unit:
        raise EvaluationError(
            f"requested unit '{requested_unit}' does not match dataset unit "
            f"'{dataset_unit}'"
        )
    unit = requested_unit or dataset_unit or default_unit
    _parse_unit(unit, "evaluation unit")

    synthetic = root.get("synthetic")
    if synthetic is not None and not isinstance(synthetic, bool):
        raise EvaluationError("dataset 'synthetic' must be a boolean when present")

    name = root.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise EvaluationError("dataset 'name' must be a non-empty string when present")
    description = root.get("description", "")
    if not isinstance(description, str):
        raise EvaluationError("dataset 'description' must be a string when present")

    queries = _dataset_queries(raw_queries, path)
    return EvaluationDataset(
        name=name.strip(),
        description=description,
        synthetic=synthetic,
        unit=unit,
        queries=queries,
        source_path=path,
        format_version=str(version),
    )


def _known_ids(knowledge_base: KnowledgeBase, unit: str) -> set[str]:
    if unit == "chunk":
        return set(knowledge_base.index.chunk_info.keys())
    return {
        info.get("doc_id")
        for info in knowledge_base.index.chunk_info.values()
        if info.get("doc_id")
    }


def validate_dataset_against_kb(
    dataset: EvaluationDataset, knowledge_base: KnowledgeBase
) -> None:
    """Fail clearly if judgments reference IDs absent from the loaded index."""
    unit = _parse_unit(dataset.unit, "evaluation unit")
    known = _known_ids(knowledge_base, unit)
    for judgment in dataset.queries:
        missing = [item for item in judgment.relevant_ids if item not in known]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} total)"
            raise EvaluationError(
                f"query '{judgment.query_id}' references unknown "
                f"{unit} IDs: {preview}"
            )


def _ranked_ids(
    knowledge_base: KnowledgeBase,
    query: str,
    mode: str,
    unit: str,
) -> list[str]:
    limit = max(1, knowledge_base.index.N)
    results = knowledge_base.search(query, mode=mode, limit=limit)
    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        identifier = result.chunk_id if unit == "chunk" else result.doc_id
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        ranked.append(identifier)
    return ranked


def _query_metrics(retrieved: list[str], relevant: tuple[str, ...], k: int) -> dict:
    independent = set(relevant)
    return {
        "precision_at_k": precision_at_k(retrieved, independent, k),
        "recall_at_k": recall_at_k(retrieved, independent, k),
        "reciprocal_rank": reciprocal_rank(retrieved, independent),
        "ndcg_at_k": ndcg_at_k(retrieved, independent, k),
        "average_precision": average_precision(retrieved, independent),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "query_count": 0,
            **{name: 0.0 for name in METRIC_NAMES},
            "mrr": 0.0,
            "map": 0.0,
        }
    count = len(rows)
    result = {
        "query_count": count,
        "precision_at_k": sum(row["precision_at_k"] for row in rows) / count,
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "ndcg_at_k": sum(row["ndcg_at_k"] for row in rows) / count,
        "map": sum(row["average_precision"] for row in rows) / count,
    }
    return result


def evaluate(
    knowledge_base: KnowledgeBase,
    dataset: EvaluationDataset,
    *,
    k: int = 5,
    modes: Sequence[str] = SUPPORTED_RANKING_MODES,
) -> dict:
    """Evaluate one or more ranking modes and return a JSON-serializable report."""
    _validate_k(k)
    _parse_unit(dataset.unit, "evaluation unit")
    if isinstance(modes, str):
        modes = [modes]
    selected: list[str] = []
    for mode in modes:
        if mode not in SUPPORTED_RANKING_MODES:
            raise EvaluationError(
                f"unsupported ranking mode '{mode}'; "
                f"choose from {', '.join(SUPPORTED_RANKING_MODES)}"
            )
        if mode not in knowledge_base.supported_modes:
            raise EvaluationError(
                f"ranking mode '{mode}' is not available in this package"
            )
        if mode not in selected:
            selected.append(mode)
    if not selected:
        raise EvaluationError("at least one ranking mode is required")

    validate_dataset_against_kb(dataset, knowledge_base)

    per_query: dict[str, list[dict]] = {}
    aggregate: dict[str, dict] = {}
    for mode in selected:
        rows: list[dict] = []
        for judgment in dataset.queries:
            retrieved = _ranked_ids(
                knowledge_base, judgment.query, mode, dataset.unit
            )
            metrics = _query_metrics(retrieved, judgment.relevant_ids, k)
            rows.append(
                {
                    "query_id": judgment.query_id,
                    "query": judgment.query,
                    "unit": dataset.unit,
                    "k": k,
                    "relevant_count": len(judgment.relevant_ids),
                    "retrieved_count": len(retrieved),
                    "retrieved_ids": retrieved[:k],
                    "relevant_retrieved_ids": [
                        item for item in retrieved[:k] if item in judgment.relevant_set
                    ],
                    **metrics,
                }
            )
        per_query[mode] = rows
        aggregate[mode] = _aggregate(rows)

    report: dict = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_FORMAT_VERSION,
        "generated_by": {"name": "kbforge", "version": __version__},
        "dataset": {
            "name": dataset.name,
            "description": dataset.description,
            "synthetic": dataset.synthetic,
            "format_version": dataset.format_version,
            "source_file": dataset.source_path.name if dataset.source_path else None,
            "query_count": len(dataset.queries),
        },
        "configuration": {
            "k": k,
            "unit": dataset.unit,
            "modes": selected,
        },
        "aggregate": aggregate,
        "per_query": per_query,
        "skipped_queries": [],
        "notes": [
            "Binary relevance judgments are used.",
            "Duplicate retrieved IDs are ignored after their first occurrence.",
            "P@k, R@k and nDCG use cutoff k; AP/MAP and MRR use the full ranking.",
            "Queries with no relevant IDs or no retrieved results score 0.0.",
            "AP/MAP use the full ranked list; missing relevant IDs reduce the score.",
        ],
    }

    if "bm25" in aggregate and "tfidf" in aggregate:
        report["comparison"] = {
            "bm25_minus_tfidf": {
                metric: aggregate["bm25"][metric] - aggregate["tfidf"][metric]
                for metric in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map")
            }
        }
    return report


def write_report(report: dict, path: str | Path) -> Path:
    """Write a deterministic JSON report atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path
