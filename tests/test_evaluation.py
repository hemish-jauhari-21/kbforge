"""Phase 6 evaluation metrics, datasets, reports and CLI tests (offline)."""

from __future__ import annotations

import json
import math

import pytest

from kbforge.cli import main
from kbforge.evaluation import (
    EvaluationDataset,
    EvaluationError,
    QueryJudgment,
    average_precision,
    evaluate,
    load_dataset,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    write_report,
)
from kbforge.package import build_package, load_package


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _doc_ids(kb):
    return {
        info["source"]: info["doc_id"]
        for info in kb.index.chunk_info.values()
        if info.get("source") and info.get("doc_id")
    }


def _dataset(unit, queries):
    return EvaluationDataset(
        name="offline",
        description="small offline judgments",
        synthetic=True,
        unit=unit,
        queries=tuple(queries),
    )


def test_precision_recall_deduplicate_and_short_lists():
    retrieved = ["a", "a", "b", "c"]
    relevant = ["a", "c"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert precision_at_k(["a"], ["a", "b"], 4) == pytest.approx(1 / 4)
    assert recall_at_k(["a"], ["a", "b"], 4) == pytest.approx(0.5)
    assert precision_at_k([], ["a"], 3) == 0.0
    assert recall_at_k([], ["a"], 3) == 0.0


def test_precision_and_recall_with_no_relevant_judgments_are_zero():
    assert precision_at_k(["a"], [], 3) == 0.0
    assert recall_at_k(["a"], [], 3) == 0.0
    assert reciprocal_rank(["a"], []) == 0.0
    assert ndcg_at_k(["a"], [], 3) == 0.0
    assert average_precision(["a"], []) == 0.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "a", "b"], ["b"]) == pytest.approx(0.5)
    assert reciprocal_rank(["a"], ["b"]) == 0.0
    assert mean_reciprocal_rank([["a", "b"], ["b"]], [["b"], ["a"]]) == pytest.approx(0.25)
    assert mean_reciprocal_rank([], []) == 0.0


def test_ndcg_uses_binary_gain_and_ideal_ranking():
    expected = 1 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c"], ["b"], 3) == pytest.approx(expected)
    assert ndcg_at_k(["b", "a"], ["b"], 3) == pytest.approx(1.0)
    assert ndcg_at_k(["a"], ["b"], 3) == 0.0


def test_average_precision_and_map():
    assert average_precision(["a", "b", "c", "d"], ["b", "d"]) == pytest.approx(0.5)
    assert average_precision(["a", "b", "c", "d"], ["b", "d"], k=2) == pytest.approx(
        0.25
    )
    assert mean_average_precision(
        [["a", "b", "c", "d"], ["x"]], [["b", "d"], ["x"]]
    ) == pytest.approx(0.75)
    assert mean_average_precision([], []) == 0.0


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True])
def test_metrics_reject_invalid_k(bad_k):
    with pytest.raises(ValueError, match="k"):
        precision_at_k(["a"], ["a"], bad_k)


def test_load_json_dataset_aliases_and_provenance(tmp_path):
    path = _write_json(
        tmp_path / "judgments.json",
        {
            "format": "kbforge-evaluation",
            "format_version": "1.0",
            "name": "docs",
            "description": "example",
            "synthetic": True,
            "unit": "document",
            "queries": [
                {
                    "id": "q1",
                    "text": "python",
                    "relevant_ids": ["d1", "d1", "d2"],
                }
            ],
        },
    )

    dataset = load_dataset(path)

    assert dataset.name == "docs"
    assert dataset.unit == "document"
    assert dataset.synthetic is True
    assert dataset.source_path == path
    query = dataset.queries[0]
    assert query.query_id == "q1"
    assert query.query == "python"
    assert query.relevant_ids == ("d1", "d2")


def test_load_jsonl_dataset_defaults_to_chunk(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "q1", "query": "python", "relevant": ["c1"]}),
                json.dumps({"query_id": "q2", "query": "index", "relevant": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(path)

    assert dataset.name == "queries"
    assert dataset.unit == "chunk"
    assert dataset.synthetic is None
    assert [query.query_id for query in dataset.queries] == ["q1", "q2"]


def test_load_single_line_jsonl_dataset(tmp_path):
    path = tmp_path / "single.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "python", "relevant": ["c1"]}) + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(path)

    assert len(dataset.queries) == 1
    assert dataset.queries[0].query_id == "q1"


def test_load_dataset_requires_requested_unit_to_match(tmp_path):
    path = _write_json(
        tmp_path / "judgments.json",
        {"unit": "document", "queries": [{"query_id": "q", "query": "x", "relevant": []}]},
    )

    with pytest.raises(EvaluationError, match="does not match"):
        load_dataset(path, requested_unit="chunk")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"queries": []}, "no queries"),
        ({"queries": {}}, "must be a list"),
        ({"format": "other", "queries": []}, "unsupported evaluation format"),
        (
            {"format_version": "99", "queries": [{"query_id": "q", "query": "x", "relevant": []}]},
            "unsupported evaluation dataset version",
        ),
        ({"synthetic": "yes", "queries": [{"query_id": "q", "query": "x", "relevant": []}]}, "synthetic"),
        ({"queries": [{"query_id": "q", "query": "", "relevant": []}]}, "query"),
        ({"queries": [{"query_id": "q", "query": "x"}]}, "relevant"),
        ({"queries": [{"query_id": "q", "query": "x", "relevant": "d1"}]}, "list"),
        (
            {
                "queries": [
                    {"query_id": "q", "query": "x", "relevant": []},
                    {"query_id": "q", "query": "y", "relevant": []},
                ]
            },
            "duplicate query_id",
        ),
    ],
)
def test_load_dataset_rejects_invalid_payloads(tmp_path, payload, message):
    path = _write_json(tmp_path / "bad.json", payload)
    with pytest.raises(EvaluationError, match=message):
        load_dataset(path)


def test_load_dataset_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_dataset(tmp_path / "missing.json")

    empty = tmp_path / "empty.json"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="empty"):
        load_dataset(empty)


def test_evaluate_document_level_both_modes(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)
    docs = _doc_ids(kb)
    dataset = _dataset(
        "document",
        [
            QueryJudgment("q-python", "python", (docs["https://kb.test/python"],)),
            QueryJudgment("q-index", "inverted index", (docs["https://kb.test/search"],)),
        ],
    )

    report = evaluate(kb, dataset, k=1, modes=("bm25", "tfidf"))

    assert report["format"] == "kbforge-evaluation-report"
    assert report["configuration"] == {"k": 1, "unit": "document", "modes": ["bm25", "tfidf"]}
    assert report["dataset"]["synthetic"] is True
    for mode in ("bm25", "tfidf"):
        aggregate = report["aggregate"][mode]
        assert aggregate["query_count"] == 2
        assert aggregate["precision_at_k"] == pytest.approx(1.0)
        assert aggregate["recall_at_k"] == pytest.approx(1.0)
        assert aggregate["mrr"] == pytest.approx(1.0)
        assert aggregate["ndcg_at_k"] == pytest.approx(1.0)
        assert aggregate["map"] == pytest.approx(1.0)
        for row in report["per_query"][mode]:
            assert len(row["retrieved_ids"]) == len(set(row["retrieved_ids"]))
            assert row["retrieved_ids"][0].startswith("web-")
    assert report["comparison"]["bm25_minus_tfidf"]["map"] == pytest.approx(0.0)


def test_evaluate_chunk_level_and_no_relevant_queries(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)
    chunk_ids = {
        info["source"]: chunk_id
        for chunk_id, info in kb.index.chunk_info.items()
        if info.get("source")
    }
    dataset = _dataset(
        "chunk",
        [
            QueryJudgment("q-python", "python", (chunk_ids["https://kb.test/python"],)),
            QueryJudgment("q-empty", "no-such-term", ()),
        ],
    )

    report = evaluate(kb, dataset, k=3, modes=["bm25"])

    assert report["configuration"]["modes"] == ["bm25"]
    assert "comparison" not in report
    rows = {row["query_id"]: row for row in report["per_query"]["bm25"]}
    assert rows["q-python"]["retrieved_ids"][0] == chunk_ids["https://kb.test/python"]
    assert rows["q-python"]["precision_at_k"] == pytest.approx(1 / 3)
    assert rows["q-empty"]["retrieved_count"] == 0
    for metric in (
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "average_precision",
    ):
        assert rows["q-empty"][metric] == 0.0
        assert not math.isnan(rows["q-empty"][metric])


def test_evaluate_rejects_unknown_ids_and_bad_modes(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)
    unknown = _dataset(
        "document",
        [QueryJudgment("q", "python", ("does-not-exist",))],
    )
    with pytest.raises(EvaluationError, match="unknown document IDs"):
        evaluate(kb, unknown, k=3)

    valid = _dataset("document", [QueryJudgment("q", "python", ())])
    with pytest.raises(EvaluationError, match="unsupported ranking mode"):
        evaluate(kb, valid, k=3, modes=["dense"])

    bad_unit = _dataset("page", [QueryJudgment("q", "python", ())])
    with pytest.raises(EvaluationError, match="evaluation unit"):
        evaluate(kb, bad_unit, k=3)


def test_write_report_is_deterministic_json(tmp_path):
    report = {"format": "kbforge-evaluation-report", "value": 0.5}
    path = write_report(report, tmp_path / "nested" / "report.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report
    first = path.read_bytes()
    write_report(report, path)
    assert path.read_bytes() == first


def test_evaluate_cli_writes_report_without_rebuilding(
    sample_kb, tmp_path, monkeypatch, capsys
):
    config, output_dir = sample_kb
    build_package(config)
    kb = load_package(output_dir)
    docs = _doc_ids(kb)
    queries = _write_json(
        tmp_path / "judgments.json",
        {
            "format": "kbforge-evaluation",
            "format_version": "1.0",
            "name": "cli synthetic",
            "synthetic": True,
            "unit": "document",
            "queries": [
                {
                    "query_id": "q-python",
                    "query": "python",
                    "relevant": [docs["https://kb.test/python"]],
                }
            ],
        },
    )
    output = tmp_path / "report.json"

    def fail(*args, **kwargs):
        raise AssertionError("evaluate must not crawl or rebuild")

    monkeypatch.setattr("kbforge.ingest.crawler.run_ingest", fail)
    monkeypatch.setattr("kbforge.process.run_process", fail)
    monkeypatch.setattr("kbforge.process.run_chunk", fail)
    monkeypatch.setattr("kbforge.process.run_index", fail)

    exit_code = main(
        [
            "evaluate",
            "--package",
            str(output_dir),
            "--queries",
            str(queries),
            "--k",
            "1",
            "--output",
            str(output),
            "--mode",
            "both",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "synthetic judgments: yes" in out
    assert "bm25" in out and "tfidf" in out
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["format"] == "kbforge-evaluation-report"
    assert report["configuration"]["modes"] == ["bm25", "tfidf"]
    assert report["per_query"]["bm25"][0]["query_id"] == "q-python"


def test_evaluate_cli_validates_inputs_and_judgments(
    sample_kb, tmp_path, capsys
):
    config, output_dir = sample_kb
    build_package(config)
    mismatched = _write_json(
        tmp_path / "document.json",
        {"unit": "document", "queries": [{"query_id": "q", "query": "x", "relevant": []}]},
    )
    unknown = _write_json(
        tmp_path / "unknown.json",
        {"unit": "document", "queries": [{"query_id": "q", "query": "x", "relevant": ["missing"]}]},
    )

    assert (
        main(
            [
                "evaluate",
                "--package",
                str(output_dir),
                "--queries",
                str(mismatched),
                "--unit",
                "chunk",
            ]
        )
        == 2
    )
    assert "does not match" in capsys.readouterr().err

    assert (
        main(
            [
                "evaluate",
                "--package",
                str(output_dir),
                "--queries",
                str(unknown),
                "--k",
                "0",
            ]
        )
        == 2
    )
    assert "--k" in capsys.readouterr().err

    assert (
        main(
            [
                "evaluate",
                "--package",
                str(output_dir),
                "--queries",
                str(unknown),
            ]
        )
        == 1
    )
    assert "unknown document IDs" in capsys.readouterr().err


def test_evaluate_cli_help_and_missing_dataset(sample_kb, tmp_path, capsys):
    config, output_dir = sample_kb
    build_package(config)

    exit_code = main(
        [
            "evaluate",
            "--package",
            str(output_dir),
            "--queries",
            str(tmp_path / "missing.json"),
        ]
    )
    assert exit_code == 2
    assert "not found" in capsys.readouterr().err

    try:
        main(["evaluate", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--queries" in out
    assert "--unit" in out
    assert "--mode" in out
