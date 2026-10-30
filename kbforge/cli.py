"""kbforge command-line interface.

Commands:
- build     run the pipeline for a client config (Phases 1-5)
- query     search a saved knowledge base (Phase 5)
- package   create/refresh the package manifest and build metadata (Phase 5)
- serve     run the local FastAPI search server (Phase 5)
- evaluate  run retrieval evaluation (Phase 6)
- update    incremental re-crawl (Phase 7)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kbforge import __version__
from kbforge.config import ConfigError, load_config


def _run_build_pipeline(config) -> dict | None:
    """Run Phases 1-5 with the shared pipeline and return phase summaries."""
    from kbforge.package import mark_build_in_progress

    print(f"kbforge {__version__} | client: {config.client_name}")
    print(f"output dir: {config.output_dir}")
    mark_build_in_progress(config.output_dir)

    try:
        from kbforge.ingest.crawler import run_ingest

        ingest_summary = run_ingest(config)
    except Exception as exc:
        print(f"\ningestion error: {exc}", file=sys.stderr)
        return None

    print("\ningestion complete")
    print(f"  pages fetched : {ingest_summary['pages_fetched']}")
    print(f"  pages failed  : {ingest_summary['pages_failed']}")
    print(f"  files loaded  : {ingest_summary['files_loaded']}")
    print(f"  files failed  : {ingest_summary['files_failed']}")
    print(
        "  documents     : "
        f"{ingest_summary['documents_written']} -> "
        f"{Path(config.output_dir) / 'documents.jsonl'}"
    )
    print(
        "  link edges    : "
        f"{ingest_summary['edges']} -> "
        f"{Path(config.output_dir) / 'link_graph.json'}"
    )

    process_summary = None
    try:
        cleaning_enabled = getattr(getattr(config, "cleaning", None), "enabled", True)
        dedup_enabled = getattr(
            getattr(config, "deduplication", None), "enabled", True
        )
        if cleaning_enabled or dedup_enabled:
            from kbforge.process import run_process

            process_summary = run_process(config)
            print("\nphase 2 complete")
            print(f"  processed documents : {process_summary['processed_documents']}")
            print(f"  duplicate removals  : {process_summary['removals']}")
            print(f"  cleaning warnings   : {process_summary['cleaning_warnings']}")
            print(f"  cleaning exclusions : {process_summary['cleaning_exclusions']}")
            print(f"  cleaned output      : {process_summary['cleaned_path']}")
            print(f"  removal log         : {process_summary['removal_log_path']}")
        else:
            print("\nphase 2 skipped (disabled in config)")
    except Exception as exc:
        print(f"\nphase 2 error: {exc}", file=sys.stderr)
        return None

    try:
        from kbforge.process import run_chunk

        chunk_summary = run_chunk(config)
        print("\nphase 3 complete")
        print(
            "  chunks written : "
            f"{chunk_summary['chunks_written']} -> {chunk_summary['chunks_path']}"
        )
    except Exception as exc:
        print(f"\nphase 3 error: {exc}", file=sys.stderr)
        return None

    try:
        from kbforge.process import run_index

        index_summary = run_index(config)
        print("\nphase 4 complete")
        print(
            "  chunks indexed : "
            f"{index_summary['chunks_indexed']} -> {index_summary['index_dir']}"
        )
    except Exception as exc:
        print(f"\nphase 4 error: {exc}", file=sys.stderr)
        return None

    try:
        from kbforge.package import build_package

        package_summary = build_package(config)
        print("\nphase 5 complete")
        print(f"  package manifest : {package_summary['manifest_path']}")
        print(f"  ranking modes    : {', '.join(package_summary['ranking_modes'])}")
    except Exception as exc:
        print(f"\nphase 5 error: {exc}", file=sys.stderr)
        return None

    return {
        "ingestion": ingest_summary,
        "processing": process_summary,
        "chunking": chunk_summary,
        "index": index_summary,
        "package": package_summary,
    }


def _build_document_count(summary: dict) -> int:
    package_summary = summary["package"]
    try:
        stats_path = Path(package_summary["build_stats_path"])
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        stats = {}
    for key in ("documents_cleaned", "documents_raw"):
        value = stats.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return summary["ingestion"].get("documents_written", 0)


def _cmd_build_url(args: argparse.Namespace) -> int:
    from kbforge.package import (
        KnowledgeBase,
        export_package,
        mark_build_in_progress,
    )
    from kbforge.url_build import (
        UrlBuildError,
        collect_url_build_options,
        display_path,
        prepare_url_build,
    )

    try:
        collected = collect_url_build_options(
            url=args.url,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            include_subdomains=args.include_subdomains,
            chunking_strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            ranking_modes=args.ranking_modes,
            output=args.output,
            yes=args.yes,
        )
    except UrlBuildError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if collected is None:
        return 0
    options, paths = collected

    try:
        prepare_url_build(options, paths, overwrite=args.overwrite)
    except (OSError, UrlBuildError) as exc:
        print(f"output error: {exc}", file=sys.stderr)
        return 1

    try:
        config = load_config(paths.config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = _run_build_pipeline(config)
    if summary is None:
        return 1

    try:
        KnowledgeBase.from_package(paths.output_dir)
    except Exception as exc:
        mark_build_in_progress(paths.output_dir)
        print(f"package validation error: {exc}", file=sys.stderr)
        return 1

    try:
        archive_summary = export_package(
            paths.output_dir,
            paths.archive_path,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"export error: {exc}", file=sys.stderr)
        return 1

    package_summary = summary["package"]
    documents = _build_document_count(summary)
    ranking_modes = package_summary.get(
        "ranking_modes", list(config.index.ranking_modes)
    )
    print("\nBuild completed successfully.\n")
    print(f"Website: {options.url}")
    print(f"Documents: {documents}")
    print(f"Chunks: {package_summary['chunks']}")
    print(f"Chunking strategy: {config.chunking.strategy}")
    print(f"Ranking modes: {', '.join(ranking_modes)}")
    print(f"\nKnowledge base: {display_path(paths.output_dir)}")
    print(f"Client ZIP: {display_path(archive_summary['archive_path'])}")
    print("README: included in ZIP")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.url is not None:
        if args.config is not None:
            print("--url and --config cannot be used together", file=sys.stderr)
            return 2
        return _cmd_build_url(args)

    url_only = (
        args.max_pages,
        args.max_depth,
        args.include_subdomains,
        args.chunking_strategy,
        args.chunk_size,
        args.chunk_overlap,
        args.ranking_modes,
        args.output,
    )
    if any(value is not None for value in url_only) or args.yes or args.overwrite:
        print("URL build options require --url", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config or "config.yaml")
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = _run_build_pipeline(config)
    return 0 if summary is not None else 1


def _cmd_package(args: argparse.Namespace) -> int:
    from kbforge.package import build_package, export_package

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        summary = build_package(config)
    except Exception as exc:
        print(f"package error: {exc}", file=sys.stderr)
        return 1

    print(f"package manifest : {summary['manifest_path']}")
    print(f"build config     : {summary['build_config_path']}")
    print(f"build stats      : {summary['build_stats_path']}")
    print(f"chunks           : {summary['chunks']}")
    print(f"ranking modes    : {', '.join(summary['ranking_modes'])}")

    if args.archive:
        try:
            archive_summary = export_package(
                summary["package_dir"],
                args.archive,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"export error: {exc}", file=sys.stderr)
            return 1

        print(f"archive path     : {archive_summary['archive_path']}")
        print(f"archive size     : {archive_summary['archive_size']} bytes")
        print(f"archive files    : {archive_summary['file_count']}")

    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from kbforge.package import KnowledgeBase, SUPPORTED_RANKING_MODES

    if args.mode is not None and args.mode not in SUPPORTED_RANKING_MODES:
        print(
            f"invalid ranking mode: {args.mode!r} "
            f"(choose from {', '.join(SUPPORTED_RANKING_MODES)})",
            file=sys.stderr,
        )
        return 2
    if args.k <= 0:
        print("--k must be greater than 0", file=sys.stderr)
        return 2
    if not args.text or not args.text.strip():
        print("query must not be empty", file=sys.stderr)
        return 2

    if args.package:
        source = Path(args.package)
    else:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        source = Path(config.output_dir)

    try:
        kb = KnowledgeBase.load(source)
    except Exception as exc:
        print(f"error loading knowledge base: {exc}", file=sys.stderr)
        if not args.package:
            print("run 'kbforge build' first", file=sys.stderr)
        return 1

    if args.mode is None:
        mode = "bm25" if "bm25" in kb.supported_modes else kb.supported_modes[0]
    else:
        mode = args.mode
    if mode not in kb.supported_modes:
        print(
            f"ranking mode {mode!r} is not available in this package "
            f"(available: {', '.join(kb.supported_modes)})",
            file=sys.stderr,
        )
        return 2

    results = kb.search(args.text, mode=mode, limit=args.k)
    if not results:
        print("no results")
        return 0

    for result in results:
        print(
            f"{result.rank + 1}. [{result.score:.4f}] {result.source} "
            f"(doc {result.doc_id}, chunk {result.chunk_id})"
        )
        if result.title:
            print(f"   {result.title}")
        print(f"   {result.text[:100]}...")
        print()
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from kbforge.evaluation import (
        EvaluationError,
        evaluate,
        load_dataset,
        write_report,
    )
    from kbforge.package import KnowledgeBase, SUPPORTED_RANKING_MODES

    if args.k <= 0:
        print("--k must be greater than 0", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(args.queries, requested_unit=args.unit)
    except EvaluationError as exc:
        print(f"evaluation dataset error: {exc}", file=sys.stderr)
        return 2

    if args.package:
        source = Path(args.package)
    else:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        source = Path(config.output_dir)

    try:
        kb = KnowledgeBase.load(source)
    except Exception as exc:
        print(f"error loading knowledge base: {exc}", file=sys.stderr)
        if not args.package:
            print("run 'kbforge build' first", file=sys.stderr)
        return 1

    if args.mode == "both":
        modes = [
            mode for mode in SUPPORTED_RANKING_MODES if mode in kb.supported_modes
        ]
    else:
        modes = [args.mode]
        if args.mode not in kb.supported_modes:
            print(
                f"ranking mode {args.mode!r} is not available in this package "
                f"(available: {', '.join(kb.supported_modes)})",
                file=sys.stderr,
            )
            return 2
    try:
        report = evaluate(kb, dataset, k=args.k, modes=modes)
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 1

    print(
        f"evaluation: {dataset.name} | queries: {len(dataset.queries)} | "
        f"unit: {dataset.unit} | k: {args.k}"
    )
    synthetic_label = (
        "unspecified"
        if dataset.synthetic is None
        else ("yes" if dataset.synthetic else "no")
    )
    print(f"synthetic judgments: {synthetic_label}")
    if dataset.synthetic is True:
        print(
            "caveat: synthetic judgments are illustrative; results do not "
            "estimate real-world retrieval quality."
        )
    elif dataset.synthetic is None:
        print(
            "caveat: judgment provenance is unspecified; results describe "
            "only this dataset."
        )
    for mode in report["configuration"]["modes"]:
        metrics = report["aggregate"][mode]
        print(
            f"{mode:>5}: P@{args.k}={metrics['precision_at_k']:.4f} "
            f"R@{args.k}={metrics['recall_at_k']:.4f} "
            f"MRR={metrics['mrr']:.4f} "
            f"nDCG@{args.k}={metrics['ndcg_at_k']:.4f} "
            f"MAP={metrics['map']:.4f}"
        )

    comparison = report.get("comparison", {}).get("bm25_minus_tfidf")
    if comparison:
        print(
            "bm25 - tfidf: "
            f"P@{args.k}={comparison['precision_at_k']:+.4f} "
            f"R@{args.k}={comparison['recall_at_k']:+.4f} "
            f"MRR={comparison['mrr']:+.4f} "
            f"nDCG@{args.k}={comparison['ndcg_at_k']:+.4f} "
            f"MAP={comparison['map']:+.4f}"
        )
        print(
            "caveat: bm25 - tfidf is a descriptive paired difference on the "
            "same judgments, not a significance test."
        )

    diagnostics = report["diagnostics"]
    if diagnostics["queries_without_relevant_ids"]:
        print(
            "warning: no relevant judgments for: "
            + ", ".join(diagnostics["queries_without_relevant_ids"])
        )
    for mode in report["configuration"]["modes"]:
        without_results = diagnostics["queries_without_results"][mode]
        if without_results:
            print(
                f"warning: {mode} returned no results for: "
                + ", ".join(without_results)
            )

    if args.output:
        try:
            report_path = write_report(report, args.output)
        except OSError as exc:
            print(f"could not write report: {exc}", file=sys.stderr)
            return 1
        print(f"report: {report_path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the API server requires fastapi and uvicorn; "
            "install the project dependencies",
            file=sys.stderr,
        )
        return 1
    if not (0 < args.port < 65536):
        print("--port must be between 1 and 65535", file=sys.stderr)
        return 2

    from kbforge.api import create_app

    app = create_app(package_path=args.package, config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_not_implemented(phase: str) -> int:
    print(f"'{phase}' arrives in a later phase of the project", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kbforge",
        description="Build, package and query a knowledge base from client data",
    )
    parser.add_argument("--version", action="version", version=f"kbforge {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build",
        help="build a knowledge base from YAML or a website URL",
        description=(
            "Run the existing ingestion, cleaning, chunking, indexing and "
            "packaging pipeline. Use --config for the YAML workflow or --url "
            "for the interactive website workflow."
        ),
    )
    p_build.add_argument(
        "--config",
        default=None,
        help="path to client config YAML (default: config.yaml)",
    )
    p_build.add_argument(
        "--url",
        help="website URL to build from; prompts for omitted options",
    )
    p_build.add_argument(
        "--max-pages",
        type=int,
        help="maximum page-fetch attempts (URL workflow; default: 300)",
    )
    p_build.add_argument(
        "--max-depth",
        type=int,
        help="maximum crawl depth (URL workflow; default: 2)",
    )
    p_build.add_argument(
        "--include-subdomains",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow links to subdomains (URL workflow; default: no)",
    )
    p_build.add_argument(
        "--chunking-strategy",
        choices=("fixed", "sentence", "paragraph"),
        help="chunking strategy (URL workflow; default: fixed)",
    )
    p_build.add_argument(
        "--chunk-size",
        type=int,
        help="maximum words per chunk (URL workflow; default: 300)",
    )
    p_build.add_argument(
        "--chunk-overlap",
        type=int,
        help="fixed-size chunk overlap in words (URL workflow; default: 50)",
    )
    p_build.add_argument(
        "--ranking-modes",
        help="comma-separated ranking modes (URL workflow; default: bm25,tfidf)",
    )
    p_build.add_argument(
        "--output",
        help="output directory inside output/ (URL workflow; default: generated)",
    )
    p_build.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="non-interactive URL build; do not prompt for confirmation",
    )
    p_build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing URL-workflow output, generated config and ZIP",
    )
    p_build.set_defaults(func=_cmd_build)

    p_query = sub.add_parser(
        "query",
        help="search a saved knowledge base without rebuilding it",
        description=(
            "Search a previously built index. Use --package to point at a "
            "package/output directory or a bare index directory; otherwise "
            "the output_dir from --config is used."
        ),
    )
    p_query.add_argument("text", help="query text")
    p_query.add_argument(
        "--mode",
        default=None,
        choices=["bm25", "tfidf"],
        help="ranking mode (default: first packaged mode, preferring bm25)",
    )
    p_query.add_argument("--k", type=int, default=5, help="maximum results (default: 5)")
    p_query.add_argument(
        "--package",
        help="package/output directory or index directory (no rebuild)",
    )
    p_query.add_argument("--config", default="config.yaml", help="client config YAML")
    p_query.set_defaults(func=_cmd_query)

    p_package = sub.add_parser(
        "package",
        help="create/refresh the package manifest for an already-built knowledge base",
    )
    p_package.add_argument("--config", default="config.yaml", help="client config YAML")
    p_package.add_argument(
        "--archive",
        metavar="PATH",
        help="export the validated package as a ZIP archive at PATH",
    )
    p_package.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing archive created with --archive",
    )
    p_package.set_defaults(func=_cmd_package)

    p_serve = sub.add_parser(
        "serve",
        help="run the local FastAPI search API",
        description=(
            "Load an existing package/index and serve /health and /search. "
            "The server never crawls or rebuilds the knowledge base."
        ),
    )
    p_serve.add_argument(
        "--package",
        help="package/output directory or index directory (overrides --config)",
    )
    p_serve.add_argument("--config", default="config.yaml", help="client config YAML")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p_serve.set_defaults(func=_cmd_serve)

    p_eval = sub.add_parser(
        "evaluate",
        help="evaluate BM25/TF-IDF retrieval against relevance judgments",
        description=(
            "Load an existing package/index and a JSON or JSONL judgment "
            "dataset, then compute IR metrics without crawling or rebuilding."
        ),
    )
    p_eval.add_argument(
        "--package",
        help="package/output directory or index directory (no rebuild)",
    )
    p_eval.add_argument("--config", default="config.yaml", help="client config YAML")
    p_eval.add_argument(
        "--queries", required=True, help="JSON or JSONL relevance-judgment dataset"
    )
    p_eval.add_argument(
        "--k",
        type=int,
        default=5,
        help="metric cutoff for P@k, R@k and nDCG@k (default: 5)",
    )
    p_eval.add_argument(
        "--unit",
        choices=["chunk", "document"],
        default=None,
        help="evaluation unit (default: dataset unit, else chunk)",
    )
    p_eval.add_argument(
        "--mode",
        choices=["bm25", "tfidf", "both"],
        default="both",
        help="ranking mode to evaluate (default: both)",
    )
    p_eval.add_argument(
        "--output",
        help="write the complete per-query report to this JSON file",
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    p_update = sub.add_parser("update", help="incremental re-crawl (Phase 7)")
    p_update.add_argument("--config", default="config.yaml")
    p_update.set_defaults(func=lambda args: _cmd_not_implemented("update"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
