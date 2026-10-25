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
import logging
import sys
from pathlib import Path

from kbforge import __version__
from kbforge.config import ConfigError, load_config


def _cmd_build(args: argparse.Namespace) -> int:
    from kbforge.ingest.crawler import run_ingest

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(f"kbforge {__version__} | client: {config.client_name}")
    print(f"output dir: {config.output_dir}")
    summary = run_ingest(config)

    print("\ningestion complete")
    print(f"  pages fetched : {summary['pages_fetched']}")
    print(f"  pages failed  : {summary['pages_failed']}")
    print(f"  files loaded  : {summary['files_loaded']}")
    print(f"  files failed  : {summary['files_failed']}")
    print(f"  documents     : {summary['documents_written']} -> {Path(config.output_dir) / 'documents.jsonl'}")
    print(f"  link edges    : {summary['edges']} -> {Path(config.output_dir) / 'link_graph.json'}")

    # Phase 2: cleaning and deduplication (optional, configurable)
    try:
        cleaning_enabled = getattr(getattr(config, 'cleaning', None), 'enabled', True)
        dedup_enabled = getattr(getattr(config, 'deduplication', None), 'enabled', True)
        if cleaning_enabled or dedup_enabled:
            from kbforge.process import run_process
            proc_summary = run_process(config)
            print("\nphase 2 complete")
            print(f"  processed documents : {proc_summary['processed_documents']}")
            print(f"  removals            : {proc_summary['removals']}")
            print(f"  cleaned output      : {proc_summary['cleaned_path']}")
            print(f"  removal log         : {proc_summary['removal_log_path']}")
        else:
            print("\nphase 2 skipped (disabled in config)")
    except Exception as exc:
        print(f"\nphase 2 error: {exc}", file=sys.stderr)
        return 0

    # Phase 3: chunking
    try:
        from kbforge.process import run_chunk
        chunk_summary = run_chunk(config)
        print("\nphase 3 complete")
        print(f"  chunks written : {chunk_summary['chunks_written']} -> {chunk_summary['chunks_path']}")
    except Exception as exc:
        print(f"\nphase 3 error: {exc}", file=sys.stderr)
        return 0

    # Phase 4: indexing
    try:
        from kbforge.process import run_index
        idx_summary = run_index(config)
        print("\nphase 4 complete")
        print(f"  chunks indexed : {idx_summary['chunks_indexed']} -> {idx_summary['index_dir']}")
    except Exception as exc:
        print(f"\nphase 4 error: {exc}", file=sys.stderr)
        return 0

    # Phase 5: packaging
    try:
        from kbforge.package import build_package
        package_summary = build_package(config)
        print("\nphase 5 complete")
        print(f"  package manifest : {package_summary['manifest_path']}")
        print(f"  ranking modes    : {', '.join(package_summary['ranking_modes'])}")
    except Exception as exc:
        print(f"\nphase 5 error: {exc}", file=sys.stderr)
        return 0

    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    from kbforge.package import build_package

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
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from kbforge.package import KnowledgeBase, SUPPORTED_RANKING_MODES

    if args.mode not in SUPPORTED_RANKING_MODES:
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

    results = kb.search(args.text, mode=args.mode, limit=args.k)
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

    p_build = sub.add_parser("build", help="run the pipeline and package the result")
    p_build.add_argument("--config", default="config.yaml", help="path to client config YAML")
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
        "--mode", default="bm25", choices=["bm25", "tfidf"], help="ranking mode"
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

    p_eval = sub.add_parser("evaluate", help="evaluate retrieval quality (Phase 6)")
    p_eval.set_defaults(func=lambda args: _cmd_not_implemented("evaluate"))

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
