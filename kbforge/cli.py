"""kbforge command-line interface.

Commands:
- build     run the pipeline for a client config (Phase 1: ingestion)
- query     search a built knowledge base (Phase 5)
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
        # Don't fail the build if Phase 2 fails
        return 0

    print("\nlater phases (chunk, index, package) are not wired yet")
    return 0



def _cmd_not_implemented(phase: str) -> int:
    print(f"'{phase}' arrives in a later phase of the project", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbforge", description="Build a knowledge base from client data")
    parser.add_argument("--version", action="version", version=f"kbforge {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="run the pipeline")
    p_build.add_argument("--config", default="config.yaml", help="path to client config YAML")
    p_build.set_defaults(func=_cmd_build)

    p_query = sub.add_parser("query", help="search the knowledge base (Phase 5)")
    p_query.add_argument("text")
    p_query.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    p_query.add_argument("--k", type=int, default=5)
    p_query.set_defaults(func=lambda args: _cmd_not_implemented("query"))

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
