# kbforge

kbforge turns a client's data (URLs, domains, uploaded files) into a
ready-to-use knowledge base for search and RAG. The client receives the
knowledge base as plain files and queries it on their own side; no
hosted service or paid API is involved.

Built for an academic Information Retrieval course project: the inverted
index, BM25 and tf-idf are implemented from scratch so they can be
explained and inspected. Crawls record a link graph, but no PageRank or
other link-based ranking is implemented.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `kbforge` console script. Running
`python -m kbforge` works as well.

## Usage

```bash
cp config.yaml my-client.yaml   # edit sources, limits, output dir
kbforge build --config my-client.yaml       # ingest, clean, chunk, index, package
kbforge query "inverted index" --config my-client.yaml --mode bm25 --k 5
kbforge package --config my-client.yaml     # refresh package metadata only
kbforge serve --package ./output/demo       # local FastAPI search API
kbforge evaluate --package ./output/demo \
  --queries evaluation/synthetic_document_judgments.json --k 5 --output report.json
```

Build directly from a website with the URL workflow:

```bash
# Interactive: prompts for crawl, chunking and output options.
kbforge build --url https://docs.example.com/guide

# Non-interactive: use defaults for omitted options.
kbforge build --url https://docs.example.com/guide --yes

# Example with explicit options and overwrite permission.
kbforge build --url https://docs.example.com/guide --yes --overwrite \
  --max-pages 200 --max-depth 2 --no-include-subdomains \
  --chunking-strategy fixed --chunk-size 300 --ranking-modes bm25,tfidf
```

The URL workflow uses the same ingestion-to-packaging pipeline as the YAML
workflow. It writes the knowledge base to `output/<hostname-derived-name>/`, a generated
configuration to `output/.kbforge-configs/`, and a client ZIP beside the
output directory (for example `output/docs-example-com.zip`). The ZIP includes
a generated `README.md`. Existing outputs, ZIPs and generated configs are only
replaced with `--overwrite`.

Phase 1 implements ingestion (crawler + file loaders). Phase 2 adds
content cleaning (trafilatura-based extraction for HTML, normalization)
and deduplication (exact hash-based and near-duplicate detection via
word shingles + MinHash/LSH). Cleaning warnings describe documents that
remain in the corpus; only unrecoverable cleaning errors are exclusions,
and only duplicate removals remove documents. Phase 3 adds chunking,
Phase 4 the inverted index with BM25/TF-IDF ranking, Phase 5 packaging,
saved-index querying and the local search API, and Phase 6 offline
retrieval evaluation.

## Configuration

See the commented `config.yaml`. The important knobs:

- `sources.urls` / `sources.allowed_domains`: seed pages and crawl scope
- `sources.files_dir`: directory of uploaded PDF/DOCX/HTML/TXT files
- `crawl.*`: max_depth, max_pages, delay_seconds, timeout, retries, UA,
  and `include_subdomains` (default `true` for YAML configs)
- `chunking.*`: strategy (fixed/sentence/paragraph), size in words, overlap
- `index.*`: lexical index settings and `ranking_modes` (subset of
  `bm25`/`tfidf`); dense/embedding retrieval is not implemented (`dense: true`
  is rejected)
- `output_dir`: where the deliverables are written

## Architecture

```
URLs, files
    -> ingest/       polite BFS crawler (robots.txt, delay, limits) that
                     records internal/external link edges; PDF/DOCX/HTML/TXT
                     loaders
    -> clean/        trafilatura extraction for HTML, Unicode/whitespace
                     normalization (Phase 2)
    -> dedup/        exact duplicates (normalized hashes) and near-duplicates
                     (word shingles + MinHash/LSH) (Phase 2)
    -> (Phase 3)     chunking (fixed / sentence / paragraph)
    -> (Phase 4)     inverted index + BM25 + tf-idf
    -> (Phase 5)     package manifest + saved-index query CLI + FastAPI server
    -> (Phase 6)     evaluation metrics and experiments
    -> (Phase 7)     incremental update (ETag / Last-Modified aware)
```

## Text processing

Indexing and queries use the same deterministic text processor: NFKC
Unicode normalization, lowercasing, punctuation removal and whitespace
tokenization. Stop-word removal and stemming are available but disabled
by default. Accent folding is not performed; NFKC normalization only
composes/decomposes Unicode, so accents are preserved.

## Package layout (Phase 5)

The built `output_dir` is the knowledge-base package. Packaging adds
metadata files and references the existing artifacts; it does not copy
large files or modify ingestion/cleaning/chunking/index outputs.

```
output_dir/
  manifest.json            package format/version, ranking modes, artifact paths
  build_config.json        safe config summary (no seed URLs or local paths)
  build_stats.json         ingestion/processing/chunk/index statistics
  documents.jsonl          raw ingestion output (optional for querying)
  documents_cleaned.jsonl  cleaned and deduplicated documents
  link_graph.json          crawl link graph (nodes/edges; not used for ranking)
  crawl_report.json        crawl counts, failures and timings
  removal_log.json         Phase 2 duplicate removals, cleaning warnings/exclusions
  phase2_stats.json        Phase 2 statistics
  chunks.jsonl             chunk text + metadata used to reconstruct results
  index/
    postings.json          term -> {chunk_id: term frequency}
    doc_lengths.json       chunk token lengths
    chunk_info.json        chunk metadata (source, title, doc_id, ...)
    df.json                document frequencies
    index_stats.json       N and average document length
```

`manifest.json` uses `format: "kbforge-knowledge-base"` and
`format_version: "1.0"`. Loaders reject newer/unknown format versions.
Artifact paths are relative to the manifest. The supported ranking modes
are recorded as `bm25` and `tfidf`.

`kbforge query` and the API load only these files. They never crawl,
clean, chunk or rebuild an index. If a required artifact is missing or
corrupt, the command/API reports a clear error instead of rebuilding.
Package loading also checks that `chunks.jsonl` chunk IDs agree with
`index/chunk_info.json` and `index/doc_lengths.json`, and that their
counts agree with `index_stats.json` and the manifest statistics.

## Build failure safety

`kbforge build` writes `.kbforge-build-incomplete` in `output_dir` before
the pipeline starts and removes it only after packaging succeeds. Package
loaders reject a package while that marker is present, so a failed rebuild
cannot be served as a current package. Previous artifacts may remain on
disk, but there is no rollback or transactional rewrite; rerun
`kbforge build`, or `kbforge package` after fixing artifacts, to clear the
marker and refresh the metadata.

## Search API

```bash
kbforge serve --package ./output/demo --host 127.0.0.1 --port 8000
# equivalent:
KBFORGE_PACKAGE=./output/demo uvicorn kbforge.api:app --port 8000
```

Routes:

- `GET /health` — reports `ok` when a package is loaded, otherwise HTTP 503.
- `POST /search` — JSON body `{"query": "...", "mode": "bm25|tfidf", "limit": 5}`.

Example:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "inverted index", "mode": "bm25", "limit": 3}'
```

Results include `score`, `chunk_id`, `doc_id`, `source`, `title`, `text`
and chunk `metadata`. Limits must be between 1 and 100. The API is local,
read-only, and has no authentication or external services.

## Development

```bash
pytest
```

Tests use small fixtures and mocked HTTP; they never touch the network.

## Politeness

The crawler respects robots.txt, sends a descriptive User-Agent, waits
`crawl.delay_seconds` between page requests, retries with exponential
backoff, and enforces `crawl.max_pages`. `max_pages` counts page-fetch
attempts, including failed responses and responses skipped by content
type; robots.txt requests are not counted. Configure it to crawl only
what you are allowed to crawl.


## Phase 3: Chunking

Chunking is supported with three strategies (configurable in `chunking`):
- `fixed`: word-based chunks with configurable `size` and `overlap`
- `sentence`: sentence-grouped chunks up to target size (words)
- `paragraph`: paragraph/heading-aware chunking

Chunks are written to `chunks.jsonl` with stable chunk IDs and metadata for traceability back to source documents.

## Phase 5: Packaging and saved-index querying

`kbforge build` writes the package metadata automatically after indexing.
To refresh metadata for an existing build without re-running the pipeline:

```bash
kbforge package --config my-client.yaml
```

Packaging writes metadata into `output_dir` and leaves the artifact files in
place; it does not copy them into a separate distribution directory.

Query a package, output directory or bare index directory without a config
or network access:

```bash
kbforge query "inverted index" --package ./output/demo --mode bm25 --k 5
kbforge query "tf idf" --package ./output/demo/index --mode tfidf
```

Invalid modes, non-positive `--k`, empty queries, missing packages and
corrupt artifacts produce a non-zero exit with a clear message. A valid
query with no matches prints `no results` and exits successfully.

## Phase 6: Retrieval evaluation

`kbforge evaluate` compares the existing BM25 and TF-IDF rankers against
offline relevance judgments. It loads an existing package, output directory
or bare index directory and never crawls, cleans, chunks or rebuilds.

```bash
# Evaluate both rankers with document-level judgments
kbforge evaluate \
  --package ./output/manual-test \
  --queries evaluation/synthetic_document_judgments.json \
  --k 5 --unit document --mode both --output report.json

# Evaluate one ranker against a chunk-level JSONL dataset
kbforge evaluate --package ./output/demo --queries judgments.jsonl \
  --k 10 --mode bm25
```

The bundled `evaluation/synthetic_document_judgments.json` targets the
`output/manual-test` package. Its labels are **synthetic examples**, not
authoritative relevance assessments; replace them with documented human
judgments for a real experiment.

### Judgment format

JSON datasets can declare metadata and a `queries` list:

```json
{
  "format": "kbforge-evaluation",
  "format_version": "1.0",
  "name": "example judgments",
  "description": "Optional provenance note",
  "synthetic": true,
  "unit": "document",
  "queries": [
    {"query_id": "q1", "query": "download python", "relevant": ["doc-1", "doc-2"]}
  ]
}
```

JSONL files contain one query object per non-empty line:

```json
{"query_id": "q1", "query": "download python", "relevant": ["doc-1"]}
{"query_id": "q2", "query": "search api", "relevant": []}
```

`query_id`/`id`, `query`/`text`, and `relevant`/`relevant_ids` are accepted
aliases. IDs are deduplicated. JSONL defaults to `unit: "chunk"`; JSON can
set `unit` to `chunk` or `document`. A disagreement between the dataset and
`--unit`, duplicate query IDs, malformed judgments, and IDs absent from the
loaded index are reported as errors.

### Metrics

Judgments are binary (relevant or not). Duplicate retrieved IDs are ignored
after their first occurrence. Empty result lists and queries without relevant
IDs score `0.0` rather than NaN.

- **P@k** = relevant results in the top `k` / `k`
- **R@k** = relevant results in the top `k` / total judged relevant
- **RR** = `1 / rank` of the first relevant result
- **MRR** = mean RR over queries
- **nDCG@k** = DCG@k / ideal DCG@k, using `gain / log2(rank + 1)`
- **AP** = mean of precision at each relevant hit, divided by judged relevant
- **MAP** = mean AP over queries

P@k, R@k and nDCG use cutoff `k`; AP/MAP and MRR use the full ranking. For
document-level evaluation, ranked chunks are collapsed to unique document IDs,
keeping the first occurrence. This avoids rewarding a document multiple times
for returning many chunks.

The JSON report contains format/version metadata, dataset provenance,
configuration, aggregate metrics per mode, per-query metrics and top-`k`
retrieved IDs, plus a `bm25_minus_tfidf` comparison when both modes run.
No query is silently skipped: queries with no relevant IDs or no retrieved
results still score `0.0`, and the report's `diagnostics` section lists them
by query ID (per mode for empty result lists).

The CLI always prints whether judgments are synthetic as `yes`, `no` or
`unspecified`; an undeclared `synthetic` field is not treated as human
provenance. Report notes and CLI caveats state that metrics describe only the
supplied judgments. Synthetic results are illustrative, not estimates of
real-world retrieval quality, and `bm25_minus_tfidf` is a descriptive paired
difference rather than a significance test. Comparing different
chunking/cleaning configurations requires separate packages built with the
same document IDs; automatic multi-package comparison is intentionally
deferred beyond Phase 6.
