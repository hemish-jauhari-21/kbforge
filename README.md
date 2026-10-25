# kbforge

kbforge turns a client's data (URLs, domains, uploaded files) into a
ready-to-use knowledge base for search and RAG. The client receives the
knowledge base as plain files and queries it on their own side; no
hosted service or paid API is involved.

Built for an academic Information Retrieval course project: the inverted
index, BM25, tf-idf and PageRank are implemented from scratch so they
can be explained and inspected.

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
```

Phase 1 implements ingestion (crawler + file loaders). Phase 2 adds
content cleaning (trafilatura-based extraction for HTML, normalization)
and deduplication (exact hash-based and near-duplicate detection via
word shingles + MinHash/LSH). Phase 3 adds chunking, Phase 4 the
inverted index with BM25/TF-IDF ranking, and Phase 5 packaging, saved-index
querying and the local search API.

## Configuration

See the commented `config.yaml`. The important knobs:

- `sources.urls` / `sources.allowed_domains`: seed pages and crawl scope
- `sources.files_dir`: directory of uploaded PDF/DOCX/HTML/TXT files
- `crawl.*`: max_depth, max_pages, delay_seconds, timeout, retries, UA
- `chunking.*`: strategy (fixed/sentence/paragraph), size, overlap
- `index.*`: which indexes to build and the embedding model
- `output_dir`: where the deliverables are written

## Architecture

```
URLs, files
    -> ingest/       polite BFS crawler (robots.txt, delay, limits);
                     PDF/DOCX/HTML/TXT loaders
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
  removal_log.json         Phase 2 removal log
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
