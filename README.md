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
kbforge build --config my-client.yaml
```

Phase 1 implements ingestion (crawler + file loaders). Phase 2 adds
content cleaning (trafilatura-based extraction for HTML, normalization)
and deduplication (exact hash-based and near-duplicate detection via
word shingles + MinHash/LSH). Later phases add chunking, indexing,
packaging, evaluation and incremental updates.

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
    -> (Phase 4)     inverted index + BM25 + tf-idf, dense index (FAISS),
                     hybrid fusion, PageRank over the link graph
    -> (Phase 5)     packaged deliverables + quality report + query CLI
                     + FastAPI server
    -> (Phase 6)     evaluation metrics and experiments
    -> (Phase 7)     incremental update (ETag / Last-Modified aware)
```

## Deliverables (final, after Phase 5)

Written to `output_dir`:

| file | contents |
|---|---|
| `documents.jsonl` | one document per line (ingestion output) |
| `documents_cleaned.jsonl` | cleaned and deduplicated documents (Phase 2) |
| `removal_log.json` | removal log with reasons and representatives (Phase 2) |
| `phase2_stats.json` | Phase 2 processing statistics (Phase 2) |
| `chunks.jsonl` | searchable chunks with metadata |
| `bm25 index files` | inverted index, postings, doc lengths |
| `faiss index` | dense vectors + embedding model name |
| `pagerank.json` | PageRank scores per URL |
| `link_graph.json` | nodes and directed hyperlink edges |
| `quality_report.json` / `.md` | build statistics and quality checks |

A client consumes these files with `sample_client.py` (Phase 5) or the
FastAPI server (`api/server.py`).

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
