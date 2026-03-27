# NomadLATAM RAG Agent

A production-grade Retrieval-Augmented Generation system for digital nomads in Latin America. Users ask natural language questions about visa requirements, cost of living, coworking spaces, and city recommendations — and receive grounded, cited answers drawn from a curated knowledge base.

Built as a portfolio project demonstrating production RAG patterns: hybrid retrieval, LangGraph orchestration, evaluation-driven iteration, and a deployable FastAPI service layer.

---

## Architecture

```
data/ (markdown)
    └── ingestion pipeline
            ├── markdown_headers chunking
            ├── OpenAI embeddings (text-embedding-3-small)
            └── pgvector + tsvector (PostgreSQL)
                    └── LangGraph agent
                            ├── query_analysis  →  parsed intents + filters
                            ├── retrieve        →  hybrid search (semantic + keyword)
                            │   └── multi_retrieve (comparison queries)
                            ├── rerank          →  LLM relevance scoring
                            ├── generate        →  grounded answer with citations
                            └── fallback        →  empty retrieval path
                                    └── FastAPI service layer
                                            ├── POST /chat
                                            ├── POST /documents
                                            └── GET /health
```

---

## Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| Orchestration | LangGraph | Graph-based architecture; explicit routing between retrieval paths |
| LLM | Claude (Anthropic) | Generation (Sonnet), reranking and query analysis (Haiku) |
| Embeddings | OpenAI `text-embedding-3-small` | 1536 dims — fits pgvector HNSW limit; sufficient quality for this domain |
| Vector store | pgvector (PostgreSQL) | Persistent, metadata filtering via SQL, already in stack |
| Keyword search | PostgreSQL `tsvector` | Hybrid retrieval in one system; no separate BM25 index to maintain |
| API | FastAPI | Same service layer pattern as financial agent |
| Deployment | Docker Compose | One-command local startup |
| Evaluation | RAGAS | Standard RAG evaluation framework; Haiku as judge to control cost |

---

## Document Types

| Type | Description | Count |
|---|---|---|
| `city_guide` | Long-form nomad guides — cost, internet, neighborhoods, visa, safety, climate | 4 |
| `visa_info` | Visa pathways and requirements — tables and structured bullet lists | 3 |
| `coworking_review` | Space reviews — price, internet speed, amenities, vibe | 2 |
| `cost_comparison` | Multi-city cost breakdowns — rent, groceries, transport, coworking | 2 |

All documents are LLM-generated synthetic data, designed to cover realistic query patterns without manual curation. Generation script: `scripts/generate_data.py`.

---

## Chunking Strategy

### The normalization insight

The pipeline treats markdown as the canonical document format. Raw documents (PDF, DOCX, etc.) would be normalized to markdown during a pre-processing step before entering the pipeline. This keeps the chunking layer simple and uniform — complexity belongs at the normalization boundary, not inside the chunker.

### Strategy comparison

Three strategies were evaluated against all document types using `scripts/compare_chunks.py`:

- **Recursive character splitting** — baseline, ignores document structure
- **Markdown header splitting** — respects `##` section boundaries
- **Semantic chunking** — embedding-based topic boundary detection

Batch comparison results across all city guides:

| File | recursive (chunks/avg) | markdown_headers (chunks/avg) | semantic (chunks/avg) |
|---|---|---|---|
| buenos_aires_argentina | 28 / 336 | 8 / 1143 | 5 / 1859 |
| florianopolis_brazil | 27 / 338 | 8 / 1101 | 5 / 1786 |
| medellin_colombia | 24 / 356 | 8 / 1060 | 4 / 2151 |
| mexico_city_mexico | 26 / 353 | 9 / 990 | 4 / 2270 |

### Decision

`markdown_headers` wins across all document types:

| Document type | Strategy | Typical chunks | Avg size |
|---|---|---|---|
| `city_guide` | `markdown_headers` | 8–9 | ~1000–1140 chars |
| `cost_comparison` | `markdown_headers` | 7–8 | ~678–826 chars |
| `coworking_review` | `markdown_headers` | 4 | ~500–1300 chars |
| `visa_info` | `markdown_headers` | 3–5 | ~868–1442 chars |

**Why markdown_headers won:**
- Consistent chunk count across all files of the same type — evidence that the strategy tracks document structure rather than arbitrary character boundaries
- Each chunk maps to one semantic unit (one city guide section, one visa pathway, one coworking space)
- Zero API cost — no embeddings needed at ingestion time, unlike semantic chunking

**Why semantic chunking lost:** Documents already have explicit structure encoded in headers. Semantic chunking pays embedding cost to rediscover structure that's already in the markup — solving a problem that doesn't exist in well-normalized data.

The visa documents initially failed with `markdown_headers` (1 chunk = entire document) because the generation prompt produced bold pseudo-headers instead of `##` markdown. Fixed by enforcing `##` headers in `scripts/generate_data.py` — confirming the principle: normalize at the data layer, not the chunking layer.

---

## Retrieval Design

### Hybrid search

Two retrieval paths run in parallel and merge via Reciprocal Rank Fusion (RRF):

**Semantic search** — cosine similarity over `text-embedding-3-small` embeddings stored in pgvector with an HNSW index. Finds conceptually related chunks even when exact terms don't match.

**Keyword search** — PostgreSQL full-text search using `tsvector`. Finds exact term matches for proper nouns, acronyms, and specific entities that semantic search may rank poorly.

**RRF merging** — combines ranked lists by position, not score. The formula `1 / (rank + 60)` rewards chunks that appear highly in both lists without requiring score normalization across different algorithms. A chunk ranked #2 in both lists outscores a chunk ranked #1 in only one.

### Why hybrid matters

A query like `"FMM tourist permit 180 days Mexico"` contains exact terms that appear verbatim in the visa document. Semantic search finds it, but keyword search finds it more precisely. A query like `"cheapest city for rent"` has no exact matches but semantic search finds cost comparison chunks reliably.

### tsvector configuration

The keyword search uses a custom `english_unaccent` text search configuration:

```sql
CREATE TEXT SEARCH CONFIGURATION english_unaccent (COPY = pg_catalog.english);
ALTER TEXT SEARCH CONFIGURATION english_unaccent
    ALTER MAPPING FOR hword, hword_part, word
    WITH unaccent, english_stem;
```

This pipeline strips diacritics (`Medellín` → `medellin`) then applies English stemming (`coworking` → `cowork`). Both the index (via trigger) and queries use the same configuration, ensuring symmetric normalization. Using `english` alone broke on accented city names; `simple` handled accents but broke stemming; `english_unaccent` solves both.

### Metadata filtering

Every chunk carries structured metadata — `document_type`, `country`, `city`, `section` — stored as regular PostgreSQL columns. Filters are applied as SQL `WHERE` clauses before vector search, scoping retrieval to the relevant subset before any similarity computation.

---

## LangGraph Agent

### Graph structure

```
START
  └── query_analysis
          ├── [single intent]  → retrieve → [empty] → fallback → END
          │                              → [results] → rerank → generate → END
          └── [multi intent]  → multi_retrieve → [empty] → fallback → END
                                               → [results] → rerank → generate → END
```

### Nodes

| Node | Model | Purpose |
|---|---|---|
| `query_analysis` | Haiku | Decompose query into intents; extract entity filters |
| `retrieve` | — | `hybrid_search` with intent[0] query and filters |
| `multi_retrieve` | — | One `hybrid_search` per intent; merge via RRF |
| `rerank` | Haiku | Score each of k=10 chunks 1–10 for relevance; keep top 4 |
| `generate` | Sonnet | Answer from top-4 chunks with inline `[1][2]` citations |
| `fallback` | — | Canned response when retrieval returns empty |

### Two-layer "I don't know" handling

**Graph-level fallback** — triggers when retrieval returns zero chunks. No LLM call. Fires for queries about entities completely outside the knowledge base (Bogotá, Tokyo, Portugal).

**LLM-level refusal** — triggers inside `generate` when retrieved chunks exist but don't answer the question. The system prompt instructs Claude to acknowledge uncertainty rather than hallucinate. Fires for questions about specific facts not in the chunks ("What's the name of the startup event at...").

### Query analysis and multi-retrieve

The `query_analysis` node decomposes comparison queries into per-entity intents:

```
"compare cost of living between Medellín and Buenos Aires"
    → [
        {"query": "cost of living", "filters": {"city": "Medellín"}},
        {"query": "cost of living", "filters": {"city": "Buenos Aires"}}
      ]
```

Each intent runs an independent `hybrid_search`, results are merged via RRF, and the reranker sees chunks from both cities. The generate node receives balanced context for a genuine comparison response.

### Model tier decisions

Haiku handles reranking and query analysis — both are structured extraction tasks (score chunks, decompose queries) that don't require Sonnet's reasoning depth. Sonnet handles generation where response quality directly affects the user experience. This cuts per-query cost significantly with no measurable quality loss on the rerank and analysis tasks.

---

## Evaluation

### Test set

25 question/expected-answer pairs generated by Claude Sonnet from the actual document content, covering five categories:

| Category | Count | Description |
|---|---|---|
| Factual | 6 | Specific facts with verifiable answers |
| Comparison | 6 | Multi-city or multi-country comparisons |
| Aggregation | 5 | Synthesis across multiple documents |
| Out of scope | 4 | Questions outside the knowledge base |
| Ambiguous | 4 | Vague queries requiring interpretation |

### M3 vs M4 baseline comparison

| Metric | M3 | M4 | Delta |
|---|---|---|---|
| Faithfulness | 0.604 | 0.658 | +8.9% |
| Answer relevancy | 0.312 | 0.495 | +58.7% |

Both runs: 25 test cases, baseline metrics (faithfulness + answer relevancy), Haiku as RAGAS judge.

**Key findings:**

1. **Answer relevancy improved 59%** — directly attributable to M4 query analysis. Decomposing comparison queries into per-entity intents produces more targeted, relevant responses that RAGAS can score reliably.

2. **Out-of-scope handling improved** — M3 incorrectly attempted to answer the Portugal visa question (faithfulness=0.5, hallucinated content). M4 correctly triggered fallback. Query analysis recognizes when a query is outside the knowledge base geography.

3. **Faithfulness is strong for in-scope queries** — excluding fallbacks and ambiguous cases, factual and comparison queries score ~0.85 faithfulness. The grounding instructions in the generate system prompt are working.

**Known limitation:** The coworking affordability query consistently scores low faithfulness (M3: 0.17, M4: 0.43) because query analysis doesn't reliably apply `document_type=coworking_review` for coworking queries, pulling city guide chunks instead of coworking review chunks. Query analysis entity vocabulary is currently hardcoded — a dynamic vocabulary loaded from the database at startup would fix this.

### Running evaluations

```bash
# Quick check — 5 samples, baseline metrics
python scripts/scorecard.py --samples 5 --metrics baseline --tag dev

# Full baseline run
python scripts/scorecard.py --metrics baseline --tag m4

# Re-score existing results with different metrics (no API calls for inference)
python scripts/scorecard.py --skip-inference --metrics full --tag m4_full
```

Results are saved as timestamped JSON snapshots in `data/evaluation/results/`.

---

## Cost Profile

| Operation | Model | Estimated cost |
|---|---|---|
| Single query (full chain) | Haiku + Sonnet | ~$0.02–0.04 |
| Full test run (25 queries) | Haiku + Sonnet | ~$0.50–1.00 |
| RAGAS evaluation (25 cases, baseline) | Haiku judge | ~$0.05–0.10 |
| Data generation (one-time) | Sonnet | ~$0.20 |

The reranker is the highest per-query cost driver — sending k=10 chunks to an LLM for scoring. Haiku reduces this to negligible cost vs using Sonnet for reranking.

---

## Known Limitations & Backlog

**Parent-child chunking** — not implemented. Small chunks for precise retrieval, parent context passed to generation. Most valuable for documents longer than ~3000 chars where `markdown_headers` produces overly large chunks. Current chunk sizes (800–1400 chars) don't justify the schema complexity yet.

**Retrieval Precision@k / Recall@k** — not implemented. Requires a ground-truth chunk mapping per test case (which chunk IDs should be retrieved). Deferred as future work.

**Keyword AND brittleness** — `websearch_to_tsquery` requires all query terms to appear in the chunk. Multi-word natural language queries sometimes return empty results when one term is missing from the tsvector. M4 query analysis partially mitigates this by rewriting queries to shorter, more targeted terms.

**City filter on coworking chunks** — coworking review chunks have `city=None` in metadata because the city appears in the section heading, not in the document-level metadata. City-level filtering on coworking queries silently returns empty until metadata is enriched at ingestion time.

**Single session** — `app.state.session` is a single SQLAlchemy session shared across all requests. Not production-safe under concurrent load. Needs a `sessionmaker` pool with per-request sessions for production deployment.

**Hardcoded entity vocabulary in query analysis** — the `QUERY_ANALYSIS_SYSTEM_PROMPT` lists valid city, country, and document_type values explicitly. Adding a new city requires updating the prompt. Should be replaced with dynamic vocabulary loaded from the database at graph build time.

---

## How to Run

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- OpenAI API key
- Anthropic API key

### Setup

```bash
git clone <repo>
cd nomad-rag

# Create virtualenv
python -m venv venv
source venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, ANTHROPIC_API_KEY
```

### Start the database

```bash
docker compose up -d postgres
```

### Run migrations

```bash
alembic upgrade head
```

### Ingest documents

```bash
python scripts/ingest.py
```

### Start the API

```bash
# Development
uvicorn src.api.main:app --reload

# Or via Docker Compose (full stack)
docker compose up -d
```

### Example queries

```bash
# Health check
curl http://localhost:8000/health

# Simple factual query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maximum tourist stay in Colombia?"}'

# Query with metadata filter
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "coworking spaces with fast internet", "filters": {"document_type": "coworking_review"}}'

# Comparison query (triggers multi-retrieve)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "compare cost of living between Medellín and Buenos Aires"}'

# Upload a new document
curl -X POST http://localhost:8000/documents \
  -F "file=@data/city_guides/medellin_colombia.md"
```

---

## Project Structure

```
nomad-rag/
├── data/
│   ├── city_guides/            # City guide markdown documents
│   ├── visa_info/              # Visa information documents
│   ├── coworking/              # Coworking space reviews
│   ├── cost_comparison/        # Multi-city cost comparisons
│   └── evaluation/
│       ├── test_set.json       # 25 evaluation test cases
│       └── results/            # Timestamped RAGAS evaluation snapshots
├── src/
│   ├── database.py             # SQLAlchemy engine and session management
│   ├── config/
│   │   └── settings.py         # Environment-backed configuration defaults
│   ├── agent/                  # LangGraph RAG agent
│   │   ├── config.py           # AgentConfig (models, k values, temperature)
│   │   ├── state.py            # GraphState TypedDict + QueryIntent
│   │   ├── prompts.py          # System prompts and chunk formatters
│   │   ├── nodes.py            # Node factory functions
│   │   └── graph.py            # build_agent() — compiled LangGraph
│   ├── ingestion/              # Document → chunk → database pipeline
│   │   ├── models.py           # ChunkMetadata (Pydantic) + ChunkRecord (SQLAlchemy)
│   │   ├── loader.py           # Markdown loader with metadata extraction
│   │   ├── chunking.py         # Three strategies behind common interface
│   │   └── vector_store.py     # upsert_chunks, similarity_search, delete_by_source
│   ├── retrieval/              # Hybrid search
│   │   ├── hybrid.py           # hybrid_search + RRF merging
│   │   └── keyword_search.py   # PostgreSQL tsvector full-text search
│   ├── services/               # Transport-agnostic business logic
│   │   ├── chat.py             # chat_service — wraps graph invocation
│   │   └── documents.py        # ingest_document — wraps ingestion pipeline
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # App factory + lifespan (shared graph, session)
│   │   └── routes/
│   │       ├── chat.py         # POST /chat
│   │       └── documents.py    # POST /documents
│   └── evaluation/             # RAGAS evaluation framework
│       ├── metrics.py          # evaluate_pipeline(), RawResult, EvaluationReport
│       └── test_set.py         # TestCase model, load_test_set()
├── scripts/
│   ├── generate_data.py        # One-time: generate synthetic markdown corpus via Claude
│   ├── generate_test_set.py    # One-time: generate 25 evaluation test cases via Claude
│   ├── ingest.py               # Chunk, embed, and upsert all documents into pgvector
│   ├── compare_chunks.py       # Compare chunking strategies on a file or document type batch
│   ├── test_retrieval.py       # Smoke-test semantic, keyword, and hybrid search directly
│   ├── test_graph.py           # End-to-end smoke-test of the full LangGraph agent
│   └── scorecard.py            # RAGAS evaluation runner with two-phase execution
├── alembic/
│   └── versions/
│       ├── dc4045f226bf_create_chunks_table.py          # Initial schema + HNSW + tsvector trigger
│       ├── a1b2c3d4e5f6_update_tsvector_simple.py       # Switch trigger to simple config
│       └── 109d08d5b6a0_english_unaccent_config.py      # Custom english_unaccent ts config
├── data/
│   ├── city_guides/            # 4 city guide markdown files
│   ├── visa_info/              # 3 visa information markdown files
│   ├── coworking/              # 2 coworking review markdown files
│   ├── cost_comparison/        # 2 cost comparison markdown files
│   └── evaluation/
│       ├── test_set.json       # 25 committed test cases (generated, not hand-written)
│       ├── raw_results.json    # Latest graph inference outputs (gitignored)
│       └── results/            # Timestamped RAGAS evaluation snapshots
├── docker-compose.yml          # PostgreSQL + pgvector + API services
├── Dockerfile                  # Python 3.11-slim; installs pyproject.toml deps
├── pyproject.toml              # Project metadata and all dependencies
└── .env.example                # Required env vars: DATABASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY
```

---

## Scripts Reference

| Script | When to run | Notes |
|---|---|---|
| `generate_data.py` | Once, before first ingest | Regenerates synthetic markdown corpus; overwrites existing files |
| `generate_test_set.py` | Once, or after major data changes | Writes `data/evaluation/test_set.json`; commit the output |
| `ingest.py` | After `generate_data.py` or adding new documents | Idempotent — deterministic chunk IDs prevent duplicates |
| `compare_chunks.py` | When evaluating chunking strategy | `--type city_guide` for batch; single file path for drill-down |
| `test_retrieval.py` | After schema or retrieval changes | Smoke-tests semantic, keyword, and hybrid search directly |
| `test_graph.py` | After agent changes | End-to-end test including query analysis and multi-retrieve |
| `scorecard.py` | At milestone checkpoints | Use `--skip-inference` to re-score without re-running the graph |

---

## Design Principles

Carried forward from prior projects:

- **Normalize at the boundary.** The chunking pipeline assumes well-formed markdown. Structural problems are fixed at document generation or conversion time, not inside the chunker.
- **Service layer pattern.** All business logic in transport-agnostic services. API routes are thin adapters. The same service functions can be called from CLI scripts, tests, or future transports.
- **Evidence-based decisions.** Chunking strategy, model tier, and retrieval design choices are backed by measurable evidence — batch comparison outputs, RAGAS scores, and cost measurements — not intuition.
- **Evaluation as a first-class deliverable.** The scorecard script and test set are committed artifacts, not afterthoughts. Pipeline changes can be quantified.

---

## License

Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE).