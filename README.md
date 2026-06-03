# RezRag: Production-Grade RAG System on the Yelp Business Review Dataset Entirely from Scratch

**RezRag** is a high-performance Retrieval-Augmented Generation (RAG) system engineered to provide grounded, location-aware restaurant recommendations  **built entirely from scratch without LangChain, LlamaIndex, or any RAG framework.** It leverages a **Hybrid Search Architecture** (Dense Vectors + Sparse Keywords) fused with a Cross-Encoder Reranker to retrieve precise context from the Yelp Academic Dataset, which is then synthesized by a 4-bit quantized LLM.

The system is architected as a set of decoupled, asynchronous microservices to ensure scalability and fault tolerance.

![Demo GIF](data/demo.gif)

> ⚡ **Live Demo:** [yelp-restaurant-rag.vercel.app](https://yelp-restaurant-rag.vercel.app)
> First request after inactivity takes ~15–20s (serverless cold start while E5 loads into memory). Subsequent queries respond faster.

---

## Why From Scratch?

To demonstrate a real understanding of everything that happens under the hood of RAG Frameworks. RezRag is built with full control and zero abstractions. No LangChain. No LlamaIndex. No RAG frameworks. There is no managed loader, no pre-built retriever, no abstracted LLM call.

---


## Architecture Overview

```
Raw Yelp JSON (~10GB)
    → Preprocessor (scoring, filtering, sentiment sampling)
    → Chunker (tiktoken-bounded, typed chunks)
    → Embedder (E5-large-v2, BM25 index)
    → Qdrant Cloud (vector DB)
         ↓
Retriever Microservice (FastAPI)
    → Geo filter (spaCy NER)
    → Qdrant vector search (HNSW)
    → Local BM25 rescoring
    → RRF fusion
    → CrossEncoder reranking
         ↓
Generator Microservice (FastAPI)
    → Intent Classification (llama-3.1-8b-instant via Groq)
         ↓ food_search / location_only → proceed
         ↓ greeting / identity / off_topic → short-circuit, no retrieval
    → Coverage Guard (COVERED_AREAS / OUT_OF_COVERAGE blocklists)
         ↓ out-of-coverage → short-circuit, no retrieval
    → Groq API (Qwen3-32B, production)
    → Local Qwen2.5-3B NF4 (offline mode)
         ↓
Next.js Frontend (Vercel)
```

## Data Pipeline — From Raw Yelp JSON to Production Grade Vector DB

The entire data pipeline is hand-built from the raw 
[Yelp Academic Dataset](https://www.yelp.com/dataset) ~10GB across multiple 
JSON files with no managed loaders, pre-processed datasets, or data APIs.

### Challenges solved:

**1. Multi-file joins at scale**: Streamed joins between `business.json` and `review.json` using `business_id`. without loading full datasets into RAM.

**2. Category filtering**: Parsed Yelp’s free-text category lists to isolate restaurants from 1000+ business types.

**3. Custom Restaurant Score**: Built a weighted ranking system combining rating, review count, and recency for more reliable quality scoring.

**4. Balanced sentiment sampling**: Sampled reviews across positive, neutral, and negative buckets to capture tradeoffs, not just high ratings.

**5. Tiktoken-bounded chunking**: Used token-based splitting to ensure chunks fit within E5-large-v2’s 512-token limit while preserving semantics.

**6. Typed chunk structure**: Created structured chunks (profile, positive, negative) instead of raw text to improve retrieval specificity.

**7. Embedding at scale**: Generated E5-large-v2 embeddings in batches and stored them as .pt tensors for fast reload.

**8. Stable deduplication**: Used UUID5-based deterministic IDs to ensure idempotent reprocessing without duplicates.

**9. Generator-based ingestion**: Streamed Qdrant uploads in batches of 256 using generators to keep memory usage constant.

---

## Under the Hood

### Streaming Filter Cascade (`preprocessor.py`)
Reviews are filtered at parse time in cheapest-to-expensive order to avoid loading 20GB+ into RAM:
1. **Business ID** — O(1) set lookup. Filters 95% of records instantly.
2. **Date filter** — lexicographic ISO string compare.
3. **Word count** — `text.split()` only runs on the survivors of (1) and (2).

### Composite Restaurant Scoring (`preprocessor.py`)
Raw star ratings are statistically broken for ranking; 3 reviews at 5 outranks
2000 reviews at 4.7 naively. I Built a two-component weighted score:
- `restaurant_score = stars × log1p(review_count)` — long-term reputation signal
- `reviews_score = mean_stars × log1p(sum_stars)` — recent review quality

Both are Min-Max normalized before combining with tunable coefficients.
City caps are adaptive, dense cities (600+ restaurants) get higher limits than sparse ones.

### Balanced Sentiment Sampling (`pipeline.py`)
Naive top-N sampling produces all 5-star reviews. The LLM then has no context
on wait times, portion sizes, or service issues. Reviews are sampled proportionally
across positive/neutral/negative buckets with ceiling rounding to preserve rare
sentiment classes.

### Tiktoken-Bounded Chunking (`chunker.py`)
Character-based splitting breaks semantic coherence mid-sentence. Token-budget
arithmetic runs with a fast estimation path (`len(text) // 3`) that only falls
back to the expensive `tiktoken.encode()` call when the estimate is borderline
avoiding unnecessary tokenizer overhead across 50k+ chunks.

### Typed Chunk Structure (`chunker.py`)
The chunks are not raw review blocks. Each restaurant produces structured chunk types:
- **Business Profile** — name, location, category, hours
- **Attribute chunks** — parsed from Yelp's nested stringified dicts (e.g. WiFi, parking, alcohol)
- **Vibe chunks** — atmosphere descriptors extracted from nested attribute maps
- **Positive / Neutral / Negative review batches** — sentiment-separated for retrieval precision

This allows the retriever to surface different facets of the same restaurant
depending on query intent, a "romantic atmosphere" query hits vibe chunks,
a "avoid if in a rush" query hits negative review chunks.

---

## Retrieval

### Hybrid Search

**Dense retrieval** — E5-large-v2 embeddings queried against Qdrant HNSW index. Captures semantic similarity ("cheap eats" -> "affordable prices").

**Sparse retrieval** — BM25 built on-the-fly over the Qdrant candidate pool. Captures exact keyword matches ("Tonkotsu Ramen", "Gluten-Free").

**RRF fusion** — Reciprocal Rank Fusion combines both rankings without score normalization:
```
rrf_score = 1/(k + vec_rank) + 1/(k + bm25_rank)    k=60
```

**Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` processes `[CLS] Query [SEP] Document` jointly, enabling deep interaction modeling between query and document tokens.

### Location Extraction
spaCy NER extracts city/state from queries. Fallback dictionary matching handles abbreviations ("Philly" -> Philadelphia, "NOLA" -> New Orleans) and state→city routing ("Pennsylvania" -> Philadelphia metro).

---

## Query Routing

Before any retrieval occurs, every query passes through two filtering layers:

### Intent Classification
`llama-3.1-8b-instant` via Groq classifies the query (~200ms) into one of:

| Intent | Action |
|:---|:---|
| `food_search` | Proceed to coverage check + retrieval |
| `location_only` | Proceed to retrieval with geo filter |
| `greeting` | Return intro message, skip retrieval |
| `identity` | Return RezRag description, skip retrieval |
| `off_topic` | Return redirect message, skip retrieval |


### Coverage Guard (Two Layers)
Out-of-coverage cities are blocked before retrieval at two independent points:

- **Generator-level** — `is_out_of_coverage()` checks the query against `COVERED_AREAS` / `OUT_OF_COVERAGE` lists before calling the retriever
- **Retriever-level** — `_detect_raw_location()` catches anything that slips through, returns `out_of_coverage: true` with empty results so irrelevant source cards never appear in the UI

Both layers use raw location detection (NER + regex) that is **not** filtered against the dataset so an unknown city like "Athens" is detected and blocked rather than silently returning unfiltered results from other cities.

---

## Inference

Two generator paths are implemented production uses Groq, local runs a fully quantized model offline.

### Production - Groq API
`Qwen3-32B` via Groq. Zero GPU cost, zero VRAM. Response is currently buffered by Modal's ASGI proxy — full response arrives after ~4s rather than true token streaming.

| Metric | Value |
|:---|:---|
| TTFT | ~4s (Modal buffered) |
| Tokens/sec | ~400k |
| GPU cost | 0 |

### Local / Offline - Qwen2.5-3B NF4

Qwen2.5-3B-Instruct in 4-bit NF4 quantization via `bitsandbytes`. Measured on RTX 3060 6GB:

| Metric | Value |
|:---|:---|
| VRAM baseline | 0 MB |
| VRAM at load | 2221 MB (~2.2GB) |
| VRAM peak generation | 2369 MB (~2.3GB) |
| Headroom remaining | ~3.7GB |
| Tokens/sec | 11.8 |
| TTFT (warm) | ~3.7s |
| TTFT (cold, first run) | ~37s (torch.compile JIT) |

Config:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

- `attn_implementation="sdpa"` — PyTorch scaled dot-product attention
- `torch.backends.cuda.matmul.allow_tf32 = True` — TF32 on Ampere
- `TextIteratorStreamer` + background thread — `model.generate()` runs in a thread, main thread yields tokens without blocking the event loop
- `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` after each generation prevents VRAM fragmentation across requests

**Why not 7B?** Qwen2.5-7B NF4 theoretically fits (~3.5GB weights) but spills onto CPU at inference time on 6GB. Measured result: TTFT 58s, 0.6 tokens/sec, ~22 minutes for a full response. So 3B NF4 is the best choice for 6GB VRAM.

### Comparison

| Path | Model | VRAM | TTFT (warm) | Tokens/sec | Streaming |
|:---|:---|:---|:---|:---|:---|
| Production | Qwen3-32B via Groq | 0 | ~4s | N/A (buffered) | Buffered |
| Local | Qwen2.5-3B NF4 | 2.2GB | ~3.7s | 11.8 | Real token stream |
| Local (attempted) | Qwen2.5-7B NF4 | 6GB+ (CPU offload) | ~58s | 0.6 | — |

---

## Retrieval Quality Evaluation

47 queries across all 12 dataset cities. Human-labeled relevance judgments. Fuzzy name matching with accent normalization, city/branch suffix stripping, and token-subset matching. Bootstrap 95% CI.

| Strategy | MRR@5 | 95% CI | Hit@3 | Hit@5 | P@5 | Avg Latency |
|:---|:---|:---|:---|:---|:---|:---|
| Hybrid + Rerank | 0.773 | [0.685, 0.854] | 0.936 | 1.000 | 0.523 | 1665ms |
| Hybrid (no rerank) | 0.737 | [0.640, 0.827] | 0.872 | 0.979 | 0.494 | 1324ms |

**Key findings:**
- Hit@5 = 1.000 with reranking every query returns at least one relevant result in top 5
- Cuisine queries are strongest (MRR@5 = 0.922), landmark queries are weakest (MRR@5 = 0.542)
- Location accuracy: 99.7% (metro-area aware matching)
- Reno and Santa Barbara lower performance due to limited Yelp dataset coverage

Eval script: `eval.py` — run with `--verbose` for per-query breakdown, `--out results.json` to save.

---

## Deployment

### Serverless Backend (Modal)

| Service | Resources |
|:---|:---|
| Retriever | 2 CPU, 2GB RAM |
| Generator | 1 CPU, 0.5GB RAM |

- E5-large-v2 and CrossEncoder pre-cached in a Modal Volume cold starts load from volume (~5-10s), not the internet
- `keep_warm=1` on retriever keeps one container warm
- Query result cache (diskcache / SQLite, 6-hour TTL) repeated queries skip retrieval entirely

### Frontend
Next.js on Vercel. Token streaming, interactive Leaflet map, restaurant cards with Maps/Yelp links, retrieval latency display.

---

## Performance

### Retrieval (shared across both paths)

| Step | Latency |
|:---|:---|
| E5 embedding (CPU) | ~200–500ms |
| Qdrant vector search | ~133–700ms |
| BM25 + RRF | ~0ms |
| CrossEncoder rerank | ~200–400ms |
| **Total retrieval** | **~700ms–1.2s** |

### End-to-End (warm, deployed)

| Path | Generator | Total |
|:---|:---|:---|
| Production (Groq) | ~4s buffered | ~5–6s |
| Local (Qwen2.5-3B NF4) | ~3.7s TTFT + ~60s generation | ~64s |

Variance is inherent to free-tier serverless infrastructure. A paid Qdrant cluster + dedicated CPU would bring down retrieval time consistently.

---


## Project Structure

```
RezRag/
├── ml_backend/
│   ├── config.py           # Centralized configuration
│   ├── retriever.py        # Hybrid search, RRF, geo-filter (FastAPI)
│   ├── generator.py        # Groq + local Qwen generator (FastAPI)
│   ├── cache.py            # diskcache SQLite query result cache
│   ├── observability.py    # Prometheus metrics + loguru logging
│   └── eval.py             # MRR, Hit@K, P@K, bootstrap CI
├── data_pipeline/
│   ├── preprocessor.py     # Raw Yelp NDJSON filtering + scoring
│   ├── chunker.py          # Tiktoken-bounded semantic chunking
│   ├── embedder.py         # E5-large-v2 embeddings + BM25 index
│   └── ingester.py         # Generator-based Qdrant ingestion
├── deployment/
│   ├── modal_retriever.py  # Modal serverless — retriever
│   └── modal_generator.py  # Modal serverless — generator
└── ui/                     # Next.js frontend
```

---

## Performance Optimizations Implemented

1. **Parquet & PyTorch Formats:** Replaced standard CSV/JSON/Pickle intermediate files with **Parquet** (for metadata) and **.pt Tensors** (for vectors). 
2. **Pandas Vectorization:** `chunker.py` used `df.explode()` and `df.melt()` instead of expensive `for` loops. This moves the iteration logic to C-level Pandas optimizations, speeding up processing.
3. **Generator-Based Ingestion:** The `ingestor.py` does not load the full dataset into RAM. It lazily reads from the disk and yields batches to the Qdrant client, allowing the ingestion of datasets larger than system RAM.
4. **Query cache** — diskcache (SQLite-backed) with 6-hour TTL; repeated queries skip retrieval entirely
5. **Qdrant co-location** — database and compute in same cloud region eliminates cross-cloud latency (~200ms saved)

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (for local inference)
- CUDA 11.8 or 12.1
- Docker (for local Qdrant)

### Install

```bash
git clone https://github.com/your-username/rezrag.git
cd rezrag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

`.env`:
```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
RETRIEVER_URL=http://127.0.0.1:8000/retrieve
GROQ_API_KEY=gsk_...
GROQ_MODEL_ID=qwen3-32b
```

`.env.local` (frontend):
```ini
NEXT_PUBLIC_GENERATOR_URL=http://localhost:9000
NEXT_PUBLIC_STADIA_API_KEY=your_stadia_key
```

### Run Data Pipeline

```bash
python data_pipeline/preprocessor.py
python data_pipeline/chunker.py
python data_pipeline/embedder.py
python data_pipeline/ingester.py
```

### Run Services

```bash
# Terminal 1
uvicorn ml_backend.retriever:app --port 8000

# Terminal 2
uvicorn ml_backend.generator:app --port 9000

# Terminal 3
cd ui && npm run dev
```

### Run Evaluation

```bash
python ml_backend/eval.py --url http://127.0.0.1:8000 --top_k 8 --verbose
```

---

## API

### Generate (user-facing)
`POST /generate`
```json
{ "query": "best tacos in Philadelphia", "top_k": 8 }
```

Response — NDJSON stream:
```
{"type": "ping"}
{"type": "meta", "data": {"retrieval_ms": 850, "results_count": 8, "reranked": true}}
{"type": "sources", "data": [...]}
{"type": "token", "data": "Here"}
{"type": "token", "data": " are"}
...
```

### Retrieve (debug)
`POST /retrieve`
```json
{ "query": "late night ramen", "top_k": 5, "do_rerank": true }
```

