# RezRag: Production-Grade RAG System on the Yelp Business Review Dataset Entirely from Scratch

**RezRag** is a high-performance Retrieval-Augmented Generation (RAG) system engineered to provide grounded, location-aware restaurant recommendations  **built entirely from scratch without LangChain, LlamaIndex, or any RAG framework.** It leverages a **Hybrid Search Architecture** (Dense Vectors + Sparse Keywords) fused with a Cross-Encoder Reranker to retrieve precise context from the Yelp Academic Dataset, which is then synthesized by a 4-bit quantized LLM.

The system is architected as a set of decoupled, asynchronous microservices to ensure scalability and fault tolerance.

![Demo GIF](data/demo.gif)

> ⚡ **Live Demo:** [yelp-restaurant-rag.vercel.app](https://yelp-restaurant-rag.vercel.app)
> First request after inactivity takes ~15–20s (serverless cold start while E5 loads into memory). Subsequent queries respond faster.

---

## Why From Scratch?

To demonstrate a real understanding of everything that happens under the hood of RAG Frameworks. RezRag is built with full control and zero abstractions. No LangChain. No LlamaIndex. No RAG frameworks. This project starts from the raw Yelp Academic Dataset (~10GB, multi-file NDJSON)
and processes it end-to-end with no managed loaders. Every component (hybrid search, BM25, RRF fusion, cross-encoder reranking, streaming microservices) is hand-rolled in Python. :


### Streaming Filter Cascade (`preprocessor.py`)
The review file contains millions of records. Loading it before filtering = 20GB RAM.
Instead, three filters run at parse time ordered cheapest → most expensive:
1. **Business ID** — O(1) set lookup. Kills 95% of records instantly.
2. **Date filter** — lexicographic ISO string compare. Free.
3. **Word count** — `text.split()` only runs on the survivors of (1) and (2).

### Composite Restaurant Scoring (`preprocessor.py`)
Raw star ratings are statistically broken for ranking — 3 reviews at 5★ outranks
2000 reviews at 4.7★ naively. Built a two-component weighted score:
- `restaurant_score = stars × log1p(review_count)` — long-term reputation signal
- `reviews_score = mean_stars × log1p(sum_stars)` — recent review quality

Both Min-Max normalized before combining with tunable coefficients.
City caps are adaptive — dense cities (600+ restaurants) get higher limits than sparse ones.

### Balanced Sentiment Sampling (`pipeline.py`)
Naive top-N sampling produces all 5-star reviews. The LLM then has no context
on wait times, portion sizes, or service issues. Reviews are sampled proportionally
across positive/neutral/negative buckets with ceiling rounding to preserve rare
sentiment classes.

### Tiktoken-Bounded Chunking (`chunker.py`)
Character-based splitting breaks semantic coherence mid-sentence. Token-budget
arithmetic runs with a fast estimation path (`len(text) // 3`) that only falls
back to the expensive `tiktoken.encode()` call when the estimate is borderline —
avoiding unnecessary tokenizer overhead across 50k+ chunks.

### Typed Chunk Structure (`chunker.py`)
Chunks are not raw review dumps. Each restaurant produces structured chunk types:
- **Business Profile** — name, location, category, hours
- **Attribute chunks** — parsed from Yelp's nested stringified dicts (e.g. WiFi, parking, alcohol)
- **Vibe chunks** — atmosphere descriptors extracted from nested attribute maps
- **Positive / Neutral / Negative review batches** — sentiment-separated for retrieval precision

This allows the retriever to surface different facets of the same restaurant
depending on query intent — a "romantic atmosphere" query hits vibe chunks,
a "avoid if in a rush" query hits negative review chunks.

### Embedding Pipeline (`embedder.py`)
E5-large-v2 (1024-dim) run over 50k+ chunks with:
- `df.melt()` + `df.explode()` vectorized flattening — no nested Python loops
- Fast estimation path avoids re-tokenizing on every iteration
- `load_precomputed` flags on both embedding and BM25 steps — skip re-embedding
  during iteration without changing any code
- BM25 index serialized to disk alongside `.pt` tensors for instant reload

---

## Data Pipeline — From Raw Yelp JSON to Production Vector DB

The entire data pipeline is hand-built from the raw 
[Yelp Academic Dataset](https://www.yelp.com/dataset) ~10GB across multiple 
JSON files with no managed loaders, pre-processed datasets, or data APIs.

### Challenges solved:

**1. Multi-file joins at scale** — `business.json` and `review.json` are separate 
files joined by `business_id`. Processed in streaming fashion to avoid loading 
gigabytes into RAM simultaneously.

**2. Category filtering** — Yelp categories are free-text comma lists. Built a 
keyword filter to isolate restaurants from the 1000+ other business types in the 
dataset.

**3. Custom Restaurant Score** — raw star ratings are unreliable (3 reviews at 5★ 
outranks 2000 reviews at 4.7★). Built a weighted formula balancing rating, review 
volume, and recency to surface genuinely high-quality restaurants.

**4. Balanced sentiment sampling** — naive top-N sampling produces all 5-star 
results. Sampled across sentiment buckets (positive/neutral/negative) so the 
retriever has context on tradeoffs, wait times, and common complaints — not just 
hype.

**5. Tiktoken-bounded chunking** — reviews range from 2 sentences to 20 paragraphs. 
Character-based splitting breaks semantic coherence. Used tiktoken to bound chunks 
by token count so every chunk fits cleanly within E5-large-v2's 512-token context 
window.

**6. Typed chunk structure** — chunks are not raw review text. Each restaurant 
produces three chunk types: Business Profile (name, location, hours, category), 
Positive Review Summary, Negative Review Summary. This allows the retriever to 
surface different facets depending on query intent.

**7. Embedding at scale** — E5-large-v2 (1024 dimensions) run over tens of 
thousands of chunks with batched inference, saved as `.pt` tensors for fast reload 
without re-embedding.

**8. Stable deduplication** — UUID5 (deterministic, content-based) IDs prevent 
duplicate points on re-ingestion. Re-running the pipeline is idempotent.

**9. Generator-based ingestion** — Qdrant upload streams in batches of 256 via 
Python generators. RAM usage stays flat regardless of dataset size.

## Production Deployment

The original local prototype has been extended into a fully deployed, cloud-native system:

### Serverless Backend (Modal)
Both microservices are deployed on [Modal](https://modal.com) — a serverless GPU/CPU platform that scales to zero between requests and cold-starts on demand.

| Service | URL | Resources |
| :--- | :--- | :--- |
| **Retriever** | `https://megumind6172--food-rag-retriever-serve.modal.run` | 2 CPU, 2GB RAM |
| **Generator** | `https://megumind6172--food-rag-generator-serve.modal.run` | 2 CPU, 0.5GB RAM |

Key deployment decisions:
- E5-large-v2 and CrossEncoder models pre-cached in a Modal **Volume** to avoid re-downloading on cold start
- `keep_warm=1` on the retriever keeps one container warm to eliminate cold start latency
- Generator calls **Groq API** (`qwen-2.5-32b`) instead of loading a local LLM — no GPU required in production

### Frontend (Vercel + Next.js)
A new Next.js frontend replaces the original chat interface, deployed on Vercel

### Retrieval Optimizations
- Reduced `INITIAL_K` from 50 → 8 (reranking fewer candidates cuts reranker time from ~2.4s → ~400ms)
- CrossEncoder truncates documents to 400 chars before reranking (attention scales quadratically)
- Qdrant cluster co-located in same cloud region as Modal deployment (GCP `us-east4`) to eliminate cross-cloud latency
- Query result cache (diskcache / SQLite) with 6-hour TTL — repeated queries return instantly

## Technical Architecture

### 1. Hybrid Search Engine (The "Retriever")

The core retrieval logic solves the "vocabulary mismatch problem" inherent in semantic search by combining two distinct indexing strategies.

![Hybrid Search](data/hybrid.png)

**Dense Retrieval (Semantic):**
* **Model:** `intfloat/e5-large-v2` (1024 dimensions).
* Captures conceptual similarity (e.g., "cheap eats" → "affordable prices").
* **Index:** HNSW (Hierarchical Navigable Small World) graph in **Qdrant** for sub-millisecond approximate nearest neighbor search.


**Sparse Retrieval (Lexical):**
* BM25 (Best Matching 25).
* Captures exact keyword matches (e.g., specific dish names like "Tonkotsu Ramen" or dietary tags like "Gluten-Free").
* An in-memory inverted index built dynamically on the candidate subset or pre-computed, optimized for high-precision filtering.


**Fusion Strategy: Reciprocal Rank Fusion (RRF)**
* Combines disparate scores from dense & sparse retrieval
* Prevents outlier scores from dominating the final ranking

### 2. Multi-Stage Filtering Pipeline

Retrieval is not a single step but a cascade of filters designed to maximize precision while minimizing latency.

1. **Geo-Spatial Filtering:** Extracts city/state via spaCy `NER`; restricts search to target geolocation.
2. **Vector Candidate Generation:** The top 50 candidates are retrieved using the dense vector index.
3. **Local Rescoring:** A lightweight BM25 index is built *on-the-fly* for just these 50 candidates to re-rank them based on exact keyword overlap with the user prompt.
4. **Cross-Encoder Reranking:** The re-ordered candidates are passed to `ms-marco-MiniLM-L-6-v2`. Unlike bi-encoders (which process query and document separately), this model processes `[CLS] Query [SEP] Document` simultaneously, allowing the self-attention mechanism to perform deep interaction modeling between query terms and document tokens.

### 3. The Generator (LLM)

* **Model(On Device):** `Qwen/Qwen2.5-7B-Instruct`. Chosen for its superior reasoning capabilities at small parameter counts.
* **Quantization:** Loaded in **4-bit NF4** (Normal Float 4) format using `bitsandbytes`. This reduces VRAM usage from ~7GB to ~2.5GB, allowing high-performance inference on consumer hardware (RTX 3060/4060).
* **Streaming:** Responses are streamed token-by-token using Python Generators and Server-Sent Events (NDJSON) to minimize Time-To-First-Token (TTFT).
* **Model(Live Demo):** `Qwen/Qwen2.5-32B` via **Groq API**. Zero GPU cost, ~500ms TTFT

---

## ⚡ Performance Optimizations Implemented

1. **Parquet & PyTorch Formats:** Replaced standard CSV/JSON/Pickle intermediate files with **Parquet** (for metadata) and **.pt Tensors** (for vectors). 
2. **Pandas Vectorization:** `chunker.py` used `df.explode()` and `df.melt()` instead of expensive `for` loops. This moves the iteration logic to C-level Pandas optimizations, speeding up processing.
3. **Generator-Based Ingestion:** The `ingestor.py` does not load the full dataset into RAM. It lazily reads from the disk and yields batches to the Qdrant client, allowing the ingestion of datasets larger than system RAM.
4. **Query cache** — diskcache (SQLite-backed) with 6-hour TTL; repeated queries skip retrieval entirely
5. **Qdrant co-location** — database and compute in same cloud region eliminates cross-cloud latency (~200ms saved)

## 📂 Project Structure

```bash
RezRag/
├── data/                    # Storage for pickle, parquet, and vectors
├── config.py                # Centralized configuration
├── pipeline.py              # Raw Data Preprocessing
├── chunker.py               # Semantic Chunking
├── embedder.py              # Vector & BM25 Generation
├── ingestor.py              # Qdrant Ingestion
├── retriever.py             # Hybrid Search, RRF, Reranker (FastAPI)
├── generator.py             # Groq LLM + Prompt Builder (FastAPI)
├── modal_retriever.py       # Modal serverless deployment — retriever
├── modal_generator.py       # Modal serverless deployment — generator
├── cache.py                 # diskcache query result cache
├── observability.py         # Prometheus metrics + loguru logging
├── food-rag/                # Next.js frontend
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # Main chat interface
│       │   └── readme/page.tsx    # Live README renderer
│       └── components/
│           ├── MapPanel.tsx       # Leaflet map + marker interactions
│           ├── RestaurantCard.tsx # Result cards with Maps/Yelp links
│           └── TimingBar.tsx
└── requirements.txt

```

---

## 🛠️ Prerequisites

* **Operating System:** Linux or Windows (WSL2 recommended).
* **Hardware:** NVIDIA GPU with at least 6GB VRAM (8GB+ recommended).
* **Software:**
* Python 3.10+
* CUDA Toolkit 11.8 or 12.1



---

## 🚀 Installation & Setup

### 1. Environment Setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/your-username/foodguru.git
cd foodguru
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```

### 2. Install Dependencies

Install PyTorch with CUDA support first, then the rest:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

### 3. Launch Vector Database

Run Qdrant using Docker. This persists data to a local `qdrant_storage` folder.

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

```

### 4. Configuration

Create a `.env` file in the root directory:

```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave blank for local instance

E5_URL=http://127.0.0.1:5000/embed
RETRIEVER_URL=http://127.0.0.1:8000/retrieve

GROQ_API_KEY=gsk_...
GROQ_MODEL_ID=qwen-2.5-32b

```

### 5. Frontend Setup

RezRag includes a Node.js frontend/backend. Follow these steps to set it up:

Create the Next.js app inside a folder called rag-chat:

```ini
npx create-next-app rag-chat --ts

```
Copy the contents of rag-chat into the main project’s src folder:

```ini
mkdir -p src
cp -r rag-chat/* src/

```
(Windows PowerShell: Copy-Item -Recurse rag-chat\* src\)

Install Node.js dependencies inside src:
```ini
mkdir -p src
cp -r rag-chat/* src/

```

Start the Node.js app:
```
npm run dev
```

```ini
# food-rag/.env.local
NEXT_PUBLIC_GENERATOR_URL=http://localhost:9000
NEXT_PUBLIC_STADIA_API_KEY=your_stadia_key
```

---

## Phase 1: Data Ingestion Pipeline

### Step 1: Preprocessing (`pipeline.py`)

Cleans the raw Yelp JSON, calculates the custom "Restaurant Score" (), and filters the top ~200 restaurants per city to remove noise.

```bash
python pipeline.py

```

### Step 2: Semantic Chunking (`chunker.py`)

Explodes the dataframe and creates semantic text chunks (Business Profiles, Positive Review Summaries, Negative Review Summaries). Uses `tiktoken` to ensure chunks fit within the embedding model's context window.

```bash
python chunker.py

```

### Step 3: Vectorization (`embedder.py`)

Generates 1024-dimension dense vectors using E5-Large and builds the sparse BM25 index. Saves artifacts to disk as `.pt` (PyTorch Tensor) and `.parquet` files.

```bash
python embedder.py

```

### Step 4: Database Ingestion (`ingestor.py`)

Streams the processed vectors and metadata into Qdrant. Uses **Python Generators** to yield batches of 256 points, ensuring RAM usage remains flat regardless of dataset size.

```bash
python ingestor.py

```

---

## Phase 2: Running the Microservices Locally

```bash
uvicorn retriever:app --port 8000
uvicorn generator:app --port 9000
```

**Service Status:**
| Microservice | URL | Port | Role |
| :--- | :--- | :--- | :--- |
| **Retrieval** | `http://127.0.0.1:8000` | 8000 | Handles Hybrid Search, RRF Fusion, and Reranking. |
| **Generator** | `http://127.0.0.1:9000` | 9000 | The main RAG Chat Interface. Streams LLM tokens. |

---

## 📡 API Reference

### Chat / Generation Endpoint

**POST** `http://127.0.0.1:9000/generate`

This is the user-facing endpoint. returns a stream of **NDJSON** events.

**Request:**

```json
{
  "query": "Where can I find the best deep dish pizza?",
  "city": "Chicago",
  "top_k": 7
}

```

**Response Stream (NDJSON):**
The first event contains the retrieved sources, followed by the LLM tokens.

```json
{"type": "sources", "data": [{"name": "Giordano's", "address": "223 W Jackson Blvd", "lat": 41.87, "lon": -87.63}, ...]}
{"type": "token", "data": "If"}
{"type": "token", "data": " you're"}
{"type": "token", "data": " looking"}
{"type": "token", "data": " for"}
...

```

### Retrieval Debug Endpoint

**POST** `http://127.0.0.1:8000/retrieve`

Used to debug the search quality without waiting for the LLM generation.

**Request:**

```json
{
  "query": "Spicy Ramen",
  "top_k": 5,
  "do_rerank": true
}

```

---


