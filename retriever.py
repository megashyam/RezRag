from http import client

import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"


import string
import time
import numpy as np
import pandas as pd
import spacy
import torch
from typing import Optional, List
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models


from observability import (
    setup_logging,
    attach_prometheus,
    Timer,
    QUERY_COUNTER,
    CACHE_COUNTER,
    RESULTS_HISTOGRAM,
    RERANK_IMPROVEMENT,
)
from cache import get_cached, set_cached, cache_stats, evict_all
from loguru import logger

load_dotenv()

import config


class HybridRetriever:
    """
    Service class handling the hybrid retrieval pipeline, including vector search,
    BM25 rescoring, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.
    """

    def __init__(self):
        self.qdrant: Optional[QdrantClient] = None
        self.reranker: Optional[CrossEncoder] = None
        self.nlp = None
        self.city_list = set()
        self.embedding_model = "intfloat/e5-large-v2"
        self.embedder = None

    def initialize(self):
        """
        Loads required machine learning models (Embedder, Reranker, Spacy)
        and connects to the Qdrant vector database.

        Raises:
            Exception: If connection to Qdrant fails.
        """
        logger.info(f"Initializing Hybrid Retriever on {config.DEVICE}...")

        # 1. Qdrant
        try:
            self.qdrant = QdrantClient(
                url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60
            )
            self.qdrant.get_collections()  # Health check
            logger.info("Connected to Qdrant.")
            for field in ["city", "state", "restaurant", "doc_id"]:
                self.qdrant.create_payload_index(
                    collection_name=config.COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info("Connected to Qdrant and indices verified.")
        except Exception as e:
            logger.error(f"Qdrant Connection Failed: {e}")
            raise e

        # 2. Reranker
        logger.info(f"Loading Reranker: {config.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL_NAME, max_length=512, device=config.DEVICE
        )

        # Embedder
        logger.info("Loading E5 embedder ...")
        self.embedder = SentenceTransformer("intfloat/e5-large-v2", device="cpu")
        logger.info(" E5 embedder ready.")

        # 3. Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 4. City Data
        try:
            if pd.io.common.file_exists(config.DATA_PATH):
                df = pd.read_csv(config.DATA_PATH)
                if "city" in df.columns:
                    self.city_list = set(df["city"].str.lower().dropna().unique())
                    logger.info(f"Loaded {len(self.city_list)} cities.")
        except Exception as e:
            logger.warning(f"Could not load city list: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates a dense embedding for the given text using the E5 model.

        Args:
            text (str): The input query string to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """

        import json as _json

        with Timer("retriever", "embedding"):
            vec = self.embedder.encode(
                f"query: {text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return vec.tolist()

    def _extract_location(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts City and State entities from the query using Spacy NER
        and a fallback dictionary lookup.

        Args:
            query (str): The search query.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the extracted (city, state).
        """
        doc = self.nlp(query)
        city, state = None, None

        for ent in doc.ents:
            if ent.label_ == "GPE":
                text = ent.text.strip().lower()
                if text in config.US_STATES:
                    state = config.US_STATES[text]
                elif text.upper() in config.STATE_ALIASES:
                    state = text.upper()
                else:
                    city = text

        # Fallback to dictionary match for city
        if not city and self.city_list:
            tokens = query.lower().split()
            for word in tokens:
                if word in self.city_list:
                    city = word
                    break
        return city, state

    def search(
        self,
        query: str,
        top_k: int,
        initial_k: int,
        k_rrf: int,
        do_rerank: bool,
        max_duplicates: int,
    ):
        """
        Executes the Hybrid Search Pipeline:
        1. Vector Search (Qdrant)
        2. Local BM25 (on retrieved chunks)
        3. RRF Fusion
        4. Cross-Encoder Reranking

        Args:
            query (str): The search query provided by the user.
            top_k (int): Final number of results to return.
            initial_k (int): Number of initial candidates to retrieve from Qdrant.
            k_rrf (int): Constant used in the RRF (Reciprocal Rank Fusion) calculation.
            do_rerank (bool): Flag indicating whether to use the cross-encoder for reranking.
            max_duplicates (int): Maximum allowable chunks from the same business/restaurant.

        Returns:
            Tuple[List[Dict[str, Any]], float]: A tuple containing the list of formatted result dictionaries and the total retrieval time in milliseconds.
        """

        t0 = time.perf_counter()
        # A. Extract Location & Build Filter
        city, state = self._extract_location(query)
        logger.info(f"Query: '{query}' | Extracted Loc: {city}, {state}")

        conditions = []
        if city:
            conditions.append(
                models.FieldCondition(
                    key="city", match=models.MatchValue(value=city.lower())
                )
            )
        if state:
            conditions.append(
                models.FieldCondition(key="state", match=models.MatchValue(value=state))
            )

        q_filter = models.Filter(must=conditions) if conditions else None
        print(f"Qdrant Filter: {q_filter}")

        # B. Get Embeddings & Query Qdrant
        with Timer("retriever", "embedding"):
            query_vec = self._get_embedding(query)
            t1 = time.perf_counter()

        with Timer("retriever", "qdrant_query"):
            try:
                points = self.qdrant.query_points(
                    collection_name=config.COLLECTION_NAME,
                    query=query_vec,
                    query_filter=q_filter,
                    limit=initial_k,
                    with_payload=True,
                ).points
                t2 = time.perf_counter()
            except Exception as e:
                logger.error(f"Qdrant Query Failed: {e}")
            # return []
        logger.info(f"Qdrant returned {len(points)} points.")
        # if not points:
        #     return []

        # C. Prepare Data for Fusion
        chunks = [
            {
                "text": p.payload.get("text_content", ""),
                "meta": p.payload,
                "vec_score": p.score,
            }
            for p in points
        ]
        corpus_texts = [c["text"] for c in chunks]

        # D. Local BM25 Rescoring (Ranking the retrieved candidates by keyword overlap)
        with Timer("retriever", "bm25"):
            translator = str.maketrans("", "", string.punctuation)
            clean_query = query.lower().translate(translator).split()

            tokenized_corpus = [
                doc.lower().translate(translator).split() for doc in corpus_texts
            ]

            if tokenized_corpus:
                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = np.array(bm25.get_scores(clean_query))
                t3 = time.perf_counter()
            else:
                bm25_scores = np.zeros(len(chunks))

            vector_scores = np.array([c["vec_score"] for c in chunks])

        # E. Reciprocal Rank Fusion (RRF)
        vec_rank = np.argsort(np.argsort(-vector_scores))
        bm25_rank = np.argsort(np.argsort(-bm25_scores))

        rrf_scores = (1 / (k_rrf + vec_rank)) + (1 / (k_rrf + bm25_rank))

        # Sort by RRF score descending
        candidate_indices = np.argsort(-rrf_scores)
        t4 = time.perf_counter()

        # F. Reranking (Cross Encoder)
        final_indices = candidate_indices
        final_scores = rrf_scores[candidate_indices]

        t5 = t4

        if do_rerank and self.reranker:
            # Only rerank the sorted candidates
            top_candidates_idx = candidate_indices  # Rerank all retrieved candidates
            with Timer("retriever", "rerank"):
                pairs = [[query, corpus_texts[i][:400]] for i in top_candidates_idx]

                if pairs:
                    rerank_scores = self.reranker.predict(pairs)
                    t5 = time.perf_counter()
                    # Sort based on reranker output
                    sorted_rerank_idx = np.argsort(-rerank_scores)

                    # Map back to original indices
                    final_indices = [top_candidates_idx[i] for i in sorted_rerank_idx]
                    final_scores = [rerank_scores[i] for i in sorted_rerank_idx]

                    # Track when reranker changes top result
                    if (
                        len(final_indices) > 0
                        and final_indices[0] != candidate_indices[0]
                    ):
                        RERANK_IMPROVEMENT.labels("retriever").inc()

        # G. Deduplication & Formatting
        results = []
        seen_biz = {}

        for idx, score in zip(final_indices, final_scores):
            c = chunks[idx]
            b_id = c["meta"].get("business_id")

            if seen_biz.get(b_id, 0) < max_duplicates:
                results.append(
                    {
                        "score": float(score),
                        "restaurant": c["meta"].get("restaurant", "Unknown"),
                        "text": c["text"],
                        "city": c["meta"].get("city"),
                        "state": c["meta"].get("state_abbr"),
                        "address": c["meta"].get("address"),
                        "latitude": c["meta"].get("latitude"),
                        "longitude": c["meta"].get("longitude"),
                    }
                )
                seen_biz[b_id] = seen_biz.get(b_id, 0) + 1

            if len(results) >= top_k:
                break
        logger.info(len(results))
        t_end = time.perf_counter()
        retrieval_ms = float((t_end - t0) * 1000)

        try:
            logger.info(
                f"[PERF] embed={int((t1-t0)*1000)}ms | "
                f"qdrant={int((t2-t1)*1000)}ms | "
                f"bm25={int((t3-t2)*1000)}ms | "
                f"rrf={int((t4-t3)*1000)}ms | "
                f"rerank={int((t5-t4)*1000)}ms | "
                f"total={int((t5-t0)*1000)}ms"
            )
        except:
            pass
        return results, retrieval_ms


# Global Service Instance
retriever = HybridRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages FastAPI application startup and shutdown events,
    initializing the retriever service upon startup.
    """
    # Startup
    setup_logging("retriever")
    try:
        retriever.initialize()
        logger.info("Retriever ready.")
    except Exception as e:
        logger.error(f"Critical Startup Error: {e}")
    yield
    # Shutdown
    logger.info("Shutting down retriever service...")


app = FastAPI(title="Retrieval Microservice", lifespan=lifespan)
attach_prometheus(app, "retriever")


# --- Models ---
class RetrieveRequest(BaseModel):
    """
    Pydantic schema for defining user retrieval request payloads.
    """

    query: str
    top_k: int = config.TOP_K
    initial_k: int = config.INITIAL_K
    k_rrf: int = config.RRF_K
    max_duplicates: int = config.MAX_DUPLICATES
    do_rerank: bool = config.DO_RERANK


class RetrieveResponse(BaseModel):
    """
    Pydantic schema for formatting the retrieval response payload.
    """

    results: List[Dict[str, Any]]
    retrieval_ms: float


# --- Endpoints ---
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify service availability.

    Returns:
        Dict: A standard ok status dictionary.
    """
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    """
    Main endpoint for executing a document retrieval query.
    Checks the local cache before running the full search pipeline.

    Args:
        req (RetrieveRequest): Request payload containing search parameters.

    Returns:
        Dict: Contains a list of matching results and execution time in ms.

    Raises:
        HTTPException: If the retriever is not initialized or an internal error occurs.
    """
    if not retriever.qdrant:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    # Cache check
    cached = get_cached(req.query, req.top_k, req.do_rerank)
    if cached is not None:
        CACHE_COUNTER.labels("hit").inc()
        QUERY_COUNTER.labels("retriever", "cache_hit").inc()
        return {"results": cached, "retrieval_ms": 0}

    # If no cached result, perform full retrieval
    CACHE_COUNTER.labels("miss").inc()

    # Full retrieval pipeline
    try:
        results, retrieval_ms = retriever.search(
            query=req.query,
            top_k=req.top_k,
            initial_k=req.initial_k,
            k_rrf=req.k_rrf,
            do_rerank=req.do_rerank,
            max_duplicates=req.max_duplicates,
        )

    except Exception as e:
        QUERY_COUNTER.labels("retriever", "error").inc()
        import traceback

        logger.error(
            f"Retrieval failed for '{req.query}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    RESULTS_HISTOGRAM.observe(len(results))
    QUERY_COUNTER.labels("retriever", "success").inc()

    set_cached(req.query, req.top_k, req.do_rerank, results)
    logger.info(f"Retrieved {(len(results))} for query: '{req.query}'")

    return {"results": results, "retrieval_ms": retrieval_ms}


@app.get("/cache/stats")
def cache_stats_endpoint():
    """
    Retrieves internal cache statistics (hits, misses, sizes).

    Returns:
        Dict: A dictionary containing cache metrics.
    """
    return cache_stats()


@app.delete("/cache")
def clear_cache_endpoint():
    """
    Clears all items currently held in the retrieval cache.

    Returns:
        Dict: Number of items cleared.
    """
    n = evict_all()
    return {"cleared": n}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
