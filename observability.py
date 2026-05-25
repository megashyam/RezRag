"""
Observability: structured logging (loguru) + Prometheus metrics.
Call setup_logging() and attach_prometheus() in each service's lifespan.
"""

import os
import sys
import time

from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import Response

# ── Metrics ───────────────────────────────────────────────────────────────────

QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total queries handled",
    ["service", "status"],
)

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency per pipeline stage",
    ["service", "stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)

CACHE_COUNTER = Counter(
    "rag_cache_total",
    "Cache hits and misses",
    ["result"],
)

RESULTS_HISTOGRAM = Histogram(
    "rag_results_returned",
    "Number of results returned per query",
    buckets=[0, 1, 2, 3, 5, 8, 12, 20],
)

RERANK_IMPROVEMENT = Counter(
    "rag_rerank_improvements_total",
    "Times reranker changed top result vs vector ranking",
    ["service"],
)


# ── Timer context manager ─────────────────────────────────────────────────────


class Timer:
    def __init__(self, service: str, stage: str):
        self.service = service
        self.stage = stage
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        STAGE_LATENCY.labels(self.service, self.stage).observe(elapsed)
        logger.debug(f"[{self.service}] {self.stage}: {elapsed*1000:.1f} ms")


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(service: str) -> None:
    logger.remove()
    os.makedirs("logs", exist_ok=True)

    logger.add(
        f"logs/{service}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "{message}"
        ),
        level="DEBUG",
        colorize=True,
    )

    logger.info(f"[{service}] logger ready")


# ── Prometheus instrumentation ────────────────────────────────────────────────


def attach_prometheus(app, service: str) -> None:
    """
    Instruments the app with Prometheus metrics and adds a /metrics endpoint
    that returns text/plain — the format Prometheus actually expects.

    prometheus-fastapi-instrumentator's expose() returns application/json
    in some FastAPI versions, which breaks scraping. This bypasses that
    by adding the endpoint manually.
    """
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,  # "text/plain; version=0.0.4; charset=utf-8"
        )

    logger.info(f"[{service}] Prometheus metrics → /metrics (text/plain)")
