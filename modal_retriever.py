import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "qdrant-client",
        "sentence-transformers",
        "spacy",
        "rank-bm25",
        "pandas",
        "numpy",
        "python-dotenv",
        "diskcache",
        "loguru",
        "prometheus-fastapi-instrumentator",
        "prometheus-client",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    # Embed your source files into the image at build time
    .add_local_file("retriever.py", "/app/retriever.py")
    .add_local_file("cache.py", "/app/cache.py")
    .add_local_file("observability.py", "/app/observability.py")
    .add_local_file("config.py", "/app/config.py")
)

model_vol = modal.Volume.from_name("food-rag-model-cache", create_if_missing=True)
cache_vol = modal.Volume.from_name("food-rag-query-cache", create_if_missing=True)

app = modal.App("food-rag-retriever", image=image)


@app.function(
    volumes={"/model-cache": model_vol},
    timeout=600,  # 10 min — E5 is 1.2GB, slow on first download
)
def download_models():

    import os

    os.environ["HF_HOME"] = "/model-cache"

    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("Downloading intfloat/e5-large-v2 (~1.2GB)...")
    SentenceTransformer("intfloat/e5-large-v2")

    print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    print("Committing to volume...")
    model_vol.commit()
    print("Done. Models are cached — cold starts will be fast from now on.")


@app.function(
    volumes={
        "/model-cache": model_vol,
        "/cache": cache_vol,
    },
    secrets=[modal.Secret.from_name("food-rag-secrets")],
    cpu=2.0,
    memory=2000,  # 3GB — E5(1.2GB) + CrossEncoder + overhead
    timeout=120,  # per-request timeout
    scaledown_window=500,
)
@modal.asgi_app()
def serve():
    import os
    import sys

    sys.path.insert(0, "/app")

    # Point HuggingFace at the persistent volume
    os.environ["HF_HOME"] = "/model-cache"
    # Point diskcache at the persistent volume
    os.environ["CACHE_DIR"] = "/cache"

    from retriever import app

    return app
