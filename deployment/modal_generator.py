import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "groq",
        "requests",
        "python-dotenv",
        "loguru",
        "prometheus-fastapi-instrumentator",
        "prometheus-client",
    )
    .add_local_file("generator.py", "/app/generator.py")
    .add_local_file("observability.py", "/app/observability.py")
    .add_local_file("config.py", "/app/config.py")
)


app = modal.App("food-rag-generator", image=image)


@app.function(
    secrets=[modal.Secret.from_name("food-rag-secrets")],
    cpu=1.0,
    timeout=120,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    import sys

    sys.path.insert(0, "/app")

    from ml_backend.generator_groq import app

    return app
