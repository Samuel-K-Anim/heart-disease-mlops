"""FastAPI production deployment script."""

from fastapi import FastAPI

app = FastAPI(title="Heart Disease MLOps")


@app.get("/health")
def health() -> dict:
    """Simple health endpoint."""
    return {"status": "ok"}
