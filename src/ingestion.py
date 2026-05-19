"""Adds simulated timestamps to raw Kaggle data."""

from datetime import datetime, timezone


def add_simulated_timestamp(record: dict) -> dict:
    """Return record with an ingestion timestamp."""
    output = dict(record)
    output.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
    return output
