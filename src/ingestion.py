"""Adds simulated timestamps to raw Kaggle data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames verbose Kaggle column names to standard medical shortcodes
    expected by the feature store and downstream API.
    """
    # Mapping of long Kaggle names to standard shortcodes
    schema_mapping = {
        "Age": "age",
        "Sex": "sex",
        "Chest pain type": "cp",
        "BP": "trestbps",
        "Cholesterol": "chol",
        "FBS over 120": "fbs",
        "EKG results": "restecg",
        "Max HR": "thalach",
        "Exercise angina": "exang",
        "ST depression": "oldpeak",
        "Slope of ST": "slope",
        "Number of vessels fluro": "ca",
        "Thallium": "thal",
        "Target": "target",
        "Heart Disease": "target",
        # "Heart Disease": "Heart Disease"
    }

    # Rename columns that exist in the mapping
    df = df.rename(columns=schema_mapping)

    # Ensure all column names are lowercase as a safety measure
    df.columns = df.columns.str.lower()
    return df


def ingest_and_timestamp_data(synthetic_path: str, clinical_path: str, output_dir: str):
    """
    Ingests flat CSVs, assigns unique patient IDs, and generates deterministic
    timestamps to simulate a production data lake for Feast.
    """
    print("Loading raw data...")
    df_synth = pd.read_csv(synthetic_path)
    df_clin = pd.read_csv(clinical_path)

    print("Standardizing Schemas...")
    df_synth = standardize_schema(df_synth)
    df_clin = standardize_schema(df_clin)

    print("Harmonizing Schemas and IDs...")

    # 1. Drop the Kaggle competition 'id' column from synthetic data if it exists
    if "id" in df_synth.columns:
        df_synth = df_synth.drop(columns=["id"])

    # Drop 'id' from clinical data just in case it was manually added
    if "id" in df_clin.columns:
        df_clin = df_clin.drop(columns=["id"])

    # 2. Generate unified, non-overlapping patient_ids
    # We start synthetic IDs at 1,000,000 so they NEVER collide with clinical IDs
    df_synth["patient_id"] = range(1000000, 1000000 + len(df_synth))

    # Clinical patients get standard IDs starting from 1
    df_clin["patient_id"] = range(1, 1 + len(df_clin))

    base_date_synth = datetime(2015, 1, 1)
    # Randomly distribute events across an 8-year span
    random_days = np.random.randint(0, 365 * 8, size=len(df_synth))
    df_synth["event_timestamp"] = [
        base_date_synth + timedelta(days=int(d)) for d in random_days
    ]
    df_synth["created_timestamp"] = pd.Timestamp.now()

    # 3. Process Clinical Data (Simulating recent/streaming clinical data: 2024-2026)

    base_date_clin = datetime(2024, 1, 1)
    # Distribute over the last ~2.5 years
    random_days_clin = np.random.randint(0, 850, size=len(df_clin))
    df_clin["event_timestamp"] = [
        base_date_clin + timedelta(days=int(d)) for d in random_days_clin
    ]
    df_clin["created_timestamp"] = pd.Timestamp.now()

    print("Saving to Parquet for Feast ingestion...")
    # Feast requires timezone-aware or strict UTC datetime formats in parquet
    df_synth["event_timestamp"] = pd.to_datetime(df_synth["event_timestamp"], utc=True)
    df_synth["created_timestamp"] = pd.to_datetime(
        df_synth["created_timestamp"], utc=True
    )
    df_clin["event_timestamp"] = pd.to_datetime(df_clin["event_timestamp"], utc=True)
    df_clin["created_timestamp"] = pd.to_datetime(
        df_clin["created_timestamp"], utc=True
    )

    df_synth.to_parquet(f"{output_dir}/synthetic_features.parquet", index=False)
    df_clin.to_parquet(f"{output_dir}/clinical_features.parquet", index=False)
    print("Ingestion complete.")


if __name__ == "__main__":

    ingest_and_timestamp_data(
        "data/raw/train.csv", "data/raw/heart.csv", "data/processed"
    )
    pass
