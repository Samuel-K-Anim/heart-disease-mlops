"""FastAPI production deployment script."""

import os
import time
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import yaml

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
app = FastAPI(
    title="Heart Disease Inference API",
    description="Production endpoint for clinical heart disease prediction. Utilizes optimal threshold (t*).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for local testing)
)
# Global variables to hold model and threshold
MODEL = None
OPTIMAL_THRESHOLD = None


class PatientPayload(BaseModel):
    """
    Expected JSON payload from the hospital's frontend or EMR system.
    """

    patient_id: str
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ==========================================
# STARTUP EVENT: LOAD MLFLOW ARTIFACTS
# ==========================================
@app.on_event("startup")
def load_production_assets():
    """
    Loads the MLflow model and the optimized threshold into memory on server start.
    In a real production environment, this would pull from a model registry like AWS S3.
    """
    global MODEL, OPTIMAL_THRESHOLD

    # Ensure MLflow tracking URI is set
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print("Initializing production microservice...")

    # Read the config
    with open("config/model_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    MODEL_URI = config["serving"]["model_uri"]
    FALLBACK_THRESHOLD = config["serving"]["fallback_threshold"]

    try:
        # Load the unified PyFunc ensemble
        MODEL = mlflow.pyfunc.load_model(MODEL_URI)
        print("Ensemble model loaded successfully.")
    except Exception as e:
        print(
            "CRITICAL: Failed to load model. Ensure fine_tune.py registered the model."
        )
        raise RuntimeError("Model loading failed.") from e

    try:
        # Dynamically fetch the optimal threshold (t*) computed by evaluate.py
        # mlflow.search_runs may return either a pandas.DataFrame (older API)
        # or a list of Run objects (mlflow 2.x). Handle both gracefully.
        runs: Any = mlflow.search_runs(search_all_experiments=True)

        # Case A: DataFrame-like result
        if hasattr(runs, "empty") and hasattr(runs, "columns"):
            df_runs = runs
            if df_runs.empty or "metrics.optimal_threshold" not in df_runs.columns:
                raise ValueError(
                    "Metric 'optimal_threshold' not found in any MLflow run."
                )

            valid_runs = df_runs.dropna(subset=["metrics.optimal_threshold"])
            if valid_runs.empty:
                raise ValueError(
                    "Metric exists in tracking server but all values are NaN."
                )

            valid_runs = valid_runs.sort_values(by="start_time", ascending=False)
            OPTIMAL_THRESHOLD = valid_runs.iloc[0]["metrics.optimal_threshold"]

        else:
            # Case B: list of Run objects
            if not isinstance(runs, list) or len(runs) == 0:
                raise ValueError("No evaluation runs found.")

            # Filter runs that have the metric present and not None
            valid_runs = [
                r
                for r in runs
                if getattr(r, "data", None)
                and getattr(r.data, "metrics", None)
                and "optimal_threshold" in r.data.metrics
                and r.data.metrics["optimal_threshold"] is not None
            ]

            if not valid_runs:
                raise ValueError(
                    "Metric 'optimal_threshold' not found in any MLflow run."
                )

            # Sort by start_time if available, newest first
            def _start_time(run):
                return float(getattr(getattr(run, "info", None), "start_time", 0) or 0)

            valid_runs = sorted(valid_runs, key=_start_time, reverse=True)
            OPTIMAL_THRESHOLD = valid_runs[0].data.metrics["optimal_threshold"]

        print(f"✅ Optimal clinical threshold loaded: t* = {OPTIMAL_THRESHOLD:.3f}")

    except Exception as e:
        print(
            "WARNING: Failed to load optimal threshold from MLflow. Falling back to default t=0.50"
        )
        OPTIMAL_THRESHOLD = FALLBACK_THRESHOLD


# ==========================================
# INFERENCE ENDPOINT
# ==========================================
@app.post("/predict")
def predict_heart_disease(payload: PatientPayload):
    """
    Accepts patient data, runs it through the ensemble, applies the clinical threshold,
    and returns a risk assessment.
    """
    start_time = time.time()

    # 1. Convert Payload to DataFrame (model expects a DataFrame)
    # Exclude patient_id as it is not a predictive feature
    input_dict = payload.dict()
    patient_id = input_dict.pop("patient_id")
    df_input = pd.DataFrame([input_dict])

    # 2. Execute Inference
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Service unavailable."
        )

    try:
        # The PyFunc wrapper handles the XGB/LGBM/CatBoost averaging automatically
        probability = MODEL.predict(df_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

    # 3. Apply the Optimal Threshold ($t^*$)
    is_high_risk = bool(probability >= OPTIMAL_THRESHOLD)

    # 4. Construct Response Payload
    inference_time_ms = (time.time() - start_time) * 1000

    return {
        "patient_id": patient_id,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "risk_assessment": {
            "is_high_risk": is_high_risk,
            "probability_score": float(probability),
            "applied_threshold": OPTIMAL_THRESHOLD,
            "clinical_guidance": (
                "Recommend immediate cardiology consult."
                if is_high_risk
                else "Standard observation."
            ),
        },
        "telemetry": {"inference_time_ms": round(inference_time_ms, 2)},
    }


# ==========================================
# HEALTHCHECK ENDPOINT
# ==========================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "active_threshold": OPTIMAL_THRESHOLD,
    }
