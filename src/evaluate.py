"""Threshold optimization and cost matrix calculation."""

import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from feast import FeatureStore

# Ensure MLflow tracking URI matches the previous phases
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart_disease_evaluation")

# ==========================================
# CLINICAL COST MATRIX CONFIGURATION
# ==========================================

# Read the central configuration file
with open("config/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Pull the variables dynamically
COST_FN = config["cost_matrix"]["cost_false_negative"]
# FN (False Negative) = Patient is sick but sent home. High risk of fatality/liability.
COST_FP = config["cost_matrix"]["cost_false_positive"]
# FP (False Positive) = Patient is healthy but flagged. Cost of ECG/Angiogram + stress.
COST_TP = config["cost_matrix"]["cost_true_positive"]
# TP (True Positive) = Correctly identified sick patient. Usually 0 cost, or negative if representing revenue/savings.
COST_TN = config["cost_matrix"]["cost_true_negative"]


# These values are usually determined in collaboration with clinical stakeholders, hospital administration, and financial analysts to reflect real-world costs and risks. For example:
# COST_FN = 10000 (cost of a missed diagnosis leading to a fatality)


def fetch_validation_data() -> pd.DataFrame:
    """
    Fetches the clinical dataset. In a true production environment,
    this would be a strict hold-out test set from Feast.
    """
    print("Fetching clinical dataset for evaluation...")
    try:
        df = pd.read_parquet("data/processed/clinical_features.parquet")
    except FileNotFoundError:
        raise FileNotFoundError("Parquet missing. Run ingestion.py first.")
    return df


def optimize_threshold():
    """
    Evaluates the production ensemble and calculates the cost-optimal threshold (t*).
    """
    df = fetch_validation_data()
    y_true = df["Heart Disease"].values
    X_val = df.drop(
        columns=["Heart Disease" , "patient_id", "event_timestamp", "created_timestamp"]
    )

    print("Loading Production Ensemble from MLflow...")
    # Load the unified PyFunc ensemble created in fine_tune.py
    # In production, use "models:/heart_disease_production/Production" alias
    try:
        model = mlflow.pyfunc.load_model("models:/heart_disease_production/latest")
    except Exception as e:
        print("Could not load model. Ensure fine_tune.py was run to register it.")
        raise e

    print("Generating ensemble probability predictions...")
    # The PyFunc model automatically averages XGB, LGBM, and CatBoost internally
    y_probs = model.predict(X_val)

    # Calculate baseline PR-AUC
    pr_auc = average_precision_score(y_true, y_probs)
    print(f"Ensemble PR-AUC: {pr_auc:.4f}")

    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []

    for t in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred_t = (y_probs >= t).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()

        # Calculate total clinical cost
        total_cost = (fn * COST_FN) + (fp * COST_FP) + (tp * COST_TP) + (tn * COST_TN)
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    optimal_t = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]

    print(f"\n--- THRESHOLD OPTIMIZATION RESULTS ---")
    print(
        f"Default 0.50 Threshold Cost:  {costs[np.abs(thresholds - 0.50).argmin()]:.2f}"
    )
    print(f"Optimal t* Threshold Cost:    {min_cost:.2f} (at t* = {optimal_t:.3f})")
    print(
        f"Cost Savings vs Default:      {costs[np.abs(thresholds - 0.50).argmin()] - min_cost:.2f}"
    )

    # Calculate final metrics at optimal threshold
    final_preds = (y_probs >= optimal_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()

    with mlflow.start_run(run_name="threshold_optimization"):
        # Log metrics
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("optimal_threshold", optimal_t)
        mlflow.log_metric("minimum_clinical_cost", min_cost)
        mlflow.log_metric("false_negatives_at_t_star", fn)
        mlflow.log_metric("false_positives_at_t_star", fp)

        # Generate a Cost Curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            thresholds, costs, label="Total Clinical Cost", color="red", linewidth=2
        )
        plt.axvline(
            x=optimal_t,
            color="blue",
            linestyle="--",
            label=f"Optimal t* = {optimal_t:.3f}",
        )
        plt.axvline(x=0.5, color="gray", linestyle=":", label="Default t = 0.5")
        plt.title("Clinical Cost vs. Decision Threshold")
        plt.xlabel("Probability Decision Threshold")
        plt.ylabel("Total Cost (Units)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to MLflow
        plot_path = "cost_curve.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up local file

        print("Optimization complete. Metrics and Cost Curve logged to MLflow.")


if __name__ == "__main__":
    optimize_threshold()
