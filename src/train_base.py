"""Phase 1: Optuna tuning and MLflow logging on synthetic data."""

import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from datetime import datetime
from feast import FeatureStore
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure MLflow tracking URI is set (e.g., local sqlite for portfolio)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart_disease_pretrain")


def fetch_synthetic_data(store: FeatureStore) -> pd.DataFrame:
    """
    Retrieves the synthetic dataset from Feast using time-travel logic.
    """
    print("Connecting to Feast Feature Store...")

    # In a real environment, you'd generate an entity dataframe containing
    # patient_ids and timestamps for when you want to retrieve the features.
    # For this baseline, we pull the raw parquet directly as a dataframe
    # to bypass Feast's point-in-time constraints for the initial bulk load.

    # Note: Feast is better utilized in fine_tune.py for streaming lookups.
    # Here, we simulate fetching the historical warehouse dump.
    try:
        df = pd.read_parquet("data/processed/synthetic_features.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Run src/ingestion.py first to generate the parquet files."
        )

    print(f"Retrieved {len(df)} synthetic records.")
    return df


def objective(trial, X, y):
    """
    Optuna objective function for tuning XGBoost parameters.
    """
    param = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        # Enable GPU acceleration if available
        "device": "cuda" if xgb.core.core.XGBoostError else "cpu",
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eta": trial.suggest_float("eta", 1e-3, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    # Use Stratified K-Fold to maintain class balance
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize XGBoost classifier with Optuna suggested parameters
    clf = xgb.XGBClassifier(**param, use_label_encoder=False)

    # We optimize for PR-AUC (Average Precision) instead of standard ROC-AUC
    # This is critical for clinical datasets where False Positives/Negatives
    # carry drastically different costs.
    scorer = make_scorer(average_precision_score, needs_proba=True)

    scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer, n_jobs=-1)

    return scores.mean()


def train_base_model():
    """
    Executes the Phase 1 pre-training pipeline.
    """
    # 1. Initialize Feast Store (assuming we run from repository root)
    store = FeatureStore(repo_path="feature_store/")

    # 2. Fetch Data
    df = fetch_synthetic_data(store)

    # 3. Preprocess
    # Assuming standard Kaggle target column 'target'
    target_col = "target"
    features = [
        col
        for col in df.columns
        if col not in ["target", "patient_id", "event_timestamp", "created_timestamp"]
    ]

    X = df[features]
    y = df[target_col]

    print("Starting Hyperparameter Tuning with Optuna...")
    study = optuna.create_study(direction="maximize")
    # Limit trials for portfolio execution speed; increase for actual production
    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (PR-AUC): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("Training final base ensemble models...")

    # 1. XGBoost Params (from Optuna)
    xgb_params = trial.params
    xgb_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cuda" if xgb.core.core.XGBoostError else "cpu",
        }
    )

    # 2. LightGBM Params (Robust defaults for synthetic data)
    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "max_depth": 6,
        "verbose": -1,
    }

    # 3. CatBoost Params
    cb_params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "verbose": 0,
    }

    with mlflow.start_run(run_name="synthetic_ensemble_pretrain"):
        # XGBoost
        dtrain_xgb = xgb.DMatrix(X, label=y)
        xgb_base = xgb.train(xgb_params, dtrain_xgb, num_boost_round=300)
        mlflow.xgboost.log_model(
            xgb_base,
            artifact_path="base_xgb",
            registered_model_name="heart_disease_xgb_base",
        )

        # LightGBM
        dtrain_lgb = lgb.Dataset(X, label=y)
        lgb_base = lgb.train(lgb_params, dtrain_lgb, num_boost_round=300)
        mlflow.lightgbm.log_model(
            lgb_base,
            artifact_path="base_lgb",
            registered_model_name="heart_disease_lgb_base",
        )

        # CatBoost
        cb_base = cb.CatBoostClassifier(**cb_params)
        cb_base.fit(X, y)
        mlflow.catboost.log_model(
            cb_base,
            artifact_path="base_cb",
            registered_model_name="heart_disease_cb_base",
        )

        print(
            f"Base ensemble registered in MLflow under run ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    train_base_model()
