"""Phase 2: Warm-start cross-validation on clinical data."""

import os
import tempfile
import pandas as pd
import yaml
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import mlflow
import mlflow.pyfunc
from feast import FeatureStore

# Ensure MLflow tracking URI is set to match Phase 1
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart_disease_finetune")


class StackedEnsembleWrapper(mlflow.pyfunc.PythonModel):
    """
    A Custom PyFunc model that loads all three fine-tuned algorithms
    and returns a soft-voting ensemble prediction. This abstracts the
    complexity away from the FastAPI serving layer.
    """

    def load_context(self, context):
        """Loads the native models from the saved artifacts."""
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model_path"])

        self.lgb_model = lgb.Booster(model_file=context.artifacts["lgb_model_path"])

        self.cb_model = cb.CatBoostClassifier()
        self.cb_model.load_model(context.artifacts["cb_model_path"])

    def predict(self, context, model_input):
        """Generates predictions from all 3 models and averages them."""
        # XGBoost requires DMatrix
        dtest = xgb.DMatrix(model_input)
        xgb_preds = self.xgb_model.predict(dtest)

        # LightGBM and CatBoost accept DataFrames directly
        lgb_preds = self.lgb_model.predict(model_input)
        cb_preds = self.cb_model.predict_proba(model_input)[
            :, 1
        ]  # Get probability of class 1

        # Soft Voting / Averaging
        ensemble_preds = (xgb_preds + lgb_preds + cb_preds) / 3.0
        return ensemble_preds


def fetch_clinical_data(store: FeatureStore) -> pd.DataFrame:
    print("Fetching small clinical dataset from Feast...")
    try:
        df = pd.read_parquet("data/processed/clinical_features.parquet")
    except FileNotFoundError:
        raise FileNotFoundError("Clinical parquet missing. Run ingestion.py first.")
    return df


def fine_tune_and_package():
    """
    Pulls the Phase 1 Base Models, warm-starts them on clinical data,
    and packages them into a production ensemble artifact.
    """
    # 1. Fetch Clinical Data
    store = FeatureStore(repo_path="feature_store/")
    df = fetch_clinical_data(store)

    target_col = "Heart Disease"  # Assuming the target column is named 'Heart Disease' in the clinical dataset
    features = [
        col
        for col in df.columns
        if col not in [target_col, "patient_id", "event_timestamp", "created_timestamp"]
    ]

    X_clin = df[features]
    y_clin = df[target_col]

    print("Loading Base Models from MLflow Registry...")
    # In production, you'd fetch by "Alias" or "Version".
    # Here we fetch the latest models registered in train_base.py
    xgb_base = mlflow.xgboost.load_model("models:/heart_disease_xgb_base/latest")
    lgb_base = mlflow.lightgbm.load_model("models:/heart_disease_lgb_base/latest")
    cb_base = mlflow.catboost.load_model("models:/heart_disease_cb_base/latest")

    with open("config/model_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Drastically reduce learning rate and constrain tree structure
    xgb_ft_params = config["fine_tuning"]["xgb_params"]
    lgb_ft_params = config["fine_tuning"]["lgb_params"]
    cb_ft_params = config["fine_tuning"]["catboost_params"]
    num_boost_round = config["fine_tuning"]["num_boost_round"]

    with mlflow.start_run(run_name="clinical_ensemble_finetuning"):
        print("Executing Warm Start Fine-Tuning on Clinical Data (N=270)...")

        dtrain_xgb = xgb.DMatrix(X_clin, label=y_clin)
        xgb_finetuned = xgb.train(
            xgb_ft_params,
            dtrain_xgb,
            num_boost_round=num_boost_round,
            xgb_model=xgb_base,  # WARM START
        )

        dtrain_lgb = lgb.Dataset(X_clin, label=y_clin)
        lgb_finetuned = lgb.train(
            lgb_ft_params,
            dtrain_lgb,
            num_boost_round=num_boost_round,
            init_model=lgb_base,  # WARM START
        )

        cb_finetuned = cb.CatBoostClassifier(**cb_ft_params)
        cb_finetuned.fit(X_clin, y_clin, init_model=cb_base)  # WARM START

        print("Packaging models into Custom PyFunc Ensemble...")

        # Save models to temporary files to package as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            xgb_path = os.path.join(tmpdir, "xgb_model.json")
            lgb_path = os.path.join(tmpdir, "lgb_model.txt")
            cb_path = os.path.join(tmpdir, "cb_model.cbm")

            xgb_finetuned.save_model(xgb_path)
            lgb_finetuned.save_model(lgb_path)
            cb_finetuned.save_model(cb_path)

            artifacts = {
                "xgb_model_path": xgb_path,
                "lgb_model_path": lgb_path,
                "cb_model_path": cb_path,
            }

            # Log the unified PyFunc model
            mlflow.pyfunc.log_model(
                artifact_path="production_ensemble",
                python_model=StackedEnsembleWrapper(),
                artifacts=artifacts,
                registered_model_name="heart_disease_production",
            )

        print(f"Production Ensemble registered! Ready for FastAPI deployment.")


if __name__ == "__main__":
    fine_tune_and_package()
