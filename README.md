# Clinical MLOps: End-to-End Heart Disease Diagnostic System

A production-grade, continuous training MLOps pipeline designed to predict cardiovascular disease risk. This project moves beyond static Jupyter notebooks by implementing a full lifecycle architecture: time-travel feature engineering, tabular transfer learning, clinical cost-matrix optimization, real-time drift monitoring, and an automated CI/CD/CD deployment pipeline.

🔴 **Live Interactive Dashboard** (Deployed via Render) - [Click Here](https://samuel-k-anim.github.io/testing-base/)

🟢 **Live API Health Endpoint** | **API Docs**

---

## Architectural Overview

This system is decoupled into specific operational phases to simulate a real-world clinical data infrastructure:

- **Feature Store (Feast)**: Eliminates data leakage through point-in-time joins, guaranteeing that historical clinical features are strictly isolated from future target variables.

- **Phase 1: Pre-training (High Compute)**: A base ensemble (XGBoost, LightGBM, CatBoost) is trained on 630,000 synthetic patient records to learn macro-distributions. Tracked via MLflow.

- **Phase 2: Fine-Tuning (Warm-Start)**: Base models are loaded from the registry and warm-started on real, low-volume clinical data ($N=270$) using aggressive L1/L2 regularization and shrinkage to prevent memorization.

- **Custom PyFunc Packaging**: The heterogeneous ensemble is packaged into a unified MLflow PyFunc artifact for simplified downstream API consumption.

- **Clinical Threshold Optimization ($t^*$)**: Rejects the default $0.5$ classification threshold. Applies a custom Cost-Benefit Matrix (heavily penalizing false negatives) to compute the mathematically optimal decision boundary.

- **Serving & Monitoring**: Deployed via a FastAPI Docker container.

---

## Continuous Deployment (CI/CD/CD) Architecture

The system utilizes a modern GitOps pipeline for automated validation and deployment:

- **Validation (CI)**: GitHub Actions triggers code formatting inspections (Black) and model validation unit testing (pytest).

- **Containerization**: App files, configurations, and trained MLflow artifact stores are wrapped inside an immutable `python:3.11-slim` Docker container.

- **Continuous Deployment (CD)**: GitHub Actions securely authenticates to Docker Hub using encrypted Repository Secrets to push the image. Render listens for image updates, clears build caches, executes cross-platform asset translation patches, and provisions the public FastAPI endpoint.

### GitHub Actions Workflow

```yaml
name: MLOps Production Engine
on:
  push:
    branches: [ main ]

jobs:
  lint_and_test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install Testing Requirements
        run: |
          pip install -r requirements.txt
          pip install pytest black
      - name: Run Code Style Validation
        run: python -m black src/ --check
      - name: Execute Validation Testing Suite
        run: python -m pytest

  docker_build:
    needs: lint_and_test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and Push Monolithic Container Image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/heart-disease-api:latest
          outputs: type=image,push=true,allow=monolithic=true
```

---

## Production Engineering & Incident Log

Transitioning from local training scripts to a cloud-native deployment required resolving several systemic environment, dependency, and cross-platform failures. Documented below are the key root-cause analyses and applied resolutions.

### 1. Cross-Platform Path Boundary Exception (Windows ➡️ Linux)

**Symptom**: Render deployment crashed. Model failed to open with `XGBoostError: Opening ... failed: No such file or directory`. Logs indicated invalid path structures (e.g., `/app/mlruns/.../artifacts\xgb_model.json`).

**Root Cause**: Training on a Windows host saves file location metadata strings with backslashes (`\`). The Linux host (Render) cannot interpret Windows backslashes as valid directory breaks. These paths were hardcoded into the internal SQLite database (`mlflow.db`) and metadata logs (`meta.yaml`).

**Resolution**: Implemented an automated cross-platform translation patch inside `src/app.py`. A startup sanitizer detects the Render environment, queries the tracking assets, and dynamically rewrites path formats into Linux-compliant forward slashes (`/`).

### 2. Python Code Deserialization Crash

**Symptom**: Cloud deployment crashed immediately with `TypeError: code expected at most 16 arguments, got 18`.

**Root Cause**: Interpreter mismatch. Local model training was serialized using Python 3.11 (cloudpickle). The production container image was initialized from an older `python:3.10-slim` base image. Python 3.11 introduced new code object parameters that 3.10 cannot deserialize.

**Resolution**: Synced runtime environments by updating the primary container configuration: `FROM python:3.11-slim`.

### 3. Missing Binary Artifacts in CI/CD Build
**Symptom**: Cloud-managed Docker compilation failed at `COPY mlruns/ mlruns/` indicating "not found".

**Root Cause**: The `.gitignore` file correctly blocked `mlruns/` and `mlflow.db` to prevent tracking massive logs. Consequently, the GitHub Actions runner checked out an empty repository context lacking model binary artifacts.

**Resolution**: Modified repository tracking behaviors using Git's force-override to upload verified local training outputs specifically for the deployment state: `git add -f mlruns/` & `git add -f mlflow.db`.

### 4. Dependency Mismatch: FastAPI vs. httpx

**Symptom**: Testing suite crashed with `TypeError: Client.init() got an unexpected keyword argument 'app'`.

**Root Cause**: Breaking updates to the HTTP transport library `httpx` (>=0.26.0) stripped out the legacy `app` argument signature explicitly expected by older versions of FastAPI (0.103.2).

**Resolution**: Pinned dependencies in `requirements.txt` to enforce structural stability: `fastapi==0.103.2` and `httpx==0.25.2`.

### 5. Pytest Module Resolution Failure

**Symptom**: Pytest failed to boot, throwing `ModuleNotFoundError: No module named 'src'`.

**Root Cause**: Executing the bare `pytest` binary does not implicitly inject the top-level project root path into Python's execution runtime array (`sys.path`).

**Resolution**: Configured an explicit test layer by creating a `pyproject.toml` containing `[tool.pytest.ini_options] pythonpath = ["."]`.

### 6. XGBoost Categorical Target Mapping

**Symptom**: Optuna tuning crashed during the first trial with `ValueError: Invalid classes inferred... Expected: [0 1], got ['Absence' 'Presence']`.

**Root Cause**: The target label evaluated to raw text. Modern XGBoost (>=2.0.0) strictly requires binary prediction outcomes as numeric integer arrays.

**Resolution**: Explicitly mapped categorical features to binary keys at ingestion/training: `y = y.map({'Absence': 0, 'Presence': 1})`.

### 7. Legacy Package Utility Deprecation

**Symptom**: Training loop crashed with `ModuleNotFoundError: No module named 'pkg_resources'`.

**Root Cause**: `pkg_resources` was removed from native initialization paths in newer environments, causing older MLflow versions (2.7.1) to crash on Python 3.11.

**Resolution**: Explicitly restored the interface via pinned installation tools: `pip install "setuptools<70.0.0"`.

### 8. Enterprise OS Security Intercepts

**Symptom**: Running local formatters (black) or Conda environments failed due to Application Control policies and restricted execution rules.

**Root Cause**: Windows OS restricted security layers and AppLocker intercepted unverified `.exe` binaries and powershell scripts.

**Resolution**: Bypassed binary restrictions by routing execution natively through the Python interpreter (`python -m black src/`) and altering execution privileges via administrative tokens (`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`).

---

## Repository Structure

```
heart-disease-mlops/
├── .github/workflows/       # CI/CD (Linting, Pytest, Docker Build)
├── config/                  # Central YAML configurations (Cost Matrix, Hyperparams)
├── data/                    # Local state (git-ignored for HIPAA compliance)
│   ├── raw/                 # Drop Kaggle CSVs here
│   └── processed/           # Parquet files generated for Feast
├── feature_store/           # Feast registry and schema definitions
├── src/
│   ├── ingestion.py         # Standardizes schema and generates synthetic timestamps
│   ├── train_base.py        # Phase 1: Pre-training on synthetic data
│   ├── fine_tune.py         # Phase 2: Warm-start tabular transfer learning
│   ├── evaluate.py          # Cost-matrix threshold optimization
│   ├── app.py               # FastAPI inference service with cross-platform patching
│   └── monitor.py           # Evidently AI drift detection
├── tests/                   # Pytest suite
├── Dockerfile               # OCI-compliant container definition
├── pyproject.toml           # Pytest path routing configuration
└── requirements.txt         # Pinned dependency graph
```

---

## Local Installation & Execution

### 1. Environment Setup

```bash
git clone https://github.com/Samuel-K-Anim/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Ingestion & Feature Store

**Note**: Raw data is not hosted in this repository.

1. Download the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) and the Synthetic Dataset.
2. Place them in `data/raw/` as `heart.csv` and `train.csv`.
3. Generate point-in-time Parquet files and apply the Feast schema:

```bash
python src/ingestion.py
cd feature_store
feast apply
cd ..
```

### 3. Pipeline Execution

Execute sequentially. MLflow tracks artifacts locally in `mlruns/` and `mlflow.db`.

```bash
python src/train_base.py      # Pre-train base models
python src/fine_tune.py       # Warm-start on clinical fold
python src/evaluate.py        # Compute optimal threshold t*
```

### 4. API Deployment

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Roadmap: Version 2.0 (Calibration)

### Identified Limitation in V1

Gradient Boosted Trees (GBDTs) do not output true empirical probabilities; their margins are transformed via a sigmoid function, clustering predictions near the median. Applying an aggressive asymmetric cost matrix (e.g., 10:1 penalty for False Negatives) to uncalibrated outputs mathematically forces the optimal threshold ($t^*$) toward degenerate states (e.g., $0.02$). While mathematically correct given the cost function, this induces "Alarm Fatigue" in clinical settings.

### V2 Architecture Upgrade

The next iteration will implement **Platt Scaling** (Logistic Calibration). The MLflow PyFunc wrapper will be updated to embed a LogisticRegression layer trained on Out-Of-Fold (OOF) predictions during Phase 2. This will map the raw ensemble scores to calibrated empirical probabilities prior to cost-matrix evaluation, stabilizing the clinical decision boundary.