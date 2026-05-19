# Slim version reduces the image size significantly

FROM python:3.10-slim

# Set environment variables to optimize Python execution

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Set the working directory in the container

WORKDIR /app

# Install system dependencies required for building C++ extensions (like LightGBM/CatBoost)

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container

COPY requirements.txt .

# Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# In a real CI/CD pipeline, you would pull the model from AWS S3.

# For this portfolio, we copy the local MLflow DB and mlruns folder so the API can load the model.

COPY src/ src/
COPY mlflow.db .
COPY config/ config/ 
COPY mlruns/ mlruns/

# Expose the port FastAPI runs on

EXPOSE 8000

# Create a non-root user for security (Docker best practice)

RUN useradd -m apiuser && chown -R apiuser /app
USER apiuser

# Command to run the FastAPI application using Uvicorn

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
