from fastapi.testclient import TestClient
from src.app import app

# Create a test client that simulates requests to our FastAPI app
client = TestClient(app)

def test_health_check():
    """
    Test that the /health endpoint returns a 200 OK status 
    and the expected JSON structure.
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_validation_error():
    """
    Test that the API correctly rejects a payload that is missing required fields.
    This proves our Pydantic schema validation is working.
    """
    # Payload missing 'age' and 'sex'
    bad_payload = {
        "patient_id": "TEST-123",
        "cp": 1,
        "trestbps": 120.0
    }
    
    response = client.post("/predict", json=bad_payload)
    
    # 422 is the HTTP status code for "Unprocessable Entity" (Validation Error)
    assert response.status_code == 422