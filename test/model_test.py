from fastapi.testclient import TestClient
from unittest.mock import patch
from api import app  # Import your FastAPI app

client = TestClient(app)

def test_get_available_models():
    # Mock data to be returned by the model manager
    models = ["logistic_regression","decision_tree"]

    # Use patch to mock the get_available_models method
    response = client.post("/get_data")
    assert response.status_code == 200
    assert response.json() == {"model": models}

# Replace "path.to.your.model_manager" with the actual import path of your model manager
