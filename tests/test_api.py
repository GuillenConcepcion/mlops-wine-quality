"""
Integration tests for FastAPI endpoints, request contracts, and guardrails.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.dependencies import get_artifacts_manager

client = TestClient(app)


def test_root_endpoint():
    """Verify service root metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["author"] == "Guillén Concepción"


def test_health_endpoint():
    """Verify healthcheck and model liveness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True
    assert len(data["available_classes"]) > 0


def test_predict_endpoint_valid_sample():
    """Verify inference for a realistic wine profile."""
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.36,
        "citric_acid": 0.30,
        "residual_sugar": 1.8,
        "chlorides": 0.075,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 45.0,
        "density": 0.9968,
        "pH": 3.38,
        "sulphates": 0.65,
        "alcohol": 11.2,
        "is_red": 1
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quality" in data
    assert data["predicted_quality"] in [3, 4, 5, 6, 7, 8, 9]
    assert "probabilities" in data
    assert len(data["probabilities"]) > 0


def test_predict_endpoint_out_of_bounds_guardrail():
    """Verify Pydantic rejects physically impossible values (e.g., pH = 12.0)."""
    invalid_payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.36,
        "citric_acid": 0.30,
        "residual_sugar": 1.8,
        "chlorides": 0.075,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 45.0,
        "density": 0.9968,
        "pH": 12.0,  # Invalid: Wine cannot have pH 12
        "sulphates": 0.65,
        "alcohol": 11.2,
        "is_red": 1
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_explain_endpoint_shap_attributions():
    """Verify XAI endpoint returns top contributing features."""
    payload = {
        "fixed_acidity": 8.0,
        "volatile_acidity": 0.28,
        "citric_acid": 0.40,
        "residual_sugar": 6.5,
        "chlorides": 0.038,
        "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 120.0,
        "density": 0.9930,
        "pH": 3.15,
        "sulphates": 0.55,
        "alcohol": 12.5,
        "is_red": 0
    }

    response = client.post("/explain?top_k=3", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) == 3
    for feat in data["top_contributing_features"]:
        assert "feature" in feat
        assert "shap_impact" in feat
        assert "direction" in feat
