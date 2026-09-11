"""
Pydantic v2 schemas for API contracts and physical boundary validation.
Ensures rigorous input guardrails (rejecting biologically/chemically impossible values).
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class WineFeatures(BaseModel):
    """
    Physical-chemical properties of wine input.
    Boundaries are established based on enological realities (Tukey bounds on UCI dataset).
    """
    fixed_acidity: float = Field(..., ge=3.0, le=16.0, description="Tartaric acid concentration (g/dm³)")
    volatile_acidity: float = Field(..., ge=0.08, le=1.60, description="Acetic acid concentration (g/dm³)")
    citric_acid: float = Field(..., ge=0.0, le=1.70, description="Citric acid concentration (g/dm³)")
    residual_sugar: float = Field(..., ge=0.4, le=66.0, description="Residual sugar content (g/dm³)")
    chlorides: float = Field(..., ge=0.009, le=0.65, description="Sodium chloride concentration (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=1.0, le=290.0, description="Free SO₂ concentration (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=6.0, le=440.0, description="Total SO₂ concentration (mg/dm³)")
    density: float = Field(..., ge=0.980, le=1.040, description="Density (g/cm³)")
    pH: float = Field(..., ge=2.70, le=4.10, description="pH level of the wine")
    sulphates: float = Field(..., ge=0.20, le=2.00, description="Potassium sulphate (g/dm³)")
    alcohol: float = Field(..., ge=8.0, le=15.0, description="Alcohol content (% vol)")
    is_red: Literal[0, 1] = Field(..., description="Wine type indicator: 1 for Red wine, 0 for White wine")

    @field_validator("total_sulfur_dioxide")
    @classmethod
    def validate_sulfur_dioxide(cls, v: float, info) -> float:
        """Physical integrity check: Total SO₂ must be >= Free SO₂."""
        free_so2 = info.data.get("free_sulfur_dioxide")
        if free_so2 is not None and v < free_so2:
            raise ValueError(f"total_sulfur_dioxide ({v}) cannot be lower than free_sulfur_dioxide ({free_so2})")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class FeatureImpact(BaseModel):
    """SHAP marginal attribution for a single feature."""
    feature: str
    value: float
    shap_impact: float
    direction: Literal["positive", "negative"]


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""
    predicted_quality: int
    probabilities: Dict[int, float]
    model_architecture: str
    target_mode: str
    status: str = "success"


class ExplanationResponse(BaseModel):
    """Response returned by the /explain endpoint with XAI factors."""
    predicted_quality: int
    probabilities: Dict[int, float]
    top_contributing_features: List[FeatureImpact]
    interpretation: str
    status: str = "success"


class HealthCheckResponse(BaseModel):
    """System health status and loaded model metadata."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
    target_mode: str
    available_classes: List[int]
    app_version: str = "0.1.0"
