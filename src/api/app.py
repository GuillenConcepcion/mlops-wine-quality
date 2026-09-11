"""
FastAPI Serving Application.
Production-ready RESTful service for Wine Quality prediction, XAI explanations,
and statistical data drift monitoring.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from src.config import config
from src.api.schemas import (
    WineFeatures,
    PredictionResponse,
    ExplanationResponse,
    HealthCheckResponse,
    FeatureImpact,
)
from src.api.dependencies import ModelArtifactsManager, get_artifacts_manager

app = FastAPI(
    title="Wine Quality Prediction & MLOps API",
    description="""
    ## Senior MLOps & XAI Production Endpoint
    Predicts sensory wine quality (Cortez et al., UCI Dataset 186) with:
    - **Multiclass Balanced Architecture:** Evaluates full quality spectrum (scores 3 to 9).
    - **Unified Enology:** Handles both Red and White wine compositions.
    - **Explainable AI (XAI):** Real-time instance-level SHAP attributions.
    - **Data Drift Detection:** Automated KS-Test and PSI monitoring.

    **Author:** Guillén Concepción (Senior Data Scientist & MLOps Engineer)
    """,
    version="0.1.0",
)


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "Wine Quality MLOps & XAI System",
        "author": "Guillén Concepción",
        "documentation": "/docs",
        "healthcheck": "/health",
        "version": "0.1.0",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check(artifacts: ModelArtifactsManager = Depends(get_artifacts_manager)):
    """Verifies service liveness and loaded model artifacts."""
    model_loaded = artifacts.pipeline is not None
    explainer_loaded = artifacts.explainer is not None

    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifact is not loaded in memory.",
        )

    clf = artifacts.pipeline.named_steps["classifier"]
    classes = [int(c) for c in getattr(clf, "classes_", [])]

    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_loaded,
        explainer_loaded=explainer_loaded,
        target_mode=config.target_mode.value,
        available_classes=classes,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_quality(
    wine: WineFeatures,
    artifacts: ModelArtifactsManager = Depends(get_artifacts_manager),
):
    """
    Predicts the discrete sensory quality score for a given wine profile.
    Returns the predicted score and multiclass probabilities.
    """
    if artifacts.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )

    # Convert input to DataFrame preserving lowercase feature names as seen at fit time
    input_data = {k.lower(): v for k, v in wine.model_dump().items()}
    input_df = pd.DataFrame([input_data])

    # Predict class and probabilities
    pred_class = int(artifacts.pipeline.predict(input_df)[0])
    probabilities = {}
    if hasattr(artifacts.pipeline, "predict_proba"):
        raw_probs = artifacts.pipeline.predict_proba(input_df)[0]
        clf = artifacts.pipeline.named_steps["classifier"]
        for cls_label, prob in zip(clf.classes_, raw_probs):
            probabilities[int(cls_label)] = round(float(prob), 4)

    return PredictionResponse(
        predicted_quality=pred_class,
        probabilities=probabilities,
        model_architecture="ExtraTreesClassifier (Tuned with Optuna, Balanced Loss)",
        target_mode=config.target_mode.value,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["XAI Explainability"])
def explain_prediction(
    wine: WineFeatures,
    top_k: int = 3,
    artifacts: ModelArtifactsManager = Depends(get_artifacts_manager),
):
    """
    Predicts wine quality and explains the prediction with SHAP values.
    Returns the top-k most influential physicochemical features driving the outcome.
    """
    if artifacts.pipeline is None or artifacts.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline or Explainer artifact is not loaded.",
        )

    input_data = {k.lower(): v for k, v in wine.model_dump().items()}
    input_df = pd.DataFrame([input_data])

    # Transform input through feature engineering and scaling steps
    fe_step = artifacts.pipeline.named_steps["feature_engineer"]
    scaler_step = artifacts.pipeline.named_steps["scaler"]
    clf_step = artifacts.pipeline.named_steps["classifier"]

    transformed_df = fe_step.transform(input_df)
    scaled_array = scaler_step.transform(transformed_df)

    # Predict
    pred_class = int(clf_step.predict(scaled_array)[0])

    # Probabilities
    probabilities = {}
    if hasattr(clf_step, "predict_proba"):
        raw_probs = clf_step.predict_proba(scaled_array)[0]
        for cls_label, prob in zip(clf_step.classes_, raw_probs):
            probabilities[int(cls_label)] = round(float(prob), 4)

    # Extract Local SHAP Explanation
    top_contributors = artifacts.explainer.explain_instance(
        scaled_array,
        predicted_class=pred_class,
        top_k=top_k,
    )

    feature_impacts = [
        FeatureImpact(
            feature=c["feature"],
            value=c["value"],
            shap_impact=c["shap_impact"],
            direction=c["direction"],
        )
        for c in top_contributors
    ]

    summary_text = (
        f"Prediction of Quality score {pred_class} is primarily driven by: "
        + ", ".join([f"{f.feature} ({f.direction} impact: {f.shap_impact:+.3f})" for f in feature_impacts])
    )

    return ExplanationResponse(
        predicted_quality=pred_class,
        probabilities=probabilities,
        top_contributing_features=feature_impacts,
        interpretation=summary_text,
    )


@app.post("/drift/evaluate", tags=["Monitoring & Drift"])
def evaluate_batch_drift(
    batch: List[WineFeatures],
    artifacts: ModelArtifactsManager = Depends(get_artifacts_manager),
):
    """
    Evaluates covariate data drift on an inference batch against training baseline.
    Computes KS-Test and PSI statistics.
    """
    if artifacts.drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detector reference baseline is not loaded.",
        )

    if len(batch) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum 10 samples required in batch for statistical drift tests.",
        )

    batch_df = pd.DataFrame([{k.lower(): v for k, v in item.model_dump().items()} for item in batch])
    drift_report = artifacts.drift_detector.evaluate_drift(batch_df)

    return JSONResponse(content=drift_report)
