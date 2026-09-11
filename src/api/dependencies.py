"""
Dependency injection and artifact loader for FastAPI serving.
Employs Singleton pattern to load serialized models and baseline reference into memory once.
"""

import logging
from typing import Optional
import joblib
import pandas as pd

from src.config import config, MODELS_DIR
from src.monitoring.drift import DataDriftDetector

logger = logging.getLogger(__name__)


class ModelArtifactsManager:
    """Manages loaded artifacts in memory for high-throughput inference."""
    _instance: Optional["ModelArtifactsManager"] = None

    def __init__(self):
        self.pipeline = None
        self.explainer = None
        self.reference_data = None
        self.drift_detector = None
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "ModelArtifactsManager":
        if cls._instance is None:
            cls._instance = ModelArtifactsManager()
        return cls._instance

    def load_artifacts(self):
        pipeline_path = MODELS_DIR / config.pipeline_artifact_name
        explainer_path = MODELS_DIR / config.explainer_artifact_name
        reference_path = MODELS_DIR / config.reference_data_name

        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            logger.info(f"Loaded production pipeline from {pipeline_path}")
        else:
            logger.warning(f"Pipeline artifact not found at {pipeline_path}")

        if explainer_path.exists():
            self.explainer = joblib.load(explainer_path)
            logger.info(f"Loaded SHAP explainer from {explainer_path}")
        else:
            logger.warning(f"Explainer artifact not found at {explainer_path}")

        if reference_path.exists():
            self.reference_data = pd.read_parquet(reference_path)
            self.drift_detector = DataDriftDetector(self.reference_data)
            logger.info(f"Loaded training reference baseline from {reference_path}")
        else:
            logger.warning(f"Reference data not found at {reference_path}")


def get_artifacts_manager() -> ModelArtifactsManager:
    return ModelArtifactsManager.get_instance()
