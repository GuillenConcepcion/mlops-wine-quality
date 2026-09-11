"""
Configuration module for the Wine Quality MLOps project.
Follows senior-level practices: centralized configuration, typed schemas, and environment-aware settings.
"""

from pathlib import Path
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class TargetMode(str, Enum):
    """Target formulation mode."""
    MULTICLASS = "multiclass"  # Pure multiclass (quality scores 3 to 9)
    SEGMENTED_3CLASS = "segmented_3class"  # Low (3-5), Medium (6), High (7-9)
    BINARY_PREMIUM = "binary_premium"  # Standard (<7) vs Premium (>=7)


class ImbalanceStrategy(str, Enum):
    """Imbalance mitigation strategy."""
    CLASS_WEIGHT = "class_weight"
    SMOTE = "smote"
    SMOTE_ENN = "smote_enn"
    NONE = "none"


class ProjectConfig(BaseModel):
    """Central configuration for data ingestion, feature engineering, and model training."""
    # Data Sources
    uci_red_wine_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    uci_white_wine_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    # Target and Problem Setup
    target_column: str = "quality"
    target_mode: TargetMode = TargetMode.MULTICLASS
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.CLASS_WEIGHT
    
    # Feature Definitions
    raw_numerical_features: List[str] = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    categorical_features: List[str] = ["is_red"]
    
    # Validation & Training
    random_state: int = 42
    test_size: float = 0.20
    n_splits_cv: int = 5
    
    # MLflow
    mlflow_experiment_name: str = "wine-quality-multiclass"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    
    # Serialization Names
    pipeline_artifact_name: str = "wine_quality_pipeline.joblib"
    explainer_artifact_name: str = "shap_explainer.joblib"
    reference_data_name: str = "train_reference_data.parquet"


config = ProjectConfig()
