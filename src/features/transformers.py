"""
Enological Feature Engineering Module.
Custom scikit-learn transformers implementing domain-specific enological features:
- Volatile to fixed acidity ratio
- Free to total sulfur dioxide ratio & bound sulfur dioxide
- Total acidity sum
- Alcohol to residual sugar ratio
- Sulphates to chlorides ratio
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EnologicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw physicochemical features into specialized enological interaction metrics.
    Ensures safe division by adding epsilon to denominators.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.feature_names_in_: Optional[List[str]] = None
        self.engineered_feature_names_: List[str] = [
            "acidity_ratio",          # volatile / fixed
            "free_so2_ratio",         # free_so2 / total_so2
            "bound_so2",              # total_so2 - free_so2
            "total_acidity",          # fixed + volatile + citric
            "alcohol_sugar_ratio",    # alcohol / residual_sugar
            "sulphates_chlorides_ratio" # sulphates / chlorides
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Store input feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute enological features and append them to the dataset."""
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise ValueError("X must be a pandas DataFrame or fitted with one.")

        X_out = X.copy()

        # 1. Volatile vs Fixed Acidity Ratio (Acetic vs Tartaric)
        X_out["acidity_ratio"] = X_out["volatile_acidity"] / (X_out["fixed_acidity"] + self.eps)

        # 2. Free vs Total SO2 Ratio & Bound SO2
        X_out["free_so2_ratio"] = X_out["free_sulfur_dioxide"] / (X_out["total_sulfur_dioxide"] + self.eps)
        X_out["bound_so2"] = np.maximum(0, X_out["total_sulfur_dioxide"] - X_out["free_sulfur_dioxide"])

        # 3. Total Acidity (Sum of primary organic acids)
        X_out["total_acidity"] = X_out["fixed_acidity"] + X_out["volatile_acidity"] + X_out["citric_acid"]

        # 4. Alcohol to Sugar Ratio (Fermentation balance)
        X_out["alcohol_sugar_ratio"] = X_out["alcohol"] / (X_out["residual_sugar"] + self.eps)

        # 5. Sulphates to Chlorides Ratio
        X_out["sulphates_chlorides_ratio"] = X_out["sulphates"] / (X_out["chlorides"] + self.eps)

        return X_out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return combined original and newly engineered feature names."""
        base_features = input_features if input_features is not None else self.feature_names_in_
        if base_features is None:
            raise ValueError("Transformer has not been fitted and no input features were provided.")
        return list(base_features) + self.engineered_feature_names_
