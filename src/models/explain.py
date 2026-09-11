"""
Explainability Module (XAI) using SHAP (SHapley Additive exPlanations).
Provides global feature attribution and local instance-level explanations for API serving.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class WineQualityExplainer:
    """
    Wraps SHAP TreeExplainer for multiclass tree models.
    Supports top-k local explanations for real-time API queries.
    """

    def __init__(self, model: Any, feature_names: List[str], class_labels: Optional[List[int]] = None):
        self.model = model
        self.feature_names = feature_names
        self.class_labels = class_labels if class_labels is not None else getattr(model, "classes_", [3, 4, 5, 6, 7, 8, 9])
        # TreeExplainer is fast and exact for ensembles (RandomForest, ExtraTrees, LightGBM, XGBoost)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(
        self,
        X_instance: np.ndarray,
        predicted_class: int,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP values for a single processed sample and returns the top_k contributors
        to the predicted class.
        """
        # Ensure 2D shape (1, n_features)
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        # shap_values: for multiclass, usually returns a list of arrays [n_classes] each of shape (1, n_features)
        # or a 3D array (1, n_features, n_classes)
        raw_shap = self.explainer.shap_values(X_instance)

        # Locate index of predicted class
        class_list = list(self.class_labels)
        try:
            class_idx = class_list.index(predicted_class)
        except ValueError:
            class_idx = 0

        if isinstance(raw_shap, list):
            # List of (n_samples, n_features) per class
            instance_shap = raw_shap[class_idx][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
            instance_shap = raw_shap[0, :, class_idx]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            # Binary or single output
            instance_shap = raw_shap[0]
        else:
            instance_shap = np.zeros(len(self.feature_names))

        # Combine with feature names and instance values
        contributions = []
        for feat_name, feat_val, s_val in zip(self.feature_names, X_instance[0], instance_shap):
            contributions.append({
                "feature": feat_name,
                "value": round(float(feat_val), 4),
                "shap_impact": round(float(s_val), 4),
                "direction": "positive" if s_val >= 0 else "negative",
                "abs_impact": abs(float(s_val)),
            })

        # Sort by absolute impact descending and take top_k
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_contributors = [
            {
                "feature": c["feature"],
                "value": c["value"],
                "shap_impact": c["shap_impact"],
                "direction": c["direction"],
            }
            for c in contributions[:top_k]
        ]

        return top_contributors

    def generate_summary_plot(
        self,
        X_sample: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 15,
    ) -> None:
        """Generates and saves a global SHAP summary bar plot."""
        shap_values = self.explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            class_names=[f"Quality {c}" for c in self.class_labels],
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("Global Feature Importance by Wine Quality Class (SHAP)", fontsize=13, pad=15)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
