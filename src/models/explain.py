"""
Explainability Module (XAI) using SHAP (SHapley Additive exPlanations).
Provides global feature attribution and local instance-level explanations for API serving.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
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

    def generate_pdp_ice_plot(
        self,
        estimator: Any,
        X_df: pd.DataFrame,
        features: List[str],
        target_class: int = 7,
        save_path: Optional[str] = None,
        subsample: int = 250,
    ) -> None:
        """
        Generates Partial Dependence Plots (PDP) overlaid with Individual Conditional Expectation (ICE)
        curves to reveal nonlinear thresholds and enological tipping points (Masís, 2021).
        """
        from sklearn.inspection import PartialDependenceDisplay

        class_list = list(self.class_labels)
        try:
            target_idx = class_list.index(target_class)
        except ValueError:
            target_idx = 0

        X_eval = X_df.sample(min(subsample, len(X_df)), random_state=42) if len(X_df) > subsample else X_df

        fig, ax = plt.subplots(1, len(features), figsize=(6 * len(features), 5), sharey=True)
        if len(features) == 1:
            ax = [ax]

        display = PartialDependenceDisplay.from_estimator(
            estimator,
            X_eval,
            features=features,
            target=target_idx,
            kind="both",  # Overlay individual ICE lines with average PDP trend
            ax=ax,
            ice_lines_kw={"color": "#4A90E2", "alpha": 0.12, "linewidth": 0.8},
            pd_line_kw={"color": "#D0021B", "linewidth": 2.8, "label": "PDP (Promedio Marginal)"},
        )

        fig.suptitle(
            f"Partial Dependence & ICE Curves — Probabilidad de Calidad >= {target_class}\n"
            f"(Fundamentación Metodológica: Masís 2021 — Interpretable ML with Python)",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_waterfall_plot(
        self,
        X_instance: np.ndarray,
        predicted_class: int,
        save_path: Optional[str] = None,
        max_display: int = 10,
    ) -> None:
        """
        Generates a local SHAP waterfall plot showing positive/negative pushes from base value E[f(x)].
        """
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        raw_shap = self.explainer.shap_values(X_instance)
        class_list = list(self.class_labels)
        try:
            class_idx = class_list.index(predicted_class)
        except ValueError:
            class_idx = 0

        if isinstance(raw_shap, list):
            instance_shap = raw_shap[class_idx][0]
            base_val = getattr(self.explainer, "expected_value", [0.0] * len(class_list))[class_idx]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            instance_shap = raw_shap[0, :, class_idx]
            base_val = getattr(self.explainer, "expected_value", [0.0] * len(class_list))[class_idx]
        else:
            instance_shap = raw_shap[0]
            base_val = 0.0

        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        explanation = shap.Explanation(
            values=instance_shap,
            base_values=base_val,
            data=X_instance[0],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(
            f"SHAP Waterfall Attribution — Instancia Predicha como Calidad {predicted_class}\n"
            f"Desglose Aditivo Local vs. Expectativa Base E[f(x)]",
            fontsize=12,
            pad=12,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def compute_enological_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_premium_leak: float = 100.0,
    cost_missed_premium: float = 30.0,
    cost_adjacent_error: float = 5.0,
) -> Dict[str, Any]:
    """
    Computes enological asymmetric financial loss inspired by Jason Brownlee (2020)
    and Richard Boire (2020):
    - Premium Leak: Defective wine (<=4) predicted as Premium (>=7) -> Severe brand damage.
    - Missed Premium: Premium wine (>=7) predicted as Low (<=4) -> Opportunity loss.
    - Adjacent error: Distance |y_true - y_pred| == 1 -> Minor friction.
    """
    total_cost = 0.0
    breakdown = {
        "premium_leak_count": 0,
        "missed_premium_count": 0,
        "adjacent_error_count": 0,
        "severe_error_count": 0,
        "exact_matches": 0,
        "total_samples": len(y_true),
    }

    for yt, yp in zip(y_true, y_pred):
        diff = abs(int(yt) - int(yp))
        if diff == 0:
            breakdown["exact_matches"] += 1
            continue

        # Asymmetric business rules
        if yt <= 4 and yp >= 7:
            total_cost += cost_premium_leak
            breakdown["premium_leak_count"] += 1
        elif yt >= 7 and yp <= 4:
            total_cost += cost_missed_premium
            breakdown["missed_premium_count"] += 1
        elif diff == 1:
            total_cost += cost_adjacent_error
            breakdown["adjacent_error_count"] += 1
        else:
            total_cost += cost_adjacent_error * (diff ** 1.5)
            breakdown["severe_error_count"] += 1

    breakdown["total_financial_loss_score"] = round(total_cost, 2)
    breakdown["average_loss_per_bottle"] = round(total_cost / max(1, len(y_true)), 2)
    return breakdown

