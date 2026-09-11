"""
Evaluation module for multiclass wine quality classification.
Provides rigorous metrics, bootstrap confidence intervals, and visualization tools.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard and imbalance-aware multiclass metrics.
    Focuses on Macro-F1 and Balanced Accuracy as senior KPIs.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = "f1_macro",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval (95% CI) for a chosen metric.
    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrapped_scores: List[float] = []

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        
        if metric_name == "f1_macro":
            score = f1_score(y_true_arr[indices], y_pred_arr[indices], average="macro", zero_division=0)
        elif metric_name == "balanced_accuracy":
            score = balanced_accuracy_score(y_true_arr[indices], y_pred_arr[indices])
        else:
            score = accuracy_score(y_true_arr[indices], y_pred_arr[indices])

        bootstrapped_scores.append(float(score))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrapped_scores, alpha * 100))
    upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))
    point_est = float(np.mean(bootstrapped_scores))

    return point_est, lower, upper


def plot_multiclass_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Any]] = None,
    normalize: str = "true",
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a high-fidelity normalized confusion matrix heatmap.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted Quality", fontsize=11)
    ax.set_ylabel("True Quality", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
