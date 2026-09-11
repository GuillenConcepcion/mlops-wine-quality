"""
Training, Benchmarking, and MLOps Pipeline Orchestrator.
Performs stratified cross-validation, hyperparameter tuning with Optuna,
MLflow experiment tracking, SHAP explainability generation, and artifact serialization.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn

from src.config import config, TargetMode, MODELS_DIR, REPORTS_DIR
from src.data.loader import get_train_test_split
from src.features.transformers import EnologicalFeatureEngineer
from src.models.evaluate import (
    compute_multiclass_metrics,
    compute_bootstrap_ci,
    plot_multiclass_confusion_matrix,
)
from src.models.explain import WineQualityExplainer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_stratified_folds(y: pd.Series, n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified folds safely handling classes with fewer samples than n_splits.
    Uses an adjacent-class grouping proxy for splitting without altering actual training labels.
    """
    y_strat = y.copy().astype(int)
    counts = y_strat.value_counts()
    rare_classes = counts[counts < n_splits].index

    # Map rare classes to nearest adjacent class for splitting purposes only
    for rc in rare_classes:
        adj = rc - 1 if rc > y_strat.min() else rc + 1
        y_strat = y_strat.replace(rc, adj)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y_strat))


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Runs cross-validation and averages multiclass metrics across folds."""
    fold_metrics = []

    for train_idx, val_idx in folds:
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        y_pred = pipeline.predict(X_fold_val)

        m = compute_multiclass_metrics(y_fold_val.values, y_pred)
        fold_metrics.append(m)

    # Average metrics
    avg_metrics = {}
    for k in fold_metrics[0].keys():
        avg_metrics[f"cv_{k}"] = float(np.mean([m[k] for m in fold_metrics]))

    return avg_metrics


def run_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Benchmarks multiple candidate models with balanced class weights.
    """
    models = {
        "LogisticRegression_Baseline": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.random_state
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=150, random_state=config.random_state, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            class_weight="balanced", n_estimators=150, random_state=config.random_state, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced",
            n_estimators=150,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    results = {}
    logger.info("--- Starting Model Benchmarking ---")

    for name, estimator in models.items():
        pipe = Pipeline([
            ("feature_engineer", EnologicalFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])

        cv_res = cross_validate_pipeline(pipe, X_train, y_train, folds)
        results[name] = {"pipeline": pipe, "cv_metrics": cv_res}
        logger.info(
            f"[{name:25s}] CV Macro-F1: {cv_res['cv_f1_macro']:.4f} | "
            f"Balanced Acc: {cv_res['cv_balanced_accuracy']:.4f} | "
            f"Accuracy: {cv_res['cv_accuracy']:.4f}"
        )

    return results


def optimize_extra_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 15,
) -> Dict[str, Any]:
    """
    Hyperparameter optimization using Optuna for the top-performing ensemble architecture.
    """
    logger.info(f"Starting Bayesian Optimization with Optuna ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 35),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "class_weight": "balanced",
            "random_state": config.random_state,
            "n_jobs": -1,
        }

        pipe = Pipeline([
            ("feature_engineer", EnologicalFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("classifier", ExtraTreesClassifier(**params)),
        ])

        cv_res = cross_validate_pipeline(pipe, X_train, y_train, folds)
        return cv_res["cv_f1_macro"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best Trial Macro-F1: {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")
    return study.best_params


def train_and_register_production_model():
    """
    Complete end-to-end execution:
    Data -> Benchmark -> Optuna -> MLflow Tracking -> Test Evaluation -> SHAP -> Artifacts
    """
    # Set MLflow tracking
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    # Ingest and split
    X_train, X_test, y_train, y_test = get_train_test_split()
    folds = get_stratified_folds(y_train, n_splits=config.n_splits_cv, random_state=config.random_state)

    # 1. Benchmarking
    benchmark_results = run_benchmark(X_train, y_train, folds)

    # 2. Hyperparameter Optimization
    best_params = optimize_extra_trees(X_train, y_train, folds, n_trials=12)

    # 3. Final Production Model Pipeline
    final_estimator = ExtraTreesClassifier(
        **best_params,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )

    final_pipeline = Pipeline([
        ("feature_engineer", EnologicalFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("classifier", final_estimator),
    ])

    logger.info("Training final production pipeline on complete training set...")
    final_pipeline.fit(X_train, y_train)

    # 4. Rigorous Test Set Evaluation
    y_test_pred = final_pipeline.predict(X_test)
    test_metrics = compute_multiclass_metrics(y_test.values, y_test_pred)

    f1_est, f1_low, f1_high = compute_bootstrap_ci(
        y_test.values, y_test_pred, metric_name="f1_macro", n_bootstraps=1000
    )
    test_metrics["test_f1_macro_ci_low"] = f1_low
    test_metrics["test_f1_macro_ci_high"] = f1_high

    logger.info("=== FINAL TEST METRICS ===")
    logger.info(f"Test Accuracy:          {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Test Macro-F1:          {test_metrics['f1_macro']:.4f} (95% CI: [{f1_low:.4f}, {f1_high:.4f}])")
    logger.info(f"Test Weighted-F1:       {test_metrics['f1_weighted']:.4f}")

    # 5. Visualizations & Artifacts
    cm_path = REPORTS_DIR / "confusion_matrix_test.png"
    plot_multiclass_confusion_matrix(
        y_test.values,
        y_test_pred,
        labels=sorted(y_train.unique()),
        save_path=str(cm_path),
    )

    # 6. SHAP Explainer Creation
    logger.info("Generating SHAP Explainer and global feature attributions...")
    fe_step = final_pipeline.named_steps["feature_engineer"]
    scaler_step = final_pipeline.named_steps["scaler"]
    clf_step = final_pipeline.named_steps["classifier"]

    X_train_fe = fe_step.transform(X_train)
    feature_names = fe_step.get_feature_names_out()
    X_train_scaled = scaler_step.transform(X_train_fe)

    explainer = WineQualityExplainer(
        model=clf_step,
        feature_names=feature_names,
        class_labels=clf_step.classes_,
    )

    shap_summary_path = REPORTS_DIR / "shap_global_summary.png"
    sample_indices = np.random.RandomState(42).choice(len(X_train_scaled), size=min(150, len(X_train_scaled)), replace=False)
    explainer.generate_summary_plot(X_train_scaled[sample_indices], save_path=str(shap_summary_path))

    # 7. MLflow Experiment Logging
    with mlflow.start_run(run_name="production_extra_trees_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_param("imbalance_strategy", "class_weight=balanced")
        mlflow.log_param("target_mode", config.target_mode.value)
        mlflow.log_metrics(test_metrics)

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(shap_summary_path))

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="pipeline",
            serialization_format="cloudpickle",
        )
        logger.info("Run successfully logged to MLflow.")

    # 8. Local Serialization for Serving & Drift Monitoring
    pipeline_save_path = MODELS_DIR / config.pipeline_artifact_name
    explainer_save_path = MODELS_DIR / config.explainer_artifact_name
    reference_data_path = MODELS_DIR / config.reference_data_name

    joblib.dump(final_pipeline, pipeline_save_path)
    joblib.dump(explainer, explainer_save_path)
    X_train.to_parquet(reference_data_path, index=False)

    logger.info(f"Production pipeline saved to: {pipeline_save_path}")
    logger.info(f"Production explainer saved to: {explainer_save_path}")
    logger.info(f"Training reference data saved to: {reference_data_path}")

    return test_metrics


if __name__ == "__main__":
    train_and_register_production_model()
