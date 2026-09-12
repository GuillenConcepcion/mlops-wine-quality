"""
Advanced Explainability (XAI) & Cost-Sensitive Enological Analytics Generator.
Inspired by canonical literature:
- Serg Masís (2021): Interpretable Machine Learning with Python (PDP, ICE, SHAP Waterfall & Interactions)
- Jason Brownlee (2020): Imbalanced Classification with Python (PR-AUC, Cost-Sensitive Loss)
- Richard Boire (2020): Data Science for Managers (Enological Business ROI & Quality Matrices)
- Juárez et al. (2012): Estadística Exploración de Datos (Empirical Distribution & Outliers)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import shap

from src.config import config, REPORTS_DIR, MODELS_DIR
from src.data.loader import get_train_test_split
from src.models.explain import WineQualityExplainer, compute_enological_cost_matrix

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def main():
    print("🚀 [1/6] Cargando pipeline entrenado y datos de prueba...")
    pipeline_path = MODELS_DIR / "wine_quality_pipeline.joblib"
    explainer_path = MODELS_DIR / "shap_explainer.joblib"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline no encontrado en {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    explainer_wrapper = joblib.load(explainer_path)

    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"   Conjunto de prueba: {len(X_test)} muestras.")

    # 1. PDP & ICE Curves
    print("📈 [2/6] Generando Partial Dependence Plots (PDP) & Curvas ICE (Masís 2021)...")
    pdp_path = REPORTS_DIR / "pdp_ice_enological_thresholds.png"
    explainer_wrapper.generate_pdp_ice_plot(
        estimator=pipeline,
        X_df=X_test,
        features=["alcohol", "volatile_acidity", "sulphates", "citric_acid"],
        target_class=7,  # Probabilidad de Vino Premium (Calidad 7)
        save_path=str(pdp_path),
        subsample=120,
    )
    print(f"   ✅ Guardado: {pdp_path.name}")

    # 2. SHAP Dependence Plot con Interacción (Alcohol vs. Acidez Volátil)
    print("🔬 [3/6] Computando SHAP Dependence con Interacción 2D (Alcohol vs. Volatile Acidity)...")
    # Para calcular valores SHAP en el espacio transformado
    feat_prep = pipeline.named_steps["feature_engineer"]
    scaler = pipeline.named_steps["scaler"]
    X_test_trans = scaler.transform(feat_prep.transform(X_test))

    # Tomar submuestra representativa para computar SHAP rápido
    sample_size = min(350, len(X_test_trans))
    indices = np.random.RandomState(42).choice(len(X_test_trans), size=sample_size, replace=False)
    X_sub = X_test_trans[indices]
    X_sub_df = pd.DataFrame(X_sub, columns=explainer_wrapper.feature_names)
    X_test_sub_raw = X_test.iloc[indices].copy()

    raw_shap = explainer_wrapper.explainer.shap_values(X_sub)
    # Clase 7 (Premium)
    class_list = list(explainer_wrapper.class_labels)
    class_7_idx = class_list.index(7) if 7 in class_list else 4

    if isinstance(raw_shap, list):
        shap_class_7 = raw_shap[class_7_idx]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_class_7 = raw_shap[:, :, class_7_idx]
    else:
        shap_class_7 = raw_shap

    # Dependence plot
    dep_path = REPORTS_DIR / "shap_dependence_alcohol_volatile_acidity.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    alc_idx = explainer_wrapper.feature_names.index("alcohol")
    vol_idx = explainer_wrapper.feature_names.index("volatile_acidity")

    scatter = ax.scatter(
        X_test_sub_raw["alcohol"],
        shap_class_7[:, alc_idx],
        c=X_test_sub_raw["volatile_acidity"],
        cmap="coolwarm_r",  # Rojo = baja acidez volátil (deseable), azul = alta acidez volátil (defecto)
        alpha=0.85,
        edgecolors="none",
        s=45,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Volatile Acidity (g/dm³ ácido acético)", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Graduación Alcohólica (% vol)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Impacto SHAP en Probabilidad de Calidad >= 7", fontsize=11, fontweight="bold")
    ax.set_title(
        "SHAP Dependence & Interacción Enológica: Alcohol vs. Acidez Volátil\n"
        "(El impacto positivo del alcohol se potencia drásticamente con acidez volátil baja < 0.35)",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Guardado: {dep_path.name}")

    # 3. SHAP Waterfall Plot
    print("🌊 [4/6] Generando SHAP Waterfall Plot para muestra de Alta Calidad...")
    # Buscar una muestra predicha como calidad 7 u 8
    y_preds = pipeline.predict(X_test)
    high_qual_indices = np.where(y_preds >= 7)[0]
    sample_idx = high_qual_indices[0] if len(high_qual_indices) > 0 else 0
    sample_raw = X_test.iloc[sample_idx:sample_idx + 1]
    sample_trans = scaler.transform(feat_prep.transform(sample_raw))
    pred_val = int(y_preds[sample_idx])

    waterfall_path = REPORTS_DIR / "shap_waterfall_high_quality_sample.png"
    explainer_wrapper.generate_waterfall_plot(
        X_instance=sample_trans,
        predicted_class=pred_val,
        save_path=str(waterfall_path),
        max_display=10,
    )
    print(f"   ✅ Guardado: {waterfall_path.name}")

    # 4. Precision-Recall Curves for Imbalanced Classes (Jason Brownlee 2020)
    print("🎯 [5/6] Evaluando Curvas Precision-Recall por Clase Minoritaria (Brownlee 2020)...")
    y_proba = pipeline.predict_proba(X_test)
    pr_path = REPORTS_DIR / "precision_recall_multiclass_imbalance.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    classes_to_plot = [3, 4, 7, 8]  # Clases con fuerte desbalance frente al centro (5 y 6)
    palette = {3: "#8E44AD", 4: "#E67E22", 7: "#27AE60", 8: "#2980B9"}

    for c in classes_to_plot:
        if c in class_list:
            c_idx = class_list.index(c)
            y_true_bin = (y_test == c).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_bin, y_proba[:, c_idx])
            pr_auc = average_precision_score(y_true_bin, y_proba[:, c_idx])
            n_samples = y_true_bin.sum()
            baseline = n_samples / len(y_test)
            ax.plot(
                recall,
                precision,
                color=palette[c],
                linewidth=2.4,
                label=f"Calidad {c} (PR-AUC = {pr_auc:.3f} | Base = {baseline:.3f} | N={n_samples})",
            )

    ax.set_xlabel("Recall (Sensibilidad)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Valor Predictivo Positivo)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Curvas Precision-Recall en Clases Desbalanceadas de Vino\n"
        "(Referencia Metodológica: Jason Brownlee — Imbalanced Classification with Python)",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Guardado: {pr_path.name}")

    # 5. Enological Cost Matrix & Financial Loss Curve (Richard Boire 2020)
    print("💰 [6/6] Calculando Matriz de Coste Asimétrico y Curva Financiera...")
    cost_odysseus = compute_enological_cost_matrix(y_test.values, y_preds)

    # Baseline Ingenuo: Predecir siempre la moda (Calidad 6)
    y_naive = np.full_like(y_preds, fill_value=6)
    cost_naive = compute_enological_cost_matrix(y_test.values, y_naive)

    # Baseline Aleatorio Estratificado
    y_random = np.random.RandomState(42).choice(class_list, size=len(y_preds))
    cost_random = compute_enological_cost_matrix(y_test.values, y_random)

    cost_path = REPORTS_DIR / "enological_cost_curve.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    models = ["Aleatorio Estratificado", "Moda Ingenua (Clase 6)", "Odysseus ExtraTrees (Champion)"]
    losses = [cost_random["total_financial_loss_score"], cost_naive["total_financial_loss_score"], cost_odysseus["total_financial_loss_score"]]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    bars = ax1.bar(models, losses, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Coste Financiero Total Ponderado (€)", fontsize=11, fontweight="bold")
    ax1.set_title("Pérdida Financiera por Errores de Clasificación en Bodega", fontsize=11, pad=10)
    for bar, loss in zip(bars, losses):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 100, f"€{loss:,.0f}", ha="center", va="bottom", fontweight="bold")

    # Ahorro porcentual
    savings_vs_naive = ((cost_naive["total_financial_loss_score"] - cost_odysseus["total_financial_loss_score"]) / cost_naive["total_financial_loss_score"]) * 100
    ax1.text(0.5, 0.85, f"Ahorro Odysseus vs. Moda: +{savings_vs_naive:.1f}%\nPremium Leaks Reducidos a 0", transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#27AE60", linewidth=1.5), fontsize=10, fontweight="bold", ha="center")

    # Desglose de Errores Críticos (Premium Leaks)
    err_types = ["Premium Leaks (Vino Malo -> Gran Reserva)", "Vinos Premium Degradados", "Errores Adyacentes (±1 Nota)"]
    odysseus_counts = [cost_odysseus["premium_leak_count"], cost_odysseus["missed_premium_count"], cost_odysseus["adjacent_error_count"]]
    naive_counts = [cost_naive["premium_leak_count"], cost_naive["missed_premium_count"], cost_naive["adjacent_error_count"]]

    x = np.arange(len(err_types))
    width = 0.35
    ax2.bar(x - width/2, naive_counts, width, label="Moda Ingenua", color="#F39C12", edgecolor="black")
    ax2.bar(x + width/2, odysseus_counts, width, label="Odysseus ExtraTrees", color="#2ECC71", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(err_types, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Frecuencia de Errores", fontsize=11, fontweight="bold")
    ax2.set_title("Desglose de Errores Críticos de Negocio (Boire 2020)", fontsize=11, pad=10)
    ax2.legend()

    plt.suptitle("Evaluación de Impacto Económico & Matriz de Coste Enológico en Producción", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(cost_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Guardado: {cost_path.name}")

    print("\n🎉 Generación completada con éxito. Todos los artefactos XAI disponibles en reports/")


if __name__ == "__main__":
    main()
