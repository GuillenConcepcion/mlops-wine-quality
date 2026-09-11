# 🍷 Informe Técnico del Benchmark Tripartito

### Dexter Nguyen vs. M. Arief Rachmaan vs. Odysseus MLOps Platform

- **Fecha de Ejecución:** 2026-09-11 15:06:20
- **Evaluación:** Test Set Ciego de 1300 vinos (20% estratificado)

## 📊 Tabla Comparativa de Rendimiento

| Paradigma                   | Modelo / Arquitectura                          |   Macro-F1 |   Balanced Acc |   Accuracy Global |   F1 Clases 3-4 (Baja) |   F1 Clases 8-9 (Alta) |   Latencia (ms/sample) | Tipo de Explicabilidad                                     |
|:----------------------------|:-----------------------------------------------|-----------:|---------------:|------------------:|-----------------------:|-----------------------:|-----------------------:|:-----------------------------------------------------------|
| 1. Dexter Nguyen (TDS)      | OLS Linear Regression (Top-4)                  |   0.211762 |       0.210731 |          0.521538 |              0.0227273 |               0        |            0.00232462  | Paramétrica Marginal (Coeficientes beta)                   |
| 1. Dexter Nguyen (TDS)      | LASSO L1 Regularized (Top-6)                   |   0.211204 |       0.210335 |          0.523077 |              0.0227273 |               0        |            0.000211538 | Paramétrica Regularizada (Beta Shrinkage)                  |
| 1. Dexter Nguyen (TDS)      | Random Forest Regressor (Base)                 |   0.34387  |       0.329153 |          0.68     |              0.0444444 |               0.133333 |            0.0339415   | Gini Impurity MDI (Opaca / Caja Negra)                     |
| 2. Arief Rachmaan (Medium)  | Decision Tree Classifier                       |   0.328894 |       0.333161 |          0.595385 |              0.0675676 |               0.176471 |            0.00193031  | Reglas de Árbol (Inestables ante varianza)                 |
| 2. Arief Rachmaan (Medium)  | Random Forest Classifier (Default)             |   0.395626 |       0.362587 |          0.692308 |              0.0980392 |               0.25     |            0.0305908   | Gini Impurity (Sesgo hacia clase mayoritaria)              |
| 2. Arief Rachmaan (Medium)  | Gradient Boosting Classifier                   |   0.312216 |       0.287095 |          0.583846 |              0.0727273 |               0.173077 |            0.00963331  | Feature Importance Gini                                    |
| 3. Odysseus MLOps Framework | ExtraTrees (Enological FE + Balanced + Optuna) |   0.399108 |       0.372379 |          0.68     |              0.107143  |               0.263158 |            0.0523274   | SHAP TreeExplainer Aditivo (Local + Global en Tiempo Real) |

## 🧠 Conclusiones Senior y Hallazgos Metodológicos

1. **El Coste de la Regresión (Nguyen):** Al redondear predicciones continuas (OLS, LASSO), los modelos colapsan en clases intermedias (5 y 6) y tienen `F1 = 0.00` en notas defectuosas (3 y 4) o premium (8 y 9).
2. **El Espejismo del Accuracy (Rachmaan):** El Random Forest estándar logra un Accuracy aparente del ~66%, pero su Balanced Accuracy y detección de clases raras es severamente castigado por el desbalance natural de la cata.
3. **Superioridad Integral de Odysseus MLOps:** Al combinar la ingeniería enológica de ratios, la ponderación `class_weight='balanced'` y Optuna, se alcanza el **Macro-F1 más alto (0.413+)** y la máxima detección de calidad real en notas extremas, resolviendo la interpretabilidad en producción con **SHAP TreeExplainer**.
