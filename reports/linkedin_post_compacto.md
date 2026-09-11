# 🍷 Post Compacto para LinkedIn (< 1,600 caracteres)

🍷 **¿Por qué la mayoría de los modelos fallan al predecir la calidad sensorial del vino?**

Al modelar el dataset de Cortez et al. (*Vinho Verde*), se suele caer en dos atajos:
1. **Binarizar trivialmente** (*"bueno vs malo"*), perdiendo la granularidad sensorial real de cata.
2. **Tratarlo como regresión continua**, donde redondear a notas enteras causa un **$F_1 = 0.00$ en vinos excelentes ($8-9$)** por colapso en la masa central ($5-6$).

En **Odysseus AI Platform** diseñé una solución **MLOps & XAI de grado productivo** bajo metodología **CRISP-DM**:

📊 **Benchmark Tripartito (Test ciego $N=1,300$):**
• *Nguyen (Regresión OLS/LASSO):* F1=0.00 en calidad alta; la regresión continua no sirve en cata.
• *Rachmaan (Random Forest Default):* Accuracy aparente alto (69%), pero sesgado a clases centrales (Balanced Acc: 36%).
• **Odysseus MLOps (ExtraTrees + FE Enológico + Balanced):** **Líder con Macro-F1 = 0.399 (0.443 CV)** y máxima detección en defectos y excelencia.

🧠 **XAI en Tiempo Real:** 
Superamos la caja negra integrando **SHAP TreeExplainer** en inferencia (<55 ms), demostrando matemáticamente el impacto del alcohol (+0.182) y el castigo por acidez volátil.

⚙️ **Arquitectura:**
• API asíncrona con **FastAPI**
• Containerización rootless en **Podman** (UID 8888)
• Tracking en **MLflow** y detección de **Data Drift** (KS-Test & PSI)

👉 **Repo:** https://github.com/GuillenConcepcion/mlops-wine-quality
📫 **Contacto:** guillenconcepcion@gmail.com | [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127)

#MachineLearning #MLOps #DataScience #XAI #Python #FastAPI #Podman
