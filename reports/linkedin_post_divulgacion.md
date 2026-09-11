# 🍷 Post de Divulgación para LinkedIn: Odysseus AI — Wine Quality MLOps & XAI System

---

### 🚀 Texto Listo para Copiar y Publicar en LinkedIn:

---

🍷 **¿Por qué la mayoría de los modelos de Machine Learning fallan al predecir la calidad sensorial del vino?**

En la literatura y tutoriales de Data Science abundan dos atajos habituales al trabajar con el clásico dataset de Cortez et al. (*Vinho Verde*):
1. **Binarizar trivialmente el problema** (*"vino bueno vs. vino malo"*), perdiendo la granularidad sensorial de notas enteras que exige un comité de cata o una bodega comercial.
2. **Tratarlo como regresión continua** ($Y \in \mathbb{R}$), redondear los decimales a notas enteras y encontrarse con que el modelo colapsa en la masa central ($5$ y $6$), arrojando un **$F_1 = 0.00$ en la detección de vinos de excelencia ($8$ y $9$)**.

Para superar estas limitaciones y resolver el clásico dilema entre **capacidad predictiva pura** e **interpretabilidad econométrica**, he diseñado e implementado dentro de la plataforma **Odysseus AI** una solución integral de **MLOps & XAI de grado productivo**, guiada rigurosamente por el estándar **CRISP-DM**.

---

### 📊 1. El Benchmark Empírico Tripartito (Mismo Test Ciego, $N=1,300$ muestras)

Contrastamos experimentalmente tres filosofías de modelado sobre el mismo conjunto holdout estratificado e independiente:

1. **Paradigma Econométrico de Dexter Nguyen (Duke / Towards Data Science):**
   * *Aproximación:* OLS Linear Regression (Top-4) y LASSO $L_1$ (Top-6).
   * *Resultado:* Aunque ofrecía coeficientes marginales $\beta$ claros, sufre un sesgo central severo: $F_1 = 0.00$ en calidad alta y $F_1 = 0.02$ en defectos. La regresión continua enmascara la naturaleza discreta del panel sensorial.
2. **Paradigma de Clasificación Convencional de M. Arief Rachmaan (Medium):**
   * *Aproximación:* Random Forest y Gradient Boosting sin balanceo de clases.
   * *Resultado:* *Accuracy Illusion*. Logra un Accuracy global aparente del $69.2\%$, pero a costa de ignorar las clases raras (el $76.5\%$ de las muestras son notas 5 y 6). Su Balanced Accuracy cae a $0.362$.
3. **Nuestra Solución: Odysseus MLOps Framework:**
   * *Aproximación:* Formulación multiclase pura ($3$ a $9$), ingeniería de características dominial (`EnologicalFeatureEngineer` con ratios de acidez tartárica/acética, equilibrio $\text{SO}_2$ libre/ligado y balance alcohol/azúcar residual), ponderación de costes `class_weight='balanced'` y sintonización bayesiana TPE con **Optuna** (12 trials).
   * *Resultado:* **Mayor Macro-F1 del benchmark ($0.3991$ en test ciego / $0.4431$ en 5-fold CV)** y la mayor sensibilidad en clases periféricas ($F_1=0.107$ en defectos $3-4$ y $F_1=0.263$ en excelencia $8-9$).

---

### 🧠 2. Superando la "Caja Negra" con SHAP ($XAI$) en Tiempo Real

El trade-off histórico planteaba: *"o uso modelos lineales explicables pero inexactos, o uso ensambles precisos pero opacos"*.

En **Odysseus**, rompemos este compromiso acoplando **SHAP TreeExplainer** directamente al motor de inferencia. En cada predicción del endpoint `POST /explain`, la API descompone el score en aditividad local y global exacta ($f(x) = \phi_0 + \sum \phi_i$) en menos de $55\text{ ms}$:
* Confirmando matemáticamente que el **alcohol (% ABV)** es el principal impulsor de calidad alta ($+0.182\text{ SHAP}$).
* Evidenciando que el exceso de **acidez volátil (ácido acético)** es el principal penalizador organoléptico (picado acético).

---

### ⚙️ 3. Arquitectura MLOps Cloud-Native & Segura

* ⚡ **FastAPI Asíncrono:** Validación estricta con esquemas Pydantic v2 para `/predict`, `/explain` y `/drift/evaluate`.
* 🔒 **Containerización Rootless (Podman / Docker Compose):** Construcción multi-stage compilada con `uv` ejecutándose con el usuario no privilegiado `appuser` (**UID 8888**).
* 📈 **Gobernanza & Tracking:** Registro automático de artefactos y métricas en **MLflow**.
* 🛡️ **Monitoreo Continuo de Data Drift:** Detección no paramétrica con **Kolmogorov-Smirnov (2-Sample KS)** y **Population Stability Index (PSI)** adaptativo para alertar desplazamientos poblacionales entre vendimias y disparar el reentrenamiento continuo.

---

El código completo, el informe técnico del benchmark, los cuadernos de exploración y los artefactos de visualización están disponibles en mi repositorio:

👉 **GitHub Repository:** https://github.com/GuillenConcepcion
👉 **LinkedIn Profile:** https://www.linkedin.com/in/guillen-concepcion-25266b127
📬 **Contacto:** guillenconcepcion@gmail.com

¿Cómo gestionas en tu organización el dilema entre precisión y explicabilidad en problemas con desbalance severo? ¡Me encantaría leer tus reflexiones en los comentarios! 👇

---

#MachineLearning #DataScience #MLOps #XAI #ArtificialIntelligence #Python #FastAPI #Podman #Docker #MLflow #SHAP #CRISPDM #AIArchitecture #WineQuality
