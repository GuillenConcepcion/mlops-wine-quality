# 🍷 Reporte Canónico de Explicabilidad Avanzada (XAI) & Economía Enológica

**Plataforma:** Odysseus AI Platform & Predictive MLOps Framework  
**Proyecto:** Wine Quality Multiclass & Imbalanced Classification Platform  
**Lead Architect:** [Guillen Concepcion](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*  
**Contacto:** [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) • [GitHub](https://github.com/GuillenConcepcion) • [Email](mailto:guillenconcepcion@gmail.com)  
**Literatura Canónica de Referencia:**  
* 📘 **Serg Masís (2021).** *Interpretable Machine Learning with Python.* Packt Publishing.
* 📗 **Jason Brownlee (2020).** *Imbalanced Classification with Python.* Machine Learning Mastery.
* 📙 **Richard Boire (2020).** *Data Science for Managers.* Springer.
* 📕 **Juárez, Ylé, Flórez & Inzunsa (2012).** *Estadística: Exploración de Datos.* UAS.

---

## 🎯 1. Resumen Ejecutivo y Motivación de Negocio

En la industria vitivinícola de alta gama, predecir la calificación de un panel de cata no es suficiente: los enólogos, directores de bodega y maestros de bodega requieren **comprender los mecanismos fisicoquímicos causales** y los **puntos de inflexión no lineales** que determinan por qué un lote alcanza la categoría de *Gran Reserva / Premium* (calidades 7, 8 o 9) o degenera en vino corriente o defectuoso (calidades 3 o 4).

Este informe documenta la integración de técnicas avanzadas de **Inteligencia Artificial Explicable (XAI)** y **Evaluación Financiera Asimétrica**, superando las limitaciones de los métodos tradicionales de caja negra e interpretabilidad global superficial.

---

## 🔬 2. Diagnósticos de Interpretabilidad Avanzada (XAI)

### 2.1. Partial Dependence Plots (PDP) & Curvas ICE (Serg Masís, 2021)
*(Artefacto: `reports/pdp_ice_enological_thresholds.png`)*

Mientras que la importancia global por árboles sólo asigna un número a cada variable, las curvas combinadas de **Partial Dependence (PDP)** y **Individual Conditional Expectation (ICE)** permiten aislar el efecto marginal promedio (línea roja continua) superpuesto con la dispersión heterogénea de cada vino individual (líneas azules tenues):

$$\text{PDP}(x_s) = \mathbb{E}_{X_c} \left[ f(x_s, X_c) \right] = \frac{1}{N} \sum_{i=1}^N f(x_s, x_{c}^{(i)})$$

#### Hallazgos y Reglas Fisicoquímicas Extraídas:
1. **Graduación Alcohólica (`alcohol`):**
   * Exhibe un comportamiento sigmoidal no lineal pronunciado.
   * **Punto de Inflexión Crítico:** Por debajo de $10.2\%$ vol, la probabilidad de que el vino sea calificado como Premium es inferior al $8\%$. A partir de los $11.5\%$ vol, la curva experimenta una aceleración marginal positiva abrupta, alcanzando un plateau en $\ge 12.8\%$ vol con probabilidades superiores al $45\%$.
2. **Acidez Volátil (`volatile_acidity` - Ácido Acético):**
   * Comportamiento monotónico decreciente estricto.
   * **Techo Letal:** Valores por encima de $0.40\text{ g/dm}^3$ colapsan la probabilidad de calidad alta a casi cero. Los vinos excepcionales requieren mantenerse estrictamente entre $0.18\text{ y } 0.32\text{ g/dm}^3$.
3. **Sulfatos (`sulphates` - Sulfato de Potasio):**
   * Efecto campana cóncava. Promueve la calidad óptima en el rango de $0.60 - 0.85\text{ g/dm}^3$. Por encima de $1.0\text{ g/dm}^3$, los sulfatos introducen aspereza y amargor que castigan la nota en cata.
4. **Acidez Cítrica (`citric_acid`):**
   * Aporta frescura aromática y preservación frutal en rangos de $0.25 - 0.45\text{ g/dm}^3$.

---

### 2.2. Interacción Enológica 2D: Alcohol vs. Acidez Volátil (SHAP Dependence)
*(Artefacto: `reports/shap_dependence_alcohol_volatile_acidity.png`)*

Mediante la descomposición de valores SHAP bivariados con mapa de calor:

$$\phi_j(x) + \sum_{k \neq j} \phi_{j,k}(x)$$

Se demuestra empíricamente la **regla de oro enológica**:
> *“Un grado alcohólico elevado ($\ge 12\%$) únicamente aporta valor positivo al perfil sensorial del vino si la acidez volátil se mantiene por debajo de $0.35\text{ g/dm}^3$. En vinos donde la acidez acética supera los $0.50\text{ g/dm}^3$, el alcohol elevado exacerba la sensación de picado y ardor, resultando en impactos SHAP netamente negativos.”*

---

### 2.3. Explicabilidad Local en Cascada: SHAP Waterfall Plot
*(Artefacto: `reports/shap_waterfall_high_quality_sample.png`)*

Para una botella evaluada por la bodega, el desglose aditivo local explica exactamente por qué el modelo clasificó la muestra como calidad 7 (o superior):
* **Expectativa Base $E[f(x)]$:** Probabilidad prior promedio en el dataset.
* **Empujes Positivos (+):** El alto contenido alcohólico ($+0.18$) y el ratio óptimo de sulfatos sobre cloruros ($+0.11$) elevan la predicción.
* **Frenos Negativos (-):** Ligero exceso de dióxido de azufre libre penaliza marginalmente ($-0.03$).

---

## 🎯 3. Evaluación de Clases Desbalanceadas (Jason Brownlee, 2020)
*(Artefacto: `reports/precision_recall_multiclass_imbalance.png`)*

En la cata de vinos, la distribución de notas presenta un desbalance acentuado:
* Calidad 5 y 6 (Vinos de mesa comunes): $> 76\%$ de las muestras.
* Calidad 3 (Defectuoso extremo): $30$ muestras ($0.46\%$).
* Calidad 8 y 9 (Gran Reserva / Joya enológica): $198$ muestras ($3.05\%$).

Bajo estas condiciones, la métrica tradicional ROC-AUC resulta engañosa. Adoptando las directrices de Brownlee:
* **Calidad 7:** PR-AUC = **`0.684`** (frente a un baseline ingenuo de $0.166$, lo que representa una **ganancia predictiva de $4.1\times$**).
* **Calidad 8:** PR-AUC = **`0.412`** (frente a un baseline aleatorio de $0.030$, ganancia de **$13.7\times$**).
* **Calidad 4:** PR-AUC = **`0.298`** (frente a baseline de $0.033$).

El modelo ExtraTrees demuestra una sobresaliente capacidad de aislamiento de los extremos sensoriales sin verse cegado por la hiperconcentración en las notas 5 y 6.

---

## 💰 4. Matriz de Coste Enológico & Retorno Financiero (Richard Boire, 2020)
*(Artefacto: `reports/enological_cost_curve.png`)*

Para conectar el algoritmo con el comité de dirección de la bodega, se parametriza la **función de pérdida económica asimétrica**:

| Tipo de Error | Consecuencia de Negocio | Penalización (€) |
| :--- | :--- | :---: |
| **Premium Leak** (Vino malo $\le 4$ clasificado como $\ge 7$) | Destrucción de imagen de marca y pérdida de clientes VIP. | **$100\text{ €}$** |
| **Missed Premium** (Vino excelente $\ge 7$ catalogado como $\le 4$) | Venta de caldo premium a precio de vino a granel (coste de oportunidad). | **$30\text{ €}$** |
| **Error Adyacente** ($|\Delta y| = 1$, ej. 5 vs. 6) | Fricción mínima aceptable en panel de cata. | **$5\text{ €}$** |

### Resultados en el Conjunto de Test ($N=1,300$ botellas):
* **Modelo Aleatorio:** Pérdida estimada de **$28,450\text{ €}$** ($21.88\text{ €}$/botella).
* **Baseline Ingenuo (Predecir siempre clase 6):** Pérdida de **$4,935\text{ €}$** ($3.80\text{ €}$/botella).
* **Odysseus ExtraTrees (Champion):** Pérdida de tan solo **$2,860\text{ €}$** ($2.20\text{ €}$/botella).
* **Impacto Financiero:**
  * **$+42.0\%$ de reducción directa en costes de reclasificación** frente a la moda.
  * **Cero (0) Premium Leaks**: Ningún lote de vino defectuoso es inadvertidamente etiquetado como Gran Reserva.

---

## 📋 5. Guía Prescriptiva para Enólogos y Bodegas

```text
========================================================================================
             FICHA PRESCRIPTIVA ENOLÓGICA DE CONTROL DE CALIDAD ODYSSEUS
========================================================================================
Variable Fisicoquímica       Rango Crítico Defectuoso     Rango Óptimo Gran Reserva
----------------------------------------------------------------------------------------
Alcohol (% vol)              < 10.2% vol                 >= 11.8% vol (Ideal: 12.5% - 13.5%)
Acidez Volátil (g/dm³)       > 0.45 g/dm³                <= 0.32 g/dm³ (Control estricto)
Sulfatos (g/dm³)             < 0.40 o > 1.00 g/dm³       0.65 - 0.85 g/dm³
Ratio SO2 Libre / Total      < 0.20 (oxidación)          0.30 - 0.45 (protección activa)
Acidez Total (g/dm³)         < 5.0 g/dm³ (plano)         6.5 - 8.5 g/dm³ (equilibrio tánico)
========================================================================================
```
