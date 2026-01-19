# Métricas recomendadas para `embeddings_experiments.py`

Este documento indica **qué métricas añadir o evitar** en tu script actual de experimentación con embeddings, y por qué.  
Está escrito para que puedas comparar **(a) entre modelos de embeddings** y **(b) contra otros enfoques** (TF‑IDF+OHE, SetFit, etc.) de forma justa y defendible.

El script actual ya guarda (por experimento): **CV F1‑macro (mean/std)** y en test **F1‑macro**, **recall_high** y **precision_high**, además del `classification_report` completo en JSON. fileciteturn6file0

---

## 1) Métricas que mantendría (ya las usas y están bien)

### A. F1‑macro (CV y test) — *métrica principal técnica*
- **Por qué:** en multiclase desbalanceado trata a cada clase por igual (no deja que Medium/Low “tapen” High).
- **Uso recomendado:** seguir usándola como métrica principal para ranking técnico y comparación entre modelos.

En tu script ya se usa como `scoring="f1_macro"` en CV y se exporta a summary. fileciteturn6file0

### B. Precision y Recall de **High** (test) — *métricas de clase crítica*
- **Por qué:** High suele ser la clase con mayor impacto de negocio (FN de High es caro; FP de High genera sobrecarga).
- **Uso recomendado:** mantener ambas; si tienes que priorizar, prioriza **Recall_high** (seguridad) y vigila que Precision_high no caiga demasiado.

Ya se guardan en `embeddings_summary.csv/json`. fileciteturn6file0

### C. `classification_report` completo en JSON — *trazabilidad*
- **Por qué:** te permite reconstruir métricas por clase sin recalcular.
- **Uso recomendado:** mantenerlo como “artefacto reproducible”.

---

## 2) Métricas que añadiría (para que quede “redondo” y comparable)

### 1) Balanced Accuracy (CV y test)
- **Qué aporta:** rendimiento global equilibrado por clase, muy interpretable con desbalance.
- **Por qué añadirla si ya tienes F1‑macro:** te da una perspectiva complementaria “por clase” sin penalizar tanto por precision/recall extremos.

**Implementación:** `balanced_accuracy_score(y_true, y_pred)` en test, y en CV puedes usar `scoring="balanced_accuracy"` (ideal: guardarlo como columna adicional).

### 2) Matriz de confusión (test) + derivados
- **Qué aporta:** identifica el *tipo* de error, que es crucial en priorización:
  - High → Medium (malo)
  - High → Low (mucho peor)
- **Qué guardar:** o bien la matriz completa (3x3) en JSON, o columnas tipo:
  - `conf_high_to_med`, `conf_high_to_low`, etc.

**Implementación:** `confusion_matrix(y_test, y_pred, labels=[...])`.

### 3) Métrica “High vs Rest” (recomendada si tu foco es detectar High)
**Opción ideal:** **Average Precision (PR‑AUC)** para High vs Rest  
- **Por qué:** PR‑AUC es más informativa que ROC‑AUC cuando hay desbalance.
- **Requisitos:** necesitas scores:
  - `LinearSVC.decision_function(X)`
  - `LogReg.predict_proba(X)` (o `decision_function` si lo soporta)

**Alternativa simple (sin AUC):** F1/Recall/Precision para High vs Rest (binario) usando `y_pred == "high"`.

> Nota: para comparar con SetFit (que suele dar scores), PR‑AUC es especialmente útil.

### 4) Cost Score (métrica de negocio) — *si quieres cerrar el círculo del TFM*
Define una **matriz de costes** (penalización) coherente con SLA/coste. Ejemplo (ajústalo a tu caso):
- FN(High): 10
- High→Low: 12
- High→Medium: 8
- FP(High): 3
- Medium↔Low: 1

Luego calcula:
- `cost_total` y `cost_per_ticket = cost_total / N`

**Por qué:** te permite elegir un “mejor modelo” con criterio operativo, no solo por métrica técnica.

---

## 3) Métricas que NO usaría como principales (y por qué)

### A) Accuracy
- **Por qué no:** en desbalance puede ser alta aunque falles High; no refleja el objetivo.

### B) Micro‑F1
- **Por qué no:** en multiclase suele comportarse parecido a accuracy (domina la clase mayoritaria).

### C) ROC‑AUC multiclase (como métrica principal)
- **Por qué no:** menos interpretable para priorización y menos útil que PR‑AUC cuando High es minoritaria.
- **Cuándo sí:** como métrica secundaria si necesitas comparar “separabilidad” general con scores calibrados.

---

## 4) Propuesta mínima de columnas nuevas en `embeddings_summary.csv`

Si quieres mejorar sin inflar demasiado el CSV, añadiría estas columnas (además de las actuales):  
- `test_balanced_accuracy`
- `test_conf_high_to_med`
- `test_conf_high_to_low`
- `test_high_vs_rest_ap` *(si guardas scores y calculas PR‑AUC)*
- `test_cost_per_ticket` *(si defines cost matrix)*

Y opcional (si quieres consistencia con CV):  
- `cv_balanced_accuracy_mean`, `cv_balanced_accuracy_std`

---

## 5) Cómo decidir “el mejor” modelo con estas métricas

**Ranking técnico (primario):**
1. `cv_f1_macro_mean` (y mirar std para estabilidad)
2. desempate con `test_f1_macro`

**Guardarraíles (clase crítica):**
- exigir `test_recall_high` ≥ umbral (p.ej. 0.75) o comparar top‑k por recall_high

**Negocio (si lo incluyes):**
- elegir el mínimo `test_cost_per_ticket` entre modelos que cumplan el guardarraíl de High.

---

## 6) Resumen ejecutivo

- **Mantén:** F1‑macro + precision/recall de High + report JSON. fileciteturn6file0  
- **Añade (recomendado):** Balanced Accuracy + Confusion Matrix (y mejor aún PR‑AUC High vs Rest).  
- **Añade (negocio):** Cost Score si quieres una conclusión orientada a SLA/coste.  
- **Evita como principal:** Accuracy, Micro‑F1, ROC‑AUC multiclase.

