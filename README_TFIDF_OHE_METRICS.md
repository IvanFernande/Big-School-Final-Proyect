# TF-IDF + OHE — Metrics Strategy and Rationale

Este documento describe **qué métricas se utilizan en cada fase del pipeline TF-IDF + OHE**, por qué se utilizan esas métricas y **por qué se descartan otras**.  
Su objetivo es dejar cerrada la **estrategia de evaluación**, tanto técnica como orientada a negocio.

---

## 1. Principios generales de evaluación

La evaluación del modelo se diseña siguiendo estos principios:

- Separación estricta entre **selección de modelo** y **evaluación final**
- Uso de métricas **robustas al desbalance**
- Priorización de errores con **impacto operativo real**
- Evitar métricas redundantes o poco interpretables

No se busca maximizar una única métrica, sino **evaluar el modelo desde distintos niveles**.

---

## 2. Nivel 1 — Evaluación para selección de modelo (Cross-Validation)

### Dónde se evalúa
- Cross-validation estratificada
- Solo sobre el conjunto de entrenamiento (80 %)

### Objetivo
Seleccionar:
- el mejor modelo (SVM, Logistic Regression, Naive Bayes)
- la mejor representación de features (Department encoding)

---

### Métrica principal: **macro-F1**

**Qué mide**  
El F1-score calculado de forma independiente por clase y promediado sin ponderación.

**Por qué se utiliza**
- El dataset presenta desbalance entre clases
- Todas las prioridades (low, medium, high) deben rendir correctamente
- Evita que la clase mayoritaria domine la evaluación

**Qué pregunta responde**
> ¿Qué modelo mantiene el mejor equilibrio global entre clases?

---

### Métricas descartadas en CV

No se utilizan en esta fase:

- Accuracy → engañosa con desbalance
- Recall de High → demasiado específica para selección global
- Confusion matrix → ruido elevado en CV
- Weighted-F1 → vuelve a favorecer la clase mayoritaria

La CV se utiliza exclusivamente para **comparabilidad y estabilidad**, no para análisis operativo.

---

## 3. Nivel 2 — Evaluación final del modelo (Test hold-out)

### Dónde se evalúa
- Conjunto de test independiente (20 %)
- Una única vez, tras cerrar todas las decisiones

### Objetivo
Estimar el rendimiento **realista del modelo en producción**.

---

### Métricas reportadas

#### 1. **macro-F1 (test)**
- Permite comparar generalización respecto a CV
- Mantiene coherencia con la métrica de selección

---

#### 2. **Recall de la clase High**
Métrica clave desde el punto de vista de negocio.

**Por qué**
- Un falso negativo en High implica riesgo de SLA
- Se prioriza no perder tickets críticos
- Acepta un mayor número de falsos positivos si es necesario

**Qué pregunta responde**
> ¿Cuántos tickets realmente críticos detecta el sistema?

---

#### 3. **Accuracy (secundaria)**
- Se reporta como referencia global
- No se utiliza para tomar decisiones

---

#### 4. **Confusion Matrix**
- Análisis cualitativo de errores
- Identificación de confusiones críticas (High → Low)
- Se genera para el modelo cargado en `MODEL_PATH` (debe corresponder al TF-IDF si se quiere atribuir a este baseline)

No se utiliza como métrica agregada, sino como **herramienta de diagnóstico**.

---

## 4. Nivel 3 — Métricas de negocio (futuro)

Estas métricas **no se calculan aún** en el baseline TF-IDF. Quedan definidas como trabajo futuro para el modelo final del proyecto.

### Qué se mediría (cuando se implemente)
- Falsos negativos de High (FNₕ)
- Falsos positivos de High (FPₕ)
- Ratio de High detectados correctamente (recall High)
- % de violaciones de SLA evitadas
- Coste operativo estimado

Estas métricas permitirán traducir el rendimiento técnico a **impacto empresarial** cuando se integre la evaluación de negocio.

---

## 5. Por qué no se utiliza un esquema 70/20/10

Aunque podría usarse un split 70/20/10, se opta por:

- Cross-validation estratificada para selección de modelo
- Test hold-out independiente para evaluación final

Este enfoque:
- reduce varianza frente a un único validation set
- maximiza el uso de datos disponibles
- mantiene el test completamente aislado

---

## 6. Métricas descartadas y justificación

No se utilizan:

- ROC / AUC → no alineadas con el problema multiclase
- Precision High → menos relevante que recall para SLA
- Weighted-F1 → oculta el desbalance
- Métricas por fold → difícil interpretación

Estas decisiones reducen complejidad sin pérdida de información relevante.

---

## 7. Conclusión

La estrategia de métricas del baseline TF-IDF + OHE:

- Está alineada con el desbalance del dataset
- Refleja prioridades de negocio reales
- Evita métricas engañosas o redundantes
- Permite una evaluación clara, reproducible y defendible

Este esquema de métricas queda **cerrado** y sirve como base para comparaciones con modelos más avanzados.
