# 📊 Ticket Priority Classification
**Modelización Predictiva y Métricas de Negocio (TFM – Proyecto 1)**

---

## 1. Introducción y contexto del problema

### 1.1 Contexto de negocio
Un sistema de soporte IT gestiona incidencias mediante tickets con información textual y metadatos. Una mala priorización provoca tiempos de respuesta inadecuados, sobrecarga del equipo de soporte y riesgo de incumplir SLA. En este contexto, **no todos los errores tienen el mismo impacto**: fallar un ticket `High` es mucho más costoso que confundir un `Low`.

### 1.2 Objetivo de negocio
Objetivo principal: **reducir violaciones de SLA** priorizando correctamente los tickets críticos. Objetivos secundarios: reducir ruido operativo y mejorar el triaje.

Preguntas clave:
- ¿Qué tickets deben atenderse de forma inmediata?
- ¿Cuántos tickets críticos (High) soy capaz de detectar?
- ¿Qué coste tiene equivocarme en cada clase?

---

## 2. Definición del problema de Machine Learning

### 2.1 Tipo de problema
Clasificación supervisada multiclase (`high`, `medium`, `low`).
- No es regresión: las clases no representan un valor continuo.
- No se trata como ordinal puro: los costes no son lineales y el error `high→low` es más grave que `medium→low`.
- Cada clase tiene un peso de negocio distinto.

### 2.2 Variable objetivo
`Priority` define la urgencia del ticket:
- `high`: impacto crítico (SLA corto).
- `medium`: impacto moderado.
- `low`: consultas o mejoras.

La clase **High** es prioritaria aunque minoritaria.

---

## 3. Datos y preparación

### 3.1 Fuente de datos
Dataset público de Kaggle (tickets de soporte IT). Archivo: `data/raw/data.csv`.
- Registros: **29,651**
- Clases: `high=11,512`, `medium=12,126`, `low=6,013`
- Variables principales: `Body`, `Department`, `Tags`, `Priority`

### 3.2 Exploración inicial del dataset
- Desbalance moderado: la clase `low` es aproximadamente la mitad de `high/medium`.
- Texto libre con señales operativas (incidencias, errores, peticiones).
- Metadatos categóricos con baja cardinalidad (`Department`).

### 3.3 Limpieza y transformación
- Normalización de `Priority` a minúsculas y filtrado a `{low, medium, high}`.
- `Body` vacío → cadena vacía; `Department` nulo → `"Unknown"`.
- `Tags` parseado a lista; se usa `n_tags`.
- Features auxiliares: `n_tags`, `len_words`.

Estas variables son adecuadas porque combinan **señal semántica** (texto) con **contexto operativo** (departamento y volumen/longitud), lo que mejora la priorización realista.

---

## 4. Investigación y diseño experimental

### 4.1 Hipótesis de partida
- TF‑IDF + modelos clásicos puede ser competitivo con buen feature engineering.
- Embeddings deberían capturar semántica más rica, pero no siempre mejoran en datasets medianos.
- SetFit permite fine‑tuning eficiente sin coste de modelos grandes.
- Zero‑shot puede servir como benchmark pero no como modelo final.

Expectativas y riesgos:
- Es plausible que un baseline interpretable gane en macro‑F1 si el dataset tiene señales léxicas fuertes.
- Los LLMs pueden sobrepriorizar `high` o producir salidas inválidas.

### 4.2 Estrategia de comparación
Se comparan familias de modelos para justificar la elección final:
- Baselines clásicos (TF‑IDF).
- Embeddings con clasificadores lineales.
- Fine‑tuning con SetFit.
- Zero‑shot / Few‑shot con LLMs.

“Ganar” significa **mejor macro‑F1 y recall High** con coste operativo razonable y trazabilidad suficiente.

---

## 5. Enfoques y modelos evaluados

### 5.1 Enfoques explorados
1. TF‑IDF + modelos clásicos
2. TF‑IDF + OHE (texto + metadatos)
3. Embeddings + clasificadores lineales
4. Fine‑tuning con SetFit
5. Zero‑shot / Few‑shot con LLMs

### 5.2 Modelos utilizados
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- SetFit (Sentence Transformers)
- LLMs (Gemini / Ollama)

Se busca **valor de negocio**, no complejidad innecesaria.

---

## 6. Estrategia de evaluación y métricas

### 6.1 Principios generales
La evaluación se trata como un **sistema de decisión**, no como una única métrica:
- Robustez al desbalance (macro‑F1).
- Foco en clase `high` (recall alto).
- Interpretabilidad y trazabilidad.

### 6.2 Métricas técnicas principales

#### 6.2.1 Macro‑F1
- **Qué mide**: promedio no ponderado del F1 por clase.
- **Por qué es relevante**: evita que la clase mayoritaria domine el resultado.
- **Qué decisión habilita**: comparar modelos de forma justa cuando hay desbalance.

#### 6.2.2 Balanced Accuracy
- **Qué mide**: promedio de recalls por clase.
- **Por qué es relevante**: corrige la limitación del accuracy cuando hay clases desiguales.
- **Qué decisión habilita**: detectar modelos que ignoran clases minoritarias.

#### 6.2.3 Métricas por clase (High)
- **Precision (High)**: tasa de acierto al priorizar tickets críticos.
- **Recall (High)**: proporción de High detectados (clave para SLA).
- **F1 (High)**: equilibrio entre precisión y recall.

Estas métricas conectan directamente con riesgo SLA.

### 6.3 Métricas complementarias
- **Accuracy**: se reporta como referencia, pero no se optimiza.
- **Weighted F1**: aporta estabilidad global, pero puede ocultar fallos críticos.

### 6.4 Confusion Matrix
Permite cuantificar errores graves (High → Medium/Low) y verificar si el modelo está “sobrepriorizando” o subpriorizando tickets críticos.

**Resumen de métricas**: se prioriza macro‑F1 y recall High; el resto se usa para diagnóstico y trazabilidad.

---

## 7. Métricas específicas por enfoque

### 7.1 Embeddings / SetFit
- Balanced Accuracy en test.
- Recall High y análisis de errores High → Low.
- Cuando está disponible, se añade precisión High para evaluar sobrepriorización.

### 7.2 Zero‑shot / Few‑shot
- **Coverage / invalid rate**: porcentaje de salidas válidas.
- **Latencia y p95**: coste operativo del uso del LLM.
- **Over‑triage**: si el modelo dispara demasiados `high`.
- **Bootstrap CI**: estabilidad de macro‑F1 y recall High.

---

## 8. Métricas descartadas y justificación
- **ROC‑AUC multiclase**: baja interpretabilidad para negocio.
- **Log Loss**: difícil de traducir a impacto SLA.
- **MCC / Cohen’s Kappa**: redundantes frente a macro‑F1 y recall High.

---

## 9. Resultados experimentales

### 9.1 Resultados cuantitativos (resumen)

| Enfoque | Configuración | Accuracy | Macro‑F1 | Recall High | Fuente |
|---|---|---:|---:|---:|---|
| TF‑IDF + OHE + num | Linear SVM (mejor CV) | 0.771 | 0.762 | 0.811 | `reports/metrics_test.json` |
| Embeddings | all‑distilroberta‑v1 + body_tags_ohe_num + Linear SVM C=0.5 | — | 0.528 | 0.653 | `reports/experiments/embeddings_summary.csv` |
| SetFit (fine‑tuning) | all‑MiniLM‑L6‑v2 (best val) | 0.714 | 0.699 | 0.742 | `results/setfit/final_test/test_metrics.json` |
| Zero‑shot (LLM) | gemini‑2.5‑flash‑lite, few‑shot=5 | 0.320 | 0.284 | 0.206 | `reports/metrics_zero_shot.json` |

### 9.2 Análisis cualitativo
- **TF‑IDF** supera al resto en macro‑F1 y recall High en esta ejecución.
- **Embeddings** mejoran sobre modelos simples, pero no superan al baseline.
- **SetFit** ofrece un rendimiento consistente, pero no alcanza el TF‑IDF en macro‑F1.
- **Zero‑shot** queda como benchmark por baja precisión y alto invalid rate.

Esto confirma la hipótesis de que los baselines pueden ser suficientes con buenas features, y contradice la expectativa de que embeddings/SetFit siempre superan a TF‑IDF.

---

## 10. De métricas técnicas a métricas de negocio

### 10.1 Definición de métricas de negocio
Se usa un proxy simple de coste:
- Penalización fuerte para errores `high→medium/low`.
- Penalización menor para errores entre `medium` y `low`.

### 10.2 Impacto potencial del modelo
El objetivo es reducir falsos negativos en `high`. En este contexto, priorizar recall High es clave para minimizar riesgo de SLA.

---

## 11. Modelo final seleccionado

### 11.1 Criterios de selección
- Mejor macro‑F1 y recall High.
- Robustez y simplicidad operativa.
- Interpretabilidad y coste de despliegue.

### 11.2 Justificación final
**Modelo seleccionado: TF‑IDF + Linear SVM.**
En la ejecución actual es el que ofrece el mejor equilibrio entre macro‑F1 y recall High. SetFit y embeddings aportan valor semántico, pero no superan al baseline en métricas clave.

---

## 12. Explicabilidad del modelo
- TF‑IDF permite inspeccionar tokens con mayor peso por clase.
- La matriz de confusión ayuda a entender errores críticos.
- Limitación: no captura relaciones semánticas complejas como un modelo profundo.

---

## 13. Esquema del sistema completo
Ingesta → Limpieza → Feature Engineering → Train/Val/Test → Evaluación → Selección → Métricas negocio

---

## 14. Despliegue conceptual
- Tipo: API en tiempo real para priorización.
- Retraining: mensual o por caída de macro‑F1/recall High.
- Monitorización: drift en distribución de texto y categorías.

---

## 15. Dificultades encontradas y decisiones clave
- Desbalance de clases: se prioriza macro‑F1 y recall High.
- Zero‑shot con LLM: alto invalid rate y bajo rendimiento.
- SetFit: coste computacional mayor sin mejora clara en métricas.

---

## 16. Limitaciones y trabajo futuro
- Explorar más modelos base de embeddings.
- Probar prompts más robustos en zero‑shot.
- Ajustar costes de negocio con datos reales.
- Incorporar validación temporal (drift real).

---

## 17. Conclusiones
- Se ha construido un pipeline reproducible y trazable.
- TF‑IDF + Linear SVM es el mejor candidato en esta fase.
- El enfoque permite justificar decisiones técnicas con impacto de negocio.
