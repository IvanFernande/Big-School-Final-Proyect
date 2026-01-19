# Embeddings-based Classification: Methodology and Rationale

Este documento describe **cómo integrar y justificar** en el TFM la experimentación completa con **embeddings puros** y **embeddings + One-Hot Encoding (OHE)**, basándose en el estado actual del script `embeddings_experiments.py`.

El objetivo es doble:

1. Evaluar empíricamente **qué combinación de representación funciona mejor**.
2. Disponer de una base sólida para **explicar por qué funciona mejor** desde el punto de vista técnico y de dominio.

---

## 1. Motivación general

La clasificación de tickets de soporte se apoya en datos heterogéneos:

* Texto libre (`Body`, `Tags`)
* Campos categóricos (`Department`)
* Variables estructurales (`n_tags`, `len_words`)

No todos estos tipos de datos deben representarse de la misma forma. Este trabajo compara distintas estrategias de representación para responder a la siguiente pregunta:

> ¿Es suficiente la semántica textual capturada por embeddings o es necesario incorporar explícitamente la estructura del dominio?

---

## 2. Bloque A — Embeddings puros

### 2.1 Descripción

En este bloque, **todas las variables textuales se representan exclusivamente mediante embeddings densos**, sin introducir codificaciones categóricas explícitas.

Esto permite evaluar el poder predictivo de la semántica textual por sí sola.

### 2.2 Modelos de embeddings evaluados

El script compara múltiples modelos base con distintos compromisos entre calidad y coste:

* `all-MiniLM-L6-v2`: rápido y ligero (baseline)
* `all-mpnet-base-v2`: mayor capacidad semántica
* `all-distilroberta-v1`: punto intermedio
* `multi-qa-MiniLM-L6-cos-v1`: orientado a recuperación semántica

Para cada modelo se registra:

* Dimensión del embedding
* Tiempo de generación (train/test)
* Uso de caché

Esto permite analizar **rendimiento vs latencia**, aspecto clave en contextos industriales.

### 2.3 Estrategias de combinación textual

Con estos modelos se evalúan varias estrategias:

* `body_dept_concat`: concatenación directa de embeddings
* `body_dept_avg`: media simple tras normalización L2
* `body_dept_avg_w30 / w70 / w90`: medias ponderadas

La ponderación permite analizar **qué campo aporta más información predictiva** (texto libre vs metadato textual).

---

## 3. Bloque B — Embeddings + estructura explícita (OHE)

### 3.1 Motivación

El campo `Department`:

* es categórico
* tiene un vocabulario cerrado
* actúa como identificador funcional, no como lenguaje natural

Representarlo mediante embeddings puede introducir ruido semántico. Por ello, se evalúa una alternativa estructural.

### 3.2 Estrategia híbrida

En este bloque, la representación final combina:

* Embedding de `Body`
* One-Hot Encoding de `Department`
* Variables numéricas escaladas (`n_tags`, `len_words`)

Estrategia implementada:

* `body_ohe_num`

Esto permite que el clasificador aprenda **reglas explícitas por categoría**, complementando la semántica del texto.

---

## 4. Clasificadores y evaluación

Todas las estrategias se evalúan bajo el mismo marco:

* Clasificadores lineales (Linear SVM, Logistic Regression)
* Ponderación por desbalance de clases
* Cross-validation estratificada
* Métricas principales:

  * F1-macro
  * Recall y Precision de la clase High

Esta consistencia garantiza una **comparación justa entre representaciones**.

---

## 5. Organización experimental

El script implementa el siguiente flujo:

1. Generación y cacheo de embeddings
2. Construcción de variantes de features
3. Evaluación cruzada + test hold-out
4. Guardado de:

   * Resultados por experimento (JSON)
   * Resumen global (`embeddings_summary.csv`)
   * Benchmark de latencia (`embeddings_benchmark.csv`)

Esto asegura reproducibilidad y trazabilidad completa.

---

## 6. Interpretación y análisis futuro

Una vez identificado el mejor enfoque, los resultados permiten:

* Analizar coeficientes de modelos lineales
* Comparar estabilidad entre folds
* Relacionar mejoras métricas con decisiones de representación

Esto habilita la redacción de conclusiones del tipo:

> “El uso exclusivo de embeddings captura adecuadamente la semántica del texto, pero la incorporación explícita de variables categóricas mediante OHE mejora la estabilidad y el recall de la clase crítica, al reflejar de forma más fiel la estructura del dominio.”

---

## 7. Conclusión

Este enfoque experimental no persigue únicamente maximizar métricas, sino **comprender el impacto de cada decisión de representación**. La comparación sistemática entre embeddings puros y representaciones híbridas proporciona una base sólida para justificar el diseño final del modelo y su alineación con el problema de negocio.
