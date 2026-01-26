# 📊 Ticket Priority Classification
Proyecto de clasificación automática de tickets de soporte IT en prioridades **high / medium / low**, combinando NLP, metadatos operativos y métricas de negocio (SLA / coste).

---

## 1. Introducción y contexto del problema

### 1.1 Contexto de negocio
Un sistema de soporte IT gestiona incidencias mediante tickets con información textual y metadatos. Una mala priorización provoca tiempos de respuesta inadecuados, sobrecarga del equipo de soporte y riesgo de incumplimiento de SLA (Acuerdo de Nivel de Servicio).

En este contexto, **no todos los errores tienen el mismo impacto**: fallar un ticket `High` es mucho más costoso que confundir un `Low`. Por tanto, el problema no puede evaluarse únicamente con métricas globales.

### 1.2 Objetivo de negocio
Objetivo principal: **reducir violaciones de SLA** priorizando correctamente los tickets, teniendo más importancia la correcta clasificación de aquellos que sean críticos (`High`).  
Objetivos secundarios:
- Reducir ruido operativo.
- Mejorar el triaje inicial.
- Disminuir el coste asociado a errores graves de priorización.

Este objetivo se aborda mediante un **sistema de clasificación supervisada**, entrenado con datos históricos de tickets.

---

## 2. Definición del problema de Machine Learning

### 2.1 Tipo de problema
Clasificación supervisada multiclase (`high`, `medium`, `low`).

- No es regresión: las clases no representan un valor continuo.
- No se trata como ordinal puro: aunque hay orden (low < medium < high), el coste del error no es lineal ni simétrico (fallar un `high` es mucho más grave).
- Cada clase tiene un peso de negocio distinto.

### 2.2 Variable objetivo
`Priority` define la urgencia del ticket:
- `high`: impacto crítico.
- `medium`: impacto moderado.
- `low`: consultas o mejoras.

La clase **High** es prioritaria aunque no sea la más abundante, porque es la que más impacta en SLA.

---

## 3. Datos y preparación

### 3.1 Fuente de datos
Dataset público de Kaggle (tickets de soporte IT).

- Registros totales: **29 651**
- Variables principales: `Body`, `Department`, `Tags`, `Priority`

### 3.2 Distribución de clases

| Clase | Nº tickets | % |
|------|------------|---|
| High | 11 512 | 38.8 |
| Medium | 12 126 | 40.9 |
| Low | 6 013 | 20.3 |

### 3.3 Exploración inicial del dataset
- Desbalance moderado: la clase `Low` es aproximadamente la mitad de `High/Medium`.
- Texto libre con señales operativas claras.
- Cardinalidad moderada en `Department`.

**Figura 3.1 — Distribución de prioridades**  
![EDA Priority](reports/figures/eda_priority_distribution.png)

**Figura 3.2 — Departamentos más frecuentes**  
![EDA Department Top 10](reports/figures/eda_department_top10.png)

**Figura 3.3 — Longitud del texto por clase**  
![EDA Len Words Box](reports/figures/eda_len_words_box.png)

### 3.4 Limpieza y feature engineering
- Normalización de `Priority` a minúsculas.
- `Body` vacío → cadena vacía.
- `Department` nulo → `"Unknown"`.
- `Tags` parseado a lista.
- Variables derivadas: `n_tags`, `len_words`.

Estas variables combinan **señal semántica** y **contexto operativo**, alineadas con el problema real.

---

## 4. Estrategia experimental

Se comparan **familias de modelos**, no solo configuraciones individuales:
- TF-IDF + modelos clásicos.
- Embeddings + clasificadores lineales.
- Fine-tuning con SetFit.
- Zero-shot / Few-shot con LLMs.

“Ganar” significa maximizar el **recall de la clase `High`** (métrica primaria de negocio), manteniendo un buen equilibrio global entre clases medido mediante **macro-F1** (métrica técnica principal), con coste operativo razonable y trazabilidad.

---

## 5. Estrategia global de evaluación y métricas

**Principios**: separar selección vs test, robustez al desbalance, foco en `High`.

### 5.1 Métricas utilizadas (definición y uso)

**Macro‑F1 (principal técnica)**  
Promedio no ponderado del F1 por clase.  
Fórmula: `F1 = 2 * (precision * recall) / (precision + recall)`  
Se usa para comparar modelos porque **no deja que la clase mayoritaria domine**.

**Recall High (principal de negocio)**  
Proporción de tickets `High` correctamente detectados.  
Ejemplo: si hay 100 `High` reales y se detectan 81 → recall = 0.81.

**Precision High (control operativo)**  
De los tickets predichos como `High`, cuántos lo son realmente.  
Ejemplo: si el modelo marca 50 como `High` y 30 son correctos → precision = 0.60.

**F1 High (equilibrio en clase crítica)**  
Media armónica entre precisión y recall en `High`.  
Evita optimizar solo recall si la precisión cae demasiado.

**Balanced Accuracy (diagnóstico)**  
Promedio del recall por clase. Útil para detectar si el modelo **ignora una clase**, pero no se usa para seleccionar el ganador.

**Accuracy / Weighted‑F1 (contexto)**  
Accuracy = proporción total de aciertos.  
Weighted‑F1 = F1 ponderado por tamaño de clase; puede ocultar fallos en `High`.

**Confusion Matrix (impacto de error)**  
Permite cuantificar errores críticos: `High→Medium` y `High→Low`.

**Métricas de negocio**  
Traducción a impacto (gap de horas) se detalla en la sección 9.

### 5.2 Protocolo de partición (80/20 vs 70/20/10)

- **80/20 + CV**: se usa cuando entrenar es barato (TF‑IDF, embeddings). La selección se hace con validación cruzada y el 20% queda como test aislado.  
- **70/20/10**: se usa cuando el método requiere validación fija o es costoso (SetFit). Permite elegir hiperparámetros sin contaminar el test.

---

## 6. Exploración de enfoques y resultados

### 6.1 TF-IDF + modelos clásicos (baseline principal)

Resumen del enfoque: **TF-IDF (Body) + metadatos + clasificador lineal** por interpretabilidad y coste bajo.

**Vectorizador (final)**: `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`, `smooth_idf=True`.

**Qué significa cada parámetro**
- `ngram_range=(1,2)`: usa unigramas y bigramas para capturar términos y expresiones frecuentes.
- `min_df=2`: descarta términos que aparecen en menos de 2 documentos, reduciendo ruido.
- `sublinear_tf=True`: evita que palabras muy repetidas dominen el vector.
- `smooth_idf=True`: suaviza IDF para estabilizar pesos en términos raros.

**Metadatos**:
- `Department` como OHE (OHE agrupado ≈ OHE completo; hashing no mejora).
- `Tags` no como texto; se usa `n_tags`.
- `len_words` como señal de longitud.

**Modelos evaluados**

Se entrenan y comparan los siguientes clasificadores:

- Linear SVM
- Logistic Regression
- Multinomial Naive Bayes

**Estrategias de `Department` (Linear SVM fijo, CV)**:

| Estrategia | Macro‑F1 |
|---|---:|
| OHE completo | 0.706 |
| OHE agrupado | 0.706 |
| Hashing (`n_hash=2048`) | 0.706 |

Conclusión: diferencias marginales; se mantiene OHE completo por simplicidad y trazabilidad.

**Estrategia de evaluación**

**División de datos**
- Split **80 / 20** (train / test).
- Estratificación por `Priority`.
- El conjunto de test se mantiene **completamente aislado** hasta la evaluación final.

**Justificación del uso de validación cruzada**

La validación cruzada estratificada se utiliza para la selección de modelos y configuraciones porque ofrece estimaciones más estables que un único split.

---

**Resultados de selección (Cross-Validation)**

Durante la validación cruzada estratificada sobre el conjunto de entrenamiento se obtuvieron los siguientes resultados medios (macro-F1):

- **Linear SVM**: 0.706  
- **Logistic Regression**: 0.645  
- **Multinomial Naive Bayes**: 0.413  

---
**Resultados obtenidos (TF-IDF)**

Una vez determinado que **Linear SVM** obtiene el mejor macro‑F1 en validación, se evalúa en test.

**Mejor configuración seleccionada**
- Modelo: **Linear SVM**
- Representación: TF-IDF + metadatos
- Estrategia de `Department`: **OHE completo**

**Resultados en conjunto de test**
- Macro-F1 **0.762**
- Recall High **0.811**
- Accuracy **0.771**

Además de estas metricas para definir el rendimiento del modelo, se obtiene una matriz de confusión:

![TF-IDF Confusion Matrix](reports/figures/confusion_matrix.png)

La matriz de confusión muestra que:
- la mayoría de errores de `High` se producen hacia `Medium`,
- los errores críticos `High → Low` son minoritarios.

Este patrón es coherente con el objetivo de negocio, ya que **prioriza no perder tickets críticos**, incluso a costa de cierta sobre-priorización.

---

**Conclusión del enfoque TF-IDF**

Esta configuración se selecciona por ofrecer el mejor equilibrio entre macro-F1 y recall de la clase High, manteniendo una baja tasa de errores críticos (High→Low) y un coste computacional reducido.

El enfoque destaca por su interpretabilidad y estabilidad, lo que permite auditar y explicar las decisiones de priorización en un entorno operativo, facilitando su adopción y la confianza en la automatización.

Por estos motivos, se establece como baseline principal y referencia para la comparación de los enfoques posteriores.

### 6.2 Embeddings + clasificadores lineales

**Objetivo**: evaluar si embeddings preentrenados mejoran la semántica frente a TF‑IDF manteniendo clasificadores lineales.
Los encoders se usan **como extractores de características** (sin fine‑tuning) para aislar el efecto de la representación.

**Encoders evaluados**:
- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `all-distilroberta-v1`
- `multi-qa-MiniLM-L6-cos-v1`

**Clasificadores**: Linear SVM y Logistic Regression (`class_weight="balanced"`, `C ∈ {0.5, 1.0, 2.0}`), CV para selección y test aislado para métricas finales.
Se mantiene el **mismo protocolo** que en TF‑IDF para que las diferencias provengan de la representación y no del split.

#### A) Embeddings “puros” (solo texto)
Estrategias: `concat`, `avg`, pesos (70/30, 30/70, 90/10). Se evalúa el aporte incremental de `Department` y `Tags`.

**Comparativa de estrategias (ejemplo: all-mpnet-base-v2, Linear SVM, C=2.0)**

| Estrategia (Body + Dept) | Macro-F1 | Recall High |
|---|---:|---:|
| concat | 0.524 | 0.636 |
| avg | 0.519 | 0.630 |
| w30 | 0.518 | 0.636 |
| w70 | 0.520 | 0.637 |
| w90 | 0.512 | 0.621 |

| Estrategia (Body + Tags) | Macro-F1 | Recall High |
|---|---:|---:|
| concat | 0.489 | 0.579 |
| avg | 0.465 | 0.544 |
| w30 | 0.447 | 0.539 |
| w70 | 0.471 | 0.547 |
| w90 | 0.471 | 0.557 |

**Figura 6.2-A — Estrategias de combinación (solo texto)**  
![Embeddings estrategias A](reports/figures/embeddings_strategies_A.png)

Conclusión A: diferencias pequeñas; `Tags` aporta señal limitada frente a `Body`.

#### B) Embeddings + estructura explícita (OHE)
Se combina `Body` (y `Tags` cuando procede) con `Department` en OHE + numéricas.

**Ejemplo (all-mpnet-base-v2, Linear SVM, C=2.0):**

| Estrategia (Body + Dept) | Macro-F1 | Recall High |
|---|---:|---:|
| concat (Dept embebido) | 0.524 | 0.636 |
| concat + num | 0.525 | 0.636 |
| OHE + num | 0.525 | 0.636 |

| Estrategia (Body + Tags) | Macro-F1 | Recall High |
|---|---:|---:|
| concat (Tags embebidos) | 0.489 | 0.579 |
| Tags embebidos + Dept OHE + num | 0.526 | 0.650 |

**Figura 6.2-B — Estrategias con estructura explícita (OHE)**  
![Embeddings estrategias B](reports/figures/embeddings_strategies_B.png)

Conclusión B: OHE + num mejora cuando se añaden campos auxiliares (especialmente `Tags`).

#### Resultados (mejor configuración por encoder)

| Modelo de embeddings | Estrategia | C | Macro-F1 | Recall High | AP High | Pred High (%) |
|---------------------|------------|---:|----------|-------------|---------|---------------|
| all-MiniLM-L6-v2 | body_tags_ohe_num | 2.0 | 0.511 | 0.648 | 0.657 | 41.8 |
| all-distilroberta-v1 | body_ohe_num | 1.0 | 0.521 | 0.656 | 0.671 | 42.1 |
| all-mpnet-base-v2 | body_ohe_num | 2.0 | 0.525 | 0.636 | 0.673 | 40.7 |
| multi-qa-MiniLM-L6-cos-v1 | body_dept_concat | 1.0 | 0.511 | 0.643 | 0.650 | 41.6 |

**Métricas específicas**: `AP High` (PR‑AUC High vs Rest) y `Pred High (%)` para monitorizar sobre‑priorización.

**Figura 6.2-C — Matriz de confusión (embeddings)**  
![Embeddings confusion](reports/figures/embeddings_confusion_matrix.png)

**Resultado**: no se supera al baseline TF‑IDF en macro‑F1 ni recall High; mayor complejidad y coste.
En términos operativos, el uso de embeddings **no reduce de forma consistente** los errores críticos en `High`, y el ligero avance en algunas métricas (AP/Pred High) no compensa la pérdida de interpretabilidad y el aumento de complejidad del pipeline.

### 6.3 Fine-tuning con SetFit

SetFit ajusta un encoder de frases con **aprendizaje contrastivo** (pares positivos/negativos) y entrena un clasificador ligero.  
Se usa **split 70/20/10** (train/val/test) por coste de entrenamiento y para evitar fugas al generar pares.

**Grid evaluado**: iteraciones {5,10}, épocas {1,2}, batch {16,32} (GPU) / {8,16} (CPU), LR {2e‑5, 5e‑5}.  
Se prueba solo `sentence-transformers/all-MiniLM-L6-v2` para reducir coste computacional y tiempo de ejecución, pero un trabajo a futuro es probar en el grid otros modelos.

**Selección** en validación con `selection_score = 0.7·macro_f1 + 0.3·recall_high` para mantener equilibrio global sin perder detección de `High`.

**Balanceo (validación)**
| Estrategia | Macro-F1 | Recall High | Precision High | Selection Score |
|---|---:|---:|---:|---:|
| none | 0.515 | 0.643 | 0.659 | 0.554 |
| oversample | 0.538 | 0.457 | 0.656 | 0.514 |
| downsample | 0.540 | 0.501 | 0.667 | 0.528 |

**Figura 6.3-B — Balanceo en SetFit (validación)**  
![SetFit Balance](reports/figures/setfit_balance_comparison.png)

Conclusión: oversample/downsample suben macro‑F1 pero **reducen recall High**; se mantiene `none`.

**Top configuraciones (validación)**
| Configuración | Iter | Epochs | Batch | LR | Macro-F1 (val) | Recall High (val) | Selection Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 10 | 2 | 32 | 5e-05 | 0.644 | 0.733 | 0.671 |
| all-MiniLM-L6-v2 | 10 | 2 | 16 | 5e-05 | 0.648 | 0.703 | 0.664 |
| all-MiniLM-L6-v2 | 10 | 2 | 16 | 2e-05 | 0.589 | 0.717 | 0.627 |
| all-MiniLM-L6-v2 | 5 | 2 | 16 | 5e-05 | 0.565 | 0.695 | 0.604 |
| all-MiniLM-L6-v2 | 10 | 2 | 32 | 2e-05 | 0.565 | 0.678 | 0.599 |

**Figura 6.3-A — Top configuraciones (validación)**  
![SetFit Top Configs](reports/figures/setfit_top_configs.png)

**Resultados (test)**  
Modelo: `all-MiniLM-L6-v2`, iter=10, epochs=2, batch=32, LR=5e‑05

| Métrica | Valor |
|--------|-------|
| Macro F1 | 0.699 |
| Recall High | 0.742 |
| Precision High | 0.770 |
| Accuracy | 0.714 |

![SetFit Confusion Matrix](results/setfit/final_test/confusion_matrix.png)

**Resultado**: rendimiento competitivo, pero **no supera a TF‑IDF** y añade coste/complexidad.

En conclusión, SetFit se considera una **alternativa robusta** cuando se prioriza adaptación semántica, pero no desplaza al baseline por coste y trazabilidad.

### 6.4 Zero-shot / Few-shot con LLMs

Objetivo: **benchmark exploratorio** (sin entrenamiento). Se evalúa sensibilidad a prompt y número de ejemplos.

**Modelos**: DeepSeek‑R1:8B (local) y Gemini 2.5 Flash Lite (cloud).  
**Prompt**: `base` y `rules`.  
**Métricas**: macro‑F1, recall/precision High, coverage, latencia, over‑triage.

**Prompts utilizados (plantilla)**

`base`:
```
Eres un asistente que clasifica tickets en: high, medium o low.

[Ejemplos opcionales]

Ahora clasifica este ticket:
Department: <dept>
Body: <body>
Priority (responde solo high, medium o low):
```

`rules`:
```
Clasifica tickets en: high, medium o low siguiendo estas reglas:
- High: caida de servicio, bloqueo total, seguridad, perdida de datos, urgencia critica.
- Medium: degradacion, errores intermitentes, retrasos relevantes, impacto moderado.
- Low: preguntas, peticiones menores, mejoras, impacto bajo.
Si dudas entre high y medium, elige high.

[Ejemplos opcionales]

Ahora clasifica este ticket:
Department: <dept>
Body: <body>
Priority (responde solo high, medium o low):
```

#### Sensibilidad al número de ejemplos (DeepSeek)

| Prompt | #Ejemplos | Macro-F1 | Recall High | Precision High | Pred High (%) |
|------|-----------|----------|-------------|----------------|-------------|
| base | 0 | 0.335 | 0.450 | 0.425 | 35.3 |
| base | 3 | 0.404 | 0.270 | 0.435 | 20.7 |
| base | 6 | 0.386 | 0.200 | 0.541 | 12.3 |

Conclusión: al aumentar ejemplos *few‑shot*, baja el **recall High** y también `Pred High (%)`; mejora la precisión, pero el trade‑off penaliza la detección de tickets críticos.

#### Sensibilidad al prompt (DeepSeek)

| Prompt | #Ejemplos | Macro-F1 | Recall High | Coverage |
|-------|-----------|----------|-------------|-------------|
| base | 6 | 0.386 | 0.200 | 1.00 |
| rules | 6 | 0.399 | 0.360 | 1.00 |

Conclusión: el prompt `rules` mejora claramente el **recall High** frente a `base` manteniendo la cobertura, aunque sin alcanzar niveles competitivos.

#### Estabilidad entre ejecuciones (DeepSeek rules 6)

| Seed | Macro-F1 | Recall High |
|------|----------|-------------|
| 0 | 0.400 | 0.460 |
| 1 | 0.383 | 0.470 |
| 2 | 0.370 | 0.410 |
| **Media ± Std** | **0.385 ± 0.012** | **0.447 ± 0.026** |

**Figura G4 — Estabilidad entre ejecuciones**  
![Figura G4](reports/figures/fig_g4_llm_estabilidad.png)

#### Comparación entre modelos (DeepSeek vs Gemini)

| Modelo | Prompt | #Ejemplos | Macro-F1 | Recall High | Coverage | Latencia media (ms) |
|-------|--------|-----------|----------|-------------|----------|---------------------|
| DeepSeek-R1:8B | rules | 6 | 0.399 | 0.360 | 1.00 | 4466.97 |
| Gemini 2.5 Flash Lite | rules | 6 | 0.394 | 0.382 | 1.00 | 529.40 |

**Figura G2 — Trade-off Precision vs Recall (LLMs)**  
![Figura G2](reports/figures/fig_g2_llm_precision_recall.png)

**Conclusión del enfoque LLMs**: rendimiento muy inferior a modelos supervisados y alta variabilidad.  
Se mantiene como **referencia exploratoria**, no como candidato productivo.

## 7. Comparativa global y selección del modelo final

En esta sección se sintetizan los resultados obtenidos para los **mejores modelos seleccionados dentro de cada familia de enfoques**, evaluados sobre el conjunto de test con las mismas métricas.

La comparativa se centra en cuatro dimensiones clave:
- **Rendimiento global** (macro-F1),
- **Detección de tickets críticos** (recall de la clase High),
- **Viabilidad operativa** (coste computacional y latencia),
- **Alineación con el objetivo de negocio** (impacto en SLA).

El objetivo no es únicamente maximizar métricas técnicas, sino seleccionar un modelo **robusto, interpretable y operativo** en un entorno real de soporte IT.

---

### 7.1 Resumen comparativo por enfoque

La siguiente tabla resume los resultados más representativos de cada enfoque:

**Figura G1 — Comparativa global por enfoques (Macro-F1 / Recall High)**  
![Figura G1](reports/figures/fig_g1_comparativa_global.png)

| Enfoque | Modelo seleccionado | Macro-F1 (test) | Recall High (test) | Complejidad | Rol en el proyecto |
|-------|---------------------|------------------|--------------------|-------------|-------------------|
| TF-IDF | Linear SVM + metadatos | **0.762** | **0.811** | Baja | ✅ Modelo final |
| Embeddings | all-mpnet-base-v2 + body_ohe_num + Linear SVM (C=2.0) | 0.525 | 0.636 | Media | Referencia avanzada |
| SetFit | all-MiniLM-L6-v2 fine-tuned | 0.699 | 0.742 | Alta | Alternativa robusta |
| Zero / Few-shot | DeepSeek-R1:8B (rules, 6) | 0.399 | 0.360 | Muy alta | Benchmark |

**Notas de interpretación**:
- TF-IDF ofrece el mejor equilibrio entre rendimiento global y protección de la clase crítica.
- SetFit logra resultados competitivos, pero con mayor complejidad y coste.
- Los embeddings densos no superan al modelo final seleccionado en métricas clave de negocio.

---

### 7.2 Análisis comparativo cualitativo

Desde una perspectiva operativa y de negocio:

- **TF-IDF + Linear SVM** destaca por:
  - alta interpretabilidad,
  - estabilidad entre ejecuciones,
  - bajo coste computacional,
  - y control explícito del impacto de errores críticos.

- **Embeddings + clasificadores lineales** aportan mayor capacidad semántica, pero:
  - no reducen de forma consistente los errores `High → Low`,
  - incrementan la complejidad del pipeline,
  - y no mejoran el impacto en SLA.

- **SetFit** muestra una buena adaptación al dominio y una estructura de errores coherente, pero:
  - requiere entrenamiento contrastivo,
  - introduce mayor coste y complejidad,
  - y no justifica el cambio frente al modelo final seleccionado más simple.

- **LLMs en zero-shot / few-shot** presentan:
  - alta dependencia del prompt,
  - variabilidad entre ejecuciones,
  - mayor latencia y menor control operativo,
  - lo que limita su uso como sistema automático de priorización.

---

### 7.3 Modelo seleccionado

A partir del análisis cuantitativo y cualitativo, se selecciona como **modelo final del proyecto**:

**➡️ TF-IDF + Linear SVM + metadatos operativos**

Esta decisión se fundamenta en:

- el mejor equilibrio entre **macro-F1** y **recall de la clase High**,
- la minimización de errores con impacto directo en SLA,
- el menor coste computacional y latencia,
- y la alta interpretabilidad del modelo, clave en entornos operativos.

El modelo seleccionado cumple el objetivo principal del proyecto:  
**priorizar correctamente los tickets críticos reduciendo el riesgo de incumplimiento de SLA**, sin introducir complejidad innecesaria en el sistema.

---

### 7.4 Profundización en el modelo ganador (TF-IDF)

- **Errores críticos**: la mayoría de fallos de `High` son `High→Medium`; `High→Low` es minoritario (ver matriz de confusión).  
- **Interpretabilidad**: pesos de n‑gramas permiten auditar decisiones (ver sección 8).  
- **Mejoras inmediatas** (futuro): ajustar umbral de `High` y métricas por segmentos (`Department`).

---

## 8. Explicabilidad del modelo

### 8.1 Global (TF-IDF)
TF‑IDF + Linear SVM es **intrínsecamente interpretable**: los coeficientes indican qué n‑gramas empujan cada clase.  
Los términos mostrados corresponden a los n‑gramas con mayor peso absoluto del clasificador.

**Interpretabilidad vs explicabilidad**:  
- *Interpretabilidad* = entender el comportamiento global del modelo a partir de su estructura (coeficientes).  
- *Explicabilidad* = justificar decisiones individuales (post‑hoc).

Los pesos mostrados son **coeficientes del clasificador lineal** sobre la representación TF‑IDF. Un peso positivo favorece esa clase; uno negativo la penaliza.  

Si aparecen términos estructurales (p. ej., `Department` o variables numéricas), se interpretan como señales operativas auxiliares.

Esta vista global permite responder preguntas clave de negocio:  
**qué términos activan `High`**, si existen señales coherentes con incidencias críticas y si el modelo se apoya en vocabulario razonable del dominio.  
Además, facilita detectar posibles sesgos (p. ej., términos que activan `High` sin ser realmente críticos).

**Figura E1 — TF-IDF global (top n‑gramas por clase)**  
![TF-IDF Global High](reports/figures/tfidf_global_high.png)
![TF-IDF Global Medium](reports/figures/tfidf_global_medium.png)
![TF-IDF Global Low](reports/figures/tfidf_global_low.png)

Conclusión (global): los términos con mayor peso por clase son coherentes con el dominio, lo que facilita validar que el modelo no se apoya en señales espurias.

**Comparación de vocabulario (top‑10 por clase)**:  
En los top‑10 de `High` no hay solapamiento con `Medium` ni `Low`, lo que indica señales léxicas **bien separadas**.  
Solo aparece un término compartido entre `Medium` y `Low` (p. ej., “sporadic”), consistente con clases menos críticas.  

Ejemplos de términos distintivos:  
`High`: *critical*, *loss*, *exists assistance*;  
`Medium`: *on solutions*, *integrations are*, *features available*;  
`Low`: *securing patient*, *configuration errors*, *growth due*.

### 8.2 Local (SHAP)
SHAP se usa **como complemento** para auditar instancias.  
Valores positivos empujan la **clase predicha**; negativos actúan en sentido contrario.

**Figura E2 — Explicaciones locales (ejemplos TF-IDF + SHAP)**  
![TF-IDF Local Example 1](reports/explainability_tfidf/local_5693_high_medium.png)
![TF-IDF Local Example 2](reports/explainability_tfidf/local_2127_high_low.png)

Estos ejemplos se eligen deliberadamente sobre `High` mal clasificados (`High→Medium` y `High→Low`) para mostrar **errores operativamente críticos**.  
Permiten explicar qué señales empujaron al modelo a fallar y qué términos “faltaron” para activar la clase `High`.

Conclusión (local): las explicaciones por instancia permiten **auditar errores críticos** y justificar por qué el modelo falla en casos concretos.

En términos operativos, esto permite:\n
- revisar tickets conflictivos con el equipo de soporte,\n
- identificar patrones de ambigüedad en el texto,\n
- y justificar por qué un ticket crítico no fue priorizado (o por qué se sobre‑priorizó).

### 8.3 Comparativa entre enfoques
- **TF‑IDF**: alta interpretabilidad y trazabilidad.  
- **Embeddings / SetFit**: mejor semántica, menor transparencia.  
- **LLMs**: explicabilidad baja y alta variabilidad.

Esta comparación justifica por qué, en un entorno con impacto SLA, un modelo **explicable y controlable** puede ser preferible a uno más complejo si la ganancia métrica no compensa la pérdida de trazabilidad.

### 8.4 Límites
Explicación léxica (no causal) y sin relaciones semánticas profundas. Para auditorías avanzadas se podrían usar técnicas locales adicionales.

Limitaciones clave:
- no captura causalidad ni dependencias largas,
- no explica interacciones complejas entre variables,
- y no sustituye la validación humana en casos críticos.

### 8.5 Conclusiones de explicabilidad
La explicabilidad del modelo es **suficiente y defendible** para un entorno de soporte IT porque:
- permite auditar decisiones con trazabilidad clara (coeficientes y SHAP),
- conecta directamente con el riesgo SLA (errores críticos visibilizados),
- y mantiene interpretabilidad sin añadir complejidad innecesaria.

En conjunto, el enfoque elegido equilibra rendimiento y control operativo, reforzando la elección del modelo final seleccionado.

## 9. Métricas de negocio y simulación de impacto

Esta sección conecta el rendimiento del modelo con el **objetivo operativo** del proyecto: priorizar correctamente incidencias con impacto directo en SLA.

### 9.1 Objetivo de negocio y foco en la clase `High`

En la operación real, **no todos los errores cuestan lo mismo**.  
Un falso negativo en `High` implica retrasar tickets críticos, con riesgo de incumplimiento de SLA y costes asociados (operativos o contractuales).  
Por eso, el análisis y la selección del modelo priorizan **recall de `High`** y el control de errores `High → Medium/Low`.

### 9.2 Traducción de errores a impacto

Se utiliza un único nivel de traducción a negocio, con supuestos explícitos:

- **Gap de horas (proxy de SLA)**  
  Se asigna un objetivo de tiempo por clase: `High=4`, `Medium=8`, `Low=12` horas.  
  Se calcula el **`mean_gap`** como `mean(|objetivo_pred - objetivo_real|)`, y se compara frente a un **baseline aleatorio estratificado por clase**.  
  Esta simulación **no modela colas ni tiempos reales**, sino la coherencia de la priorización con los objetivos de respuesta.

### 9.3 Resultados

En la simulación con semilla fija, el modelo reduce el `mean_gap` frente al baseline aleatorio:

- Modelo: **1.08h** (mean_gap) y **77.12%** de acierto.  
- Aleatorio: **3.21h** (mean_gap) y **35.72%** de acierto.

Esto supone una **reducción media de 2.13h** en el gap respecto a un baseline no entrenado.

Cómo leer estos valores:
- Con objetivos `4/8/12h`, los gaps posibles son **0h, 4h o 8h**.  
- **0h** = acierto exacto de clase.  
- **4h** = error adyacente (`High ↔ Medium` o `Medium ↔ Low`).  
- **8h** = error extremo (`High ↔ Low`), el más costoso en operación.

Para visualizar la severidad del error, se muestra la distribución del gap:
- **Modelo**: 0h **77.12%**, 4h **18.72%**, 8h **4.17%**.  
- **Aleatorio**: 0h **35.72%**, 4h **48.26%**, 8h **16.02%**.

La mejora no es solo en media: el modelo **concentra la mayoría de casos en 0h** y reduce drásticamente los errores extremos (8h), que son los más dañinos para el SLA.

### 9.4 Ejecución y trazabilidad

El cálculo se obtiene con el script `src.business_metrics`, que guarda un resumen en:
- `reports/business_metrics_simple.json`

```
BUSINESS_SEED=42 python -m src.business_metrics
```

Parámetros útiles:
- `BUSINESS_AGENTS`: número de agentes en la simulación.
- `BUSINESS_HORIZON_H`: horizonte temporal en horas.

---

## 10. Despliegue conceptual en un entorno real

**Escenario**: integración en un Service Desk para asistir el triaje inicial de tickets.

**Pipeline**: ingesta → limpieza → TF‑IDF + metadatos → predicción → priorización.

**Operación**: modo near‑real‑time (latencia baja) con trazabilidad de decisiones.

**Monitorización**:
- distribución de predicciones,
- recall High proxy (auditorías),
- drift léxico.

**Reentrenamiento**: ante caída sostenida de macro‑F1/recall High o cambios de dominio.

**Arquitectura conceptual (sin implementación técnica)**:
1. **Entrada**: ticket con `Body`, `Department`, `Tags`.  
2. **Preprocesado**: limpieza básica + derivación de `n_tags`, `len_words`.  
3. **Vectorización**: TF‑IDF + OHE + numéricas con el vocabulario entrenado.  
4. **Predicción**: Linear SVM devuelve prioridad.  
5. **Salida**: prioridad guardada y usada para ordenación/triage.

**Control operativo**:
- alertas si `Pred High (%)` se dispara o cae de forma anómala,
- revisión humana de una muestra de `High` para estimar recall real,
- auditoría de errores críticos (`High→Low`).

**Actualización y versionado**:
- congelar vocabulario y modelo por versión,
- comparar métricas antes/después de cada release,
- mantener rollback si aparece degradación.

## 11. Conclusiones y trabajo futuro

### 11.1 Aprendizajes clave

- El **modelo final seleccionado (TF-IDF + Linear SVM)** ofrece el mejor equilibrio entre rendimiento, interpretabilidad y control operativo.  
- En problemas con SLA, **recall High** es la métrica determinante y debe guiar la selección.  
- La explicabilidad no es un añadido: es un requisito para adopción y confianza en operación.  

### 11.2 Limitaciones reales

- No se dispone de validación temporal para evaluar drift real.  
- La simulación de negocio utiliza costes proxy, no costes reales del soporte.  
- No se incorpora feedback humano en producción (loop de mejora continua).  
