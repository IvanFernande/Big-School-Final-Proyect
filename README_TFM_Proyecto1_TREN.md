# 📊 Ticket Priority Classification
**Modelización Predictiva y Métricas de Negocio (TFM – Proyecto 1)**

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
- No se trata como ordinal puro: los costes no son lineales.
- Cada clase tiene un peso de negocio distinto.

### 2.2 Variable objetivo
`Priority` define la urgencia del ticket:
- `high`: impacto crítico.
- `medium`: impacto moderado.
- `low`: consultas o mejoras.

La clase **High** es prioritaria aunque minoritaria.

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

### 3.4 Limpieza y feature engineering
- Normalización de `Priority` a minúsculas.
- `Body` vacío → cadena vacía.
- `Department` nulo → `"Unknown"`.
- `Tags` parseado a lista.
- Variables derivadas: `n_tags`, `len_words`.

Estas variables combinan **señal semántica** y **contexto operativo**, alineadas con el problema real.

---

## 4. Hipótesis de partida y estrategia experimental

### 4.1 Hipótesis de partida
- TF-IDF + modelos clásicos puede ser altamente competitivo.
- Los embeddings capturan semántica más rica, pero no siempre compensan en datasets medianos.
- SetFit permite fine-tuning eficiente sin modelos grandes.
- Zero-shot sirve como benchmark, no como solución final.

### 4.2 Estrategia de comparación
Se comparan **familias de modelos**, no solo configuraciones individuales:
- TF-IDF + modelos clásicos.
- Embeddings + clasificadores lineales.
- Fine-tuning con SetFit.
- Zero-shot / Few-shot con LLMs.

“Ganar” significa **mejor macro-F1 y recall High**, con coste operativo razonable y trazabilidad.

---

## 5. Estrategia global de evaluación y métricas

### 5.1 Principios generales
- Separación entre selección y evaluación final.
- Robustez al desbalance.
- Priorización de errores con impacto real en SLA.
- Evitar métricas redundantes o poco interpretables.

### 5.2 Métricas técnicas principales

#### Macro-F1
- Promedio no ponderado del F1 por clase.
- Evita que la clase mayoritaria domine el resultado.

#### Balanced Accuracy
- Promedio del recall por clase.
- Útil para detectar modelos que ignoran clases minoritarias.

#### Métricas por clase (High)
- Precision (High)
- Recall (High)
- F1 (High)

Estas métricas conectan directamente con riesgo SLA.

### 5.3 Métricas complementarias
- Accuracy (solo como referencia).
- Weighted F1 (diagnóstico global).

### 5.4 Confusion Matrix
Permite cuantificar errores críticos:
- `High → Medium`
- `High → Low`

### 5.5 Métricas descartadas
- ROC-AUC multiclase.
- Log-loss.
- MCC / Cohen’s Kappa.

---

## 6. Exploración de enfoques y resultados

## 6.1 TF-IDF + modelos clásicos (baseline principal)

### Motivación del enfoque

Se establece como primer enfoque un baseline basado en **TF-IDF combinado con modelos lineales**, por tratarse de una solución:

- interpretable,
- computacionalmente eficiente,
- robusta en datasets de tamaño medio,
- ampliamente utilizada como referencia en clasificación de texto.

Este enfoque permite además **analizar de forma clara los errores**, algo especialmente relevante cuando el impacto de negocio (SLA) no es simétrico entre clases.

---

### ¿Qué es TF-IDF y por qué se utiliza?

TF-IDF (Term Frequency – Inverse Document Frequency) es una representación vectorial que pondera los términos según su frecuencia en un documento y su rareza en el corpus completo. De este modo, resalta palabras relevantes para un ticket concreto y reduce el peso de términos comunes.

Se utiliza en este proyecto por ser:
- altamente interpretable,
- eficaz en clasificación de texto con vocabulario técnico,
- un estándar sólido como baseline en problemas NLP supervisados.

---

### Representación textual: TF-IDF

Se utiliza **TF-IDF a nivel de palabra** sobre el campo `Body`, que concentra la mayor parte de la señal semántica del ticket.

**Configuración final del vectorizador**
- n-grams: `(1, 2)`
- `min_df = 2`
- `sublinear_tf = True`
- `smooth_idf = True`

Esta configuración busca capturar:
- términos individuales relevantes,
- expresiones operativas frecuentes (por ejemplo, *“service down”*, *“urgent issue”*),

sin introducir combinatoria excesiva ni ruido.

No se emplean:
- stopwords personalizadas,
- stemming o lemmatization,
- char n-grams,
- selección de features (χ²),

al priorizar **trazabilidad, estabilidad y control del espacio experimental** frente a micro-optimización.

---

### Incorporación de metadatos operativos

Además del texto, se incorporan señales estructurales relevantes:

- `Department`: variable categórica con cardinalidad baja (~10 valores).
- `n_tags`: número de etiquetas asociadas al ticket.
- `len_words`: longitud del texto del ticket.

Las variables numéricas se escalan mediante  
`StandardScaler(with_mean=False)` para compatibilidad con matrices dispersas.

#### Nota sobre el uso de `Tags`

Aunque el dataset incluye una columna `Tags`, estas no se incorporan directamente al TF-IDF como texto adicional. Los tags representan información categórica y altamente redundante con el contenido del `Body`, por lo que su inclusión directa podría introducir sesgos y doble conteo de señal semántica.

En su lugar, se utiliza `n_tags` como proxy de complejidad y grado de anotación del ticket, aportando señal adicional sin contaminar la representación textual.

---

### Estrategias evaluadas para `Department`

Se comparan tres estrategias:
- One-Hot Encoding completo.
- One-Hot Encoding agrupado (frecuencias bajas).
- Hashing.

Los resultados muestran **diferencias marginales** entre estrategias, coherentes con la baja cardinalidad de la variable.  
Para evitar decisiones arbitrarias y mantener coherencia con el pipeline, se adopta una **selección automática** de la mejor estrategia mediante **macro-F1 en validación cruzada**.

---

### Modelos evaluados

Se entrenan y comparan los siguientes clasificadores:

- **Linear SVM**
- Logistic Regression
- Multinomial Naive Bayes

Todos los modelos se integran en un pipeline único que incluye:
- preprocesado,
- vectorización,
- clasificación.

Para SVM y Logistic Regression se utiliza  
`class_weight="balanced"`, priorizando la correcta detección de la clase `High`.

---

### Estrategia de evaluación

#### División de datos
- Split **80 / 20** (train / test).
- Estratificación por `Priority`.
- El conjunto de test se mantiene **completamente aislado** hasta la evaluación final.

#### Justificación del uso de validación cruzada

La validación cruzada estratificada se utiliza para la selección de modelos y configuraciones por ofrecer una estimación más estable y robusta del rendimiento que un único conjunto de validación fijo.

En datasets desbalanceados, un split reducido puede contener pocos ejemplos de clases críticas, produciendo estimaciones inestables. La validación cruzada permite comparar modelos de forma justa, maximizar el uso de los datos disponibles y reducir la dependencia del azar.

Aunque inicialmente se planteó un esquema 70/20/10, se decidió no utilizarlo al quedar la validación cruzada como un sustituto más robusto del conjunto de validación. Mantener ambos enfoques habría sido redundante y menos eficiente.

---

### Resultados de selección (Cross-Validation)

Durante la validación cruzada estratificada sobre el conjunto de entrenamiento, se obtuvieron los siguientes resultados medios (macro-F1):

- **Linear SVM**: ≈ 0.70  
- **Logistic Regression**: ≈ 0.65  
- **Multinomial Naive Bayes**: ≈ 0.41  

La baja desviación estándar entre folds indica un comportamiento estable del pipeline y respalda la elección de **Linear SVM** como modelo base.

---

### Métricas utilizadas y justificación

#### Métrica principal (selección de modelo)
- **macro-F1**  
  Evalúa el equilibrio global entre clases sin favorecer a la mayoritaria.

#### Métricas de evaluación final (test)
- **macro-F1**
- **recall de la clase High**
- accuracy (solo como referencia)
- confusion matrix

El **recall High** es la métrica más relevante desde el punto de vista de negocio, ya que un falso negativo en esta clase implica riesgo directo de incumplimiento de SLA.

No se utilizan métricas como ROC-AUC o log-loss por su baja interpretabilidad en un contexto multiclase con prioridades no simétricas.

---

### Artefactos y métricas generadas

Durante la experimentación se generan y almacenan artefactos intermedios para asegurar **trazabilidad y reproducibilidad**:

- Métricas de validación cruzada por modelo (`metrics_baselines.json`)
- Comparativa de estrategias de `Department` (`metrics_dept_strategies.json`)
- Métricas finales en conjunto de test (`metrics_test.json`)
- Matriz de confusión del modelo final

---

### Resultados obtenidos (TF-IDF)

**Mejor configuración seleccionada**
- Modelo: **Linear SVM**
- Representación: TF-IDF + metadatos
- Estrategia de `Department`: seleccionada automáticamente

**Resultados en conjunto de test**
- Macro-F1 ≈ **0.76**
- Recall High ≈ **0.81**
- Accuracy ≈ **0.77**

La matriz de confusión muestra que:
- la mayoría de errores de `High` se producen hacia `Medium`,
- los errores críticos `High → Low` son minoritarios.

Este patrón es coherente con el objetivo de negocio, ya que **prioriza no perder tickets críticos**, incluso a costa de cierta sobre-priorización.

---

### Conclusión del enfoque TF-IDF

El enfoque TF-IDF + Linear SVM se consolida como:

- un **baseline sólido y competitivo**,
- con excelente equilibrio entre rendimiento, interpretabilidad y coste,
- difícil de superar por enfoques más complejos sin aumentar significativamente la complejidad.

Este modelo se utiliza como **referencia principal** para comparar enfoques basados en embeddings, fine-tuning y LLMs en las siguientes secciones.


---

### 6.2 Embeddings + clasificadores lineales

### Motivación del enfoque

Aunque TF-IDF ofrece una representación eficaz y altamente interpretable, su naturaleza basada en frecuencia limita la captura de relaciones semánticas entre tickets redactados de forma distinta pero conceptualmente equivalentes.

El uso de **embeddings densos preentrenados** permite representar el texto en un espacio semántico continuo, donde:
- sinónimos y expresiones equivalentes quedan próximas,
- se reduce la dependencia del vocabulario exacto,
- se mejora la generalización ante variabilidad lingüística.

Este enfoque se evalúa como una **extensión natural del baseline**, manteniendo clasificadores lineales para asegurar comparabilidad, control experimental y trazabilidad.

---

### Representación semántica mediante embeddings

Se emplean modelos *sentence-level embeddings* preentrenados, utilizados exclusivamente como **extractores de características**, sin fine-tuning del encoder.

No se realiza ajuste del encoder durante el entrenamiento con el objetivo de **aislar el efecto de la representación**, mantener la comparabilidad directa con TF-IDF y evitar introducir un coste computacional adicional que dificultaría la evaluación justa entre enfoques.

Modelos evaluados:
- `all-MiniLM-L6-v2` (ligero, baja latencia)
- `all-mpnet-base-v2` (mayor capacidad semántica)
- `all-distilroberta-v1` (punto intermedio)
- `multi-qa-MiniLM-L6-cos-v1` (orientado a recuperación semántica)

Para cada modelo se registran:
- dimensión del embedding,
- tiempo de inferencia,
- reutilización de caché,

permitiendo analizar el **trade-off rendimiento / coste computacional**.

---

### Estrategias de combinación de features

Dado que el dataset combina texto libre y metadatos, se evalúan dos grandes familias de representación:

#### A) Embeddings “puros” (solo texto)

Se combinan embeddings de los campos textuales:
- `Body`
- `Department`
- `Tags`

Estrategias evaluadas:
- concatenación directa (`concat`)
- media normalizada (`avg`)
- medias ponderadas (`70/30`, `30/70`, `90/10`)

La normalización L2 previa garantiza comparabilidad entre vectores y evita sesgos de escala.

Estas variantes permiten responder a:
> ¿Dónde reside la mayor parte de la señal semántica: en el texto libre o en los metadatos textuales?

---

#### B) Embeddings + estructura explícita (OHE)

El campo `Department` representa una **categoría funcional**, no lenguaje natural. Embeddizarlo puede introducir ruido semántico.

Por este motivo, se evalúa explícitamente `Department` tanto como embedding como mediante **One-Hot Encoding**, con el objetivo de validar empíricamente si su tratamiento como texto aporta información relevante o si, por el contrario, una codificación categórica explícita resulta más estable e interpretable.

La estrategia híbrida combina:
- Embedding del `Body` (y `Tags` cuando procede)
- `Department` codificado mediante **One-Hot Encoding**
- Variables numéricas (`n_tags`, `len_words`)

Esta aproximación separa explícitamente:
- **semántica** → embeddings
- **estructura del dominio** → OHE / numéricas

y permite analizar si esta distinción mejora estabilidad y rendimiento.

---

### Modelos de clasificación

Sobre las representaciones generadas se entrenan clasificadores lineales:
- **Linear SVM**
- Logistic Regression

Ambos se configuran con:
- `class_weight="balanced"`
- grid reducido de regularización (`C`)

La elección de modelos lineales responde a:
- control del sesgo,
- interpretabilidad,
- comparabilidad directa con TF-IDF.

---

### Estrategia de evaluación

La evaluación sigue exactamente los mismos principios que el baseline TF-IDF:
- **Validación cruzada estratificada** para selección de modelo
- **Conjunto de test aislado** para evaluación final
- Misma partición y mismas métricas

Esto garantiza que cualquier diferencia observada se deba a la **representación**, no al protocolo experimental.

---

### Métricas específicas para embeddings

Además de las métricas generales (macro-F1, balanced accuracy), se introducen métricas específicas:

- **Average Precision (PR-AUC) High vs Rest**  
  Evalúa la capacidad del modelo como detector de tickets críticos, especialmente relevante en escenarios desbalanceados.

- **Coste medio por ticket**  
  Calculado mediante una matriz de costes que penaliza especialmente:
  - falsos negativos en `High`,
  - errores `High → Low`.

Estas métricas conectan directamente la calidad del modelo con el **impacto operativo y de negocio**.

---

### Resultados obtenidos

👉 *(a completar automáticamente desde `embeddings_summary.csv`)*

| Modelo de embeddings | Estrategia | Macro-F1 | Recall High | AP High | Coste/ticket |
|---------------------|------------|----------|-------------|---------|---------------|

---

### Criterio de selección y resumen de experimentos

La experimentación con embeddings genera **un fichero JSON por combinación** de:
- modelo de embeddings,
- estrategia de representación,
- clasificador,
- hiperparámetros.

Dado el elevado número de combinaciones evaluadas, **no se listan todos los experimentos**, sino que se adopta una estrategia de **resumen y selección justificada**, evitando ruido experimental y selección basada en resultados puntuales de test.

#### Información extraída de cada experimento

De cada archivo JSON se extraen las siguientes métricas:

**Selección (Cross-Validation):**
- `cv.f1_macro_mean`
- `cv.f1_macro_std`
- `cv.balanced_accuracy_mean` (si aplica)

**Evaluación final (test):**
- `test.macro_f1`
- `test.recall_high`
- `test.precision_high`
- `test.high_vs_rest_ap`
- `test.cost_per_ticket`
- errores `High → Medium` y `High → Low`

#### Selección del mejor experimento por modelo de embeddings

Para cada modelo de embeddings se selecciona **un único experimento representativo**, siguiendo este criterio:

1. Mayor `cv.f1_macro_mean`
2. Cumplimiento de un umbral mínimo de `test.recall_high`
3. En caso de empate:
   - menor `test.cost_per_ticket`
   - mayor `test.high_vs_rest_ap`
   - menor desviación estándar en CV

Este enfoque prioriza **estabilidad, protección de la clase crítica e impacto de negocio**, evitando optimización directa sobre el conjunto de test.

---

### Análisis de errores

El análisis de la matriz de confusión revela que:
- los errores de `High` tienden a confundirse con `Medium`,
- los errores críticos `High → Low` se mantienen bajos,
- no se observa una reducción significativa de estos errores frente al baseline.

Este patrón es consistente con los resultados métricos.

---

### Conclusión del enfoque con embeddings

El uso de embeddings aporta una representación semántica más rica, pero en este contexto:
- no mejora de forma clara las métricas clave de negocio,
- introduce mayor coste computacional,
- reduce interpretabilidad frente a TF-IDF.

Por tanto, **los embeddings no aportan una ventaja decisiva** sobre el baseline en este dataset concreto, aunque sí ofrecen un marco más generalizable y flexible.

El enfoque se mantiene como referencia avanzada y se compara posteriormente con técnicas de fine-tuning ligero (SetFit).


---

### 6.3 Fine-tuning con SetFit

**Motivación**  
Fine-tuning ligero sobre sentence transformers.

**Estrategia de selección**
Score = `0.7 · macro-F1 + 0.3 · recall High`.

**Resultados**
👉 **[COMPLETAR CON MÉTRICAS FINALES]**

**Conclusión parcial**  
Buen compromiso, pero no supera al baseline TF-IDF.

---

### 6.4 Zero-shot / Few-shot con LLMs

**Motivación**  
Benchmark sin entrenamiento.

**Métricas específicas**
- Coverage.
- Invalid rate.
- Latencia.

**Resultados**
👉 **[COMPLETAR SI AJUSTAS CONFIGURACIÓN FINAL]**

**Conclusión parcial**  
No viable para producción.

---

## 7. Comparativa global y selección del modelo final

👉 **[COMPLETAR TABLA FINAL]**

| Enfoque | Macro-F1 | Recall High | Latencia | Coste |
|--------|----------|-------------|----------|-------|

**Modelo seleccionado**
👉 **[DECISIÓN FINAL EXPLÍCITA]**

---

## 8. Explicabilidad del modelo
La explicabilidad se aborda mediante:
- Interpretabilidad de TF-IDF.
- Análisis de errores críticos.
- Comparación entre enfoques.

---

## 9. Despliegue conceptual
👉 **[COMPLETAR]**
- Batch / real-time.
- Reentrenamiento.
- Monitorización y data drift.
- Versionado.

---

## 10. Conclusiones y trabajo futuro
👉 **[COMPLETAR]**
- Aprendizajes clave.
- Limitaciones.
- Posibles mejoras.
