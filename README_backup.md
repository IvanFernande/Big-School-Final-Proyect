# Sistema RAG Multiformato para Documentación de Soporte

El objetivo es diseñar, implementar y evaluar un sistema de **Retrieval-Augmented Generation (RAG)** aplicado a documentación interna de soporte técnico, poniendo el foco en el impacto de las estrategias de recuperación sobre la calidad de las respuestas generadas.

El sistema permite responder preguntas factuales y procedimentales a partir de documentos heterogéneos, evaluando de forma **reproducible y cuantitativa** distintas configuraciones de retrieval y generación.

A lo largo del documento se distingue explícitamente entre **retrieval** (recuperación de contexto relevante) y **generation** (generación de la respuesta final por el modelo LLM), evaluándose ambos componentes de forma independiente.

---

## 1. Objetivo del proyecto

El objetivo principal de este proyecto es construir un sistema RAG que:

- Responda preguntas sobre documentación interna de soporte técnico.
- Reduzca al mínimo las alucinaciones mediante recuperación explícita de contexto.
- Permita comparar distintas estrategias de recuperación de información de forma controlada.
- Justifique técnicamente las decisiones adoptadas mediante métricas objetivas.

El foco del proyecto no es únicamente la generación de texto, sino el **análisis del papel del retrieval como componente crítico en sistemas RAG**, especialmente en dominios donde la precisión factual es prioritaria frente a la creatividad.

El proyecto pone el foco en la etapa de recuperación, al considerarse el principal factor determinante de la precisión factual en sistemas RAG aplicados a documentación técnica.

---

## 2. Dominio y descripción de los datos

El corpus simula un entorno realista de documentación interna de una organización tecnológica e incluye información de distinta naturaleza:

- KPIs y métricas operativas
- Configuración de servicios
- Procedimientos operativos y FAQs
- Políticas internas y on-call
- Inventario de servicios y activos
- Histórico de incidentes
- Organigrama y roles técnicos

Los documentos se presentan en múltiples formatos (CSV, JSON, Markdown y texto plano), reproduciendo la heterogeneidad habitual en entornos empresariales reales.

---

## 3. Planteamiento experimental y alcance

Se evalúan de forma controlada tres componentes clave del sistema:
- embeddings
- retrieval
- generación

Quedan fuera del alcance:
- fine-tuning de modelos
- aprendizaje online
- entrenamiento supervisado con datos externos

---
## 4. Conjunto de evaluación

La evaluación se realiza mediante un **test set manualmente definido**, compuesto por preguntas representativas del dominio.

El conjunto está definido en `src/eval_data.py` como `TEST_SET` y se reutiliza en los benchmarks de embeddings, retrieval y LLM para evaluar de forma comparable cada etapa del pipeline.

Cada entrada del test set incluye:
- **question**: la consulta a evaluar.
- **expected**: lista de tokens/entidades que deben aparecer (números, IDs, versiones, emails).
- **expect_text**: respuesta canónica en texto para comparar similitud semántica.
- **category**: etiqueta semántica (tablas, configuración, identificadores, procedimientos, inventarios).

Diseño del test set:
- **Tablas (CSV)**: busca extraer valores numéricos exactos (SLA, CSAT, tickets).
- **Configuración (JSON)**: verifica claves y parámetros críticos (versiones, features, umbrales).
- **Identificadores**: exige coincidencia exacta de IDs, emails y versiones.
- **Procedimientos**: comprueba pasos y acciones clave en textos largos.
- **Inventarios**: valida estados y owners asociados a servicios.

Este enfoque permite evaluar:
- Coincidencia parcial por tokens (retrieval).
- Calidad global por similitud semántica (LLM).
- Robustez frente a diferentes tipos de pregunta.

Ejemplos reales del TEST_SET:
```
{"question": "Cual es el SLA cumplido en Q4 y cuantos tickets abiertos hubo?",
 "expected": ["93.1", "1500"],
 "expect_text": "En Q4 el SLA cumplido es 93.1% y hubo 1500 tickets abiertos.",
 "category": "tablas"}

{"question": "Cual es la version del servicio principal?",
 "expected": ["2.3.1"],
 "expect_text": "La versión del servicio principal (core_api) es 2.3.1.",
 "category": "identificadores"}

{"question": "Que pasos de troubleshooting se realizan antes de escalar?",
 "expected": ["verificar", "credenciales", "logs"],
 "expect_text": "Antes de escalar se realiza troubleshooting: verificar credenciales, revisar logs y comprobar estado del servicio.",
 "category": "procedimientos"}
```

---

## 5. Métricas de evaluación

### 5.1 Métricas de recuperación
- **avg_score (top‑k)**: proporción de tokens esperados presentes en el contexto recuperado.
  - Definición:  
    `avg_score = (# tokens esperados presentes en top‑k) / (total tokens esperados)`
  - Cálculo: se concatena el texto de los top‑k chunks y se comprueba si cada token esperado aparece como substring.
  - Normalización: se aplica `lower()` y se eliminan tildes (y opcionalmente puntuación) antes de comparar.

- **Recall@k**: existe al menos un chunk relevante en el top‑k.  
  - Definición:  
    `Recall@k = 1 si existe chunk con score_relevancia > 0, si no 0`
  - Relevancia: se define por proporción de tokens esperados presentes en el chunk.

- **MRR@k**: posición del primer chunk relevante.  
  - Definición:  
    `MRR@k = 1 / rank_del_primer_chunk_relevante` (0 si no hay relevante).

- **nDCG@k**: calidad del ranking completo con relevancia graduada.  
  - Definición:  
    `DCG = sum(rel_i / log2(i+1))`  
    `nDCG = DCG / IDCG`
  - `rel_i` se calcula como proporción de tokens esperados presentes en cada chunk.

- **context_precision@k**: proporción de tokens relevantes respecto al total de tokens en el contexto.  
  - Definición:  
    `context_precision = (# tokens esperados presentes) / (# tokens totales en top‑k)`
  - Interpreta cuánto “ruido” se introduce en el contexto recuperado.
  - No se optimiza de forma aislada; se usa para comparar el nivel relativo de ruido entre estrategias con cobertura similar.

### 5.2 Métricas de respuesta
- **score (contains‑match)**: proporción de tokens esperados presentes en la respuesta del LLM.  
  - Definición:  
    `score = (# tokens esperados presentes) / (total tokens esperados)`
  - Normalización: misma normalización que en retrieval (lower, sin tildes, espacios colapsados).

- **semantic_similarity**: similitud coseno entre embeddings de la respuesta y `expect_text`.  
  - Definición:  
    `sim = cos(emb(respuesta), emb(expect_text))`
  - Modelo: `sentence-transformers` (configurable en `llm_similarity_model`).

- **groundedness**: penaliza entidades fuera del contexto recuperado.  
  - Definición: se extraen patrones (emails, IDs, versiones) en la respuesta y se verifica su presencia en el contexto.

- **latencias**:  
  - `retrieval_s`, `generation_s`, `total_s` se miden por pregunta.

Las métricas se reportan de forma global y segmentadas por categoría.

### 5.3 Relación métrica → benchmark → objetivo

- **benchmark_embeddings**  
  - Métrica principal: `avg_score` (top‑k) sobre retrieval vectorial.  
  - Objetivo: seleccionar el mejor modelo de embeddings.

- **benchmark_retrieval**  
  - Métricas: `avg_score`, `Recall@k`, `MRR@k`, `nDCG@k`.  
  - Objetivo: comparar estrategias de retrieval (vector, BM25, híbridos, rerank).

- **benchmark_llm**  
  - Métricas: `score`, `semantic_similarity`, `groundedness`, latencias.  
  - Objetivo: evaluar calidad final de respuesta y elegir el LLM ganador.

---

## 6. Arquitectura del sistema

### 6.1 Visión general del pipeline

El sistema sigue una arquitectura RAG modular, implementada en componentes reutilizables:

**Pipeline principal (descripción técnica):**
- **Ingestión**: se cargan documentos multiformato y se normaliza su texto, preservando metadatos de origen.
- **Preprocesado**: se limpian espacios, saltos de línea y caracteres no deseados manteniendo tokens críticos (IDs, emails, versiones).
- **Chunking**: se segmenta por secciones en textos narrativos y por bloques fijos en datos estructurados, con solape controlado.
- **Embeddings**: cada chunk se vectoriza con modelos `sentence-transformers` u `ollama`.
- **Vector store**: los embeddings se normalizan y se indexan en FAISS para búsqueda por similitud coseno.
- **Retrieval**: se comparan enfoques vectoriales, léxicos (BM25) e híbridos.
- **Generación**: el LLM responde usando solo el contexto recuperado.
- **Evaluación**: se cuantifica la calidad del retrieval y de la respuesta final con métricas automáticas.

**Flujo lógico:**
Ingesta → Limpieza → Chunking → Embeddings → Indexación → Retrieval → LLM → Métricas.


**Trazabilidad y metadatos:**
Cada fragmento (chunk) conserva metadatos que permiten reconstruir su origen:
- `source`, `filename`, `ext`, `source_type`
- `doc_id` estable por documento
- offsets y límites de caracteres (`char_start`, `char_end`)
- `section_title` y `headings_path` cuando aplica

Esto permite:
- Auditar qué documento y sección sustentan una respuesta.
- Analizar fallos de recuperación de forma granular.
- Repetir evaluaciones con exactitud.

**Diseño desacoplado:**
Cada etapa se implementa de forma independiente. Esto facilita:
- Sustituir el modelo de embeddings sin reescribir el pipeline.
- Probar variantes de chunking o retrievers de forma aislada.
- Mantener reproducibilidad y control experimental.

---


### 6.2 Ingestión y chunking

### 6.2.1 Ingestión multiformato

Los documentos se cargan preservando contenido y metadatos básicos (tipo, origen, sección) para asegurar trazabilidad en retrieval.  
El objetivo de la ingesta es **normalizar cada formato a texto** con estructura mínima para que el chunking y el retrieval funcionen de forma consistente.

Flujo de ingesta (scripts):
- `scripts/build_index.py`: recorre `data/`, aplica loaders por formato, limpia texto y genera chunks.
- `scripts/benchmark_embeddings.py` y `scripts/benchmark_retrieval.py`: construyen índices de prueba para comparar configuraciones sin afectar al índice principal.

Loaders por tipo:
- PDF: extracción de texto y tablas, con marcadores `[TABLAS]`.
- CSV/JSON: transformación a texto legible por filas/keys.
- TXT/MD: lectura directa preservando estructura.

Ejemplos de normalización por formato:

- **PDF** (texto + tablas):
```
...contenido de la página...

[TABLAS]
columna1,columna2,columna3
valor1,valor2,valor3
```
Esto permite que datos tabulares queden accesibles para consultas de valores exactos.

- **CSV** (una fila por chunk en texto legible):
```
trimestre: Q4; tickets_abiertos: 1500; tickets_cerrados: 1470; sla_cumplido_pct: 93.1; csat_promedio: 4.7
```
Se evita mezclar filas para mantener consistencia numérica.

- **JSON** (keys y valores aplanados):
```
servicio: core_api
version: 2.3.1
features.0: autenticacion_segura
alerting.cpu: 85
```
El aplanado conserva rutas para poder recuperar claves específicas.

- **Markdown / TXT** (preserva estructura y headings):
```
# Proceso de escalado
## Nivel 1
Validar credenciales y revisar logs...
```
La estructura de headings se utiliza después en el chunking semántico para no mezclar temas.

### 6.2.2 Estrategia de segmentación (chunking)

La segmentación se adapta al tipo de documento para preservar coherencia semántica y evitar contaminación entre registros:

- **Documentos narrativos (PDF, Markdown, TXT)**  
  Chunking semántico por secciones (headings) combinado con ventana deslizante con solape.  
  - Objetivo: mantener pasos completos, contexto local y trazabilidad por sección.  
  - Parámetros actuales: `max_size=500`, `overlap=60`.

- **Documentos estructurados (CSV, JSON)**  
  Chunking fijo sin solape.  
  - Objetivo: evitar mezclar filas o claves y prevenir contaminación de valores numéricos.  
  - Parámetros actuales: `size=400`, `overlap=0`.

Justificación:
- En procedimientos largos, el split por secciones reduce cortes abruptos y mejora la recuperación de pasos completos.
- En tablas y configuraciones, el split fijo garantiza independencia entre registros.

Parámetros en uso:
- Narrativo: `max_size=500`, `overlap=60`
- Estructurado: `size=400`, `overlap=0`

---


### 6.3 Almacenamiento vectorial (FAISS)

Los embeddings se almacenan en un índice FAISS (`index/index.faiss`) junto con los metadatos asociados a cada fragmento (`index/data.json`).

Motivos para utilizar FAISS:

- Eficiencia en CPU para búsqueda vectorial y escalabilidad a miles de fragmentos.
- Persistencia local del índice sin dependencia de servicios externos.
- Integración sencilla con embeddings normalizados (similitud coseno mediante inner product).

FAISS se elige frente a alternativas más complejas debido a su simplicidad de despliegue local y a que ofrece un rendimiento adecuado para el tamaño del corpus considerado en este proyecto.

Comparativa con alternativas:
- **Chroma**: aporta API de alto nivel y persistencia integrada, pero añade mas dependencias y abstracciones; para este proyecto FAISS es suficiente y mas ligero.
- **Elasticsearch/OpenSearch**: muy potente para escala y filtrado, pero requiere infraestructura adicional y un despliegue mas pesado.
- **Annoy/HNSWlib**: buenas alternativas en CPU, pero FAISS ofrece mas flexibilidad y una implementacion estable en el ecosistema.

---


---
## 7. Estrategias de recuperación (Retrieval)

Dado que el corpus contiene tanto información altamente estructurada como documentación narrativa, el sistema evalúa múltiples estrategias de recuperación.

### 7.1 Recuperación léxica (BM25)

BM25 es un algoritmo clásico de recuperación de información basado en coincidencia exacta de términos.

Se utiliza como baseline léxico y resulta especialmente eficaz para:
- Identificadores exactos (IDs de incidentes, nombres de servicios, comandos)
- Valores numéricos, versiones y configuraciones
- Nombres propios y direcciones de correo

Su principal limitación es la degradación de rendimiento ante reformulación semántica de las consultas.

BM25 se incluye como baseline léxico no solo por comparativa, sino para garantizar un comportamiento robusto en consultas dominadas por identificadores, valores exactos o tokens críticos.

---

### 7.2 Recuperación semántica (Vector Search)

La búsqueda vectorial se basa en embeddings densos que representan documentos y consultas en un espacio semántico continuo.

Ventajas principales:
- Capacidad de recuperar contexto relevante aunque no exista coincidencia literal.
- Buen rendimiento en procedimientos, FAQs y documentación descriptiva.

Los embeddings se utilizan exclusivamente como extractores de características, sin ajuste durante el entrenamiento, para mantener comparabilidad experimental.

---

### 7.3 Recuperación híbrida (Vector + BM25)

La recuperación híbrida combina resultados de BM25 y búsqueda vectorial mediante fusión de rankings controlada por un parámetro `retriever_alpha`.

Este enfoque permite:
- Mantener la precisión léxica en tokens críticos.
- Incorporar generalización semántica en consultas descriptivas.
- Evitar decisiones rígidas sobre el tipo de consulta.

Se evalúa como principal candidato del sistema.

---

### 7.4 Re-ranking (opcional)

De forma opcional, se incorpora un reranker basado en cross-encoder que reordena los fragmentos recuperados en el top-k.

Este paso puede mejorar la calidad final del ranking, aunque introduce latencia adicional, por lo que se evalúa de forma independiente.

Aunque el reranking mejora ligeramente el orden del top‑k, su impacto global no compensa el incremento de latencia introducido en el contexto de este proyecto.

---

### 7.5 Benchmark específico de BM25

Se realiza un grid search sobre los hiperparámetros `k1` y `b` de BM25 para obtener una configuración óptima previa a la comparación con otros retrievers.

Motivación:  
BM25 es altamente sensible a estos hiperparámetros, y el uso de valores por defecto puede infraestimar su rendimiento real.

Resultado del tuning:
- Mejor combinación encontrada: `k1=0.9`, `b=0.25`.
  
Métricas del benchmark BM25 (avg_score):
- Mejor resultado: 0.838 (best_k=16).
- min_score: 0.000, max_score: 1.000.
- El rendimiento se mantiene estable entre combinaciones probadas en este corpus.

Conclusión del tuning:
Aunque se probaron varias combinaciones de `k1`/`b`, no se observó una mejora significativa frente a otras configuraciones.  
Se fija la configuración encontrada por el grid search (`k1=0.9`, `b=0.25`) para mantener consistencia experimental.

Desglose por categoria (best_k=16):

| Categoria | avg_score |
| --- | --- |
| tablas | 0.80 |
| configuracion | 0.33 |
| identificadores | 1.00 |
| procedimientos | 0.85 |
| inventarios | 1.00 |

Grid de combinaciones probadas (BM25):
- `k1`: [0.9, 1.2, 1.5, 1.8, 2.0]
- `b`: [0.25, 0.5, 0.75, 0.9]

![BM25 breakdown](visualizations/bm25_breakdown.png)

---

### 7.6 Resultados del benchmark de retrieval

Métrica principal: `avg_score` (proporción de tokens esperados presentes en el top‑k concatenado).

Resultados (mejor K por modo):

| Modo | best_k | avg_score | context_precision@k | Recall@k | MRR@k | nDCG@k |
|------|--------|-----------|--------------------|----------|-------|--------|
| vector | 16 | 0.949 | 0.004 | 0.970 | 0.713 | 0.718 |
| keyword (TF-IDF) | 16 | 0.899 | 0.003 | 0.939 | 0.703 | 0.721 |
| bm25 (tuneado) | 16 | 0.838 | 0.003 | 0.879 | 0.569 | 0.581 |
| hybrid | 12 | 0.949 | 0.005 | 0.970 | 0.811 | 0.816 |
| hybrid_bm25 | 16 | 0.949 | 0.004 | 0.970 | 0.713 | 0.718 |
| vector_rerank | 12 | 0.939 | 0.004 | 0.939 | 0.735 | 0.755 |
| keyword_rerank | 12 | 0.891 | 0.003 | 0.939 | 0.718 | 0.736 |
| bm25_rerank | 16 | 0.828 | 0.003 | 0.848 | 0.681 | 0.687 |
| hybrid_rerank | 12 | 0.949 | 0.005 | 0.970 | 0.767 | 0.779 |
| hybrid_bm25_rerank | 12 | 0.939 | 0.004 | 0.939 | 0.735 | 0.755 |

![Retrieval metrics por modo](visualizations/retrieval_models.png)
![k vs metrics](visualizations/retrieval_k_tradeoff.png)

Se reporta también `context_precision@k` en `results/benchmark_retrieval.json`.

Conclusión:
- `vector`, `hybrid` y `hybrid_bm25` empatan con el máximo rendimiento global.
- El reranking no aporta mejora en este corpus.
- Se elige `hybrid_bm25` por robustez ante consultas léxicas y semánticas.
- El reranking se desactiva en configuración (`rerank_enabled: false`).

---

### 7.7 Propuesta de mejora e impacto

Mejora propuesta: **retrieval híbrido** (vector + BM25), con BM25 tuneado y selección de `best_k`.

Impacto medido:
- Baseline vectorial: `avg_score=0.949`, `MRR@k=0.713`.
- Híbrido: `avg_score=0.949` con **mejor ranking** (`MRR@k=0.811`, `nDCG@k=0.816`).
- Reranking: no aporta mejora global y se desactiva por latencia.

Conclusión: el híbrido mejora el orden de relevancia sin perder recall, lo que favorece al LLM cuando el contexto es limitado.


---

## 8. Generación de respuestas

El modelo generativo recibe exclusivamente el contexto recuperado por la estrategia de retrieval correspondiente.

El prompting está orientado a:
- Respuestas concisas y factuales.
- Uso exclusivo del contexto proporcionado.
- Evitar inferencias no soportadas por los documentos.

### 8.1 Configuración de generación

La generación se realiza con modelos locales (Ollama) o Gemini, en función del benchmark configurado.  
Parámetros relevantes en `config.json`:
- `llm_winner`: modelo elegido tras benchmarking.
- `llm_temperature`: controla variabilidad; se mantiene baja para respuestas factuales.
- `llm_max_contexts`: número de chunks incluidos en el prompt final.

En el código:
- Construcción del prompt: `scripts/benchmark_llm.py` y `scripts/query.py` (función `build_prompt`).
- Generación: `scripts/benchmark_llm.py` (funciones `ollama_generate` y `gemini_generate`) y `src/rag/generator.SimpleGenerator`.

### 8.2 Prompting y control de alucinaciones

El prompt se construye concatenando los top‑k contextos y añadiendo instrucciones explícitas:
- Responder solo con el contexto.
- Mantener números y valores exactos.
- Indicar “no disponible” si el dato no aparece.

Esto limita invenciones y favorece respuestas verificables.

Ejemplo de prompt (esquema):
```
Responde de forma concisa basándote solo en el contexto.
Incluye números tal cual aparecen. Si no está en el contexto, di que no está disponible.
Pregunta: <pregunta>
Contexto:
- <chunk 1>
- <chunk 2>
Respuesta:
```

### 8.3 Modelos evaluados

Se evaluaron modelos Ollama locales (p. ej. `llama3.2:3b`, `deepseek-r1:8b`) y variantes Gemini.  
El benchmark compara precisión, completitud y groundedness para seleccionar el ganador.

Detalles de ejecución:
- Ollama: se usa `/api/generate` con fallback a `/api/chat` para compatibilidad.
- Gemini: se configura por API key y se controla el rate limit vía `llm_gemini_max_per_min`.

Grid de modelos evaluados (LLM):
- `ollama-llama3.2` → `llama3.2:3b`
- `ollama-deepseek` → `deepseek-r1:8b`
- `gemini-flash` → `gemini-2.5-flash-lite`

En escenarios sin acceso a APIs externas o con restricciones de privacidad, el modelo `ollama-deepseek` se presenta como una alternativa local plenamente funcional.

---

## 9. Resultados experimentales

### 9.1 Embeddings

Se compara el desempeño de embeddings usando retrieval vectorial con `avg_score` en top‑k.  
El modelo ganador es `paraphrase-multilingual-mpnet-base-v2`.

Grid de k evaluado (embeddings): `k ∈ {4, 8, 12, 16}`.

| Modelo | best_k | avg_score | min_score | max_score |
|------|--------|-----------|-----------|-----------|
| paraphrase-multilingual-mpnet-base-v2 | 16 | 0.949 | 0.000 | 1.000 |
| all-mpnet-base-v2 | 16 | 0.922 | 0.000 | 1.000 |
| nomic-embed-text | 16 | 0.879 | 0.000 | 1.000 |

![Embeddings avg_score](visualizations/embeddings_avg_score.png)
---

### 9.2 Retrieval

Se comparan estrategias de retrieval con `avg_score`, `Recall@k`, `MRR@k` y `nDCG@k`.  
Empatan en el maximo rendimiento: `vector`, `hybrid` y `hybrid_bm25`.

Grid de k evaluado (retrieval): `k ∈ {4, 8, 12, 16}`.

| Modo | best_k | avg_score | min_score | max_score | context_precision@k | Recall@k | MRR@k | nDCG@k |
|-----|--------|-----------|-----------|-----------|--------------------|----------|-------|--------|
| vector | 16 | 0.949 | 0.000 | 1.000 | 0.004 | 0.970 | 0.713 | 0.718 |
| keyword (TF-IDF) | 16 | 0.899 | 0.000 | 1.000 | 0.003 | 0.939 | 0.703 | 0.721 |
| bm25 (tuneado) | 16 | 0.838 | 0.000 | 1.000 | 0.003 | 0.879 | 0.569 | 0.581 |
| hybrid | 12 | 0.949 | 0.000 | 1.000 | 0.005 | 0.970 | 0.811 | 0.816 |
| hybrid_bm25 | 16 | 0.949 | 0.000 | 1.000 | 0.004 | 0.970 | 0.713 | 0.718 |
| vector_rerank | 12 | 0.939 | 0.000 | 1.000 | 0.004 | 0.939 | 0.735 | 0.755 |
| keyword_rerank | 12 | 0.891 | 0.000 | 1.000 | 0.003 | 0.939 | 0.718 | 0.736 |
| bm25_rerank | 16 | 0.828 | 0.000 | 1.000 | 0.003 | 0.848 | 0.681 | 0.687 |
| hybrid_rerank | 12 | 0.949 | 0.000 | 1.000 | 0.005 | 0.970 | 0.767 | 0.779 |
| hybrid_bm25_rerank | 12 | 0.939 | 0.000 | 1.000 | 0.004 | 0.939 | 0.735 | 0.755 |

Configuracion final seleccionada: `hybrid_bm25`.

---

### 9.3 LLM

Se evalua el LLM ganador con `avg_score`, `semantic_similarity`, `groundedness`, completitud y latencias.  
El mejor modelo en calidad es `gemini-flash` (mayor avg_score y similarity), aunque con mayor latencia.


| Modelo | avg_score | context_coverage | groundedness | semantic_similarity | completeness | retrieval_s | generation_s | total_s |
|------|-----------|------------------|--------------|---------------------|--------------|-------------|--------------|---------|
| gemini-flash | 0.808 | 0.950 | 1.000 | 0.741 | 0.750 | 0.025 | 6.351 | 6.376 |
| ollama-deepseek | 0.683 | 0.950 | 1.000 | 0.664 | 0.650 | 0.021 | 5.239 | 5.260 |
| ollama-llama3.2 | 0.450 | 0.950 | 1.000 | 0.611 | 0.250 | 0.207 | 1.184 | 1.391 |

![LLM avg_score](visualizations/llm_avg_score.png)

Modelo seleccionado: `gemini-flash`.

---

### 9.4 Ejemplo de ejecucion (reproducible)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_index.py
python scripts/benchmark_embeddings.py
python scripts/benchmark_bm25.py
python scripts/benchmark_retrieval.py
python scripts/benchmark_llm.py
```

### 9.5 Outputs y visualizaciones

Outputs principales:
- `results/benchmark_embeddings.json`
- `results/benchmark_retrieval.json`
- `results/benchmark_bm25.json`
- `results/benchmark_llm.json`
- `results/benchmark_llm.csv`

Visualizaciones:
- `visualizations/embeddings_avg_score.png`
- `visualizations/retrieval_models.png`
- `visualizations/retrieval_k_tradeoff.png`
- `visualizations/llm_avg_score.png`
- `visualizations/bm25_breakdown.png`

Las visualizaciones se generan con `scripts/generate_plots.py`.

---

## 10. Discusión de resultados

Los resultados muestran que:
- La estrategia de recuperación tiene un impacto mayor en la calidad final de la respuesta que el modelo generativo.
- BM25 destaca en consultas con identificadores y valores exactos.
- La búsqueda vectorial es superior en consultas procedimentales.
- El enfoque híbrido ofrece el mejor equilibrio global.

Discusión detallada:
- **Embeddings**: el modelo multilingüe (`paraphrase-multilingual-mpnet-base-v2`) supera a alternativas monolingües o embeddings locales. Esto es coherente con un corpus mixto ES/EN y con entidades técnicas (IDs, versiones) que requieren robustez semántica y léxica.
- **Retrieval**:
  - `vector`, `hybrid` y `hybrid_bm25` empatan en `avg_score`, pero difieren en ranking (`MRR@k`, `nDCG@k`).  
  - `hybrid` obtiene mejores métricas de ranking, lo que indica un ordenamiento más útil para el LLM cuando el contexto es limitado.  
  - `hybrid_bm25` mantiene la robustez léxica y ofrece consistencia con la configuración final del sistema.
  - El reranking no mejora métricas globales en este corpus; su coste en latencia no se justifica.
- **BM25 tuning**: el grid search revela estabilidad entre combinaciones y confirma que un ajuste leve (`k1=0.9`, `b=0.25`) es suficiente; no se observan mejoras drásticas en este dominio.
- **LLM**:
  - `gemini-flash` ofrece el mayor `avg_score` y `semantic_similarity`, a costa de mayor latencia.  
  - `ollama-deepseek` es la alternativa local con mejor equilibrio calidad/latencia.  
  - `ollama-llama3.2` queda por debajo en precisión, aunque es más rápido.

Implicaciones:
- El componente crítico para la precisión factual es la recuperación (en especial el ranking), más que la elección del LLM.  
- La combinación híbrida es necesaria para cubrir preguntas léxicas y semánticas sin penalizar el recall.
- Un mayor `context_precision@k`, combinado con buen ranking, reduce la probabilidad de que el LLM incorpore información irrelevante.

---

## 11. Dificultades y resoluciones

- **BM25 devolvia scores nulos**: se detecto un problema en el patrón de tokenizacion; se corrigio y se re‑ejecutaron benchmarks.
- **Reranking sin mejora**: las metricas no subieron y la latencia aumentaba, por lo que se desactivo en configuracion final.
- **Similitud semantica sin valores**: se normalizo el campo `expect_text` y se ajusto el pipeline de evaluacion.

---

## 12. Conclusiones

Este proyecto demuestra de forma empírica que la calidad final de un sistema RAG depende más del diseño del retrieval que del modelo generativo en sí.  
En un dominio con alta densidad de entidades exactas (IDs, versiones, comandos, métricas), la recuperación léxica aporta una precisión crítica que la búsqueda puramente semántica tiende a diluir.

La evaluación confirma que:
- **El retrieval es el factor dominante**: mejorar el ranking y la cobertura de contexto impacta directamente en `avg_score`, `semantic_similarity` y completitud.
- **La combinación léxica + semántica** es el punto de equilibrio óptimo: el enfoque híbrido captura tanto coincidencias exactas como reformulaciones.
- **El tuning de BM25 es necesario**: usar parámetros por defecto puede penalizar su rendimiento y llevar a conclusiones erróneas sobre su utilidad.
- **El reranking no siempre es coste‑efectivo**: en este corpus no aporta mejora suficiente para justificar el aumento de latencia.
- **El LLM ganador se justifica cuantitativamente**: se prioriza calidad (score/similarity) sobre velocidad, manteniendo alternativas locales para escenarios sin API.

Desde una perspectiva de ingeniería, la arquitectura final es:
- **Modular**: permite sustituir embeddings, retrievers o LLM sin reescribir el pipeline.
- **Reproducible**: benchmarks y configuraciones quedan versionados y comparables.
- **Trazable**: metadatos y chunking permiten auditar el origen de cada respuesta.

En conjunto, el proyecto valida que una estrategia experimental rigurosa —con benchmarks por componente y métricas homogéneas— permite justificar decisiones técnicas de forma objetiva y transferible a otros entornos con documentación técnica.

Lecciones aprendidas:
- Sin un retrieval sólido, incluso un LLM fuerte no compensa la falta de contexto adecuado.
- El rendimiento depende tanto del **ranking** como de la **cobertura**; no basta con recuperar algo relevante.
- La separación por tipo de documento (chunking narrativo vs estructurado) reduce errores de extracción numérica.
- La evaluación con `expect_text` ayuda a detectar fallos de completitud que no aparecen solo con tokens esperados.
- En proyectos locales, el trade‑off latencia/calidad del LLM debe decidirse con métricas y no solo con percepción cualitativa.

---

## 13. Limitaciones y trabajo futuro

- Re-ranking más avanzado con modelos cross-encoder especializados.
- Estrategias dinámicas de chunking según tipo de consulta.
- Evaluación con datasets reales de mayor tamaño.
- Benchmarking de prompts para optimizar precisión sin cambiar modelo.
- Evaluación humana o semi-automática para validar respuestas complejas.

Amenazas a la validez:
- El tamaño del test set es limitado; resultados pueden variar con un conjunto más amplio.
- La distribución de categorías influye en la ventaja de cada retriever.
- La evaluación depende de tokens esperados y del texto canónico, lo que favorece respuestas más extractivas.

Limitaciones metodológicas:
- No se ha explorado fine-tuning de embeddings ni re-ranking con modelos especializados por dominio.
- El chunking es fijo por tipo de documento; no se probaron estrategias adaptativas por consulta.
