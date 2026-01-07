# RAG multiformato (TFM Proyecto 2)

Pipeline RAG local para 5 formatos (PDF/MD/TXT/CSV/JSON) con evaluación reproducible. Este README resume decisiones técnicas, comparativas y métricas conforme al enunciado del proyecto.

## Objetivo
Construir un asistente RAG que responda preguntas sobre documentos internos de soporte y evaluar, con métricas, qué combinación de embeddings, retrieval y LLM funciona mejor.

## Alcance del proyecto
- Asistente RAG multiformato (5 formatos mínimos).
- Recuperación y generación evaluadas con benchmarks reproducibles.
- Enfoque en respuestas factuales (KPIs, incidentes, emails, comandos, versiones).

## Datos y formatos
Los documentos están en `data/` y cubren:
- PDF (manuales y políticas)
- Markdown (procedimientos y organigramas)
- TXT (FAQ e incidentes)
- CSV (KPIs e inventarios)
- JSON (configuración de servicios)

## Esquema del sistema (texto)
data/ -> loaders -> limpieza -> chunking -> embeddings -> FAISS -> retrieval -> LLM -> respuesta

## Estructura del repo
- `src/`: loaders, pipeline, embeddings, retriever, generator
- `scripts/`: build/query/eval/benchmarks
- `results/`: métricas de benchmarks

Motivo de la estructura:
- Separar **lógica reusable** (`src/`) de **ejecuciones reproducibles** (`scripts/`).
- Facilitar benchmarking y trazabilidad de resultados (`results/`).

Detalle por carpeta:
- `src/`: módulos de ingesta, preprocesado, chunking, embeddings, vectorstore y retrieval.
- `scripts/`: scripts de orquestación (indexación, consulta, benchmarks).
- `results/`: salidas de benchmarks para documentación y comparación.

## Arquitectura técnica (detalle)
Este proyecto sigue un flujo clásico de RAG con decisiones explícitas en cada etapa:

1) **Ingesta por formato**
- `src/loaders/*`: cada loader normaliza un formato distinto (PDF/MD/TXT/CSV/JSON).
- Objetivo: convertir cada fuente a texto consistente y aportar metadatos de trazabilidad.

2) **Preprocesado**
- `src/pipeline/preprocessing.clean_text`: normaliza espacios y saltos sin perder caracteres críticos.
- Se evita eliminar acentos para no romper tokens exactos (emails, INC‑xxxx, versiones).

3) **Chunking**
- `src/pipeline/chunking.semantic_chunk`: separa por secciones (headings) y aplica ventana con solape.
- `src/pipeline/chunking.fixed_chunk`: fijo para CSV/JSON para no mezclar filas ni claves.
- Se añade metadata de offsets y headings para rastreo.

4) **Embeddings**
- `src/embeddings/embedder.Embedder`: embeddings por chunk (no por doc).
- Normalización L2 para FAISS IP (similitud coseno).
- Caché en `index/embeddings_cache.json` para no recomputar.

5) **Vector store**
- `src/rag/vectorstore.VectorStore`: FAISS + persistencia en `index/`.
- `index/index.faiss` + `index/data.json` (texto + metadatos por chunk).

6) **Retrieval**
- `src/rag/retriever.HybridRetriever`: combina embeddings + keyword (TF‑IDF).
- Motivación: tokens exactos (INC, versiones, emails) se benefician de keyword.
- `retriever_k` y `retriever_alpha` controlan tamaño y mezcla.

7) **LLM**
- Benchmark con Ollama (local) y Gemini (API).
- Prompt restrictivo: responde solo con contexto.

## Uso rápido
1) Crear entorno: `python -m venv .venv && source .venv/bin/activate`
2) Instalar deps: `pip install -r requirements.txt`
3) Construir índice: `python scripts/build_index.py`
4) Consultar: `python scripts/query.py`
5) Benchmark embeddings: `python scripts/benchmark_embeddings.py`
6) Benchmark retrieval: `python scripts/benchmark_retrieval.py`
7) Benchmark LLM: `python scripts/benchmark_llm.py`

## Configuración actual (resumen)
- Embeddings: `sentence-transformers` con `paraphrase-multilingual-mpnet-base-v2`
- GPU: `embed_device: cuda`
- Retrieval: híbrido (dense + keyword) con `retriever_k: 12` y `retriever_alpha: 0.6`
- LLM ganador: `llm_winner` (actualmente `gemini-flash`)

## Configuración detallada (config.json)
Parámetros clave:
- `embed_backend`, `st_model_name`, `embed_device`
- `retriever_type`, `retriever_k`, `retriever_alpha`
- `llm_winner`, `llm_temperature`, `llm_max_contexts`
- `llm_benchmark_models`, `llm_benchmark_limit`, `llm_gemini_max_per_min`

## Decisiones técnicas clave
- **Embeddings**: se priorizó un modelo multilingüe por mezcla de términos en ES/EN y entidades exactas.
- **Chunking**: se eligió un split por secciones para no mezclar temas y preservar contexto en procedimientos.
- **Retrieval**: híbrido porque mejora el orden de relevancia; el recall iguala al vectorial en K altos.
- **LLM**: se priorizó calidad de respuesta; se mantiene alternativa local por coste/rate limit.

## Estrategia de split y metadatos
- PDF/MD/TXT: chunking por secciones + ventana con solape.
- CSV/JSON: chunking fijo por filas o bloques lógicos.
- Metadatos por chunk: `doc_id`, `filename`, `ext`, `source_type`, offsets y headings cuando aplica.

Detalle técnico del chunking:
- **Secciones (headings)**: secciones agregadas con `headings_path` para mantener contexto de tema.
- **Ventana con solape**: evita cortes que rompan frases/pasos.
- **CSV/JSON**: bloques pequeños para consultas numéricas exactas.

## Dataset de evaluación (TEST_SET)
- 33 preguntas alineadas a los documentos reales en `data/`.
- Categorías: tablas, configuración, identificadores y procedimientos.
- Cada pregunta incluye tokens esperados para evaluar retrieval y LLM.

## Razonamiento de los tests (por qué existen)
El TEST_SET no busca cubrir "todas" las preguntas posibles, sino representar los tipos de consultas reales que el sistema debe resolver y los riesgos típicos en RAG.

1) **Tablas (CSV)**
- **Qué se prueba**: exactitud numérica y extracción de KPIs (SLA, CSAT, tickets).
- **Por qué**: los números suelen fallar por chunking agresivo o por mezclas de filas.
- **Decisión**: se mantienen filas como unidades y se evalúan tokens exactos (ej. 93.1, 1500).

2) **Configuración (JSON)**
- **Qué se prueba**: lectura de parámetros clave y listas (features, alerting).
- **Por qué**: JSON suele perderse si se aplana sin estructura; se requieren claves específicas.
- **Decisión**: chunk por bloque lógico y expected con claves/valores relevantes.

3) **Identificadores exactos**
- **Qué se prueba**: recuperación de entidades con formato rígido (INC-2024-001, emails, versiones).
- **Por qué**: embeddings tienden a diluir tokens exactos; se necesita garantizar precisión.
- **Decisión**: tests con tokens exactos y groundedness para evitar invenciones.

4) **Procedimientos**
- **Qué se prueba**: pasos, tiempos y acciones correctas (on-call, escalado, políticas).
- **Por qué**: son textos largos donde el LLM tiende a resumir o omitir pasos.
- **Decisión**: expected con verbos clave y requisitos mínimos (ej. "10 minutos", "escalado").

## Cómo se evalúa cada test
- **Retrieval**: se concatena top‑k y se comprueba si aparecen los tokens esperados.
- **LLM**:
  - contains‑match: proporción de tokens esperados presentes.
  - completeness: exige todos los tokens esperados.
  - groundedness: penaliza entidades que no están en el contexto.
- **Latencias**: se miden tiempos de retrieval y generación por pregunta.

## Estrategia de retrieval (baseline + mejora)
Baseline: retrieval vectorial con FAISS.  
Mejora propuesta: retrieval híbrido (vector + keyword), manteniendo top‑k fijo.

Comparativa (mejor K por modo):
- Vector (K=16): Recall 0.970, MRR 0.713, nDCG 0.717
- Keyword (K=16): Recall 0.939, MRR 0.694, nDCG 0.722
- Hybrid (K=12): Recall 0.970, MRR 0.795, nDCG 0.816

Decisión: híbrido por mejor ordenamiento (MRR/nDCG) manteniendo el recall máximo.

## Benchmark de embeddings
Resultados clave (avg_score con K=16):
- `paraphrase-multilingual-mpnet-base-v2`: 0.985 (ganador)
- `all-mpnet-base-v2`: 0.952
- `nomic-embed-text`: 0.942

Archivos: `results/benchmark_embeddings.json`, `results/benchmark_embeddings.csv`

## Benchmark de retrieval
Se evalúa Recall@K, MRR@K y nDCG@K para vector/keyword/hybrid.
Archivo: `results/benchmark_retrieval.json`

Cómo se calcula:
- **Recall@K**: existe al menos un chunk relevante en top‑K.
- **MRR@K**: posición del primer chunk relevante.
- **nDCG@K**: calidad del ranking completo.
- Relevancia se define por proporción de tokens esperados presentes en el chunk.

## Benchmark de LLM
Métricas:
- contains-match: % tokens esperados en la respuesta
- completeness: % de preguntas con respuesta completa
- groundedness: no inventa entidades fuera del contexto
- latencias: retrieval/generación/total

Resultados globales (avg_score / completeness):
- Gemini: 0.796 / 0.70 (mejor calidad)
- DeepSeek: 0.738 / 0.65 (local y sin rate limit)
- Llama3.2: 0.571 / 0.45

Decisión (llm_winner):
- Ganador por calidad: Gemini (mejor avg_score y completitud).
- Pros: mayor precisión y mejor desempeño en procedimientos.
- Contras: rate limits y mayor latencia frente a local.
- Motivo: priorizamos calidad de respuesta en la evaluación; se mantiene la alternativa local (DeepSeek) para escenarios sin API.

Breakdown por tipo (tablas, configuración, identificadores, procedimientos) en:
- `results/benchmark_llm.json`
- `results/benchmark_llm.csv`

## Dificultades y cómo se resolvieron
- **Tokens exactos** (INC-xxxx, emails, versiones): se mantuvieron intactos en limpieza y se evaluaron explícitamente.
- **PDFs con estructura irregular**: se optó por chunking por secciones y ventana para reducir ruido.
- **Rate limits en LLM**: se añadió throttling configurable para Gemini.

## Por qué se probaron estos modelos y no otros
- **Embeddings**: se comparó un modelo local rápido (nomic), uno fuerte en inglés (all‑mpnet) y uno multilingüe (paraphrase‑multilingual) por mezcla ES/EN en el corpus.
- **Retrieval**: se comparó vector, keyword e híbrido por presencia de tokens exactos en el dominio.
- **LLM**: se comparó local (ollama) vs API (Gemini) para balancear calidad vs coste/rate limits.

## Scripts principales (qué hace cada uno)
- `scripts/build_index.py`: ingesta, limpieza, chunking, embeddings y FAISS.
- `scripts/query.py`: consulta interactiva con retrieval + LLM.
- `scripts/benchmark_embeddings.py`: compara embeddings con retrieval fijo.
- `scripts/benchmark_retrieval.py`: compara estrategias de retrieval.
- `scripts/benchmark_llm.py`: compara LLMs con métricas y latencias.

## Limitaciones conocidas
- El benchmark usa tokens esperados (no evaluación semántica completa).
- No se calcula coste por tokens (pendiente).
- La calidad depende del corpus sintético de `data/`.

## Ejemplos de ejecución
- Construcción de índice: `python scripts/build_index.py`
- Benchmark LLM: `python scripts/benchmark_llm.py`

## Evaluación y por qué estas métricas bastan
Estas métricas son suficientes para un corpus cerrado y respuestas factuales: miden recuperación correcta (retrieval) y respuesta fiel (LLM).  
Métricas más complejas (ROUGE, BERTScore, LLM‑as‑judge) añaden coste/variabilidad y no aportan mucho en respuestas cortas con datos exactos.

## Reproducibilidad
- Configuración central en `config.json`.
- Benchmarks guardados en `results/`.
- Scripts independientes para embeddings/retrieval/LLM.

## Evidencias
- Embeddings: `results/benchmark_embeddings.json`, `results/benchmark_embeddings.csv`
- Retrieval: `results/benchmark_retrieval.json`
- LLM: `results/benchmark_llm.json`, `results/benchmark_llm.csv`
- TEST_SET: `src/eval_data.py`

## Pendiente / no implementado
- Cálculo de coste (tokens/€ por consulta).
- Reranking del contexto antes del prompt (mejora futura del LLM).
