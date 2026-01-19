# Guia interna RAG (TFM)

Objetivo: documentar decisiones tecnicas con valores por defecto, alternativas y como evaluarlas. Este documento se actualiza a medida que se implementan mejoras.

========================================================
1) ESTRATEGIA DE SPLIT DE TEXTO + EMBEDDINGS
========================================================
Objetivo:
- Crear chunks autosuficientes sin perder precision en preguntas concretas (Q4, INC-xxxx, emails, versiones, comandos).

1.1 Split recomendado (baseline defendible)
- Split jerarquico por estructura del documento:
  A) Markdown: separar por headings (#, ##, ###) => chunk por seccion con headings_path.
  B) PDF: separar por titulos/parrafos. Si no se detectan bien, split por ventana fija.
  C) TXT: separar por bloques/parrafos.
  D) CSV: convertir filas a texto y crear chunks por fila (y opcionalmente un chunk de cabecera).
  E) JSON: chunk por bloque logico (keys principales: features, alerting, contactos).

- Tamaños recomendados:
  - target_chunk_tokens: 350–700 (o 1800–3500 caracteres si no hay tokenizador).
  - overlap: 10–20% (p.ej. 80–120 tokens o 200–400 chars).
  - hard_max: 900–1000 tokens (si supera, re-splitting).

1.2 Metadatos obligatorios por chunk
- doc_id, filename, ext, source_type (pdf/md/txt/csv/json)
- section_title, headings_path (si aplica)
- chunk_id, chunk_index
- offsets (char_start/end o pagina si pdf)

1.3 Limpieza/normalizacion
- Mantener acentos y UTF-8.
- Normalizar espacios y saltos; eliminar caracteres nulos.
- Mantener tokens exactos: Q1/Q2/Q3/Q4, INC-2024-001, emails, versiones (2.3.1), comandos (tn-monitor).

1.4 Generacion de embeddings
- Embeddings por chunk (no por documento).
- Guardar embedding + texto + metadatos.
- Normalizacion L2 si uso cosine/dot.

1.5 Checklist
- ¿Las tablas (SLA, KPIs) estan en chunks separados y consultables?
- ¿Los chunks incluyen suficiente contexto (ej. “Ingeniero Nivel 2: Carlos Ruiz...”)?
- ¿No hay chunks gigantes que mezclan temas distintos?
- ¿Over-splitting? (chunks demasiado pequenos que pierden contexto)

Opcional futuro (si queremos afinar mas):
- Detectar headings en PDF y propagarlos como headings_path.
- Guardar pagina del PDF como offset (ademas de char_start/end).
- doc_id mas semantico (nombre + fecha) para trazabilidad humana.

========================================================
2) ESTRATEGIA DE RETRIEVAL
========================================================
Objetivo:
- Maximizar Recall@K sin introducir ruido excesivo.

2.1 Baseline: Dense retrieval
- Similaridad: cosine (o dot con embeddings normalizados).
- topK inicial: 10–12
- topK final para contexto LLM: 5–8

2.2 Metadata routing/boost (heuristicas baratas)
- Si query contiene “Q1/Q2/Q3/Q4” => boost a kpis_trimestrales.csv
- Si query contiene “INC-” => boost a historico_incidentes_criticos.txt
- Si query contiene “email”, “responsable”, “rol” => boost a config_servicio.json + organigrama_soporte.md
- Si query contiene “tn-” => boost a procedimiento_oncall.md

2.3 Reranking (si esta disponible)
- Recuperar top 30 candidatos (dense/hybrid) y reordenar con cross-encoder.
- Beneficio: reduce chunks equivocados cuando hay secciones parecidas.

2.4 Checklist
- ¿El topK trae el documento correcto de forma consistente?
- ¿Queries exactas (INC-2024-001, 2.3.1, emails) donde dense falla? => justificar hibrido.
- ¿Duplicados del mismo doc? => dedup/diversidad.

========================================================
3) BENCH DE MODELOS
========================================================
Objetivo:
- Demostrar con datos que combinacion funciona mejor en el corpus.

3.1 Que comparar (minimo viable)
A) Embeddings (2–3 opciones) manteniendo retrieval fijo.
B) Retrieval (dense vs hybrid vs hybrid+rerank) con embeddings fijo (el mejor).
C) LLM de respuesta (1–2 opciones) con retrieval fijo.

3.2 Dataset de evaluacion
- TEST_SET actual (KPIs, config, FAQ, manual, politicas, oncall, inventarios, incidentes).
- Anadir 10–20 preguntas negativas (no estan en docs) para medir alucinacion.

3.3 Que reportar
- Recall@K, MRR@K, nDCG@K (retrieval).
- contains-match y groundedness (QA).
- Latencia end-to-end.

========================================================
5) METRICAS DE RAG
========================================================
Separar retrieval vs generacion.

5.1 Retrieval
- Recall@K, MRR@K, nDCG@K.
- Chunk relevante: contiene la mayoria (o todos) los tokens esperados.

5.2 Respuesta (QA)
- contains-match = tokens esperados presentes / total tokens esperados.
- completeness: si pregunta pide 2 campos, exigir ambos.
- groundedness: no introducir entidades que no aparezcan en los chunks recuperados.

5.3 Operativas
- Latencia: retrieval, rerank, generacion, total.
- Coste (si aplica): tokens usados.

5.4 Entregable recomendado
- Reporte por configuracion con promedios globales y breakdown por tipo de pregunta:
  - tablas (CSV)
  - identificadores exactos (INC, versiones, emails)
  - procedimentales (oncall, escalado)
