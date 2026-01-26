1. Estrategia split de text
    - Como genero embeddings
2. Estrategia de retrieval
    - Mejores
    - Similitud
    - Etc
3. Bench de modelos
4. Forma de hacerlo mas avanzado (key word search)
5. Metricas de RAG

---

Que el expected sea un parrafo y comparar textos (embeddings)

Tecnicas de split [Falta justificar]
Reranking/retrieval y porque [Falta justificar]
FAISS [Falta justificar]
BM25 [X]

En el repo para reproducirlo y como configurar

---

## Estrategias de recuperacion (Retrieval)

El sistema RAG implementa y evalua distintas estrategias de recuperacion de informacion con el objetivo de analizar su impacto en la calidad final de las respuestas generadas por el LLM.

Dado que el corpus incluye tanto informacion altamente estructurada (identificadores, valores numericos, comandos) como documentacion narrativa (procedimientos, politicas, FAQs), se evaluan enfoques lexicos, semanticos e hibridos.

---

### Recuperacion lexica: BM25

BM25 es un algoritmo clasico de recuperacion de informacion basado en coincidencia de terminos, ampliamente utilizado en motores de busqueda tradicionales.

Este enfoque prioriza documentos que contienen exactamente los mismos tokens que la consulta, ponderando su frecuencia y distribucion.

Se utiliza BM25 como baseline lexico, siendo especialmente eficaz para:
- Identificadores exactos (IDs de incidentes, nombres de servicios, comandos)
- Valores numericos y versiones
- Nombres propios y direcciones de correo

No obstante, su rendimiento disminuye ante parafraseo o reformulacion semantica de las consultas.

---

### Recuperacion semantica: Vector Search

La recuperacion semantica se basa en embeddings densos que representan documentos y consultas en un espacio vectorial continuo.

Este enfoque permite recuperar fragmentos relevantes aunque no exista coincidencia exacta de terminos, siendo especialmente adecuado para:
- Procedimientos operativos
- FAQs
- Politicas y documentacion descriptiva

Los embeddings se utilizan unicamente como extractores de caracteristicas, sin ajuste del modelo durante el entrenamiento, con el objetivo de mantener comparabilidad y control experimental.

---

### Recuperacion hibrida: BM25 + Vector

La recuperacion hibrida combina los resultados de BM25 y de la busqueda vectorial para aprovechar las ventajas de ambos enfoques.

En este proyecto, la estrategia hibrida se implementa mediante fusion de rankings, permitiendo:
- Mantener la precision lexica de BM25 en tokens criticos
- Incorporar la generalizacion semantica del vector search

Este enfoque se evalua como candidato principal del sistema, ya que ofrece una cobertura mas robusta ante la diversidad de tipos de preguntas presentes en el corpus.

---

### Justificacion experimental

Las tres estrategias se evaluan de forma independiente y comparativa, permitiendo analizar:
- Diferencias de rendimiento segun el tipo de pregunta
- Fortalezas y limitaciones de cada enfoque
- Beneficios de la combinacion hibrida frente a enfoques individuales

Los resultados se reportan tanto a nivel global como segmentados por categoria semantica.
