
# 🔍 Asistente RAG Multiformato  
**Trabajo Final de Máster – Proyecto 2**  
**Autor:** Iván Fernández  

---

## 🧠 1. Introducción

El volumen de documentación dentro de un equipo técnico, departamento de soporte o área de ingeniería suele estar disperso en múltiples formatos: informes en PDF, procedimientos internos en Word, logs y configuraciones en JSON, métricas operativas en CSV y documentación general en TXT o Markdown.

Este proyecto aborda ese problema mediante la construcción de un **asistente inteligente basado en RAG (Retrieval-Augmented Generation)** capaz de:

- Leer información de **al menos 5 formatos diferentes**  
- Indexarla en una base vectorial  
- Recuperar fragmentos relevantes según una pregunta del usuario  
- Generar respuestas precisas usando un LLM  
- Mejorar la calidad de las respuestas mediante una técnica avanzada  

El sistema pretende simular un entorno real donde los equipos necesitan acceder rápidamente a información crítica para la operación diaria.

---

## 🎯 2. Objetivo del proyecto

El objetivo principal es:

> **Construir un asistente RAG multiformato que permita consultar documentación técnica heterogénea mediante lenguaje natural, mejorando la eficiencia, accesibilidad y precisión del conocimiento dentro de una organización.**

Objetivos específicos:

1. Ingerir documentos de **5 formatos**: PDF, DOCX, CSV, JSON, TXT/Markdown.  
2. Procesar y normalizar el contenido textual.  
3. Implementar un sistema de **chunking** adecuado para creación de embeddings.  
4. Generar embeddings usando un modelo avanzado.  
5. Indexar todo en un **vector store** (FAISS o ChromaDB).  
6. Construir un **RAG básico**: retrieval + generación.  
7. Diseñar una **mejora técnica** que incremente la calidad de las respuestas:  
   - re-ranking,  
   - chunking semántico,  
   - prompting avanzado,  
   - o combinación de técnicas.  
8. Evaluar cualitativamente las respuestas antes y después de la mejora.  
9. Documentar todo el sistema con un **esquema visual**.  
10. Describir cómo se llevaría a producción (conceptualmente).

---

## 🗂️ 3. Tipos de documentos utilizados

El asistente procesa los siguientes formatos:

| Formato | Ejemplo de Contenido | Uso |
|---------|-----------------------|-----|
| **PDF** | Manual técnico, políticas internas | Información estructurada y normativa |
| **DOCX** | Procedimientos operativos | Documentación editable |
| **CSV** | KPIs, estadísticas, histórico de tickets | Datos tabulares |
| **JSON** | Logs, configuraciones, parámetros | Información técnica o estructurada |
| **TXT / MD** | FAQ, notas, documentación interna | Conocimiento no estructurado |

Cada formato se ingiere mediante un **loader especializado** para garantizar la extracción correcta del texto.

---

## ⚙️ 4. Arquitectura General del Sistema

El sistema RAG se compone de las siguientes etapas:

### **1. Ingesta Multiformato**
- Módulos independientes para cada tipo de documento.  
- Extracción de texto + metadatos básicos.  
- Almacenamiento en una estructura estandarizada.

### **2. Procesamiento y Limpieza**
- Eliminación de caracteres inválidos  
- Normalización del texto  
- Unión de contenido y metadatos  

### **3. Chunking**
- División en fragmentos apropiados para embeddings.  
- Dos métodos:
  - *Chunking fijo* (básico)  
  - *Chunking semántico* (opción de mejora)

### **4. Embeddings**
- Modelo seleccionado (ej.: SentenceTransformers “all-MiniLM-L6-v2”)  
- Generación de vectores densos para cada chunk  

### **5. Vector Store**
- FAISS o ChromaDB para almacenar embeddings  
- Permite búsquedas vectoriales eficientes  

### **6. Retrieval**
- Recuperación de los *k* chunks más relevantes  
- Filtrado por similitud coseno  

### **7. LLM (Generación)**
- Instrucciones + pregunta + contexto recuperado  
- Respuesta en lenguaje natural  

### **8. Mejora de rendimiento**
- Re-ranking de los chunks recuperados vía Cross-Encoder  
**o**  
- Chunking semántico  
**o**  
- Prompting avanzado  
**o**  
- Recuperación híbrida (BM25 + embeddings)

### **9. Evaluación**
- Comparación entre RAG básico y RAG mejorado  
- Ejemplos reales evaluados manualmente  

---

## 🧩 5. Estructura del repositorio

```text
rag-multiformato/
├── data/
│   ├── pdf/
│   ├── docx/
│   ├── csv/
│   ├── json/
│   └── txt/
├── src/
│   ├── loaders/
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── csv_loader.py
│   │   ├── json_loader.py
│   │   └── txt_loader.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── generator.py
│   └── reranker.py        # Mejora propuesta
├── notebooks/
│   ├── 01_ingesta_pruebas.ipynb
│   ├── 02_retrieval_test.ipynb
│   └── 03_evaluacion.ipynb
├── results/
│   ├── ejemplos_respuestas.md
│   └── comparativa_rag.xlsx
└── README.md
```

---

## 🧪 6. RAG básico — Pipeline

1. El usuario hace una pregunta.  
2. Se genera el embedding de la pregunta.  
3. El sistema busca en el vector store los chunks más similares.  
4. Los top-k chunks se incorporan al prompt.  
5. El LLM genera una respuesta basada en ese contexto.  

Este es el sistema base.

---

## 🚀 7. Mejora técnica implementada (obligatoria)

**Ejemplo (recomendado): Reranking con Cross-Encoder**

1. El retriever devuelve **top 20** chunks.  
2. Se pasa cada chunk + pregunta por un **modelo de re-ranking** (bi-encoder o cross-encoder).  
3. Se reordenan según relevancia real.  
4. El LLM recibe la versión final más precisa.

Beneficios:
- Menos alucinaciones  
- Mejores respuestas en preguntas complejas  
- Mayor cohesión del contexto  

---

## 📊 8. Evaluación del sistema

La evaluación incluye:

### **1. Comparación cualitativa básica vs mejorado**
- Calidad factual  
- Precisión  
- Completitud  
- Relevancia  
- Estabilidad de la respuesta  
- Uso correcto de la fuente  

### **2. Tabla de evaluación**
| Pregunta | RAG básico | RAG mejorado | Mejor |
|----------|------------|--------------|--------|
| ¿Cuál es el flujo de escalado? | Parcial | Completa | Mejorado |
| ¿Qué KPIs del CSV son del Q2? | Incorrecta | Correcta | Mejorado |
| ¿Qué parámetros del JSON configuran X? | Difusa | Precisa | Mejorado |

### **3. Análisis global**
El modelo mejorado:
- Recupera mejor contexto largo  
- Reduce contradicciones  
- Da respuestas más precisas  

---

## 🧱 9. Esquema visual del sistema

```
        DOCUMENTOS (5 formatos)
 PDF ─┐
 DOCX ├─> Loaders → Limpieza → Chunking → Embeddings → VectorStore 
 CSV ─┘

 Usuario → Pregunta → Embedding → Retriever → (Reranker) → LLM → Respuesta
```

---

## ☁️ 10. Despliegue conceptual

El sistema podría desplegarse con:

- **API REST** en FastAPI  
- **Vector store persistente** (FAISS / ChromaDB)  
- **Job programado** para reindexar documentos  
- **Uso de Docker** para portabilidad  
- **Monitorización** del rendimiento del asistente  
- **Control de versiones** al actualizar documentos  

---

## 🧾 11. Limitaciones

- Los documentos requieren cierta calidad estructural  
- El asistente no puede inventar información ausente  
- El rendimiento depende de:
  - embeddings  
  - chunking  
  - técnica de mejora aplicada  

---

## 🔮 12. Mejoras futuras

- Recuperación híbrida BM25 + embeddings  
- Dashboard interactivo (Streamlit/Gradio)  
- RAG multimodal (imágenes + texto)  
- Supervisión humana sobre respuestas  
- Integración con sistemas reales (ServiceNow, Jira, etc.)

---

