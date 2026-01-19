# Embeddings-based Classification – Technical Extensions

Este documento describe **las extensiones técnicas propuestas** para mejorar la fase de clasificación basada en embeddings del Proyecto 1 del TFM, detallando **qué se añade**, **cómo se implementa** y **por qué se elige una opción frente a otras**.

El objetivo no es aumentar la complejidad por sí misma, sino **evaluar de forma controlada el impacto real de decisiones de representación** sobre métricas técnicas y de negocio.

---

## 1. Contexto

En el enfoque base, cada ticket se representa mediante embeddings de texto generados a partir de distintos campos del CSV (principalmente `Body` y `Department`). Estos embeddings se utilizan como entrada a clasificadores supervisados clásicos (Logistic Regression, Linear SVM).

Las extensiones propuestas buscan responder a dos preguntas clave:

1. ¿Qué modelo de embeddings ofrece mejor equilibrio entre calidad y coste?
2. ¿Qué peso informativo aporta cada campo textual al embedding final?

---

## 2. Extensión 1 — Comparativa de modelos de embeddings

### 2.1 Qué se añade

Se amplía el conjunto de modelos de embeddings evaluados, incorporando arquitecturas con distintos compromisos entre:

* calidad semántica
* dimensión del vector
* latencia de inferencia

Modelos propuestos:

* `all-MiniLM-L6-v2` (baseline ligero)
* `all-mpnet-base-v2` (baseline de alta calidad)
* `all-distilroberta-v1` (intermedio)
* `multi-qa-MiniLM-L6-cos-v1` (orientado a recuperación semántica)

### 2.2 Cómo se implementa

Para cada modelo:

1. Se generan embeddings para el mismo conjunto de tickets.
2. Se cachean los embeddings para evitar recomputación.
3. Se mide:

   * tiempo total de generación de embeddings
   * dimensión del embedding
   * métricas de clasificación (F1-macro, recall de High)

Los resultados se almacenan en un CSV/JSON de benchmarking.

### 2.3 Por qué se hace así

* Limitar el número de modelos evita ruido experimental.
* Comparar modelos de la misma familia permite conclusiones claras.
* Medir latencia permite justificar decisiones desde un punto de vista industrial.

No se realiza fine-tuning de los modelos, ya que el objetivo es evaluar **representaciones preentrenadas reutilizables**, coherente con un escenario realista de despliegue.

---

## 3. Extensión 2 — Ponderación de campos en el embedding

### 3.1 Qué se añade

Se introduce una estrategia de **media ponderada** entre embeddings de distintos campos:

```
embedding_final = α · embedding_body + (1 − α) · embedding_department
```

Valores evaluados:

* α = 0.3
* α = 0.5 (baseline)
* α = 0.7
* α = 0.9

### 3.2 Cómo se implementa

1. Se calculan embeddings independientes para `Body` y `Department`.
2. Ambos embeddings se normalizan (L2).
3. Se combinan mediante media ponderada.
4. El embedding resultante se usa como entrada al clasificador.

Cada valor de α se trata como una estrategia distinta.

### 3.3 Por qué media ponderada y no concatenación ponderada

* En la concatenación, el clasificador ya aprende pesos relativos.
* Ponderar antes de concatenar introduce efectos de escala difíciles de interpretar.
* La media ponderada permite **aislar explícitamente la contribución semántica** de cada campo.

Esta estrategia permite responder directamente:

> “¿Dónde está la mayor parte de la señal predictiva: en el texto libre o en el metadato?”

---

## 4. Alternativa evaluada — Department como variable categórica

### 4.1 Motivación

`Department` suele ser:

* corto
* categórico
* con un número limitado de valores

En estos casos, los embeddings pueden introducir ruido semántico innecesario.

### 4.2 Estrategia

Se añade una variante híbrida:

* Embedding de `Body`
* One-Hot Encoding de `Department`
* Variables numéricas (longitud, número de palabras, etc.)

Todo ello concatenado en un único vector de entrada.

### 4.3 Por qué esta opción es relevante

* Permite comparar embedding vs OHE para un mismo campo.
* Mejora la interpretabilidad.
* Refleja prácticas habituales en sistemas industriales.

---

## 5. Orden recomendado de experimentación

1. Media ponderada Body/Department (impacto alto, coste bajo).
2. Department como OHE frente a embedding.
3. Comparativa de modelos de embeddings con métricas y latencia.

Este orden maximiza el valor experimental sin incrementar innecesariamente la complejidad.

---

## 6. Conclusión

Estas extensiones no buscan mejorar métricas de forma aislada, sino **entender qué decisiones de representación aportan valor real**. El resultado es un pipeline más explicable, comparable y alineado con criterios técnicos y de negocio.
