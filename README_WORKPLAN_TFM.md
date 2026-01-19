# 🧭 Workplan — Completar README TFM Proyecto 1 (Paso a Paso)

Este documento describe **el orden exacto** y **qué añadir en cada sección** del README para completar el Proyecto 1 del TFM con nivel alto.

👉 Regla general:  
No avances a la siguiente sección hasta que la actual esté “cerrada” (aunque luego se pula).

---

## PASO 0 — Preparación en VSCode
- Abre `README_TFM_Proyecto1.md`
- Activa Markdown Preview y Word Wrap
- No toques resultados ni tablas todavía

---

## PASO 1 — Marco del problema (Secciones 1–2)
- Verifica que se mencionen SLA, coste y asimetría de errores.
- Comprueba que se justifique por qué no es regresión ni ordinal puro.
👉 Si está correcto: **no tocar**.

---

## PASO 2 — Datos y preparación (Sección 3)
Añade una frase final explicando por qué las variables elegidas son adecuadas para el problema.

---

## PASO 3 — Investigación y diseño experimental (Sección 4)
### 4.1 Hipótesis
Explicita expectativas y riesgos antes de ver resultados.

### 4.2 Estrategia de comparación
Justifica por qué comparas familias de modelos y qué significa “ganar”.

---

## PASO 4 — MÉTRICAS (Sección 6)
### 6.1 Principios generales
Aclara que la evaluación es un sistema de decisión, no una métrica única.

### 6.2 Métricas principales
Para cada métrica responde:
- Qué mide
- Por qué es relevante
- Qué decisión habilita

### 6.3 Métricas complementarias
Indica que se reportan pero no se optimizan.

### 6.4 Confusion Matrix
Explica errores críticos y su impacto.

Cierra con un resumen de la estrategia de métricas.

---

## PASO 5 — Métricas por enfoque (Sección 7)
- Embeddings/SetFit: justifica AP y análisis High→Low.
- Zero-shot: coverage, latencia, over-triage, CI.

---

## PASO 6 — Métricas descartadas (Sección 8)
Para cada métrica descartada, explica por qué no ayuda a tomar decisiones.

---

## PASO 7 — Resultados (Sección 9)
Añade lectura guiada: qué confirma y qué contradice tus hipótesis.

---

## PASO 8 — Métricas → Negocio (Sección 10)
Define costes relativos y explica por qué priorizas recall High.

---

## PASO 9 — Modelo final (Sección 11)
Justifica por qué eliges uno y descartas los demás.

---

## PASO 10 — Cierre (Secciones 12–17)
Explicabilidad, despliegue, dificultades, limitaciones y conclusiones coherentes.

---

## Señal de finalización
El README está listo si:
- Se entiende por qué se eligió el modelo
- Está claro qué errores son más graves
- Las métricas cuentan una historia coherente
