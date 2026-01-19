# Zero-shot / Few-shot Classification with LLMs

Este módulo evalúa el uso de **Large Language Models (LLMs)** como clasificadores de tickets de soporte en un escenario **zero-shot y few-shot**, sin entrenamiento específico sobre el dataset.  
El objetivo es **analizar su viabilidad como baseline**, su sensibilidad al prompt y al número de ejemplos, y compararlos conceptualmente con modelos entrenados.

---

## Objetivo del experimento

Evaluar si un LLM es capaz de clasificar correctamente la prioridad de un ticket (`high`, `medium`, `low`) únicamente a partir de:
- El texto del ticket (`Body`)
- El departamento (`Department`)
- Instrucciones en lenguaje natural (prompt)

Este experimento **no busca maximizar rendimiento**, sino:
- Medir **capacidad de generalización sin entrenamiento**
- Analizar **estabilidad y sensibilidad**
- Comparar contra modelos clásicos entrenados

---

## Alcance

Se evalúan:
- Zero-shot (0 ejemplos)
- Few-shot (1, 3, 5, 10 ejemplos)
- Diferentes modelos LLM (cloud y local)
- Diferentes estrategias de prompt
- Métricas técnicas y métricas con impacto de negocio

---

## Modelos evaluados

Dependiendo de la configuración:

| Tipo | Modelo | Proveedor |
|----|----|----|
| Cloud LLM | Gemini 2.5 Flash | Google |
| Cloud LLM | Gemini 2.5 Flash Lite | Google |
| Local LLM | DeepSeek-R1:8B | Ollama |

---

## Estrategia Zero-shot / Few-shot

Cada inferencia incluye:
- Instrucciones de clasificación
- `N_EXAMPLES` ejemplos (opcional)
- Un ticket objetivo a clasificar

---

## Variación experimental

Los experimentos se ejecutan variando:
- Número de ejemplos (`N_EXAMPLES ∈ {0,1,3,5,10}`)
- Prompt utilizado
- Modelo LLM
- Semilla aleatoria
- Selección estratificada de ejemplos

---

## Selección de datos

- Se utiliza el conjunto de test
- Selección estratificada por clase
- Sin solapamiento entre ejemplos y evaluación
- Semilla fija para reproducibilidad

---

## Métricas evaluadas

### Métricas técnicas
- Precision / Recall / F1 por clase
- **Macro F1**
- Matriz de confusión

### Métricas de negocio
- **Recall (High)**: minimizar falsos negativos en tickets críticos

---

## Baselines de referencia

- Clase mayoritaria
- Clasificador aleatorio estratificado
- Modelos entrenados (TF-IDF + SVM)

---

## Resultados y logging

Los resultados se almacenan en:

```
reports/
├── metrics_zero_shot.json
├── zero_shot_predictions_*.csv
```

---

## Reproducibilidad

Variables de entorno principales:

```bash
ZERO_SHOT_PROVIDER=ollama|gemini
ZERO_SHOT_N_EXAMPLES=5
ZERO_SHOT_N_EVAL=100
ZERO_SHOT_SEED=42
ZERO_SHOT_STRATIFIED=1
```

Ejecución:

```bash
python -m src.train.zero_shot
```

---

## Conclusiones esperadas

- Rendimiento razonable en zero-shot
- Mejora limitada con few-shot
- Alta sensibilidad al prompt
- Coste computacional elevado frente a modelos entrenados

---

Este módulo actúa como **baseline conceptual** y refuerza la elección de modelos entrenados en el proyecto.
