# TFM — Ticket Priority Classification with SetFit (Fine‑Tuning)

Este README describe **de forma completa y operativa** cómo implementar y justificar un pipeline de **clasificación de prioridad de tickets con SetFit**, alineado con los requisitos de un **TFM bien evaluado**: metodología clara, separación selección/evaluación, métricas justificadas, reproducibilidad y trazabilidad experimental.

Este documento está pensado para:
- Servir como **guía de trabajo en Visual Studio Code**
- Dejar **documentada la metodología** para la memoria del TFM
- Permitir **reproducir exactamente los resultados**

---

## 1. Objetivo del experimento

El objetivo es realizar **fine‑tuning con SetFit** para clasificar tickets en:
- `High`
- `Medium`
- `Low`

Mejorando el baseline TF‑IDF mediante:
- uso de **embeddings semánticos**
- **selección de modelo basada en validación**
- métricas alineadas con **impacto de negocio (SLA)**

---

## 2. Cambios clave respecto a la versión anterior

Este experimento introduce explícitamente:

1. Uso de **SetFit** como método de fine‑tuning supervisado sobre embeddings
2. Comparación de **múltiples modelos base**
3. Split **70 / 20 / 10** (train / validation / test)
4. Eliminación de Cross‑Validation para SetFit
5. Métricas ampliadas y justificadas
6. Búsqueda controlada de hiperparámetros
7. FAST_MODE para desarrollo rápido
8. Guardado de artefactos experimentales

---

## 3. Entorno de trabajo (Visual Studio Code)

### 3.1. Requisitos
- Python ≥ 3.9
- Visual Studio Code
- (Recomendado) WSL + Ubuntu

### 3.2. Crear entorno virtual

#### Linux / WSL
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn torch setfit sentence-transformers datasets matplotlib
```

#### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install pandas numpy scikit-learn torch setfit sentence-transformers datasets matplotlib
```

---

## 4. Estructura del proyecto

```
project/
├── data/
│   └── data_clean.csv
├── scripts/
│   └── deep_setfit.py
├── results/
│   └── setfit/
│       ├── runs/
│       ├── best/
│       └── final_test/
├── models/
│   └── setfit/
└── README_SET_FIT.md
```

---

## 5. Dataset y features

### 5.1. Texto principal
- `Body`
- `Department`

### 5.2. Metadatos (opcional pero recomendado)
- `n_tags`
- `len_words`

Estos metadatos **no pueden usarse directamente en SetFit**, por lo que se incorporan como **tokens textuales discretizados**, por ejemplo:

```
Body [SEP] dept=IT tags=3‑5 len=100‑200
```

Esto permite justificar:
> “La integración de metadatos estructurados en el espacio semántico mediante tokens textuales”.

---

## 6. Split de datos (OBLIGATORIO)

Se utiliza un split estratificado:

- **Train (70%)** → entrenamiento
- **Validation (20%)** → selección de modelo e hiperparámetros
- **Test (10%)** → evaluación final (una única vez)

> El conjunto de test **no se utiliza** durante la selección.

---

## 7. Modelos base evaluados

Se evalúan distintos modelos Sentence Transformers:

- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- `paraphrase-multilingual-MiniLM-L12-v2`
- `thenlper/gte-small`

La elección final se basa exclusivamente en rendimiento en **validación**.

---

## 8. Hiperparámetros explorados

Se realiza una búsqueda controlada sobre:

- `num_iterations`
- `num_epochs`
- `batch_size`
- `learning_rate`

### FAST_MODE
- `FAST_MODE=1`: grid reducido + menos datos
- `FAST_MODE=0`: ejecución completa (resultados finales)

---

## 9. Métricas reportadas (y justificación)

### Métricas principales
- **Macro‑F1**
  - Robusta al desbalance
  - Trata todas las clases por igual
- **Recall (High)**
  - Minimiza falsos negativos en tickets críticos
  - Directamente alineada con SLA y coste operativo
- **Confusion Matrix**
  - Explicabilidad operativa
  - Permite analizar errores High → Medium / Low

### Métrica complementaria
- **Balanced Accuracy**
  - Media de recalls por clase

---

## 10. Criterio de selección de modelo

El modelo final se selecciona usando el conjunto de validación mediante:

```
selection_score = 0.7 · macro_f1 + 0.3 · recall_high
```

Esto equilibra:
- rendimiento global
- prioridad de negocio (High)

---

## 11. Protocolo experimental

### 11.1. Selección
1. Entrenar en `train`
2. Evaluar en `validation`
3. Guardar métricas por configuración
4. Elegir la mejor configuración

### 11.2. Evaluación final
1. Reentrenar con `train + validation`
2. Evaluar **una sola vez** en `test`
3. Guardar métricas finales

---

## 12. Artefactos guardados

### Por experimento
- Configuración
- Métricas de validación
- selection_score

`results/setfit/runs/run_*.json`

### Mejor modelo
- `best_config.json`
- `best_val_metrics.json`

`results/setfit/best/`

### Evaluación final
- `test_metrics.json`
- `confusion_matrix.png` (opcional)

`results/setfit/final_test/`

---

## 13. Ejecución

### Desarrollo
```bash
FAST_MODE=1 python scripts/deep_setfit.py
```

### Resultados finales
```bash
FAST_MODE=0 python scripts/deep_setfit.py
```

---

## 14. Reproducibilidad (TFM)

- Seed fija
- Versiones de librerías
- Artefactos persistidos
- Test aislado

---

## 15. Checklist final TFM

- [ ] Split 70/20/10
- [ ] Comparativa de modelos base
- [ ] Métricas justificadas
- [ ] Selección en validación
- [ ] Test una sola vez
- [ ] Resultados reproducibles
- [ ] Artefactos guardados

---

Este README puede incorporarse directamente al repositorio y referenciarse en la memoria del TFM como **descripción metodológica del experimento SetFit**.
