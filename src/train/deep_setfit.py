"""
Entrenamiento y evaluación con SetFit (pocos-shots eficiente) usando all-MiniLM-L6-v2.
- Usa Body + Department concatenados como texto de entrada.
- Entrena y evalúa en el split 80/20 definido en data_prep.
- Guarda métricas en reports/metrics_setfit.json y el modelo en models/setfit_model.

Ejecución:
  python -m src.train.deep_setfit
  FAST_MODE=1 python -m src.train.deep_setfit   # menos pasos/epochs
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import REPORTS_DIR, SEED
from src.data_prep import add_features, load, EXPECTED_PRIORITIES

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
BASE_MODEL = os.getenv("SETFIT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_DIR = Path(os.getenv("SETFIT_OUTPUT_DIR", "models/setfit_model"))
MAX_TRAIN_SAMPLES = int(os.getenv("SETFIT_MAX_TRAIN_SAMPLES", "8000" if not FAST_MODE else "4000"))


def make_dataset(X_df, y):
    texts = (X_df["Body"].astype(str) + " [SEP] " + X_df["Department"].astype(str)).tolist()
    labels = y.astype(str).tolist()
    return texts, labels


def main():
    print(f"[setfit] Inicio con modelo base {BASE_MODEL}. FAST_MODE={FAST_MODE}")
    df = add_features(load())
    df = df[df["Priority"].isin(EXPECTED_PRIORITIES)]
    X = df[["Body", "Department", "n_tags", "len_words"]]
    y = df["Priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Submuestreo para evitar explosión de pares contrastivos y problemas de memoria
    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_train), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        print(f"[setfit] Submuestreo aplicado: {len(X_train)} ejemplos de entrenamiento (max {MAX_TRAIN_SAMPLES})")

    # Codificar labels a enteros para SetFit y mantener mapping para reporte
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    train_texts, train_labels = make_dataset(X_train, y_train_enc)
    test_texts, test_labels = make_dataset(X_test, y_test_enc)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    model = SetFitModel.from_pretrained(BASE_MODEL)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        metric="f1_macro",
        num_iterations=5 if FAST_MODE else 10,
        num_epochs=2 if FAST_MODE else 4,
        batch_size=8 if FAST_MODE else 16,
        learning_rate=2e-5,
    )

    print("[setfit] Entrenando...")
    trainer.train()
    # Guardar checkpoint tras entrenamiento para evitar perderlo si falla la evaluaci�n
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(f"[setfit] Checkpoint guardado en {OUTPUT_DIR} tras entrenar.")

    print("[setfit] Evaluando en test...")
    y_pred_enc = trainer.model.predict(test_texts)
    y_pred = le.inverse_transform(np.asarray(y_pred_enc, dtype=int))
    y_true = y_test.astype(str)

    report = classification_report(y_true, y_pred, labels=le.classes_, output_dict=True)
    accuracy = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "metrics_setfit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "fast_mode": FAST_MODE,
                "train_samples": len(train_texts),
                "test_samples": len(test_texts),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    print(f"[setfit] Métricas guardadas en {out_path}")
    print(classification_report(y_true, y_pred, labels=le.classes_))

    # Guardado final tras evaluar (mismo path)
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(f"[setfit] Modelo final guardado en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
