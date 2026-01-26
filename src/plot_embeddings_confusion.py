"""
Genera matriz de confusion para el modelo de embeddings seleccionado.
Uso:
  python -m src.plot_embeddings_confusion
"""

import csv
import os
from pathlib import Path

import numpy as np

from src.config import FIG_DIR, REPORTS_DIR

# Evita escribir cache en home no accesible.
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / "mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402


MODEL_NAME = os.getenv("EMB_CM_MODEL", "sentence-transformers/all-mpnet-base-v2")
CLASSIFIER = os.getenv("EMB_CM_CLASSIFIER", "linear_svm")
FEATURE_STRATEGY = os.getenv("EMB_CM_STRATEGY", "body_ohe_num")
C_VALUE = float(os.getenv("EMB_CM_C", "2.0"))


def _find_experiment_id():
    summary_path = REPORTS_DIR / "experiments" / "embeddings_summary.csv"
    with summary_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("embedding_model") == MODEL_NAME
                and row.get("classifier") == CLASSIFIER
                and row.get("feature_strategy") == FEATURE_STRATEGY
                and float(row.get("C", "0")) == C_VALUE
            ):
                return row.get("id")
    raise ValueError("No matching experiment found in embeddings_summary.csv")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = _find_experiment_id()
    exp_path = REPORTS_DIR / "experiments" / f"{exp_id}.json"
    data = __import__("json").load(exp_path.open(encoding="utf-8"))
    cm = np.array(data["test"]["confusion_matrix"])
    labels = ["high", "medium", "low"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (embeddings)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = FIG_DIR / "embeddings_confusion_matrix.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[embeddings_plots] Guardado {out_path}")


if __name__ == "__main__":
    main()
