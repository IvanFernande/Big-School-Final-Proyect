"""
Genera figuras para comparar estrategias de combinación de features en embeddings.
Uso:
  python -m src.plot_embeddings_strategies
"""

import csv
import os
from pathlib import Path

from src.config import FIG_DIR, REPORTS_DIR

# Evita escribir cache en home no accesible.
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / "mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402


MODEL_NAME = os.getenv("EMB_PLOT_MODEL", "sentence-transformers/all-mpnet-base-v2")
CLASSIFIER = os.getenv("EMB_PLOT_CLASSIFIER", "linear_svm")
C_VALUE = float(os.getenv("EMB_PLOT_C", "2.0"))


def _load_rows():
    summary_path = REPORTS_DIR / "experiments" / "embeddings_summary.csv"
    with summary_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_metrics(rows, strategy: str):
    for r in rows:
        if (
            r.get("embedding_model") == MODEL_NAME
            and r.get("classifier") == CLASSIFIER
            and float(r.get("C", "0")) == C_VALUE
            and r.get("feature_strategy") == strategy
        ):
            return float(r["test_f1_macro"]), float(r["test_recall_high"])
    raise ValueError(f"Missing strategy {strategy} for {MODEL_NAME} {CLASSIFIER} C={C_VALUE}")


def _plot_group(ax, title, strategies, labels, rows):
    f1_vals = []
    rec_vals = []
    for strat in strategies:
        f1, rec = _get_metrics(rows, strat)
        f1_vals.append(f1)
        rec_vals.append(rec)

    x = list(range(len(labels)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], f1_vals, width, label="Macro-F1")
    ax.bar([i + width / 2 for i in x], rec_vals, width, label="Recall High")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.4, 0.7)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def plot_A(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_group(
        axes[0],
        "Body + Dept (solo texto)",
        ["body_dept_concat", "body_dept_avg", "body_dept_avg_w30", "body_dept_avg_w70", "body_dept_avg_w90"],
        ["concat", "avg", "w30", "w70", "w90"],
        rows,
    )
    _plot_group(
        axes[1],
        "Body + Tags (solo texto)",
        ["body_tags_concat", "body_tags_avg", "body_tags_avg_w30", "body_tags_avg_w70", "body_tags_avg_w90"],
        ["concat", "avg", "w30", "w70", "w90"],
        rows,
    )
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Estrategias A (modelo={MODEL_NAME}, clf={CLASSIFIER}, C={C_VALUE})")
    fig.tight_layout()
    out_path = FIG_DIR / "embeddings_strategies_A.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[embeddings_plots] Guardado {out_path}")


def plot_B(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _plot_group(
        axes[0],
        "Body + Dept (estructura)",
        ["body_dept_concat", "body_dept_concat_num", "body_ohe_num"],
        ["concat", "concat+num", "OHE+num"],
        rows,
    )
    _plot_group(
        axes[1],
        "Body + Tags (estructura)",
        ["body_tags_concat", "body_tags_ohe_num"],
        ["concat", "tags+OHE+num"],
        rows,
    )
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Estrategias B (modelo={MODEL_NAME}, clf={CLASSIFIER}, C={C_VALUE})")
    fig.tight_layout()
    out_path = FIG_DIR / "embeddings_strategies_B.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[embeddings_plots] Guardado {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    plot_A(rows)
    plot_B(rows)


if __name__ == "__main__":
    main()
