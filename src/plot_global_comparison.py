"""
Generate global comparison figure (Macro-F1 vs Recall High) across approaches.

Outputs:
- reports/figures/fig_g1_comparativa_global.png

Data sources:
- TF-IDF: reports/metrics_test.json
- Embeddings: reports/experiments/embeddings_summary.csv (config below)
- SetFit: results/setfit/final_test/test_metrics.json
- LLM: reports/metrics_zero_shot.json (DeepSeek rules 6, seed=42 if available)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"

# Avoid writing matplotlib cache to home.
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / "mplconfig"))

# Embeddings config used for comparison (matches README).
EMB_MODEL = "sentence-transformers/all-distilroberta-v1"
EMB_CLASSIFIER = "linear_svm"
EMB_STRATEGY = "body_tags_ohe_num"
EMB_C = "0.5"


def _load_tfidf_metrics():
    p = REPORTS_DIR / "metrics_test.json"
    data = json.loads(p.read_text())
    return {
        "macro_f1": float(data["macro avg"]["f1-score"]),
        "recall_high": float(data["high"]["recall"]),
    }


def _load_embeddings_metrics():
    p = REPORTS_DIR / "experiments" / "embeddings_summary.csv"
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (
            r.get("embedding_model") == EMB_MODEL
            and r.get("classifier") == EMB_CLASSIFIER
            and r.get("feature_strategy") == EMB_STRATEGY
            and r.get("C") == EMB_C
        ):
            return {
                "macro_f1": float(r["test_f1_macro"]),
                "recall_high": float(r["test_recall_high"]),
            }
    # Fallback: best macro-F1 for this encoder if exact row missing.
    candidates = [r for r in rows if r.get("embedding_model") == EMB_MODEL]
    if not candidates:
        raise FileNotFoundError("No embeddings rows found for comparison.")
    best = max(candidates, key=lambda r: float(r.get("test_f1_macro", "0")))
    return {
        "macro_f1": float(best["test_f1_macro"]),
        "recall_high": float(best["test_recall_high"]),
    }


def _load_setfit_metrics():
    p = Path("results/setfit/final_test/test_metrics.json")
    data = json.loads(p.read_text())
    return {
        "macro_f1": float(data["macro_f1"]),
        "recall_high": float(data["high_recall"]),
    }


def _load_llm_metrics():
    p = REPORTS_DIR / "metrics_zero_shot.json"
    rows = json.loads(p.read_text())
    # Prefer DeepSeek rules 6 with seed=42 (matches README tables).
    filtered = [
        r
        for r in rows
        if r.get("provider") == "ollama"
        and r.get("prompt_style") == "rules"
        and r.get("n_examples") == 6
        and r.get("seed") == 42
    ]
    if not filtered:
        # Fallback: latest DeepSeek rules 6 if seed 42 not present.
        filtered = [
            r
            for r in rows
            if r.get("provider") == "ollama"
            and r.get("prompt_style") == "rules"
            and r.get("n_examples") == 6
        ]
        filtered.sort(key=lambda r: r.get("timestamp", ""))
    if not filtered:
        raise FileNotFoundError("No LLM metrics found for DeepSeek rules 6.")
    r = filtered[-1]
    k = r.get("kpis", {})
    return {"macro_f1": float(k.get("macro_f1", 0.0)), "recall_high": float(k.get("high_recall", 0.0))}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "TF-IDF": _load_tfidf_metrics(),
        "Embeddings": _load_embeddings_metrics(),
        "SetFit": _load_setfit_metrics(),
        "Zero/Few-shot": _load_llm_metrics(),
    }

    labels = list(data.keys())
    macro = [data[k]["macro_f1"] for k in labels]
    recall = [data[k]["recall_high"] for k in labels]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], macro, width, label="Macro-F1")
    ax.bar([i + width / 2 for i in x], recall, width, label="Recall High")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Comparativa global por enfoques")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = FIG_DIR / "fig_g1_comparativa_global.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[global_comparison] Guardado {out_path}")


if __name__ == "__main__":
    main()
