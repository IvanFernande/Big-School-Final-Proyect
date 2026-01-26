from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid writing cache in non-writable locations.
os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path("results")
OUT_DIR = Path("visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bar_plot(title: str, labels: List[str], values: List[float], outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, outfile)


def embeddings_plot() -> None:
    data = _load_json(RESULTS_DIR / "benchmark_embeddings.json")
    labels = []
    values = []
    label_map = {
        "sentence-transformers-paraphrase-multilingual-mpnet-base-v2": "mpnet-multi",
        "sentence-transformers-all-mpnet-base-v2": "mpnet-base",
        "ollama-nomic-embed-text": "nomic",
    }
    for run in data.get("runs", []):
        metrics = run.get("metrics_by_k", {})
        if not metrics:
            continue
        best_k = max(metrics, key=lambda k: metrics[k]["avg_score"])
        labels.append(label_map.get(run["name"], run["name"]))
        values.append(float(metrics[best_k]["avg_score"]))
    _bar_plot("Embeddings avg_score (best_k)", labels, values, OUT_DIR / "embeddings_avg_score.png")


def bm25_breakdown_plot() -> None:
    data = _load_json(RESULTS_DIR / "benchmark_bm25.json")
    rows = data.get("rows", [])
    if not rows:
        return
    best_row = max(rows, key=lambda r: r.get("avg_score", 0))
    breakdown = best_row.get("breakdown", {})
    labels = list(breakdown.keys())
    values = [float(breakdown[k].get("avg_score", 0.0)) for k in labels]
    _bar_plot("BM25 avg_score por categoria", labels, values, OUT_DIR / "bm25_breakdown.png")


def llm_plot() -> None:
    data = _load_json(RESULTS_DIR / "benchmark_llm.json")
    labels = []
    values = []
    for run in data.get("runs", []):
        labels.append(run.get("name", "llm"))
        values.append(float(run.get("avg_score", 0.0)))
    _bar_plot("LLM avg_score", labels, values, OUT_DIR / "llm_avg_score.png")


def retrieval_models_plot() -> None:
    data = _load_json(RESULTS_DIR / "benchmark_retrieval.json")
    modes = []
    avg_scores = []
    mrrs = []
    ndcgs = []
    for run in data.get("runs", []):
        mode = run.get("mode")
        metrics = run.get("metrics_by_k", {})
        if not mode or not metrics:
            continue
        best_k = max(metrics, key=lambda k: metrics[k]["avg_score"])
        m = metrics[best_k]
        modes.append(mode)
        avg_scores.append(float(m.get("avg_score", 0.0)))
        mrrs.append(float(m.get("mrr_at_k", 0.0)))
        ndcgs.append(float(m.get("ndcg_at_k", 0.0)))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(modes)))
    width = 0.25
    ax.bar([i - width for i in x], avg_scores, width, label="avg_score", color="#4c78a8")
    ax.bar(x, mrrs, width, label="MRR@k", color="#f58518")
    ax.bar([i + width for i in x], ndcgs, width, label="nDCG@k", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Retrieval metrics por modo (best_k)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    _save_fig(fig, OUT_DIR / "retrieval_models.png")


def retrieval_k_tradeoff_plot() -> None:
    data = _load_json(RESULTS_DIR / "benchmark_retrieval.json")
    modes = ["vector", "bm25", "hybrid"]
    metrics_by_mode: Dict[str, Dict[int, Dict[str, float]]] = {}
    for run in data.get("runs", []):
        mode = run.get("mode")
        if mode not in modes:
            continue
        metrics_by_k = run.get("metrics_by_k", {})
        metrics_by_mode[mode] = {int(k): v for k, v in metrics_by_k.items()}

    if not metrics_by_mode:
        return

    ks = sorted({k for m in metrics_by_mode.values() for k in m.keys()})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metric_map = [
        ("avg_score", "avg_score"),
        ("mrr_at_k", "MRR@k"),
        ("ndcg_at_k", "nDCG@k"),
    ]

    for ax, (key, label) in zip(axes, metric_map):
        for mode in modes:
            series = [metrics_by_mode.get(mode, {}).get(k, {}).get(key, 0.0) for k in ks]
            ax.plot(ks, series, marker="o", label=mode)
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(ks)
    axes[-1].legend(loc="lower right")
    _save_fig(fig, OUT_DIR / "retrieval_k_tradeoff.png")


def main() -> None:
    embeddings_plot()
    retrieval_models_plot()
    retrieval_k_tradeoff_plot()
    llm_plot()
    bm25_breakdown_plot()
    print(f"Graficas generadas en {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
