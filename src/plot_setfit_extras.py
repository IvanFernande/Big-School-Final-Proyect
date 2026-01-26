"""
Genera graficas adicionales para SetFit:
1) Top configuraciones por selection score (validación).
2) Comparativa de balanceo (none/oversample/downsample).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.config import FIG_DIR, REPORTS_DIR

# Evita escribir cache en home no accesible.
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / "mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402


RUNS_DIR = Path("results/setfit/runs")
BALANCE_PATH = REPORTS_DIR / "metrics_setfit_balance.json"


def _load_runs():
    rows = []
    for path in RUNS_DIR.glob("run_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fast_mode"):
            continue
        cfg = data.get("config", {})
        val = data.get("val_metrics", {})
        rows.append(
            {
                "run_id": data.get("id"),
                "base_model": cfg.get("base_model"),
                "num_iterations": cfg.get("num_iterations"),
                "num_epochs": cfg.get("num_epochs"),
                "batch_size": cfg.get("batch_size"),
                "learning_rate": cfg.get("learning_rate"),
                "macro_f1": val.get("macro_f1"),
                "high_recall": val.get("high_recall"),
                "selection_score": val.get("selection_score"),
            }
        )
    return rows


def plot_top_configs(top_n: int = 5):
    rows = sorted(_load_runs(), key=lambda r: r.get("selection_score") or 0, reverse=True)[:top_n]
    labels = [
        f"it={r['num_iterations']},ep={r['num_epochs']},b={r['batch_size']},lr={r['learning_rate']}"
        for r in rows
    ]
    scores = [r["selection_score"] for r in rows]
    macro = [r["macro_f1"] for r in rows]
    recall = [r["high_recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(rows))
    ax.bar([i - 0.25 for i in x], scores, width=0.25, label="Selection score")
    ax.bar([i for i in x], macro, width=0.25, label="Macro-F1 (val)")
    ax.bar([i + 0.25 for i in x], recall, width=0.25, label="Recall High (val)")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_title("SetFit: Top configuraciones (validación)")
    ax.set_ylim(0.4, 0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = FIG_DIR / "setfit_top_configs.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[setfit_plots] Guardado {out_path}")


def plot_balance():
    data = json.loads(BALANCE_PATH.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    rows = sorted(rows, key=lambda r: ["none", "oversample", "downsample"].index(r["balance_mode"]))
    labels = [r["balance_mode"] for r in rows]
    macro = [r["val_metrics"]["macro_f1"] for r in rows]
    recall = [r["val_metrics"]["high_recall"] for r in rows]
    score = [r["val_metrics"]["selection_score"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(rows))
    ax.bar([i - 0.25 for i in x], macro, width=0.25, label="Macro-F1 (val)")
    ax.bar([i for i in x], recall, width=0.25, label="Recall High (val)")
    ax.bar([i + 0.25 for i in x], score, width=0.25, label="Selection score")
    ax.set_xticks(list(x), labels)
    ax.set_title("SetFit: Balanceo (validación)")
    ax.set_ylim(0.4, 0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = FIG_DIR / "setfit_balance_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[setfit_plots] Guardado {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_configs()
    plot_balance()


if __name__ == "__main__":
    main()
