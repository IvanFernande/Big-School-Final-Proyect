"""
Plot LLM zero-shot / few-shot figures from reports/metrics_zero_shot.json.

Outputs:
- reports/figures/fig_g2_llm_precision_recall.png
- reports/figures/fig_g4_llm_estabilidad.png

Run:
  python -m src.plot_llm_figures
"""

import json
import os
from pathlib import Path

mpl_dir = os.getenv("MPLCONFIGDIR")
if not mpl_dir:
    mpl_dir = str(Path("/tmp/mplconfig"))
    os.environ["MPLCONFIGDIR"] = mpl_dir
Path(mpl_dir).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402


FIG_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics_zero_shot.json")


def _load_rows():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {METRICS_PATH}")
    data = json.loads(METRICS_PATH.read_text())
    return [d for d in data if isinstance(d, dict) and "kpis" in d]


def _pick_latest(rows, **criteria):
    filtered = []
    for row in rows:
        ok = True
        for k, v in criteria.items():
            if row.get(k) != v:
                ok = False
                break
        if ok:
            filtered.append(row)
    if not filtered:
        return None
    filtered.sort(key=lambda r: r.get("timestamp", ""))
    return filtered[-1]


def _pick_seed_preferred(rows, seed=42, **criteria):
    preferred = [r for r in rows if r.get("seed") == seed and all(r.get(k) == v for k, v in criteria.items())]
    if preferred:
        preferred.sort(key=lambda r: r.get("timestamp", ""))
        return preferred[-1]
    return _pick_latest(rows, **criteria)


def plot_g2(rows):
    configs = []
    # DeepSeek base 0/3/6
    for shots in (0, 3, 6):
        row = _pick_latest(
            rows,
            provider="ollama",
            prompt_style="base",
            n_examples=shots,
        )
        if row:
            configs.append(("DeepSeek base %s" % shots, row))
    # DeepSeek rules 6
    row = _pick_seed_preferred(rows, seed=42, provider="ollama", prompt_style="rules", n_examples=6)
    if row:
        configs.append(("DeepSeek rules 6", row))
    # Gemini rules 6
    row = _pick_seed_preferred(rows, seed=42, provider="gemini", prompt_style="rules", n_examples=6)
    if row:
        configs.append(("Gemini rules 6", row))

    if not configs:
        print("[llm_plots] No hay datos para G2.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, row in configs:
        k = row["kpis"]
        x = k.get("high_recall")
        y = k.get("high_precision")
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=120, alpha=0.8, label=label)

    ax.set_xlabel("Recall High")
    ax.set_ylabel("Precision High")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path = FIG_DIR / "fig_g2_llm_precision_recall.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[llm_plots] Guardado {out_path}")


def plot_g4(rows):
    filtered = [
        r
        for r in rows
        if r.get("provider") == "ollama"
        and r.get("prompt_style") == "rules"
        and r.get("n_examples") == 6
        and r.get("seed") is not None
    ]
    if not filtered:
        print("[llm_plots] No hay datos para G4.")
        return

    filtered.sort(key=lambda r: r.get("seed"))
    seeds = [r.get("seed") for r in filtered]
    macro = [r["kpis"].get("macro_f1") for r in filtered]
    recall = [r["kpis"].get("high_recall") for r in filtered]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(seeds, macro, marker="o")
    axes[0].set_title("Macro-F1")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(seeds, recall, marker="o", color="#F58518")
    axes[1].set_title("Recall High")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_g4_llm_estabilidad.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[llm_plots] Guardado {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    plot_g2(rows)
    plot_g4(rows)


if __name__ == "__main__":
    main()
