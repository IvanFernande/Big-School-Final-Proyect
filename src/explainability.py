"""
Explainability for the final SetFit model using LIME (global + local).

Outputs:
- reports/figures/explain_global_<label>.png
- reports/explainability/local_<idx>_<true>_<pred>.png
- reports/explainability/local_<idx>_<true>_<pred>.html
- reports/explainability_summary.json

Run:
  python -m src.explainability
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from lime.lime_text import LimeTextExplainer
from setfit import SetFitModel

from src.config import FIG_DIR, REPORTS_DIR, SEED
from src.data_prep import add_features, load, split

SETFIT_MODEL_DIR = Path(os.getenv("SETFIT_OUTPUT_DIR", "models/setfit_model"))
SETFIT_DEVICE = os.getenv("SETFIT_DEVICE", "auto")

N_EVAL = int(os.getenv("EXPLAIN_N_EVAL", "120"))
N_LOCAL_PER_CLASS = int(os.getenv("EXPLAIN_N_LOCAL_PER_CLASS", "2"))
NUM_FEATURES_LOCAL = int(os.getenv("EXPLAIN_NUM_FEATURES_LOCAL", "10"))
NUM_FEATURES_GLOBAL = int(os.getenv("EXPLAIN_NUM_FEATURES_GLOBAL", "15"))
EXPLAIN_N_GROUPS = int(os.getenv("EXPLAIN_N_GROUPS", "3"))

EXPLAIN_DIR = REPORTS_DIR / "explainability"


def _select_device():
    try:
        import torch

        if SETFIT_DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return SETFIT_DEVICE
    except Exception:
        return None


def _format_text(df):
    return (df["Body"].astype(str) + " [SEP] " + df["Department"].astype(str)).tolist()


def _load_labels(model, fallback_labels):
    labels = getattr(model, "labels", None)
    if labels:
        return list(labels)
    config_path = Path(SETFIT_MODEL_DIR) / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            id2label = cfg.get("id2label")
            if isinstance(id2label, dict) and id2label:
                ids = sorted(int(k) for k in id2label.keys())
                return [id2label[str(i)] for i in ids]
        except Exception:
            pass
    return list(fallback_labels)


def main():
    np.random.seed(SEED)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(load())
    X_train, X_test, y_train, y_test = split(df)

    model = SetFitModel.from_pretrained(SETFIT_MODEL_DIR)
    device = _select_device()
    if device:
        try:
            model.to(device)
        except Exception:
            pass

    labels = _load_labels(model, sorted(y_train.unique()))
    fallback_labels = sorted(y_train.unique())
    label_map = {label: label for label in labels}
    if all(str(l).isdigit() for l in labels) and len(labels) == len(fallback_labels):
        label_map = {str(i): fallback_labels[i] for i in range(len(fallback_labels))}
    labels_display = [label_map[label] for label in labels]
    label_to_idx = {label: i for i, label in enumerate(labels_display)}

    def predict_proba(texts):
        try:
            probs = model.predict_proba(list(texts))
            return np.asarray(probs)
        except Exception:
            preds = model.predict(list(texts))
            probs = np.zeros((len(texts), len(labels)), dtype=float)
            for i, p in enumerate(preds):
                probs[i, label_to_idx.get(p, 0)] = 1.0
            return probs

    explainer = LimeTextExplainer(class_names=labels_display, random_state=SEED)

    test_texts = _format_text(X_test)
    test_labels = y_test.astype(str).tolist()

    if N_EVAL > len(test_texts):
        raise ValueError("EXPLAIN_N_EVAL exceeds test size.")

    rng = np.random.default_rng(SEED)
    eval_indices = rng.choice(len(test_texts), size=N_EVAL, replace=False)

    # Global explainability: aggregate absolute weights per class.
    instance_explanations = []
    global_weights = {label: Counter() for label in labels_display}
    for idx in eval_indices:
        text = test_texts[idx]
        probs = predict_proba([text])[0]
        pred_label = labels[int(np.argmax(probs))]
        pred_label_display = label_map[pred_label]
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=NUM_FEATURES_GLOBAL,
            top_labels=len(labels),
        )
        for label in labels_display:
            label_idx = label_to_idx[label]
            for token, weight in exp.as_list(label=label_idx):
                global_weights[label][token] += abs(weight)

        pred_items = exp.as_list(label=label_to_idx[label_map[pred_label]])
        instance_explanations.append(
            {
                "idx": int(idx),
                "true": test_labels[idx],
                "pred": pred_label_display,
                "items": pred_items,
            }
        )

    global_top = {}
    for label, counter in global_weights.items():
        top_items = counter.most_common(NUM_FEATURES_GLOBAL)
        global_top[label] = [{"token": t, "weight": float(w)} for t, w in top_items]

        if top_items:
            tokens = [t for t, _ in top_items][::-1]
            weights = [w for _, w in top_items][::-1]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(tokens, weights, color="#4C78A8")
            ax.set_title(f"Global LIME - {label}")
            ax.set_xlabel("avg |weight|")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"explain_global_{label}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    # Grouped explainability: cluster explanations by token weights.
    grouped_summary = []
    if EXPLAIN_N_GROUPS > 1 and len(instance_explanations) >= EXPLAIN_N_GROUPS:
        vocab = sorted({token for row in instance_explanations for token, _ in row["items"]})
        if vocab:
            token_idx = {t: i for i, t in enumerate(vocab)}
            X = np.zeros((len(instance_explanations), len(vocab)), dtype=float)
            for i, row in enumerate(instance_explanations):
                for token, weight in row["items"]:
                    X[i, token_idx[token]] = weight

            try:
                kmeans = KMeans(n_clusters=EXPLAIN_N_GROUPS, random_state=SEED, n_init=10)
                clusters = kmeans.fit_predict(X)
            except Exception:
                clusters = None

            if clusters is not None:
                for cluster_id in range(EXPLAIN_N_GROUPS):
                    mask = clusters == cluster_id
                    if not np.any(mask):
                        continue
                    mean_abs = np.mean(np.abs(X[mask]), axis=0)
                    top_idx = np.argsort(mean_abs)[::-1][:NUM_FEATURES_GLOBAL]
                    top_items = [(vocab[i], float(mean_abs[i])) for i in top_idx if mean_abs[i] > 0]

                    grouped_summary.append(
                        {
                            "cluster": int(cluster_id),
                            "count": int(np.sum(mask)),
                            "top_tokens": [{"token": t, "weight": w} for t, w in top_items],
                        }
                    )

                    if top_items:
                        tokens = [t for t, _ in top_items][::-1]
                        weights = [w for _, w in top_items][::-1]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.barh(tokens, weights, color="#54A24B")
                        ax.set_title(f"Grouped LIME - cluster {cluster_id}")
                        ax.set_xlabel("avg |weight|")
                        fig.tight_layout()
                        fig.savefig(FIG_DIR / f"explain_group_{cluster_id}.png", dpi=200, bbox_inches="tight")
                        plt.close(fig)

    # Local explainability: a few examples per class.
    local_rows = []
    for label in labels_display:
        label_indices = [i for i, y in enumerate(test_labels) if y == label]
        if not label_indices:
            continue
        pick_n = min(N_LOCAL_PER_CLASS, len(label_indices))
        sample_idx = rng.choice(label_indices, size=pick_n, replace=False)

        for idx in sample_idx:
            text = test_texts[idx]
            true_label = test_labels[idx]
            pred_label = labels[int(np.argmax(predict_proba([text])[0]))]
            pred_label = label_map[pred_label]

            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=NUM_FEATURES_LOCAL,
                top_labels=len(labels),
            )
            label_idx = label_to_idx[pred_label]
            items = exp.as_list(label=label_idx)

            safe_pred = pred_label.replace("/", "_")
            safe_true = true_label.replace("/", "_")
            html_path = EXPLAIN_DIR / f"local_{idx}_{safe_true}_{safe_pred}.html"
            exp.save_to_file(str(html_path))

            if items:
                tokens = [t for t, _ in items][::-1]
                weights = [w for _, w in items][::-1]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(tokens, weights, color="#F58518")
                ax.set_title(f"Local LIME - true={true_label} pred={pred_label}")
                ax.set_xlabel("weight")
                fig.tight_layout()
                fig.savefig(EXPLAIN_DIR / f"local_{idx}_{safe_true}_{safe_pred}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

            local_rows.append(
                {
                    "idx": int(idx),
                    "true": true_label,
                    "pred": pred_label,
                    "top_tokens": [{"token": t, "weight": float(w)} for t, w in items],
                }
            )

    summary = {
        "model_dir": str(SETFIT_MODEL_DIR),
        "device": device or "auto",
        "n_eval": int(N_EVAL),
        "n_local_per_class": int(N_LOCAL_PER_CLASS),
        "labels": labels_display,
        "global_top_tokens": global_top,
        "grouped_explainability": grouped_summary,
        "local_examples": local_rows,
    }

    with (REPORTS_DIR / "explainability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[explainability] Global plots saved in reports/figures/")
    print("[explainability] Local plots and HTML in reports/explainability/")
    print("[explainability] Summary in reports/explainability_summary.json")


if __name__ == "__main__":
    main()
