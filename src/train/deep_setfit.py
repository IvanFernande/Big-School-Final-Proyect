"""
SetFit fine-tuning for ticket priority classification.

Aligned with README_SETFIT_TFM.md:
- Stratified split 70/20/10 (train/val/test)
- Multiple base models
- No cross-validation for SetFit
- Selection score on validation: 0.7 * macro_f1 + 0.3 * recall_high
- Retrain on train+val, evaluate once on test
- Artifacts persisted in results/setfit/*
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gc
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import SEED
from src.data_prep import add_features, load, EXPECTED_PRIORITIES

# Avoid tokenizer parallelism memory spikes (can be overridden)
os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))



FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
SAVE_CM = os.getenv("SETFIT_SAVE_CM", "1") == "1"
SAVE_RUN_MODEL = os.getenv("SETFIT_SAVE_RUN_MODEL", "0") == "1"

BASE_MODELS = os.getenv(
    "SETFIT_MODELS",
    ",".join(
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "thenlper/gte-small",
        ]
    ),
)
BASE_MODELS = [m.strip() for m in BASE_MODELS.split(",") if m.strip()]

OUTPUT_ROOT = Path(os.getenv("SETFIT_RESULTS_DIR", "results/setfit"))
MODEL_DIR = Path(os.getenv("SETFIT_OUTPUT_DIR", "models/setfit"))

# Where (if any) checkpoints are written (some libs save to ./checkpoints by default)
CHECKPOINT_DIR = Path(os.getenv("SETFIT_CHECKPOINT_DIR", str(OUTPUT_ROOT / "checkpoints")))


SETFIT_DATA_MAX_ROWS = int(os.getenv("SETFIT_DATA_MAX_ROWS", "0"))  # 0 = use full dataset
FAST_TRAIN_MAX = int(os.getenv("SETFIT_FAST_TRAIN_MAX", "4000"))
FAST_VAL_MAX = int(os.getenv("SETFIT_FAST_VAL_MAX", "1000"))

# Max sequence length for the encoder (reduces RAM/VRAM). Typical safe values: 128 or 256
MAX_SEQ_LEN = int(os.getenv("SETFIT_MAX_SEQ_LEN", "128"))


USE_CUDA = torch.cuda.is_available() and os.getenv("SETFIT_FORCE_CPU", "0") != "1"

if FAST_MODE:
    GRID_NUM_ITER = [5]
    GRID_NUM_EPOCHS = [1]          # fast smoke test
    GRID_BATCH = [16] if USE_CUDA else [8]
    GRID_LR = [2e-5]
else:
    GRID_NUM_ITER = [5, 10]
    GRID_NUM_EPOCHS = [1, 2]       # SetFit often saturates quickly; keeps memory stable
    GRID_BATCH = [16, 32] if USE_CUDA else [8, 16]
    GRID_LR = [2e-5, 5e-5]



@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class RunConfig:
    base_model: str
    num_iterations: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    fast_mode: bool


def _bucket(value: int, bins: List[int]) -> str:
    if value <= bins[0]:
        return f"0-{bins[0]}"
    for lo, hi in zip(bins, bins[1:]):
        if lo < value <= hi:
            return f"{lo+1}-{hi}"
    return f">{bins[-1]}"


def _format_metadata(n_tags: int, len_words: int) -> str:
    tags_bucket = _bucket(int(n_tags), [0, 1, 2, 5, 10])
    len_bucket = _bucket(int(len_words), [0, 20, 50, 100, 200, 500])
    return f"tags={tags_bucket} len={len_bucket}"


def make_texts(X_df: pd.DataFrame) -> List[str]:
    texts = []
    for _, row in X_df.iterrows():
        body = str(row["Body"])
        dept = str(row["Department"])
        meta = _format_metadata(row.get("n_tags", 0), row.get("len_words", 0))
        texts.append(f"{body} [SEP] {dept} [SEP] {meta}")
    return texts


def to_dataset(texts: List[str], labels: List[int]) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})




def _ordered_labels(classes: List[str]) -> List[str]:
    base = ["high", "medium", "low"]
    classes_set = {str(c).strip().lower() for c in classes}
    ordered = [lbl for lbl in base if lbl in classes_set]
    if "high" not in ordered:
        print("[setfit] WARNING: class 'high' not found in labels; check Priority normalization.")
    return ordered if ordered else base


def eval_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "high_recall": float(report.get("high", {}).get("recall", 0.0)),
        "high_precision": float(report.get("high", {}).get("precision", 0.0)),
        "high_f1": float(report.get("high", {}).get("f1-score", 0.0)),
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    metrics["selection_score"] = 0.7 * metrics["macro_f1"] + 0.3 * metrics["high_recall"]
    return metrics


def sample_fast(X_df: pd.DataFrame, y: pd.Series, max_n: int, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    if len(X_df) <= max_n:
        return X_df.reset_index(drop=True), y.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_df), size=max_n, replace=False)
    return X_df.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def stratified_reduce_df(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    """Reduce total rows while preserving class proportions (stratified sampling)."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    counts = df[label_col].value_counts()
    ratios = counts / counts.sum()

    desired = (ratios * max_rows).round().astype(int)

    # ensure >=1 per present class (if possible)
    for cls in counts.index:
        if desired.loc[cls] == 0 and counts.loc[cls] > 0:
            desired.loc[cls] = 1

    diff = int(max_rows - desired.sum())
    if diff != 0:
        if diff > 0:
            remaining = (counts - desired).sort_values(ascending=False)
            for cls in remaining.index:
                if diff == 0:
                    break
                if counts.loc[cls] > desired.loc[cls]:
                    desired.loc[cls] += 1
                    diff -= 1
        else:
            for cls in desired.sort_values(ascending=False).index:
                if diff == 0:
                    break
                if desired.loc[cls] > 1:
                    desired.loc[cls] -= 1
                    diff += 1

    parts = []
    for cls, n in desired.items():
        n = int(min(int(n), int(counts.loc[cls])))
        if n <= 0:
            continue
        part = df[df[label_col] == cls].sample(n=n, replace=False, random_state=seed)
        parts.append(part)

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def run_experiment(
    cfg: RunConfig,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    le: LabelEncoder,
) -> Dict:
    train_ds = to_dataset(train_texts, train_labels)

    model = SetFitModel.from_pretrained(cfg.base_model)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Limit sequence length to reduce memory usage
    try:
        model.model_body.max_seq_length = MAX_SEQ_LEN
    except Exception:
        pass

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        loss_class=CosineSimilarityLoss,
        num_iterations=cfg.num_iterations,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
    )
    with _chdir(CHECKPOINT_DIR):
        trainer.train()

    y_pred_enc = trainer.model.predict(val_texts)
    y_pred = le.inverse_transform(np.asarray(y_pred_enc, dtype=int))
    y_true = le.inverse_transform(np.asarray(val_labels, dtype=int))

    labels = _ordered_labels(list(le.classes_))
    return eval_metrics(y_true.tolist(), y_pred.tolist(), labels)


def main() -> None:
    experiment_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[setfit] FAST_MODE={FAST_MODE} | timestamp={experiment_ts}")
    print(f"[setfit] USE_CUDA={USE_CUDA} | MAX_SEQ_LEN={MAX_SEQ_LEN} | SAVE_RUN_MODEL={SAVE_RUN_MODEL}")
    print(f"[setfit] Models: {BASE_MODELS}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs_dir = OUTPUT_ROOT / "runs"
    best_dir = OUTPUT_ROOT / "best"
    final_dir = OUTPUT_ROOT / "final_test"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(load())
    df["Priority"] = df["Priority"].astype(str).str.lower()
    df = df[df["Priority"].isin(EXPECTED_PRIORITIES)]

    # Optional: reduce dataset size while preserving class proportions (helps RAM/time)
    if SETFIT_DATA_MAX_ROWS and SETFIT_DATA_MAX_ROWS > 0:
        before = df["Priority"].value_counts().to_dict()
        df = stratified_reduce_df(df, "Priority", SETFIT_DATA_MAX_ROWS, SEED)
        after = df["Priority"].value_counts().to_dict()
        print(f"[setfit] Dataset reduced to {len(df)} rows (stratified). Before={before} After={after}")

    X = df[["Body", "Department", "n_tags", "len_words"]].reset_index(drop=True)
    y = df["Priority"].reset_index(drop=True)

    # Split 70/20/10
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=SEED
    )
    val_size = 0.2 / 0.9
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=SEED
    )

    if FAST_MODE:
        X_train, y_train = sample_fast(X_train, y_train, FAST_TRAIN_MAX, SEED)
        X_val, y_val = sample_fast(X_val, y_val, FAST_VAL_MAX, SEED + 1)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_val_enc = le.transform(y_val.astype(str))

    train_texts = make_texts(X_train)
    val_texts = make_texts(X_val)
    test_texts = make_texts(X_test)

    n_train = len(train_texts)
    n_val = len(val_texts)
    n_test = len(test_texts)

    best_cfg = None
    best_metrics = None

    for base_model in BASE_MODELS:
        for num_iterations in GRID_NUM_ITER:
            for num_epochs in GRID_NUM_EPOCHS:
                for batch_size in GRID_BATCH:
                    for learning_rate in GRID_LR:
                        cfg = RunConfig(
                            base_model=base_model,
                            num_iterations=num_iterations,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            fast_mode=FAST_MODE,
                        )
                        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        print(f"[setfit] Run {run_id} | {cfg}")

                        metrics = run_experiment(
                            cfg,
                            train_texts,
                            y_train_enc,
                            val_texts,
                            y_val_enc,
                            le,
                        )

                        run_payload = {
                            "id": run_id,
                            "timestamp": run_id,
                            "experiment_timestamp": experiment_ts,
                            "seed": SEED,
                            "n_train": n_train,
                            "n_val": n_val,
                            "n_test": n_test,
                            "fast_mode": FAST_MODE,
                            "fast_train_max": FAST_TRAIN_MAX if FAST_MODE else None,
                            "fast_val_max": FAST_VAL_MAX if FAST_MODE else None,
                            "config": asdict(cfg),
                            "val_metrics": metrics,
                        }
                        (runs_dir / f"run_{run_id}.json").write_text(
                            json.dumps(run_payload, indent=2), encoding="utf-8"
                        )

                        if best_metrics is None or metrics["selection_score"] > best_metrics["selection_score"]:
                            best_cfg = cfg
                            best_metrics = metrics                        # Free memory between runs (robust: variables may not exist depending on code-path)
                        for _name in ("trainer", "model", "train_ds"):
                            try:
                                del locals()[_name]
                            except Exception:
                                pass
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    if best_cfg is None or best_metrics is None:

        raise RuntimeError("No se pudo seleccionar un modelo (best_cfg vacío).")

    best_payload = {
        "config": asdict(best_cfg),
        "val_metrics": best_metrics,
        "seed": SEED,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "fast_mode": FAST_MODE,
        "fast_train_max": FAST_TRAIN_MAX if FAST_MODE else None,
        "fast_val_max": FAST_VAL_MAX if FAST_MODE else None,
        "experiment_timestamp": experiment_ts,
    }
    (best_dir / "best_config.json").write_text(json.dumps(asdict(best_cfg), indent=2), encoding="utf-8")
    (best_dir / "best_val_metrics.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    # Final train on train + val, evaluate once on test
    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    y_trainval_enc = le.fit_transform(y_trainval.astype(str))

    trainval_texts = make_texts(X_trainval)

    trainval_ds = to_dataset(trainval_texts, y_trainval_enc)
    model = SetFitModel.from_pretrained(best_cfg.base_model)
    try:
        model.model_body.max_seq_length = MAX_SEQ_LEN
    except Exception:
        pass

    trainer = SetFitTrainer(
        model=model,
        train_dataset=trainval_ds,
        loss_class=CosineSimilarityLoss,
        num_iterations=best_cfg.num_iterations,
        num_epochs=best_cfg.num_epochs,
        batch_size=best_cfg.batch_size,
        learning_rate=best_cfg.learning_rate,
    )
    trainer.train()

    y_pred_enc = trainer.model.predict(test_texts)
    y_pred = le.inverse_transform(np.asarray(y_pred_enc, dtype=int))
    y_true = y_test.astype(str).tolist()
    labels = _ordered_labels(list(le.classes_))
    test_metrics = eval_metrics(y_true, y_pred.tolist(), labels)

    (final_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    if SAVE_CM:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("SetFit Confusion Matrix")
        fig.tight_layout()
        fig.savefig(final_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    trainer.model.save_pretrained(MODEL_DIR)

    print(f"[setfit] Best config: {asdict(best_cfg)}")
    print(f"[setfit] Test macro_f1: {test_metrics['macro_f1']:.4f} | high_recall: {test_metrics['high_recall']:.4f}")
    print(f"[setfit] Artifacts in {OUTPUT_ROOT}")
    print(f"[setfit] Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
