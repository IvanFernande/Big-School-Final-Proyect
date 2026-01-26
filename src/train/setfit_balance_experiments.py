"""
Experimentos de balanceo para SetFit (none/oversample/downsample).
Genera métricas comparables usando el mismo protocolo 70/20/10.

Uso:
  python -m src.train.setfit_balance_experiments
  SETFIT_BALANCE_FAST=1 python -m src.train.setfit_balance_experiments
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import REPORTS_DIR, SEED
from src.data_prep import add_features, load, EXPECTED_PRIORITIES

os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))

FAST_MODE = os.getenv("SETFIT_BALANCE_FAST", "0") == "1"
FAST_TRAIN_MAX = int(os.getenv("SETFIT_FAST_TRAIN_MAX", "4000"))
FAST_VAL_MAX = int(os.getenv("SETFIT_FAST_VAL_MAX", "1000"))
MAX_SEQ_LEN = int(os.getenv("SETFIT_MAX_SEQ_LEN", "128"))

BALANCE_MODES = ["none", "oversample", "downsample"]

BEST_CONFIG_PATH = Path(os.getenv("SETFIT_BALANCE_CONFIG", "results/setfit/best/best_config.json"))


@dataclass
class BalanceConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_iterations: int = 10
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 5e-5


def _load_best_config() -> BalanceConfig:
    if BEST_CONFIG_PATH.exists():
        raw = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
        return BalanceConfig(
            base_model=raw.get("base_model", "sentence-transformers/all-MiniLM-L6-v2"),
            num_iterations=int(raw.get("num_iterations", 10)),
            num_epochs=int(raw.get("num_epochs", 2)),
            batch_size=int(raw.get("batch_size", 32)),
            learning_rate=float(raw.get("learning_rate", 5e-5)),
        )
    return BalanceConfig()


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


def balance_df(df: pd.DataFrame, label_col: str, mode: str, seed: int) -> pd.DataFrame:
    if mode == "none":
        return df
    counts = df[label_col].value_counts()
    if mode == "oversample":
        target = counts.max()
        parts = [
            df[df[label_col] == cls].sample(n=target, replace=True, random_state=seed)
            for cls in counts.index
        ]
    elif mode == "downsample":
        target = counts.min()
        parts = [
            df[df[label_col] == cls].sample(n=target, replace=False, random_state=seed)
            for cls in counts.index
        ]
    else:
        raise ValueError(f"Balance mode not supported: {mode}")
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


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


def _sample_fast(X_df: pd.DataFrame, y: pd.Series, max_n: int, seed: int):
    if len(X_df) <= max_n:
        return X_df.reset_index(drop=True), y.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_df), size=max_n, replace=False)
    return X_df.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def run_balance_mode(cfg: BalanceConfig, X_train, y_train, X_val, y_val, le, mode: str) -> Dict:
    train_df = X_train.copy()
    train_df["Priority"] = y_train.values
    train_df = balance_df(train_df, "Priority", mode, SEED)

    y_train_bal = train_df["Priority"]
    X_train_bal = train_df.drop(columns=["Priority"])

    if FAST_MODE:
        X_train_bal, y_train_bal = _sample_fast(X_train_bal, y_train_bal, FAST_TRAIN_MAX, SEED)

    train_texts = make_texts(X_train_bal)
    val_texts = make_texts(X_val)

    y_train_enc = le.fit_transform(y_train_bal.astype(str))
    y_val_enc = le.transform(y_val.astype(str))

    train_ds = to_dataset(train_texts, y_train_enc)

    model = SetFitModel.from_pretrained(cfg.base_model)
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
    trainer.train()

    y_pred_enc = trainer.model.predict(val_texts)
    y_pred = le.inverse_transform(np.asarray(y_pred_enc, dtype=int))
    y_true = y_val.astype(str).tolist()

    labels = ["high", "medium", "low"]
    metrics = eval_metrics(y_true, y_pred.tolist(), labels)

    # Free memory
    for _name in ("trainer", "model", "train_ds"):
        try:
            del locals()[_name]
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "balance_mode": mode,
        "train_size": len(X_train_bal),
        "train_distribution": y_train_bal.value_counts().to_dict(),
        "val_size": len(X_val),
        "val_distribution": y_val.value_counts().to_dict(),
        "config": asdict(cfg),
        "val_metrics": metrics,
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = _load_best_config()
    print(f"[setfit-balance] Using config: {cfg}")

    df = add_features(load())
    df["Priority"] = df["Priority"].astype(str).str.lower()
    df = df[df["Priority"].isin(EXPECTED_PRIORITIES)]

    X = df[["Body", "Department", "n_tags", "len_words"]].reset_index(drop=True)
    y = df["Priority"].reset_index(drop=True)

    # 70/20/10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=SEED
    )
    val_size = 0.2 / 0.9
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=SEED
    )

    if FAST_MODE:
        X_val, y_val = _sample_fast(X_val, y_val, FAST_VAL_MAX, SEED + 1)

    le = LabelEncoder()

    results = []
    for mode in BALANCE_MODES:
        print(f"[setfit-balance] Running mode={mode}")
        payload = run_balance_mode(cfg, X_train, y_train, X_val, y_val, le, mode)
        results.append(payload)

    out_path = REPORTS_DIR / "metrics_setfit_balance.json"
    out_path.write_text(json.dumps({"timestamp": ts, "results": results}, indent=2), encoding="utf-8")
    print(f"[setfit-balance] Saved {out_path}")


if __name__ == "__main__":
    main()
