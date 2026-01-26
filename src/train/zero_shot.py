"""
zero_shot.py

Zero-shot / Few-shot ticket priority classification with LLMs.

Providers:
- Ollama (local): POST {OLLAMA_URL} with {model,prompt,stream=False}
- Gemini (cloud): google-genai Client (supports rotating multiple free API keys)

TFM-oriented features:
- Few-shot examples sampled from TRAIN (support set); evaluation performed on TEST.
- Prompt style configurable to test prompt sensitivity.
- Optional strict JSON output + strict parsing.
- Optional exclusion of invalid outputs from metrics (coverage reported).
- Per-sample logging (raw answer, parse_ok, latency, key index).
- Confusion matrix + baselines (majority, stratified-random).
- Extra KPIs: balanced accuracy, explicit High KPIs, cost-sensitive score, coverage + over-triage.
- Bootstrap confidence intervals for macro-F1 and recall(High).

Usage:
  python -m src.train.zero_shot

Env vars:
  ZERO_SHOT_PROVIDER=ollama|gemini
  GEMINI_API_KEY=...                  # single key
  GEMINI_API_KEYS=key1,key2,...       # multiple keys (rotation)
  GEMINI_RPM=5                        # requests per minute per key (soft, capped at 5)
  GEMINI_RPD=20                       # requests per day per key (soft, local counter)
  GEMINI_COOLDOWN_SEC=60              # cooldown after errors per key
  GEMINI_MAX_RETRIES=2                # retries per request (on transient errors)
  GEMINI_MIN_INTERVAL_SEC=10          # minimum seconds between requests (not counted as latency)
  GEMINI_MODEL=gemini-2.5-flash-lite  # optional

  OLLAMA_MODEL=deepseek-r1:8b
  OLLAMA_URL=http://localhost:11434/api/generate

  ZERO_SHOT_N_EXAMPLES=0|1|3|5|10
  ZERO_SHOT_N_EVAL=100
  ZERO_SHOT_SEED=42
  ZERO_SHOT_STRATIFIED=1
  ZERO_SHOT_PROMPT_STYLE=base|business|rules|strict|strict_json
  ZERO_SHOT_BOOTSTRAP_N=500
  ZERO_SHOT_EXCLUDE_INVALID=1         # if 1, compute metrics only on valid parses; report coverage

Notes:
- Configure GEMINI_API_KEYS or GEMINI_API_KEY in .env when using provider=gemini.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, deque
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import google.genai as genai

from src.data_prep import add_features, load, split

# -----------------
# Configuration
# -----------------
DEFAULT_API_KEY = "PON_AQUI_TU_API_KEY"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

PROVIDER = os.getenv("ZERO_SHOT_PROVIDER", "ollama").strip().lower()
PROMPT_STYLE = os.getenv("ZERO_SHOT_PROMPT_STYLE", "base").strip().lower()

N_EXAMPLES = int(os.getenv("ZERO_SHOT_N_EXAMPLES", "5"))
N_EVAL = int(os.getenv("ZERO_SHOT_N_EVAL", "100"))
RANDOM_SEED = int(os.getenv("ZERO_SHOT_SEED", "42"))
ZERO_SHOT_STRATIFIED = os.getenv("ZERO_SHOT_STRATIFIED", "1") == "1"
BOOTSTRAP_N = int(os.getenv("ZERO_SHOT_BOOTSTRAP_N", "500"))
EXCLUDE_INVALID = os.getenv("ZERO_SHOT_EXCLUDE_INVALID", "1") == "1"

GEMINI_RPM = min(int(os.getenv("GEMINI_RPM", "5")), 5)
GEMINI_RPD = min(int(os.getenv("GEMINI_RPD", "20")), 20)
GEMINI_COOLDOWN_SEC = int(os.getenv("GEMINI_COOLDOWN_SEC", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_MIN_INTERVAL_SEC = max(int(os.getenv("GEMINI_MIN_INTERVAL_SEC", "10")), 0)

LABELS = ["high", "medium", "low"]
LABEL_SET = set(LABELS)

# Cost map (tune later): connect errors to business impact
# NOTE: These costs are a starting point. You can raise FP(high) costs to penalize over-triage.
COST_MAP: Dict[Tuple[str, str], float] = {
    ("high", "medium"): 10.0,
    ("high", "low"): 10.0,
    ("medium", "high"): 3.0,
    ("low", "high"): 3.0,
    ("medium", "low"): 1.0,
    ("low", "medium"): 1.0,
}


class GeminiKeyManager:
    """
    Rotates among multiple Gemini API keys with local (best-effort) limits:
    - RPM per key (sliding 60s window)
    - RPD per key (resets when local date changes)
    Also applies cooldown after errors per key.
    """

    def __init__(self, keys: Sequence[str], *, rpm: int, rpd: int, cooldown_sec: int) -> None:
        if not keys:
            raise ValueError("GeminiKeyManager requires at least one key.")
        self.keys = list(keys)
        self.rpm = max(int(rpm), 1)
        self.rpd = max(int(rpd), 1)
        self.cooldown_sec = max(int(cooldown_sec), 1)

        self._cursor = 0
        self._minute_usage = {idx: deque() for idx in range(len(self.keys))}
        self._daily_usage = {idx: 0 for idx in range(len(self.keys))}
        self._cooldown_until = {idx: 0.0 for idx in range(len(self.keys))}
        self._clients = {idx: genai.Client(api_key=key) for idx, key in enumerate(self.keys)}
        self._day = date.today().isoformat()

    def _reset_if_new_day(self) -> None:
        today = date.today().isoformat()
        if today != self._day:
            self._day = today
            for idx in range(len(self.keys)):
                self._daily_usage[idx] = 0

    def _prune_minute(self, now: float) -> None:
        for dq in self._minute_usage.values():
            while dq and (now - dq[0]) >= 60.0:
                dq.popleft()

    def _has_daily_capacity(self) -> bool:
        self._reset_if_new_day()
        return any(self._daily_usage[idx] < self.rpd for idx in range(len(self.keys)))

    def _pick_available(self, now: float) -> Optional[int]:
        self._reset_if_new_day()
        self._prune_minute(now)

        for _ in range(len(self.keys)):
            idx = self._cursor % len(self.keys)
            self._cursor += 1

            if self._daily_usage[idx] >= self.rpd:
                continue
            if now < self._cooldown_until[idx]:
                continue
            if len(self._minute_usage[idx]) >= self.rpm:
                continue
            return idx
        return None

    def _next_wait(self, now: float) -> Optional[float]:
        self._reset_if_new_day()
        self._prune_minute(now)

        waits: List[float] = []
        for idx in range(len(self.keys)):
            if self._daily_usage[idx] >= self.rpd:
                continue

            dq = self._minute_usage[idx]
            wait_rpm = 0.0
            if len(dq) >= self.rpm:
                wait_rpm = max(0.0, 60.0 - (now - dq[0]))

            wait_cd = max(0.0, self._cooldown_until[idx] - now)
            waits.append(max(wait_rpm, wait_cd))

        if not waits:
            return None
        return min(waits)

    def report_error(self, idx: int, exc: Exception) -> None:
        # cooldown on error (best effort). For obvious rate limits, cool down longer.
        msg = str(exc).lower()
        cooldown = self.cooldown_sec
        if "429" in msg or "rate" in msg or "quota" in msg:
            cooldown = max(cooldown, self.cooldown_sec * 3)
        self._cooldown_until[idx] = max(self._cooldown_until[idx], time.time() + cooldown)

    def acquire(self) -> Tuple[genai.Client, int]:
        if not self._has_daily_capacity():
            raise RuntimeError("Límite diario alcanzado para todas las GEMINI_API_KEYS (según contador local).")

        while True:
            now = time.time()
            idx = self._pick_available(now)
            if idx is not None:
                self._minute_usage[idx].append(now)
                self._daily_usage[idx] += 1
                return self._clients[idx], idx

            wait = self._next_wait(now)
            if wait is None:
                raise RuntimeError("Límite diario alcanzado para todas las GEMINI_API_KEYS (según contador local).")
            if wait > 0:
                print(f"[zero_shot] Rate-limit Gemini: esperando {wait:.1f}s para siguiente clave...")
                time.sleep(wait)


def _normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"high", "h"}:
        return "high"
    if s in {"medium", "med", "m"}:
        return "medium"
    if s in {"low", "l"}:
        return "low"
    return s


def build_prompt(
    examples: Sequence[Tuple[str, str, str]],
    query_body: str,
    query_dept: str,
    *,
    style: str = "base",
) -> str:
    style = (style or "base").strip().lower()

    if style == "business":
        system = (
            "Eres un asistente que clasifica tickets de soporte en: high, medium o low. "
            "Prioriza el impacto en SLA y coste operativo. "
            "High = servicio caido/bloqueo/impacto alto; Medium = degradacion/impacto moderado; "
            "Low = consulta/mejora/impacto bajo."
        )
    elif style == "rules":
        system = (
            "Clasifica tickets en: high, medium o low siguiendo estas reglas:\n"
            "- High: caida de servicio, bloqueo total, seguridad, perdida de datos, urgencia critica.\n"
            "- Medium: degradacion, errores intermitentes, retrasos relevantes, impacto moderado.\n"
            "- Low: preguntas, peticiones menores, mejoras, impacto bajo.\n"
            "Si dudas entre high y medium, elige high."
        )
    else:
        system = "Eres un asistente que clasifica tickets en: high, medium o low."

    lines: List[str] = [system, ""]

    if examples:
        lines.append("Ejemplos:")
        for i, (body, dept, prio) in enumerate(examples, 1):
            lines.append(f"Ejemplo {i}:")
            lines.append(f"Department: {dept}")
            lines.append(f"Body: {body}")
            lines.append(f"Priority: {prio}")
            lines.append("")

    lines.append("Ahora clasifica este ticket:")
    lines.append(f"Department: {query_dept}")
    lines.append(f"Body: {query_body}")

    if style == "strict_json":
        lines.append('Responde SOLO con JSON valido y EXACTO, sin texto extra: {"priority":"high"} (o medium/low).')
        lines.append("Output:")
    elif style == "strict":
        lines.append("Responde SOLO con una palabra: high, medium o low.")
        lines.append("Priority:")
    else:
        lines.append("Priority (responde solo high, medium o low):")

    return "\n".join(lines)


def _stratified_indices(y, n_samples: int, rng: np.random.Generator) -> List[int]:
    labels = sorted(y.unique())
    n_classes = len(labels)
    if n_classes == 0 or n_samples <= 0:
        return []

    base = n_samples // n_classes
    remainder = n_samples % n_classes

    counts = {label: base for label in labels}
    for label in labels[:remainder]:
        counts[label] += 1

    indices: List[int] = []
    for label in labels:
        label_idx = y[y == label].index.to_numpy()
        if len(label_idx) < counts[label]:
            raise ValueError(
                f"No hay suficientes ejemplos para la clase {label} (necesito {counts[label]}, hay {len(label_idx)})."
            )
        chosen = rng.choice(label_idx, size=counts[label], replace=False)
        indices.extend(chosen.tolist())
    return indices


def _predict_one(*, provider: str, client, prompt: str) -> Tuple[str, Optional[int]]:
    provider = (provider or "").strip().lower()

    if provider == "gemini":
        if isinstance(client, GeminiKeyManager):
            last_exc: Optional[Exception] = None
            for attempt in range(max(GEMINI_MAX_RETRIES, 1)):
                gem_client, key_idx = client.acquire()
                try:
                    resp = gem_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                    return (resp.text or ""), key_idx
                except Exception as exc:  # broad to rotate on API errors
                    last_exc = exc
                    client.report_error(key_idx, exc)
                    print(f"[zero_shot] Gemini error (attempt {attempt+1}/{GEMINI_MAX_RETRIES}) key_idx={key_idx}: {exc}")
                    time.sleep(min(2 ** attempt, 8))
            raise RuntimeError(f"Gemini failed after retries: {last_exc}")
        else:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (resp.text or ""), None

    if provider == "ollama":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", ""), None

    raise RuntimeError(f"Proveedor no soportado: {provider}")


def _parse_prediction(answer_raw: str, *, prompt_style: str) -> Tuple[str, bool, str]:
    prompt_style = (prompt_style or "base").strip().lower()
    answer = (answer_raw or "").strip()

    if prompt_style == "strict_json":
        try:
            obj = json.loads(answer)
            pred = _normalize_label(obj.get("priority", ""))
            if pred not in LABEL_SET:
                raise ValueError(f"priority invalida: {pred}")
            return pred, True, ""
        except Exception as e:
            return "medium", False, str(e)

    ans = answer.strip().lower()
    if ans in LABEL_SET:
        return ans, True, ""

    # fallback heuristic
    if "high" in ans:
        return "high", True, ""
    if "medium" in ans:
        return "medium", True, ""
    if "low" in ans:
        return "low", True, ""

    return "medium", False, "No label found"


def cost_per_ticket(y_true: Sequence[str], y_pred: Sequence[str], cost_map: Dict[Tuple[str, str], float]) -> float:
    total = 0.0
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        total += float(cost_map.get((t, p), 1.0))
    return total / max(len(y_true), 1)


def kpis_from_report(rep: dict) -> dict:
    return {
        "accuracy": rep.get("accuracy", None),
        "macro_f1": rep.get("macro avg", {}).get("f1-score", None),
        "weighted_f1": rep.get("weighted avg", {}).get("f1-score", None),
        "high_precision": rep.get("high", {}).get("precision", None),
        "high_recall": rep.get("high", {}).get("recall", None),
        "high_f1": rep.get("high", {}).get("f1-score", None),
    }


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric_fn: Callable[[Sequence[str], Sequence[str]], float],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None}

    idxs = np.arange(n)
    vals = []
    for _ in range(max(n_boot, 1)):
        sample = rng.choice(idxs, size=n, replace=True)
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        vals.append(metric_fn(yt, yp))
    vals = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
    }


def _unpack_split(res):
    """Be tolerant to different split() return orders."""
    if not isinstance(res, (tuple, list)) or len(res) != 4:
        raise RuntimeError("split(df) debe devolver 4 elementos: X_train, X_test, y_train, y_test (o equivalente).")

    a, b, c, d = res

    def is_series(x):
        return hasattr(x, "dtype") and hasattr(x, "values") and (not hasattr(x, "columns"))

    def is_df(x):
        return hasattr(x, "columns")

    candidates = [a, b, c, d]
    dfs = [x for x in candidates if is_df(x)]
    ys = [x for x in candidates if is_series(x)]

    if len(dfs) == 2 and len(ys) == 2:
        if is_df(a) and is_df(b) and is_series(c) and is_series(d):
            return a, b, c, d
        if is_df(a) and is_series(b) and is_df(c) and is_series(d):
            return a, c, b, d

    return a, b, c, d


def main():
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=repo_root / ".env")

    provider = PROVIDER

    # Provider client init
    if provider == "gemini":
        env_keys = os.getenv("GEMINI_API_KEYS", "").strip()
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        else:
            key = os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY)
            keys = [key] if key else []

        if not keys or any("PON_AQUI" in k for k in keys):
            raise RuntimeError("Configura GEMINI_API_KEYS o GEMINI_API_KEY en .env.")

        # best-effort local check to avoid guaranteed failure
        if N_EVAL > len(keys) * GEMINI_RPD:
            raise ValueError(f"N_EVAL={N_EVAL} excede el límite diario total estimado ({len(keys)} * {GEMINI_RPD}).")

        if len(keys) == 1:
            client = genai.Client(api_key=keys[0])
        else:
            client = GeminiKeyManager(keys, rpm=GEMINI_RPM, rpd=GEMINI_RPD, cooldown_sec=GEMINI_COOLDOWN_SEC)
            print(f"[zero_shot] Gemini keys: {len(keys)} | RPM/key={GEMINI_RPM} | RPD/key={GEMINI_RPD} | cooldown={GEMINI_COOLDOWN_SEC}s | retries={GEMINI_MAX_RETRIES}")
    else:
        client = None

    # Data
    df = add_features(load())
    X_train, X_test, y_train, y_test = _unpack_split(split(df))

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True).map(_normalize_label)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True).map(_normalize_label)

    rng = np.random.default_rng(RANDOM_SEED)

    if N_EVAL > len(X_test):
        raise ValueError("N_EVAL excede el tamaño del test.")

    if N_EXAMPLES > len(X_train):
        raise ValueError("N_EXAMPLES excede el tamaño del train.")

    # Few-shot examples from TRAIN
    if N_EXAMPLES > 0:
        if ZERO_SHOT_STRATIFIED:
            ex_idx = _stratified_indices(y_train, N_EXAMPLES, rng)
        else:
            ex_idx = rng.choice(len(X_train), size=N_EXAMPLES, replace=False).tolist()

        examples = list(
            zip(
                X_train.iloc[ex_idx]["Body"].astype(str),
                X_train.iloc[ex_idx]["Department"].astype(str),
                y_train.iloc[ex_idx].astype(str),
            )
        )
        if ZERO_SHOT_STRATIFIED:
            counts = y_train.iloc[ex_idx].value_counts().to_dict()
            print(f"[zero_shot] Examples (TRAIN) class counts: {counts}")
    else:
        examples = []

    # Eval indices from TEST
    if ZERO_SHOT_STRATIFIED:
        eval_idx = _stratified_indices(y_test, N_EVAL, rng)
    else:
        eval_idx = rng.choice(len(X_test), size=N_EVAL, replace=False).tolist()

    print(f"[zero_shot] Provider={provider} Model={OLLAMA_MODEL if provider=='ollama' else GEMINI_MODEL}")
    print(f"[zero_shot] PromptStyle={PROMPT_STYLE} Examples={N_EXAMPLES} Eval={N_EVAL} Seed={RANDOM_SEED} Stratified={ZERO_SHOT_STRATIFIED} ExcludeInvalid={EXCLUDE_INVALID}")

    rows: List[dict] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    parse_ok_all: List[int] = []
    latencies: List[float] = []

    # for metrics on valid only
    y_true_valid: List[str] = []
    y_pred_valid: List[str] = []

    total = len(eval_idx)
    for i, idx in enumerate(eval_idx, 1):
        query_row = X_test.iloc[idx]
        query_body = str(query_row["Body"])
        query_dept = str(query_row["Department"])
        true_label = str(y_test.iloc[idx])

        prompt = build_prompt(examples, query_body, query_dept, style=PROMPT_STYLE)

        t0 = time.perf_counter()
        answer_raw, key_idx = _predict_one(provider=provider, client=client, prompt=prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred, parse_ok, parse_error = _parse_prediction(answer_raw, prompt_style=PROMPT_STYLE)

        y_true_all.append(true_label)
        y_pred_all.append(pred)
        parse_ok_all.append(int(bool(parse_ok)))
        latencies.append(float(latency_ms))

        if (not EXCLUDE_INVALID) or parse_ok:
            y_true_valid.append(true_label)
            y_pred_valid.append(pred)

        rows.append(
            {
                "idx": int(idx),
                "true": true_label,
                "pred": pred,
                "provider": provider,
                "model": OLLAMA_MODEL if provider == "ollama" else GEMINI_MODEL,
                "api_key_idx": key_idx,
                "prompt_style": PROMPT_STYLE,
                "n_examples": int(N_EXAMPLES),
                "latency_ms": round(float(latency_ms), 2),
                "parse_ok": int(bool(parse_ok)),
                "parse_error": (parse_error or "")[:200],
                "answer_raw": (answer_raw or "")[:500],
            }
        )

        if provider == "gemini" and GEMINI_MIN_INTERVAL_SEC > 0 and i < total:
            time.sleep(GEMINI_MIN_INTERVAL_SEC)

        if i % 10 == 0 or i == total:
            print(f"[zero_shot] Progreso: {i}/{total}")

    # Coverage / robustness
    parse_ok_rate = float(sum(parse_ok_all) / max(len(parse_ok_all), 1))
    invalid_rate = 1.0 - parse_ok_rate
    coverage = parse_ok_rate if EXCLUDE_INVALID else 1.0  # if not excluding invalid, coverage isn't used for metrics

    # Metrics subset selection
    yt = y_true_valid if EXCLUDE_INVALID else y_true_all
    yp = y_pred_valid if EXCLUDE_INVALID else y_pred_all

    report = classification_report(yt, yp, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=LABELS)

    # Headline KPIs
    kpis = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(yt) else None,
        "high_precision": float(report.get("high", {}).get("precision", 0.0)),
        "high_recall": float(report.get("high", {}).get("recall", 0.0)),
        "high_f1": float(report.get("high", {}).get("f1-score", 0.0)),
        "avg_cost": float(cost_per_ticket(yt, yp, COST_MAP)) if len(yt) else None,
        "invalid_rate": float(invalid_rate),
        "coverage": float(coverage),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
    }

    # Over-triage indicators (business-ish)
    if len(yt):
        true_high_rate = float(sum(1 for t in yt if t == "high") / len(yt))
        pred_high_rate = float(sum(1 for p in yp if p == "high") / len(yt))
        kpis["true_high_rate"] = true_high_rate
        kpis["pred_high_rate"] = pred_high_rate
        kpis["overtriage_factor"] = float(pred_high_rate / true_high_rate) if true_high_rate > 0 else None

    # Baselines on SAME subset (valid-only if EXCLUDE_INVALID)
    counts = Counter(yt)
    majority = max(counts, key=counts.get) if counts else "medium"
    majority_pred = [majority] * len(yt)

    probs = [counts.get(lbl, 0) / max(len(yt), 1) for lbl in LABELS] if len(yt) else [1/3, 1/3, 1/3]
    rng2 = np.random.default_rng(RANDOM_SEED + 1337)
    random_pred = rng2.choice(LABELS, size=len(yt), p=probs).tolist()

    baseline_majority_rep = classification_report(yt, majority_pred, labels=LABELS, output_dict=True, zero_division=0)
    baseline_random_rep = classification_report(yt, random_pred, labels=LABELS, output_dict=True, zero_division=0)

    baselines = {
        "majority": {
            "classification_report": baseline_majority_rep,
            "kpis": {
                **kpis_from_report(baseline_majority_rep),
                "balanced_accuracy": float(balanced_accuracy_score(yt, majority_pred)) if len(yt) else None,
                "avg_cost": float(cost_per_ticket(yt, majority_pred, COST_MAP)) if len(yt) else None,
                "pred_high_rate": float(sum(1 for p in majority_pred if p == "high") / len(yt)) if len(yt) else None,
            },
        },
        "random_stratified": {
            "classification_report": baseline_random_rep,
            "kpis": {
                **kpis_from_report(baseline_random_rep),
                "balanced_accuracy": float(balanced_accuracy_score(yt, random_pred)) if len(yt) else None,
                "avg_cost": float(cost_per_ticket(yt, random_pred, COST_MAP)) if len(yt) else None,
                "pred_high_rate": float(sum(1 for p in random_pred if p == "high") / len(yt)) if len(yt) else None,
            },
        },
    }

    # Bootstrap CI (macro-F1, recall(High)) on metrics subset
    def macro_f1_fn(yt2, yp2) -> float:
        _, _, f1, _ = precision_recall_fscore_support(yt2, yp2, labels=LABELS, average="macro", zero_division=0)
        return float(f1)

    def high_recall_fn(yt2, yp2) -> float:
        rep2 = classification_report(yt2, yp2, labels=LABELS, output_dict=True, zero_division=0)
        return float(rep2.get("high", {}).get("recall", 0.0))

    bootstrap = {
        "macro_f1": bootstrap_ci(yt, yp, macro_f1_fn, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED),
        "high_recall": bootstrap_ci(yt, yp, high_recall_fn, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED),
    }

    # Persist
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / "metrics_zero_shot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": ts,
        "provider": provider,
        "model": OLLAMA_MODEL if provider == "ollama" else GEMINI_MODEL,
        "prompt_style": PROMPT_STYLE,
        "n_examples": int(N_EXAMPLES),
        "n_eval": int(N_EVAL),
        "seed": int(RANDOM_SEED),
        "bootstrap_n": int(BOOTSTRAP_N),
        "exclude_invalid": bool(EXCLUDE_INVALID),
        "n_valid": int(len(yt)),
        "kpis": kpis,
        "bootstrap_ci": bootstrap,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "classification_report": report,
        "baselines": baselines,
        "cost_map": {f"{a}->{b}": float(v) for (a, b), v in COST_MAP.items()},
    }

    if report_path.exists():
        try:
            history = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except Exception:
            history = []
    else:
        history = []

    history.append(entry)
    report_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    preds_path = Path("reports") / f"zero_shot_predictions_{ts}.csv"
    fieldnames = [
        "idx",
        "true",
        "pred",
        "provider",
        "model",
        "api_key_idx",
        "prompt_style",
        "n_examples",
        "latency_ms",
        "parse_ok",
        "parse_error",
        "answer_raw",
    ]
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[zero_shot] Métricas guardadas en {report_path}")
    print(f"[zero_shot] Predicciones guardadas en {preds_path}")
    print("\n[zero_shot] KPI resumen:")
    for k, v in kpis.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
