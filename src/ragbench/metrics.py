"""Retrieval, answer-quality and latency metrics.

Retrieval (one gold passage per question):
  Recall@k  1 if the gold passage is in the top k
  MRR@d     1 / rank of the gold passage, 0 if not in the top d
  nDCG@d    1 / log2(rank + 1); with a single relevant passage the ideal DCG is 1

Answers (official SQuAD 2.0 definitions):
  EM        normalised prediction equals any gold answer
  F1        token-overlap F1 against the best-matching gold answer
  For unanswerable questions the gold answer is "", so a refusal scores 1.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def gold_rank(ranked_pids: list[str], gold_pid: str) -> int | None:
    return ranked_pids.index(gold_pid) + 1 if gold_pid in ranked_pids else None


def retrieval_metrics(ranks: list[int | None], ks: list[int], depth: int) -> dict[str, float]:
    within = [r if r is not None and r <= depth else None for r in ranks]
    out = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in within])) for k in ks}
    out[f"mrr@{depth}"] = float(np.mean([1 / r if r else 0.0 for r in within]))
    out[f"ndcg@{depth}"] = float(np.mean([1 / math.log2(r + 1) if r else 0.0 for r in within]))
    return out


def normalize_answer(text: str) -> str:
    """Official SQuAD normalisation: lowercase, drop punctuation and articles, squash spaces."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, golds: tuple[str, ...]) -> float:
    golds = golds or ("",)
    return float(max(normalize_answer(prediction) == normalize_answer(g) for g in golds))


def _f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, golds: tuple[str, ...]) -> float:
    return max(_f1(prediction, g) for g in (golds or ("",)))


def qa_metrics(predictions: list[str], golds: list[tuple[str, ...]]) -> dict[str, float]:
    """predictions are cleaned answers where "" means the model refused."""
    em = [exact_match(p, g) for p, g in zip(predictions, golds)]
    f1 = [f1_score(p, g) for p, g in zip(predictions, golds)]
    has_ans = [bool(g) for g in golds]
    refused = [p == "" for p in predictions]

    def mean_where(values, mask):
        picked = [v for v, m in zip(values, mask) if m]
        return float(np.mean(picked)) if picked else float("nan")

    no_ans = [not h for h in has_ans]
    return {
        "em": float(np.mean(em)),
        "f1": float(np.mean(f1)),
        "em_answerable": mean_where(em, has_ans),
        "f1_answerable": mean_where(f1, has_ans),
        "refusal_accuracy": mean_where(refused, no_ans),     # refused when it should
        "false_refusal_rate": mean_where(refused, has_ans),  # refused when it shouldn't
    }


def latency_summary(ms: list[float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_p50_ms": float(np.percentile(ms, 50)), f"{prefix}_p95_ms": float(np.percentile(ms, 95))}
