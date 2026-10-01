"""Turn SQuAD 2.0 dev into an open-retrieval benchmark.

SQuAD gives each question its own paragraph. Here every unique paragraph is
pooled into one document store, so the system must first *find* the right
paragraph among ~1,200 before it can answer. The original paragraph becomes
the gold label for retrieval; the original answers become the gold labels for
generation.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

import yaml
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        # Article title adds topical signal ("Normans. The Normans were ...").
        return f"{self.title.replace('_', ' ')}. {self.text}"


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    gold_pid: str
    answers: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return len(self.answers) > 0


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Unicode-normalise (NFKC folds look-alike characters) and lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    """Normalise, split into word tokens, drop English stop words."""
    return [t for t in _TOKEN_RE.findall(normalize_text(text)) if t not in ENGLISH_STOP_WORDS]


def load_squad_open(dataset: str, split: str) -> tuple[list[Passage], list[Query]]:
    ds = load_dataset(dataset, split=split)
    pid_by_context: dict[str, str] = {}
    passages: list[Passage] = []
    queries: list[Query] = []

    for row in ds:
        context = row["context"]
        if context not in pid_by_context:
            pid = f"p{len(passages):05d}"
            pid_by_context[context] = pid
            passages.append(Passage(pid, row["title"], context))
        # Several annotators may give the same answer; keep unique ones in order.
        answers = tuple(dict.fromkeys(a.strip() for a in row["answers"]["text"]))
        queries.append(Query(row["id"], row["question"].strip(), pid_by_context[context], answers))

    return passages, queries


def sample_balanced(queries: list[Query], n_per_class: int, seed: int) -> list[Query]:
    """Fixed-seed sample with equal answerable and unanswerable questions."""
    rng = random.Random(seed)
    answerable = [q for q in queries if q.answerable]
    unanswerable = [q for q in queries if not q.answerable]
    sample = rng.sample(answerable, min(n_per_class, len(answerable)))
    sample += rng.sample(unanswerable, min(n_per_class, len(unanswerable)))
    rng.shuffle(sample)
    return sample


def lexical_overlap(question: str, passage_text: str) -> float:
    """Fraction of the question's content words that also appear in the passage."""
    q = set(tokenize(question))
    return len(q & set(tokenize(passage_text))) / max(len(q), 1)
