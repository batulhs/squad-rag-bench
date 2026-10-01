"""Four retrievers behind one interface: index(passages) then search(questions, k).

1. TF-IDF + cosine   sparse, exact-word matching, classic weighting
2. BM25              sparse, adds term-frequency saturation and length normalisation
3. Dense             bi-encoder embeddings, matches meaning rather than exact words
   Dense (fine-tuned) the same encoder after contrastive training on SQuAD train
4. Hybrid            BM25 + dense fused with Reciprocal Rank Fusion, then a
                     cross-encoder reranks the fused candidates
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Passage, tokenize

Hit = tuple[str, float]  # (passage id, score)


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, highest first, without a full sort."""
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class Retriever:
    name = "base"

    def index(self, passages: list[Passage]) -> "Retriever":
        start = time.perf_counter()
        self.passages = passages
        self.pids = [p.pid for p in passages]
        self._build(passages)
        self.index_seconds = time.perf_counter() - start
        return self

    def _build(self, passages: list[Passage]) -> None:
        raise NotImplementedError

    def search(self, questions: list[str], k: int) -> list[list[Hit]]:
        raise NotImplementedError

    def _rank(self, scores: np.ndarray, k: int) -> list[Hit]:
        return [(self.pids[i], float(scores[i])) for i in top_k_indices(scores, k)]


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=sublinear_tf
        )

    def _build(self, passages):
        # Rows are L2-normalised by default, so a dot product is cosine similarity.
        self.doc_matrix = self.vectorizer.fit_transform([p.full_text for p in passages])

    def search(self, questions, k):
        scores = (self.vectorizer.transform(questions) @ self.doc_matrix.T).toarray()
        return [self._rank(row, k) for row in scores]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def _build(self, passages):
        self.bm25 = BM25Okapi([tokenize(p.full_text) for p in passages], k1=self.k1, b=self.b)

    def search(self, questions, k):
        return [self._rank(np.asarray(self.bm25.get_scores(tokenize(q))), k) for q in questions]


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model: str, query_prefix: str = "", batch_size: int = 128, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model, device=device or default_device())
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def _build(self, passages):
        self.doc_emb = self._encode([p.full_text for p in passages])

    def search(self, questions, k):
        q_emb = self._encode([self.query_prefix + q for q in questions])
        return [self._rank(row, k) for row in q_emb @ self.doc_emb.T]


class HybridRerankRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self, sparse: BM25Retriever, dense: DenseRetriever, reranker: str,
        candidates: int = 50, rrf_k: int = 60, batch_size: int = 256, device: str | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.sparse, self.dense = sparse, dense
        self.reranker = CrossEncoder(reranker, device=device or default_device())
        self.candidates, self.rrf_k, self.batch_size = candidates, rrf_k, batch_size

    def _build(self, passages):
        for component in (self.sparse, self.dense):
            if not hasattr(component, "pids"):
                component.index(passages)
        self.text_by_pid = {p.pid: p.full_text for p in passages}

    def _fuse(self, sparse_hits: list[Hit], dense_hits: list[Hit]) -> list[str]:
        """Reciprocal Rank Fusion: score = sum over lists of 1 / (rrf_k + rank).
        Uses ranks, not raw scores, so BM25 and cosine scales never need calibrating."""
        fused: dict[str, float] = defaultdict(float)
        for hits in (sparse_hits, dense_hits):
            for rank, (pid, _) in enumerate(hits, start=1):
                fused[pid] += 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=fused.get, reverse=True)[: self.candidates]

    def search(self, questions, k):
        sparse = self.sparse.search(questions, self.candidates)
        dense = self.dense.search(questions, self.candidates)
        candidate_lists = [self._fuse(s, d) for s, d in zip(sparse, dense)]

        # Score every (question, candidate) pair in one batched call.
        pairs = [(q, self.text_by_pid[pid]) for q, cands in zip(questions, candidate_lists) for pid in cands]
        scores = np.asarray(self.reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))

        results, offset = [], 0
        for cands in candidate_lists:
            chunk = scores[offset: offset + len(cands)]
            offset += len(cands)
            results.append([(cands[i], float(chunk[i])) for i in top_k_indices(chunk, k)])
        return results


ALL_RETRIEVERS = ("tfidf", "bm25", "dense", "dense_ft", "hybrid")


def build_retrievers(cfg: dict, passages: list[Passage], names=ALL_RETRIEVERS) -> dict[str, Retriever]:
    """Build and index the requested retrievers. BM25 and (pretrained) dense are shared with hybrid."""
    rc = cfg["retrieval"]
    built: dict[str, Retriever] = {}
    if "tfidf" in names:
        built["tfidf"] = TfidfRetriever(**rc["tfidf"]).index(passages)
    if "bm25" in names or "hybrid" in names:
        built["bm25"] = BM25Retriever(**rc["bm25"]).index(passages)
    if "dense" in names or "hybrid" in names:
        built["dense"] = DenseRetriever(**rc["dense"]).index(passages)
    if "dense_ft" in names:
        if not Path(rc["dense_ft"]["model"]).exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {rc['dense_ft']['model']}. Run `python -m ragbench.train` first, "
                "or leave dense_ft out with --retrievers."
            )
        built["dense_ft"] = DenseRetriever(**rc["dense_ft"]).index(passages)
    if "hybrid" in names:
        hybrid = HybridRerankRetriever(built["bm25"], built["dense"], **rc["hybrid"]).index(passages)
        hybrid.index_seconds += built["bm25"].index_seconds + built["dense"].index_seconds
        built["hybrid"] = hybrid
    return {n: built[n] for n in names}
