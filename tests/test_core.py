import math

import numpy as np
import pytest

from ragbench.data import Passage, lexical_overlap, tokenize
from ragbench.generator import clean_answer, is_refusal
from ragbench.metrics import (exact_match, f1_score, gold_rank, normalize_answer,
                              qa_metrics, retrieval_metrics)
from ragbench.retrievers import BM25Retriever, TfidfRetriever, top_k_indices

PASSAGES = [
    Passage("p0", "Normans", "The Normans were people who gave their name to Normandy in France."),
    Passage("p1", "Computational_complexity", "Complexity theory classifies problems by resources such as time."),
    Passage("p2", "Amazon_rainforest", "The Amazon rainforest covers much of the Amazon basin in South America."),
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The NORMANS were in France") == ["normans", "france"]


def test_lexical_overlap():
    assert lexical_overlap("Where is Normandy?", "Normandy is in France") == 1.0
    assert lexical_overlap("Where is Normandy?", "Nothing relevant") == 0.0


def test_top_k_indices_sorted_and_clipped():
    scores = np.array([0.1, 0.9, 0.5])
    assert top_k_indices(scores, 2).tolist() == [1, 2]
    assert top_k_indices(scores, 10).tolist() == [1, 2, 0]


def test_retrieval_metrics_single_gold():
    ranks = [1, 2, None]  # hit at 1, hit at 2, miss
    m = retrieval_metrics(ranks, ks=[1, 5], depth=10)
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["mrr@10"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 3)


def test_gold_rank():
    assert gold_rank(["a", "b", "c"], "b") == 2
    assert gold_rank(["a"], "z") is None


def test_squad_normalisation_and_scores():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("eiffel tower", ("The Eiffel Tower",)) == 1.0
    assert f1_score("the tower", ("Eiffel Tower",)) == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert exact_match("", ()) == 1.0          # correct refusal on unanswerable
    assert f1_score("Paris", ()) == 0.0        # hallucination on unanswerable


def test_qa_metrics_refusals():
    m = qa_metrics(["France", "", "", "Paris"], [("France",), (), ("Rome",), ()])
    assert m["refusal_accuracy"] == 0.5
    assert m["false_refusal_rate"] == 0.5
    assert m["em"] == 0.5


def test_refusal_parsing():
    assert is_refusal("NO_ANSWER") and is_refusal("no_answer.")
    assert clean_answer("NO_ANSWER") == ""
    assert clean_answer('"France."') == "France"


@pytest.mark.parametrize("cls", [TfidfRetriever, BM25Retriever])
def test_sparse_retrievers_find_obvious_passage(cls):
    retriever = cls().index(PASSAGES)
    hits = retriever.search(["Which rainforest is in South America?", "Who gave their name to Normandy?"], k=2)
    assert hits[0][0][0] == "p2"
    assert hits[1][0][0] == "p0"
    assert len(hits[0]) == 2
