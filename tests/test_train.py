from ragbench.train import build_training_pairs


def row(context, question, answers, title="Some_Title"):
    return {"context": context, "question": question, "title": title, "answers": {"text": answers}}


def test_one_pair_per_paragraph_answerable_only():
    rows = [
        row("Paris is in France.", "Where is Paris?", ["France"]),
        row("Paris is in France.", "What is in France?", ["Paris"]),
        row("Rome is in Italy.", "Where is Rome?", ["Italy"]),
        row("Oslo is in Norway.", "Who founded Oslo?", []),  # unanswerable: excluded
    ]
    pairs = build_training_pairs(rows, max_pairs=10, seed=0, query_prefix="Q: ")
    passages = [p for _, p in pairs]
    assert len(pairs) == 2
    assert len(set(passages)) == 2
    assert all(q.startswith("Q: ") for q, _ in pairs)
    assert "Some Title. Rome is in Italy." in passages
    assert not any("Oslo" in p for p in passages)


def test_max_pairs_and_determinism():
    rows = [row(f"Paragraph {i}.", f"Question {i}?", ["x"]) for i in range(50)]
    assert len(build_training_pairs(rows, max_pairs=10, seed=1)) == 10
    assert build_training_pairs(rows, 10, seed=1) == build_training_pairs(rows, 10, seed=1)
