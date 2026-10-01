"""Error analysis on saved results: where does each retriever win or fail?

  python -m ragbench.analyze

Writes:
  breakdown_by_question_type.csv   Recall@5 per retriever for what/who/when/...
  breakdown_by_overlap.csv         Recall@5 per retriever by question-passage word overlap
  demo_cases.json                  one question per scenario, used by ragbench.demo
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .data import load_config

QUESTION_WORDS = ("what", "who", "when", "where", "which", "why", "how")


def question_type(question: str) -> str:
    words = question.lower().split()
    return next((w for w in words if w in QUESTION_WORDS), "other")


def overlap_bucket(overlap: float) -> str:
    if overlap < 0.34:
        return "low (<34%)"
    if overlap < 0.67:
        return "medium (34-67%)"
    return "high (>=67%)"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def recall_table(df: pd.DataFrame, names: list[str], group: str, k: int) -> pd.DataFrame:
    hit = {n: df[n].apply(lambda r: r is not None and r == r and r <= k) for n in names}
    table = pd.DataFrame({group: df[group], **hit}).groupby(group).mean()
    table.insert(0, "n", df.groupby(group).size())
    return table.round(3)


def pick(rng: random.Random, candidates: list[dict]) -> dict | None:
    return rng.choice(candidates) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["paths"]["results"])
    rows = read_jsonl(out / "retrieval_per_query.jsonl")
    names = list(rows[0]["ranks"])

    df = pd.DataFrame([{**{n: r["ranks"][n] for n in names}, "qid": r["qid"],
                        "qtype": question_type(r["question"]),
                        "overlap": overlap_bucket(r["overlap"])} for r in rows])
    by_type = recall_table(df, names, "qtype", args.k)
    by_overlap = recall_table(df, names, "overlap", args.k)
    by_type.to_csv(out / "breakdown_by_question_type.csv")
    by_overlap.to_csv(out / "breakdown_by_overlap.csv")
    print(f"Recall@{args.k} by question type\n{by_type}\n")
    print(f"Recall@{args.k} by question-passage word overlap\n{by_overlap}\n")

    # Demo cases: one real example of each scenario worth showing live.
    rng = random.Random(cfg["seed"])
    top1 = lambda r, n: r["ranks"].get(n) == 1
    miss = lambda r, n: n in r["ranks"] and (r["ranks"][n] is None or r["ranks"][n] > args.k)
    scenarios = {
        "all_retrievers_correct": [r for r in rows if all(top1(r, n) for n in names)],
        "only_bm25_correct": [r for r in rows if top1(r, "bm25") and miss(r, "dense")],
        "only_dense_correct": [r for r in rows if top1(r, "dense") and miss(r, "bm25")],
        "only_finetuned_correct": [r for r in rows if top1(r, "dense_ft") and miss(r, "dense")],
        "hybrid_rescue": [r for r in rows if top1(r, "hybrid") and not top1(r, "bm25") and not top1(r, "dense")],
    }
    cases = []
    for label, candidates in scenarios.items():
        chosen = pick(rng, candidates)
        if chosen:
            cases.append({"scenario": label, "qid": chosen["qid"], "question": chosen["question"]})
        print(f"[demo] {label}: {len(candidates)} candidates")

    gen_path = out / "generation_predictions.jsonl"
    if gen_path.exists():
        preds = read_jsonl(gen_path)
        conditions = [c for c in dict.fromkeys(p["condition"] for p in preds) if c not in ("closed_book", "oracle")]
        best = "hybrid" if "hybrid" in conditions else conditions[-1]  # strongest retriever that was run
        unanswerable = [p for p in preds if p["condition"] == best and not p["answerable"]]
        hallucinated = [p for p in unanswerable if p["prediction"]]
        chosen = pick(rng, hallucinated) or pick(rng, unanswerable)
        if chosen:
            label = "unanswerable_hallucinated" if chosen["prediction"] else "unanswerable_refused"
            cases.append({"scenario": label, "qid": chosen["qid"], "question": chosen["question"]})
    else:
        print("[demo] no generation results yet; run the generation stage to add an unanswerable case")

    (out / "demo_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"[done] {len(cases)} demo cases written to {out / 'demo_cases.json'}")


if __name__ == "__main__":
    main()
