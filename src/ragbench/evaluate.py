"""Run the benchmark.

  python -m ragbench.evaluate retrieval            # all retrievers, all answerable questions
  python -m ragbench.evaluate generation           # needs the vLLM server running
  python -m ragbench.evaluate all
  python -m ragbench.evaluate retrieval --retrievers tfidf bm25 --limit 300   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from .data import lexical_overlap, load_config, load_squad_open, sample_balanced
from .generator import Generator, clean_answer
from .metrics import gold_rank, latency_summary, qa_metrics, retrieval_metrics
from .retrievers import ALL_RETRIEVERS, build_retrievers


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_retrieval(cfg, passages, queries, retrievers, out_dir: Path, limit: int | None) -> pd.DataFrame:
    rc = cfg["retrieval"]
    depth = rc["depth"]
    # Only answerable questions have a passage that actually contains the answer.
    evalset = [q for q in queries if q.answerable][:limit]
    questions = [q.question for q in evalset]
    text_by_pid = {p.pid: p.full_text for p in passages}
    rng = random.Random(cfg["seed"])
    latency_qs = rng.sample(questions, min(rc["latency_sample"], len(questions)))

    rows, ranks_by_name = [], {}
    for name, retriever in retrievers.items():
        print(f"[retrieval] {name}: searching {len(questions)} questions")
        start = time.perf_counter()
        hits = retriever.search(questions, depth)
        batch_seconds = time.perf_counter() - start

        ranks = [gold_rank([pid for pid, _ in h], q.gold_pid) for h, q in zip(hits, evalset)]
        ranks_by_name[name] = ranks

        single_ms = []
        for q in latency_qs:  # one question at a time, as a live user would send it
            t0 = time.perf_counter()
            retriever.search([q], depth)
            single_ms.append((time.perf_counter() - t0) * 1000)

        rows.append({
            "retriever": name,
            **retrieval_metrics(ranks, rc["recall_ks"], depth),
            **latency_summary(single_ms, "query"),
            "batch_qps": len(questions) / batch_seconds,
            "index_s": retriever.index_seconds,
        })

    per_query = [{
        "qid": q.qid,
        "question": q.question,
        "gold_pid": q.gold_pid,
        "overlap": round(lexical_overlap(q.question, text_by_pid[q.gold_pid]), 4),
        "ranks": {name: ranks_by_name[name][i] for name in retrievers},
    } for i, q in enumerate(evalset)]

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "retrieval_metrics.csv", index=False)
    write_jsonl(out_dir / "retrieval_per_query.jsonl", per_query)
    return df


def run_generation(cfg, passages, queries, retrievers, out_dir: Path, limit: int | None) -> pd.DataFrame:
    gc = cfg["generation"]
    k = gc["top_k_context"]
    n_per_class = min(gc["n_per_class"], limit // 2) if limit else gc["n_per_class"]
    sample = sample_balanced(queries, n_per_class, cfg["seed"])
    questions = [q.question for q in sample]
    golds = [q.answers for q in sample]
    text_by_pid = {p.pid: p.full_text for p in passages}
    generator = Generator.from_config(cfg)

    # Baselines bracket the retrievers: closed_book = no retrieval (lower bound),
    # oracle = the gold passage handed over directly (upper bound).
    conditions = ["closed_book", *retrievers, "oracle"]
    rows, predictions = [], []
    for cond in conditions:
        print(f"[generation] {cond}: {len(sample)} questions")
        if cond == "closed_book":
            pid_lists = [[] for _ in sample]
        elif cond == "oracle":
            pid_lists = [[q.gold_pid] for q in sample]
        else:
            pid_lists = [[pid for pid, _ in h] for h in retrievers[cond].search(questions, k)]

        items = [(q, [text_by_pid[p] for p in pids]) for q, pids in zip(questions, pid_lists)]
        start = time.perf_counter()
        outputs = generator.answer_many(items)
        wall = time.perf_counter() - start

        cleaned = [clean_answer(raw) for raw, _ in outputs]
        answerable_hits = [q.gold_pid in pids for q, pids in zip(sample, pid_lists) if q.answerable]
        rows.append({
            "condition": cond,
            **qa_metrics(cleaned, golds),
            "context_recall": sum(answerable_hits) / max(len(answerable_hits), 1),
            **latency_summary([ms for _, ms in outputs], "gen"),
            "throughput_qps": len(sample) / wall,
        })
        predictions += [{
            "condition": cond, "qid": q.qid, "question": q.question, "answerable": q.answerable,
            "gold_answers": list(q.answers), "gold_pid": q.gold_pid, "retrieved": pids,
            "raw_output": raw, "prediction": pred,
        } for q, pids, (raw, _), pred in zip(sample, pid_lists, outputs, cleaned)]

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "generation_metrics.csv", index=False)
    write_jsonl(out_dir / "generation_predictions.jsonl", predictions)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="SQuAD 2.0 open-retrieval RAG benchmark")
    parser.add_argument("stage", choices=["retrieval", "generation", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--retrievers", nargs="+", choices=ALL_RETRIEVERS, default=list(ALL_RETRIEVERS))
    parser.add_argument("--limit", type=int, default=None, help="cap the number of questions (smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    passages, queries = load_squad_open(cfg["data"]["dataset"], cfg["data"]["split"])
    n_ans = sum(q.answerable for q in queries)
    print(f"[data] {len(passages)} passages | {len(queries)} questions ({n_ans} answerable)")

    retrievers = build_retrievers(cfg, passages, args.retrievers)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if args.stage in ("retrieval", "all"):
            print(run_retrieval(cfg, passages, queries, retrievers, out_dir, args.limit).to_string(index=False))
        if args.stage in ("generation", "all"):
            print(run_generation(cfg, passages, queries, retrievers, out_dir, args.limit).to_string(index=False))
    print(f"[done] results written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
