"""Run the demo cases end to end: every retriever, then the LLM, side by side.

  python -m ragbench.demo                 # cases from results/demo_cases.json
  python -m ragbench.demo -q "Who ruled Normandy in the 10th century?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_config, load_squad_open
from .generator import Generator, clean_answer
from .retrievers import build_retrievers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-q", "--question", help="ask a custom question instead of the saved cases")
    args = parser.parse_args()

    cfg = load_config(args.config)
    passages, queries = load_squad_open(cfg["data"]["dataset"], cfg["data"]["split"])
    by_pid = {p.pid: p for p in passages}
    by_qid = {q.qid: q for q in queries}
    retrievers = build_retrievers(cfg, passages)
    generator = Generator.from_config(cfg)
    k = cfg["generation"]["top_k_context"]

    if args.question:
        cases = [{"scenario": "custom", "qid": None, "question": args.question}]
    else:
        cases = json.loads((Path(cfg["paths"]["results"]) / "demo_cases.json").read_text(encoding="utf-8"))

    for case in cases:
        query = by_qid.get(case["qid"])
        print("=" * 90)
        print(f"[{case['scenario']}] {case['question']}")
        if query:
            gold = " | ".join(query.answers) or "(unanswerable)"
            print(f"gold passage: {query.gold_pid} ({by_pid[query.gold_pid].title})   gold answer: {gold}")
        for name, retriever in retrievers.items():
            hits = retriever.search([case["question"]], k)[0]
            pids = [pid for pid, _ in hits]
            mark = "" if not query else ("  <- gold found" if query.gold_pid in pids else "  <- gold missed")
            answer = clean_answer(generator.answer(case["question"], [by_pid[p].full_text for p in pids]))
            print(f"  {name:<8} top-{k}: {', '.join(pids)}{mark}")
            print(f"           answer: {answer or '(refused: NO_ANSWER)'}")


if __name__ == "__main__":
    main()
