"""Fine-tune the dense retriever on SQuAD 2.0 *train* (question, paragraph) pairs.

  python -m ragbench.train

Train and dev use different Wikipedia articles, so the fine-tuned model never
sees a test paragraph. The result is evaluated as the fifth retriever, `dense_ft`,
against the same base model before fine-tuning (`dense`).

Loss: MultipleNegativesRankingLoss (contrastive, in-batch negatives). For a batch
of B pairs, each question must score its own paragraph above the other B-1
paragraphs in the batch, which act as negatives at no extra cost.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .data import load_config


def build_training_pairs(rows, max_pairs: int, seed: int, query_prefix: str = "") -> list[tuple[str, str]]:
    """One answerable question per unique paragraph.

    Taking a single question per paragraph means no paragraph appears twice in a
    batch, so no in-batch "negative" is secretly a correct answer.
    """
    questions_by_context: dict[str, list[str]] = defaultdict(list)
    title_by_context: dict[str, str] = {}
    for row in rows:
        if row["answers"]["text"]:  # answerable only: the paragraph really contains the answer
            questions_by_context[row["context"]].append(row["question"].strip())
            title_by_context[row["context"]] = row["title"]

    rng = random.Random(seed)
    contexts = sorted(questions_by_context)
    rng.shuffle(contexts)
    pairs = []
    for context in contexts[:max_pairs]:
        passage = f"{title_by_context[context].replace('_', ' ')}. {context}"  # same format as Passage.full_text
        pairs.append((query_prefix + rng.choice(questions_by_context[context]), passage))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the dense retriever on SQuAD 2.0 train")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    import torch
    from datasets import Dataset, load_dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    cfg = load_config(args.config)
    tc = cfg["training"]
    prefix = cfg["retrieval"]["dense_ft"]["query_prefix"]
    out_dir = Path(tc["output_dir"])

    rows = load_dataset(cfg["data"]["dataset"], split="train")
    pairs = build_training_pairs(rows, tc["max_pairs"], cfg["seed"], prefix)
    train_set = Dataset.from_dict({"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]})
    print(f"[train] {len(pairs)} (question, paragraph) pairs from SQuAD 2.0 train")

    use_gpu = torch.cuda.is_available()
    model = SentenceTransformer(tc["base_model"], device="cuda" if use_gpu else "cpu")
    loss = losses.MultipleNegativesRankingLoss(model)

    steps = max(1, len(pairs) // tc["batch_size"]) * tc["epochs"]
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        learning_rate=tc["lr"],
        warmup_steps=int(tc["warmup_ratio"] * steps),
        fp16=use_gpu,                              # T4 supports fp16, not bf16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # extra guard against duplicate texts in a batch
        logging_steps=max(1, steps // 20),
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
    )

    start = time.perf_counter()
    SentenceTransformerTrainer(model=model, args=training_args, train_dataset=train_set, loss=loss).train()
    seconds = time.perf_counter() - start

    model.save(str(out_dir))
    summary = {"base_model": tc["base_model"], "pairs": len(pairs), "epochs": tc["epochs"],
               "batch_size": tc["batch_size"], "lr": tc["lr"], "steps": steps,
               "train_seconds": round(seconds, 1), "device": torch.cuda.get_device_name(0) if use_gpu else "cpu"}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] fine-tuned model saved to {out_dir.resolve()} in {seconds:.0f}s")


if __name__ == "__main__":
    main()
