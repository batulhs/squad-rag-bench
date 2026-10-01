"""LLM answer generation through a vLLM server's OpenAI-compatible API.

vLLM batches concurrent requests on the GPU (continuous batching), so sending
many requests from a thread pool raises throughput without extra code.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

NO_ANSWER = "NO_ANSWER"

GROUNDED_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Reply with the shortest span copied from the context that answers the question, "
    "with no explanation. If the context does not contain the answer, reply with exactly: "
    f"{NO_ANSWER}"
)

CLOSED_BOOK_PROMPT = (
    "Answer the question with the shortest possible phrase and no explanation. "
    f"If you do not know the answer, reply with exactly: {NO_ANSWER}"
)


def is_refusal(output: str) -> bool:
    return NO_ANSWER in output.upper()


def clean_answer(output: str) -> str:
    """Map refusals to the empty string (SQuAD 2.0 convention) and trim formatting."""
    if is_refusal(output):
        return ""
    return output.strip().strip('"').rstrip(".").strip()


class Generator:
    def __init__(self, base_url: str, model: str, max_tokens: int = 32,
                 temperature: float = 0.0, workers: int = 32, api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.max_tokens, self.temperature, self.workers = model, max_tokens, temperature, workers

    @classmethod
    def from_config(cls, cfg: dict) -> "Generator":
        g = cfg["generation"]
        return cls(g["base_url"], g["model"], g["max_tokens"], g["temperature"], g["workers"])

    def answer(self, question: str, contexts: list[str]) -> str:
        if contexts:
            context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
            messages = [
                {"role": "system", "content": GROUNDED_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"},
            ]
        else:  # closed-book baseline: no retrieval at all
            messages = [
                {"role": "system", "content": CLOSED_BOOK_PROMPT},
                {"role": "user", "content": f"Question: {question}\nAnswer:"},
            ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _timed(self, item: tuple[str, list[str]]) -> tuple[str, float]:
        start = time.perf_counter()
        output = self.answer(*item)
        return output, (time.perf_counter() - start) * 1000

    def answer_many(self, items: list[tuple[str, list[str]]]) -> list[tuple[str, float]]:
        """Returns (raw output, latency in ms) per item, in input order."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._timed, items))
