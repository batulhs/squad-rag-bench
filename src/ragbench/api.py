"""HTTP service: ask a question, choose a retriever, get a grounded answer with sources.

  uvicorn ragbench.api:app --host 127.0.0.1 --port 8080
  curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
       -d '{"question": "In what country is Normandy located?", "retriever": "hybrid"}'
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import load_config, load_squad_open
from .generator import Generator, clean_answer, is_refusal
from .retrievers import build_retrievers

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build every index once at startup; requests only search.
    cfg = load_config(os.environ.get("RAGBENCH_CONFIG", "config.yaml"))
    passages, _ = load_squad_open(cfg["data"]["dataset"], cfg["data"]["split"])
    STATE["passages"] = {p.pid: p for p in passages}
    STATE["retrievers"] = build_retrievers(cfg, passages)
    STATE["generator"] = Generator.from_config(cfg)
    yield
    STATE.clear()


app = FastAPI(title="SQuAD RAG Bench", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    retriever: Literal["tfidf", "bm25", "dense", "dense_ft", "hybrid"] = "hybrid"
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    pid: str
    title: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "passages": len(STATE.get("passages", {})), "retrievers": list(STATE.get("retrievers", {}))}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # Plain `def`: FastAPI runs it in a worker thread, so blocking calls don't stall the server.
    t0 = time.perf_counter()
    hits = STATE["retrievers"][req.retriever].search([req.question], req.k)[0]
    t1 = time.perf_counter()
    passages = [STATE["passages"][pid] for pid, _ in hits]
    raw = STATE["generator"].answer(req.question, [p.full_text for p in passages])
    t2 = time.perf_counter()
    return AskResponse(
        answer=clean_answer(raw),
        refused=is_refusal(raw),
        sources=[Source(pid=p.pid, title=p.title, score=s, text=p.text) for p, (_, s) in zip(passages, hits)],
        retrieval_ms=round((t1 - t0) * 1000, 2),
        generation_ms=round((t2 - t1) * 1000, 2),
    )
