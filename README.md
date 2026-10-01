# squad-rag-bench

A controlled comparison of retrieval methods for Retrieval-Augmented Generation (RAG), including a dense retriever fine-tuned for the task. The generator, prompt and data are fixed; only the retriever changes, so differences in answer quality are attributable to retrieval.

| Retriever | Idea |
|---|---|
| TF-IDF + cosine | Weighted exact-word overlap |
| BM25 | TF-IDF with term-frequency saturation and document-length normalisation |
| Dense (`bge-small-en-v1.5`) | Embeddings: matches meaning, not exact words |
| Dense, fine-tuned | The same model after contrastive fine-tuning on SQuAD 2.0 train |
| Hybrid + rerank | BM25 and dense fused with Reciprocal Rank Fusion, then a cross-encoder reranks the top 50 |

`dense` vs `dense_ft` isolates the effect of fine-tuning: same architecture, same size, same inference code.

Two baselines bracket them: **closed-book** (no retrieval, lower bound) and **oracle** (gold passage given, upper bound).

## Dataset

SQuAD 2.0 dev set ([Rajpurkar et al., 2018](https://arxiv.org/abs/1806.03822)): 11,873 questions over 35 Wikipedia articles, about half deliberately unanswerable. All unique paragraphs are pooled into one store, so each question must first find its paragraph among ~1,200. The original paragraph is the retrieval label; the original answers are the generation labels. Unanswerable questions test whether the model refuses instead of inventing an answer.

## Fine-tuning

`python -m ragbench.train` builds one (question, paragraph) pair per unique paragraph in SQuAD 2.0 **train** (answerable questions only; up to 20,000 pairs) and trains `bge-small-en-v1.5` for one epoch with `MultipleNegativesRankingLoss`: each question must score its own paragraph above the other 63 paragraphs in its batch. One question per paragraph means no in-batch negative is secretly a correct match. Train and dev use different Wikipedia articles, so no test paragraph is seen in training.

## Pipeline

```
question ──► retriever (1 of 4) ──► top-3 passages ──► prompt ──► Qwen2.5-7B-Instruct (vLLM) ──► answer | NO_ANSWER
                 ▲                                                                             │
   ~1,200 SQuAD paragraphs (normalised, tokenised, indexed)                     EM / F1 / refusal accuracy
```

## Metrics

| Stage | Metrics |
|---|---|
| Retrieval | Recall@1/5/10, MRR@10, nDCG@10, query latency p50/p95, batch throughput |
| Generation | EM, F1 (official SQuAD 2.0), refusal accuracy, false-refusal rate, context recall, latency p50/p95, throughput |

## Results

_Fill in from `results/*.csv` after running._

| Retriever | Recall@5 | MRR@10 | EM | F1 | Refusal acc. | p50 latency |
|---|---|---|---|---|---|---|
| closed-book | – | – | 0.316 | 0.3406 | 0.532 | - |
| TF-IDF | 0.9238 | 0.8284 | 0.584 | 0.6556 | 0.596 | 1.1333 |
| BM25 | 0.9312 | 0.853 | 0.604 | 0.6706 | 0.612 | 1.0988 |
| Dense | 0.9062 | 0.7883 | 0.606 | 0.6707 | 0.624 | 13.6244 |
| Dense, fine-tuned | 0.9197 | 0.8072 | 0.606 | 0.6684 | 0.616 | 13.4986 |
| Hybrid + rerank | 0.9843 | 0.9367 | 0.594 | 0.6647 | 0.572 | 245.6712 |
| oracle | 1.000 | 1.000 | 0.632 | 0.7086 | 0.6 | |

## Run it

Step-by-step instructions, including a free route on Kaggle: **[RUN.md](RUN.md)**.

Quick reference:

```bash
pip install -e ".[dev]"
python -m ragbench.train                               # fine-tune the dense retriever
python -m ragbench.evaluate retrieval                  # no LLM needed
python -m ragbench.evaluate all                        # needs the LLM server running
python -m ragbench.evaluate retrieval --retrievers tfidf bm25 --limit 300   # quick CPU smoke test
python -m ragbench.analyze
python -m ragbench.demo -q "Who ruled Normandy in the 10th century?"
uvicorn ragbench.api:app --host 127.0.0.1 --port 8080  # HTTP API, docs at /docs
```

`config.yaml` serves the generator with vLLM (full precision); `config_kaggle.yaml` uses Ollama (4-bit) on a free T4.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `retrieval_metrics.csv` | One row per retriever |
| `retrieval_per_query.jsonl` | Gold rank of every question under every retriever, plus word overlap |
| `generation_metrics.csv` | One row per condition (closed-book, 4 retrievers, oracle) |
| `generation_predictions.jsonl` | Every prompt's retrieved passages, raw output and cleaned answer |
| `breakdown_by_question_type.csv` | Recall@5 for what / who / when / ... questions |
| `breakdown_by_overlap.csv` | Recall@5 by how many question words appear in the gold passage |
| `demo_cases.json` | One real example per scenario (BM25-only win, dense-only win, fine-tuned-only win, hybrid rescue, unanswerable) |

## Layout

```
src/ragbench/
  data.py        load SQuAD 2.0, build passage store, normalise + tokenise
  retrievers.py  TF-IDF, BM25, dense, fine-tuned dense, hybrid + rerank behind one interface
  train.py       contrastive fine-tuning of the dense retriever
  generator.py   vLLM client, grounded prompt, refusal handling
  metrics.py     retrieval, SQuAD EM/F1, refusal and latency metrics
  evaluate.py    benchmark CLI
  analyze.py     error analysis and demo-case selection
  demo.py        side-by-side demo
  api.py         FastAPI service
tests/           unit tests for metrics, parsing, sparse retrievers, training-pair construction
config.yaml      every setting, one seed (vLLM generator)
config_kaggle.yaml  same, with Ollama generator for free Kaggle GPUs
kaggle_run.ipynb    the full run as a Kaggle notebook
RUN.md           step-by-step instructions
```

## Known limitation

SQuAD's "unanswerable" labels were written against a single paragraph. After pooling, a few may be answerable from a neighbouring paragraph, so refusal accuracy is slightly pessimistic. Inspect `generation_predictions.jsonl` for such cases.

## References

- Lewis et al., 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Robertson & Zaragoza, 2009. The Probabilistic Relevance Framework: BM25 and Beyond.
- Karpukhin et al., 2020. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Henderson et al., 2017. Efficient Natural Language Response Suggestion for Smart Reply. (in-batch negatives loss)
- Cormack, Clarke & Büttcher, 2009. Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Nogueira & Cho, 2019. Passage Re-ranking with BERT.
- Rajpurkar, Jia & Liang, 2018. Know What You Don't Know: Unanswerable Questions for SQuAD. ACL.
