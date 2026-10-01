# How to run squad-rag-bench

Everything below is free: **GitHub** hosts the code, **Kaggle** provides two NVIDIA T4 GPUs (30 hours/week), and **Ollama** runs the language model on one of them.

**What the run does, in order:**

1. Fine-tunes a small embedding model on SQuAD 2.0 *train*.
2. Evaluates 5 retrievers on SQuAD 2.0 *dev*.
3. Generates answers with Qwen2.5-7B for 6 conditions (closed-book, 5 retrievers, oracle).
4. Writes the analysis and demo cases.

---

## Part A: One-time setup

### A1. Kaggle account

1. Sign up at <https://www.kaggle.com>.
2. Go to your profile picture → **Settings** → **Phone verification** and verify your number. Without this, Kaggle won't let you turn on GPUs or internet in notebooks.

### A2. GitHub account and Git

1. Sign up at <https://github.com>.
2. In PowerShell, check Git is installed:
   ```powershell
   git --version
   ```
   If you get an error, install Git from <https://git-scm.com/download/win>, then close and reopen PowerShell.
3. If this is your first time using Git on this laptop, set your name and email:
   ```powershell
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

---

## Part B: Put the code on GitHub

### B1. Unzip

Right-click `squad-rag-bench.zip` → **Extract All**. You get a folder `squad-rag-bench` containing `README.md`, `RUN.md`, `src`, and so on.

### B2. Create an empty repository

On GitHub: **+** (top right) → **New repository**.

- Name: `squad-rag-bench`
- Visibility: **Public**
- Leave "Add a README", ".gitignore" and "license" **unticked**, because the project already has them.

Click **Create repository**.

### B3. Push the code

In PowerShell, go into the extracted folder and push. Replace `YOUR_USERNAME`.

```powershell
cd $HOME\Downloads\squad-rag-bench\squad-rag-bench   # adjust to where you extracted it
git init
git add .
git commit -m "RAG retriever benchmark on SQuAD 2.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/squad-rag-bench.git
git push -u origin main
```

A browser window may open asking you to sign in to GitHub. Do that. Then refresh the repository page: you should see the files.

---

## Part C: Run on Kaggle

### C1. Import the notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. In the notebook editor: **File → Import Notebook** → upload `kaggle_run.ipynb` from the project folder.

### C2. Turn on GPU and internet

In the right-hand panel under **Session options**:

- **Accelerator:** `GPU T4 x2`
- **Internet:** `On`

### C3. Set your repository URL

In the first code cell, change:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/squad-rag-bench.git"
```

### C4. Run the cells in order

Click into each cell and press **Shift + Enter**. Wait for a cell to finish (the spinner stops) before running the next.

| Cell | What it does | What you should see |
|---|---|---|
| 1 | Downloads your code | `/kaggle/working/squad-rag-bench` |
| 2 | Installs packages, runs tests | `12 passed` |
| 3 | Lists GPUs | Two rows saying `Tesla T4` |
| 4 | Fine-tunes the retriever | A progress bar with falling `loss` values, then `[done] fine-tuned model saved ...` and a JSON summary |
| 5a | Installs Ollama | `>>> Install complete` (warnings about systemd are normal) |
| 5b | Starts Ollama, downloads Qwen2.5-7B | Download progress, then `success` |
| 6 | Test run on 40 questions | Two tables: retrieval metrics (5 rows) and generation metrics (7 rows) |
| 7 | Full run | Same tables, now on all questions |
| 8 | Analysis | Recall tables by question type and by word overlap; a list of demo scenarios |
| 9 | Demo | Each demo question with the top passages and answer from every retriever |
| 10 | Packages results | `results.zip` with its size |

I haven't timed this on a T4. Expect the full run (cells 4 to 9) to take tens of minutes, mostly downloads and cell 7. **Keep the browser tab open while it runs**; an idle interactive session can be shut down.

### C5. Download the results

Open the **Output** panel (right side, or **File → Output**) → `results.zip` → download.

### C6. Stop the session

**Run → Stop Session.** A running session uses your 30-hour quota even when nothing is executing.

---

## Part D: What you get

Inside `results.zip`:

| File | Use it for |
|---|---|
| `retrieval_metrics.csv` | Main retrieval table: Recall@1/5/10, MRR, nDCG, latency, throughput for 5 retrievers |
| `generation_metrics.csv` | Main answer table: EM, F1, refusal accuracy, false-refusal rate, latency for 7 conditions |
| `training_summary.json` | Fine-tuning details: base model, pairs, steps, time, GPU |
| `breakdown_by_question_type.csv` | Where each retriever wins: what / who / when / ... |
| `breakdown_by_overlap.csv` | Keyword vs. meaning: how recall changes when the question shares few words with the passage |
| `demo_cases.json`, `demo_output.txt` | Your demo test cases and their outputs |
| `retrieval_per_query.jsonl`, `generation_predictions.jsonl` | Every single question's result, for error analysis |

**The headline comparison is `dense` vs `dense_ft`:** the same model before and after your fine-tuning.

After the run, fill in the Results table in `README.md` and push again:

```powershell
git add README.md
git commit -m "Add results"
git push
```

---

## Part E: Live demo during the presentation

Start a Kaggle session and run cells 1, 2, 4, 5a and 5b. Then run:

```python
!CUDA_VISIBLE_DEVICES=1 python -m ragbench.demo --config config_kaggle.yaml -q "Your question here"
```

Session files are deleted when a Kaggle session ends, so the trained model has to be rebuilt in a new session. Cell 4 is quick compared with the full run.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Accelerator options are greyed out | Phone not verified | Part A1 |
| Cell 1: `Could not resolve host: github.com` | Internet is off | Session options → Internet: On, then re-run |
| Cell 1: `Repository not found` | Wrong URL or private repo | Check `REPO_URL`; make the repo public |
| Cell 2: some tests fail | Package versions differ on Kaggle | Copy the error output and send it to me |
| Cell 4 or 7: `CUDA out of memory` | Both jobs on one GPU | Make sure the cell includes `CUDA_VISIBLE_DEVICES=1`; restart session if needed |
| Cell 6 hangs or shows connection errors | Ollama not running | New cell: `!tail -30 /kaggle/working/ollama.log`, then re-run cell 5b |
| `Fine-tuned model not found` | Session restarted, so cell 4's output is gone | Re-run cell 4, or add `--retrievers tfidf bm25 dense hybrid` to skip it |
| Session ended mid-run | Idle timeout or 12-hour limit | Re-run from cell 1. Kaggle deletes session files when a session ends, so download `results.zip` after each full run |

---

## Optional: on your own Linux GPU server (vLLM instead of Ollama)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install vllm   # separate env: vLLM pins its own torch
tmux new -s llm 'source ~/venvs/vllm/bin/activate && GPU=0 bash scripts/serve_llm.sh'
EVAL_GPU=1 bash scripts/run_all.sh
```

This uses `config.yaml` (full-precision Qwen2.5-7B via vLLM) instead of `config_kaggle.yaml` (4-bit via Ollama). Report which one you used.

---

## Optional: run the tests on your Windows laptop

No GPU needed. This only checks the code installs and the core logic works.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
