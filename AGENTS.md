# TokLab: Agent Rulebook

## What we are building (read this first)

TokLab is a Python tool that answers ONE research question:

  "If we tokenize code using (a) its structure and (b) a custom
   vocabulary of company-specific names, together, do we need fewer
   tokens than using either one alone, WITHOUT damaging the code?"

A "token" is a piece of text a language model reads. The identifier
`CustomerPaymentTransactionService` may be split into 6 tokens by a normal
tokenizer. Fewer tokens = more code fits in the model's memory window =
cheaper and faster.

TokLab builds 6 tokenizers, runs them on the SAME code, measures them
fairly, and writes an honest report. It is a research workbench, NOT a
chatbot, NOT a SaaS product.

## The 6 conditions (C1 to C6)

| ID | Name            | Splitting by | Vocabulary        |
|----|-----------------|--------------|-------------------|
| C1 | General BPE     | normal text  | GPT-2 vocab       |
| C2 | Code tokenizer  | normal text  | code model's vocab|
| C3 | Domain-adapted  | normal text  | code vocab + custom names |
| C4 | Structure-aware | code structure (tree-sitter) | code model's vocab |
| C5 | Combined        | code structure | code vocab + custom names |
| C6 | Byte-level      | raw bytes    | 256 bytes         |

C2, C3, C4, C5 form a 2x2 grid (structure on/off x custom vocab on/off).
That grid is how we prove "better than either alone". NEVER break it:
C2..C5 must differ ONLY in those two switches.

## Golden rules (breaking any of these = task failed)

1. LOSSLESS: for every tokenizer, decode(encode(x)) == x exactly,
   byte for byte. No lowercasing, no whitespace cleanup, no anonymizing.
2. NO LEAKAGE: split data by REPOSITORY (never by file). The custom
   vocabulary is built ONLY from the adapt split. Never look at test data
   when building anything.
3. APPEND-ONLY IDS: custom tokens get IDs AFTER the base vocabulary.
   Never renumber an existing token.
4. COMPRESSION RATIO = original UTF-8 bytes of the source / number of
   tokens. Always the ORIGINAL bytes. Higher = better.
5. NEVER INVENT RESULTS. No made-up numbers in code, docs, tests,
   or reports. If something was not run, say "not run".
6. REPORT NEGATIVE RESULTS. If the combined method does not win,
   the report must say so plainly.
7. QUALITY METRIC = bits-per-byte (BPB), not per-token perplexity.
   Per-token numbers are not comparable across tokenizers.
8. MODEL HONESTY: if a tokenizer has new vocabulary rows that were
   never trained, quality results for it are SUPPRESSED, not shown.
9. DETERMINISM: every run uses seeds from config. Same config + seed
   = same token metrics, exactly.
10. KEEP IT SIMPLE: one Python package, files + SQLite. No Docker,
    no Kubernetes, no Redis, no microservices, no extra frameworks.

## Tech stack (do not swap without asking)

Python 3.11, uv or venv, pytest, hypothesis, ruff, typer (CLI),
transformers + tokenizers (HF), tree-sitter + tree-sitter-python,
libcst, datasketch, numpy, scipy, pandas/pyarrow, jinja2,
torch (only in toklab/models and toklab/evaluation/systems.py).
Later: fastapi, pydantic, streamlit, plotly.

## Repo layout (create folders only when a phase needs them)

src/toklab/core/         types, interfaces, hashing
src/toklab/data/         ingest, splits, dedup, synth/
src/toklab/parsing/      segmenters, roles
src/toklab/vocab/        mining, selection, adapted_vocab
src/toklab/tokenizers/   native_hf, composed, byte_level, registry
src/toklab/evaluation/   fidelity, intrinsic, systems, vocab_cost,
                         quality, drift, stats, report
src/toklab/models/       loader, gate, adapt_embeddings, finetune
src/toklab/experiments/  config, runner, store, env_log
src/toklab/cli.py
src/toklab/api/          (phase 10 only)
ui/                      (phase 10 only)
tests/{unit,integration,api,research,fixtures}
configs/  scripts/spikes/  reports/templates/  docs/decisions/

Import rules: `core` imports nothing from toklab. `evaluation` never
imports `api` or `ui`. `api`/`ui` only call `experiments.runner` and read
the store. Never put science logic in the UI. `models` is optional:
intrinsic runs must work without importing torch.

## How you must work on every task

1. Read this file and the task prompt fully.
2. Do ONLY the task. Do not build future phases.
3. Write tests FIRST or together with code.
4. Run `ruff check .` and `pytest -q`. Both must pass.
5. Run the task's "Acceptance command". Paste its real output
   in your final message.
6. Final message must contain: files created/changed, commands run,
   real output, anything you could not do, and any assumption you made.
7. If something is unclear or a library behaves differently than
   expected, STOP and say so. Do not guess silently.
8. Commit with message "phase-N: short description".

## Definition of done (every task)

[ ] code written  [ ] tests written and passing  [ ] ruff clean
[ ] acceptance command run, real output shown
[ ] no golden rule broken  [ ] no fake numbers anywhere
