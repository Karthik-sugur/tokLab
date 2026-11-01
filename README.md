# TokLab

TokLab is a research workbench that asks one question: does tokenizing code
by its **structure** (tree-sitter) together with a **custom vocabulary** of
company-specific names need fewer tokens than either idea alone, without
changing a single byte of the code? It builds the four tokenizers of a 2×2
design (C2 code tokenizer, C3 + custom names, C4 + structure, C5 both) on the
same base vocabulary, checks that every one of them round-trips every file
exactly, and measures tokens per KB of original source with
repository-level confidence intervals. Results, including negative ones, are
written to a generated report; nothing in this README is a result.

**Results snapshot:** a copy of the demo run's generated report is kept at
[`docs/results/report_a63c8847.md`](docs/results/report_a63c8847.md)
(labelled inside as a snapshot of that run; not re-computed here). The live
report under `runs/<hash>/` and `demo_assets/report.html` are git-ignored and
come from the pipeline or `demo_bundle.zip` (hash in
[`configs/demo.yaml`](configs/demo.yaml)).

## Install

Python 3.11, macOS or Linux.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'          # library, CLI, API, UI, tests
pip install -e '.[models]'       # optional: torch (systems latency, fine-tuning code)
```

The base tokenizer (`Qwen/Qwen2.5-Coder-0.5B`) is downloaded from the
Hugging Face Hub on first use and cached afterwards.

## Reproduce

Each command prints what the next one needs (`manifest_hash=`,
`config_hash=`, `k_star=`).

```bash
# 1. fetch the pinned repositories listed in configs/repos.yaml
python scripts/fetch_repos.py --config configs/repos.yaml --out data/repos

# 2. ingest + repository-level splits -> data/processed/manifest.parquet
toklab data build --repos data/repos --out data/processed --seed 0

# 3. synthetic renaming layer (all splits + drift files) and its validation gate
toklab data synth --manifest data/processed/manifest.parquet \
  --repos data/repos --out data/synthetic --seed 0
python scripts/validate_synth.py --repos data/repos --synthetic data/synthetic

# 4. experiment: K sweep, K* chosen on val (docs/decisions/005-k-selection.md),
#    fidelity, tokens/KB, 2x2 interaction, context utilisation, vocab cost
toklab run experiment --config configs/experiments/mvp_py.yaml   # prints config_hash=<HASH>

# 5. drift at K*
toklab run drift --run <HASH>

# 6. systems latency at K* (needs '.[models]'; CPU; optional)
toklab run systems --run <HASH>

# 7. report (Markdown) and a self-contained HTML copy
toklab report --run <HASH>
toklab report --run <HASH> --format html     # -> demo_assets/report.html
```

Steps 2–7 in one go, with gates and per-step timings:
`bash scripts/run_real_pipeline.sh` (`SKIP_SYSTEMS=1` skips step 6).

Tests and lint: `pytest -q` and `ruff check .`.

## Demo

```bash
bash scripts/demo.sh
```

Activates `.venv`, checks that `runs/<demo_run>/` exists, starts the API,
waits for it, starts Streamlit headless and opens <http://localhost:8501>.
The sidebar is pre-filled with the demo run (`TOKLAB_DEFAULT_RUN=<hash>`
overrides `configs/demo.yaml`). If the base tokenizer is already cached the
script sets `HF_HUB_OFFLINE=1`; otherwise it warns that Preview needs
internet once. Ctrl-C stops both servers. The talk track is in
[`docs/DEMO.md`](docs/DEMO.md).

**No server at all:** open `demo_assets/report.html` in any browser
(`toklab report --run <HASH> --format html`).

**Second laptop:**

```bash
# on the machine with the run
toklab report --run <HASH> --format html     # optional, adds the HTML copy
bash scripts/export_demo_bundle.sh           # -> demo_bundle.zip

# on the second laptop (repo cloned and installed as above)
unzip -o demo_bundle.zip -d .                # restores runs/<HASH>/, data/synthetic/...
bash scripts/demo.sh                         # or open demo_assets/report.html
```

The bundle holds the run's report, tables, `drift_table.txt` and the K\*
vocabularies (for Preview); files larger than `MAX_FILE_MB` (default 5) are
skipped and listed in `BUNDLE_MANIFEST.txt` inside the zip. Preview on the
second laptop needs the base tokenizer, i.e. internet once.

## Implemented / not implemented

Implemented:

- C2–C5 as a strict 2×2 (structure on/off × custom vocabulary on/off) over
  one base vocabulary; custom tokens get ids after the base vocabulary.
- Byte-exact round-trip check on every evaluated file for every condition;
  a failing condition is marked INVALID and excluded from claims.
- Repository-level splits (adapt / val / test / drift_holdout) plus the
  file-level `test_inorg` set; vocabulary mined from the adapt split only.
- Synthetic renaming layer with a validation gate, and pre-generated drift
  variants.
- Tokens per KB of original UTF-8 bytes with cluster-bootstrap CIs over
  repositories, the 2×2 interaction, context utilisation, K sweep with K\*
  chosen on val, hypotheses H1–H5 judged at K\*.
- Drift (STATIC vs append-only REFRESH vocabulary) at K\*.
- CPU latency benchmark at K\* (separate step; the report says "not run"
  when it was skipped).
- Vocabulary / KV-cache / FLOPs overhead, labelled ESTIMATE.
- Generated Markdown + HTML report, FastAPI, display-only Streamlit UI.

Not implemented / not measured:

- **Quality is not measured.** No bits-per-byte result exists; C3/C5 add
  embedding rows that were never trained, so any quality figure for them is
  suppressed.
- **Real fine-tuning was not done.** Only a dry-run path exists
  (`toklab run finetune-dry`).
- **C1 (GPT-2) and C6 (byte-level) are intrinsic only:** fidelity and
  tokens/KB via `toklab run baselines` on the original repositories. They
  are not part of the 2×2 experiment, the demo run or the UI.
- Downstream task performance (completion, retrieval, ...) is not measured.
- Python only.

## Limitations

<!-- limitations:start -->
_Copied verbatim from `runs/a63c884760b93db98798e5e5e6dadba5be8653ae956b66cd4ef9f011908c3403/report.md` by `scripts/sync_readme_limitations.py`; re-run it after a new run._

- Shared Zipf lexicon: adapt and evaluation splits are renamed from one lexicon, so company-style names overlap heavily between adapt and the evaluation sets (measured, structural identifiers): test: 497312 of 614955 identifier occurrences (0.8087) and 8266 of 22135 distinct identifiers (0.3734) also occur in adapt; test_inorg: 363595 of 390658 identifier occurrences (0.9307) and 13433 of 20121 distinct identifiers (0.6676) also occur in adapt.
- Related repositories in different splits: repositories are split as whole units, but related projects (same ecosystem, shared dependencies or test fixtures) can sit in different splits; relatedness is not controlled. Split membership: adapt: attrs, black, fastapi, httpx, itsdangerous, jinja, jsonschema, marshmallow, mkdocs, networkx, poetry, requests, scrapy, sqlalchemy, structlog, urllib3, uvicorn; drift_holdout: rich, virtualenv, werkzeug; test: celery, click, flask, pydantic, tornado; test_inorg: attrs, black, fastapi, httpx, itsdangerous, jinja, jsonschema, marshmallow, mkdocs, networkx, poetry, requests, scrapy, sqlalchemy, structlog, urllib3, uvicorn; val: aiohttp, pytest, starlette.
- Only 5 test repositories: CIs are cluster bootstraps over 5 repositories and are coarse.
- test_inorg is a FILE-level hold-out inside adapt repositories, an explicit exception to repository-level splitting: it measures reuse of a known codebase's names, not generalisation to new codebases. Hypothesis claims use the primary set (test) first.
- Alpha-equivalence is not runtime equivalence: the synthetic renamer checks that renamed code has the same syntax tree up to consistent identifier renaming; it does not prove the code behaves the same (string-based lookups such as getattr, reflection or external APIs can break).
- K chosen on val: K*=16384 minimises C3 tokens/KB on the val split (440 files), chosen with C3 only (docs/decisions/005-k-selection.md). Hypotheses are judged at K* only; other K values are descriptive. Selecting K on val still tunes the setup to this corpus.
- Selection uses min_freq=1 and min_repos=1. With min_repos=1, entries frequent in a single adapt repository can be selected: they inflate gains on test_inorg (same repositories) and overfit relative to test (new repositories).
- Quality not measured: no fine-tuning was run, so there is no bits-per-byte result; C3/C5 add untrained embedding rows and any quality figure for them is suppressed.
- 5 file(s) excluded at ingest for every condition because the base tokenizer cannot round-trip them (NFC normalisation): pydantic/tests/test_networks.py (NFC normalizer: U+F96E CJK compatibility ideograph -> U+8449 (base tokenizer, C2 fails)); black/tests/data/cases/docstring.py (NFC normalizer: U+2001 EM QUAD -> U+2003 EM SPACE (base tokenizer, C2 fails)); pytest/testing/io/test_terminalwriter.py (NFC normalizer: 'e' + U+0301 combining acute -> U+00E9 (base tokenizer, C2 fails)); aiohttp/tests/test_http_parser.py (NFC normalizer: U+212A KELVIN SIGN -> 'K' (base tokenizer, C2 fails)); aiohttp/tests/test_multipart_helpers.py (NFC normalizer: 'a' + U+0308 combining diaeresis -> U+00E4 (base tokenizer, C2 fails)).
- FLOPs / KV / param overhead figures are ESTIMATE labels, not measured.
- Drift REFRESH may append tokens mined from a drifted v1 sample (size recorded in drift_meta); old adapted ids stay unchanged (append-only). Drift BPB is reported only for gate-valid fine-tuned variants; otherwise marked not run. This does not show robustness to large-scale renaming.
- All evaluated conditions passed fidelity on this run.
<!-- limitations:end -->

## Folder layout

```text
configs/            experiment, ingest, repos, lexicon, demo configs; hypotheses.yaml (pre-registered)
docs/decisions/     design decisions (model, round-trip, splits, test_inorg, K selection)
docs/DEMO.md        10-minute demo cue sheet
reports/templates/  Jinja2 report template
scripts/            fetch, pipeline, demo, bundle export, diagnostics
src/toklab/
  core/             types, interfaces, hashing, byte-exact source I/O
  data/             ingest, splits, dedup, corpus reader, synth/ (renaming layer)
  parsing/          tree-sitter segmenters, roles
  vocab/            candidate mining, selection, adapted vocabulary
  tokenizers/       native HF, composed (C3–C5), byte-level, registry
  evaluation/       fidelity, intrinsic, stats, factorial, drift, systems, vocab cost, report
  models/           loader, gate, embedding adaptation, fine-tuning (dry run)
  experiments/      config, runner, run store, preview
  api/              FastAPI app
  cli.py            `toklab` command
ui/                 Streamlit UI (display only; talks to the API)
tests/              unit, api
data/  runs/  logs/ generated, git-ignored
```
