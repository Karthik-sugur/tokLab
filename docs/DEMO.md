# TokLab demo: 10-minute cue sheet

Every number on screen comes from the stored run (`runs/<demo_run>/`). This
sheet contains no results on purpose: read values and verdicts off the
screen, never from memory.

## Before you go on (T−10 min)

1. `bash scripts/demo.sh` and wait for the browser to open
   <http://localhost:8501>. The sidebar should say **API connected** and the
   experiment id should be pre-filled.
2. Click through every page once. The first Preview click loads the
   tokenizer and the K\* vocabulary, so do it now, not on stage.
3. Keep `demo_assets/report.html` open in a second browser tab as the fallback.
4. Wi-Fi: if the terminal printed `tokenizer not cached, Preview needs
   internet once`, run one Preview while online before the talk.

## Running order

| time | page | goal |
|------|------|------|
| 0:00 | (title) | the question |
| 0:30 | Preview | what a token is; lossless |
| 2:00 | Comparison | the measured answer per condition |
| 3:30 | 2x2 interaction | do the two ideas add up? |
| 5:30 | Context utilisation | what it means for a context window |
| 6:30 | Drift | does a frozen vocabulary age? |
| 8:00 | Cost | only if systems ran |
| 8:30 | Report | verdicts, limitations |
| 9:30 | close | measured / not measured |

### 0:00 — The question (no click)

Say: "If we tokenize code by its structure and add our own company names to
the vocabulary, do we need fewer tokens than with either idea alone, without
changing a single byte?"

### 0:30 — Preview

- **Click:** sidebar **Page → Preview**. The snippet is pre-filled (first
  30 lines of one val-split file, set in `configs/demo.yaml`). Leave all four
  conditions selected, click **Tokenize**.
- **Answers:** how each tokenizer cuts the same text. Each coloured box is one
  token; the header shows the token count per condition. The blue note
  names the run and K\* whose vocabulary is loaded.
- **Say:** "Same lines, four tokenizers; every box is one token and the text
  decodes back byte for byte. This is one hand-picked file, an illustration;
  the measured answer is on the next page."

### 2:00 — Comparison

- **Click:** **Page → Comparison**. Eval set **test** (new repositories,
  primary), K left on the value marked **(K\*)**.
- **Answers:** tokens per KB of original source for C2–C5 on repositories
  that never contributed to the vocabulary. Lower is better. Whiskers are
  confidence intervals from resampling whole repositories (level in the
  chart title).
- **Say:** "Lower bar means fewer tokens for the same code; if two whiskers
  overlap we do not claim a difference."
- Optional: switch Eval set to **test_inorg** (new files in known
  repositories) to show the same chart where names are reused.

### 3:30 — 2x2 interaction

- **Click:** **Page → 2x2 interaction**, Eval set **test**.
- **Answers:** whether the custom vocabulary helps more (or less) when
  structure is on. The line is I = (C5 − C4) − (C3 − C2) per K with its CI;
  dotted line = zero, dashed line = K\*. The table's last column is the
  pre-registered decision rule: does C5 beat the better of C3 and C4 with a
  CI excluding zero?
- **Say:** "If the band crosses zero there is no evidence the ideas
  interact; below zero means names save more with structure on. 'Combined
  beats both' needs the last column to say True, and we report it either way."

### 5:30 — Context utilisation

- **Click:** **Page → Context utilisation**, Eval set **test**.
- **Answers:** for a fixed token budget, how many bytes of source fit
  (greedy packing of files in a fixed order, at K\*). Higher is better.
- **Say:** "Fewer tokens only matter if more code fits the window; this is
  that, measured in bytes, not claimed."

### 6:30 — Drift

- **Click:** **Page → Drift**.
- **Answers:** in repositories held out for drift, a growing share of
  definitions is renamed after the vocabulary was frozen. STATIC = frozen K\*
  vocabulary; REFRESH = a few appended entries (old ids unchanged). The y
  axis is token reduction against the C2/C4 baseline.
- **Say:** "This is how much of the saving survives when the codebase moves
  on, and whether a small append-only refresh changes it."

### 8:00 — Cost (only if systems ran)

- **Click:** **Page → Cost**. If it shows **Systems latency: not run**, say
  so in one sentence and move on.
- **Answers:** measured CPU forward latency per sequence length (if run) and
  the parameter / KV-cache overhead of the extra vocabulary, labelled
  ESTIMATE.
- **Say:** "The extra vocabulary has a cost in parameters; these rows are
  estimates, not measurements."

### 8:30 — Report

- **Click:** **Page → Report**. Scroll to **Hypotheses verdicts [test]** and
  **Limitations**. **Download report.md** if someone asks for it.
- **Answers:** H1–H5 verdicts at K\*, computed by the pre-registered rule in
  `configs/hypotheses.yaml`, and the generated limitations.
- **Say:** "Every chart came from this file; the verdicts are computed by a
  rule written before the run, and negative ones are printed as they are."

### 9:30 — Close: what was and was not measured

Measured (in this run):

- byte-exact round-trip of every evaluated file, per condition;
- tokens per KB with repository-level CIs on new repositories and on new
  files in known repositories;
- the 2×2 interaction and the decision rule, at K\* chosen on val;
- context-window packing, and drift with a frozen vs refreshed vocabulary;
- CPU latency only if the Cost page shows it.

Not measured:

- model quality (bits per byte): the new embedding rows were never trained,
  so no quality number is shown for C3/C5;
- real fine-tuning, downstream tasks (completion, retrieval);
- C1 (GPT-2) and C6 (bytes) are not in this demo (intrinsic baselines only);
- runtime equivalence of the renamed code (it is checked as syntax, not
  behaviour); names are synthetic, not a real company's code.

Say: "So: lossless and measured on token counts; whether the model gets
better at the code is the next experiment, not a claim we make today."

## If it breaks

| symptom | do this |
|---------|---------|
| `ERROR: no .venv found` | `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'` |
| `run directory not found: runs/<hash>` | `unzip -o demo_bundle.zip -d .`, or `TOKLAB_DEFAULT_RUN=<hash> bash scripts/demo.sh` |
| `something is already listening on port 8000` | `API_PORT=8001 bash scripts/demo.sh` (the UI follows automatically) |
| `API did not answer GET /datasets ... within 20s` | read the uvicorn output above the error; try `toklab serve-api` by hand |
| Streamlit port busy | `UI_PORT=8502 bash scripts/demo.sh` |
| browser did not open | go to <http://localhost:8501> by hand |
| sidebar says **API not reachable** | the API died: Ctrl-C and re-run `bash scripts/demo.sh` |
| Preview hangs or errors on first click | tokenizer not cached and no internet: skip Preview, start at Comparison |
| Preview note: "no adapted vocabulary loaded, so C3/C5 equal C2/C4" | the sidebar experiment id is empty; paste `demo_run` from `configs/demo.yaml` |
| a page says "not run" | that stage was not run for this run; say so, it is by design |
| `API error 404 ... not found` | wrong experiment id in the sidebar; use `demo_run` from `configs/demo.yaml` |
| anything else | switch to the `demo_assets/report.html` tab and present from the report |
