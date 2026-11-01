# 005 — K sweep and choice of K* (vocabulary size)

Status: accepted. Written **before** any run with the K sweep below; no
result of such a run existed when this rule was fixed. The only earlier run
(`0250decc…`, "preliminary v1") used K ∈ {0, 8} on `data/repos` and is not
used to choose K.

`configs/hypotheses.yaml` was **NOT** changed.

## The sweep

`configs/experiments/mvp_py.yaml`: `vocab.k_values: [0, 256, 1024, 4096, 16384]`.

Each value is capped at the number of positive-gain adapt candidates
(`gain = freq × (n_base_tokens − 1) > 0` after `min_freq` / `min_repos`).
The cap is the larger of the native (C3) and structural (C5) candidate
counts; each mode's vocabulary at K has `min(K, n_positive[mode])` entries.
Capping can merge K values; duplicates are dropped. The capped sweep is
recorded in `runs/<hash>/k_sweep.json`.

Candidates are mined once per mode from the ADAPT split only and sliced to
each K (top-K by gain, alphabetical tie-break), so the vocabulary at a
smaller K is a prefix of the vocabulary at a larger K.

## Choosing K*

- **Split:** `val` (`data.k_selection_split`). Never `test` or `test_inorg`.
- **Condition:** C3 only (domain vocabulary, no structure). C5 and the 2×2
  play no part, so the choice cannot favour the combined method.
- **Metric:** mean tokens/KB over val files (original UTF-8 bytes; lower is
  better). This is the point estimate the report uses for every condition.
- **Rule:** K* = the K with the lowest val tokens/KB. Ties go to the smaller K.
- **Eligibility:** a K whose C3 tokenizer fails the round-trip check on any
  val file is not eligible.
- **K = 0 is a candidate.** If no vocabulary size lowers val tokens/KB, K* = 0
  and C3/C5 equal C2/C4 at K*. That is reported as it is.
- **No val files / no eligible K:** K* is "not selected". Hypotheses are then
  reported as "not run". Descriptive tables use the largest K and say so.

The per-K val values, eligibility and K* are written to
`runs/<hash>/k_selection.json` and shown in the report.

## What uses K*

- **Hypotheses H1–H5** on `test` and `test_inorg` are judged at K* only.
- **Descriptive only:** the full sweep (token efficiency, 2×2 table, per-K
  vocab overhead) on `test` and `test_inorg`. No verdict is drawn from a
  K ≠ K* row.
- **Drift** (`toklab run drift`): STATIC = the saved K* vocabularies
  (`runs/<hash>/vocab/{native,structural}_k<K*>.json`).
- **Systems** (`toklab run systems`): C2–C5 at K*.
- **Context utilisation** and the KV / FLOPs estimate table: K*.
- **Preview** (`/tokenize/preview` with an experiment id): K* vocabularies.

## Why

A single large K, or the K that looks best on test, would tune the setup on
the data used to judge it. Choosing K on val with the domain-only condition
keeps the choice away from both the test sets and the combined method. Since
the vocabulary is still chosen on this corpus, the report lists "K chosen on
val" under limitations.
