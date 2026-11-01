# 004 — Second evaluation set `test_inorg` (new files in a known codebase)

Status: accepted. Changes the evaluation setup **after** seeing results of run
`0250decc37f179b589bec3bf9f1f190b84bdc77f12fad58d10b6879bae319a0f`, which is
now labelled **preliminary v1** in its report (`configs/run_labels.yaml`).

`configs/hypotheses.yaml` was **NOT** changed. Hypotheses, contrasts and the
decision rule are identical before and after this decision.

## What we saw in run 0250decc (preliminary v1)

All numbers below are copied from `runs/0250decc.../report.md` and from
`scripts/diagnose_vocab.py` output (`logs/taskA_vocab_diag_*.log`).

- Drift table: STATIC vocabulary |V| = 8, REFRESH |V| = 12; STATIC_C5
  coverage 0.004, STATIC_C3 coverage 0.000; token reduction 0.000 for every
  C3/C5 row at every drift proportion.
- 2×2 table: C3 = C2 = 247.49 tokens/KB and C5 = C4 = 307.56 tokens/KB at
  both K = 0 and K = 8.
- Fidelity: C2, C3, C4 and C5 each failed round-trip on 1 of 1022 test files,
  so every hypothesis verdict is "inconclusive" (INVALID grid). Not
  investigated in this decision.

Diagnosis (adapt split):

| corpus | mode | distinct identifiers | occurrences | pass min_freq=1, min_repos=1 | positive gain | vocab at K=8 |
|---|---|---|---|---|---|---|
| data/synthetic | structural | 52790 | 1940467 | 52790 | 48230 | 8 |
| data/synthetic | native | 76457 | 1509746 | 76457 | 65469 | 8 |
| data/repos | structural | 86502 | 1940467 | 86502 | 81664 | 8 |

Why the adapted vocabulary had no effect:

1. The K sweep in `configs/experiments/mvp_py.yaml` is `[0, 8]`: |V| = 8 is
   the configured cap, not the outcome of filtering. REFRESH adds
   `--k-refresh 4` (default), giving 12.
2. `min_freq` and `min_repos` are already 1, so neither removes any
   candidate. `min_repos` is **not** the limiting filter; we therefore did
   not change it, and we did not add K selection on val.
3. The runner reads `data.repos_dir = data/repos` (original sources); the
   synthetic layer in `data/synthetic` only enters the run hash. The
   enterprise names were never tokenized in this run.
4. With min_repos = 1 the top-gain candidates are dominated by names
   specific to single large adapt repositories (e.g. `map_imperatively`,
   `dialect`, `assert_compile`: 1 repo each), which a repo-level test split
   (new codebases) cannot contain.
5. Native (C3) mining emits truncated identifier fragments (e.g.
   `ome_rather_long_text_value`, `archCatalogAccountController`) because the
   joiner skips a leading space-prefixed pre-token; such entries rarely match
   at encode time. Recorded here; not fixed by this decision.

Test-split ceiling (share of test identifier occurrences whose string also
occurs in adapt, structural mining): 0.8184 on data/synthetic, 0.7671 on
data/repos. By distinct identifiers: 0.3924 and 0.1669.

## Decision

Add a second, clearly labelled evaluation set:

- `test` (unchanged, primary): held-out **repositories**, "new codebase".
- `test_inorg` (new, secondary): a seeded 20 % of the **files** of every adapt
  repository (`inorg_holdout_fraction: 0.20` in `configs/ingest.yaml`),
  "new files in a known codebase", which is the setting where a
  company-specific vocabulary is expected to help.

Leakage controls:

- `test_inorg` files get split `test_inorg`, so `ManifestView("adapt")`,
  which is the only input that mining/selection accept, never contains them.
- Duplicate clusters (exact and MinHash near-duplicates) move as one unit,
  so no cluster spans adapt (mining) and `test_inorg`.
- Selection is seeded per repository (`inorg:{seed}:{repo}`); each repo keeps
  at least one adapt file.

Every per-file metric (fidelity, token efficiency with CI, 2×2 factorial,
context utilisation, hypothesis verdicts) is reported for both sets, in
separate labelled sections. Hypothesis claims use `test` first.

## Why change the setup after seeing results

Run 0250decc showed that a repo-level test split measures only transfer of a
vocabulary to *unseen* codebases, which is the hardest case and not the main
use case of a company-specific vocabulary (new code in the company's own
repositories). Adding `test_inorg` is a change of evaluation design made
after observing results. To limit the risk of tuning on the outcome:

- the repo-level `test` set and all hypotheses are kept unchanged;
- `test_inorg` is reported in addition to `test`, never instead of it;
- run 0250decc keeps its artifacts and is labelled "preliminary v1".

## Deviation from golden rule 2

Golden rule 2 says "split data by REPOSITORY (never by file)". `test_inorg`
is an explicit, file-level exception requested for this research question.
It is confined to a separate, labelled evaluation set; no vocabulary or
selection input ever includes its files, and the primary `test` set remains
repository-level.

## Risks / limitations

- Overfitting: with `min_repos = 1`, entries frequent in one adapt repo are
  selected; they help `test_inorg` (same repos) and inflate in-org gains
  relative to `test`. K is not chosen on val.
- `test_inorg` shares authors, style and naming with adapt. Gains there do
  not imply gains on new codebases.
- Not run yet: no results exist for `test_inorg`. The next pipeline run must
  rebuild the manifest (`toklab data build`), because `test_inorg` is assigned
  at build time.
