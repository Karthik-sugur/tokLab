> **Snapshot** of the generated report for run `a63c884760b93db98798e5e5e6dadba5be8653ae956b66cd4ef9f011908c3403` (first 8 chars: `a63c8847`). Copied after the audit follow-up; not re-computed in this file. Live copy: `runs/<hash>/report.md` (git-ignored).

# TokLab report: mvp_py

**config_hash:** `a63c884760b93db98798e5e5e6dadba5be8653ae956b66cd4ef9f011908c3403`  
**run_dir:** `/Users/lenovo/tokLab/tokLab/runs/a63c884760b93db98798e5e5e6dadba5be8653ae956b66cd4ef9f011908c3403`

## Setup

| Field | Value |
|-------|-------|
| seed | 0 |
| git_sha | 908a35c1d98fda3753800af89e995eef51cabd85 |
| python | 3.11.17 |
| platform | macOS-26.5.2-arm64-arm-64bit |
| cpu | arm |
| gpu | cpu |
| manifest_hash | `7952b8287359369a81de27f9957a4cc77a034103781e28ebd81a0aacb9cd9c70` |
| vocab_hash | `f7ce395681e2fa83d4252d8dac4ba8ad424da855ca271a9bb3364cdf578c376f` |
| model | Qwen/Qwen2.5-Coder-0.5B |

Library versions: libcst=1.9.0, numpy=2.4.6, pandas=3.1.0, tokenizers=0.23.3, torch=2.14.1, transformers=5.20.2, tree_sitter=0.26.0

## Corpus

| Field | Value |
|-------|-------|
| kind | **synthetic** |
| dir | `data/synthetic` |
| manifest | `data/processed/manifest.parquet` |
| manifest_hash | `7952b8287359369a81de27f9957a4cc77a034103781e28ebd81a0aacb9cd9c70` |
| synthetic_hash | `47e7fc822111585a7b8d9b0872a0c30c7e77e632d3dc98000da61810ed5c08e0` |

| split | files | repos |
|-------|-------|-------|
| adapt | 2790 | 17 |
| drift_holdout | 399 | 3 |
| test | 1021 | 5 |
| test_inorg | 696 | 17 |
| val | 440 | 3 |

Synthetic layer (from `synth_manifest.parquet`):

| split | variant | files | renamed | copied unchanged | unchanged reasons |
|-------|---------|-------|---------|------------------|-------------------|
| adapt | adapt_lexicon | 2790 | 2431 | 359 | ast_parse_error=21, libcst_parse_error=2, no_eligible_identifiers=336 |
| drift_holdout | drift_p0 | 399 | 0 | 399 | proportion_0=399 |
| drift_holdout | drift_p10 | 399 | 195 | 204 | no_eligible_identifiers=204 |
| drift_holdout | drift_p25 | 399 | 295 | 104 | no_eligible_identifiers=104 |
| drift_holdout | drift_p50 | 399 | 324 | 75 | no_eligible_identifiers=75 |
| test | adapt_lexicon | 1021 | 950 | 71 | no_eligible_identifiers=71 |
| test_inorg | adapt_lexicon | 696 | 595 | 101 | ast_parse_error=6, no_eligible_identifiers=95 |
| val | adapt_lexicon | 440 | 416 | 24 | no_eligible_identifiers=24 |

## K sweep and K selection

Rule (docs/decisions/005-k-selection.md): argmin over K; ties -> smaller K; K with any C3 round-trip failure on the selection split is not eligible. Selection
split: `val` (440 files), condition
C3 only, metric tokens_per_kb (mean over files, lower is better).
Nominal K values [0, 256, 1024, 4096, 16384]; positive-gain candidates
native=65239, structural=42175;
swept K values [0, 256, 1024, 4096, 16384].

| K | C3 vocab size | val tokens/KB | C3 round-trip failures | eligible |
|---|---------------|---------------|------------------------|----------|
| 0 | 0 | 237.5674 | 0 | True |
| 256 | 256 | 232.0908 | 0 | True |
| 1024 | 1024 | 228.1932 | 0 | True |
| 4096 | 4096 | 224.3779 | 0 | True |
| 16384 (K*) | 16384 | 221.9153 | 0 | True |

**K\* = 16384** (ok).
Hypotheses below are judged at K* only; other K rows are descriptive.

## Evaluation sets

| eval set | role | description | status |
|----------|------|-------------|--------|
| `test` | primary | new codebase: held-out repositories (repo-level split, never seen by vocabulary mining) | 1021 files |
| `test_inorg` | secondary | new files in known codebases: seeded fraction of files held out from each adapt repository (excluded from vocabulary mining) | 696 files |

## Method notes (CIs and hypotheses)

- Cluster bootstrap: n_boot=50, alpha=0.05 (95% percentile CI), repositories resampled with replacement.
- H1–H3 compare each condition's separate CI with C2's CI (overlap): supported only if the condition's CI lies entirely below C2's CI (and the effect estimate is negative); not supported if entirely above; otherwise inconclusive. H4 and H5 use paired per-repository difference CIs (repository-weighted).
- configs/hypotheses.yaml states the claims and the H4 decision rule but does not specify whether contrasts use overlapping separate CIs or paired per-repository CIs; the report implements the overlap rule for H1–H3 and paired CIs for H4/H5.
- With 5 test repositories there are only 126 distinct bootstrap resamples (combinations with replacement); CIs are therefore coarse.

# Eval set `test`: new codebase: held-out repositories (repo-level split, never seen by vocabulary mining)

## Fidelity [test]

| K | condition | n_files | roundtrip_ok | roundtrip_fail | status |
|---|-----------|---------|--------------|----------------|--------|
| 0 | C2 | 1021 | 1021 | 0 | **OK** |
| 0 | C3 | 1021 | 1021 | 0 | **OK** |
| 0 | C4 | 1021 | 1021 | 0 | **OK** |
| 0 | C5 | 1021 | 1021 | 0 | **OK** |
| 256 | C2 | 1021 | 1021 | 0 | **OK** |
| 256 | C3 | 1021 | 1021 | 0 | **OK** |
| 256 | C4 | 1021 | 1021 | 0 | **OK** |
| 256 | C5 | 1021 | 1021 | 0 | **OK** |
| 1024 | C2 | 1021 | 1021 | 0 | **OK** |
| 1024 | C3 | 1021 | 1021 | 0 | **OK** |
| 1024 | C4 | 1021 | 1021 | 0 | **OK** |
| 1024 | C5 | 1021 | 1021 | 0 | **OK** |
| 4096 | C2 | 1021 | 1021 | 0 | **OK** |
| 4096 | C3 | 1021 | 1021 | 0 | **OK** |
| 4096 | C4 | 1021 | 1021 | 0 | **OK** |
| 4096 | C5 | 1021 | 1021 | 0 | **OK** |
| 16384 | C2 | 1021 | 1021 | 0 | **OK** |
| 16384 | C3 | 1021 | 1021 | 0 | **OK** |
| 16384 | C4 | 1021 | 1021 | 0 | **OK** |
| 16384 | C5 | 1021 | 1021 | 0 | **OK** |

## Token efficiency [test] (tokens/KB, lower is better)

Cluster-bootstrap mean over files with a CI from resampling repositories.
C2 and C4 do not depend on K; they are repeated so each K is a full 2×2.
Rows for INVALID conditions are listed but excluded from claims.

| K | condition | mean | CI low | CI high | n_repos | notes |
|---|-----------|------|--------|---------|---------|-------|
| 0 | C2 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 0 | C3 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 0 | C4 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 0 | C5 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 256 | C2 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 256 | C3 | 236.5709 | 226.1159 | 246.5647 | 5 | — |
| 256 | C4 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 256 | C5 | 291.0199 | 277.5866 | 303.3725 | 5 | — |
| 1024 | C2 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 1024 | C3 | 232.9346 | 223.1005 | 242.3466 | 5 | — |
| 1024 | C4 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 1024 | C5 | 286.9173 | 273.9450 | 298.8803 | 5 | — |
| 4096 | C2 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 4096 | C3 | 229.7570 | 220.1662 | 238.8127 | 5 | — |
| 4096 | C4 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 4096 | C5 | 282.7932 | 270.1237 | 294.4230 | 5 | — |
| 16384 (K*) | C2 | 240.9745 | 230.4639 | 251.1042 | 5 | — |
| 16384 (K*) | C3 | 227.6532 | 218.5207 | 236.0481 | 5 | — |
| 16384 (K*) | C4 | 296.9584 | 283.3555 | 309.6778 | 5 | — |
| 16384 (K*) | C5 | 280.1440 | 267.9275 | 291.4435 | 5 | — |

## 2×2 interaction [test] (structure × domain)

Pre-registered decision rule (quoted from configs/hypotheses.yaml): "Combined beats either alone on a metric only if C5 beats the better of C3/C4 with a CI excluding zero."

**Interaction summary:** Super-additive interaction (CI excludes zero): I=-4.3771, 95% CI [-6.9985, -3.0628] at K=16384. On tokens/KB the domain vocabulary saves more tokens when structure is on. Decision rule not satisfied: C5 is not better than the better of C3/C4 with a CI excluding zero.

| K | C2 | C3 | C4 | C5 | effect_S | effect_D | I | I CI | C5 better than best of C3/C4 (CI excl. 0) |
|---|----|----|----|----|----------|----------|---|------|------------------------------|
| 0 | 240.97 | 240.97 | 296.96 | 296.96 | 56.70 | 0.00 | 0.0000 | [0.000, 0.000] | False |
| 256 | 240.97 | 236.57 | 296.96 | 291.02 | 56.70 | -4.15 | -1.9836 | [-3.143, -1.335] | False |
| 1024 | 240.97 | 232.93 | 296.96 | 286.92 | 56.70 | -7.86 | -2.3101 | [-3.155, -1.868] | False |
| 4096 | 240.97 | 229.76 | 296.96 | 282.79 | 56.70 | -10.83 | -3.1162 | [-4.068, -2.623] | False |
| 16384 (K*) | 240.97 | 227.65 | 296.96 | 280.14 | 56.70 | -12.13 | -4.3771 | [-6.998, -3.063] | False |

**How these columns are averaged:** C2–C5 are the cell means of the Token efficiency table: the mean over all files, each file weighted equally. effect_S (C4−C2), effect_D (C3−C2) and I = (C5−C4) − (C3−C2) are the mean over the 5 repositories of per-repository differences of repository means, each repository weighted equally, with cluster-bootstrap CIs over repositories. They therefore differ from differences of the printed cell means: at K=16384 the cell means give D=-13.3214, S=+55.9839, I=-3.4930; the table shows D=-12.1257, S=+56.6984, I=-4.3771. The decision-rule column uses the same repository-weighted paired difference (C5 minus the better of C3/C4, the better one picked by the cell means) and is True only if that difference and its whole CI are below zero.

## Context utilisation [test]

Greedy first-fit packing in fixed `file_id` order at K=16384.
``files_fit`` is not monotonic in the budget because packing is greedy
first-fit in that fixed order: a larger budget can admit an earlier large
file that blocks later small ones. Read ``bytes_fit`` and ``tokens_used``.

| condition | budget | files_fit | bytes_fit | tokens_used |
|-----------|--------|-----------|-----------|-------------|
| C2 | 64 | 2 | 286 | 62 |
| C2 | 128 | 4 | 569 | 125 |
| C2 | 256 | 2 | 987 | 255 |
| C2 | 512 | 4 | 2073 | 510 |
| C3 | 64 | 2 | 298 | 64 |
| C3 | 128 | 5 | 605 | 128 |
| C3 | 256 | 2 | 1116 | 256 |
| C3 | 512 | 5 | 2207 | 510 |
| C4 | 64 | 4 | 218 | 59 |
| C4 | 128 | 4 | 457 | 127 |
| C4 | 256 | 4 | 899 | 252 |
| C4 | 512 | 3 | 1785 | 508 |
| C5 | 64 | 2 | 286 | 64 |
| C5 | 128 | 2 | 521 | 124 |
| C5 | 256 | 1 | 973 | 255 |
| C5 | 512 | 4 | 1972 | 510 |

## Hypotheses verdicts [test] (primary) at K*=16384
### H1 [test] — not supported

Structure alone (C4) reduces tokens relative to the code tokenizer (C2).

- contrast: C4 vs C2
- verdict: **not supported**
- reason: C4 tokens/KB CI entirely above C2

### H2 [test] — inconclusive

Domain-adapted vocabulary alone (C3) reduces tokens relative to C2.

- contrast: C3 vs C2
- verdict: **inconclusive**
- reason: C3 vs C2 CI overlap or zero not excluded

### H3 [test] — not supported

Combined (C5) reduces tokens relative to C2.

- contrast: C5 vs C2
- verdict: **not supported**
- reason: C5 tokens/KB CI entirely above C2

### H4 [test] — not supported

Combined beats the better of structure-alone or domain-alone (decision_rule).

- contrast: C5 vs max(C3, C4)
- verdict: **not supported**
- reason: C5 does not beat better of C3/C4 with CI excluding zero

### H5 [test] — supported

The structure×domain interaction is non-zero (super-/sub-additivity vs independence expectation 1-(1-r_S)(1-r_D) for token reduction).

- contrast: interaction I
- verdict: **supported**
- reason: interaction CI excludes zero (I=-4.3771, [-6.9985,-3.0628])

# Eval set `test_inorg`: new files in known codebases: seeded fraction of files held out from each adapt repository (excluded from vocabulary mining)

## Fidelity [test_inorg]

| K | condition | n_files | roundtrip_ok | roundtrip_fail | status |
|---|-----------|---------|--------------|----------------|--------|
| 0 | C2 | 696 | 696 | 0 | **OK** |
| 0 | C3 | 696 | 696 | 0 | **OK** |
| 0 | C4 | 696 | 696 | 0 | **OK** |
| 0 | C5 | 696 | 696 | 0 | **OK** |
| 256 | C2 | 696 | 696 | 0 | **OK** |
| 256 | C3 | 696 | 696 | 0 | **OK** |
| 256 | C4 | 696 | 696 | 0 | **OK** |
| 256 | C5 | 696 | 696 | 0 | **OK** |
| 1024 | C2 | 696 | 696 | 0 | **OK** |
| 1024 | C3 | 696 | 696 | 0 | **OK** |
| 1024 | C4 | 696 | 696 | 0 | **OK** |
| 1024 | C5 | 696 | 696 | 0 | **OK** |
| 4096 | C2 | 696 | 696 | 0 | **OK** |
| 4096 | C3 | 696 | 696 | 0 | **OK** |
| 4096 | C4 | 696 | 696 | 0 | **OK** |
| 4096 | C5 | 696 | 696 | 0 | **OK** |
| 16384 | C2 | 696 | 696 | 0 | **OK** |
| 16384 | C3 | 696 | 696 | 0 | **OK** |
| 16384 | C4 | 696 | 696 | 0 | **OK** |
| 16384 | C5 | 696 | 696 | 0 | **OK** |

## Token efficiency [test_inorg] (tokens/KB, lower is better)

Cluster-bootstrap mean over files with a CI from resampling repositories.
C2 and C4 do not depend on K; they are repeated so each K is a full 2×2.
Rows for INVALID conditions are listed but excluded from claims.

| K | condition | mean | CI low | CI high | n_repos | notes |
|---|-----------|------|--------|---------|---------|-------|
| 0 | C2 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 0 | C3 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 0 | C4 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 0 | C5 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 256 | C2 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 256 | C3 | 241.8749 | 218.4066 | 264.3409 | 17 | — |
| 256 | C4 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 256 | C5 | 292.0409 | 274.9030 | 311.7971 | 17 | — |
| 1024 | C2 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 1024 | C3 | 238.3405 | 214.7157 | 261.0887 | 17 | — |
| 1024 | C4 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 1024 | C5 | 288.2129 | 270.7988 | 308.1943 | 17 | — |
| 4096 | C2 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 4096 | C3 | 234.0889 | 210.2373 | 257.7496 | 17 | — |
| 4096 | C4 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 4096 | C5 | 283.0365 | 265.6216 | 302.8782 | 17 | — |
| 16384 (K*) | C2 | 245.7236 | 222.4757 | 267.6121 | 17 | — |
| 16384 (K*) | C3 | 229.5973 | 206.4919 | 253.0913 | 17 | — |
| 16384 (K*) | C4 | 297.0537 | 280.0656 | 316.1516 | 17 | — |
| 16384 (K*) | C5 | 278.6849 | 261.3923 | 298.1506 | 17 | — |

## 2×2 interaction [test_inorg] (structure × domain)

Pre-registered decision rule (quoted from configs/hypotheses.yaml): "Combined beats either alone on a metric only if C5 beats the better of C3/C4 with a CI excluding zero."

**Interaction summary:** Super-additive interaction (CI excludes zero): I=-3.1832, 95% CI [-4.6572, -2.1858] at K=16384. On tokens/KB the domain vocabulary saves more tokens when structure is on. Decision rule not satisfied: C5 is not better than the better of C3/C4 with a CI excluding zero.

| K | C2 | C3 | C4 | C5 | effect_S | effect_D | I | I CI | C5 better than best of C3/C4 (CI excl. 0) |
|---|----|----|----|----|----------|----------|---|------|------------------------------|
| 0 | 245.72 | 245.72 | 297.05 | 297.05 | 51.01 | 0.00 | 0.0000 | [0.000, 0.000] | False |
| 256 | 245.72 | 241.87 | 297.05 | 292.04 | 51.01 | -3.64 | -1.3603 | [-2.038, -0.873] | False |
| 1024 | 245.72 | 238.34 | 297.05 | 288.21 | 51.01 | -7.76 | -1.8252 | [-2.335, -1.354] | False |
| 4096 | 245.72 | 234.09 | 297.05 | 283.04 | 51.01 | -10.86 | -3.3590 | [-4.995, -2.329] | False |
| 16384 (K*) | 245.72 | 229.60 | 297.05 | 278.68 | 51.01 | -15.65 | -3.1832 | [-4.657, -2.186] | False |

**How these columns are averaged:** C2–C5 are the cell means of the Token efficiency table: the mean over all files, each file weighted equally. effect_S (C4−C2), effect_D (C3−C2) and I = (C5−C4) − (C3−C2) are the mean over the 17 repositories of per-repository differences of repository means, each repository weighted equally, with cluster-bootstrap CIs over repositories. They therefore differ from differences of the printed cell means: at K=16384 the cell means give D=-16.1263, S=+51.3302, I=-2.2426; the table shows D=-15.6525, S=+51.0088, I=-3.1832. The decision-rule column uses the same repository-weighted paired difference (C5 minus the better of C3/C4, the better one picked by the cell means) and is True only if that difference and its whole CI are below zero.

## Context utilisation [test_inorg]

Greedy first-fit packing in fixed `file_id` order at K=16384.
``files_fit`` is not monotonic in the budget because packing is greedy
first-fit in that fixed order: a larger budget can admit an earlier large
file that blocks later small ones. Read ``bytes_fit`` and ``tokens_used``.

| condition | budget | files_fit | bytes_fit | tokens_used |
|-----------|--------|-----------|-----------|-------------|
| C2 | 64 | 3 | 309 | 63 |
| C2 | 128 | 2 | 563 | 127 |
| C2 | 256 | 5 | 945 | 255 |
| C2 | 512 | 6 | 2159 | 508 |
| C3 | 64 | 3 | 319 | 60 |
| C3 | 128 | 3 | 680 | 128 |
| C3 | 256 | 5 | 1081 | 256 |
| C3 | 512 | 7 | 2443 | 508 |
| C4 | 64 | 2 | 258 | 62 |
| C4 | 128 | 2 | 324 | 128 |
| C4 | 256 | 3 | 770 | 254 |
| C4 | 512 | 5 | 1952 | 506 |
| C5 | 64 | 3 | 266 | 58 |
| C5 | 128 | 1 | 512 | 127 |
| C5 | 256 | 4 | 844 | 253 |
| C5 | 512 | 7 | 2193 | 512 |

## Hypotheses verdicts [test_inorg] (secondary) at K*=16384
### H1 [test_inorg] — not supported

Structure alone (C4) reduces tokens relative to the code tokenizer (C2).

- contrast: C4 vs C2
- verdict: **not supported**
- reason: C4 tokens/KB CI entirely above C2

### H2 [test_inorg] — inconclusive

Domain-adapted vocabulary alone (C3) reduces tokens relative to C2.

- contrast: C3 vs C2
- verdict: **inconclusive**
- reason: C3 vs C2 CI overlap or zero not excluded

### H3 [test_inorg] — inconclusive

Combined (C5) reduces tokens relative to C2.

- contrast: C5 vs C2
- verdict: **inconclusive**
- reason: C5 vs C2 CI overlap or zero not excluded

### H4 [test_inorg] — not supported

Combined beats the better of structure-alone or domain-alone (decision_rule).

- contrast: C5 vs max(C3, C4)
- verdict: **not supported**
- reason: C5 does not beat better of C3/C4 with CI excluding zero

### H5 [test_inorg] — supported

The structure×domain interaction is non-zero (super-/sub-additivity vs independence expectation 1-(1-r_S)(1-r_D) for token reduction).

- contrast: interaction I
- verdict: **supported**
- reason: interaction CI excludes zero (I=-3.1832, [-4.6572,-2.1858])

# Run-wide results

## Descriptive paired differences (post-hoc, not the pre-registered rule)

Added after seeing the results; does **not** change any verdict. Each row is
the mean over repositories of (condition A − condition B) on per-repository
mean tokens/KB, with a cluster-bootstrap CI (n_boot=1000,
seed=0). ``repos_better`` counts repositories
where the difference is negative (fewer tokens/KB for A).

_Descriptive paired differences added after seeing results; not the pre-registered rule; does not change any verdict._

| eval set | K | contrast | mean | CI low | CI high | n_repos | repos_better | repos_worse | repos_tie |
|----------|---|----------|------|--------|---------|---------|--------------|-------------|-----------|
| test | 16384 | C3-C2 | -12.1257 | -14.7003 | -9.5511 | 5 | 5 | 0 | 0 |
| test | 16384 | C4-C2 | 56.6984 | 52.3944 | 61.5284 | 5 | 0 | 5 | 0 |
| test | 16384 | C5-C2 | 40.1956 | 37.0422 | 44.4132 | 5 | 0 | 5 | 0 |
| test | 16384 | C5-min(C3,C4) | 52.3213 | 47.0838 | 58.2400 | 5 | 0 | 5 | 0 |
| test_inorg | 16384 | C3-C2 | -15.6525 | -17.3345 | -14.0872 | 17 | 17 | 0 | 0 |
| test_inorg | 16384 | C4-C2 | 51.0088 | 47.4751 | 54.5328 | 17 | 0 | 17 | 0 |
| test_inorg | 16384 | C5-C2 | 32.1731 | 27.9547 | 36.6352 | 17 | 0 | 17 | 0 |
| test_inorg | 16384 | C5-min(C3,C4) | 47.8256 | 43.0185 | 52.0593 | 17 | 0 | 17 | 0 |

## Intrinsic baselines (not part of the 2×2)

Intrinsic baselines on the same eval text as the 2×2 (not part of the factorial grid). Conditions: C1 (GPT-2), C2 (code tokenizer), C6 (byte-level).

| condition | status | n_files | roundtrip_fail | mean tokens/KB | CI low | CI high | n_repos |
|-----------|--------|---------|----------------|----------------|--------|---------|---------|
| C1 | **OK** | 1021 | 0 | 412.6826 | 399.5962 | 419.8275 | 5 |
| C2 | **OK** | 1021 | 0 | 240.9745 | 230.4639 | 251.1042 | 5 |
| C6 | **OK** | 1021 | 0 | 1024.0000 | 1024.0000 | 1024.0000 | 5 |

## Systems cost (K*=16384)
Systems stage not run (pending: separate step `toklab run systems --run a63c884760b93db98798e5e5e6dadba5be8653ae956b66cd4ef9f011908c3403`).

## Vocab overhead per K (ESTIMATE)

delta_params / delta_bytes are **ESTIMATE** labels (not measured). Tokens
saved per KB = baseline mean tokens/KB minus condition mean tokens/KB on the
primary eval set (`test`); C3 vs C2, C5 vs C4.

| K | condition | delta_V | delta_params | delta_bytes | tokens saved per KB (test) |
|---|-----------|---------|--------------|-------------|------------------------------|
| 0 | C3 vs C2 | 0 | 0 | 0 | 0.0000 |
| 0 | C5 vs C4 | 0 | 0 | 0 | 0.0000 |
| 256 | C3 vs C2 | 256 | 229376 | 458752 | 4.4036 |
| 256 | C5 vs C4 | 256 | 229376 | 458752 | 5.9384 |
| 1024 | C3 vs C2 | 1024 | 917504 | 1835008 | 8.0399 |
| 1024 | C5 vs C4 | 1024 | 917504 | 1835008 | 10.0411 |
| 4096 | C3 vs C2 | 4096 | 3670016 | 7340032 | 11.2175 |
| 4096 | C5 vs C4 | 4096 | 3670016 | 7340032 | 14.1652 |
| 16384 (K*) | C3 vs C2 | 16384 | 14680064 | 29360128 | 13.3214 |
| 16384 (K*) | C5 vs C4 | 16384 | 14680064 | 29360128 | 16.8144 |

### KV cache / FLOPs at the reported K (ESTIMATE)

| seq_len | delta_V (C3) | delta_params | delta_bytes | kv_cache_bytes | flops/token |
|---------|--------------|--------------|-------------|----------------|-------------|
| 64 | 16384 | 14680064 | 29360128 | 786432 | 2.340e+08 |
| 128 | 16384 | 14680064 | 29360128 | 1572864 | 2.367e+08 |
| 512 | 16384 | 14680064 | 29360128 | 6291456 | 2.532e+08 |
| 1024 | 16384 | 14680064 | 29360128 | 12582912 | 2.753e+08 |
| 2048 | 16384 | 14680064 | 29360128 | 25165824 | 3.193e+08 |

## Vocabulary drift (drift_holdout repositories)

Proportions p ∈ {0, 10, 25, 50} (% of each file's eligible definitions renamed with their uses). The drift_holdout files keep their **original** names (they are not
lexicon-renamed like ``test`` / ``adapt``), so the drift baseline is not
comparable to ``test``. Baselines C2/C4; STATIC = frozen adapt
vocab (K=16384; drift texts: pre-generated (synthetic drift/p<P>)).
REFRESH = append-only new entries from a drifted v1 sample of
1 file, adding up to 4 entries (old ids unchanged). BPB only for gate-valid fine-tuned variants.
This does not show robustness to large-scale renaming.

Renamed shares on the drift_holdout texts (generated from the rename map and
the ``drift/p*`` files):

| p | files touched | renamed definitions | share of eligible defs | renamed occurrences | share of all identifier occurrences |
|---|---------------|---------------------|------------------------|---------------------|-------------------------------------|
| 10 | 195/399 | 394 / 4219 | 0.0934 | 926 / 142537 | 0.0065 |
| 25 | 295/399 | 1041 / 4219 | 0.2467 | 2293 / 142537 | 0.0161 |
| 50 | 324/399 | 2096 / 4219 | 0.4968 | 4469 / 142537 | 0.0314 |

| p | variant | baseline | n_tok | n_base | reduction | coverage | bpb | \|V\| |
|---|---------|----------|-------|--------|-----------|----------|-----|------|
| 0 | C2 | C2 | 885216 | 885216 | 0.000 | 0.000 | not run | 0 |
| 0 | C4 | C4 | 1056482 | 1056482 | 0.000 | 0.000 | not run | 0 |
| 0 | STATIC_C3 | C2 | 874494 | 885216 | 0.012 | 0.102 | not run | 16384 |
| 0 | STATIC_C5 | C4 | 1046952 | 1056482 | 0.009 | 0.101 | not run | 16384 |
| 0 | REFRESH_C3 | C2 | 874478 | 885216 | 0.012 | 0.102 | not run | 16388 |
| 0 | REFRESH_C5 | C4 | 1046933 | 1056482 | 0.009 | 0.101 | not run | 16386 |
| 10 | C2 | C2 | 888240 | 888240 | 0.000 | 0.000 | not run | 0 |
| 10 | C4 | C4 | 1059486 | 1059486 | 0.000 | 0.000 | not run | 0 |
| 10 | STATIC_C3 | C2 | 877543 | 888240 | 0.012 | 0.102 | not run | 16384 |
| 10 | STATIC_C5 | C4 | 1050012 | 1059486 | 0.009 | 0.101 | not run | 16384 |
| 10 | REFRESH_C3 | C2 | 877527 | 888240 | 0.012 | 0.102 | not run | 16388 |
| 10 | REFRESH_C5 | C4 | 1049993 | 1059486 | 0.009 | 0.101 | not run | 16386 |
| 25 | C2 | C2 | 892769 | 892769 | 0.000 | 0.000 | not run | 0 |
| 25 | C4 | C4 | 1064078 | 1064078 | 0.000 | 0.000 | not run | 0 |
| 25 | STATIC_C3 | C2 | 882038 | 892769 | 0.012 | 0.102 | not run | 16384 |
| 25 | STATIC_C5 | C4 | 1054558 | 1064078 | 0.009 | 0.101 | not run | 16384 |
| 25 | REFRESH_C3 | C2 | 882022 | 892769 | 0.012 | 0.102 | not run | 16388 |
| 25 | REFRESH_C5 | C4 | 1054539 | 1064078 | 0.009 | 0.101 | not run | 16386 |
| 50 | C2 | C2 | 899907 | 899907 | 0.000 | 0.000 | not run | 0 |
| 50 | C4 | C4 | 1071351 | 1071351 | 0.000 | 0.000 | not run | 0 |
| 50 | STATIC_C3 | C2 | 889201 | 899907 | 0.012 | 0.101 | not run | 16384 |
| 50 | STATIC_C5 | C4 | 1061893 | 1071351 | 0.009 | 0.100 | not run | 16384 |
| 50 | REFRESH_C3 | C2 | 889193 | 899907 | 0.012 | 0.101 | not run | 16388 |
| 50 | REFRESH_C5 | C4 | 1061887 | 1071351 | 0.009 | 0.100 | not run | 16386 |

## Limitations

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
