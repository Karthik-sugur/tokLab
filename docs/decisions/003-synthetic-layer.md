# 003 — Synthetic rename layer (spike 3)

## Question

Can libcst rename top-level function/class names to long enterprise-style
identifiers while preserving structure (normalized `ast.dump` equality),
and what happens to token counts under the code-model tokenizer?

## Observed

From `runs/spikes/spike_3.out`:

```
Loaded tokenizer: Qwen/Qwen2.5-Coder-0.5B
modules: ['quopri.py', 'uu.py', 'imaplib.py']
```

**quopri.py:**

```
ast_dump_equal_after_normalising_identifiers: True
token_count_original (Qwen/Qwen2.5-Coder-0.5B): 1996
token_count_renamed  (Qwen/Qwen2.5-Coder-0.5B): 2142
token_count_delta: 146
```

**uu.py:**

```
ast_dump_equal_after_normalising_identifiers: True
token_count_original (Qwen/Qwen2.5-Coder-0.5B): 1719
token_count_renamed  (Qwen/Qwen2.5-Coder-0.5B): 1816
token_count_delta: 97
```

**imaplib.py:**

```
ast_dump_equal_after_normalising_identifiers: True
token_count_original (Qwen/Qwen2.5-Coder-0.5B): 13314
token_count_renamed  (Qwen/Qwen2.5-Coder-0.5B): 13408
token_count_delta: 94
```

FINDINGS block:

```
- Renamed top-level function/class names in 3 stdlib modules via libcst; see per-module mapping and ast equality above.
- Token counts use tokenizer Qwen/Qwen2.5-Coder-0.5B; deltas printed per module.
```

All three modules reported `ast_dump_equal_after_normalising_identifiers: True`.
All three showed **higher** token counts after rename (deltas +146, +97, +94).

## Decision

The libcst top-level rename approach is **good enough as a structure-preserving
synthetic stressor**: normalized AST equality held on all three printed
modules. Keep using rename-for-enterprise-names to create long identifiers
that a plain BPE must split.

The observed token-count **increases** without a custom vocabulary are
expected under that stress (long names cost more tokens). They do **not**
by themselves show that domain-adapted vocab will win; that comparison
belongs to later C3/C5 experiments. No change to the rename *goal* is
required based on this spike alone.

## Open risks

- Only top-level names in three stdlib modules were renamed; methods,
  nested defs, constants, and `__all__` handling were not proven here
  (later `toklab data synth` work is out of scope for this note’s evidence).
- Token deltas are for the base code tokenizer only; no adapted vocab was
  applied in this spike.
- AST equality after normalizing identifiers does not prove runtime or
  import-graph correctness of the renamed modules.
