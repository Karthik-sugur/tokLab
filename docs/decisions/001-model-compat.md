# 001 — Model compatibility (spike 1)

## Question

Can we load a small causal code LM, inspect its tokenizer/embedding layout,
add new vocabulary rows via `resize_token_embeddings`, and still compute
causal LM loss on a Python snippet?

## Observed

From `runs/spikes/spike_1.out`:

```
Loaded model: Qwen/Qwen2.5-Coder-0.5B
model_name: Qwen/Qwen2.5-Coder-0.5B
license: unavailable
tokenizer_class: Qwen2Tokenizer
vocab_size (tokenizer): 151665
embedding_rows: 151936
embedding_dim: 896
tie_word_embeddings: True
hidden_size: 896
num_hidden_layers: 24
num_key_value_heads: 2
loss_before_resize: 0.8479969501495361
tokens_requested: 100
tokens_added: 100
embedding_rows_before_resize: 151936
embedding_rows_after_resize: 151765
input_embedding_shape: (151765, 896)
output_embedding_shape: (151765, 896)
input_output_embeddings_still_tied: True
loss_after_resize: 0.8479969501495361
```

FINDINGS block (same file):

```
- Loaded model Qwen/Qwen2.5-Coder-0.5B with tokenizer class Qwen2Tokenizer.
- Vocab size 151765; embedding rows after resize (151765, 896).
- tie_word_embeddings=True; after resize still_tied=True.
- Causal LM loss on snippet before=0.8479969501495361, after=0.8479969501495361.
- Added 100 dummy tokens; new rows initialized to mean of existing embeddings.
```

No fallback model was loaded; the preferred model succeeded on the first try.

## Decision

Use **`Qwen/Qwen2.5-Coder-0.5B`** as the base code model/tokenizer for
C2–C5 (already reflected in `configs/models/base_model.yaml`). Do not
switch to a fallback unless a later run fails to load this model.

Treat tied input/output embeddings as the expected post-resize state for
this checkpoint (`still_tied=True`).

## Open risks

- **License:** printed as `license: unavailable`; the spike did not
  establish redistribution or commercial-use terms.
- Stock `embedding_rows` (151936) exceeded tokenizer `vocab_size`
  (151665) before resize; after adding 100 tokens and resizing to
  `len(tok)`, rows became 151765. The spike did not explain the reserved
  rows or whether shrinking away unused rows matters for training.
- Loss was measured on one short snippet only; that does not prove
  quality after real domain adaptation.
- New embedding rows were mean-initialized; the spike did not evaluate
  whether that initialization is adequate for downstream quality (and
  golden rule 8 still requires suppressing quality metrics until rows
  are trained).
