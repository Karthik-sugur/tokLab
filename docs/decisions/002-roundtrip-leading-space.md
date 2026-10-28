# 002 — Round-trip, leading space, and add_tokens (spike 2)

## Question

Do the code-model tokenizer and GPT-2 round-trip real Python sources
losslessly? How do leading spaces affect token IDs? What happens when a
long identifier is added with `tokenizer.add_tokens`?

## Observed

From `runs/spikes/spike_2.out`:

**Round-trip on 100 stdlib `.py` files — both tokenizers, zero failures:**

```
code (Qwen/Qwen2.5-Coder-0.5B): 100/100 exact round-trips
code_failures: 0
gpt2 (gpt2): 100/100 exact round-trips
gpt2_failures: 0
```

**Leading spaces (code = Qwen):**

```
  ['foo'] ids=[7975]
  ['foo'] tokens=['foo']
  ['foo'] decode==input: True (decoded='foo')
  [' foo'] ids=[15229]
  [' foo'] tokens=['Ġfoo']
  [' foo'] decode==input: True (decoded=' foo')
  ['x = foo'] ids=[87, 284, 15229]
  ['x = foo'] tokens=['x', 'Ġ=', 'Ġfoo']
  ['x = foo'] decode==input: True (decoded='x = foo')
```

**Leading spaces (gpt2):** same pattern — `'foo'` → `foo`; `' foo'` and
`'x = foo'` use `Ġfoo` with a different id than bare `foo`:

```
  ['foo'] ids=[21943]
  ['foo'] tokens=['foo']
  [' foo'] ids=[22944]
  [' foo'] tokens=['Ġfoo']
  ['x = foo'] tokens=['x', 'Ġ=', 'Ġfoo']
```

**`add_tokens` long identifier (Qwen tokenizer):**

```
before_add tokens for 'AcmePaymentSettlementBatchProcessor': ['Ac', 'me', 'Payment', 'Set', 'tlement', 'Batch', 'Processor']
add_tokens returned: 1
after_add tokens for 'AcmePaymentSettlementBatchProcessor': ['AcmePaymentSettlementBatchProcessor']
  text='AcmePaymentSettlementBatchProcessor'
    tokens=['AcmePaymentSettlementBatchProcessor']
    contains_new_token_id(151665): True
  text='AcmePaymentSettlementBatchProcessorHelper'
    tokens=['AcmePaymentSettlementBatchProcessor', 'Helper']
    contains_new_token_id(151665): True
  text='preAcmePaymentSettlementBatchProcessor'
    tokens=['pre', 'AcmePaymentSettlementBatchProcessor']
    contains_new_token_id(151665): True
  text=' AcmePaymentSettlementBatchProcessor'
    tokens=['Ġ', 'AcmePaymentSettlementBatchProcessor']
    contains_new_token_id(151665): True
  text='x = AcmePaymentSettlementBatchProcessor()'
    tokens=['x', 'Ġ=', 'Ġ', 'AcmePaymentSettlementBatchProcessor', '()']
    contains_new_token_id(151665): True
  text='class AcmePaymentSettlementBatchProcessor:'
    tokens=['class', 'Ġ', 'AcmePaymentSettlementBatchProcessor', ':']
    contains_new_token_id(151665): True
```

Warnings also appeared about sequences longer than the model max length
while encoding large stdlib files; round-trip still succeeded for those
files.

## Decision

1. **`attach_leading_space` default = `True`.**  
   In normal code, both tokenizers encode a space-prefixed identifier as
   one `Ġ…` token (`'x = foo'` → `Ġfoo`, same id as `' foo'`). Encoding a
   bare identifier segment after a separate space gap would not match
   that pretrained unit. Merging a single leading space gap into the
   following segment is the setting that matches the observed BPE
   behavior.

2. **Do not rely on HF `add_tokens` matching for custom vocab.**  
   After `add_tokens`, the new id matched **inside larger words**
   (`…ProcessorHelper`, `preAcme…`) and **did not** produce a
   space-prefixed `ĠAcme…` form: leading space became a separate `Ġ`
   token plus the bare new token. Custom / adapted vocabulary lookup for
   C3/C5 must use our own **boundary-aware longest-match** (on
   structure-aware segments), not raw HF `add_tokens` substring matching.

3. Round-trip evidence supports using these HF tokenizers as lossless
   bases for C1/C2 when decode uses the same settings as the spike
   (`clean_up_tokenization_spaces=False` path used in the spike script).

## Open risks

- Round-trip was checked on 100 stdlib files only, not on enterprise /
  synthetic corpora.
- Max-length warnings were printed; the spike did not measure model
  forward-pass correctness on long files, only tokenizer round-trip.
- Only one long custom token was added; behavior with many custom tokens
  was not measured here (spike 1 added 100 dummy tokens for embeddings,
  not for matching).
- Whether `attach_leading_space=False` can still be lossless was not
  disproven — the spike only shows what pretrained BPE *does* with
  spaces, which drives the **default**, not a uniqueness proof.
