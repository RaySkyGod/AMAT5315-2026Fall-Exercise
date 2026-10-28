# Week 8: Language Models and the Transformer

AMAT5315 (Fall 2026) week 8 exercises: directions of meaning inside a
pretrained embedding table (potion-base-8M, shuffled-row control), count
models on TinyStories (n-gram ladder with add-k smoothing, exact
normalization, chi-squared sampling check, curse of dimensionality),
key-value memory stores and the one-token MLP (capacity ~d, ~2d, ~d^p; the
MLP stores the bigram table), attention as a memory written by the text
(softmax store recalls a 32x32 picture of 「算」), a two-layer pre-norm
transformer trained on TinyStories with exact-target samplers, and a
replication refit of the Chinchilla scaling law with bootstrap errors and
the compute-optimal split.

Environment (uv, CPU baseline):

```bash
uv sync                       # creates .venv from uv.lock
uv run pytest tests/          # all unit tests
uv run python scripts/<name>.py
```

Figures/tables live in `evidence/` (in git); raw runs, weights and caches in
`artifacts/` (not in git); downloads in `data/` and `inputs/` (not in git).

## Inputs and provenance

All pinned downloads (fetched via `https://hf-mirror.com` mirror on this
machine; the huggingface.co endpoints are the canonical addresses):

- TinyStories validation split (22,502,601 bytes, 27,630 stories; sheet
  command verbatim, canonical host `huggingface.co`):

  ```bash
  mkdir -p data
  base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve
  rev=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
  curl -L -o data/TinyStoriesV2-GPT4-valid.txt "$base/$rev/TinyStoriesV2-GPT4-valid.txt"
  ```

- Byte-pair encoding `tok512.model` (V=512, eos `</s>` id 2) from
  `karpathy/tinyllamas` revision `0bd21da`, folder `stories260K`, saved as
  `inputs/tinyllamas/tok512.model`:

  ```bash
  curl -L -o inputs/tinyllamas/tok512.model \
    https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model
  ```

- Model2Vec embedding table `minishlab/potion-base-8M` revision `bf8b056`
  (`model.safetensors`, 30,236,760 bytes, tensor `embeddings` 29,528 x 256
  float32, loaded with safetensors/numpy without PyTorch; `tokenizer.json`
  wordpiece vocab, e.g. `cat` -> 3943, `kitten` -> 17407):

  ```bash
  curl -L -o inputs/potion-base-8M/model.safetensors \
    https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors
  curl -L -o inputs/potion-base-8M/tokenizer.json \
    https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json
  ```

- Chinchilla training runs (245 rows, columns `Model Size`, `Training
  FLOP`, `loss`) from `epoch-research/analyzing-chinchilla` revision
  `92258837425e1b5f2851d624287f0120583a3d0e`, sheet command verbatim:

  ```bash
  base=https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla
  rev=92258837425e1b5f2851d624287f0120583a3d0e
  curl -L -o data/chinchilla-runs.csv "$base/$rev/data/svg_extracted_data.csv"
  ```

## Evidence

(populated by later tasks)
