# Week 7: Diffusion and Flow Matching

AMAT5315 (Fall 2026) week 7 exercises: DDPM on a four-cluster Gaussian
mixture (forward diffusion, time-conditioned noise predictor, 1000-step
sampling), deterministic DDIM with selectable level grids plus the mixture's
exact conditional-mean noise, flow matching with a learned velocity field and
a three-sampler step-count comparison, generation with the downloaded
pretrained CIFAR-10 noise predictor in JAX, and RePaint inpainting.

Environment (uv, CPU baseline):

```bash
uv sync                       # creates .venv from uv.lock (deps pinned to week6 versions)
uv run pytest tests/          # all unit tests
uv run python scripts/<name>.py
```

Figures/animations live in `evidence/` (in git); raw runs and settings in
`artifacts/` (not in git); downloads in `inputs/` (not in git).

## Inputs and provenance

Pinned model download (sheet command, revision
`267b167dc01f0e4e61923ea244e8b988f84deb80`, files `config.json` +
`diffusion_pytorch_model.safetensors`, 143,020,084 bytes; fetched via
`HF_ENDPOINT=https://hf-mirror.com` on a slow link):

```bash
uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \
  config.json diffusion_pytorch_model.safetensors \
  --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
  --local-dir inputs/cifar-model
```

Test images (sheet command; `inputs/images.npz`, 36,444 bytes; key `test`
holds the four 32x32 RGB images stored CHW float32 in [-1, 1]; other keys:
`validation`, `test_ids`, `validation_ids`, `labels`, `validation_labels`):

```bash
curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz \
  -o inputs/images.npz
```

## Evidence

| Figure | Command |
|---|---|
| `evidence/part1/loss_history.png` | `uv run python scripts/part1_train.py` |
| `evidence/part1/forward_diffusion.gif`, `forward_stats.txt` | `uv run python scripts/part1_forward.py` |
| `evidence/part1/stages.gif` | `uv run python scripts/part1_stages.py` |
| `evidence/part1/generate.gif`, `generate_final.png`, `mode_stats.txt` | `uv run python scripts/part1_generate.py` |
| `evidence/part2/determinism.png`, `samples_compare.png`, `max_diff.txt` | `uv run python scripts/part2_determinism.py` |
| `evidence/part3/flow_loss_history.png` | `uv run python scripts/part3_train_flow.py` |
| `evidence/part3/velocity_fields.png`, `velocity_rms.txt` | `uv run python scripts/part3_fields.py` |
| `evidence/part3/steps_sweep.png`, `steps_table.txt` | `uv run python scripts/part3_compare.py` |
| `evidence/part4/cifar_stages.png`, `cifar_final.png`, `cifar_vs_real.png` | `uv run python scripts/part4_generate.py` |

## Training settings

(filled in as tasks complete)
