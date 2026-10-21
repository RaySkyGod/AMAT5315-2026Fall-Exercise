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
| `evidence/part5/inpaint_gallery.png`, `mse_table.txt` | `uv run python scripts/part5_inpaint.py` |

All evidence (7 figures + 4 GIFs + tables) totals about 2.5 MB.

## Training settings (both networks: plain SGD, learning rate decaying linearly to zero)

| network | updates | batch | lr0 (pilot) | width | time features | final loss |
|---|---|---|---|---|---|---|
| noise predictor (Part 1) | 25000 | 512 | 0.3 | 3 FC layers of 128 (SiLU) | 32 sinusoidal of t/1000 | ~0.245 (reference ~0.25) |
| velocity network (Part 3) | 25000 | 512 | 0.1 | same architecture, fresh init | 32 sinusoidal of s | ~5.22 (= the conditional-variance floor, measured 5.229 by Monte Carlo) |

lr0 was chosen by 2000-update pilots; for the velocity regression lr0 >= 0.2
diverges (targets z - x0 have E||v||^2 ~ 11). Ten training stages (updates
0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 25000) are stored in
`artifacts/part1/noise_predictor.npz` with the loss history. The CIFAR-10
predictor is used as downloaded, with all weights fixed.

## Verification summary (sheet thresholds in parentheses)

- Part 1 forward: t=1000 coordinate means [-0.002, 0.058] (|.| < 0.1),
  variances [0.989, 0.947] (|.-1| < 0.1).
- Part 1 stages: initialization paths leave the frame (max |x| ~ 581); by
  update 100 all ten endpoints sit on the clusters; the loss falls in the
  first few hundred updates and levels off at ~0.245 (nonzero, as the
  conditional-mean identity predicts).
- Part 1 generation: populations max dev 0.007 (< 0.04), centres max dev
  0.032 (< 0.05), spreads max dev 0.021 (< 0.03).
- Part 2: two DDIM runs from the same noise differ by exactly 0; DDIM
  cluster fractions max dev 0.009 (< 0.015); cross-sampler same-cluster
  rate 0.262 (~1/4). Learned-DDIM widths span [0.300, 0.325], slightly
  above the reference band [0.29, 0.32] at the top; the exact-noise DDIM
  spans [0.294, 0.312] (inside).
- Part 3 velocity: relative RMS 0.0095 / 0.0172 / 0.0411 at s = 0.1/0.5/0.9
  (< 0.1; reference 0.011/0.017/0.058), measured over 200k marginal
  samples x_s per s. The metric follows the sheet's "grid over the
  samples"; a uniform grid also scores rarely-visited far-field points
  where the trained MLP is unconstrained.
- Part 3 steps: flow at 20 steps has energy distance -0.0001 (within the
  null; < 0.005) and width error 0.024 (< 0.03); 1-2 steps fail for every
  sampler, including the exact noise/velocity variants (collapse toward
  the mean). The 20-pair null 95% quantile is 0.0036 here (reference
  0.0013); a 100-pair check gives 0.0023 with 20-pair estimates ranging
  0.0004-0.0065, so both values are statistically consistent.
- Part 4: 1000-step DDPM on 4 images took 182 s (0.045 s/image/eval,
  reference ~0.05). Structure appears late: neighbor-pixel correlation
  0.003 at t=1000, 0.254 at t=200, 0.799 at t=50, 0.915 at t=0 (real
  CIFAR: 0.936). Noising real images with known noise, the downloaded
  predictor recovers it with correlation 0.96-1.00.
- Part 5: observed pixels change by exactly 0 (< 1e-5); mean hidden-pixel
  MSE 0.203 (reference ~0.24); outside the hole the four completions are
  bit-identical, inside they differ.

## Explanations

**Transport comparison (Part 3).** All three samplers move a Gaussian cloud
to the four clusters, but they buy accuracy differently. DDPM must walk
every level because each reverse step only justifies a variance-beta_t
noise around its posterior mean; skipping levels needs ~50 before its
widths settle. DDIM reuses the same predictor but jumps between selected
levels deterministically (Tweedie estimate re-noised to the destination
level), reaching the null scale by ~20 steps. Flow matching predicts the
motion itself, and Euler-following the learned marginal velocity reaches
the null band at ~20 steps (0.0015-scale energy distance); one or two steps
fail for every method because a single straight move cannot reproduce four
separated modes - the samples collapse toward the mean.

**Generated photographs (Part 4).** What the download already contains: the
35.7M weights of the U-Net noise predictor, i.e. the learned mapping
(x_t, t) -> eps_hat for CIFAR-10 photographs. What this exercise computes:
everything else - the variance schedule and cumulative coefficients, the
DDPM posterior mean, the 1000-step reverse loop that subtracts each step's
predicted noise share and re-injects variance-beta_t randomness, and the
initial pure-noise draw. No training or fine-tuning happens here; only the
sampler turns someone else's learned denoising knowledge into images.

**Inpainting (Part 5).** The observation fixes the visible border but says
nothing definite about the hidden square: many photographs agree with the
same border, so the posterior over the hole is genuinely spread out.
RePaint clamps the forward-noised observation into the sampler state after
every jump (making observed pixels exact by construction), while the hidden
region is filled from whatever the noise predictor proposes. Different
seeds therefore give different plausible completions - each consistent
with the border, none guaranteed to match the withheld truth - which is
exactly why the pixelwise-variance map lights up inside the hole and stays
dark outside, and why a plausible completion can still have a large
hidden-pixel MSE.
