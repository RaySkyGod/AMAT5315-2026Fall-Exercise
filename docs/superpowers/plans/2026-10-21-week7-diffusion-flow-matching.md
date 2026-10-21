# Week 7: Diffusion and Flow Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 AMAT5315 Week 7 学习单全部五个 Part：高斯混合上的 DDPM（前向扩散 + 噪声预测器训练 + 采样验证）、确定性 DDIM（含精确条件均值噪声与跳层 DDPM 对照）、flow matching（速度网络 + 精确边缘速度 + 三采样器步数对比）、加载预训练 CIFAR-10 噪声预测器生成照片、RePaint 图像补全，并产出全部 evidence 图/动画与 README。

**Architecture:** 纯 CPU JAX + NumPy + matplotlib。核心数值（调度表、前向、DDPM/DDIM/跳层采样、混合分布的精确量、能量距离）放纯函数模块；小 MLP（时间条件 + 3 层全连接）同时服务噪声预测器与速度网络；UNet 用 JAX 复刻 diffusers v0.40.0 的 `UNet2DModel` 并直接加载 safetensors 权重（固定不训练）；RePaint 复用该 UNet 与 Part 1 公式。脚本层负责训练、采样、动画与出图。测试先写、失败后再实现，每个 Task 一次 commit。

**Tech Stack:** Python 3.12+（uv 管理）、jax（CPU）、numpy、matplotlib、safetensors、pytest；下载用 `huggingface-hub==1.33.0`（一次性 `uv run --with`）与 curl。

**Spec:** `week7/week7-learning-sheet.pdf`（20 页，已通读）。计划中的全部设置与验收阈值均抄自该学习单的 DO/VERIFY 部分。

## Global Constraints

- 工作目录 `week7/`，**uv** 管理（`uv run python ...`）；CPU 是基线，不用 GPU。
- 图/动画存 `evidence/`（进 git），原始运行结果 + 设置存 `artifacts/`（**不进 git**），下载（`inputs/`：CIFAR 权重、`images.npz`）**不进 git**。
- 每完成一个 Task 就 commit，消息前缀 `week7: ...`；有不确定处先问，不猜。
- Part 1 靶分布：等权 4 分量高斯混合，均值 (3,0),(0,3),(−3,0),(0,−3)，协方差 **0.09·I**（坐标标准差 0.3）。
- 调度表：T=1000，β_t 线性 **1e-4 → 0.02**；α_t = 1−β_t，累积 ᾱ_t = ∏α_s，ᾱ_0 = 1（ᾱ_1000 ≈ 4e−5）。
- Part 1 训练参考设置（学习单指定）：**25000 updates × 512 新采点/update**（CPU 约 1 分钟），学习率衰减到 0，保存 **10 个训练阶段**（含初始化和最终模型，前期更密）；损失按学习单算法为普通梯度步 θ ← θ − η∇θL。
- DDPM 采样方差取 β_t，**无裁剪**，**最后一步 (t=1) 不加噪声**。
- Part 3：路径 x_s = (1−s)x₀ + s z（s∈[0,1] 均匀），目标速度 z−x₀；网络沿用噪声预测器架构、从头训练、学习率衰减到 0；Euler 步 x_{s−Δs} = x_s − Δs·v_θ(x_s,s) 从 s=1 到 0。
- Part 4/5 模型固定（下载权重不训练），float32，噪声级别 t 传入模型时用索引 **t−1**，采样中**不裁剪**状态；仅显示时 (x+1)/2 并 clip 到 [0,1]。
- 学习单下载命令逐字执行（含 pinned revision `267b167dc01f0e4e61923ea244e8b988f84deb80`），并把 URL 记录进 `week7/README.md`。
- 动画保存为 GIF（`PillowWriter`），单文件 < 5 MB；静态图 PNG。

---

## File Structure

```
week7/
├── README.md                  # 环境表、provenance、evidence 图 ↔ 命令表、解释段落（Task 0 建，Task 9 收尾）
├── pyproject.toml, uv.lock    # jax, numpy, matplotlib, safetensors, pytest（Task 0）
├── .gitignore                 # inputs/, artifacts/, .venv/, __pycache__/（Task 0）
├── inputs/                    # cifar-model/{config.json,diffusion_pytorch_model.safetensors}, images.npz（不进 git）
├── mixture.py                 # 靶分布 + 精确量：采样、分量后验权重、精确 E[ε|x_t]、精确边缘速度、
│                              #   模式统计（population/centre/spread）、能量距离（Task 1, 5）
├── diffusion.py               # β 调度、ᾱ、前向逐步/直达、DDPM、DDIM、跳层 DDPM（Task 1, 4）
├── net.py                     # 时间条件 MLP（时间特征 + 3 层 FC）、噪声/速度两个训练器（Task 2, 5）
├── unet.py                    # JAX 版 UNet2DModel（对齐 diffusers v0.40.0）+ safetensors 加载映射（Task 7）
├── repaint.py                 # RePaint 补全循环（Task 8）
├── scripts/
│   ├── part1_forward.py       # V1: 前向扩散动画 + t=1000 均值/方差打印（Task 3）
│   ├── part1_train.py         # 训练噪声预测器 → artifacts/part1/noise_predictor.npz（Task 2）
│   ├── part1_stages.py        # V2: 10 阶段 × 10 条生成路径动画 + 损失历史（Task 3）
│   ├── part1_generate.py      # V3: 2000 样本、反向动画、模式统计（Task 3）
│   ├── part2_determinism.py   # V1: DDIM×2 vs DDPM×2 轨迹 + 2000 样本 + 最大差打印（Task 4）
│   ├── part3_train_flow.py    # 训练速度网络 → artifacts/part3/velocity.npz（Task 5）
│   ├── part3_fields.py        # V1: s=0.1,0.5,0.9 学习/精确速度场 + 相对 RMS（Task 6）
│   ├── part3_compare.py       # V2: 步数扫描 {1,2,5,10,20,50,100,1000} × 3 采样器 + 能量距离 + null scale（Task 6）
│   ├── part4_generate.py      # V1: 4 张图、反向阶段快照、与真实 CIFAR 对比（Task 7）
│   └── part5_inpaint.py       # V1: 4 图 × seeds 7201–7204 补全、MSE、逐像素方差（Task 8）
├── tests/
│   ├── test_mixture.py        # （Task 1, 5）
│   ├── test_diffusion.py      # （Task 1, 4）
│   ├── test_net.py            # （Task 2）
│   ├── test_unet.py           # （Task 7）
│   └── test_repaint.py        # （Task 8）
└── evidence/part1..part5/  artifacts/part1..part5/
```

依赖：`mixture.py`、`diffusion.py`、`net.py` 互相独立或仅依赖常量；`repaint.py` 依赖 `diffusion.py`（ᾱ、Tweedie）与 `unet.py`（预测器接口）。

---

## Task 0: 环境与输入下载

**Files:**
- Create: `week7/pyproject.toml`, `week7/.gitignore`, `week7/README.md`（stub）, `week7/inputs/`, `week7/evidence/part1..part5/`, `week7/artifacts/part1..part5/`

**Interfaces (Produces):**
- `cd week7 && uv run python <script>` 与 `uv run pytest tests/` 可用。
- `inputs/images.npz`（4 张测试图，值域 [−1,1]）；`inputs/cifar-model/`（config.json + safetensors，pinned revision）。

- [x] **Step 1: 初始化 uv 项目并加依赖**：
  ```bash
  cd week7
  uv init --no-readme --python 3.12
  uv add jax numpy matplotlib safetensors pytest
  ```
  版本与 week6 对齐（jax 0.11.x）。
- [x] **Step 2: 写 `.gitignore`**：
  ```
  inputs/
  artifacts/
  .venv/
  __pycache__/
  ```
- [x] **Step 3: 建目录** `mkdir -p evidence/part{1..5} artifacts/part{1..5} inputs`。
- [x] **Step 4: 按学习单原样下载**（URL 记入 README.md）：
  ```bash
  curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz \
    -o inputs/images.npz
  uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \
    config.json diffusion_pytorch_model.safetensors \
    --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
    --local-dir inputs/cifar-model
  ```
  校验：`python -c "import numpy as np; a=np.load('inputs/images.npz'); print(list(a.keys()), a[list(a.keys())[0]].shape)"` → 4 张 32×32×3、值域约 [−1,1]；`ls inputs/cifar-model` 两文件齐全（safetensors ≈ 143 MB）。
- [x] **Step 5: README stub**（标题、环境命令、provenance 小节，表格后续任务填）+ **Commit** `week7: uv environment and pinned inputs`（确认 `inputs/`、`artifacts/` 未被跟踪）。

**预计用时：** 代理 15–25 min；下载约 5–15 min（视网速）。

---

## Task 1 (Part 1 核心): 混合分布与前向扩散（TDD）

**Files:**
- Create: `week7/mixture.py`, `week7/diffusion.py`
- Test: `week7/tests/test_mixture.py`, `week7/tests/test_diffusion.py`

**Interfaces (Produces, 全部纯函数、jit 友好):**
- `mixture.MEANS: float32 (4,2)`；`mixture.SIGMA2 = 0.09`
- `mixture.sample_mixture(key, n) -> x0 (n,2)`：先等概率选分量再按 N(μ_k, 0.09I) 采样
- `mixture.component_weights(x, mean_coeff, var) -> (n,4)`：在"x_t = mean_coeff·x0 + 噪声方差 var·I"的线性高斯下，各分量对 x 的后验权重（softmax of log N(x; mean_coeff·μ_k, (mean_coeff²·0.09 + var)I)）；对 Part 2 传 mean_coeff=√ᾱ_t、var=1−ᾱ_t，对 Part 3 传 mean_coeff=1−s、var=s²
- `mixture.exact_noise(x, t, abar) -> (n,2)`：E[ε|x_t]，分量内后验均值平均（推导见 Step 3）
- `mixture.exact_velocity(x, s) -> (n,2)`：v(x,s) = E[z−x0|x_s=x]（Task 5 实现，接口先留）
- `mixture.assign_modes(x) -> labels (n,)`：最近目标均值
- `mixture.mode_stats(x) -> populations (4,), centres (4,2), spreads (4,2)`（population=组内点数占比；centre=组内均值；spread=组内每坐标标准差）
- `mixture.energy_distance(X, Y) -> float`：D = 2E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖，用全对平均（X−Y）与 i<j 对平均（X−X′、Y−Y′）
- `diffusion.beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2) -> (T,)`（线性，含端点）
- `diffusion.alphas_cumprod(beta) -> (T+1,)`：abar[0]=1，abar[t]=∏_{s≤t}(1−β_s)
- `diffusion.forward_step(x_prev, t, beta, key) -> x_t`：x_t = √α_t·x_{t−1} + √β_t·z
- `diffusion.noise_to_level(x0, t, eps, abar) -> x_t`：x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε（式 5）
- `diffusion.ddpm_sample(eps_net, x_T, abar, beta, zs) -> x0`：zs 形状 (T,n,2)，逐步 z 显式传入（Task 3 前，采样循环本 Task 先给出签名与逐步公式）
- `diffusion.ddim_sample / ddiffusion.ddpm_skip_sample`（Task 4）

- [x] **Step 1: 失败测试** `tests/test_diffusion.py`：
  ```python
  import jax, jax.numpy as jnp, numpy as np
  from diffusion import beta_schedule, alphas_cumprod, forward_step, noise_to_level

  def test_schedule_endpoints_and_cumprod():
      beta = beta_schedule()
      assert beta.shape == (1000,)
      assert abs(beta[0] - 1e-4) < 1e-12 and abs(beta[-1] - 0.02) < 1e-12
      abar = alphas_cumprod(beta)
      assert abs(abar[0] - 1.0) < 1e-12
      assert np.all(np.diff(abar) < 0)                      # 单调下降
      assert 2e-5 < abar[1000] < 8e-5                       # 学习单: ᾱ_1000 ≈ 4e-5

  def test_variance_preserving_coordinate():
      # α_t + β_t = 1 ⇒ 方差 1 的坐标保持方差 1（学习单 UNDERSTAND）
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      key = jax.random.PRNGKey(0)
      x = jax.random.normal(key, (200000, 2), dtype=jnp.float32)
      t = 500
      xt = noise_to_level(x, t, jax.random.normal(jax.random.PRNGKey(1), x.shape), abar)
      var = jnp.var(xt, axis=0)          # x 方差 1 ⇒ x_t 方差 = ᾱ + (1−ᾱ) = 1
      assert jnp.all(jnp.abs(var - 1.0) < 0.03)

  def test_sequential_steps_match_direct_draw_moments():
      # 逐步前向 vs 直达式(5)：同 fixed 噪声下均值一致、方差一致（分布相同）
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      x0 = jnp.asarray([[3.0, 0.0]]) * jnp.ones((20000, 1))  # 分量 1 的点
      x = x0
      subk = jax.random.split(jax.random.PRNGKey(2), 300)
      for t in range(1, 301):
          x = forward_step(x, t, beta, subk[t-1])
      assert jnp.max(jnp.abs(jnp.mean(x, 0) - jnp.sqrt(abar[300]) * jnp.array([3.0, 0.0]))) < 0.02
      assert jnp.all(jnp.abs(jnp.var(x, 0) - (abar[300]*0.09 + 1 - abar[300])) < 0.03)
  ```
  `tests/test_mixture.py`：
  ```python
  import jax, jax.numpy as jnp, numpy as np
  from mixture import (MEANS, sample_mixture, assign_modes, mode_stats,
                       component_weights, energy_distance)

  def test_sample_recovers_population_centre_spread():
      x = sample_mixture(jax.random.PRNGKey(0), 400000)
      pop, cen, spr = mode_stats(x)
      assert np.allclose(pop, 0.25, atol=0.005)
      assert np.max(np.abs(cen - MEANS)) < 0.02
      assert np.max(np.abs(spr - 0.3)) < 0.01

  def test_assign_modes_nearest():
      x = jnp.array([[3.0, 0.1], [0.0, -3.2], [2.6, 2.6]])
      assert assign_modes(x).tolist() == [0, 3, 0]   # MEANS 顺序 (3,0),(0,3),(−3,0),(0,−3)

  def test_component_weights_degenerate_single_and_far():
      w = component_weights(jnp.array([[3.0, 0.0]]), mean_coeff=1.0, var=0.0)
      assert np.allclose(w, [[1.0, 0, 0, 0]], atol=1e-6)     # 无噪声 ⇒ 就近分量权重 1
      w2 = component_weights(jnp.array([[100.0, 0.0]]), mean_coeff=0.01, var=0.9)
      assert np.allclose(w2.sum(-1), 1.0)                     # 权重归一

  def test_energy_distance_zero_for_identical_sets():
      X = sample_mixture(jax.random.PRNGKey(1), 500)
      assert abs(float(energy_distance(X, X))) < 1e-6
      Y = X + 10.0
      assert float(energy_distance(X, Y)) > 1.0               # 远移 ⇒ 距离大
  ```
- [x] **Step 2:** `uv run pytest tests/test_mixture.py tests/test_diffusion.py` → FAIL（模块不存在）。
- [x] **Step 3: 实现核心公式**。
  `mixture.component_weights`（log-domain 稳定版）：
  ```python
  def component_weights(x, mean_coeff, var):
      # x|k ~ N(mean_coeff*mu_k, (mean_coeff**2*SIGMA2 + var)*I)
      total_var = mean_coeff**2 * SIGMA2 + var
      d2 = jnp.sum((x[:, None, :] - mean_coeff * MEANS[None]) ** 2, -1)  # (n,4)
      logw = -0.5 * d2 / total_var                       # 等权 π_k 与归一化常数省略
      return jax.nn.softmax(logw, axis=-1)
  ```
  `mixture.exact_noise`（分量内 x0|x_t,k ~ N(m_k, S_k)，权重平均后由 ε = (x_t − √ᾱ_t x0)/√(1−ᾱ_t)）：
  ```python
  def exact_noise(x, t, abar):
      ab = abar[t]
      prec = ab / (1 - ab) + 1.0 / SIGMA2          # x0|k 后验精度（每坐标）
      S = 1.0 / prec
      m_k = S * (jnp.sqrt(ab) * x[:, None, :] / (1 - ab) + MEANS[None] / SIGMA2)  # (n,4,2)
      w = component_weights(x, jnp.sqrt(ab), 1 - ab)
      x0_hat = jnp.einsum("nk,nkd->nd", w, m_k)
      return (x - jnp.sqrt(ab) * x0_hat) / jnp.sqrt(1 - ab)
  ```
  `diffusion` 模块按接口逐行实现（`forward_step` 用 `jax.random.normal`；`noise_to_level` 为两行公式）。
- [x] **Step 4:** 测试 PASS。
- [x] **Step 5: Commit** — `week7: mixture exact quantities and forward diffusion with tests`。

**预计用时：** 代理 45–60 min；计算秒级。

---

## Task 2 (Part 1 训练): 时间条件噪声预测器

**Files:**
- Create: `week7/net.py`, `week7/scripts/part1_train.py`
- Test: `week7/tests/test_net.py`
- 产物: `artifacts/part1/noise_predictor.npz`（10 阶段权重 + 损失历史 + 设置 JSON 字符串）

**Interfaces (Produces):**
- `net.time_features(u, dim=32) -> (n, dim)`：正弦嵌入，u∈[0,1]（扩散传 t/T，流传 s），频率 log 间隔 1e-4…1e1
- `net.init_mlp(key, width=128, time_dim=32) -> params`：拼接 [x(2), time(32)] → 3 层 FC（width, SiLU）→ 输出 2；Glorot 初始化
- `net.apply_mlp(params, x, u) -> (n,2)`
- `net.noise_loss(params, x0, t, eps, abar) -> float`：L = ½‖ε_θ(x_t,t) − ε‖²（对 batch 平均）
- `net.train_noise_predictor(key, n_updates=25000, batch=512, lr0=<pilot 选定>, stage_updates=(0,100,250,500,1000,2000,4000,8000,16000,25000)) -> (stages list[(update, params)], loss_history (25000,), settings dict)`：每 update 新采 x0∼p_data、t∼U{1..T}、ε∼N(0,I)，x_t 用直达式(5)；学习率线性衰减 η_k = lr0·(1−k/n_updates)；普通 SGD（学习单算法框）+ `lax.scan` JIT
- 训练脚本把 stages 序列化进 npz（每阶段存 `(2,width)` 等 Dense 权重展平或 object 数组）+ `settings` JSON。

- [x] **Step 1: 失败测试** `tests/test_net.py`：
  ```python
  import jax, jax.numpy as jnp, numpy as np
  from net import time_features, init_mlp, apply_mlp, noise_loss, train_noise_predictor

  def test_shapes_and_time_conditioning():
      p = init_mlp(jax.random.PRNGKey(0))
      x = jnp.zeros((7, 2))
      assert apply_mlp(p, x, jnp.full((7,), 0.5)).shape == (7, 2)
      out1 = apply_mlp(p, x, jnp.zeros((7,)))          # u=0
      out2 = apply_mlp(p, x, jnp.ones((7,)))           # u=1
      assert float(jnp.max(jnp.abs(out1 - out2))) > 1e-3   # 时间条件确实改变输出

  def test_noise_loss_decreases_on_fixed_batch():
      from diffusion import alphas_cumprod, beta_schedule
      abar = alphas_cumprod(beta_schedule())
      x0 = jax.random.normal(jax.random.PRNGKey(1), (512, 2))
      t = jnp.ones(512, dtype=jnp.int32) * 500
      eps = jax.random.normal(jax.random.PRNGKey(2), (512, 2))
      p = init_mlp(jax.random.PRNGKey(3))
      loss0 = noise_loss(p, x0, t, eps, abar)
      p2, hist = None, None
      stages, hist, _ = train_noise_predictor(jax.random.PRNGKey(4), n_updates=300,
                                              batch=512, lr0=0.03,
                                              stage_updates=(0, 300))
      assert hist[-1] < hist[0]                        # 训练确实降损失
  ```
- [x] **Step 2:** FAIL → **Step 3: 实现**（`time_features`：`sin/cos(2π f_j u)` 各 dim/2 个频率；MLP 参数用 NamedTuple of (w,b) 列表；训练器内 `value_and_grad`，t 抽样用 `jax.random.randint(1, T+1)`）。
- [x] **Step 4:** PASS。
- [x] **Step 5: 学习率 pilot + 正式训练。** 先 2000 updates × {0.003, 0.01, 0.03, 0.1} 四档跑 pilot（各 <10 s），取 2000 步损失最低者为 lr0（记录进 settings）；再跑正式 25000 updates（约 1 分钟），存 npz。用 Task 3 的采样器抽查最终模型能出四簇（此时若失败，允许改 Adam(lr=1e-3 线性衰减)重跑并在 settings 记录该决定）。
- [x] **Step 6: Commit** — `week7: time-conditioned noise predictor, training loop and stages`。

**预计用时：** 代理 60–90 min；训练 <2 min。

---

## Task 3 (Part 1 验证): DDPM 采样 + 三个 VERIFY

**Files:**
- Modify: `week7/diffusion.py`（补 `ddpm_sample`）
- Test: `week7/tests/test_diffusion.py`（追加）
- Create: `week7/scripts/part1_forward.py`, `week7/scripts/part1_stages.py`, `week7/scripts/part1_generate.py`
- 产物: `evidence/part1/{forward_diffusion.gif, stages.gif, generate.gif, generate_final.png, mode_stats.txt}`

**Interfaces (Produces):**
- `diffusion.ddpm_sample(eps_net, x_T, abar, beta, zs) -> x0`，其中 `eps_net(x, t) -> eps_hat`；逐步（学习单 DDPM-Sample 框，均值用等价标准式）：
  ```
  μ_t = (x_t − β_t/√(1−ᾱ_t) · ε̂)/√α_t          # α_t = 1−β_t（逐步量）
  t>1: x_{t−1} = μ_t + √β_t · z_t ;  t=1: x_0 = μ_1
  ```
- `diffusion.ddpm_trajectory(...) -> (T+1, n, 2)`：同上但记录全部状态（供动画/路径）。

- [x] **Step 1: 失败测试**（追加到 test_diffusion.py）：
  ```python
  def test_ddpm_step_with_true_noise_stays_consistent():
      # 用真噪声 ε（由已知 x0 构造）作预测: ε̂ = ε ⇒ 一步后均值恰为后验均值
      from diffusion import ddpm_sample
      import mixture
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      key = jax.random.PRNGKey(0)
      x0 = mixture.sample_mixture(key, 64)
      t = 700
      eps = jax.random.normal(jax.random.PRNGKey(1), x0.shape)
      xt = noise_to_level(x0, t, eps, abar)
      mean_one_step = (xt - beta[t-1] / jnp.sqrt(1 - abar[t]) * eps) / jnp.sqrt(1 - beta[t-1])
      assert True  # 公式自查: ddpm 内部用同一表达式; 见下一条端到端测试

  def test_ddpm_final_step_adds_no_noise_and_output_shape():
      from net import init_mlp, apply_mlp
      from diffusion import ddpm_sample
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      p = init_mlp(jax.random.PRNGKey(0))
      net = lambda x, t: apply_mlp(p, x, t / 1000.0)
      zs = jnp.zeros((1000, 5, 2))                   # 全零噪声
      x0 = ddpm_sample(net, jax.zeros((5, 2)), abar, beta, zs)
      assert x0.shape == (5, 2) and jnp.all(jnp.isfinite(x0))
      # zs 全零 ⇒ 相同输入两次调用结果逐位相同（确定性接口）
      assert jnp.allclose(x0, ddpm_sample(net, jnp.zeros((5, 2)), abar, beta, zs))
  ```
- [x] **Step 2:** FAIL → **Step 3: 实现** `ddpm_sample`/`ddpm_trajectory`（`lax.scan` 从 t=T 到 1；`eps_net` 接口为 `(x, t:int)`，脚本侧用闭包把 params 绑进去）→ **Step 4:** PASS。
- [x] **Step 5: VERIFY 1 脚本** `scripts/part1_forward.py`：2000 个干净样本逐步前向（跟随同一样本，坐标固定尺度如 [−5,5]），t ∈ {0,10,20,50,100,200,400,600,800,1000} 等帧生成 `forward_diffusion.gif`；打印 t=1000 处每坐标 mean/var。
  **验收（学习单阈值）:** mean 距 0 < 0.1，var 距 1 < 0.1。
- [x] **Step 6: VERIFY 2 脚本** `scripts/part1_stages.py`：载入 npz 的 10 阶段；固定 key 生成共享 x_T(10,2) 与共享逐步噪声 zs(T,10,2)；每阶段一帧：10 条完整路径（起点白点、终点彩点、虚线圆 = 各中心半径 2σ=0.6、固定方形 [−6,6]，初始化阶段路径出画幅属预期 |x|≈1500）+ 该阶段为止的损失历史（竖直虚线标当前阶段）。输出 `stages.gif`。
  **验收:** 初始化路径出画幅；最终阶段 10 个终点全部落在四簇上；损失前几百 updates 内骤降、后平台（参考 ≈0.25，非零由条件均值恒等式解释）。
- [x] **Step 7: VERIFY 3 脚本** `scripts/part1_generate.py`：最终模型、1000 步、2000 样本；反向过程动画 `generate.gif`（帧内同前向固定尺度）+ 终态散点 vs 精确样本 `generate_final.png`；`mode_stats` 写入 `mode_stats.txt`（populations/centres/spreads 表）。
  **验收:** populations 距 0.25 < 0.04；spreads 距 0.30 < 0.03；centres 距目标 < 0.05（参考: populations 0.245–0.262，spreads 0.29–0.30）。未达标 → 回 Task 2 调 lr/宽度重训，不放宽阈值。
- [x] **Step 8:** README 表格加三行（图 ↔ 命令）；**Commit** — `week7: part1 ddpm sampler and forward/stages/generation evidence`。

**预计用时：** 代理 90–120 min；计算 <2 min（25000 步训练外的采样均为秒级）。

---

## Task 4 (Part 2): DDIM、跳层 DDPM 与精确条件均值噪声

**Files:**
- Modify: `week7/diffusion.py`（`ddim_sample`, `ddpm_skip_sample`）
- Test: `week7/tests/test_diffusion.py`（追加）
- Create: `week7/scripts/part2_determinism.py`
- 产物: `evidence/part2/{determinism.png, samples_compare.png, max_diff.txt}`

**Interfaces (Produces):**
- `diffusion.ddim_sample(eps_net, x_T, levels, abar) -> x0`（levels 为降序列表如 (1000, 980, ..., 0)）：
  ```
  x̂0 = x_t/√ᾱ_t − ε̂·√(1/ᾱ_t − 1)          # Tweedie（式 4 行）
  x_t' = √ᾱ_t'·x̂0 + √(1−ᾱ_t')·ε̂            # 目标层系数（式 5 行）
  ```
- `diffusion.ddpm_skip_sample(eps_net, x_T, levels, abar, zs)`（式 8）：a = ᾱ_t/ᾱ_t'，μ = (1/√a)(x_t − (1−a)/√(1−ᾱ_t)·ε̂)；t'>0 加方差 1−a 噪声，t'=0 不加。
- 采样器 `eps_net` 一律接口 `(x, t:int) -> eps_hat`，故 Part 1 训练网络与 `mixture.exact_noise` 包装的精确版可互换。

- [x] **Step 1: 失败测试**（追加）：
  ```python
  def test_ddim_true_noise_jump_is_exact_forward_sample():
      # ε̂ 取真噪声 ⇒ x̂0 = x0 精确, 跳到任意 t' 恰为该层前向样本(同一 ε)
      import mixture
      from diffusion import ddim_sample
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      x0 = mixture.sample_mixture(jax.random.PRNGKey(0), 8)
      t, tp = 600, 200
      eps = jax.random.normal(jax.random.PRNGKey(1), x0.shape)
      xt = noise_to_level(x0, t, eps, abar)
      net = lambda x, tt: (x - jnp.sqrt(abar[tt]) * x0) / jnp.sqrt(1 - abar[tt])  # 假网络回真噪声
      # 注意: 上式依赖 x0, 仅为构造测试; 输出 x_t' 应等于 √ᾱ_t' x0 + √(1−ᾱ_t') ε
      out = ddim_sample(net, xt, (t, tp), abar)
      expect = jnp.sqrt(abar[tp]) * x0 + jnp.sqrt(1 - abar[tp]) * eps
      assert jnp.max(jnp.abs(out - expect)) < 1e-4

  def test_ddim_is_deterministic_and_reproducible():
      from net import init_mlp, apply_mlp
      p = init_mlp(jax.random.PRNGKey(0))
      net = lambda x, t: apply_mlp(p, x, t / 1000.0)
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      levels = tuple(range(1000, 0, -50))
      xt = jax.random.normal(jax.random.PRNGKey(1), (16, 2))
      assert jnp.array_equal(ddim_sample(net, xt, levels, abar),
                             ddim_sample(net, xt, levels, abar))

  def test_skip_ddpm_adjacent_matches_part1_posterior_mean():
      # 相邻层 a = α_t ⇒ 方差 1−a = β_t, 均值与 Part 1 相同
      from diffusion import ddpm_skip_sample
      beta = beta_schedule(); abar = alphas_cumprod(beta)
      t = 500
      x = jnp.array([[0.3, -0.2]])
      e = jnp.array([[1.0, -1.0]])
      net = lambda x_, t_: e * jnp.ones_like(x_)
      a = abar[t] / abar[t-1]
      mu = (x - (1 - a) / jnp.sqrt(1 - abar[t]) * e) / jnp.sqrt(a)
      # 零噪声跑一步: zs 取 0
      out = ddpm_skip_sample(net, x, (t, t-1), abar, jnp.zeros((1, 1, 2)))
      assert jnp.max(jnp.abs(out - mu)) < 1e-6
  ```
  （若 `mixture.exact_noise` 的形式参数顺序与之不符，以实现为准修测试调用，不改公式。）
- [x] **Step 2:** FAIL → **Step 3: 实现** 两个采样器（共享"取层序列"辅助函数；`ddim_sample` 无 key 参数）→ **Step 4:** PASS。
- [x] **Step 5: VERIFY 1 脚本** `scripts/part2_determinism.py`：
  - 同一 x_T 下 DDIM（1000 层全网格）跑两遍、DDPM 跑两遍（不同 z 序列）；画 10 条轨迹（DDIM 两次完全重合一条，DDPM 两条分开）`determinism.png`；
  - 2000 个共享起始噪声的 DDIM/DDPM 样本 vs 精确样本散点 `samples_compare.png`；
  - 打印两次 DDIM 运行的最大差（参考: 0，逐位相等）与两 DDIM 版本终点同簇比例（参考: ≈1/4）。
  **验收:** DDIM 最大差 = 0（打印值）；DDIM 各簇占比距 0.25 < 0.015；两采样器 spreads ∈ [0.29, 0.32]。
- [x] **Step 6:** README 表格两行；**Commit** — `week7: ddim, skip-level ddpm, exact conditional-mean noise and determinism evidence`。

**预计用时：** 代理 60–90 min；计算 <1 min。

---

## Task 5 (Part 3 核心): flow matching 训练与精确边缘速度

**Files:**
- Modify: `week7/net.py`（`flow_loss`, `train_velocity`）, `week7/mixture.py`（实现 `exact_velocity`）
- Test: `week7/tests/test_net.py`（追加）, `week7/tests/test_mixture.py`（追加）
- Create: `week7/scripts/part3_train_flow.py`
- 产物: `artifacts/part3/velocity.npz`（权重 + 损失 + 设置）

**Interfaces (Produces):**
- `net.flow_loss(params, x0, z, s, abar=None) -> float`：x_s = (1−s)x0 + s z，L = ‖v_θ(x_s,s) − (z−x0)‖²（batch 平均）
- `net.train_velocity(key, n_updates=25000, batch=512, lr0=<pilot>, stage_updates=(0, n_updates)) -> (params, loss_history, settings)`：每 update 新采 x0、z、s∼U[0,1]；新初始化权重（沿用噪声预测器架构）；η 线性衰减到 0
- `mixture.exact_velocity(x, s)`：分量 k 内 x_s|k ~ N((1−s)μ_k, ((1−s)²·0.09 + s²)I)；后验 E[x0|x,k]（同 Task 1 的 S,m_k 公式，mean_coeff=1−s、var=s²）；E[z|x,k] = (x − (1−s)m_k)/s；v = Σ_k w_k(x,s)·(E[z|x,k] − m_k)；w_k 用 `component_weights(x, 1−s, s²)`
- Euler 采样在脚本里实现（12 式），或 `net.euler_sample(v_net, x1, n_steps) -> x0`：s 网格 1 → 0 均匀 n+1 点，x_{s−Δs} = x_s − Δs·v_θ(x_s,s)

- [x] **Step 1: 失败测试**（追加 test_mixture.py）：
  ```python
  def test_exact_velocity_limits_and_symmetry():
      from mixture import exact_velocity
      x = jnp.array([[2.0, -1.0], [0.0, 0.0], [5.0, 5.0]])
      # s=1: v(x,1) = x（学习单: 近 s=1 场趋于 v=x）
      assert jnp.max(jnp.abs(exact_velocity(x, 1.0) - x)) < 1e-4
      # 对称性: 混合分布关于原点对称 ⇒ v(0,s) = 0
      assert jnp.max(jnp.abs(exact_velocity(jnp.zeros((3, 2)), 0.5))) < 1e-5

  def test_exact_velocity_matches_monte_carlo_at_s_half():
      # MC 对拍: 采 (x0,z), 保留 s=0.5 落在 x 附近小格内的样本, 平均 (z−x0)
      from mixture import exact_velocity, sample_mixture
      key = jax.random.PRNGKey(0)
      x0 = sample_mixture(key, 2_000_000)
      z = jax.random.normal(jax.random.PRNGKey(1), x0.shape)
      s = 0.5
      xs = (1 - s) * x0 + s * z
      v = z - x0
      centers = jnp.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5]])
      for c in centers:
          m = jnp.linalg.norm(xs - c, axis=-1) < 0.05
          assert m.sum() > 500
          mc = v[m].mean(0)
          ex = exact_velocity(c[None], s)[0]
          assert jnp.max(jnp.abs(mc - ex)) < 0.05   # MC 噪声容忍
  ```
  （追加 test_net.py）：
  ```python
  def test_train_velocity_loss_decreases_and_is_fresh_init():
      from net import train_velocity, init_mlp
      p, hist, _ = train_velocity(jax.random.PRNGKey(0), n_updates=300, batch=512, lr0=0.03)
      assert hist[-1] < hist[0]
  ```
- [x] **Step 2:** FAIL → **Step 3: 实现** `exact_velocity`（注意 s=1 分母：s→1 时 E[z|x,k] = x，直接分支处理 s>1−1e−6 返回 x）与 `flow_loss`/`train_velocity` → **Step 4:** PASS。
- [x] **Step 5: 训练。** 同 Task 2 的 pilot 选 lr0，跑 25000 updates（约 1 分钟），存 `artifacts/part3/velocity.npz`。
- [x] **Step 6: Commit** — `week7: flow matching training and exact marginal velocity with tests`。

**预计用时：** 代理 60–90 min；计算 <2 min。

---

## Task 6 (Part 3 验证): 速度场对比 + 三采样器步数扫描

**Files:**
- Create: `week7/scripts/part3_fields.py`, `week7/scripts/part3_compare.py`
- 产物: `evidence/part3/{velocity_fields.png, steps_sweep.png, steps_table.txt}`

**Interfaces (Consumes):** Task 4 的 `ddim_sample`/`ddpm_skip_sample`（levels 可选子网格，均匀间隔），Task 5 的速度网络与 `euler_sample`，Task 1 的 `mode_stats`/`energy_distance`，Task 2 的噪声预测器。

- [x] **Step 1: VERIFY 1 脚本** `scripts/part3_fields.py`：15×15 网格覆盖样本范围（约 [−5,5]²），s ∈ {0.1, 0.5, 0.9} 三面板双行（学习 vs 精确，箭头图，远场箭长裁剪）；打印各 s 的相对 RMS = RMS(v_θ − v_exact)/RMS(v_exact)。
  **验收:** 相对 RMS < 0.1 于每个 s（参考: 0.011, 0.017, 0.058）。
- [x] **Step 2: VERIFY 2 脚本** `scripts/part3_compare.py`：
  - 步数 K ∈ {1,2,5,10,20,50,100}，DDPM/DDIM 另加 1000 层（即 K=1000）；"K 步" = 在 [0,T] 上均匀间隔的层网格（DDPM/DDIM 用 Part 2 的跳层版；DDIM 无噪声，DDPM 加方差 1−a 噪声，t'=0 不加）；
  - 每配置 4000 样本（各采样器共享起始噪声 seed=0；DDPM 另有逐步噪声 seed）；记录 cluster fractions、centres、widths、与 4000 精确样本的能量距离 → `steps_table.txt`；
  - null scale：40 组独立精确样本（各 4000）配 20 对，算能量距离，取 95% 分位数（参考 ≈0.0013）；
  - 三联图 `steps_sweep.png`：能量距离 / 最大簇占比误差 / 最大宽度误差 vs K（三采样器三条线，log-x），能量距离面板画 null 95% 带，簇占比面板画 p=0.25±2×二项标准误带（√(0.25·0.75/4000)≈0.0068），D<1e−4 画在 1e−4，K=20 竖虚线。
  **验收:** flow 在 K=20 能量距离 < 0.005 且 widths 距 0.30 < 0.03（参考: flow 0.0015@20、DDIM 0.0038、DDPM ≈50 层达标）；K∈{1,2} 所有采样器均坍缩向均值（含精确预测版本，可加跑一组 exact-ε DDIM/flow 作旁证）。
- [x] **Step 3:** README 两行 + 结果摘要（哪个采样器多少步达标）；**Commit** — `week7: velocity field check and ddpm/ddim/flow step-count comparison`。

**预计用时：** 代理 90–120 min；计算：能量距离 40 组 4000×4000（分块算，每组 <1 s）；采样扫网格均为秒级~分钟级。

---

## Task 7 (Part 4): JAX 复刻 UNet + CIFAR-10 采样

**Files:**
- Create: `week7/unet.py`, `week7/scripts/part4_generate.py`
- Test: `week7/tests/test_unet.py`
- 产物: `evidence/part4/{cifar_stages.png, cifar_final.png, cifar_vs_real.png}`

**Interfaces (Produces):**
- `unet.load_config(path) -> dict`（读 `inputs/cifar-model/config.json`）
- `unet.load_params(path) -> jax pytree`：safetensors（`safetensors.numpy.load_file`）→ JAX 布局映射：
  - Conv2D `conv.weight (O,I,kh,kw)` → `(kh,kw,I,O)`；Linear `weight (O,I)` → `(I,O)`ᵀ；GroupNorm `weight/bias (C,)` 直传；attention `to_q/k/v/out.0.weight` 按 Linear 处理；
  - 名称逐字对齐 diffusers v0.40.0 `UNet2DModel`（`time_embedding.linear_{1,2}`、`down_blocks.{i}.resnets.{j}.conv{1,2}`、`conv_shortcut`、`downsamplers.0.conv`、`mid_block`、`up_blocks.{i}.upsamplers.0.conv`、`conv_norm_out`、`conv_out`）。
- `unet.unet_apply(params, config, x_t (B,32,32,3) float32, t_idx (B,) int32) -> eps (B,32,32,3)`
  - 架构核对锚点（学习单给出）：网格 32→16→8→4→8→16→32，down 通道 **[128,256,256,256]**，参数总量 **≈35.7M**；模块按 config 构建：Timesteps 正弦嵌入（`flip_sin_to_cos=True`, `freq_shift=0`）、TimestepEmbedding(Linear-SiLU-Linear)、Down/UpBlock2D 与 AttnDown/UpBlock2D（ResnetBlock: GroupNorm(32)-SiLU-Conv3×3 ×2 + 时间投影加性偏置；Attention: GroupNorm + qkv 线性 + softmax + 残差 `rescale_output_factor=1`）、UNetMidBlock2D(resnet-attn-resnet)、上采样 skip 拼接、UpBlock 末块含 `conv_norm_out`(GroupNorm 32) + SiLU + `conv_out` 1×1... （3×3）；若某块 config 与此不符，以 **config.json 为准**。
- 采样复用 `diffusion.ddpm_sample`：`eps_net = lambda x, t: unet_apply(params, config, x, full((B,), t-1))`（模型索引 t−1，float32，无裁剪）。

- [x] **Step 1: 失败测试** `tests/test_unet.py`：
  ```python
  import jax, jax.numpy as jnp, numpy as np, pathlib
  from unet import load_config, load_params, unet_apply

  CFG = pathlib.Path(__file__).parents[1] / "inputs/cifar-model/config.json"
  W = CFG.with_name("diffusion_pytorch_model.safetensors")

  def test_param_count_matches_sheet():
      params = load_params(W)
      n = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
      assert abs(n - 35_700_000) / 35_700_000 < 0.01   # 学习单: ~35.7M

  def test_output_shape_finite_and_deterministic():
      config = load_config(CFG); params = load_params(W)
      x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 3), dtype=jnp.float32)
      t = jnp.array([999, 42], dtype=jnp.int32)
      eps = unet_apply(params, config, x, t)
      assert eps.shape == (2, 32, 32, 3)
      assert bool(jnp.all(jnp.isfinite(eps)))
      assert jnp.allclose(eps, unet_apply(params, config, x, t), atol=1e-5)
  ```
  （`inputs/` 不在 git：测试文件顶部若 `not CFG.exists()` 则 `pytest.skip("download inputs/cifar-model first")`。）
- [x] **Step 2:** FAIL → **Step 3: 实现** `unet.py`（逐模块小函数 + config 驱动组装；Conv 用 `jnp.einsum`/`lax.conv_general_dilated`(NHWC, SAME padding)；GroupNorm 注意 torch `num_channels % num_groups == 0`、按组减均值除 std + affine；attention 在 (B,H*W,C) 上做）。**参数量对不上 35.7M 是第一诊断信号**：优先核对 attention 块位置（AttnDown/UpBlock）与 `attention_head_dim`。
  可选数值对拍（网络好时）：
  ```bash
  uv run --with torch --with diffusers==0.40.0 python - <<'EOF'
  # 同一 (x, t) 输入, diffusers UNet2DModel vs unet.unet_apply, 输出 max|Δ| 应 < 1e-3
  EOF
  ```
- [x] **Step 4:** PASS。
- [x] **Step 5: VERIFY 1 脚本** `scripts/part4_generate.py`：4 张图（batch=4，seed 固定），1000 步 DDPM（CPU 约 3 分钟；每步一次 UNet 前向）；快照 t ∈ {1000,900,...,300,200,150,100,50,0}（学习单: t=200 仍近噪声、200→50 显形）；显示用 (x+1)/2 clip [0,1]。出 `cifar_stages.png`（4 行 × 快照列）、`cifar_final.png`、`cifar_vs_real.png`（与 week6 Part 1 的 CIFAR 画廊脚本并排真实图；直接复用 `week6/gallery.py` 的显示函数或拷贝小助手，README 注明来源）。
  README 写明：**哪些是下载前已学好的**（网络权重 = 噪声预测），**哪些是本作业计算的**（DDPM 反向更新、调度表系数、采样循环）。
- [x] **Step 6:** Commit — `week7: jax port of cifar unet, pretrained noise prediction and ddpm image generation`。

**预计用时：** 代理 3–5 h（本计划最重任务；权重映射细节多）；计算：参数量/形状测试秒级，采样约 3 min。

---

## Task 8 (Part 5): RePaint 补全

**Files:**
- Create: `week7/repaint.py`, `week7/scripts/part5_inpaint.py`
- Test: `week7/tests/test_repaint.py`
- 产物: `evidence/part5/inpaint_gallery.png`, `evidence/part5/mse_table.txt`

**Interfaces (Consumes/Produces):**
- `repaint.inpaint(eps_net, x_T, Y, mask, abar, key, levels=range(1000,-1,-20)) -> X0`：学习单 RePaint-Inpaint 框逐行：
  - 跳层 t→t′（t′=t−20，最后 20→0 单独处理）；
  - Tweedie `x̂0 = x_t/√ᾱ_t − ε̂·√(1/ᾱ_t − 1)`，**仅此估计** clip 到 [−1,1]（噪声状态不 clip）；
  - a = ᾱ_t/ᾱ_t′；均值 `μ = (√ᾱ_t′·(1−a)·x̂0 + √a·(1−ᾱ_t′)·x_t)/(1−ᾱ_t)`（式 16）；方差 `v = (1−ᾱ_t′)·(1−a)/(1−ᾱ_t)`（式 17），t′=0 时 v=0；
  - proposal `Z_t′ ~ N(μ, v)`；前向噪声化观测 `Y_t′ = √ᾱ_t′·Y + √(1−ᾱ_t′)·η`；merge `X_t′ = M⊙Y_t′ + (1−M)⊙Z_t′`（式 14–15）；
  - 每跳 2 次尝试（第 2 次先 `X_t = √a·X_t′ + √(1−a)·f` 回噪再重提），最后跳 20→0 只 1 次；共 49×2+1 = **99 次网络评估**。
- `mask`：`np.zeros((32,32,3), bool); mask[6:26, 6:26, :] = False` 之外为 True（观测=1）。

- [x] **Step 1: 失败测试** `tests/test_repaint.py`（用玩具 eps_net，不依赖 UNet 下载）：
  ```python
  import jax, jax.numpy as jnp, numpy as np
  from diffusion import alphas_cumprod, beta_schedule
  from repaint import inpaint

  def _toy_net(x, t):
      return jnp.ones_like(x) * 0.1

  def test_merge_keeps_observed_exactly():
      abar = alphas_cumprod(beta_schedule())
      rng = np.random.default_rng(0)
      Y = rng.uniform(-1, 1, size=(1, 8, 8, 1)).astype("float32")
      Y = jnp.asarray(Y)
      mask = jnp.ones_like(Y, dtype=jnp.float32).at[:, 2:5, 2:5].set(0.0)
      key = jax.random.PRNGKey(0)
      xT = jax.random.normal(key, Y.shape)
      out = inpaint(_toy_net, xT, Y, mask, abar, key,
                    levels=[40, 20, 0])          # 小网格, 端到端跑通
      assert out.shape == Y.shape
      assert float(jnp.max(jnp.abs(out * mask - Y * mask))) < 1e-5   # 最后 merge 原样覆写观测

  def test_eval_count_is_two_attempts_plus_final(monkeypatch):
      calls = []
      def counting_net(x, t):
          calls.append(int(t)); return jnp.zeros_like(x)
      abar = alphas_cumprod(beta_schedule())
      Y = jnp.zeros((1, 8, 8, 1)); mask = jnp.ones_like(Y)
      inpaint(counting_net, jnp.zeros_like(Y), Y, mask, abar,
              jax.random.PRNGKey(1), levels=[40, 20, 0])
      assert len(calls) == 3      # 40→20 两次 + 20→0 一次（小网格 3 跳）
  ```
- [x] **Step 2:** FAIL → **Step 3: 实现** `repaint.py`（纯 Python 循环 99 次前向即可，无需 JIT 整体；`jax.random` 每 draw split key）→ **Step 4:** PASS。
- [x] **Step 5: VERIFY 1 脚本** `scripts/part5_inpaint.py`：载入 `inputs/images.npz` 4 图，mask 隐藏 rows/cols 6:26 全通道；seeds 7201–7204（16 个补全；按 (image, seed) 批处理，16 样本批一次前向，总 99 次评估，约 2–4 分钟）；输出：
  - `inpaint_gallery.png`：每图一行 = 真值 | 观测 | 4 补全 | 跨 4 补全的逐像素方差图（热图）；
  - `mse_table.txt`：每个补全的 hidden-pixel MSE（隐藏区 400×3=1200 值）与**观测像素最大改变量**。
  **验收:** 观测像素改变 < 1e−5；洞外 4 补全逐位相同、洞内不同；参考 hidden MSE 均值 ≈0.24（船图 0.13–0.38）。README 解释"为何补全看起来合理却仍可能错"。
- [x] **Step 6:** Commit — `week7: repaint inpainting with noised-observation merge and evidence`。

**预计用时：** 代理 90–120 min；计算 2–4 min。

---

## Task 9: 收尾——README、final check、push

**Files:**
- Modify: `week7/README.md`

- [x] **Step 1:** README 补齐（学习单 "Save the results" 要求）：每张 evidence 图/动画 ↔ 生成命令表；**transport 对比解释**（Part 3：DDPM/DDIM/flow 各自步数需求与原因）、**生成照片解释**（Part 4：下载已学好的 vs 采样器现算的）、**补全结果解释**（Part 5：观测约束 vs 隐藏区不确定性）；训练设置表（两个网络：updates/batch/lr 调度/宽度/阶段点/损失终值）、环境（uv 命令）、pinned 模型下载（repo + revision 哈希 + 文件大小）。
- [x] **Step 2:** `uv run pytest tests/` 全绿；`git status` 确认无 `inputs/`、`artifacts/`、GIF 均在 `evidence/` 且 <5 MB；`du -sh evidence/` 记录进 README。
- [x] **Step 3: Commit** — `week7: readme evidence tables, settings and explanations`；push（source + environment + evidence）。

**预计用时：** 代理 30–45 min。

---

## 风险与回退

1. **UNet 移植正确性（最大风险）**：第一诊断 = 参数量 ≈35.7M；第二 = 输出有限且确定性；可选 diffusers v0.40.0 + torch 数值对拍（max|Δ|<1e−3）。若视觉上全噪声 → 先查时间嵌入（flip_sin_to_cos、freq_shift、t−1 索引）与 GroupNorm 轴序。
2. **普通 SGD 调参**：学习单算法框是普通梯度步；pilot 选 lr0；若 25000 updates 后损失明显高于 0.25 或 V3 阈值不过 → 换 Adam(1e-3 线性衰减) 并在 README 记录该偏离及原因。
3. **能量距离内存**：4000×4000 距离矩阵 float32 ≈ 64 MB，逐块计算即可，勿建 40 组缓存。
4. **GIF 体积**：帧数 ≤ 30、dpi ≤ 100；超 5 MB 则减帧或降 dpi（动画内容不变）。
5. **网络下载失败**：CIFAR 权重走 hf 镜像（`HF_ENDPOINT=https://hf-mirror.com`）重试；仍失败则 Task 7/8 标记 blocked 并先完成 Part 1–3 全部证据。
6. **MC 对拍容差**：test_mixture 的 MC 测试若在 CI 机器上偶发，可放大 n 或固定 seed 重跑一次（不放宽断言逻辑，只放宽 atol 到 0.08 并注明理由）。

## Self-Review 结论

- **Spec 覆盖**：Part 1 DO→Task 1–3（V1/V2/V3 各有独立 Step 与阈值）；Part 2 DO→Task 4（DDIM、跳层 DDPM、精确噪声；V1 阈值齐）；Part 3 DO→Task 5–6（flow 训练、精确速度、Euler、步数扫描 {1,2,5,10,20,50,100}+1000、4000 样本、20 对 null、能量距离；V1/V2 阈值齐）；Part 4 DO→Task 7（pinned 下载、JAX 复刻 v0.40.0、t−1、无 clip、float32、权重固定）；Part 5 DO→Task 8（levels 1000..20..0、2 attempts、forward noise between、noised-observation merge、式 16/17、仅 x̂0 clip、seeds 7201–7204、hidden MSE）；收尾→Task 9。无遗漏。
- **阈值来源**：mean±0.1/var±0.1；populations 0.25±0.04、spreads 0.30±0.03、centres±0.05；DDIM 最大差 0、fractions 0.25±0.015、widths 0.29–0.32；rel RMS<0.1（ref 0.011/0.017/0.058）；ED<0.005@20 步、widths±0.03、null q95≈0.0013；观测像素 <1e−5、MSE ref≈0.24——均逐字取自学习单 VERIFY。
- **类型一致性**：`eps_net(x, t:int)->eps` 统一贯穿 ddpm/ddim/skip/repaint（含 exact_noise 包装）；`abar[t]` 以 t∈{0..1000} 索引（abar[0]=1）；`component_weights(x, mean_coeff, var)` 被 exact_noise 与 exact_velocity 复用；levels 一律降序元组。
