# Week 6: Energy-Based Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 AMAT5315 Week 6 学习单的全部四个 Part：数据集检查、JAX RBM 生成式手写数字模型、Hopfield 联想记忆的容量测量、modern Hopfield（softmax attention）数字检索，并产出全部 evidence 图与 README。

**Architecture:** 纯 CPU JAX + NumPy。每个 Part 一个核心库模块（可测试的纯函数）+ 若干脚本（训练/实验/绘图，产出 `evidence/` 图和 `artifacts/` 原始数据）。数据解析独立成 `datasets.py` 供 Part 1/2/4 复用。测试先写、失败后再实现，每个 Task 一次 commit。

**Tech Stack:** Python 3.12+ (uv 管理), jax (CPU), numpy, matplotlib, pytest。

**Spec:** `week6/week6-learning-sheet.pdf`（16 页，已通读）。计划中的所有验收阈值均抄自该学习单的 VERIFY 部分。

## Global Constraints

- 工作目录 `week6/`，使用 **uv** 管理环境（`uv run python ...`）；CPU 是基线，不用 GPU。
- 图存 `evidence/`（进 git），原始运行结果 + 设置存 `artifacts/`（**不进 git**），下载的数据（`inputs/`）**不进 git**。
- 每完成一个 Task 就 commit，消息前缀 `week6: ...`；有不确定处先问，不猜。
- 训练参考设置（学习单指定）：**128 hidden units，20 epochs，plain gradient descent，lr=0.1，batch=100**；CD-1 与 CD-10 必须共享初始化、数据顺序和全部其他设置，只差链长。
- 二值化阈值：`v_i = 1 if pixel ≥ 128 else 0`；自旋表示 `s_i = 2v_i - 1`。
- 绘图保持 native pixels（`imshow(..., interpolation="nearest")`），不放大数据不加细节。
- 下载命令把 URL 追加进 `week6/README.md`（学习单脚本自带 `--write-out ... >> README.md`）。

---

## File Structure

```
week6/
├── README.md                  # provenance、环境、每张 evidence 图 ↔ 命令表（Task 0 建，Task 8 完成收尾）
├── pyproject.toml, uv.lock    # uv 环境（Task 0）
├── .gitignore                 # inputs/, artifacts/, .venv/（Task 0）
├── inputs/                    # mnist/ cifar10/ 下载原始数据，不进 git（Task 0）
├── datasets.py                # IDX/CIFAR 解析 + 二值化 + 切分（Task 1）
├── gallery.py                 # 灰度/RGB 画廊 + 类频率图（Task 1）
├── rbm.py                     # RBM 核心数值函数（Task 2）
├── hopfield.py                # 经典 Hopfield（Task 5）
├── modern_hopfield.py         # modern Hopfield + attention 等价（Task 7）
├── scripts/
│   ├── download.py            # 学习单 curl 命令的封装，或直接照抄 shell（Task 0）
│   ├── part1_gallery.py       # 图: mnist_gallery, cifar_gallery, class_freq（Task 1）
│   ├── part2_train.py         # CD-1/CD-10 训练 + 曲线 + 存参数（Task 3）
│   ├── part2_generate.py      # 重构 vs 生成，1/10/100/1000 步（Task 4）
│   ├── part2_complete.py      # 补全 + ink overlap（Task 4b，见 Task 4 合并执行）
│   ├── part3_sweep.py         # 负载扫描 α∈[0.05,0.25]×N×20 sets（Task 6）
│   └── part4_retrieve.py      # 检索网格 + 画廊（Task 8）
├── tests/
│   ├── test_datasets.py       # (Task 1)
│   ├── test_rbm.py            # (Task 2)
│   ├── test_hopfield.py       # (Task 5)
│   └── test_modern.py         # (Task 7)
└── evidence/ artifacts/       # 图进 git；artifacts 不进 git
```

---

## Task 0: 环境与数据下载

**Files:**
- Create: `week6/pyproject.toml`, `week6/.gitignore`, `week6/README.md`, `week6/inputs/`

**Interfaces (Produces):**
- 命令 `cd week6 && uv run python <script>` 可运行任何后续脚本。
- `inputs/mnist/*.gz`（4 个文件）、`inputs/cifar10/cifar-10-binary.tar.gz`（已解压出 `data_batch_*`、`test_batch`）。

- [x] **Step 1: 安装 uv**（系统当前没有）：
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # 或 pip install --user uv
  ```
- [x] **Step 2: 初始化项目并加依赖**：
  ```bash
  cd week6
  uv init --no-readme --python 3.12
  uv add jax numpy matplotlib pytest
  ```
  与 week5 的版本对齐（jax==0.11.x, numpy, matplotlib, pytest）。
- [x] **Step 3: 写 `.gitignore`**：
  ```
  inputs/
  artifacts/
  .venv/
  __pycache__/
  ```
- [x] **Step 4: 按学习单原样执行下载**（命令会把 URL 追加到 README.md）：
  ```bash
  mkdir -p inputs/mnist inputs/cifar10
  mnist_url=https://storage.googleapis.com/cvdf-datasets/mnist
  for split in train t10k; do
    for kind in images-idx3-ubyte labels-idx1-ubyte; do
      name="$split-$kind.gz"
      curl -fL "$mnist_url/$name" -o "inputs/mnist/$name" --write-out "%{url_effective}\n"
    done
  done >> README.md
  curl -fL https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz \
    -o inputs/cifar10/cifar-10-binary.tar.gz \
    --write-out "%{url_effective}\n" >> README.md
  tar -xzf inputs/cifar10/cifar-10-binary.tar.gz -C inputs/cifar10
  ```
- [x] **Step 5: Commit** — `git add week6/pyproject.toml week6/uv.lock week6/.gitignore week6/README.md` → `week6: uv environment and dataset provenance`（确认 `inputs/` 未被跟踪）。

**预计用时：** 代理 15–25 min；下载约 10–15 min（MNIST ≈11 MB + CIFAR ≈170 MB，视网速）。

---

## Task 1 (Part 1): 数据解析、画廊与类频率

**Files:**
- Create: `week6/datasets.py`, `week6/gallery.py`, `week6/scripts/part1_gallery.py`
- Test: `week6/tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `load_mnist(split) -> (images uint8 (N,28,28), labels int8 (N,))`，split ∈ {"train","test"}
  - `load_cifar10(split) -> (images uint8 (N,32,32,3), labels (N,))`
  - `binarize(imgs) -> bool (N,28,28)`（`>= 128`）
  - `class_frequencies(labels, n_classes=10) -> float[10]`
- 验收（学习单 VERIFY 1–2）：MNIST 60000/10000；digit 1 最多、digit 5 最少（两个 split 皆是）；CIFAR 50000/10000，每类恰 5000/1000（全部频率 0.1）。

- [x] **Step 1: 失败测试**（IDX 魔数 2051/2049、大端；CIFAR 记录 = 1 label 字节 + 3072 像素字节，RGB 三个平面）：
  ```python
  # tests/test_datasets.py
  import numpy as np
  from datasets import load_mnist, load_cifar10, binarize, class_frequencies

  def test_mnist_shapes_and_split():
      tr, tr_y = load_mnist("train"); te, te_y = load_mnist("test")
      assert tr.shape == (60000, 28, 28) and tr.dtype == np.uint8
      assert te.shape == (10000, 28, 28)
      assert 0 <= tr.min() and tr.max() <= 255

  def test_mnist_class_frequencies_match_reference():
      f_tr = class_frequencies(load_mnist("train")[1])
      assert f_tr.argmax() == 1 and f_tr.argmin() == 5
      f_te = class_frequencies(load_mnist("test")[1])
      assert f_te.argmax() == 1 and f_te.argmin() == 5

  def test_cifar_shapes_and_balance():
      tr, tr_y = load_cifar10("train"); te, te_y = load_cifar10("test")
      assert tr.shape == (50000, 32, 32, 3)
      assert np.allclose(class_frequencies(tr_y), 0.1)
      assert np.allclose(class_frequencies(te_y), 0.1)

  def test_binarize_threshold_at_128():
      assert binarize(np.array([[[127, 128, 255]]])) .tolist() == [[[False, True, True]]]
  ```
- [x] **Step 2:** `uv run pytest tests/test_datasets.py` → FAIL（模块不存在）。
- [x] **Step 3: 实现** `datasets.py`（用 `struct.unpack(">IIII", ...)` 解头、`np.frombuffer` 读像素；CIFAR 逐记录 `reshape(3,32,32).transpose(1,2,0)`）与 `gallery.py`（`imshow(..., interpolation="nearest", cmap="gray" / None)`，每类等量采样；频率图双子图 train/test）。
- [x] **Step 4:** 测试 PASS；跑 `scripts/part1_gallery.py` 产出 `evidence/part1/{mnist_gallery,cifar_gallery,mnist_class_freq,cifar_class_freq}.png`。
- [x] **Step 5: Commit** — `week6: part1 dataset galleries and class frequencies`。

**预计用时：** 代理 45–60 min；计算 <1 min。

---

## Task 2 (Part 2 核心): RBM 数值函数（TDD）

**Files:**
- Create: `week6/rbm.py`
- Test: `week6/tests/test_rbm.py`

**Interfaces (Produces, 全部纯函数、接受 params dataclass `(a (784,), b (128,), W (784,128))`):**
- `energy(v, h, p) -> float`：`-a·v - b·h - v·W·h`
- `free_energy(v, p) -> float`：`-a·v - Σ_j softplus(b_j + (vW)_j)`
- `gibbs_vh(v, key, p) -> v'`（先采样 h|v 再 v|h，各一次二项采样）
- `cd_surrogate(v0, vk, p) -> float`：`mean(F(v0)) - mean(F(stop_gradient(vk)))`
- `hamming_fraction(v0, v1) -> float`：`mean(v0 != v1)`
- `init_params(key, n_vis=784, n_hid=128, scale=0.01)`

- [x] **Step 1: 失败测试**（核心技巧：在 n_vis=4, n_hid=3 的小 RBM 上与暴力枚举对拍）：
  ```python
  # tests/test_rbm.py
  import itertools, numpy as np, jax, jax.numpy as jnp
  from rbm import energy, free_energy, gibbs_vh, cd_surrogate, init_params

  def brute_force_free_energy(v, p):
      F = -jnp.dot(p.a, v)
      for hj in itertools.product([0., 1.], repeat=p.b.shape[0]):
          h = jnp.array(hj)
          F = F - jax.nn.logsumexp(jnp.array([0.0]))  # placeholder removed below
      # 正确写法见下：F(v) = -a·v - logsumexp(0, b + vW) 逐 bit 独立 → softplus
      ...
  ```
  实际采用的更干净的对拍（避免上面占位）：
  ```python
  def brute(v, p):
      vals = []
      for hj in itertools.product([0., 1.], repeat=len(p.b)):
          h = jnp.array(hj)
          vals.append(-energy(v, h, p))
      return -jax.nn.logsumexp(jnp.array(vals))

  def test_free_energy_matches_logsumexp_over_hidden():
      p = init_params(jax.random.PRNGKey(0), n_vis=4, n_hid=3)
      v = jnp.array([1., 0., 1., 1.])
      assert abs(free_energy(v, p) - brute(v, p)) < 1e-5

  def test_gibbs_step_preserves_shapes_and_returns_binary():
      p = init_params(jax.random.PRNGKey(1), n_vis=4, n_hid=3)
      v = jnp.array([1., 0., 1., 0.])
      v2 = gibbs_vh(v, jax.random.PRNGKey(2), p)
      assert v2.shape == v.shape and set(np.asarray(v2)) <= {0.0, 1.0}

  def test_cd_surrogate_gradient_is_free_energy_difference():
      # dL/dW = ∇F(v0) - ∇F(vk)：手算小例对拍
      p = init_params(jax.random.PRNGKey(3), n_vis=2, n_hid=2)
      v0, vk = jnp.array([1., 0.]), jnp.array([0., 1.])
      g = jax.grad(lambda W: cd_surrogate(v0, vk, p._replace(W=W)))(p.W)
      g_expect = (jax.grad(lambda W: free_energy(v0, p._replace(W=W)))(p.W)
                  - jax.grad(lambda W: free_energy(vk, p._replace(W=W)))(p.W))
      assert jnp.allclose(g, g_expect, atol=1e-6)

  def test_hamming_fraction():
      assert hamming_fraction(jnp.zeros(4), jnp.array([0., 1., 0., 0.])) == 0.25
  ```
- [x] **Step 2:** 跑测试确认 FAIL。
- [x] **Step 3: 实现** `rbm.py` 关键片段：
  ```python
  def free_energy(v, p):
      pre = p.b + v @ p.W                      # (n_hid,)
      return -jnp.dot(p.a, v) - jnp.sum(jax.nn.softplus(pre))

  def gibbs_vh(v, key, p):
      k1, k2 = jax.random.split(key)
      h = jax.random.bernoulli(k1, jax.nn.sigmoid(p.b + v @ p.W)).astype(v.dtype)
      v2 = jax.random.bernoulli(k2, jax.nn.sigmoid(p.a + h @ p.W.T)).astype(v.dtype)
      return v2

  def cd_surrogate(v0, vk, p):
      return jnp.mean(free_energy(v0, p)) - jnp.mean(free_energy(jax.lax.stop_gradient(vk), p))
  ```
  Gibbs 的 k 步链用 `jax.lax.fori_loop`（或 `jax.lax.scan`）实现 `gibbs_chain(v, key, p, k)`。
- [x] **Step 4:** 测试 PASS。
- [x] **Step 5: Commit** — `week6: rbm energy, free energy, block gibbs, cd surrogate with tests`。

**预计用时：** 代理 45–60 min；计算秒级。

---

## Task 3 (Part 2 训练): CD-1 vs CD-10 曲线

**Files:**
- Create: `week6/scripts/part2_train.py`
- Uses: Task 1 的 `load_mnist/binarize`、Task 2 的 `rbm.py`

**Interfaces:**
- Produces: `evidence/part2/learning_curves.png`（双子图：CD surrogate + held-out Hamming fraction，各两条曲线 CD-1/CD-10）、`evidence/part2/spins_gallery.png`（前 16 张二值化训练图 + 一张放大的自旋配置 + RBM 结构示意）、`artifacts/rbm_cd1.npz`、`artifacts/rbm_cd10.npz`（参数 + 每 epoch 记录 + 设置 JSON，不进 git）。
- 训练设置：128 hidden、20 epochs、lr=0.1 plain GD、batch=100、**两模型用同一个 PRNGKey 初始化和同一次数据打乱序列**；epoch 内记录 surrogate；每 epoch 在 10000 张测试图上算一步重构 Hamming fraction。

- [x] **Step 1:** 实现 `part2_train.py`：`jax.jit` 的批更新 `θ ← θ - 0.1 · ∇_θ L̃_CD`；`scan` 跑 20 epochs；结束存 npz + 设置。
- [x] **Step 2:** 运行并核对学习单阈值：CD-1 ≈ 18 s、CD-10 ≈ 105 s（参考值，本机量级相同即可）；重构误差应从 ~17% 降到 ~4.7%，**必须低于初始值的一半**；surrogate 可升可降（在 README 里解释为什么不能当 NLL 曲线）。
- [x] **Step 3:** 生成 `spins_gallery.png`（验证 VERIFY 1：标出 +1/−1 像素对应关系；结构图可用 matplotlib 画两层 + 连线示意）。
- [x] **Step 4: Commit** — `week6: part2 rbm training curves for cd-1 and cd-10`。

**预计用时：** 代理 45–60 min；计算 3–5 min。

---

## Task 4 (Part 2 采样): 重构 vs 生成、补全与 ink overlap

**Files:**
- Create: `week6/scripts/part2_generate.py`（重构 + 生成链）、`week6/scripts/part2_complete.py`（补全 + overlap）

**验收（学习单 VERIFY 3–4）：**
- 两个 RBM 从**相同的随机自旋初始配置**（±1 等概率，即 v 随机 0/1）出发，保留 1/10/100/1000 步样本的画廊；1000 步样本应呈 blob/环状连通笔画（非孤立像素）。
- 补全：64 个 held-out 数字 × 8 条链 × {1,10,100,1000} 步，clamp 上半部；画廊显示 1/10/100/1000 步；ink overlap `2|A∩B|/(|A|+|B|)` vs 步数曲线（CD-1 与 CD-10 各一条），再画"错误上半部"（取自另一数字）对照。
- 阈值：**1000 步时 true-half overlap 比 wrong-half 高至少 0.1**（参考：CD-10 ≈0.51±0.02，CD-1 ≈0.40±0.02；wrong ≈0.27/0.24，且 1 步时两者几乎相同）。

- [x] **Step 1:** `part2_generate.py`：同一 key 初始化 N_CHAINS 条链，`gibbs_chain` 各步数快照；并排展示若干测试图的一步重构作对照。检查两模型起点一致（assert 同一 v0）。
- [x] **Step 2:** `part2_complete.py`：clamp 实现 = 每次 `v ← v.at[:, :14].set(upper_half)`（Gibbs 步后重置观测位）；错误上半部实验把 clamp 换成另一数字的上半部。计算 overlap 均值 ± 标准误（64 个数字上）。
- [x] **Step 3:** 产出 `evidence/part2/generation_chains.png`、`completion_gallery.png`、`ink_overlap.png`；核对阈值；把结论写进 README（不同链补全不同、CD-1 常把 1 的下半留白等观察）。
- [x] **Step 4: Commit** — `week6: part2 generation chains and clamped completion with ink overlap`。

**预计用时：** 代理 60–75 min；计算 5–10 min（参考：两核 CPU 约 5 min）。

*（可选扩展，不列入必做：AIS 估计 held-out NLL。若时间富余再做，学习单 §Optional。）*

---

## Task 5 (Part 3 核心): Hopfield 网络（TDD）

**Files:**
- Create: `week6/hopfield.py`
- Test: `week6/tests/test_hopfield.py`

**Interfaces (Produces):**
- `hebbian(xi) -> J`：`J = (1/N) ξ ξᵀ`，`J_ii = 0`，xi 为 (N,P)
- `local_field(s, J) -> h`
- `recall(s0, J, key) -> s_final, n_flips`：异步更新 `s_i ← sign(h_i)`（`h_i=0` 不动），随机顺序，直到一整轮无翻转（`jax.lax.while_loop` + `scan`；上界如 100 轮防死循环）
- `overlap(s, xi_mu) -> m`、`unstable_fraction(J, xi) -> float`（所有存储模式上 `ξ_i h_i < 0` 的比例）

- [x] **Step 1: 失败测试：**
  ```python
  # tests/test_hopfield.py
  import jax, jax.numpy as jnp, numpy as np
  from hopfield import hebbian, recall, unstable_fraction, overlap, energy

  def test_diagonal_zero_and_symmetry():
      rng = np.random.default_rng(0)
      xi = jnp.asarray(rng.choice([-1., 1.], size=(50, 5)))
      J = hebbian(xi)
      assert jnp.all(jnp.diag(J) == 0) and jnp.allclose(J, J.T)

  def test_recall_restores_pattern_at_low_load():
      rng = np.random.default_rng(1)
      xi = jnp.asarray(rng.choice([-1., 1.], size=(100, 3)))
      J = hebbian(xi)
      s0 = xi[:, 0].at[:20].multiply(-1.0)   # 翻转 20% 位
      s, _ = recall(s0, J, jax.random.PRNGKey(0))
      assert overlap(s, xi[:, 0]) == 1.0     # P=3, α=0.03 << 0.138

  def test_unstable_fraction_near_zero_at_low_load():
      assert unstable_fraction(hebbian(xi_n100_p5), xi_n100_p5) < 5e-3
  ```
  另加一个能量单调测试：recall 每轮结束后 `E(s)` 不增。
- [x] **Step 2:** FAIL → **Step 3: 实现**（注意 sign(0)=0 的坑：`jnp.where(h == 0, s_i, jnp.sign(h))`）→ **Step 4: PASS**。
- [x] **Step 5: Commit** — `week6: hopfield hebbian couplings and asynchronous recall with tests`。

**预计用时：** 代理 45–60 min；计算秒级。

---

## Task 6 (Part 3 实验): 负载扫描与容量坍缩

**Files:**
- Create: `week6/scripts/part3_sweep.py`

**设置（学习单 DO 原文）:** α 从 0.05 到 0.25 步长 0.01（21 点）× N ∈ {100,200,400,800} × 20 套独立随机模式；`P = round(αN)`；每套从**前 10 个模式**（P<10 时全部）出发做 recall 记录最终 overlap；记录 seeds 与 runtime 到 `artifacts/part3/settings.json`。

**验收（VERIFY 1–2）：**
- 图 1：unstable-bit fraction vs α（各 N 一条曲线 + 误差棒），叠加理论 `Φ(-1/√α)` 与有限 N 形式 `Φ(-√((N-1)/(P-1)))`；参考值：N=800, α=0.20 → 1.25e-2（理论 1.27e-2）；N=100, α=0.10 → 3.5e-4（有限 N 预测 4.6e-4）。若出现 ~0.5 说明 J 建错；若小 4 倍说明忘了置零对角。
- 图 2：mean final overlap 与 P(m ≥ 0.9) 比例 vs α，各 N 一条曲线，标出 α=0.138 竖线。**crossing（m 跌破 0.9 处）须随 N 递减，且落在参考值 ±0.02 内**：N=100/200/400/800 → 0.186/0.170/0.156/0.150。

- [x] **Step 1:** 实现（`jax.vmap` 或逐 set 循环；recall 是纯 Python 循环 + jit 的单轮 scan，性能足够：参考 7 min @ 2 核）。
- [x] **Step 2:** 运行、出图 `evidence/part3/{unstable_bits,capacity}.png`、对照阈值。
- [x] **Step 3: Commit** — `week6: part3 hopfield load sweep matches crosstalk prediction`。

**预计用时：** 代理 45–60 min；计算 7–10 min。

---

## Task 7 (Part 4 核心): modern Hopfield = attention（TDD）

**Files:**
- Create: `week6/modern_hopfield.py`
- Test: `week6/tests/test_modern.py`

**Interfaces (Produces):**
- `modern_update(x, X, beta) -> x'`：`X @ softmax(beta * (X.T @ x))`，X (784,P) 列为存储的 ±1 模式
- `attention_update(x, X, beta) -> x'`：`jax.nn.dot_product_attention`（或手写 `Vᵀ softmax(β K q)`，K=V=Xᵀ, q=x）
- `retrieval_success(x_ret, X, target_idx) -> bool`：`overlap(sign(x_ret), X[:,target]) ≥ 0.95` **且** target 是与 sign(x_ret) 重叠最大的存储模式
- 经典对照复用 Task 5 的 `recall`（模式即 X 的列）。

- [x] **Step 1: 失败测试：**
  ```python
  # tests/test_modern.py
  import jax, jax.numpy as jnp, numpy as np
  from modern_hopfield import modern_update, attention_update

  def test_update_equals_attention_below_1e5():
      rng = np.random.default_rng(0)
      X = jnp.asarray(rng.choice([-1., 1.], size=(784, 100)).astype(np.float32))
      x = jnp.asarray(rng.normal(size=784).astype(np.float32))
      for beta in [0.001, 0.03, 0.3, 1.0]:
          d = jnp.max(jnp.abs(modern_update(x, X, beta) - attention_update(x, X, beta)))
          assert float(d) < 1e-5   # 参考: 6.0e-7 @ fp32；若 ~2 说明 scale 传错

  def test_small_beta_averages_large_beta_retrieves():
      q = X[:, 0]
      w_small = jax.nn.softmax(0.001 * (X.T @ modern_update(q, X, 0.001)))
      assert float(w_small.max()) < 0.05          # β=0.001: 权重近均匀
      s_big = jnp.sign(modern_update(q, X, 10.0))
      assert float(jnp.mean(s_big == q)) > 0.95   # 大 β: 一次更新检索回目标
  ```
- [x] **Step 2:** FAIL → **Step 3: 实现**（注意 `dot_product_attention` 的 scale 参数就是 beta，不要用默认 `1/√d`）→ **Step 4: PASS**。
- [x] **Step 5: Commit** — `week6: modern hopfield update equals one softmax attention step`。

**预计用时：** 代理 30–45 min；计算秒级。

---

## Task 8 (Part 4 实验 + 收尾): 检索网格、画廊、README、final verification

**Files:**
- Create: `week6/scripts/part4_retrieve.py`
- Modify: `week6/README.md`

**设置（学习单 DO 原文）:** 存 P 个二值化 MNIST 训练图为自旋（X 的列）；查询 100 个存储数字（循环遍历），每个把下半部换成随机自旋；对 **P ∈ {3,10,30,100,300,1000} × β ∈ {0.001,0.003,0.01,0.03,0.1,0.3,1}** 记录 modern 一步检索成功率，以及同样式上的经典 recall 成功率。

**验收（VERIFY 1–3）：**
- softmax 权重图（一个查询 × β ∈ {0.003, 0.03, 0.3}；参考权重 0.01/0.25/1.00）+ 打印 update vs attention 最大差（<1e-5）。
- β≤0.01 全部返回同一张模糊平均数字；β≥0.1 一步取回存储数字；经典 recall 对每个查询都返回同一堆稀疏笔画碎片。
- 成功率 vs P 曲线（modern 多条 β + classical）：**β=0.3 时每个 P 成功率 ≥ 0.95；classical 恒为 0**。P=100 的画廊：targets / queries / 两种检索并排。

- [x] **Step 1:** 实现 `part4_retrieve.py`（modern 一步是矩阵乘，秒级；classical recall 到 P=1000 用 Task 5 的 recall，从查询位形出发）。
- [x] **Step 2:** 出图 `evidence/part4/{attention_weights, beta_gallery, success_vs_P}.png`，核对阈值。
- [x] **Step 3: README 收尾**（学习单 Wrap up）：每张 evidence 图 ↔ 生成命令表；按 Final verification 清单写解释——p_data vs 经验分布 vs 拟合模型 vs 采样算法的区分、CD 在哪近似似然梯度、clamp 如何把生成变成推断、crosstalk 曲线与坍缩、attention ≡ modern Hopfield、β 的作用；记录环境、训练设置、数据 provenance。
- [x] **Step 4:** `uv run pytest tests/` 全绿；`git status` 确认无 inputs/artifacts 混入；Commit — `week6: part4 retrieval grid, readme evidence and final verification`。

**预计用时：** 代理 60–90 min；计算 5–15 min。

---

## Self-Review 结论

- **Spec 覆盖：** Part 1→Task 1；Part 2 DO/VERIFY 1-2→Task 2-3，VERIFY 3→Task 4a（generate），VERIFY 4→Task 4b（complete）；Part 3→Task 5-6；Part 4→Task 7-8；Wrap-up/Final verification→Task 8 Step 3。AIS 可选项明确标为可选。无遗漏。
- **阈值来源：** recon < 初始值一半；overlap 差 ≥0.1；attention 差 <1e-5；crossing ±0.02；β=0.3 成功率 ≥0.95、classical=0——均逐字取自学习单。
- **类型一致性：** `params (a,b,W)`、`X (N,P) 列为模式`、`recall(s0,J,key)`、`overlap(s,ξ)` 在各 Task 间一致。
