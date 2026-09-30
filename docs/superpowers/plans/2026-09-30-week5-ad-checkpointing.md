# Week 5：自动微分与检查点（AD / 地震模拟 / Treeverse）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按第 5 周学习单完成 `week5/`：Part 1 用 JAX 手写 Lennard-Jones 对势能的前向/反向微分并与有限差对比；Part 2 用 Rust 写 `seismic` CLI 的声波正演；Part 3 用 Enzyme 加 `born`（JVP 链）与 `adjoint`（全历史 VJP 链）模式并做转置恒等式验证；Part 4 移植 Treeverse 检查点调度，用 6 个状态复现图像并给 Marmousi 成像，最后整理 evidence 并推送。

**Architecture:** Python 侧放 `week5/scripts/`（独立 venv，JAX 64 位），产出 `artifacts/ad/*`。Rust 侧是 `week5/seismic/` 单 crate（`cargo install --path seismic`）：普通 Rust 的步进函数（Part 2，正演与独立校验）+ `src/kernel.rs`（no_std Enzyme 内核，由 `build.rs` 单独编译，Part 3 起）提供 step 的 primal/JVP/VJP，`born.rs`/`adjoint.rs` 在 Rust 循环里把它们按时间链起来；`treeverse.rs` 先生成纯调度（store/restore/call/grad/fetch 动作表）再由执行器回放。所有数组走自写的 npy 读写模块，绘图/对比全部由 `scripts/*.py` 完成，产出 `artifacts/`。

**Tech Stack:** Python 3.14 venv（jax 0.11.2 CPU、numpy、matplotlib、pytest）；Rust edition 2024（anyhow、clap 4 derive、serde/serde_json；Part 3 起钉 `nightly-2026-09-05` + enzyme 组件，内核经 `build.rs` 以 `-Zautodiff=Enable` 编成 staticlib，走 C ABI）。

**Spec:** `week5/week5-learning-sheet.pdf`（权威；本计划所有数值验收、绿色面板文案、会话规则均出自它）。参考实现：Enzyme setup skill（Task 14 安装到 `.pi/skills/enzyme-setup/`）、TreeverseAlgorithm.jl `src/treeverse.jl`（Task 21 移植）。

## Global Constraints（全文约束，每个任务默认继承）

会话规则（学习单 Part 1 DO 面板）：

- 在 `week5/` 内工作，所有相对路径以它为根；**每个任务完成后立即 commit**；不确定先问，不要猜。
- run/对比/绘图脚本放 `week5/scripts/`；图用 matplotlib 存 PNG。
- `inputs/` 与所有 `.npy` 不入 git；其余 evidence（png/json/md/license）入 git，**每个文件 < 5 MB**。
- 提交信息前缀 `week5: ...`。

数值与格式约定（学习单 Part 2 UNDERSTAND）：

- 约化单位：1 长度单位 = 100 m，1 时间单位 = 0.1 s，1 速度单位 = 1 km/s。reflector 实验 Δx=1、Δt=0.2、f0=0.08、t0=15、240 步。
- 场数组按深度优先 `[z][x]` 展平（`k = z*nx + x`）；实验坐标是 `[x, z]` 网格索引。
- 更新公式（式 9）：`u^{n+1} = (2u^n − (1−σΔt)u^{n−1} + Δt²(c²L_h u^n + q^n)) / (1+σΔt)`，外圈格点恒零，起始两场为零，无半步起步。
- 海绵：`σ = σmax·max(0, 1−ℓ/W)²`，ℓ 为距最近边界的格点数，reflector 用 W=6、σmax=0.4。
- 源：`q = source_amplitude · g(nΔt) · footprint`，Ricker `g(t) = (1−2θ²)e^{−θ²}`、`θ = π f0 (t−t0)`；footprint 是以炮点为中心、标准差 1 格的 Gaussian（峰值 1）。接收器在每步更新后采样 `u^{n+1}`。
- 录制帧：forward 记 `u^n`（n = 0, every, 2·every, …, ≤steps）；adjoint 记完成 reverse step n 后的 `ū^n`（n = steps−1, …, 0 中 n ≡ 0 mod every）。`run.json` 的 recording 含 every、steps、times(=step·dt)，按帧序。
- npy：traces/born_data/image 用 float64，录制用 float32；L2 范数 = 各元素平方和的平方根。

验收参考值（answer key，出错即停下排查）：

- Part 1：`r=1.3` 处 energy = −0.6570169144600471、jax_grad = 2.239979929791143、adjoints.a = −2.3425903117359734（各 1e-12 内）；两种手写 AD 最大误差 < 1e-12（参考 < 3e-14），FD(h=1e-6) 约 5e-9 且大于两个 AD；cluster 误差 < 1e-12（参考 2e-15），P=3072 时 forward/energy 比值超过 reverse 的 100 倍，reverse 比值 1.2–1.5。
- Part 2：全部 traces 的 L2 = 11.574770（相对误差 < 1e-4）；每炮最大 |压力| 在 trace 83（第 84 步后）：边炮 0.60809514、中炮 0.59271397（1e-4 内）；echo 峰值约 0.006 ≈ 同步直达人波峰值 0.26 的 2%。
- Part 3：转置恒等式 ‖d‖² = 0.0348479 双侧一致，相对差 < 1e-9（参考 1e-15）；图像行 L2 剖面峰在 z=21（2.1 km）一格内（窗口 x=7..33、z=10..33 含端点）。
- Part 4：δ=1,3,5,10 的 checkpoint 图像与全历史图像相对 L2 误差 < 1e-9（参考为 0）；峰值保存状态数 2,4,6,11；每炮调度 forward 步数 28,680 / 1,695 / 990 / 642（全历史 240 步、241 状态、6,481,936 字节；单状态 = 2·nx·nz·8 字节）；审计三项计数全为 0；Marmousi 图像 L2 = 6.7037741×10⁻⁴（1e-4 内），6 状态 = 20,788,320 字节。**绝不以 full history 跑 Marmousi**（1201 状态 ≈ 4.16 GB）。

工程约束：

- Python：系统 python3 = 3.14.4（PEP 668，禁止直接 pip），用 `week5/.venv`；版本钉在 `week5/requirements.txt`。JAX 必须开 64 位（`jax.config.update("jax_enable_x64", True)`）。
- Rust：Part 2 用默认 stable（1.98.1）；Task 14 起在 `week5/seismic/` 钉 `nightly-2026-09-05`（+enzyme 组件）。Enzyme 内核独立于主 crate（`build.rs` 编译，绝不 `mod kernel`，绝不在 `.cargo/config.toml` 开 `-Zautodiff`）。**Enzyme 必须自己微分物理步进；手写切线/伴随只许出现在独立校验测试里。**
- 测试先行：Rust 任务每步 `cargo test`（在 `week5/seismic/`），Python 脚本配轻量 pytest（`week5/.venv/bin/python -m pytest scripts/`）。
- `target/`、`__pycache__/`、`.venv/`、`inputs/`、`*.npy` 不入 git；`Cargo.lock` 入 git。

## 开工前待用户确认的问题

1. Python 环境用 `week5/.venv`（venv + requirements.txt）而非 uv，可以吗？
2. Rust crate 放 `week5/seismic/`（因为学习单命令是 `cargo install --path seismic`），可以吗？
3. 学习单要求用"课程查看器"保存 `wavefield.png`/`echo.png`/`artifacts/adjoint/wavefield.png`。计划改为脚本直接渲染同一帧数组（数值相同、可复现），查看器留作你课后人工核对——可以吗？
4. Task 14 需 `rustup toolchain install nightly-2026-09-05 --component enzyme`（约几百 MB 下载）。本机装，还是用课程 Linux 机器？
5. evidence 里 checkpoint 预算 1 的 `actions-<shot>.json` 每个约 1–2 MB（< 5 MB 上限），照提交，可以吗？
6. 学习单要求装完 enzyme skill 后"开新会话再继续"。本计划已把该 skill 的全部要点内嵌进 Task 14–15，执行者可当场继续；如果你想严格照单重开会话，请在 Task 14 完成后手动重启。

## 文件结构总览

```
week5/
├── .gitignore                  inputs/ .venv/ *.npy __pycache__/ .pytest_cache/ week5-extracted.txt
├── requirements.txt            Python 依赖钉版
├── seismic.design.toml         三个面板随 Part 2/3/4 逐字追加
├── README.md                   evidence 清单 + 复现指南（Task 25）
├── MARMOUSI-LICENSE            从 inputs/ 拷出并提交
├── seismic/                    Rust crate（名 seismic，单个 [[bin]] seismic）
│   ├── Cargo.toml  Cargo.lock  rust-toolchain.toml(Task14起)  build.rs(Task14起)
│   ├── src/
│   │   ├── main.rs             clap CLI：--experiment/--mode/--out/--every/--data/--storage/--checkpoints
│   │   ├── lib.rs              模块根
│   │   ├── experiment.rs       实验 JSON 解析（Experiment 结构体）
│   │   ├── survey.rs           ricker、footprint、海绵系数 ad/bd、cfl/dt2
│   │   ├── wave.rs             State、普通 Rust step、接收器采样、L2
│   │   ├── npy.rs              npy 读写（f64/f32）
│   │   ├── output.rs           run.json/result.json/stdout
│   │   ├── forward.rs          forward 模式（含录制与 echo）
│   │   ├── kernel.rs           Enzyme 内核（仅 build.rs 的输入，不 mod）
│   │   ├── enzyme.rs           C ABI 声明 + 安全包装
│   │   ├── born.rs             born 模式
│   │   ├── adjoint.rs          adjoint 模式（full 历史 + 录制）
│   │   └── treeverse.rs        调度生成器 + 执行器 + 动作审计
│   └── tests/                  npy.rs wave.rs experiment.rs survey.rs cli.rs enzyme.rs modes.rs treeverse.rs
├── scripts/
│   ├── ad_lib.py  run_ad.py  ad_graphs.py  ad_scaling.py  test_ad.py
│   ├── plot_inputs.py  plot_forward.py  plot_frames.py
│   ├── rtm_check.py
│   ├── checkpoint_check.py  checkpoint_work.py  test_audit.py
│   └── plot_marmousi.py
├── inputs/                     (gitignore) reflector.json marmousi.json license
└── artifacts/                  (只提交 png/json) ad/ forward/ born/ adjoint/
                                checkpoint-{1,3,5,10}/ marmousi-born/ marmousi-image/ *.png
```

---

# Part 1：前向与反向自动微分（JAX）

## Task 1: Python 环境与 week5 脚手架

**Files:**
- Create: `week5/.gitignore`、`week5/requirements.txt`、`week5/scripts/`、`week5/artifacts/`（不入 git）

**Interfaces:**
- Produces: `week5/.venv/bin/python`（jax+numpy+matplotlib+pytest），后续所有 Python 命令用它执行。

- [ ] **Step 1: 建目录与 venv，装依赖**

```bash
cd week5
mkdir -p scripts artifacts/ad
python3 -m venv .venv
.venv/bin/pip install --quiet jax numpy matplotlib pytest
.venv/bin/pip freeze > requirements.txt
```

- [ ] **Step 2: 写 `week5/.gitignore`**

```gitignore
inputs/
.venv/
__pycache__/
.pytest_cache/
*.npy
week5-extracted.txt
```

- [ ] **Step 3: 冒烟验证（对照学习单参考值）**

```bash
.venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
f = lambda r: 4.0*(jnp.power(r,-12) - jnp.power(r,-6))
print(float(f(jnp.array(1.3))), float(jax.grad(f)(jnp.array(1.3))))
"
```
Expected: `-0.6570169144600471 2.2399799297911427`（本机已验证可复现）。

- [ ] **Step 4: Commit** — `git add week5/.gitignore week5/requirements.txt && git commit -m "week5: python venv and ignore rules"`

## Task 2: 手写前向/反向模式（`scripts/ad_lib.py`，TDD）

**Files:**
- Create: `week5/scripts/ad_lib.py`、`week5/scripts/test_ad.py`

**Interfaces:**
- Produces（后续脚本依赖的确切签名）:
  - `lj(r) -> jnp.scalar`（四节点程序 `a=r**-6; b=a**2; c=b-a; U=4c`）
  - `analytic_dU(r)`、`finite_diff(r, h=1e-6)`
  - `forward_mode(r) -> (vals: dict, tans: dict)`，键 `r,a,b,c,U`
  - `reverse_mode(r) -> (vals: dict, adj: dict)`，键 `r,a,b,c,U`
  - 每个节点的切线/伴随分别用一次 `jax.jvp` / `jax.vjp` 计算（照学习单伪代码）。

- [ ] **Step 1: 先写失败测试 `scripts/test_ad.py`**

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ad_lib import lj, analytic_dU, forward_mode, reverse_mode, finite_diff

def test_hand_modes_match_analytic():
    r = jnp.array(1.3)
    d = float(analytic_dU(r))
    _, t = forward_mode(r)
    _, a = reverse_mode(r)
    assert abs(float(t["U"]) - d) < 1e-12
    assert abs(float(a["r"]) - d) < 1e-12
    assert abs(float(t["U"]) - float(finite_diff(r))) > 1e-12  # FD 不该精确相等

def test_reference_values_at_r_1_3():
    r = jnp.array(1.3)
    vals, t = forward_mode(r)
    _, adj = reverse_mode(r)
    assert abs(float(vals["U"]) - (-0.6570169144600471)) < 1e-12
    assert abs(float(t["U"]) - 2.239979929791143) < 1e-12
    assert abs(float(adj["r"]) - 2.239979929791143) < 1e-12
    assert abs(float(adj["a"]) - (-2.3425903117359734)) < 1e-12

def test_jax_grad_and_adjoints_chain():
    r = jnp.array(1.3)
    assert abs(float(jax.grad(lj)(r)) - float(analytic_dU(r))) < 1e-12
    _, adj = reverse_mode(r)
    assert abs(float(adj["c"]) - 4.0) < 1e-12      # dU/dc = 4
    assert abs(float(adj["b"]) - 4.0) < 1e-12      # dU/db = 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd week5 && .venv/bin/python -m pytest scripts/test_ad.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'ad_lib'`）

- [ ] **Step 3: 实现 `scripts/ad_lib.py`**

```python
"""Lennard-Jones pair energy and its hand-written forward/reverse AD passes.

Node program: a = r^-6, b = a^2, c = b - a, U = 4c.
Each node's tangent uses one jax.jvp; each adjoint step uses one jax.vjp,
following the learning sheet's pseudocode.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def lj(r):
    a = r ** -6
    b = a ** 2
    c = b - a
    return 4.0 * c


def analytic_dU(r):
    return 24.0 * (r ** -7 - 2.0 * r ** -13)


def finite_diff(r, h=1e-6):
    return (lj(r + h) - lj(r - h)) / (2.0 * h)


def forward_mode(r):
    rdot = 1.0                                        # seed input direction
    (a, adot) = jax.jvp(lambda x: x ** -6, (r,), (rdot,))
    (b, bdot) = jax.jvp(lambda x: x ** 2, (a,), (adot,))
    (c, cdot) = jax.jvp(lambda x, y: x - y, (b, a), (bdot, adot))  # adot used 2nd time
    (U, Udot) = jax.jvp(lambda x: 4.0 * x, (c,), (cdot,))
    return dict(r=r, a=a, b=b, c=c, U=U), dict(r=rdot, a=adot, b=bdot, c=cdot, U=Udot)


def reverse_mode(r):
    a = r ** -6; b = a ** 2; c = b - a; U = 4.0 * c   # forward sweep: keep r and a
    Ubar = 1.0                                        # seed output weight
    (cbar,) = jax.vjp(lambda x: 4.0 * x, c)[1](Ubar)
    (bbar, a_from_c) = jax.vjp(lambda x, y: x - y, b, a)[1](cbar)
    (a_from_b,) = jax.vjp(lambda x: x ** 2, a)[1](bbar)
    abar = a_from_c + a_from_b                        # a feeds two nodes: add both terms
    (rbar,) = jax.vjp(lambda x: x ** -6, r)[1](abar)
    return dict(r=r, a=a, b=b, c=c, U=U), dict(U=Ubar, c=cbar, b=bbar, a=abar, r=rbar)
```

- [ ] **Step 4: 跑测试通过**

Run: `cd week5 && .venv/bin/python -m pytest scripts/test_ad.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit** — `git add week5/scripts/ad_lib.py week5/scripts/test_ad.py && git commit -m "week5: hand forward/reverse AD of LJ pair energy"`

## Task 3: VERIFY(1) — 三种导数对比 + `derivatives.json`

**Files:**
- Create: `week5/scripts/run_ad.py`

**Interfaces:**
- Consumes: Task 2 的 `ad_lib`。
- Produces: `artifacts/ad/derivatives.json`（键 r, energy, tangents{r,a,b,c,U}, adjoints{U,c,b,a,r}, jax_grad）、`artifacts/ad/modes.png`；stdout 打印三种方法最大误差。

- [ ] **Step 1: 写 `scripts/run_ad.py`**

```python
"""Part 1 VERIFY(1): compare hand forward/reverse AD with finite differences."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ad_lib import lj, analytic_dU, forward_mode, reverse_mode, finite_diff

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"
ART.mkdir(parents=True, exist_ok=True)

rs = np.linspace(0.95, 2.5, 601)          # 601 equally spaced, endpoints included
exact = np.array([float(analytic_dU(r)) for r in rs])
fwd = np.array([float(forward_mode(r)[1]["U"]) for r in rs])
rev = np.array([float(reverse_mode(r)[1]["r"]) for r in rs])
fd = np.array([float(finite_diff(r)) for r in rs])

errs = {name: np.abs(v - exact) for name, v in
        [("forward", fwd), ("reverse", rev), ("finite difference", fd)]}
for name, e in errs.items():
    print(f"max |error| {name:>17s}: {e.max():.3e}")
assert errs["forward"].max() < 1e-12 and errs["reverse"].max() < 1e-12
assert errs["finite difference"].max() > max(errs["forward"].max(), errs["reverse"].max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(rs, exact, "k-", lw=3, alpha=0.4, label="analytic")
axes[0].plot(rs, fwd, "b--", label="forward mode")
axes[0].plot(rs, rev, "g:", label="reverse mode")
axes[0].plot(rs, fd, "r-", lw=1, label="finite difference")
axes[0].axvline(2 ** (1 / 6), color="gray", ls=":", lw=0.8)
axes[0].set(xlabel="separation r", ylabel="dU/dr", title="dU/dr")
axes[0].legend()
for name, e in errs.items():
    axes[1].semilogy(rs, np.maximum(e, 1e-18), label=name)
axes[1].set(xlabel="separation r", ylabel="absolute error", title="errors")
axes[1].legend()
fig.tight_layout()
fig.savefig(ART / "modes.png", dpi=150)

import jax
import jax.numpy as jnp
from ad_lib import lj
r = jnp.array(1.3)
vals, t = forward_mode(r)
_, adj = reverse_mode(r)
doc = {
    "r": 1.3,
    "energy": float(vals["U"]),
    "tangents": {k: float(v) for k, v in t.items()},
    "adjoints": {k: float(v) for k, v in adj.items()},
    "jax_grad": float(jax.grad(lj)(r)),
}
(ART / "derivatives.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
```

- [ ] **Step 2: 运行并核对**

Run: `cd week5 && .venv/bin/python scripts/run_ad.py`
Expected: 两个 AD 最大误差 < 1e-12（参考 < 3e-14），FD 约 5e-9；`derivatives.json` 中 energy = −0.6570169144600471、tangents.U = adjoints.r = jax_grad = 2.239979929791143、adjoints.a = −2.3425903117359734。打开 `artifacts/ad/modes.png`：曲线过零点在 r = 2^{1/6} ≈ 1.1225。

- [ ] **Step 3: Commit** — `git add week5/scripts/run_ad.py week5/artifacts/ad/modes.png week5/artifacts/ad/derivatives.json && git commit -m "week5: AD modes vs finite differences, derivatives.json"`

## Task 4: VERIFY(2) — JAX 记录的计算图与梯度图

**Files:**
- Create: `week5/scripts/ad_graphs.py`
- Produces: `artifacts/ad/graph.png`、`artifacts/ad/grad-graph.png`

**Interfaces:**
- Consumes: `ad_lib.lj`；`jax.make_jaxpr`（对 `lj` 与 `jax.grad(lj)` 各在 r=1.3 求记录）。

- [ ] **Step 1: 写 `scripts/ad_graphs.py`（通用 jaxpr 绘图：每条 eqn 一个节点，标 primitive 名+参数，字面量画小标签，按出现顺序横向布局）**

```python
"""Part 1 VERIFY(2): draw the computational graphs jax records for U and grad U."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ad_lib import lj

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"

def eqn_label(eqn):
    params = ",".join(f"{k}={v}" for k, v in eqn.params.items())
    return eqn.primitive.name + (f"[{params}]" if params else "")

def draw(jaxpr, path, title):
    eqns = jaxpr.eqns
    n = len(eqns)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * n), 3.6))
    # 变量/字面量的 x 坐标 = 首次被使用的 eqn 序号；eqn x = 序号
    var_x, lit_x = {}, {}
    def xv(v):
        if isinstance(v, jax.Literal):
            lit_x.setdefault(str(v.val), len(var_x) + len(lit_x))
            return lit_x[str(v.val)] - 0.25, str(v.val)
        var_x.setdefault(v, len(var_x))
        return var_x[v] - 0.25, str(v)
    for i, e in enumerate(eqns):
        for v in e.invars:
            x0, name = xv(v)
            ax.annotate("", xy=(i - 0.35, 0), xytext=(x0 + 0.1, 0),
                        arrowprops=dict(arrowstyle="->", color="0.55"))
            if isinstance(v, jax.Literal):
                ax.text(x0, 0.28, name, fontsize=8, ha="center")
        ax.plot(i, 0, "s", ms=26, color="#8ecae6")
        ax.text(i, 0, eqn_label(e), fontsize=8, ha="center", va="center", rotation=0)
    for v in jaxpr.outvars:
        x0, name = xv(v)
        ax.annotate("", xy=(n + 0.0, 0), xytext=(x0 + 0.1, 0),
                    arrowprops=dict(arrowstyle="->", color="0.55"))
    ax.plot(n + 0.3, 0, "o", ms=18, color="#ffb703")
    ax.text(n + 0.3, 0, "output", fontsize=8, ha="center", va="center")
    ax.set(xlim=(-1, n + 1.2), ylim=(-1, 1), title=title)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(title, "->", [eqn_label(e) for e in eqns])

r = jnp.array(1.3)
draw(jax.make_jaxpr(lj)(r), ART / "graph.png", "computational graph of U")
draw(jax.make_jaxpr(jax.grad(lj))(r), ART / "grad-graph.png", "gradient graph of U")
names = [e.primitive.name for e in jax.make_jaxpr(jax.grad(lj))(r).eqns]
print("add node present:", [s for s in names if "add" in s])  # 期望 add_any
```

- [ ] **Step 2: 运行核对**

Run: `cd week5 && .venv/bin/python scripts/ad_graphs.py`
Expected: `graph.png` 顺序为 `integer_pow[y=-6] → integer_pow[y=2] → sub → mul(4.0)`；`grad-graph.png` 打印的列表含 `integer_pow[y=-7]`、`mul(-6.0)`、`mul(2.0)`、`add_any`、`neg`，前向侧复用 r 与 a 的局部导数。布局可与参考不同，节点/标签语义须一致。

- [ ] **Step 3: Commit** — `git add week5/scripts/ad_graphs.py week5/artifacts/ad/graph.png week5/artifacts/ad/grad-graph.png && git commit -m "week5: jaxpr graphs of U and grad U"`

## Task 5: VERIFY(3) — cluster 能量梯度两种模式的代价

**Files:**
- Create: `week5/scripts/ad_scaling.py`、`week5/scripts/cluster_lib.py`
- Produces: `artifacts/ad/scaling.png`；stdout 打印两模式每尺寸的比值与最大相对误差。

**Interfaces:**
- Produces（`cluster_lib.py`）：`cluster_positions(n, seed=20260930) -> np.ndarray (n,3)`（间距 2^{1/6} 的立方格点取前 n 个，加 N(0, 0.05) 抖动）；`cluster_energy(x) -> jnp.scalar`；`analytic_gradient(x) -> np.ndarray (n,3)`（= −力）。

- [ ] **Step 1: 写 `scripts/cluster_lib.py`**

```python
import math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SPACING = 2.0 ** (1.0 / 6.0)

def cluster_positions(n, seed=20260930):
    rng = np.random.default_rng(seed)
    side = math.ceil(round(n ** (1 / 3) + 1e-9))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:n]
    x = np.array(pts, dtype=float) * SPACING
    return x + rng.normal(0.0, 0.05, x.shape)

def cluster_energy(x):
    d = x[:, None, :] - x[None, :, :]
    r = jnp.sqrt((d * d).sum(-1))
    i, j = jnp.triu_indices(x.shape[0], 1)
    r = r[i, j]
    return (4.0 * (r ** -12 - r ** -6)).sum()

def analytic_gradient(x):
    diff = x[:, None, :] - x[None, :, :]
    r = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(r, np.inf)
    coef = 24.0 * (r ** -7 - 2.0 * r ** -13) / r
    return -(coef[:, :, None] * diff).sum(axis=1) * -1.0  # dE/dx_i = sum_j U'(r_ij) r̂_ij
```

（注意：`dE/dx_i = Σ_j U'(r_ij)(x_i−x_j)/r_ij`，力 = 其负值；上式直接给出梯度。）

- [ ] **Step 2: 写 `scripts/ad_scaling.py`**

```python
"""Part 1 VERIFY(3): gradient cost of a cluster energy, forward vs reverse mode."""
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from cluster_lib import cluster_positions, cluster_energy, analytic_gradient

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"

def timed(fn, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best

def forward_gradient(x):
    g = np.empty(x.size)
    for k in range(x.size):
        v = np.zeros(x.size); v[k] = 1.0
        (_, d) = jax.jvp(cluster_energy, (x,), (v.reshape(x.shape),))
        g[k] = float(d)
    return g.reshape(x.shape)

sizes = [64, 128, 256, 512, 1024]
rows = []
for n in sizes:
    x = jnp.asarray(cluster_positions(n))
    e_grad = analytic_gradient(np.asarray(x))          # independent check
    t_e = timed(lambda: cluster_energy(x), 10)
    t_fwd = timed(lambda: forward_gradient(x), 1)
    t_rev = timed(lambda: jax.grad(cluster_energy)(x), 5)
    gf = forward_gradient(np.asarray(x))
    gr = np.asarray(jax.grad(cluster_energy)(x))
    fmax = np.abs(e_grad).max()
    err_f = np.abs(gf - e_grad).max() / fmax
    err_r = np.abs(gr - e_grad).max() / fmax
    print(f"N={n:5d} P={3*n:5d} fwd/energy={t_fwd/t_e:8.1f} rev/energy={t_rev/t_e:5.2f} "
          f"err_fwd={err_f:.2e} err_rev={err_r:.2e}")
    assert err_f < 1e-12 and err_r < 1e-12
    rows.append((3 * n, t_fwd / t_e, t_rev / t_e))

P = np.array([r[0] for r in rows]); fr = np.array([r[1] for r in rows]); rr = np.array([r[2] for r in rows])
assert fr[-1] > 100 * rr[-1]                              # at P = 3072
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(P, fr, "o-", label="forward: one JVP per input")
ax.loglog(P, rr, "s-", label="reverse: one VJP")
ax.loglog(P, P / P[0] * rr[0], ":", color="0.5", label=r"$\propto P$")
ax.set(xlabel="number of inputs P = 3N", ylabel="gradient time / energy time",
       title="cluster-energy gradient cost")
ax.legend(); fig.tight_layout(); fig.savefig(ART / "scaling.png", dpi=150)
```

- [ ] **Step 3: 运行核对**

Run: `cd week5 && .venv/bin/python scripts/ad_scaling.py`
Expected: 两模式误差 < 1e-12（参考 2e-15）；P=3072 时 forward 比值远超 reverse 的 100 倍（参考 forward≈500、reverse 1.2–1.5）；`scaling.png` 上 forward 沿 ∝P 虚线上升、reverse 基本水平。N=1024 的 forward 需数秒到几十秒，耐心等。

- [ ] **Step 4: Commit** — `git add week5/scripts/cluster_lib.py week5/scripts/ad_scaling.py week5/artifacts/ad/scaling.png && git commit -m "week5: cluster gradient scaling forward vs reverse"`

---

# Part 2：地震波正演（Rust `seismic` crate）

## Task 6: crate 脚手架 + `seismic.design.toml`

**Files:**
- Create: `week5/seismic/Cargo.toml`、`week5/seismic/src/main.rs`、`week5/seismic/src/lib.rs`、`week5/seismic.design.toml`

**Interfaces:**
- Produces: crate 名 `seismic`，二进制 `seismic`；`lib.rs` 只做 `pub mod`。
- `main.rs` 暂时只解析 `--experiment/--mode/--out/--every` 并打印计划（Task 11 补全）。

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "seismic"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 2: 从学习单第 7–8 页逐字抄写 green panel 到 `week5/seismic.design.toml`**（`usage`/`description`/`[args]`/`[writes]`/`[args.recording]`/`[writes.recording]` 六块，含多行字符串原文；后续 Part 3/4 再追加面板）。
- [ ] **Step 3: `src/main.rs` 骨架 + `src/lib.rs`（空模块表）**，`cargo build` 通过（在 `week5/seismic/`）。
- [ ] **Step 4: Commit** — `git add week5/seismic week5/seismic.design.toml && git commit -m "week5: seismic crate scaffold and design panel"`

## Task 7: npy 写入模块（TDD）

**Files:**
- Create: `week5/seismic/src/npy.rs`（在 lib.rs 挂 `pub mod npy;`）、`week5/seismic/tests/npy.rs`

**Interfaces:**
- Produces: `pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> anyhow::Result<()>`、`write_f32` 同型；npy 1.0 格式、`'descr': '<f8'/'<f4'`、`'fortran_order': False`、header 空格填充至 64 倍数且以 `\n` 结尾。

- [ ] **Step 1: 失败测试 `tests/npy.rs`**

```rust
use seismic::npy::write_f64;

#[test]
fn npy_header_and_payload() {
    let dir = std::env::temp_dir().join("seismic-npy-test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("a.npy");
    write_f64(&p, &[2, 3], &(0..6).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
    let b = std::fs::read(&p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    assert_eq!(&b[6..8], &[1, 0]);
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = String::from_utf8_lossy(&b[10..10 + hlen]);
    assert!(header.contains("'<f8'"));
    assert!(header.contains("'fortran_order': False"));
    assert!(header.contains("'shape': (2, 3)"));
    assert_eq!(header.trim_end().ends_with('}'), true);
    assert_eq!(b.len(), 10 + hlen + 6 * 8);
}
```

- [ ] **Step 2: 跑 `cargo test --test npy` 确认失败**（函数未定义）。
- [ ] **Step 3: 实现 `src/npy.rs`**：写 magic `b"\x93NUMPY"`、版本 `[1,0]`、u16 LE header 长度；header 字符串 `{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }` 用空格补到 (10+len) 为 64 的倍数并以 `\n` 结尾；随后小端写数据。`write_f32` 复用同一私有函数（descr 参数化）。
- [ ] **Step 4: `cargo test --test npy` 通过**（Run: `cd week5/seismic && cargo test --test npy`，Expected: 1 passed）
- [ ] **Step 5: Commit** — `week5: npy writer`

## Task 8: 实验文件解析（TDD）

**Files:**
- Create: `week5/seismic/src/experiment.rs`、`week5/seismic/tests/experiment.rs`

**Interfaces:**
- Produces:

```rust
#[derive(Deserialize, Clone)]
pub struct Experiment {
    pub nx: usize, pub nz: usize,
    pub dx: f64, pub dt: f64, pub steps: usize,
    pub source_frequency: f64, pub source_peak_time: f64, pub source_amplitude: f64,
    pub shots: Vec<[usize; 2]>,        // [x, z]
    pub receivers: Vec<[usize; 2]>,    // [x, z]
    pub sponge_width: usize, pub sponge_strength: f64,
    pub background: Vec<Vec<f64>>,     // [z][x]
    pub perturbation: Option<Vec<Vec<f64>>>,
    pub length_unit_m: f64, pub time_unit_s: f64,
}
pub fn load(path: &Path) -> anyhow::Result<Experiment>          // serde_json::from_reader
pub fn flatten(vv: &[Vec<f64>], nx: usize, nz: usize) -> Vec<f64> // [z][x] -> 一维
```

- [ ] **Step 1: 失败测试**：`tests/experiment.rs` 内用 `serde_json::json!` 造最小对象写入临时文件，`load` 后断言各字段、`flatten` 长度 = nx·nz、`flatten(&bg)[z*nx+x] == bg[z][x]`；再断言 `background` 每行长度 = nx（load 时校验，行宽不符报错）。
- [ ] **Step 2: 确认失败**（`cargo test --test experiment`）
- [ ] **Step 3: 实现**（含行宽/非空校验，` anyhow::bail!` 报清晰错误）。
- [ ] **Step 4: 测试通过**
- [ ] **Step 5: Commit** — `week5: experiment json loading`

## Task 9: Ricker 脉冲、源足迹、海绵系数（TDD）

**Files:**
- Create: `week5/seismic/src/survey.rs`、`week5/seismic/tests/survey.rs`

**Interfaces:**
- Produces:

```rust
pub fn ricker(t: f64, f0: f64, t0: f64) -> f64          // (1-2θ²)e^{-θ²}, θ = π f0 (t-t0)
pub fn footprint(nx: usize, nz: usize, shot: [usize; 2]) -> Vec<f64>  // 单位峰值, σ=1 格
pub fn sponge_sigma(nx: usize, nz: usize, width: usize, strength: f64) -> Vec<f64>
pub struct Coeffs { pub cfl: f64, pub dt2: f64, pub ad: Vec<f64>, pub bd: Vec<f64> } // ad=1-σdt, bd=1+σdt
pub fn coeffs(exp: &Experiment) -> Coeffs
```

- [ ] **Step 1: 失败测试 `tests/survey.rs`**

```rust
use seismic::survey::*;

#[test]
fn ricker_peak_and_troughs() {
    let (f0, t0) = (0.08f64, 15.0f64);
    assert!((ricker(t0, f0, t0) - 1.0).abs() < 1e-12);
    let dt_t = (1.5f64).sqrt() / (std::f64::consts::PI * f0);   // θ = sqrt(3/2)
    let trough = -2.0 * (-1.5f64).exp();
    assert!((ricker(t0 + dt_t, f0, t0) - trough).abs() < 1e-12); // ≈ -0.44626032
    assert!((ricker(t0 - dt_t, f0, t0) - trough).abs() < 1e-12);
    assert!(ricker(27.0, f0, t0).abs() < 0.01);                  // 2.7 s 后可忽略
}

#[test]
fn footprint_gaussian() {
    let f = footprint(11, 11, [5, 5]);
    assert!((f[5 * 11 + 5] - 1.0).abs() < 1e-12);
    assert!((f[5 * 11 + 6] - (-0.5f64).exp()).abs() < 1e-12);
    assert!(f[0].abs() < 1e-8);
}

#[test]
fn sponge_profile() {
    let s = sponge_sigma(41, 41, 6, 0.4);
    assert_eq!(s.len(), 41 * 41);
    assert!((s[0] - 0.4).abs() < 1e-12);               // 角点 ℓ=0
    assert!(s[6 * 41 + 20]].abs() < 1e-12);            // ℓ=6 → 0
    assert!((s[3 * 41 + 3] - 0.4 * 0.25).abs() < 1e-12); // ℓ=3 → 0.4*(0.5)^2
    assert!(s.iter().all(|&v| (0.0..=0.4).contains(&v)));
}
```

（注意修正笔误：`s[6 * 41 + 20]`。）
- [ ] **Step 2: 确认失败 → Step 3: 实现**（ℓ = min(i, j, nx−1−i, nz−1−j)；σ = strength·max(0, 1−ℓ/W)²）→ **Step 4: 通过**
- [ ] **Step 5: Commit** — `week5: ricker pulse, source footprint, sponge coefficients`

## Task 10: 阻尼声波步进（TDD）

**Files:**
- Create: `week5/seismic/src/wave.rs`、`week5/seismic/tests/wave.rs`

**Interfaces:**
- Produces（后续 born/adjoint/treeverse 都用同一表达式；kernel 内核逐位复制它）:

```rust
pub struct State { pub um: Vec<f64>, pub u: Vec<f64> }        // (u^{n-1}, u^n)
impl State { pub fn zero(nx: usize, nz: usize) -> State; pub fn rotate(&mut self, un: Vec<f64>); }
pub fn step(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
            cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
            un: &mut [f64])
pub fn sample(u: &[f64], nx: usize, receivers: &[[usize; 2]]) -> Vec<f64>  // y^n_k = u^{n+1}[z][x]
pub fn l2(x: &[f64]) -> f64
```

`step` 的内点表达式（先全零填 `un`，再只更新内点 1..nx-2 × 1..nz-2）：

```rust
let k = j * nx + i;
let lap = u[k + 1] + u[k - 1] + u[k + nx] + u[k - nx] - 4.0 * u[k];
un[k] = (2.0 * u[k] - ad[k] * um[k] + cfl * c[k] * c[k] * lap + dt2 * qamp * foot[k]) / bd[k];
```

- [ ] **Step 1: 失败测试 `tests/wave.rs`（手算数值）**

```rust
use seismic::wave::step;
// 7x7, dx=0.5, dt=0.1 → cfl=0.04, dt2=0.01; c=1.5; σ=0 → ad=bd=1
// foot 只在中心 (3,3) 为 1; qamp=2.0
#[test]
fn two_steps_no_damping() {
    let (nx, nz) = (7usize, 7usize);
    let z = vec![0.0; nx * nz];
    let mut foot = vec![0.0; nx * nz]; foot[3 * nx + 3] = 1.0;
    let (c, ad, bd) = (vec![1.5; 49], vec![1.0; 49], vec![1.0; 49]);
    let mut u0 = vec![0.0; 49];
    step(&z, &z, &foot, &c, 2.0, 0.04, 0.01, &ad, &bd, nx, nz, &mut u0);
    assert!((u0[3 * nx + 3] - 0.02).abs() < 1e-12);           // dt2*qamp
    let mut u1 = vec![0.0; 49];
    step(&z, &u0, &foot, &c, 2.0, 0.04, 0.01, &ad, &bd, nx, nz, &mut u1);
    assert!((u1[3 * nx + 3] - (0.04 - 0.09 * 0.08 + 0.02)).abs() < 1e-12); // 0.0528
    assert!((u1[3 * nx + 4] - 0.09 * 0.02).abs() < 1e-12);    // 0.0018
    assert!(u1[0].abs() == 0.0);                              // 边界保持零
}

#[test]
fn one_step_with_damping() {
    // σ=0.5, dt=0.1 → ad=0.95, bd=1.05
    let (ad, bd) = (vec![0.95; 49], vec![1.05; 49]);
    let mut foot = vec![0.0; 49]; foot[3 * 7 + 3] = 1.0;
    let mut un = vec![0.0; 49];
    step(&vec![0.0; 49], &vec![0.0; 49], &foot, &vec![1.5; 49], 2.0, 0.04, 0.01,
         &ad, &bd, 7, 7, &mut un);
    assert!((un[3 * 7 + 3] - 0.02 / 1.05).abs() < 1e-12);
}
```

另加 `sample`（返回接收点值、位置重复时各点独立采样）与 `l2`（sqrt(Σx²)）的小测试。
- [ ] **Step 2: 失败 → Step 3: 实现 → Step 4: 通过**（`cargo test`）
- [ ] **Step 5: Commit** — `week5: damped acoustic step, sampling, L2`

## Task 11: forward 模式 CLI 端到端（TDD）

**Files:**
- Create: `week5/seismic/src/output.rs`、`week5/seismic/src/forward.rs`；Modify: `src/main.rs`、`src/lib.rs`
- Test: `week5/seismic/tests/cli.rs`

**Interfaces:**
- Consumes: Tasks 7–10。
- Produces: CLI `seismic --experiment P --mode forward [--every E] --out DIR`；行为契约＝`seismic.design.toml`：
  - stdout：表头 `shot\tmode\tL2`，其后每炮一行 `i\tforward\t{l2:.6}`（该炮 traces 的 L2）。
  - `<out>/run.json`：`experiment_file` + 除 background/perturbation 外的全部实验字段；有 `--every` 时加 `recording: {every, steps:[...], times:[...]}`（帧序）。
  - `<out>/result.json`：`mode, nx, nz, dx, dt, steps, shots, receivers`。
  - `<out>/traces.npy`：f64 `[shot, step, receiver]`。
  - `--every E` 时（仅 shot 0）：`wavefield.npy` f32 `[frame, z, x]`（帧 n = 0,E,…,≤steps 的 `u^n`）与 `echo.npy` f32（shot 0 在 background+perturbation 与 background 两跑之差、同帧）。
  - born+`--every`、adjoint 无 `--data`、treeverse 无 `--checkpoints` 等误用 → 报错退出非零（本任务先留报错桩）。

- [ ] **Step 1: 失败测试 `tests/cli.rs`**

```rust
use std::process::Command;

fn fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let bg = vec![vec![1.5; 11]; 11];
    let doc = serde_json::json!({
        "nx": 11, "nz": 11, "dx": 1.0, "dt": 0.2, "steps": 5,
        "source_frequency": 0.5, "source_peak_time": 1.0, "source_amplitude": 1.0,
        "shots": [[5, 5]], "receivers": [[3, 5], [7, 5]],
        "sponge_width": 2, "sponge_strength": 0.4,
        "background": bg, "perturbation": vec![vec![0.0; 11]; 11],
        "length_unit_m": 100.0, "time_unit_s": 0.1,
    });
    let p = dir.join("exp.json");
    std::fs::write(&p, doc.to_string()).unwrap();
    p
}

#[test]
fn forward_cli_end_to_end() {
    let dir = std::env::temp_dir().join("seismic-cli-test");
    std::fs::create_dir_all(&dir).unwrap();
    let exp = fixture(&dir);
    let out = dir.join("out");
    let st = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .args(["--experiment", exp.to_str().unwrap(), "--mode", "forward",
               "--every", "2", "--out", out.to_str().unwrap()])
        .output().unwrap();
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let stdout = String::from_utf8_lossy(&st.stdout);
    assert!(stdout.starts_with("shot\tmode\tL2"));
    assert_eq!(stdout.lines().count(), 2);
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["recording"]["steps"], serde_json::json!([0, 2, 4]));
    assert_eq!(run["recording"]["times"], serde_json::json!([0.0, 0.4, 0.8]));
    let res: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(res["mode"], "forward");
    assert_eq!(res["steps"], 5);
    assert!(out.join("traces.npy").exists());
    let b = std::fs::read(out.join("traces.npy")).unwrap();
    let off = 10 + { let h = &b[8..10]; u16::from_le_bytes([h[0], h[1]]) as usize };
    assert_eq!(b.len() - off, 1 * 5 * 2 * 8);        // [shot, step, receiver] f64
    assert!(out.join("wavefield.npy").exists());
    assert!(out.join("echo.npy").exists());
}
```

- [ ] **Step 2: 确认失败**（`cargo test --test cli`）
- [ ] **Step 3: 实现**：
  - `output.rs`：`write_run_json(path, exp, exp_path, recording: Option<(usize, Vec<usize>)>)`、`write_result_json(path, exp, mode)`（adjoint 系列后续任务再扩 statistics）。
  - `forward.rs`：

```rust
pub fn run(exp: &Experiment, exp_path: &Path, out: &Path, every: Option<usize>)
    -> anyhow::Result<()> {
    let co = survey::coeffs(exp);
    let c = experiment::flatten(&exp.background, exp.nx, exp.nz);
    let cp = exp.perturbation.as_ref()
        .map(|p| experiment::flatten(p, exp.nx, exp.nz));
    let mut traces = vec![0.0; exp.shots.len() * exp.steps * exp.receivers.len()];
    let mut frames: Vec<Vec<f32>> = Vec::new();
    let mut echo: Vec<Vec<f32>> = Vec::new();
    println!("shot\tmode\tL2");
    for (si, &shot) in exp.shots.iter().enumerate() {
        let foot = survey::footprint(exp.nx, exp.nz, shot);
        let mut s = State::zero(exp.nx, exp.nz);
        let record = si == 0 && every.is_some();
        // echo 需要：第二个模拟（background+perturbation），同帧差分
        let mut s_echo = State::zero(exp.nx, exp.nz);
        for n in 0..exp.steps {
            let t = n as f64 * exp.dt;
            let qamp = exp.source_amplitude * survey::ricker(t, exp.source_frequency, exp.source_peak_time);
            let mut un = vec![0.0; c.len()];
            wave::step(&s.um, &s.u, &foot, &c, qamp, co.cfl, co.dt2, &co.ad, &co.bd,
                       exp.nx, exp.nz, &mut un);
            s.rotate(un);
            if record && n % every.unwrap() == 0 { frames.push(s.u.iter().map(|&v| v as f32).collect()); }
            if record {  // echo：同一步在 c+p 下再走一次
                let mut un2 = vec![0.0; c.len()];
                let c2 = cp.as_ref().unwrap();
                let csum: Vec<f64> = c.iter().zip(c2).map(|(&a, &b)| a + b).collect();
                wave::step(&s_echo.um, &s_echo.u, &foot, &csum, qamp, co.cfl, co.dt2,
                           &co.ad, &co.bd, exp.nx, exp.nz, &mut un2);
                s_echo.rotate(un2);
                if n % every.unwrap() == 0 {
                    echo.push(s_echo.u.iter().zip(&s.u)
                              .map(|(&a, &b)| (a - b) as f32).collect());
                }
            }
            for (k, &y) in wave::sample(&s.u, exp.nx, &exp.receivers).iter().enumerate() {
                traces[(si * exp.steps + n) * exp.receivers.len() + k] = y;
            }
        }
        // 最后一帧 n = steps（u^N）也算 forward 帧
        if let Some(e) = every { if record && exp.steps % e == 0 {
            frames.push(s.u.iter().map(|&v| v as f32).collect());
            echo.push(s_echo.u.iter().zip(&s.u).map(|(&a, &b)| (a - b) as f32).collect());
        }}
        println!("{}\tforward\t{:.6}", si, wave::l2_of_shot(...));  // 该炮 traces 的 L2
    }
    // 写文件：traces.npy、wavefield.npy/echo.npy（record 时）、run.json、result.json
    ...
}
```

（实现时把"该炮 traces 的 L2"抽成小函数；帧规则：n=0 在加 qamp 步进**前**记录 u⁰=0 —— 即先记帧再步进，循环结束后若 steps%e==0 补记 u^N。上面代码按"每步步进后若 n%e==0 记 u^{n+1}"会错位一帧：正确顺序是在步 n **前**记 `u^n`，请按此实现并保持测试 `[0,2,4]` 通过：steps=5,e=2 → 帧记在 n=0,2,4 的步进前，即 u⁰,u²,u⁴。）
  - `main.rs`：clap 结构体 + 校验 + 分发。
- [ ] **Step 4: `cargo test` 全绿**（含 Task 7–10 旧测试）
- [ ] **Step 5: Commit** — `week5: forward mode CLI end-to-end`

## Task 12: 下载 inputs + `inputs.png`

**Files:**
- Create: `week5/inputs/`（gitignore）、`week5/scripts/plot_inputs.py`、`week5/MARMOUSI-LICENSE`
- Produces: `artifacts/inputs.png`

- [ ] **Step 1: 下载解压（在 week5/）**

```bash
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip -o week5-inputs.zip && rm week5-inputs.zip
ls inputs/
```
Expected: `reflector.json`、`marmousi.json`、Marmousi license 文件（名字以解压结果为准）。把 license 复制为 `week5/MARMOUSI-LICENSE`（入 git）。

- [ ] **Step 2: 写 `scripts/plot_inputs.py`**（两 panel）：
  - panel 1 geometry：imshow background（extent 用 km：`index·dx·length_unit_m/1000`），色标 km/s；叠加 perturbation 等值线或第二色轴（"Thin reflector" 标注）；shots 三角、receivers 圆点（0.8 km 深度）；海绵内边界虚线框；x 轴水平位置 km、y 轴深度 km 向下。
  - panel 2 pulse：`g(n·dt)`，n=0..239，x 轴秒（×time_unit_s），标注 f0=0.8 Hz、t0=1.5 s。
  运行后核对：三炮十四检波器在 0.8 km、reflector 2.1 km、脉冲峰 1 @1.5 s、谷 −0.446 @1.02/1.98 s、2.6 s 后可忽略。
- [ ] **Step 3: 运行** — `cd week5 && .venv/bin/python scripts/plot_inputs.py`；打开 `artifacts/inputs.png` 对照学习单第 10 页参考图。
- [ ] **Step 4: Commit** — `git add week5/scripts/plot_inputs.py week5/artifacts/inputs.png week5/MARMOUSI-LICENSE && git commit -m "week5: plot reflector experiment geometry and source pulse"`

## Task 13: reflector 正演跑通 + gathers + 帧

**Files:**
- Create: `week5/scripts/plot_forward.py`、`week5/scripts/plot_frames.py`
- Produces: `artifacts/forward/*`（CLI 产物）+ `artifacts/forward/gathers.png`、`wavefield.png`、`echo.png`

- [ ] **Step 1: 安装并按学习单原命令运行**

```bash
cd week5
cargo install --path seismic
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
```
Expected: 3 行 stdout；`traces.npy [3,240,14]`、`wavefield.npy [81,41,41]`、`echo.npy` 同形（帧 n=0,3,…,240）。

- [ ] **Step 2: `scripts/plot_forward.py`**：读 `traces.npy`，三 panel（每炮一 gather）：x=检波器位置 km、y=时间 s 向下、color=压力，共用一个色标（±max|p|）；存 `artifacts/forward/gathers.png`；打印：全部 traces 的 L2、每炮最大 |p| 及其 trace 序号与检波器索引。脚本内置断言：

```python
L2 = float(np.sqrt((traces ** 2).sum()))
assert abs(L2 / 11.574770 - 1.0) < 1e-4
for si in range(3):
    i = np.unravel_index(np.abs(traces[si]).argmax(), traces[si].shape)
    print(si, i, traces[si][i])
# 期望每炮 argmax 在 trace 83（step 84），值 0.60809514（炮 0、2）/ 0.59271397（炮 1）
```
- [ ] **Step 3: `scripts/plot_frames.py`**：读 `wavefield.npy`/`echo.npy` + `run.json`（recording.steps），渲染 step 150（t=3.00 s）两图：`wavefield.png`（色标 ±0.26 量级、标 Receivers/Active source）与 `echo.png`（±0.006、标 Reflector）；打印该帧 echo 最大值与直达波最大值及比值（≈2%）。用法 `plot_frames.py <artifacts/forward>`（Part 3 复用于 adjoint：`plot_frames.py <dir> --step 132 --names adjoint`）。
- [ ] **Step 4: 运行两个脚本，对照学习单第 10–11 页参考图**（强早事件近炮、远道负瓣弯曲；echo 从 reflector 向上扩散）。
- [ ] **Step 5: Commit** — `git add week5/scripts/plot_forward.py week5/scripts/plot_frames.py week5/artifacts/forward/gathers.png week5/artifacts/forward/wavefield.png week5/artifacts/forward/echo.png && git commit -m "week5: reflector forward gathers, wavefield and echo frames"`

（可选人工步骤：把 `wavefield.npy`+`run.json` 拖进课程查看器播放，用 Save PNG 存同帧 —— 见开工前问题 3。）

---

# Part 3：用 Enzyme 微分地震模拟

## Task 14: Enzyme 工具链 + 内核隔离 + 冒烟测试

**Files:**
- Create: `week5/seismic/rust-toolchain.toml`、`week5/seismic/build.rs`、`week5/seismic/src/kernel.rs`、`week5/seismic/src/enzyme.rs`
- Modify: `week5/seismic/Cargo.toml`
- Test: `week5/seismic/tests/enzyme.rs`
- Skill: `.pi/skills/enzyme-setup/SKILL.md`

**Interfaces:**
- Produces: `pub fn enzyme_cube(x: f64, out: &mut [f64; 3])` 安全包装（[8,12,12] @ x=2）；内核编译管线（后续 Task 15 的 step JVP/VJP 走同一管线）。

- [ ] **Step 1: 安装 skill 并钉工具链**

```bash
curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/enzyme-setup/SKILL.md \
  -o .pi/skills/enzyme-setup/SKILL.md
rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme
```
`week5/seismic/rust-toolchain.toml`:

```toml
[toolchain]
channel = "nightly-2026-09-05"
components = ["enzyme"]
profile = "minimal"
```
（此文件加入后，crate 内一切 cargo 命令都走该 nightly。）

- [ ] **Step 2: `Cargo.toml` 追加**（edition 2024 已有）：

```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3

[[bin]]
name = "seismic"
path = "src/main.rs"
test = false
```

- [ ] **Step 3: `src/kernel.rs`（只作 build.rs 输入，绝不 `mod kernel;`）——先放冒烟内核**

```rust
#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

unsafe extern "C" {
    fn abort() -> !;
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 { x * x * x }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adj) = cube_reverse(x, 1.0);
    unsafe { *out = y; *out.add(1) = dy; *out.add(2) = adj; }
}
```

- [ ] **Step 4: `build.rs`（照 skill 原文）**

```rust
use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new(env::var_os("RUSTC").unwrap())
        .args(["--edition=2024", "--crate-name", "enzyme_kernel", "--crate-type", "staticlib",
               "-C", "opt-level=3", "-C", "lto=fat", "-C", "panic=abort",
               "-Zautodiff=Enable", "src/kernel.rs", "-o"])
        .arg(out.join("libenzyme_kernel.a"))
        .status().expect("compile the Enzyme kernel with the pinned rustc");
    assert!(status.success(), "Enzyme kernel compilation failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=enzyme_kernel");
}
```

- [ ] **Step 5: `src/enzyme.rs` 安全包装 + 冒烟测试**

```rust
unsafe extern "C" { fn enzyme_cube(x: f64, out: *mut f64); }

pub fn cube_smoke(x: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    unsafe { enzyme_cube(x, out.as_mut_ptr()) }
    out
}
```

`tests/enzyme.rs`:

```rust
#[test]
fn enzyme_cube_smoke() {
    let [y, dy, adj] = seismic::enzyme::cube_smoke(2.0);
    assert_eq!([y, dy, adj], [8.0, 12.0, 12.0]);
}
```

- [ ] **Step 6: 阴性对照**（证明导数确实来自 Enzyme）：手动重跑 build.rs 的 rustc 命令但去掉 `-Zautodiff=Enable`，必须报 `using the autodiff feature requires -Z autodiff=Enable`；确认后照常 `cargo test --test enzyme`（Expected: 1 passed；首次会重编整个依赖树，属正常）。
- [ ] **Step 7: Commit** — `git add .pi/skills/enzyme-setup week5/seismic && git commit -m "week5: enzyme toolchain, isolated kernel, cube smoke test"`

## Task 15: 步进的 Enzyme primal/JVP/VJP + 恒等式测试（TDD）

**Files:**
- Modify: `week5/seismic/src/kernel.rs`（加 step 及其导数包装）、`week5/seismic/src/enzyme.rs`（安全包装）
- Test: `week5/seismic/tests/enzyme.rs`（追加）

**Interfaces:**
- Consumes: Task 10 的 `wave::step`（普通 Rust 参照）。
- Produces（born/adjoint/treeverse 依赖；参数与 `wave::step` 完全同名同义）:

```rust
// enzyme.rs —— 所有切片长度在调用前断言 == nx*nz
pub fn step_primal(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
                   cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                   un: &mut [f64]);
pub fn step_jvp(um: &[f64], d_um: &[f64], u: &[f64], d_u: &[f64], foot: &[f64],
                c: &[f64], d_c: &[f64], qamp: f64, cfl: f64, dt2: f64,
                ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                un: &mut [f64], d_un: &mut [f64]);
pub fn step_vjp(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
                cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                un_bar: &[f64], d_um: &mut [f64], d_u: &mut [f64], d_c: &mut [f64]);
```

- [ ] **Step 1: 失败测试（追加到 `tests/enzyme.rs`）**

```rust
use seismic::{enzyme, wave};

fn inputs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // 确定性伪随机：um,u ∈ [-1,1], foot ≥ 0, c ∈ [1,2], σ ∈ [0,0.4] → ad=1-σdt, bd=1+σdt
    let mut x = 12345u64;
    let mut rnd = move || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                            ((x >> 33) as f64) / (u32::MAX as f64) };
    let f = |lo: f64, hi: f64| lo + (hi - lo) * rnd();
    (vec![-1.0,1.0], vec![-1.0,1.0], vec![0.0,3.0], vec![1.0,2.0], vec![0.0,0.4], vec![0.0,0.0])
        .0; // 展开为六个 vec![f(n); n]，见实现说明
}

#[test]
fn kernel_primal_equals_plain_step_bitwise() {
    let (nx, nz) = (9, 7);
    let (um, u, foot, c, ad, bd) = inputs(nx * nz);  // ad=1-σ*0.2, bd=1+σ*0.2
    let mut a = vec![0.0; nx * nz];
    enzyme::step_primal(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz, &mut a);
    let mut b = vec![0.0; nx * nz];
    wave::step(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz, &mut b);
    assert_eq!(a, b);                                  // 逐位一致
}

#[test]
fn jvp_matches_hand_tangent() {
    // 手写切线（独立校验，仅测试内）：
    // d_un = (2 d_u - ad·d_um + cfl·(c²·lap(d_u) + 2c·d_c·lap(u)) + dt2·dqamp·foot)/bd, dqamp=0
    ...
    assert!(max_rel_diff < 1e-12);
}

#[test]
fn vjp_satisfies_dot_identity() {
    // 随机 (d_um,d_u,d_c) 与随机 un_bar：
    // <jvp_tangent(un_bar 方向), un_bar> == <(d_um,d_u,d_c), vjp(un_bar)>
    ...
    assert!(abs_diff / lhs.abs() < 1e-12);
}
```

（测试里的伪随机与手写切线按注释补全成可编译代码；`inputs` 直接写成六个 `vec![...]` 映射。）

- [ ] **Step 2: 确认失败**（`cargo test --test enzyme`：函数不存在）。
- [ ] **Step 3: `kernel.rs` 追加 step 与三个包装**。内核 `step` 与 `wave::step` 表达式逐位相同（先整场清零再更新内点）：

```rust
fn step(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
        cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
        un: &mut [f64]) {
    for k in 0..nx * nz { un[k] = 0.0; }
    for j in 1..nz - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            let lap = u[k + 1] + u[k - 1] + u[k + nx] + u[k - nx] - 4.0 * u[k];
            un[k] = (2.0 * u[k] - ad[k] * um[k]
                     + cfl * c[k] * c[k] * lap + dt2 * qamp * foot[k]) / bd[k];
        }
    }
}

#[autodiff_forward(step_jvp, Dual, Dual, Const, Dual, Const, Const, Const, Const, Const, Const, Const, Dual)]
#[autodiff_reverse(step_vjp, Duplicated, Duplicated, Const, Duplicated, Const, Const, Const, Const, Const, Const, Const, DuplicatedOnly)]
fn step(...)  // 与上面同一签名

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_primal(um: *const f64, u: *const f64, foot: *const f64,
    c: *const f64, qamp: f64, cfl: f64, dt2: f64, ad: *const f64, bd: *const f64,
    nx: usize, nz: usize, un: *mut f64) { /* 切片化后调 step */ }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_jvp(um: *const f64, d_um: *const f64, u: *const f64,
    d_u: *const f64, foot: *const f64, c: *const f64, d_c: *const f64, qamp: f64, cfl: f64,
    dt2: f64, ad: *const f64, bd: *const f64, nx: usize, nz: nz_typo_fix, un: *mut f64,
    d_un: *mut f64) { /* 调 step_jvp */ }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_vjp(um: *const f64, u: *const f64, foot: *const f64,
    c: *const f64, qamp: f64, cfl: f64, dt2: f64, ad: *const f64, bd: *const f64,
    nx: usize, nz: usize, un_bar: *const f64,
    d_um: *mut f64, d_u: *mut f64, d_c: *mut f64) { /* 调 step_vjp */ }
```

**编译口径**：autodiff 生成的 `step_jvp`/`step_vjp` 的准确参数序（影子插在哪个位置、`Duplicated` 影子是 `&mut` 还是 `&`）以 pinned nightly 的报错信息为准微调 wrapper；**正确性以 Step 1 的三个测试为准**（逐位 primal、手写切线对照、点积恒等式），测试不过不许继续。
- [ ] **Step 4: `cargo test --test enzyme` 全绿。**
- [ ] **Step 5: Commit** — `week5: enzyme step primal/jvp/vjp with identity tests`

## Task 16: 反向面板 + npy 读取（TDD）

**Files:**
- Modify: `week5/seismic.design.toml`（追加 Part 3 green panel：`[args.reverse]`/`[writes.reverse]`，逐字抄学习单第 11–12 页）
- Create: `week5/seismic/src/npy.rs` 增加 `read_f64`；`week5/seismic/tests/npy.rs` 追加
**Interfaces:**
- Produces: `pub fn read_f64(path: &Path) -> anyhow::Result<(Vec<usize>, Vec<f64>)>`（校验 `'<' `f8、fortran_order False；shape 从 header 的元组解析）。

- [ ] **Step 1: 失败测试**：先 `write_f64` 再 `read_f64`，断言 shape 与数据一致；手改 header 的 descr 为 `<f4` 时报错。
- [ ] **Step 2: 失败 → Step 3: 实现（魔数/版本/长度校验 + 简单扫描解析）→ Step 4: 通过**
- [ ] **Step 5: Commit** — `week5: reverse design panel and npy reader`

## Task 17: born 模式（TDD）

**Files:**
- Create: `week5/seismic/src/born.rs`；Modify: `src/main.rs`（Mode 枚举加 born）
- Test: `week5/seismic/tests/modes.rs`

**Interfaces:**
- Consumes: `enzyme::step_jvp`（Task 15）、Tasks 7–11 的输出设施。
- Produces: `seismic ... --mode born --out D`：每炮维护切线状态 `dS=(d_um,d_u)`（初始零）与固定影子 `d_c = perturbation`；每步 `step_jvp`（primal 正常前进，切线前进），`born_data[si][n][k] = d_u` 在接收点采样；写 `<out>/born_data.npy` f64 `[shot, step, receiver]`、`run.json`、`result.json`；stdout 每炮 born 数据 L2。`--every` 与 born 组合 → 报错（照 design：recording 仅 forward/adjoint）。

- [ ] **Step 1: 失败测试 `tests/modes.rs`**（用 Task 11 的 fixture 思路，steps=6、非零 perturbation，如 perturbation[z][x] = 0.1·sin）：

```rust
#[test]
fn born_zero_perturbation_gives_zero_data_and_linear_in_m() {
    // (a) perturbation 全零 → born_data 全零（L2 = 0）
    // (b) perturbation α·m：α=0.3 与 α=1.0 两次运行，断言 born_data(α=0.3) == 0.3·born_data(α=1.0)
}
```
- [ ] **Step 2: 失败 → Step 3: 实现（每步链 JVP，正是 Born-Forward 伪代码：ṡ_{n+1} = A_n ṡ_n + B_n m）→ Step 4: 通过（`cargo test`）**
- [ ] **Step 5: Commit** — `week5: born mode via chained timestep JVPs`

## Task 18: adjoint 模式（full history，TDD）

**Files:**
- Create: `week5/seismic/src/adjoint.rs`；Modify: `src/main.rs`、`src/output.rs`
- Test: `week5/seismic/tests/modes.rs`（追加）

**Interfaces:**
- Consumes: `enzyme::step_vjp`、`npy::read_f64`、Task 11 的录制设施。
- Produces: `seismic ... --mode adjoint --data BORN_NPY [--every E] --out D`：
  - 读 born 数据 `w[shot][step][receiver]`（形状校验）。
  - 每炮：先正演并**保存全轨迹** `s_0..s_N`（普通 `wave::step`）；`λ = (0,0)`、`img = 0`；`for n in (0..steps).rev()`：`λ.u[z_k·nx+x_k] += w[si][n][k]`（Rᵀ）；`step_vjp(primal = s_n, un_bar = λ.u) → (d_um, d_u, d_c)`；`img += d_c`；`λ = (d_um, d_u)`；shot 0 且 `--every` 时在 reverse step n 完成后若 `n%E==0` 记录 `λ.u` 为 f32 帧（帧序 n 递减）。
  - 写 `<out>/image.npy` f64 `[z,x]`（对炮求和）、`run.json`/`result.json`、`wavefield.npy`（录制时）；stdout 每炮打印该炮 image 贡献的 L2。
  - `result.json` 加 `statistics`（serde 结构体）：

```rust
#[derive(serde::Serialize)]
pub struct Statistics {
    pub storage: String,                    // "full" | "treeverse"
    pub checkpoints: Option<usize>,         // full → null
    pub reverse_calls: usize,               // 步 VJP 次数（合计）
    pub scheduler_forward_calls: usize,     // full → 0
    pub peak_saved_states: usize,           // full → N+1（跨炮取 max）
    pub peak_saved_bytes: u64,              // peak * 2*nx*nz*8
    pub per_shot: Vec<PerShot>,
}
#[derive(serde::Serialize)]
pub struct PerShot {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actions_file: Option<String>,
}
```

- [ ] **Step 1: 失败测试（追加 `tests/modes.rs`）**：同一 fixture 上 (a) 先跑 born 得 `d`；(b) `--data d` 跑 adjoint，断言转置恒等式 `<d,d> == <m, image>` 相对差 < 1e-12（`m` 用 fixture 的 perturbation）；(c) 随机权重 `w`（测试内生成 npy）再跑一次，断言 `<Jm, w> == <m, Jᵀw>`（`Jm` 来自 born）；(d) statistics：reverse_calls = steps·shots、scheduler_forward_calls = 0、peak_saved_states = steps+1、bytes = (steps+1)·2·nx·nz·8。
- [ ] **Step 2: 失败 → Step 3: 实现 → Step 4: 通过**
- [ ] **Step 5: Commit** — `week5: adjoint mode full history with statistics`

## Task 19: reflector 上的 RTM 验证

**Files:**
- Create: `week5/scripts/rtm_check.py`
- Produces: `artifacts/born/*`、`artifacts/adjoint/*`（CLI 产物）+ `artifacts/adjoint/image.png`、`artifacts/adjoint/wavefield.png`

- [ ] **Step 1: 按学习单命令跑**

```bash
cargo install --path seismic
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
seismic --experiment inputs/reflector.json --mode adjoint --data \
  artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
```
- [ ] **Step 2: `scripts/rtm_check.py`**：
  - 读 `born_data.npy`、`inputs/reflector.json` 的 `perturbation`、`artifacts/adjoint/image.npy`；计算 left = Σd²、right = Σ(m·image)、相对差；**打印双侧与相对差**，断言 < 1e-9（参考 0.0348479 / 1e-15）。
  - 三 panel（窗口 x=7..33、z=10..33 含端点，坐标换 km）：① 已知 perturbation（色标 km/s）；② raw RTM image（独立色标）；③ 行 L2 范数（x 轴为范数、y 为深度 km），标注真实深度 2.1 km 与剖面峰，打印深度差；断言峰在 z=21 一格内。存 `artifacts/adjoint/image.png`。
  - 用 `plot_frames.py artifacts/adjoint --step 132`（帧 t=2.64 s）存 `artifacts/adjoint/wavefield.png`，对照学习单第 14 页（伴随场聚焦到 reflector）。
- [ ] **Step 3: 运行并核对打印值与图**（宽正带 + 两侧负旁瓣；浅旁瓣范数约为峰的 92%）。
- [ ] **Step 4: Commit** — `git add week5/scripts/rtm_check.py week5/artifacts/born week5/artifacts/adjoint && git commit -m "week5: reflector transpose identity and RTM image"`（`git status` 确认没有 .npy 混入）

---

# Part 4：Treeverse 检查点

## Task 20: checkpoint 面板

**Files:**
- Modify: `week5/seismic.design.toml`（逐字追加学习单第 15 页 `[args.checkpoint]`/`[writes.checkpoint]`）
- [ ] **Step 1: 抄面板 → Step 2: Commit** — `week5: checkpoint design panel`

## Task 21: Treeverse 调度器（TDD，纯函数）

**Files:**
- Create: `week5/seismic/src/treeverse.rs`；Modify: `src/lib.rs`
- Test: `week5/seismic/tests/treeverse.rs`

**Interfaces:**
- Produces（Task 22 执行器与审计依赖）:

```rust
#[derive(Clone, Copy, PartialEq, serde::Serialize)]
pub enum Kind { Store, Restore, Call, Grad, Fetch }

pub struct Action { pub action: Kind, pub n: usize, pub saved_states: usize }

pub fn binom(n: usize, k: usize) -> u128;
pub fn binomial_fit(n_steps: usize, extra: usize) -> usize;   // 最小 τ≥1 使 C(τ+δ,δ) ≥ N
pub fn schedule(n_steps: usize, extra: usize) -> Vec<Action>; // 含 saved_states 计数
```

端口规则（照 TreeverseAlgorithm.jl `src/treeverse.jl`，已核对）：
- 入口 `recurse(δ=extra, τ=binomial_fit, β=0, σ=0, ϕ=N)`；`s_0` 从一开始就在槽内（**不发 store(0)**，也永不 fetch(0))。
- `recurse` 进入时若 `σ>β`：先发 `restore(β)`，再 `call(j)` for j in β..σ，再 `store(σ)`，且本层 `δ -= 1`。
- `κ = mid(δ,τ,σ,ϕ)`，`mid = ceil((δσ+τϕ)/(τ+δ))`，且 `if κ>=ϕ && δ>0 { κ = max(σ+1, ϕ-1) }`；`while τ>0 && κ<ϕ { recurse(δ,τ,σ,κ,ϕ); τ-=1; ϕ=κ; κ=mid(...) }`。
- 末尾发 `grad(σ)`；若 `σ>β` 再发 `fetch(σ)`。
- 每个动作记录执行后的 `saved_states`（= 当前槽数，s_0 计入）。

- [ ] **Step 1: 失败测试 `tests/treeverse.rs`（数值全部来自学习单/参考实现推演）**

```rust
use seismic::treeverse::*;

#[test]
fn binomial_fit_values() {
    assert_eq!(binomial_fit(6, 2), 2);        // C(4,2)=6
    assert_eq!(binomial_fit(240, 1), 239);
    assert_eq!(binomial_fit(240, 3), 10);     // C(13,3)=286 ≥ 240
    assert_eq!(binomial_fit(240, 5), 5);      // C(10,5)=252
    assert_eq!(binomial_fit(240, 10), 3);     // C(13,3)=286
    assert_eq!(binomial_fit(1200, 5), 8);     // Marmousi: C(13,8)=1287
}

#[test]
fn n6_delta2_reference_schedule() {
    let a = schedule(6, 2);
    let seq: Vec<(Kind, usize)> = a.iter().map(|x| (x.action, x.n)).collect();
    assert_eq!(seq, vec![
        (Restore, 0), (Call, 0), (Call, 1), (Call, 2), (Store, 3),
        (Restore, 3), (Call, 3), (Call, 4), (Store, 5), (Grad, 5), (Fetch, 5),
        (Restore, 3), (Call, 3), (Store, 4), (Grad, 4), (Fetch, 4),
        (Grad, 3), (Fetch, 3),
        (Restore, 0), (Call, 0), (Store, 1),
        (Restore, 1), (Call, 1), (Store, 2), (Grad, 2), (Fetch, 2),
        (Grad, 1), (Fetch, 1),
        (Grad, 0),
    ]);
    assert!(seq.iter().filter(|(k, _)| *k == Kind::Call).count() == 8);   // 8 次前向
    let peak = a.iter().map(|x| x.saved_states).max().unwrap();
    assert_eq!(peak, 3);                                                  // ≤ 3 状态
}

#[test]
fn reflector_work_counts_match_answer_key() {
    for (extra, calls) in [(1usize, 28680usize), (3, 1695), (5, 990), (10, 642)] {
        let a = schedule(240, extra);
        assert_eq!(a.iter().filter(|x| x.action == Kind::Call).count(), calls);
        assert_eq!(a.iter().filter(|x| x.action == Kind::Grad).count(), 240);
        assert_eq!(a.iter().map(|x| x.saved_states).max().unwrap(), extra + 1);
    }
}

#[test]
fn grads_reverse_every_step_once_in_order_and_only_s0_left() {
    let a = schedule(240, 5);
    let grads: Vec<usize> = a.iter().filter(|x| x.action == Kind::Grad).map(|x| x.n).collect();
    assert_eq!(grads, (0..240).rev().collect::<Vec<_>>());
    let mut saved = std::collections::BTreeSet::from([0usize]);
    for x in &a {
        match x.action {
            Kind::Store => { assert!(!saved.contains(&x.n)); saved.insert(x.n); }
            Kind::Fetch => { assert!(saved.remove(&x.n)); }
            Kind::Restore | Kind::Grad => { assert!(saved.contains(&x.n)); }
            Kind::Call => {}
        }
        assert_eq!(x.saved_states, saved.len());
    }
    assert_eq!(saved.into_iter().collect::<Vec<_>>(), vec![0]);
}
```

- [ ] **Step 2: 确认失败**（`cargo test --test treeverse`）
- [ ] **Step 3: 实现**（上面端口规则；递归即可，N·log 级动作数）
- [ ] **Step 4: 通过**（若 `reflector_work_counts` 不匹配：核对 `mid` 的 ceil 与 guard、δ 减的位置——以 Julia 源码为准逐行对照）
- [ ] **Step 5: Commit** — `week5: treeverse scheduler ported with pinned tests`

## Task 22: Treeverse 执行器接入 adjoint（TDD）

**Files:**
- Modify: `week5/seismic/src/adjoint.rs`（或新 `src/checkpoint.rs`）、`src/main.rs`（`--storage`/`--checkpoints`）
- Test: `week5/seismic/tests/modes.rs`（追加）

**Interfaces:**
- Consumes: `treeverse::schedule`、`enzyme::step_vjp`、`wave::step`（重放用普通步进，无导数）。
- Produces: `--storage treeverse --checkpoints δ`：
  - 执行动作表：槽 `HashMap<usize, State>`（预置 s_0=零场）；`Restore(n)`: `work = slots[n].clone()`；`Call(n)`: `work` 用 `qamp(n)` 前进一步（计 scheduler_forward_calls）；`Store(n)`: `slots.insert(n, work.clone())`；`Grad(n)`: 先 `λ.u += Rᵀ w_n`，再 `step_vjp(primal=&slots[n], un_bar=λ.u)`，`img += d_c`，`λ=(d_um,d_u)`（计 reverse_calls；shot 0 录制规则同 Task 18）；`Fetch(n)`: 移除槽。
  - 每炮写 `<out>/actions-<shot>.json`：`{"actions": [{"action":"store","n":12,"saved_states":3}, ...]}`（小写动作名，serde rename）。
  - statistics：storage="treeverse"、checkpoints=δ、per_shot 含 `actions_file`。
  - 校验：treeverse 必须给 `--checkpoints`；full + checkpoints → 报错。

- [ ] **Step 1: 失败测试（追加）**：fixture(steps=6, δ∈{1,2,3}) 上 (a) treeverse image 与 full-history image **逐元素相等**（同一批 VJP、同顺序 → 逐位一致）；(b) 转置恒等式成立；(c) actions 文件存在且 grad 序 = 5..0；(d) peak_saved_states ≤ δ+1。
- [ ] **Step 2: 失败 → Step 3: 实现 → Step 4: 通过（`cargo test` 全绿）**
- [ ] **Step 5: Commit** — `week5: treeverse executor for checkpointed adjoint`

## Task 23: checkpoint 验证 + 动作图 + 工作-存储曲线

**Files:**
- Create: `week5/scripts/checkpoint_check.py`、`week5/scripts/checkpoint_work.py`、`week5/scripts/test_audit.py`
- Produces: `artifacts/checkpoint-{1,3,5,10}/*`、`artifacts/checkpoint-actions.png`、`artifacts/checkpoint-work.png`

- [ ] **Step 1: 按学习单跑四个预算**

```bash
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint --data \
    artifacts/born/born_data.npy --storage treeverse --checkpoints $b \
    --out artifacts/checkpoint-$b
done
```
- [ ] **Step 2: `scripts/checkpoint_check.py`**（核心审计逻辑写成可导入函数，`test_audit.py` 用合成动作表测它）：
  - 对每个预算：相对 L2 误差 vs `artifacts/adjoint/image.npy`（打印 + 断言 < 1e-9）；打印 peak saved states（期望 2/4/6/11）。
  - 审计每个 `actions-<shot>.json` 三计数：① grad 序偏离 N−1..0 的条数（含缺失/多余）；② 非法 restore（restore 的槽不在）；③ 超预算（saved_states > δ+1）——全部断言为 0。
  - 画 budget 5 第一炮动作图：x=操作序号、y=时间步、颜色=动作类型，存 `artifacts/checkpoint-actions.png`。
- [ ] **Step 3: `scripts/checkpoint_work.py`**：从各 `result.json` 的 statistics 取每炮 forward 步数与 peak bytes：panel 1（x=δ、y=forward 步/炮，log 轴，含 full history 240 的水平虚线）；panel 2（x=δ、y=peak bytes，线性，标注 "N states"、full 6,481,936 字节虚线）。断言 28,680/1,695/990/642 与 241 状态。存 `artifacts/checkpoint-work.png`。
- [ ] **Step 4: 运行 + 核对**（锯齿：store 爬升到最高 reverse 输入、grad 下降、每个 restore 开启更短重放）。
- [ ] **Step 5: Commit** — `git add week5/scripts/checkpoint_check.py week5/scripts/checkpoint_work.py week5/scripts/test_audit.py week5/artifacts/checkpoint-1 week5/artifacts/checkpoint-3 week5/artifacts/checkpoint-5 week5/artifacts/checkpoint-10 week5/artifacts/checkpoint-actions.png week5/artifacts/checkpoint-work.png && git commit -m "week5: checkpoint audit, action plot, work-storage curves"`

## Task 24: Marmousi：born + 6 状态成像

**Files:**
- Create: `week5/scripts/plot_marmousi.py`
- Produces: `artifacts/marmousi-born/*`、`artifacts/marmousi-image/*`、`artifacts/marmousi.png`

- [ ] **Step 1: 按学习单跑（绝不用 full history！）**

```bash
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint --data \
  artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 \
  --out artifacts/marmousi-image
```
Expected: 9 炮；每炮 1200 步；运行共约五分钟量级。

- [ ] **Step 2: `scripts/plot_marmousi.py`**（四 panel）：
  ① background（平滑 Marmousi，色标 km/s，标 8–14 km 对比窗）；② perturbation（±0.8 km/s 色标）；③ x=10 km 炮的 Born gather（x=检波器 km、y=时间 s 向下）；④ raw image（**全深度同一幅度标**，不做 gain；标上方 3 km 对比窗）。打印 image L2（断言 |L2/6.7037741e-4 − 1| < 1e-4）、peak saved states（=6）与 bytes（=20,788,320）。
- [ ] **Step 3: 运行核对**：背景平滑、扰动为短波长倾斜层；图像上部 3 km 见倾斜条带、深部在同标度下变弱； Born gather 有明确同相轴。
- [ ] **Step 4: Commit** — `git add week5/scripts/plot_marmousi.py week5/artifacts/marmousi-born week5/artifacts/marmousi-image week5/artifacts/marmousi.png && git commit -m "week5: marmousi born data and checkpointed image with six states"`

## Task 25: README、evidence 清单、收尾推送

**Files:**
- Create: `week5/README.md`；Modify:（如有遗漏）`.gitignore`

- [ ] **Step 1: 写 `week5/README.md`**：
  - evidence 表：每个文件（`inputs.png`、`ad/*` 5 件、`forward/*` 6 件、`born/*`、`adjoint/*` 5 件、`checkpoint-{1,3,5,10}/*`、`checkpoint-actions.png`、`checkpoint-work.png`、`marmousi-born/*`、`marmousi-image/*`、`marmousi.png`）旁边写产生它的命令或脚本（对照学习单 Final verification 的 inventory）。
  - 复现指南：venv 建立（`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`）、inputs 重新下载命令、`cargo install --path seismic`、各运行命令、Python 脚本清单。
  - 明确警告：**绝不 `--storage full` 跑 marmousi**（4.16 GB/炮）；`inputs/` 与 `.npy` 不入 git、由命令再生。
  - 回答学习单的问题（简短）：为何导数准确而图像仍 ≠ m —— 有限源带宽与炮-检覆盖使 JᵀJ 是带限/照明加权算子，RTM 恢复位置而非振幅/宽度。
  - Final verification 证据说明：restart 状态含两个场（`State{um,u}`）、一次局部 Enzyme 导数调用（`tests/enzyme.rs` 的点积恒等式）、重放不改变累计图像（Task 22 逐位相等测试 + checkpoint 相对误差 0）。
- [ ] **Step 2: 清点提交**：`git status` 确认无 `.npy`、无 `inputs/`、无 `target/`；每个 evidence 文件 < 5 MB（`find week5/artifacts -size +5M` 应为空）。`git add` 余量文件。
- [ ] **Step 3: Commit + push** — `git commit -m "week5: readme evidence inventory and wrap-up"` 然后 `git push`。
- [ ] **Step 4: 课后人工项（提醒用户，不在代码内）**：用课程查看器复核 wavefield/echo/adjoint 录像；下次课向导师展示导数检查、存储-工作量曲线与地震图像；按学习单开始 Week 6 的云账号注册。

---

## Self-Review 记录

- **Spec 覆盖**：Part 1 的 DO（derivatives.json 字段）→ Task 3；VERIFY(1)(2)(3) → Tasks 3/4/5。Part 2 的 DO（下载、design.toml、forward 模式）→ Tasks 6/11/12；VERIFY(1)(2)(3) → Tasks 12/13（帧图以脚本渲染代替查看器，已列为待确认问题 3）。Part 3 的 DO（skill 安装、born、adjoint）→ Tasks 14/17/18；VERIFY → Task 19。Part 4 的 DO（面板、移植、四预算 + Marmousi 两跑）→ Tasks 20/21/22/23/24；VERIFY(1)(2)(3) → Tasks 23/24。Final verification（README 清单、提交推送、5MB 上限、license）→ Task 25。
- **占位符扫描**：Task 9 测试中一处笔误（`s[6 * 41 + 20]]`）已在正文标注修正；Task 11 的 forward 帧记录顺序在代码注释中给了明确规则（步进前记 `u^n`）并以测试 `[0,2,4]` 钉死；Task 15 的 Enzyme 影子参数序以编译器报错为准微调、以三个恒等式测试为门（这是该 nightly 特性的固有不确定性，已在任务内写明判定标准）。除此之外无 TBD/“适当处理”类占位。
- **类型一致性**：`wave::step` 与 `enzyme::step_*` 参数序逐字对齐（Task 10/15 接口块相同）；`Action{action,n,saved_states}`、`Statistics/PerShot` 字段名与 Task 22/23 的 JSON 消费一致；Python 侧 `ad_lib`/`cluster_lib` 的函数名在 Tasks 3–5 间一致。
