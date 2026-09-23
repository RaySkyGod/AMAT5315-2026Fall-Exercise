# Week 4 谱方法流体求解器（field / fluid CLI）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按第 4 周学习单完成 `week4/`：一个 Rust crate（`Integrator` trait 库 + `field`/`fluid` 两个命令行）求解周期 [0,2π)² 上的二维不可压涡度方程（Fourier 伪谱 + 2/3 规则），加 Python 证据脚本，产出 9 个 evidence 文件并通过全部 VERIFY 验收。

**Architecture:** 单 crate `week4/`（名 `fluid`，lib + 两个 `[[bin]]`：`field` 生成初始场 JSON，`fluid` 从 stdin 读场并用 Euler/rk2/rk4 积分、写 tsv + run.json + fields.jsonl）。Python 脚本放在 `week4/scripts/`（从那里运行），负责跑管线（必要时 subprocess 调 `field|fluid`）、对比、画图，输出到 `week4/evidence/`。`artifacts/` 与 `target/` 不入 git。

**Tech Stack:** Rust edition 2024（clap 4 derive、serde/serde_json、rustfft 6、num-complex、anyhow），Python 3 + numpy + matplotlib + pytest。

**Spec:** `week4/week4-learning-sheet.pdf`（权威；本计划所有数值验收来自其 DO/VERIFY 面板）。

## Global Constraints（全文约束，每个任务默认继承）

- 周期域 [0,2π)²，格点 x_j = jΔx、Δx = 2π/N，端点 2π 不重复；数组下标 `l*N + j` 存 f(x_j, y_l)。
- FFT 约定（式 7）：正变换带 1/N（二维 1/N²），逆变换不带；系数顺序 0,1,…,N/2−1,−N/2,…,−1；Nyquist 模 k=−N/2 的一阶导为零（不传播），二阶导 −k² 保留。
- 积分器（式 10）逐字实现：Euler、midpoint(rk2)、RK4（权重 1,2,2,1）；另有"等权重 RK4"（h/4·(k1+k2+k3+k4)）供 Part 1 验证用。
- 稳定性常数：RK4 实轴 −2.785、虚轴 ±2.83i；Euler/midpoint 实轴 −2、虚轴无。
- 2/3 规则：涡度本身与所有乘积只保留 |kx|,|ky| ≤ K = floor(N/3)；k²_max = 2K²。
- 式 9 谱：λ_k = −νk² − ick；单波精确解 û_k(t) = û_k(0)e^{λ_k t}。
- 式 12 微分乘子；式 13 Poisson：ψ̂_k = ω̂_k/(kx²+ky²)，ψ̂_0 = 0；u = ∂yψ，v = −∂xψ。
- 涡度方程（式 3）：∂tω = −(u∂xω + v∂yω) + ν∇²ω；midpoint/RK4 每个 stage 用自己的 ω 重建速度。
- 能量 E = ½⟨u²+v²⟩，涡量 Z = ½⟨ω²⟩（⟨⟩ 为盒平均）；相对误差 ‖f−f_exact‖/‖f_exact‖（网格平方和范数）。
- Taylor–Green（式 15）：u = cos x sin y·e^{−2νt}，v = −sin x cos y·e^{−2νt}，ω = −2cos x cos y·e^{−2νt}；E = e^{−4νt}/4，Z = e^{−4νt}/2；其平流项恒为零。
- `fluid` 输出契约（fluid.design.toml）：stdout 先表头再每快照一行 `t  E  Z`（6 位小数、tab 分隔）；能量首次非有限 → 打印该行并 exit 1；`<out>/run.json` = case,n,seed,k_band（照抄 stdin）+ method,nu,dt,t_end,snapshot_every；`<out>/fields.jsonl` 每快照 {t, step, u, v, omega}（n² 数组、6 位小数）。
- 快照规则：snapshot_every = round(every/dt) 步，第 0 步存、之后每 snapshot_every 步存；步长绝不缩短凑快照；积分直到 t ≥ t-end（整步）。
- 敏感性扰动：δω = −7e-5·M·cos(3x)cos(4y)，M = 该工况初始 u,v 绝对值最大者。
- Richardson（式 18）：e_h ≈ ‖ω_{2h}−ω_h‖/((2⁴−1)‖ω_h‖)；预测 e_{h′} ≈ e_h(h′/h)⁴。
- 随机场：环带 k-min ≤ |k| ≤ k-max 内涡度模等幅、相位由 --seed 的 PRNG 在 [0,2π) 均匀抽取，抽取顺序与 n 无关；缩放使 E(0)=0.5。
- 工作规则（学习单 Part 1）：每次用户提示后先 commit 再回复；不确定先问；所有脚本存 `week4/scripts/` 并从中运行。提交信息前缀 `week4: ...`。
- 每个 Rust 任务结束 `cargo test`（在 `week4/`）全绿；Python 脚本配轻量 pytest。
- `artifacts/`、`target/`、`__pycache__/` 不入 git；`Cargo.lock` 入 git。

## 开工前待用户确认的问题

1. 三条工作规则是否照单采纳（每 prompt 后 commit；不确定必问；脚本入 `week4/scripts/` 并从那里运行）？
2. crate 名 `fluid`（两个二进制 `field`、`fluid`）是否可以？
3. FFT 依赖用 `rustfft`（纯 Rust、无系统依赖）是否可以？
4. Part 1 需"用库里的 RK4 实测增长因子/积分脉冲"：计划用 `cargo test`（integration test 写 JSON 到 `artifacts/`）供 Python 画图 —— 机制是否符合预期？
5. 灰色面板的管线命令（`field … | fluid …`）：由我用 bash 代跑，还是你自己在第二终端跑？（计划默认我代跑，README 仍按你手动可复现写）
6. Challenge（ETDRK4、Zeitlin 等谱方法）默认不做，是否需要？

---

## Part 1：线上的积分器（学习单 pp.3–7）

### Task 0: 脚手架 + 设计文件

**Files:** Create `week4/Cargo.toml`, `week4/src/lib.rs`, `week4/.gitignore`, `week4/field.design.toml`, `week4/fluid.design.toml`, `week4/scripts/`, `week4/evidence/`（空目录放 `.gitkeep`）。

**Interfaces:** Produces crate `fluid`（lib 名 fluid）；设计文件内容照抄学习单绿色面板逐字（`field.design.toml`: usage/args/writes + [taylor-green] + [random]；`fluid.design.toml`: usage/args/writes）。

- [ ] Step 1: `mkdir -p week4/{src,scripts,evidence} week4/artifacts`（artifacts 不入 git）；写 `Cargo.toml`：

```toml
[package]
name = "fluid"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rustfft = "6"
num-complex = "0.4"

[[bin]]
name = "field"
path = "src/bin/field.rs"

[[bin]]
name = "fluid"
path = "src/bin/fluid.rs"
```

- [ ] Step 2: 从学习单第 8–9 页逐字抄两个 design.toml（块对块，不许改写）。
- [ ] Step 3: `cargo build` 通过；commit `week4: scaffold crate and design files`。

### Task 1: `Integrator` trait + 四个积分器

**Files:** Create `src/integrators.rs`（挂到 lib.rs），Test `tests/integrators.rs`。

**Interfaces:** Produces:

```rust
use num_complex::Complex64 as C;
/// 速率函数 F: 状态 -> d(state)/dt。状态一律用复向量（实数问题虚部为 0）。
pub type RateFn<'a> = &'a dyn Fn(&[C]) -> Vec<C>;
pub trait Integrator {
    fn step(&self, u: &[C], f: RateFn, h: f64) -> Vec<C>;
    fn name(&self) -> &'static str;
}
pub struct Euler; pub struct Midpoint; pub struct Rk4; pub struct EqualWeightsRk4;
pub fn by_name(method: &str) -> anyhow::Result<Box<dyn Integrator>>; // "euler"|"rk2"|"rk4"
```

（RK4 内部：k1=F(u); k2=F(u+h/2·k1); k3=F(u+h/2·k2); k4=F(u+h·k3); u+h/6·(k1+2k2+2k3+k4)。EqualWeightsRk4 权重各 1/4。）

- [ ] Step 1: 失败测试 `tests/integrators.rs`：
  - 单步 vs 稳定性函数：u′=λu（λ 取 −1.3+0.7i 等 3 个值），一步 h 后 |y1 − e^{λh}| < 1e-14（对 Euler/Midpoint/Rk4/EqualWeightsRk4 分别等于式 11 的多项式与 e^z 的差）。
  - 阶数：积分 u′=u, u(0)=1 到 t=1，h = 1/4,1/8,…,1/128，log-log 斜率 ∈ p±0.1（Euler p=1，Midpoint 2，Rk4 4，EqualWeightsRk4 2）。
  - `by_name("rk4").name() == "rk4"`，未知名返回 Err。
- [ ] Step 2: `cargo test` 确认失败（模块未建）。
- [ ] Step 3: 实现 `integrators.rs`；`cargo test` 全绿。
- [ ] Step 4: commit `week4: integrator trait with euler/midpoint/rk4/equal-weights`。

### Task 2: 周期线（FFT 微商 / 中心差分 / 谱 / 最大稳定步）

**Files:** Create `src/fft.rs`（rustfft 封装：正变换带 1/N、逆不带、`wavenumbers(n) -> Vec<i64>` 即 0..n/2-1,−n/2..−1）、`src/line.rs`，Test `tests/line.rs`。

**Interfaces:** Produces:

```rust
pub struct Line { pub n: usize, pub c: f64, pub nu: f64 }
impl Line {
    pub fn fourier_rate(&self, u: &[C]) -> Vec<C>;  // IFFT[(−νk²−ick)·FFT(u)]，Nyquist 一阶导置零
    pub fn fd_rate(&self, u: &[C]) -> Vec<C>;       // 式 8 中心差分（周期回绕）
    pub fn spectrum(&self) -> Vec<C>;               // 每个格点序的 λ_k
    pub fn largest_stable_step(&self, integ: &dyn Integrator) -> f64;
    // 对所有 k 用积分器实测一步增长因子 |R(λ_k h)|，二分求 max_k |R|=1 的 h
}
```

- [ ] Step 1: 失败测试：
  - 单波精确：初始 û_k=1（k=3 等三个模），RK4 dt=0.01 积分到 t=1（ν=0.05,c=1），误差 < 1e-13（fourier_rate）。
  - Nyquist：初始 Nyquist 模，c=1：平流 10 步后幅值不变（只按 −νk² 衰减）。
  - fd_rate 与手算三点模板一致（小 n=8 硬编码对照）。
  - `largest_stable_step(&Rk4)`（ν=0.05,n=64,c=1）∈ [0.0489, 0.0499]（学习单：精确 0.0494；对 Euler 应得 ≈ 抛物线穿单位圆盘的更小值）。
- [ ] Step 2: 跑测试确认失败 → 实现 → 全绿 → commit `week4: advection-diffusion line with fourier and fd rates`。

### Task 3: Part 1 证据（line-stability.png、line-accuracy.png）

**Files:** Create `tests/evidence_line.rs`（写 `artifacts/growth-map.json`、`artifacts/pulse-h{0.045,0.056}.json`、`artifacts/line-accuracy.json`），`scripts/line_stability.py`、`scripts/line_accuracy.py`，Test `scripts/test_line_scripts.py`。

- [ ] Step 1: `tests/evidence_line.rs`：
  - 增长因子图数据：z 网格 x∈[−4.5,1.5]×y∈[−4,4]（约 480×400），**用库里的 RK4** 对 y′=λy（λ=z, h=1）积分一步，记录 |y1|（即实测 |R_RK4(z)|）。
  - 脉冲面板数据：σ=0.35、中心 π/2、周期像求和的高斯脉冲（n=64,c=1,ν=0.05），库 RK4 积分到 t=6，dt=0.045 与 0.056，存 u(x,t) 网格（每步一行）。
  - 精度面板数据：面板 1 = σ=0.25, ν=0.002, c=1, n=64, t=2π 三条终态曲线（RK4 dt0.02 Fourier；RK4 dt0.02 中心差分；Euler dt0.005 Fourier）+ 精确解 + 各自最大误差；面板 2 = σ=0.35, ν=0.05, t=1，四方法 × dt{0.02,0.01,0.005,0.0025} 的最大误差。精确解 = 平流热核高斯：σ_t=√(σ²+2νt)，u=σ/σ_t·exp(−(x−π/2−ct)²/(2σ_t²))（周期像求和）。
  - JSON 全部写 `artifacts/`；测试同时断言若干锚点（0.045 无 |u|>2 的爆发、0.056 有 |u|>10；误差序列单调下降）。
- [ ] Step 2: Python 脚本（在 `scripts/` 内、从那里运行）：
  - `line_stability.py`：左面板 pcolormesh(log|R| 实测)，叠加解析 |R|=1 等值线（式 11，三方法，黑/虚线），点出 λ_k·h（h=0.045 圆点、0.056 叉点），坐标轴标 −2.785、±2.83i；中、右面板 u 的 x–t 色图（0.045、0.056）→ `../evidence/line-stability.png`。
  - `line_accuracy.py`：面板 1 四条曲线 + 图例含最大误差；面板 2 log-log 四方法误差点 + 最小二乘拟合线，图例标斜率（numpy polyfit on log10）→ `../evidence/line-accuracy.png`，终端打印所有误差与斜率。
- [ ] Step 3: `pytest scripts/`（测：拟合斜率函数对已知幂律数据返回 ±0.01；JSON 键存在）+ `cargo test` 全绿。
- [ ] Step 4: 生成两图，commit `week4: line stability map and accuracy evidence`。

**🚧 Part 1 人工确认（做完暂停，请你看）**
1. 打开 `week4/evidence/line-stability.png`：着色边界贴住黑色 |R_RK4|=1 曲线；曲线过轴点 −2.785、±2.83i；0.045 的点全在区域内、0.056 的最外点在外、抛物线尖端在 h≈0.049 触界；0.045 面板为一条干净斜条纹直到 t=6，0.056 面板在 t=6 前爆发成两格宽条纹。
2. 打开 `week4/evidence/line-accuracy.png`：RK4+Fourier 与精确解重叠；中心差分脉冲迟到且带尾迹；Euler 回来更高；斜率 ≈ 1.03 / 2.01 / 4.00 / 2.00（各在 1、2、4、2 的 15% 内）；终端打印的三个最大误差数量级合理。
3. 确认两个 design.toml 与学习单绿色面板逐块一致（后面 wrap-up 还要与参考件 diff）。

---

## Part 2：流场离散与求解器（field / fluid CLI）（学习单 pp.8–12）

### Task 4: 二维谱算子（微分、Poisson、2/3 规则）

**Files:** Create `src/spectral.rs`（含 fft2/ifft2、每轴 wavenumbers、K=floor(n/3) 掩模），Test `tests/spectral.rs`。

**Interfaces:** Produces:

```rust
pub struct Spectral { pub n: usize, pub kmax2: f64 /* = 2K² */ }
impl Spectral {
    pub fn fft2(&self, a: &[f64]) -> Vec<C>;          // 1/n²
    pub fn ifft2(&self, a: &[C]) -> Vec<f64>;
    pub fn deriv(&self, f: &[f64], ax: u32, ay: u32) -> Vec<f64>; // (ikx)^a(iky)^b，奇阶 Nyquist 置零
    pub fn laplacian(&self, f: &[f64]) -> Vec<f64>;
    pub fn vorticity(&self, u: &[f64], v: &[f64]) -> Vec<f64>;    // ∂x v − ∂y u，再 2/3 截断
    pub fn velocity_from_omega_hat(&self, w: &[C]) -> (Vec<f64>, Vec<f64>); // Poisson(式13)+微商
    pub fn dealias(&self, w: &mut Vec<C>);
    pub fn energy_enstrophy(&self, w: &[C]) -> (f64, f64); // Fourier 空间：E=½Σ|ω̂|²/k²（k=0 模跳过），Z=½Σ|ω̂|²
    pub fn omega_rhs(&self, w: &[C]) -> Vec<C>;  // ω→网格，逐 stage 重建 (u,v)，−(uω_x+vω_y) 正变换后 dealias，−νk²·w 相加
}
```

- [ ] Step 1: 失败测试（学习单 VERIFY(1)，n=32，g=sin3x·cos2y）：
  - `deriv(g,1,0)`、`deriv(g,2,0)`、`deriv(g,1,1)`、`laplacian(g)` vs 解析 3cos3x cos2y、−9g、−6cos3x sin2y、−13g：最大误差 < 1e-10。
  - 同样四个导数用中心差分（周期回绕）在 n=32 与 n=64 算，测试打印对照表并断言 64/32 误差比 ∈ [3.5, 4.5]（二阶收敛）。参考值：dx 0.17050/0.04318、dxx 0.25724/0.06487、dxdy 0.48534/0.12429、∇² 0.30838/0.07771。
  - Poisson：给 ψ=cos x cos y，ω=−∇²ψ=2ψ → `velocity_from_omega_hat` 还原 u=cos x sin y、v=sin x cos y，误差 < 1e-12。
  - 2/3 规则：n=64 时 K=21、kmax2=882；放一个 k=(25,0) 的模，dealias 后为零。
- [ ] Step 2: 失败 → 实现 → 绿 → commit `week4: 2d spectral operators with two-thirds rule`。

### Task 5: `field` CLI

**Files:** Create `src/field_gen.rs`（Taylor–Green 与随机场生成 + 小型 xoshiro256++ 相位 PRNG，沿用 week3 的实现模式）、`src/bin/field.rs`（clap 子命令 taylor-green / random，参数与默认值严格按 field.design.toml）。

**Interfaces:** Produces stdout JSON：`{case, n, seed, k_band, u, v}`（u,v 为 n² 数组；TG 的 seed/k_band 为 null；random 的 k_band=[k_min,k_max]）。相位抽取顺序与 n 无关：按 规范字典序遍历环带内"正半"模，逐个抽 φ∈[0,2π)，负模取共轭；幅度 A 使 E(0)=0.5（E=½Σ|ω̂_k|²/k²）。

- [ ] Step 1: 失败测试 `tests/field_gen.rs`：
  - TG（n=64,ν=0.1,t=0）：u,v 与式 15 逐点一致（<1e-15）；由 u,v 反算 ω = −2cos x cos y；E=0.25、Z=0.5。
  - random（n=128, seed 2026, k 2–6）：E(0)=0.5（<1e-12）；Z(0) ≈ 6.63–6.66（与答案键 6.657 同量级；等幅无相位依赖，差 >1% 即错）；同 seed、n=64 与 n=128 的低 k 相位一致（顺序无关性）。
- [ ] Step 2: 失败 → 实现 → 绿 → commit `week4: field generator cli (taylor-green, random)`。

### Task 6: `fluid` CLI（求解器主体）

**Files:** Create `src/solver.rs`（把 `omega_rhs` 包成 RateFn）、`src/run.rs`（主循环 + 快照 + 非有限停止 + 输出）、`src/bin/fluid.rs`。

**Interfaces:** Consumes Task 1 的 `by_name`、Task 4 的 `Spectral`。Produces 管线 `field … | fluid --method rk4 --nu … --dt … --t-end … --every … --out …`。

- [ ] Step 1: 失败测试 `tests/fluid_run.rs`（库层，不经进程）：
  - TG 一致性：ω̂(0) 由 TG u,v 反算；RK4 dt=0.01、ν=0.1 积到 t=1：E=0.167580、Z=0.335160（六位小数）；速度场相对误差 vs 式 15（t=1）< 1e-5（预期 ≈7e-7）。
  - 快照规则：dt=0.03、every=0.1 → snapshot_every=round(0.1/0.03)=3；步不缩短：t_end=1 时最后 t=30·0.03=0.9 <1 再走一步到 0.93？——按"积分直到 t≥t_end"共 ceil(1/0.03)=34 步、末 t=1.02。
  - 非有限：TG ν=0.1、dt=0.04 → 某步 E 为 NaN，记录该行、返回 Err(exit 1 语义)；fields.jsonl 只含停止前的快照。
- [ ] Step 2: 失败 → 实现：
  - 主循环：`while t < t_end − 1e-12 { w = integ.step(&w, &rate, dt); t += dt; E,Z = energy_enstrophy(w); if !E.is_finite() {打印并退出 1} if step % every_steps == 0 {存快照 u,v,ω(6 位小数) + 打印行} }`。
  - run.json 键序照 fluid.design.toml：case,n,seed,k_band,method,nu,dt,t_end,snapshot_every。
- [ ] Step 3: `tests/cli.rs` 端到端（`CARGO_BIN_EXE_field`/`_fluid`，管道接 stdin）：TG 管线 stdout 首两行 = `t Energy E Enstrophy Z` / `0.0 0.250000 0.500000`，末行 `1.0 0.167580 0.335160`；run.json/fields.jsonl 存在且键正确。
- [ ] Step 4: 全绿后 commit `week4: fluid solver cli with snapshots and blowup stop`。

### Task 7: Part 2 证据（taylor-green.png）

**Files:** Create `scripts/taylor_green.py`（若 artifacts 缺失则自己起管线 subprocess），Test `scripts/test_part2_scripts.py`。

- [ ] Step 1: 脚本流程：`cargo install --path . --quiet`（README 记录；脚本只假设 field/fluid 在 PATH）；`field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out ../artifacts/taylor-green > ../artifacts/taylor-green.tsv`；`field taylor-green --n 64 --nu 0.1 --t 1 > ../artifacts/taylor-green/exact-t1.json`；读末帧 fields.jsonl vs exact-t1.json 打印速度场相对误差；画 t=0 与 t=1 涡度色图（共享色标，速度箭头 quiver 下采样）→ `../evidence/taylor-green.png`。
- [ ] Step 2: pytest（相对误差函数、6 位小数解析）+ 跑出图；commit `week4: taylor-green evidence`。

**🚧 Part 2 人工确认（做完暂停，请你看）**
1. 终端导数对照表：Fourier 列全部 < 1e-10；FD 64/32 误差比 ≈ 4（二阶）；数值与学习单参考（0.17050 等）同量级。
2. TG 管线输出：`0.0 0.250000 0.500000` → `1.0 0.167580 0.335160` 六位小数一致；速度场相对误差 ≈ 7e-7（必须 < 1e-5）。
3. 打开 `evidence/taylor-green.png`：两面板共享色标；四个涡胞位置与符号正确（红涡跨周期边界、两半在对面相接）；核不动、只整体变淡（t=1 max|ω|≈1.637）；箭头沿等 ψ 线。若是"条纹"而非涡胞 → 空间微商有错。
4. 抽查 `run.json`/`fields.jsonl` 与 fluid.design.toml 描述逐项一致（键名、6 位小数、seed/k_band 照抄）。

---

## Part 3：找稳定极限（学习单 pp.13–16）

### Task 8: 基线运行 + 越界运行 + 扫描 + blowup.png

**Files:** Create `scripts/run_pipelines.py`（可复用的 subprocess 管线函数）、`scripts/blowup.py`。

- [ ] Step 1: 基线（学习单灰色面板等价命令，脚本代跑）：random（seed 2026,n=128,k 2–6,ν=0.004,dt 0.01,t 10,every 0.1）→ `artifacts/random{.tsv,/}`；TG 越界 dt=0.04,t 4 → `artifacts/unstable/taylor-green*`。脚本打印随机初场的最大速度 U_max（答案键 2.43，我们相位不同、数值会不同——**记录它**）。
- [ ] Step 2: 扫描（存 `artifacts/scan/`，快照 every 0.5）：TG n=64,ν=0.1,t=8 RK4 dt∈{0.032,0.033}；random n=128,ν=0.004,t=10 RK4 dt∈{0.038,0.040} 与 Euler dt=0.01。断言性检查写进脚本输出：TG 0.033 应 non-finite（预测临界 0.0316）；Euler 0.01 应在前两个时间单位内 non-finite。
- [ ] Step 3: `blowup.py`：两面板能量–时间（log 纵轴）；TG 两条都画（0.032/0.033，虚线画精确 e^{−0.4t}/4）；random 画一条到 t=10 的 RK4、一条爆掉的 RK4、Euler；不稳定曲线标注停止时间 → `evidence/blowup.png`。

### Task 9: 敏感性对跑 + sensitivity.png、random.png

**Files:** Create `scripts/sensitivity.py`、`scripts/random_flow.py`。

- [ ] Step 1: `sensitivity.py`：对 TG(n=64,ν=0.1) 与 random(n=128,ν=0.004) 各跑两次 RK4 dt=0.01 到 t=20（every 0.5）：第一次原始初场；第二次在 Python 里读初场 JSON、按 δω=−7e-5·M·cos3x·cos4y（M=该场 max|u|,|v|）加到 ω 上再管道给 fluid（field 的契约不加参数，扰动在脚本内加）。画 ‖ω1−ω2‖/‖ω1‖–t（log 纵轴，两曲线）→ `evidence/sensitivity.png`。
- [ ] Step 2: `random_flow.py`：读 `artifacts/random.tsv` 头尾 + `fields.jsonl`，画 t=0,2,5,10 四帧涡度（一行、共享色标、每帧标 E,Z）→ `evidence/random.png`。
- [ ] Step 3: pytest（扰动幅度计算、tsv 解析）+ 生成三图；commit `week4: stability limit and sensitivity evidence`。

**🚧 Part 3 人工确认（做完暂停，请你看；本部分最可能需要你拍板）**
1. `random.tsv` 首尾：E 从 0.50 降到 ≈0.29、Z 从 ≈6.66 降到 ≈0.95（能量减不到一半、涡量降约 7 倍）。若 Z(0) 偏离 6.657 超过 ~1% → 环带或幅度定义有错（先人工判断再改）。
2. 随机流 RK4 边界 bracketing 结果需要你确认：0.038/0.040 是否恰好"一稳一爆"。**若都爆或都稳，按学习单指示换更小/更大 dt 重跑**——增减方向和新步长请你确认后我再跑。验收：测量边界落在 2.83/(U_max·59.4) 的 1–3 倍（答案键 bound 0.0196、边界 0.038–0.040）。
3. TG 边界必须在 0.032–0.033 之间（预测 0.0316 的 5% 内）；dt=0.04 的 tsv 末行 non-finite 且在 t<4 内。
4. 打开 `evidence/blowup.png`：TG 面板 0.032 贴着精确虚线、0.033 在 t≈6.5 后抬升直至 non-finite；random 面板 Euler 从第一步就增、t≈1.4 前后 non-finite；每条不稳曲线有停止时间标注。
5. 打开 `evidence/sensitivity.png`：随机对到 t=20 分离 >10 倍（答案键 ×40、e-folding ≈3）；TG 对衰减到舍入地板（~1e-6，两位一致到六位小数）；两条曲线都不"造能量"。
6. 打开 `evidence/random.png`：t=0 小涡满盒 → t=2 出细丝 → t=5 大核存活细丝消退 → t=10 只剩宽淡结构；四帧同一色标。

---

## Part 4：测精度阶并选步长（学习单 pp.16–18）

### Task 10: 阶数测量（order.png）

**Files:** Create `scripts/order.py`。

- [ ] Step 1: 跑 TG n=8,ν=0.5,t=2，RK4 dt∈{0.4,0.25,0.2}，存 `artifacts/order/rk4-dt<dt>/`；`field taylor-green --n 8 --nu 0.5 --t 2` 精确场。相对误差（速度，网格范数）三点 log-log + 最小二乘拟合（图例标斜率）+ 6 位小数地板虚线（≈1.5e-7 量级）→ `evidence/order.png`。预期斜率 ≈4.10（验收：4 的 ±15%）。稳定性自查：νk²_max·dt = 0.5·8·0.4 = 1.6 < 2.785 ✓。
- [ ] Step 2: pytest（拟合函数复用）+ commit `week4: rk4 order measurement on the fluid`。

### Task 11: 自收敛 + Richardson 选步（convergence.json/png）

**Files:** Create `scripts/convergence.py`，Test `scripts/test_convergence.py`。

- [ ] Step 1: 跑 random n=128,ν=0.004,seed 2026,k 2–6,t=2，RK4 dt∈{0.02,0.0125,0.01} + 参考 dt=0.0025（同网格同初场），存 `artifacts/convergence/`。对每 run 取 t=2 的 ω（fields.jsonl 末帧）与参考比，得相对误差；三点 log-log 斜率。写 `evidence/convergence.json`：{runs:[{dt, error}], slope, richardson:{e_h_0.01_est, predictions:{0.02,0.0125,0.01}}, chosen_dt, predicted, measured}。
- [ ] Step 2: Richardson（式 18）：由 dt=0.02 与 0.01 的保留末帧 ω：e_{0.01} ≈ ‖ω_{0.02}−ω_{0.01}‖/(15·‖ω_{0.01}‖)；预测 e(h′)=e_{0.01}(h′/0.01)⁴；选预测 < 5e-6 的最大候选步（预期 0.0125、预测 ≈3.46e-6、实测 ≈3.40e-6）；打印所选步及预测/实测误差。
- [ ] Step 3: 画 `evidence/convergence.png`：误差点 + 拟合线（图例斜率，验收 3.7≤q≤4.3，键 4.024）+ 标记所选步长的点/竖线。
- [ ] Step 4: pytest：Richardson 公式对合成四阶数据反推误差正确；commit `week4: self-convergence and richardson step choice`。

**🚧 Part 4 人工确认（做完暂停，请你看）**
1. `evidence/order.png`：斜率 ≈4.10（4±15% 内）；每个误差点都明显高于 6 位小数地板（否则斜率不可信）。
2. `cat evidence/convergence.json` + 打开 `evidence/convergence.png`：斜率 3.7–4.3；所选步 dt=0.0125（预测 ≈3.46e-6、实测 ≈3.40e-6，均 < 5e-6）；dt=0.02 超标；图上标注了所选步。预测值与实测值同数量级是 Richardson 成立的证据，请人工核对打印输出。
3. 确认参考 run 与被测 run 用了完全相同的初场（同 seed、同网格）——JSON 里的 seed/n 请你过目。

---

## 收尾：README、清点、推送（学习单 pp.18–19）

### Task 12: README + 证据清点 + push

**Files:** Create `week4/README.md`；Modify `.gitignore`（week4/artifacts、target、__pycache__）。

- [ ] Step 1: README 按学习单要求写：干净 clone 的再生指南——install 命令（`cargo install --path . --quiet`）、两条管线及其参数值、再按序每个脚本与其产物文件；`evidence/` 逐文件列在命令旁；说明 `artifacts/` 不入库、本地保留以再生图。
- [ ] Step 2: 清点 evidence 9 项：line-stability.png、line-accuracy.png、taylor-green.png、blowup.png、sensitivity.png、random.png、order.png、convergence.png、convergence.json；src/ 结构：三积分器、线、求解器、两条命令行；scripts/ 五类脚本齐全。
- [ ] Step 3: 两个 design.toml 与参考件（教师分发的参考 `field.design.toml`/`fluid.design.toml`）逐块 diff——若本周没有参考件可 diff，则与学习单绿色面板人工对照。
- [ ] Step 4: commit `week4: readme and evidence inventory`；**push 前请你确认 remote 目标**，然后 push。

**🚧 收尾人工确认**
1. README 的命令序列从干净 clone 可逐条复跑（我可以代跑一遍全流程验证，你确认是否要）。
2. evidence/ 9 个文件齐全、都在 git 里；artifacts/ 与 target/ 未被跟踪（`git status` 你过目）。
3. push 目标与提交历史（`week4: …` 前缀）你确认后执行 push。
4. 是否要写 Challenge（ETDRK4 / Zeitlin 等谱）——默认不做，留待你决定。

---

## 风险与开放点（开工前已知）

- **随机相位依赖**：同 seed 不同 PRNG → U_max 与答案键 2.43 不同，advective 边界随之移动；Part 3 的 bracketing 可能需要按学习单的"都爆则减小 / 都稳则增大"规则迭代——已列为人工确认点 3.2。
- **Z(0) 锚点**：等幅无相位依赖，理论 ≈6.63–6.66；若实测偏离 >1% 说明环带/幅度约定与答案键不同，先停下来讨论而不是硬调。
- **rustfft 可用性**：需联网拉 crate；若离线则需换 vendored 方案（到时候问你）。
- **证据数据由 Rust 侧生成的机制**（cargo test 写 artifacts JSON）：学习单要求"用库里的 RK4"，此机制是我的设计选择，开工前请你确认（见问题 4）。
- Challenge 为可选研究性内容，本计划不含。
