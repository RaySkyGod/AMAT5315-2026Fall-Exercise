# Week 8: Language Models and the Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 AMAT5315 Week 8 学习单全部六个 Part：预训练嵌入表中的"关系即方向"（potion-base-8M + 洗牌对照）、TinyStories 上的 n-gram 计数模型（精确归一化、add-k 平滑、χ² 采样检验、维度灾难）、键–值记忆存储器与 MLP 语言模型（容量 ≈ d、d^p，MLP 存下 bigram 表）、注意力=从文本写入的 softmax 存储器（「算」字图片召回）、两层 pre-norm transformer（训练、采样、四种 sampler 的精确目标分布、surprise 自洽）以及 Chinchilla 缩放律重拟合（Huber + bootstrap + compute-optimal 分配），并产出全部 evidence 图与 README。

**Architecture:** 纯 CPU；JAX 写模型（MLP LM、transformer），NumPy 做检查与计数（n-gram 用 int64 打包键 + `np.unique`/`searchsorted`，避免 512^n 稠密表）；matplotlib 出图。核心数值均为纯函数模块（`embeddings.py`、`stories.py`、`ngram.py`、`stores.py`、`mlp_lm.py`、`picture.py`、`transformer.py`、`samplers.py`、`scaling.py`），脚本层只做训练/评估/出图。测试先写、失败后再实现，每个 Task 一次 commit。

**Tech Stack:** Python 3.12+（uv 管理）、jax（CPU，与 week7 同版本线 0.11.x）、numpy、matplotlib、safetensors（不装 PyTorch）、sentencepiece（tok512 BPE）、scipy（L-BFGS、χ² 分布）、pytest；下载用 `curl` 与 `huggingface-hub==1.33.0`（一次性 `uv run --with`）。

**Spec:** `week8/week8-learning-sheet.pdf`（24 页，已通读）。本计划的全部设置、下载命令与验收阈值均逐字抄自学习单 DO/VERIFY 部分；阈值见各 Task 的"验收"。

## Global Constraints

- 工作目录 `week8/`，**uv** 管理（`uv run python ...`、`uv run pytest tests/`）；CPU 基线；学习单称全周计算约 15 分钟。
- 图与结果表存 `evidence/`（进 git）；原始运行产物（权重、token 流缓存、拟合 JSON）存 `artifacts/`（**不进 git**）；下载（`data/` 语料与 CSV、`inputs/` 模型文件）**不进 git**。
- 每个 Task 结束 commit，消息前缀 `week8: ...`；有不确定处先问，不猜；对学习单未指明的超参，在 README 记录所选值。
- 语料：`week8/data/TinyStoriesV2-GPT4-valid.txt`，按 `<|endoftext|>` 切故事、去首尾空白、丢空故事，文件序前 90% 训练（**24,867** 篇）/ 后 10% 验证（**2,763** 篇）。
- 分词器：`karpathy/tinyllamas` revision `0bd21da` 的 `stories260K/tok512.model`，sentencepiece 加载，V=512；两个 split 都编码，故事之间放 end-of-sequence token；token 流以 eos 开头（首故事首 token 以 eos 为上文，对应学习单 p(x₁|eos)）。
- 嵌入表：`minishlab/potion-base-8M` revision `bf8b056`（`model.safetensors` + `tokenizer.json`），不用 PyTorch 加载；对照表=行按 seed **1730** 洗牌。
- n-gram：n=1..6 由训练 token 计数；add-k，k ∈ {0.0001, 0.001, 0.01, 0.1, 1}，在**训练 token 的最后 10%** 上选；报告 nats/token、perplexity=e^L、bits/char（总 nats ÷ ln2 ÷ 验证集解码字符数）。
- Part 3 存储：d ∈ {16,32,64}，M = 8..65536（2 的幂），随机单位键/值，查询用存储键，R=干扰范数/信号范数，3 次独立抽取取中位数；容量=R 过 1 的 M（log-log 插值）。
- Part 3 MLP LM：64 维嵌入、单隐层 512 tanh、4000 步 AdamW × 16384 样本、峰值 lr 0.006（warmup + cosine decay；warmup 步数学习单未指定，取 400 并记录）。
- Part 4 图片：32×32 的「算」字位图（墨 +1/底 −1，硬编码，见 Task 8），d=64 随机单位键、其余记忆随机 ±1 图片；outer / ReLU / ReLU² / softmax(β=8) 四种存储，M ∈ {64, 256, 1024, 4096}。
- Part 5 transformer：pre-norm decoder，学习式位置向量，每线性层带 bias；2 层、宽 96、4 头、上下文 64；6000 步 AdamW × 32 条序列，峰值 lr 0.003，300 步线性 warmup，cosine 衰减到 1/10；训练窗口 65 token（读 64 预测 64）。AdamW weight decay 学习单未指定，取 0.01 并记录。
- Part 5 采样设定：temperature 0.8；top-k 20（无温度限定，按 T=1 处理并记录）；top-p 0.9 @ T=0.8；min-p 0.1 @ T=0.7；贪心检查用 T=0.0001。
- Part 6：CSV 来自 `epoch-research/analyzing-chinchilla` rev `92258837425e1b5f2851d624287f0120583a3d0e` 的 `data/svg_extracted_data.csv`；D=C/(6N)；丢 loss 最高的 **5** 条（245→240）；拟合 L=E+A/N^α+B/D^β 用 **log L 上的 Huber（δ=0.001）**，L-BFGS 网格起点取最优；**400 次 bootstrap** 报告各参数标准误；固定 C 求 N*、D*（式 27），C 从 1e18 到 1e25；C=5.76e23 处 D*/N* 及 95% bootstrap 区间。
- 学习单全部下载命令逐字执行并把 URL 记入 `week8/README.md`；慢链可加 `HF_ENDPOINT=https://hf-mirror.com`（week7 先例，记录用了哪个）。
- README 最后更新：每张 evidence 图旁给出产生它的命令；解释 tokenizer、召回图片、bpc 模型对比、scaling 拟合；记录语料下载地址；`data/`、`inputs/`、`artifacts/` 不进 git；push 源码与 evidence。

---

## File Structure

```
week8/
├── README.md                  # Task 0 建 stub，Task 15 收尾（evidence 表 + 解释 + provenance）
├── pyproject.toml, uv.lock, .python-version, .gitignore
├── conftest.py                # 空文件（pytest 将 week8/ 根加入 sys.path，week7 同款）
├── embeddings.py              # Part 1: 载入 potion 表与词表、单位行、洗牌对照、offset 余弦
├── stories.py                 # Part 2: 语料切分、tokenizer、token 流、解码校验
├── ngram.py                   # Part 2: 打包键计数、add-k 模型、评分、bigram 矩阵、祖先采样
├── stores.py                  # Part 3/4: 五种存储器、R 测量、容量扫描、预测线公式
├── mlp_lm.py                  # Part 3: JAX 单 token MLP 语言模型 + AdamW 训练循环
├── picture.py                 # Part 4: 「算」32×32 位图常量 + 图片召回实验
├── transformer.py             # Part 5: JAX pre-norm decoder、损失、位置损失、生成循环
├── samplers.py                # Part 5: temperature/top-k/top-p/min-p + 精确目标分布
├── scaling.py                 # Part 6: CSV 读取、Huber-L-BFGS 拟合、bootstrap、compute-optimal
├── scripts/
│   ├── part1_embeddings.py    # V1: 三关系箭头图 + 平均余弦表 (+20 次洗牌稳健性)
│   ├── part2_losses.py        # V1: bigram 长度 1/2/3 枚举归一化; V2: k 选择 + 损失表 + 解码校验
│   ├── part2_sampler.py       # V3: bigram 25 万 token 祖先采样 + 合并 χ²
│   ├── part2_curse.py         # V4: 未见 n-gram 比例 + bpc 对 n 作图
│   ├── part3_capacity.py      # V1: d=64 R–M log-log 图 + 预测线; V2: 容量–d 斜率
│   ├── part3_mlp.py           # V3: MLP LM 训练 + 与 bigram 损失对比（+ 损失历史图）
│   ├── part4_recall.py        # V1: 4×4 点图网格 + 每次召回的 R (+20 抽取中位数检验)
│   ├── part5_train.py         # 训练 transformer → artifacts/part5/transformer.npz + 损失历史
│   ├── part5_eval.py          # V1: bpc 对比图 + 同 seed 各模型 60-token 开头样本表
│   ├── part5_samplers.py      # V3: 各 sampler 10 万次 χ² + T=1e-4 贪心; V4: surprise z 值
│   ├── part5_context.py       # V2: 位置损失曲线（+可选注意力热图）
│   └── part6_scaling.py       # V1–V3: 四张图 + 参数/区间 JSON
├── tests/
│   ├── test_embeddings.py  test_stories.py  test_ngram.py  test_stores.py
│   ├── test_mlp_lm.py  test_picture.py  test_transformer.py  test_samplers.py
│   └── test_scaling.py
├── evidence/part1..part6/     # 进 git
├── artifacts/part1..part6/    # 不进 git（token 流缓存、权重、拟合结果）
├── data/                      # TinyStoriesV2-GPT4-valid.txt, chinchilla-runs.csv（不进 git）
└── inputs/                    # potion-base-8M/, tinyllamas/stories260K/tok512.model（不进 git）
```

依赖关系：`ngram.py` 依赖 `stories.py` 的常量（EOS、V）；`mlp_lm.py`、`transformer.py` 依赖 token 流缓存；`picture.py` 依赖 `stores.py`；`scaling.py` 独立。各模块互不反向依赖。

---

## Task 0: 环境、目录与全部下载

**Files:**
- Create: `week8/pyproject.toml`、`week8/.gitignore`、`week8/.python-version`、`week8/conftest.py`（空）、`week8/README.md`（stub）、`week8/data/`、`week8/inputs/`、`week8/evidence/part{1..6}/`、`week8/artifacts/part{1..6}/`

**Interfaces (Produces):**
- `cd week8 && uv run python <script>` 与 `uv run pytest tests/` 可用。
- `data/TinyStoriesV2-GPT4-valid.txt`（≈22.5 MB，27,630 篇）、`data/chinchilla-runs.csv`（245 行）、`inputs/potion-base-8M/{model.safetensors,tokenizer.json}`（表 29,528×256）、`inputs/tinyllamas/stories260K/tok512.model`。

- [x] **Step 1: 初始化 uv 项目**：
  ```bash
  cd week8
  uv init --no-readme --python 3.12
  uv add jax numpy matplotlib safetensors sentencepiece scipy pytest
  ```
  jax 版本与 week7 对齐（0.11.x，CPU）。`touch conftest.py`。
- [x] **Step 2: `.gitignore`**（内容：`data/`、`inputs/`、`artifacts/`、`.venv/`、`__pycache__/`、`.pytest_cache/`）+ `mkdir -p evidence/part{1..6} artifacts/part{1..6} data inputs`。
- [x] **Step 3: 学习单原文下载（语料与 CSV，URL 记入 README）**：
  ```bash
  mkdir -p week8/data
  base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve
  rev=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
  curl -L -o week8/data/TinyStoriesV2-GPT4-valid.txt "$base/$rev/TinyStoriesV2-GPT4-valid.txt"

  base=https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla
  rev=92258837425e1b5f2851d624287f0120583a3d0e
  curl -L -o week8/data/chinchilla-runs.csv "$base/$rev/data/svg_extracted_data.csv"
  ```
- [x] **Step 4: 模型文件（pinned revisions）**：
  ```bash
  uv run --with huggingface-hub==1.33.0 hf download minishlab/potion-base-8M \
    model.safetensors tokenizer.json --revision bf8b056 \
    --local-dir inputs/potion-base-8M
  uv run --with huggingface-hub==1.33.0 hf download karpathy/tinyllamas \
    stories260K/tok512.model --revision 0bd21da --local-dir inputs/tinyllamas
  ```
  慢链可加 `HF_ENDPOINT=https://hf-mirror.com`（记录）。校验：
  ```bash
  uv run python -c "
  from safetensors.numpy import load_file
  import json, sentencepiece as spm
  t = load_file('inputs/potion-base-8M/model.safetensors')
  k = [v.shape for v in t.values()]; print(k)          # 期望 [(29528, 256)]
  vocab = json.load(open('inputs/potion-base-8M/tokenizer.json'))['model']['vocab']
  print(vocab['cat'], vocab['kitten'])                  # 期望 3943 17407（图 3）
  sp = spm.SentencePieceProcessor(model_file='inputs/tinyllamas/stories260K/tok512.model')
  print(sp.piece_size(), sp.eos_id())                   # 期望 512 与 eos id
  print(open('data/chinchilla-runs.csv').readline())    # 记录列名，Task 13 用
  "
  ```
- [x] **Step 5: README stub**（标题、环境命令、provenance 小节）+ **Commit** `week8: uv environment, directories and pinned downloads`（确认 data/、inputs/ 未被跟踪）。

**预计用时：** 代理 15–25 min；下载 5–15 min（视网速）。

---

## Task 1 (Part 1): 嵌入表：关系即方向

**Files:**
- Create: `week8/embeddings.py`、`week8/scripts/part1_embeddings.py`
- Test: `week8/tests/test_embeddings.py`
- Evidence: `evidence/part1/relations.png`、`evidence/part1/cosines.txt`

**Interfaces (Produces):**
- `embeddings.RELATIONS: dict[str, list[tuple[str, str]]]` — 三关系各 9 对（逐字抄学习单：country→capital 9 对、male→female 9 对、adjective→comparative 9 对）。
- `embeddings.load_table(path) -> np.ndarray (29528, 256) float32`：safetensors.numpy 载入（无 PyTorch），行=token 向量。
- `embeddings.load_vocab(path) -> dict[str, int]`：解析 tokenizer.json 的 `model.vocab`。
- `embeddings.unit_rows(E) -> E_hat`：每行除自身范数。
- `embeddings.shuffled_control(E, seed=1730) -> E_shuf`：`np.random.default_rng(seed).permutation(V)` 重排行。
- `embeddings.leave_one_out_cosines(E_hat, pairs, vocab) -> np.ndarray (9,)`：对每对 j，offset_j = E_hat[b]−E_hat[a]，cos_j = ⟨offset_j, mean_{i≠j} offset_i⟩ / (|offset_j| |mean|)。

- [x] **Step 1: 失败测试** `tests/test_embeddings.py`：
  ```python
  import numpy as np
  from embeddings import unit_rows, shuffled_control, leave_one_out_cosines

  def test_unit_rows():
      E = np.random.default_rng(0).normal(size=(5, 8))
      U = unit_rows(E)
      assert np.allclose(np.linalg.norm(U, axis=1), 1.0)

  def test_control_is_permutation_and_reproducible():
      E = np.arange(20, dtype=float).reshape(4, 5)
      C1, C2 = shuffled_control(E, 1730), shuffled_control(E, 1730)
      assert np.array_equal(C1, C2)                       # seed 可复现
      assert sorted(C1.ravel()) == sorted(E.ravel())      # 同一组向量，换了行

  def test_loo_cosine_recovers_exact_direction():
      # 构造 9 对，offset 完全同向 → 每对 cos=1
      d = 16; u = np.zeros(d); u[0] = 1.0
      E = np.zeros((18, d)); key = {f"w{i}": i for i in range(18)}
      pairs = []
  ```
  （test 续：`E[2*i] = rng.normal(...)` 任意起点 a_i，`E[2*i+1] = E[2*i] + u`，则 `leave_one_out_cosines(unit_rows 的非必须——直接传已单位化的 E, pairs, key)` 全为 1。再加一个洗牌后 |mean cos| 小的对照断言。）
- [x] **Step 2: 跑 `uv run pytest tests/test_embeddings.py -v` 确认失败**（ModuleNotFoundError: embeddings）。
- [x] **Step 3: 实现 `embeddings.py`**（纯 NumPy；54 个词全部在词表中，载入时 assert）。
- [x] **Step 4: 跑测试通过。**
- [x] **Step 5: evidence 脚本** `scripts/part1_embeddings.py`：
  - 表与对照（seed 1730）各自算三关系的 9 个 cos；每关系一个子图：上半天蓝箭头=表（与水平轴夹角 arccos(cos)），下半灰箭头=对照，黑色水平箭头=平均 offset 方向，90° 虚线；箭头尖端横坐标=cosine。
  - `cosines.txt`：每关系的 mean cosine（表/对照），另跑 **20 个不同 seed 的洗牌**打印 max |mean|（学习单参考 ≤0.09）。
  - 打印词表抽查：cat=3943、kitten=17407（图 3 一致性）。
- [x] **Step 6: 验收（学习单 V1 阈值）**：表的每关系 mean cos ≥ **0.3**（参考 0.47/0.55/0.43）；对照在 **±0.15** 内（参考 0.03/−0.01/0.01）。不达标 → 检查单位化顺序（先单位化行再作差）、词表大小写（potion 词表全小写）。
- [x] **Step 7: Commit** `week8: part1 embedding relations as directions, shuffled control`

**预计用时：** 代理 30–45 min。

---

## Task 2 (Part 2a): 语料切分与 token 流

**Files:**
- Create: `week8/stories.py`
- Test: `week8/tests/test_stories.py`
- Artifacts: `artifacts/part2/train_tokens.npy`、`val_tokens.npy`

**Interfaces (Produces):**
- `stories.load_stories(path) -> list[str]`：按 `<|endoftext|>` 切、strip、丢空。
- `stories.split_stories(stories) -> (train, val)`：文件序前 90%/后 10%。
- `stories.load_tokenizer(path) -> SentencePieceProcessor`（V=512 断言）。
- `stories.EOS_ID: int`、`stories.V = 512`。
- `stories.encode_stream(sp, stories) -> np.ndarray int32`：`[EOS] + s1 + [EOS] + s2 + [EOS] + ...`（流以 eos 开头，故事间以 eos 分隔；末篇后不补）。
- `stories.decode_story(sp, tokens) -> str`。
- `stories.cache_streams() -> (train_tokens, val_tokens)`：首次编码后存 npy 缓存到 artifacts/。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_split_on_endoftext_strips_and_drops_empty():
      text = "<|endoftext|>\n  A story.  \n<|endoftext||>\n\n<|endoftext|>B<|endoftext|>"
      # 注意 typo 场景：只有精确 "<|endoftext|>" 是分隔符
      assert load_stories_from_text(text) == ["A story.", "B"]

  def test_encode_stream_starts_with_eos_and_separates():
      # 用假 tokenizer（stub 两个 token）验证结构 [EOS, a, b, EOS, c]；EOS 出现次数 = 故事数
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: 编码两 split 并缓存**；打印：24,867 / 2,763 篇；训练 token 数 ≈ 9.7e6；验证解码匹配篇数（期望 **2,763−3** 篇逐字节还原；差异源于双空格合并，逐篇打印不匹配的 3 篇索引）；chars/token ≈ **2.05**（验证集解码字符数 ÷ token 数）。
- [x] **Step 6: Commit** `week8: part2 corpus split, tok512 encoding, decode check (all but 3 stories)`

**预计用时：** 代理 30–45 min。

---

## Task 3 (Part 2b): n-gram 计数模型

**Files:**
- Create: `week8/ngram.py`
- Test: `week8/tests/test_ngram.py`

**Interfaces (Produces):**
- `ngram.pack_context(tokens: int64 (N,), n) -> int64 (N,)`：把前 n−1 个 token（不足则左侧补 EOS）编码成 base-512 整数（n−1 ≤ 5 → 45 bit，安全）。
- `ngram.build_counts(tokens, n) -> CountTable`：`keys = ctx*512+next`，`np.unique(..., return_counts)` 排序存储 + 上下文总数表（`np.unique(ctx)`）。
- `CountTable.log_prob_stream(tokens, k) -> float64 (N,)`：逐 token 的 ln p(x_t|x_{t−n+1..t−1})；上下文未见 → ln(1/512)；否则 ln((c+k)/(total+kV))。向量化：`np.searchsorted`。
- `CountTable.bigram_matrix(k) -> (512, 512)`：仅 n=2，行归一转移矩阵（行=eos 语境用 pack 一致）。
- `ngram.choose_k(train_tokens, n, grid=(1e-4,1e-3,1e-2,1e-1,1))`：在**训练 token 最后 10%** 上选 k（计数用全训练流，记录选择口径）。
- `ngram.seq_prob_sums(P, eos) -> (s1, s2, s3)`：显式枚举长度 1/2/3 的全 token 序列概率和（`np.einsum('i,ij,jk->', a, P, P)` 形式的向量化枚举，T=3 时 512³ 项全部列出）。
- `ngram.sample_stream(model_row_fn, n, rng, n_tokens) -> tokens`：祖先采样（上下文不足补 EOS）。
- `ngram.chi2_pooled(observed: dict[ctx, Counter], expected: dict[ctx, probs], n_draws_per_ctx) -> (stat, dof, p)`：稀有 token 并桶至期望计数 ≥5，统计量与自由度跨 20 个上下文相加，`scipy.stats.chi2.sf`。

- [x] **Step 1: 失败测试**（要点，全部小合成例）：
  ```python
  def test_packing_roundtrip_and_width():
      toks = np.array([3, 511, 7, 0], dtype=np.int64)
      ctx = pack_context(toks, 3)          # 左补 EOS
      # 手算一例断言 + 不同 n 不冲突

  def test_addk_formula_and_unseen_context():
      # 手工 8 token 流、V=4、n=2：c(a→b)=2, total(a)=3 → p=(2+k)/(3+4k)
      # 未见上下文 → 1/4

  def test_rows_of_bigram_matrix_sum_to_one():
      P = ...; assert np.allclose(P.sum(1), 1.0)

  def test_seq_prob_sums_are_one_for_valid_chain():
      # 用随机行归一矩阵，|s_T − 1| < 1e-12（学习单 V1 的缩小版）

  def test_choose_k_prefers_smooth_zero_infinite_loss():
      # 构造含未见 bigram 的"验证"流：k=0 时损失 inf → choose_k 不返回 0
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现（float64 计数；int64 打包键）→ Step 4: 通过。**
- [x] **Step 5: Commit** `week8: part2 n-gram counts, add-k smoothing, exact normalization helpers`

**预计用时：** 代理 45–60 min。

---

## Task 4 (Part 2c): Part 2 evidence（V1 归一化、V2 损失表、V3 采样 χ²、V4 维度灾难）

**Files:**
- Create: `week8/scripts/part2_losses.py`、`week8/scripts/part2_sampler.py`、`week8/scripts/part2_curse.py`
- Evidence: `evidence/part2/normalization.txt`、`evidence/part2/losses_table.txt`、`evidence/part2/sampler_chi2.txt`、`evidence/part2/curse.png`
- Artifacts: `artifacts/part2/ngram_losses.json`（后续 Part 5 对比用）

- [x] **Step 1: `part2_losses.py`**：
  - V1：对选好 k 的 bigram 矩阵算 `seq_prob_sums`；打印 |sum−1|（学习单阈值 **1e-12**，参考 ≤ 3.2e-15）；对照组：不做行归一的得分（如 `exp(n(ab)/max n)`）打印 512 / 2.6e5 / 1.3e8 以对照。
  - V2：uniform、unigram..6-gram 七个模型；每 n 先 `choose_k`（打印所选 k），再在验证流上评分；输出表：nats/token、perplexity=e^L、bits/char（总 nats ÷ ln2 ÷ 验证解码字符数）；损失历史存 JSON。
  - 验收：uniform 行精确 **6.238 / 512 / 4.386**；unigram **4.839 / 126.4 / 3.403**；bigram **3.038 / 20.9 / 2.136**；trigram **2.094 / 8.12 / 1.473**；各值差 **≤0.03 nats**、perplexity差 **≤3%**。不达标先查：k 选择口径、流首 EOS、chars 口径（解码字符）。
- [x] **Step 2: `part2_sampler.py`（V3）**：bigram 祖先采样 **250,000** token；取训练流中 **20 个最频繁 token** 为上下文，比较样本中各"下一 token 计数"与模型概率（期望=该上下文在样本中作为前一 token 的次数 × P(·|a)），合并 χ²；打印 p 值（阈值 **> 0.001**，参考 0.37）。附一个反例 sampler（条件于往前第 2 个 token）打印 p ≈ 0，作对照。
- [x] **Step 3: `part2_curse.py`（V4）**：n=1..6 的验证流未见 n-gram 比例（held-out n-gram 从未在训练 token 中出现的比例）与 bpc 双图（参考图 7 布局）。
  - 验收：bpc 在 **n=4 最低 ≈1.310**；n=5 ≈ **1.46**、n=6 ≈ **1.79**（各 ±0.03）；未见比例 n=3..6 距 **0.6% / 4.2% / 12.2% / 23.5%** 在 **2 个百分点** 内；打印 6-gram perplexity（参考 12.8）> trigram 8.12。
- [x] **Step 4: Commit**（可分三条：`week8: part2 bigram exact normalization + loss ladder (…)` / `week8: part2 ancestral sampling passes pooled chi-square` / `week8: part2 curse of dimensionality, 4-gram best at 1.310 bpc`）。

**预计用时：** 代理 45–60 min；计算几分钟。

---

## Task 5 (Part 3a): 键–值存储器与干扰–信号比

**Files:**
- Create: `week8/stores.py`
- Test: `week8/tests/test_stores.py`

**Interfaces (Produces):**
- `stores.random_unit_rows(rng, M, d) -> (M, d)`。
- `stores.outer_store(K, V) -> W (d, d)`：`W = V.T @ K`；查询 `f(x) = W @ x`。
- `stores.lstsq_store(K, V) -> W`：`np.linalg.lstsq(K, V, rcond=None)` 解 min Σ‖Wk_i−v_i‖²。
- `stores.mean_R(K, V, sigma, rng=None, n_queries=None) -> float`：σ 取 `'outer' | ('relu', p) | ('softmax', beta)`；对每个查询键 j：信号=w_j·v_j（w_j=σ(k_j·k_j)，ReLU^p 与 softmax 归一后均按同除处理，比值不变）、干扰=Σ_{i≠j}w_i v_i；R_j=‖干扰‖/‖信号‖；返回查询上的均值。M ≤ 4096 查全部键，更大 M 查随机 **2048** 个键（学习单"query every store with stored keys"，对大 M 抽样子集，记录于 README）；softmax 权重减最大值防溢出。ReLU^p 用 float32 分块矩阵乘（`S = A[batch] @ A.T`）。
- `stores.capacity(Ms, Rs) -> float | None`：R 过 1 的 M，log R–log M 线性插值。
- 预测线：`stores.R_PRED = {'outer': sqrt(M/d), 'lstsq': max(0, sqrt(M/d)−1)（M>d）, ('relu',p): sqrt((2p−1)!!·M/(2d^p))}`。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_orthogonal_keys_outer_store_recalls_exactly():
      K = np.eye(6)[:, :4]                # 4 个正交单位键, d=6 → M=d
      V = random unit rows (4, 6)
      R = mean_R(K, V, 'outer'); assert R < 1e-10

  def test_lstsq_exact_up_to_M_equals_d():
      rng…; K (6,6) 单位行; assert mean_R(K, V, 'outer'→lstsq 通道) < 1e-10

  def test_relu_weights_zero_for_negative_similarity():
      # 手工 2 记忆：k1·k2 < 0 → ReLU^p 干扰为 0，R=0

  def test_softmax_weights_sum_to_one_and_R_matches_unnormalized():
      # softmax(β=8) 归一前后 R 相等（比例不变）

  def test_capacity_interpolation():
      Ms = [8, 32, 128]; Rs = [0.2, 0.5, 1.4] → 容量在 (32,128) 内、log-log 插值可复现
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: Commit** `week8: part3 key-value stores (outer, lstsq, relu^p, softmax) with interference ratio`

**预计用时：** 代理 45–60 min。

---

## Task 6 (Part 3b): 容量实验 evidence（V1 + V2）

**Files:**
- Create: `week8/scripts/part3_capacity.py`
- Evidence: `evidence/part3/interference_d64.png`、`evidence/part3/capacity.png`、`evidence/part3/capacity_table.txt`
- Artifacts: `artifacts/part3/capacity.json`

- [x] **Step 1: 脚本**：对 d ∈ {16, 32, 64}、M = 8..65536（2 的幂）× {outer, lstsq, relu1, relu2, relu3}，每配置 3 次独立抽取取中位数；R>1.5 后可提前停（容量已括出）；d=64 图：R–M log-log + 预测虚线 + R=1 水平线；容量–d log-log + 最小二乘斜率。float32；大 M 分块。
- [x] **Step 2: 验收（学习单 V1/V2）**：
  - 曲线在 R ≳ 0.1 后斜率 ≈ 1/2；
  - outer 在 d=64 于 **M ∈ [50, 85]** 过 1（参考 67.4）；
  - lstsq 在 M ≤ 64 时 **R < 1e-10**，在 **[100, 170]** 过 1（参考 129，参考 R=5e-14）；
  - ReLU²/ReLU³ 过 1 于 **≈2,871 / ≈38,124**（高斯预测 2,731 / 34,953）；
  - 斜率：outer/lstsq/relu1 距 1 **≤0.1**（参考 0.96/0.99/0.95）；relu2 距 2 **≤0.15**（参考 1.96）；relu3 **>2.5**（参考 2.79）。
- [x] **Step 3: Commit** `week8: part3 capacity ~d, ~2d, ~d^p with slopes 0.96/0.99/0.95/1.96/2.79`

**预计用时：** 代理 30–45 min；计算：float32 BLAS 数分钟（ReLU³ d=64 最重；若超 10 min，把大 M 查询键降到 1024）。

---

## Task 7 (Part 3c): MLP 语言模型存下 bigram 表（V3）

**Files:**
- Create: `week8/mlp_lm.py`、`week8/scripts/part3_mlp.py`
- Test: `week8/tests/test_mlp_lm.py`
- Evidence: `evidence/part3/mlp_loss.txt`、`evidence/part3/mlp_loss_history.png`
- Artifacts: `artifacts/part3/mlp.npz`

**Interfaces (Produces):**
- `mlp_lm.init_params(rng, V=512, d_emb=64, d_hidden=512) -> dict`（E、W1(512,64)、b1、W2(512,512)、b2；参数总数 328,704）。
- `mlp_lm.forward(params, prev_tokens) -> logits (B, V)`：h1=tanh(W1·E[prev]+b1)，z=W2·h1+b2。
- `mlp_lm.train(train_tokens, val_tokens, steps=4000, batch=16384, peak_lr=0.006, warmup=400) -> (params, history)`：AdamW（weight_decay=0.01，记录），每步从训练 bigram 位置均匀抽 batch；lr 线性 warmup 后 cosine 衰减到 0。
- `mlp_lm.heldout_loss(params, val_tokens) -> float`：与 Part 2 bigram 完全相同的 (prev,next) 位置集合上的平均 CE（nats/token）。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_forward_shapes_and_param_count():
      p = init_params(jax.random.PRNGKey(0))
      assert sum(x.size for x in jax.tree_util.tree_leaves(p)) == 328704
      assert forward(p, jnp.zeros((3,), jnp.int32)).shape == (3, 512)

  def test_loss_gradients_finite_and_loss_matches_direct_ce():
      # 小合成 batch：train_step 可执行、grad 有限；heldout_loss == -mean log softmax[label]

  def test_training_pushes_toy_distribution_to_counts():
      # V=4、手造 bigram 计数 → 训练后 p(·|a) 逼近行归一计数（tolerance 1e-2）
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现（JAX，jit 训练步）→ Step 4: 通过。**
- [x] **Step 5: `scripts/part3_mlp.py`**：训练（4000×16384，CPU 约一两分钟），保存权重与损失历史，报告 held-out CE。
- [x] **Step 6: 验收（V3）**：|MLP held-out − 3.038| ≤ **0.01** nats/token（参考 +0.003）。附 512 样本/步的对照跑（参考：差 0.02 nats，可只在 README 记一句，脚本加 `--batch 512` 开关）。
- [x] **Step 7: Commit** `week8: part3 one-token MLP stores the bigram table (held-out 3.041 vs 3.038)`

**预计用时：** 代理 45–60 min。

---

## Task 8 (Part 4): softmax 存储器召回「算」字图片（V1）

**Files:**
- Create: `week8/picture.py`、`week8/scripts/part4_recall.py`
- Test: `week8/tests/test_picture.py`
- Evidence: `evidence/part4/recall_grid.png`、`evidence/part4/ratios.txt`

**Interfaces (Produces):**
- `picture.SUAN_BITMAP: list[int]`（32 个 32-bit 整数，1=墨；由学习单图 13 提取，249 个墨点；「⺮+目+廾」= 算）：
  ```python
  SUAN_BITMAP = [
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00300600, 0x00700e00, 0x007feffc, 0x00ffdff8,
      0x018e3880, 0x030e70c0, 0x03003000, 0x007fffc0,
      0x007000c0, 0x007fffc0, 0x007fffc0, 0x007000c0,
      0x007fffc0, 0x007000c0, 0x007000c0, 0x007fffc0,
      0x000e0e00, 0x000e0e00, 0x07fffffc, 0x000e0e00,
      0x000e0e00, 0x001c0e00, 0x00780e00, 0x01f00e00,
      0x00800000, 0x00000000, 0x00000000, 0x00000000,
  ]
  ```
- `picture.target_picture() -> (1024,) float32`：±1，+1=墨。
- `picture.recall(target_value, M, d, store, rng) -> (recalled (1024,), R)`：键全随机单位（含目标的键），其余值随机 ±1；用其键查询；softmax 权重减最大值；返回**除以自身权重 w_j 后**的召回与 R。
- 预测：`R_softmax = sqrt(M)·exp(β²/d−β)`（d=64, β=8, M=4096 → 0.058）、`sqrt(M/d)`、`sqrt(M/(2d))`、`sqrt(3M/(2d²))`。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_bitmap_shape_values_and_ink_count():
      pic = target_picture()
      assert pic.shape == (1024,) and set(np.unique(pic)) <= {-1.0, 1.0}
      assert int((pic == 1).sum()) == 249

  def test_softmax_store_recalls_lone_picture_exactly():
      # M=1（无干扰）：召回≈图片本身（除以自身权重后 R≈0）

  def test_outer_recall_matches_prediction_at_zero_noise():
      # M=64, d=64：R ≈ sqrt(M/d)=1（tolerance 宽）
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: `scripts/part4_recall.py`**：M ∈ {64,256,1024,4096} × {outer, relu, relu2, softmax(β=8)}，固定 seed 单次抽取画 4×4 点图（点面积∝召回像素/自身权重，R<1 标红），打印全部 R；另对 M=1024 跑 **20 次抽取中位数**（outer/ReLU 检验用）。
- [x] **Step 6: 验收（V1）**：M=1024 的 outer/ReLU R 距 **4 / 2.83** ≤15%（参考中位数 4.05/2.84）；ReLU² 在 M=1024 处 **R≈0.60**（预测 0.61）且图片可读；M=4096 时 ReLU² **R>1**（参考 1.2）；softmax 每个 M 都 **R<0.1**（M=4096 参考 0.057，预测 0.058）且四行图片同样清晰。
- [x] **Step 7: Commit** `week8: part4 softmax store (beta=8) recalls the picture among 4096 memories`

**预计用时：** 代理 45–60 min。

---

## Task 9 (Part 5a): pre-norm transformer decoder（JAX）

**Files:**
- Create: `week8/transformer.py`
- Test: `week8/tests/test_transformer.py`

**Interfaces (Produces):**
- `transformer.init_params(rng, V=512, d=96, L=2, h=4, T=64) -> dict`：E(512,96)、P(64,96)；每块 LN1(g,b)、WQ/WK/WV/WO(96,96)+b、LN2(g,b)、W1(384,96)+b1、W2(96,384)+b2；final LN；readout(512,96)+b。参数总数 **328,448**。
- `transformer.forward(params, tokens (B, T)) -> logits (B, T, V)`：x = E[tok]+P[pos]；每块 `x += Attn(LN1(x))`、`x += MLP(LN2(x))`（MLP: GELU(W1·+b1)→W2·+b2）；多头：每头宽 d/h=24，scores = QKᵀ/√24 + causal mask(−1e30 上三角)，softmax 行归一；`MultiHead = [head1..head4]WO+bO`；final LN → readout。
- `transformer.loss_fn(params, batch65) -> float`：输入 (B,65)，x=batch[:,:64] 预测 batch[:,1:65]，逐位 CE 平均。
- `transformer.eval_positions(params, batch65) -> (B, 64) per-token nats`：位置损失（V2 用，第 1 位不评分 → 返回 63 个值或首位置置 NaN，接口注明）。
- `transformer.next_token_logits(params, tokens (1, t)) -> (V,)`：生成用（t ≤ 64）。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_param_count_matches_figure14_formula():
      p = init_params(PRNGKey(0))
      # E+P + 2*(4d²+4d + 8d²+5d + 4d) + 2d + dV+V = 328,448
      assert tree_size(p) == 328448

  def test_causality_future_tokens_cannot_change_past_logits():
      p = init_params(PRNGKey(1))
      a = forward(p, jnp.array([[1, 2, 3, 4, 5]]))
      b = forward(p, jnp.array([[1, 2, 3, 9, 9]]))
      assert np.allclose(a[0, :3], b[0, :3])

  def test_attention_rows_are_causal_and_normalized():
      # 导出单头权重检查：行 t 只在 j≤t 非零、行和为 1（forward 提供可选返回注意力的调试接口）

  def test_loss_zero_when_logits_one_hot_on_target():
      # 手工参数构造或对 loss_fn 用 logits 注入 stub
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: Commit** `week8: part5 pre-norm decoder, causal multi-head attention, 328,448 params`

**预计用时：** 代理 60–90 min。

---

## Task 10 (Part 5b): 训练与 V1（bpc 对比 + 各模型同 seed 样本）

**Files:**
- Create: `week8/scripts/part5_train.py`、`week8/scripts/part5_eval.py`
- Evidence: `evidence/part5/bpc_compare.png`、`evidence/part5/samples.txt`、`evidence/part5/loss_history.png`
- Artifacts: `artifacts/part5/transformer.npz`、`artifacts/part5/train_log.json`

- [x] **Step 1: `part5_train.py`**：训练窗口 65 连续 token，每步随机偏移抽 32 条；AdamW(wd=0.01，记录)、6000 步、峰值 lr 0.003、300 步线性 warmup、cosine 衰减至 1/10；保存权重 + 损失历史 + 每 500 步验证 bpc 曲线。CPU 预计 ≈2–5 分钟（参考机器 88 s）。
- [x] **Step 2: `part5_eval.py`（V1）**：
  - held-out bpc：验证 token 流切成"64 上下文 + 64 评分"的窗口（stride 64），总 nats ÷ ln2 ÷ 验证解码字符数（与 Part 2 同 chars 口径）。
  - 图：uniform/unigram/bigram/trigram/transformer 五柱 bpc 图（参考图 16）。
  - 样本表：五模型各以 **seed 1848**、T=1 采 60 token 故事开头（统一从 EOS 起采；解码时排除 eos，"␣"→可视需要保留空格，学习单表 1 风格），按 bpc 升序排列。
- [x] **Step 3: 验收（V1）**：transformer bpc **< 1.310**（4-gram），参考 **1.207**。未达标：查因果 mask、warmup/衰减、lr 峰值、bpc 口径。
- [x] **Step 4: Commit** `week8: part5 transformer trained, 1.2x bpc beats every count model`

**预计用时：** 代理 45–60 min；训练几分钟。

---

## Task 11 (Part 5c): samplers 与 V3（χ²）/ V4（surprise 自洽）

**Files:**
- Create: `week8/samplers.py`、`week8/scripts/part5_samplers.py`
- Test: `week8/tests/test_samplers.py`
- Evidence: `evidence/part5/sampler_chi2.txt`、`evidence/part5/surprise.txt`

**Interfaces (Produces):**
- `samplers.probs(logits, kind, T=1.0, k=20, p=0.9, min_p=0.1) -> (V,)`：精确目标分布。temperature：softmax(logits/T)；top-k：保留前 k 个、其余置 0 后归一（T 可选作用于 logits）；top-p：按概率降序累积，**保留越过阈值的那一个**（最小集合 cumsum ≥ p）后归一；min-p：保留 p_v ≥ min_p·max(p)；greedy：T=1e-4 等价 argmax。
- `samplers.draw(rng, probs) -> int`。
- `samplers.binned_chi2(counts (V,), probs (V,)) -> (stat, dof)`：稀有并桶至期望 ≥5。

- [x] **Step 1: 失败测试**（要点，全部手工小例）：
  ```python
  def test_temperature_matches_softmax_formula():           # 式 25
  def test_topk_keeps_k_and_renormalizes():
  def test_topp_keeps_token_crossing_threshold():
      # logits 使概率 [0.5, 0.3, 0.15, 0.05]，p=0.8 → 保留前三（0.5+0.3=0.8<0.8? 注意：
      # 0.5+0.3 = 0.8 恰达阈值 → 保留 {1,2}；改 p=0.85 → 保留 {1,2,3}；
      # 学习单：丢掉越界 token 的错误实现 χ² p≈0，测试锁定正确语义）
  def test_minp_keeps_fraction_of_max():
  def test_greedy_low_temperature_returns_argmax_everywhere():
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: `part5_samplers.py`**：
  - V3：从验证集选一个 next-token 分布足够宽（熵高）的固定上下文；四 sampler（T=0.8 / top-k20 / top-p0.9@0.8 / min-p0.1@0.7）各采 **100,000** 次；与各自精确分布做分桶 χ²，打印 p 值（全部 **> 0.001**，参考 0.12–0.98）；T=0.0001 采 1000 次每次都是 argmax。
  - V4：T=1 从 eos 生成 **128 条 × 128 token**；z = (mean surprise − mean predicted entropy) / SE(按序列)；surprise=−ln p_{T=1}(v)。再以 T=0.7 生成同样规模、用 T=1 模型打分。打印两个 z。
- [x] **Step 6: 验收（V3/V4）**：T=1 时 **|z| < 4**（参考 1.6）；T=0.7 时 **|z| > 40**（参考约 −68@128 序列）。
- [x] **Step 7: Commit** `week8: part5 samplers match exact targets (chi2 p>0.001), T=1 samples as surprising as predicted`

**预计用时：** 代理 60–75 min；计算几分钟（生成可 jit 批量化）。

---

## Task 12 (Part 5d): V2 位置损失（+ 可选注意力热图扩展）

**Files:**
- Create: `week8/scripts/part5_context.py`
- Evidence: `evidence/part5/context_loss.png`（可选：`evidence/part5/attention_heads.png`）

- [x] **Step 1: `part5_context.py`**：验证 token 流切**连续 64-token 窗口**，每窗从首 token 起评分（位置 1 无上文不评分；位置 t 的损失=t−1 个 token 已读）；对每位置取多窗平均，画 transformer 位置损失 vs 已读 token 数（±2 标准误带），bigram(3.038)/trigram(2.094) 全流平均损失画水平虚线。
- [x] **Step 2: 验收（V2）**：读 **≥32** token 后损失 **< 2.09**（trigram；参考最后 8 位平均 1.65）；读 1 token 时**差于** bigram；**3 token** 后超过 trigram 并持续下降至 64。
- [x] **Step 3（可选扩展，README 注明为扩展）**：从验证故事开头取 16–24 token，画 8 个头的注意力热图（行=预测位置，列=已读位置）；观察一层头偏局部、二层头回读更远（如 "park" 的 "k" 读 "to"）。
- [x] **Step 4: Commit** `week8: part5 loss falls with tokens read; passes trigram after 3 tokens`

**预计用时：** 代理 30–45 min。

---

## Task 13 (Part 6a): Chinchilla 缩放律拟合

**Files:**
- Create: `week8/scaling.py`
- Test: `week8/tests/test_scaling.py`
- Artifacts: `artifacts/part6/fit.json`

**Interfaces (Produces):**
- `scaling.PRINTED = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)`（论文印刷参数）。
- `scaling.load_runs(csv_path) -> (N, D, L)`：解析列（Task 0 已记录列名；按含 param/flop|compute/loss 的列名取）；D=C/(6N)；丢 loss 最高 **5** 行 → 240 条。
- `scaling.huber_log_fit(N, D, L, grid, delta=1e-3) -> params`：残差 r=ln L_pred−ln L_meas，Huber(r,δ) 之和；`scipy.optimize.minimize(method='L-BFGS-B')`，bounds E∈[1,3]、A,B∈[1,1e5]、α,β∈[0.1,0.8]；网格起点 E∈{1.6,1.8,2.0}×A∈{100,400,1600}×B∈{100,400,1600}×α∈{0.30,0.40}×β∈{0.25,0.35,0.45}（162 起）取最优。
- `scaling.bootstrap(fit_fn, N, D, L, n=400, start=params) -> params_sample (400, 5)`：有放回抽 240 条重拟合。
- `scaling.compute_optimal(params, C) -> (N_star, D_star)`：N\*=G(C/6)^{β/(α+β)}，D\*=(C/6)^{α/(α+β)}/G，G=(αA/βB)^{1/(α+β)}（式 27）。

- [x] **Step 1: 失败测试**（要点）：
  ```python
  def test_compute_optimal_matches_numeric_minimisation():
      # 玩具参数 α=β=0.5, A=B=2, E=1：L(N)=1+2/√N+2N^0.5/c^0.5，c=C/6
      # 数值最小化沿 ND=c 与式 27 一致（rtol 1e-6）；α=β、A=B 时 N*=D*=√c

  def test_compute_optimal_matches_reference_numbers():
      # 用学习单参考参数（α=0.347, β=0.367, A=478, B=2143）在 C=5.76e23：
      # D*/N* ≈ 17.9；印刷参数 → ≈ 93
      # （对拟合参数不做硬断言，只锁定公式实现）

  def test_huber_loss_values():
      # |r|≤δ: r²/2；|r|>δ: δ(|r|−δ/2)

  def test_drop_five_highest_loss():
      # 合成 10 行 → 保留 loss 最低 5+... 即去掉最高的 5 条
  ```
- [x] **Step 2: 确认失败 → Step 3: 实现 → Step 4: 通过。**
- [x] **Step 5: Commit** `week8: part6 chinchilla fit: huber log-loss, lbfgs grid, bootstrap, compute-optimal`

**预计用时：** 代理 45–60 min；bootstrap 400 次 L-BFGS（5 参数、240 行，单起点）约 1–2 分钟。

---

## Task 14 (Part 6b): Part 6 evidence（V1–V3）

**Files:**
- Create: `week8/scripts/part6_scaling.py`
- Evidence: `evidence/part6/fit_law.png`、`evidence/part6/residuals.png`、`evidence/part6/optimal_split.png`、`evidence/part6/params.txt`
- Artifacts: `artifacts/part6/fit.json`（含 bootstrap 样本与区间）

- [x] **Step 1: 脚本与四图**：
  - V1 左：L 对 N 的散点 + 三个预算（1e19/1e20/1e21 FLOP，±15% 内的 run 着色）处拟合律曲线 L(N)=E+A/N^α+B(6N/C)^β；右：measured vs fitted loss（对角线）。
  - V2：两个参数组（自己拟合 vs PRINTED）的 log-loss 残差（预测−实测）对 N 散点 + 直方图；各自报告平均残差。
  - V3：N\*、D\* 对 C（1e18–1e25，log-log）双线（自己拟合实线、印刷参数虚线），Chinchilla 星标（70e9 参数、1.4e12 token）；C=5.76e23 的 D\*/N\* 与 95% bootstrap 区间。
  - `params.txt`：α、β、E、A、B ± bootstrap SE；两斜率（N\*~C^x、D\*~C^y）。
- [x] **Step 2: 验收（学习单 V1/V2/V3 阈值）**：
  - **0.32 ≤ α ≤ 0.38**、**0.32 ≤ β ≤ 0.42**、**1.75 ≤ E ≤ 1.90**（参考 α=0.347±0.014、β=0.367±0.021、E=1.817±0.025、A=478、B=2143）；
  - 自己拟合平均残差 ∈ **±0.005**（参考 −0.0007）；印刷参数 **> 0.01**（参考 +0.017）；
  - C=5.76e23 处 **10 ≤ D\*/N\* ≤ 30**（参考 17.9，区间 7.7–37 含 Chinchilla 的 20；印刷参数给 ≈93）；N\*~C^{≈0.51}、D\*~C^{≈0.49}。
  - 不达标：查 Huber 的 log 口径、丢 5 条、C/(6N)。
- [x] **Step 3: Commit** `week8: part6 refit recovers alpha 0.35 beta 0.37, printed params miss runs, ~18 tokens/param`

**预计用时：** 代理 30–45 min。

---

## Task 15: README 收尾、根 README、push

**Files:**
- Modify: `week8/README.md`、根 `README.md`

- [x] **Step 1: `week8/README.md`** 按学习单"Save the results and prepare to show your work"：
  - evidence 图 ↔ 产生命令表（week7 风格）；
  - 解释段：tokenizer（BPE 512、~2.05 字符/token、解码除 3 篇）、召回图片（R<1 才可读、softmax 容量 ~e^{2β}）、bpc 模型对比（每加一层上下文 perplexity 除多少）、scaling 拟合（重拟合 vs 印刷参数、20 token/参数）；
  - provenance：全部下载 URL + revision + 是否用了 hf-mirror；
  - 记录学习单未指定而自选的超参：AdamW weight decay 0.01、Part 3 warmup 400、top-k 的 T=1、大 M 存储 R 的 2048 键子采样。
- [x] **Step 2: 根 README** 的仓库布局小节加 `week8/`。
- [x] **Step 3: 终检**：`cd week8 && uv run pytest tests/ -q` 全绿；`git status` 确认无 data/inputs/artifacts 泄漏；evidence 尺寸合理（<5 MB/图）。
- [x] **Step 4: Commit + push** `week8: readme evidence tables, explanations, provenance`（如推送失败/需凭据，询问用户）。

**预计用时：** 代理 30 min。

---

## 可选扩展（学习单 Extension，时间富余再做）

- Part 3 扩展：把 512 个 tanh 单元读作记忆（每侧 5 个最强前/后 token、平方激活份额中位数 ≈0.12 vs 未训练 0.135），脚本 `scripts/part3_units.py` + `evidence/part3/units.png`。
- Part 5 扩展已并入 Task 12 Step 3。

## Self-Review（对照学习单逐项核查）

- Part 1 V1 ✓ Task 1；Part 2 V1–V4 ✓ Task 3/4；Part 3 V1/V2 ✓ Task 5/6、V3 ✓ Task 7；Part 4 V1 ✓ Task 8；Part 5 V1 ✓ Task 10、V2 ✓ Task 12、V3/V4 ✓ Task 11；Part 6 V1–V3 ✓ Task 13/14；收尾 README/push ✓ Task 15。
- 学习单全部 pinned revision 与数字阈值均已逐字抄入 Global Constraints 与各 Task 验收；无占位符。
- 接口一致性：`stores.mean_R` 被 Task 6/8 共用；`samplers.probs` 被 Task 11 两处共用；`stories.encode_stream` 的 EOS 约定被 ngram/mlp_lm/transformer 三处共用；`scaling.compute_optimal` 被 V3 图与区间共用。
- 两处学习单歧义已显式记录决策（top-k 温度、k 选择口径），并在 README 中注明——若参考值偏差超容差，先按验收栏的排查清单自查，仍不符则询问用户。
