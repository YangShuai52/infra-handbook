# Xing4.0-29B-A4B 模型适配分析：MLA 计算流程、310P Gap 与 mHC 架构

> 以 Xing4.0-29B-A4B 模型为例，梳理 MLA (Multi-head Latent Attention) 的完整计算流程、mHC (multi-step HyperConnection) 架构原理、Ascend 310P 在 vLLM-Ascend 上适配的代码层面 Gap，以及模型适配 PR 的合入状态。

---

## 目录

- [〇、模型适配 PR 合入状态](#〇模型适配-pr-合入状态)
- [一、MLA 计算流程](#一mla-计算流程)
  - [1.1 模型参数](#11-模型参数)
  - [1.2 核心思想：低秩压缩](#12-核心思想低秩压缩)
  - [1.3 为什么需要分离 RoPE 和 NoPE](#13-为什么需要分离-rope-和-nope)
  - [1.4 计算流程总览](#14-计算流程总览)
  - [1.5 KV Cache 存储：压缩存储](#15-kv-cache-存储压缩存储)
  - [1.6 Prefill 路径（解压后全维度 attention）](#16-prefill-路径解压后全维度-attention)
  - [1.7 Decode 路径（潜空间 attention，不解压 K/V）](#17-decode-路径潜空间-attention不解压-kv)
  - [1.8 Prefill vs Decode 计算量对比](#18-prefill-vs-decode-计算量对比)
- [二、910B MLA 实现剖析](#二910b-mla-实现剖析)
  - [2.1 核心架构](#21-核心架构)
  - [2.2 关键设计：KV Cache 存储压缩](#22-关键设计kv-cache-存储压缩)
  - [2.3 Decode 路径：潜空间注意力](#23-decode-路径潜空间注意力)
  - [2.4 Prefill 路径：解压后全维度 attention](#24-prefill-路径解压后全维度-attention)
  - [2.5 融合算子](#25-融合算子)
- [三、310P 在 vLLM-Ascend 适配 MLA 的 Gap](#三310p-在-vllm-ascend-适配-mla-的-gap)
  - [3.1 Gap 总览](#31-gap-总览)
  - [3.2 各 Gap 代码层面详细分析](#32-各-gap-代码层面详细分析)
  - [3.3 310P MLA 适配方案](#33-310p-mla-适配方案)
  - [3.4 修改清单汇总](#34-修改清单汇总)
- [四、mHC 架构原理与 310P 适配 Gap](#四mhc-架构原理与-310p-适配-gap)
  - [4.1 mHC 核心概念](#41-mhc-核心概念)
  - [4.2 mHC 计算流程](#42-mhc-计算流程)
  - [4.3 910B 上的 mHC 实现](#43-910b-上的-mhc-实现)
  - [4.4 310P 适配 mHC 的 Gap](#44-310p-适配-mhc-的-gap)
- [五、根因总结](#五根因总结)

---

## 〇、模型适配 PR 合入状态

### 当前状态：所有 PR 均未合入主分支

截至 2026 年 10 月，Xing4.0-29B-A4B 的模型适配代码**尚未合入任何主流推理框架的主分支**。官方通过预构建镜像提供可用部署：

| 框架 | PR | 状态 | Commit Hash | 预构建镜像 |
|------|-----|------|------------|-----------|
| vLLM | [vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135) | **Open / Pending** | `25aa52a` | `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0` |
| vLLM-Ascend | （内部 fork） | **未提交 PR** | — | `quay.io/xingchen-agi/xingchen-inference-vllm-ascend:v0.26.0rc1-xing4_0` |
| SGLang | [sgl-project/sglang#39793](https://github.com/sgl-project/sglang/pull/39793) | **Open / Pending** | `f22026fa` | `quay.io/xingchen-agi/xingchen-inference-sglang:v0.5.20.rc1-xing4_0` |
| TensorRT-LLM | [NVIDIA/TensorRT-LLM#19283](https://github.com/NVIDIA/TensorRT-LLM/pull/19283) | **Open / Pending** | `1b03b640` | — |
| llama.cpp | [ggml-org/llama.cpp#29012](https://github.com/ggml-org/llama.cpp/pull/29012) | **Open / Pending** | `63c16fb9` | — |
| KTransformers | [kvcache-ai/ktransformers#2168](https://github.com/kvcache-ai/ktransformers/pull/2168) | **Open / Pending** | `f2f0e3a8` | — |

### 关键说明

1. **vLLM PR #57135**：由 `zyp2014` 提交，添加 `Xing4_0ForCausalLM` 模型实现。该 PR 复用 DeepSeek-V2/V3 的 MLA attention + MoE 构建块，使用 vLLM 共享的 mHC ops（`mhc_pre` / `mhc_post`）。截至 2026-09-28，有测试者报告在 NVIDIA DGX Spark (GB10) 上存在精度问题（词级丢字），疑似 MLA attention 路径的 bug，尚未解决。

2. **vLLM-Ascend 无公开 PR**：vLLM-Ascend 仓库中**没有任何 Xing4.0 / mHC 相关代码**（`grep xing4_0/Xing4/xing4` 结果为空）。910B 上的部署能力通过预构建镜像提供，镜像内部打包了未公开的适配代码。

3. **910B 能跑 ≠ 上游仓库能跑**：
   - 官方预构建镜像 ✅ 可在 910B2 上部署
   - vLLM 上游主分支 ❌ 无 Xing4.0 支持（PR 未合入）
   - vLLM-Ascend 上游主分支 ❌ 无 Xing4.0 支持（无公开 PR）

### Xing4.0 模型架构概览

| 特性 | 值 | 说明 |
|------|-----|------|
| 架构类型 | `Xing4_0ForCausalLM` | 基于 DeepSeek-V2/V3 backbone |
| 注意力机制 | **MLA** (Multi-head Latent Attention) | q_lora=768, kv_lora=512 |
| 残差连接 | **mHC** (multi-step HyperConnection) | 独创架构，4 流并行 + Sinkhorn 归一化 |
| MoE | 64 专家 / 4 激活 / 1 共享专家 | sigmoid scoring, noaux_tc |
| MTP | 1 层 NextN 预测 | 多 token 预测头 |
| 参数量 | 29B (4B 激活) | bf16 |
| 上下文 | 256K (YaRN factor=64) | |

---

## 一、MLA 计算流程

### 1.1 模型参数

以 Xing4.0-29B-A4B 为例：

```
hidden_size        = 3584
num_heads          = 32
q_lora_rank        = 768    (query 压缩维度)
kv_lora_rank       = 512    (KV 压缩维度)
qk_nope_head_dim   = 128    (每头 nope 维度)
qk_rope_head_dim   = 64     (每头 rope 维度)
v_head_dim         = 128    (每头 value 维度)
```

### 1.2 核心思想：低秩压缩

MLA 发现 K 和 V 之间存在高度冗余，可以用一个**低秩 latent 向量**同时表示：

```
标准: 分别存储 K[H, D_k] 和 V[H, D_v]  → 每头两个向量
MLA:  存储一个压缩向量 c_kv[L]           → 所有头共享一个低维向量 (L << H×D)

        c_kv ──┬── W_UK ──→ K[H, D_k]   (解压出 Key)
               └── W_UV ──→ V[H, D_v]   (解压出 Value)
```

以 Xing4.0 为例：

```
标准 MHA:  32 × (128+128) × 2 = 16,384 维/token
MLA:       512 维/token (kv_lora_rank=512)

压缩比: 16,384 / 512 = 32×
```

### 1.3 为什么需要分离 RoPE 和 NoPE

#### 核心矛盾：RoPE 是位置相关的旋转

RoPE 对每个位置 p 的 key 施加一个旋转矩阵 R_p：

```
K_p^rope = R_p · K_p
```

不同位置的旋转矩阵**不同**（R_p ≠ R_q when p ≠ q）。

#### 压缩与 RoPE 不兼容

假设我们想压缩带 RoPE 的 key：

```
带 RoPE 的 key: K_rope[p] = R_p · W_UK · c_kv[p]

要压缩它，需要找到 c' 使得: K_rope[p] = W' · c'[p]
即: R_p · W_UK · c_kv[p] = W' · c'[p]
```

问题在于 R_p 随位置变化，**无法提取一个固定的解压矩阵 W'**。对每个位置都需要不同的 W'，压缩就失去了意义。

数学上说：**RoPE 旋转矩阵 R_p 与压缩矩阵 W_UK 不可交换**：

```
R_p · W_UK ≠ W_UK · R_p
```

#### MLA 的解决方案：拆分

把 key 拆成两部分：

```
Q = [Q_nope | Q_rope]    K = [K_nope | K_rope]
     ↑           ↑             ↑           ↑
  可压缩      不可压缩       可压缩      不可压缩
  (不用RoPE)  (用RoPE)     (不用RoPE)  (用RoPE)
     ↑           ↑             ↑           ↑
  维度大       维度小       维度大       维度小
 (128 dim)   (64 dim)    (128 dim)    (64 dim)
```

Attention score 是两部分之和（点积可加性）：

```
score(Q, K) = Q_nope · K_nope^T + Q_rope · R_p^T R_q · K_rope^T
```

因为点积是可加的，所以 nope 部分和 rope 部分可以**独立计算**。

### 1.4 计算流程总览

```
hidden_states [*, 3584]
    │
    ├───────────────────────────────────────────────────────┐
    │ Q 路径                                                │ KV 路径
    │                                                       │
    │  q_a_proj: 3584→768                                   │  kv_a_proj: 3584→576
    │         ↓                                              │  576 = 512(latent) + 64(rope)
    │  c_q [*, 768]                                         │         ↓
    │         ↓                                              │  kv_no_split [*, 576]
    │  q_a_layernorm                                        │    ┌─────┴──────┐
    │         ↓                                              │    ▼            ▼
    │  q_b_proj: 768→6144                                   │  c_kv[512]  k_rope[64]
    │         ↓                                              │    │            │
    │  ┌──────┴──────┐                                      │  kv_a_ln     RoPE
    │  ▼             ▼                                      │    │            │
    │  Q_nope      Q_rope                                   │  c_kv_normed  k_pe
    │  [*,32,128]  [*,32,64]                                │  [512]       [64]
    │                ↓                                       │    │            │
    │              RoPE                                     │    ▼            ▼
    │                ↓                                      │  k_cache     v_cache
    │  ┌─────────────┐  ← Prefill: 解压后全维度 attention    │  (latent)    (rope)
    │  │             │  ← Decode: Q 下投影到 latent 空间     │
    │  └─────────────┘                                      │
    │         ↓                                              │
    │  o_proj: 4096→3584                                    │
    │         ↓                                              │
    └───────────────────────────────────────────────────────┘
         ▼
    output [*, 3584]
```

### 1.5 KV Cache 存储：压缩存储

这是 MLA 的核心设计——**压缩存储**：

```
存储内容:
  k_cache (latent cache):  c_kv_normed [kv_lora_rank = 512]  ← 当K又当V
  v_cache (rope cache):    k_pe [qk_rope_head_dim = 64]      ← key的rope部分

每 token 存储: 512 + 64 = 576 维
对比标准 MHA:  32 × (128+128) × 2 = 16,384 维
压缩比:        28×
```

> **命名说明**：vLLM 框架强制要求 `kv_cache = (k_cache, v_cache)` 二元组结构。MLA 利用了这个结构但重新定义了含义：
> - `k_cache` 存的是 latent（同时充当 K 和 V）
> - `v_cache` 存的不是 Value，而是 key 的 rope 部分
>
> 代码注释也印证了这一点（`mla_v1.py:5729-5751`）：`# k_cache: nope_cache  v_cache: rope_cache`

### 1.6 Prefill 路径（解压后全维度 attention）

Prefill 是 **compute-bound（算力瓶颈）**，attention 计算量大（O(N²)），解压代价占比小（~10%），直接解压用 Flash Attention：

```
─── 解压 K/V ──────────────────────────────────────────────

c_kv_normed [N, 512]
    │  kv_b_proj: 512 → 8192 (= 32×128 + 32×128)
    │  FLOPs = N × 512 × 8192 × 2
    ▼
kv_nope [N, 32, 256]
    ├── K_nope [N, 32, 128]    V [N, 32, 128]
    │
    │  K_nope 与 K_rope 拼接
    ▼
K_full [N, 32, 192]  = concat(K_nope[128], K_rope[64])

─── 全维度 Attention ──────────────────────────────────────

Q_full = concat(Q_nope[128], Q_rope[64])  [N, 32, 192]

score = Q_full · K_full^T       [N, 32, N]
       FLOPs = N × N × 192 × 32 × 2

attn_output = softmax(score/√192) · V   [N, 32, 128]
             FLOPs = N × N × 128 × 32 × 2
```

### 1.7 Decode 路径（潜空间 attention，不解压 K/V）

Decode 是 **memory-bound（带宽瓶颈）**，attention 计算极少（Q 只有 1 行），解压 N 个历史 token 代价过高。利用矩阵乘法结合律，把 W_UK/W_UV 从 KV 侧挪到 Q/输出侧：

#### 数学原理

```
标准做法（先解压 K，再算 attention）:
  K_nope = c_kv · W_UK          # 解压: N个token × 512 → N × 128
  score = Q · K_nope^T
        = Q · (c_kv · W_UK)^T
        = Q · W_UK^T · c_kv^T       # 矩阵转置律: (AB)^T = B^T A^T

MLA Decode 做法（把 W_UK 挪到 Q 侧）:
  Q · W_UK^T · c_kv^T
  = (Q · W_UK^T) · c_kv^T       # 加括号改变计算顺序
    └──────────┘
       Q_latent                  # 先算这个！只处理1个token
```

两边的 score 数学上完全相等，但计算代价天差地别。

#### 具体计算流

```
─── Q 下投影到 latent 空间 ────────────────────────────────

Q_nope [1, 32, 128]
    │  × W_UK_T [32, 128, 512]    ← batch matmul, 只处理1个token
    │  FLOPs = 1 × 128 × 512 × 32 × 2 = 4.19M
    ▼
Q_latent [1, 32, 512]

─── 从 cache 读取（不解压）────────────────────────────────

K_latent = k_cache [N, 512]   ← 直接读，不乘任何矩阵
K_rope   = v_cache [N, 64]    ← 直接读

─── latent 空间 Attention ─────────────────────────────────

score_nope = Q_latent · K_latent^T    [1, 32, N]
             FLOPs = 1 × N × 512 × 32 × 2

score_rope = Q_rope · K_rope^T         [1, 32, N]
             FLOPs = 1 × N × 64 × 32 × 2

score = score_nope + score_rope         [1, 32, N]
        ↑ nope部分分数 + rope部分分数（点积可加性）

attn_latent = softmax(score) · K_latent  [1, 32, 512]
              ↑ 注意: V = K_latent, latent cache 同时充当 K 和 V

─── 输出上投影（只处理1个token）──────────────────────────

attn_latent [1, 32, 512]
    │  × W_UV [32, 512, 128]
    │  FLOPs = 1 × 512 × 128 × 32 × 2 = 4.19M
    ▼
attn_output [1, 32, 128]
```

W_UV 同理，利用结合律从 V 侧挪到输出侧：

```
标准: V = c_kv · W_UV  →  out = softmax(score) · V  =  softmax(score) · c_kv · W_UV
MLA:  out_latent = softmax(score) · c_kv  →  out = out_latent · W_UV
                                                   ↑ 只处理1个token
```

### 1.8 Prefill vs Decode 计算量对比

以 N=1000 个历史 token 为例：

```
                    Prefill (1000 token)    Decode (1 token + 1000 cache)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
解压 kv_b_proj       8.39 GFLOPS (12%)      如解压: 8.39G (99.2%) ← 灾难
Q 下投影 W_UK^T      0 (不需要)              4.19 MFLOPS (4.6%)    ← 只处理1个token
Attention 计算       20.48 GFLOPS (30%)      69.64 MFLOPS (76%)
V 上投影 W_UV         0 (已解压)              4.19 MFLOPS (4.6%)
o_proj               29.36 GFLOPS (44%)     29.36 MFLOPS (32%)
其他                 9.07 GFLOPS (14%)      14.07 MFLOPS (15%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计                 67.3 GFLOPS            0.091 GFLOPS (不解压)
                                            8.46 GFLOPS (如解压)
```

**核心差异**：解压时 `kv_b_proj` 作用在 1000 个历史 token 上（8.39G），而不解压时 `W_UK_T` 和 `W_UV` 只作用在 1 个 query token 上（共 8.38M），**计算量差 1000 倍**。

| 特性 | Prefill | Decode |
|------|---------|--------|
| **处理 token 数** | 一次处理整个 prompt (如 1000) | 一次只处理 1 个 token |
| **计算量** | 巨大（O(N²) attention） | 极少（O(N) attention） |
| **瓶颈** | 算力（Compute-bound） | 显存带宽（Memory-bound） |
| **KV cache 解压** | ✅ 解压后用 Flash Attention | ❌ 不解压，在 latent 空间做 |
| **Q 处理** | 不需要额外压缩 | 下投影到 latent 空间 |
| **核心优化目标** | 减少计算量 | 减少显存读写 |

---

## 二、910B MLA 实现剖析

### 2.1 核心架构

910B MLA 实现在 `vllm_ascend/attention/mla_v1.py`，核心类 `AscendMLAImpl`（line 810）。

主要组件：

| 组件 | 类名 | 文件位置 | 说明 |
|------|------|---------|------|
| Attention Backend | `AscendMLABackend` | `mla_v1.py:85` | 注册名 `"ASCEND_MLA"`，kernel block size 128 |
| MLA Attention Layer | `AscendMLAAttention` | `mla_v1.py:125` | 继承上游 `MLAAttention` |
| MLA Impl | `AscendMLAImpl` | `mla_v1.py:810` | 核心 forward 实现（2200 行） |
| Metadata Builder | `AscendMLAMetadataBuilder` | `mla_v1.py:256` | 构建 prefill/decode metadata |
| Model Wrapper | `AscendMultiHeadLatentAttention` | `ops/mla.py:111` | 模型层封装 |
| DeepSeek Patch | `_deepseek_v2_mla_attention_init` | `patch/worker/patch_deepseek_v2.py:61` | DeepSeek MLA 初始化 patch |

### 2.2 关键设计：KV Cache 存储压缩

910B 存储的是**压缩后的低秩 KV**，而非完整的 K/V：

```python
# model_runner_v1.py:4860-4876 — KV cache 维度
# K dimension = kv_lora_rank (e.g. 512)  ← 压缩后的 latent
# V dimension = qk_rope_head_dim (e.g. 64) ← rope 部分
```

### 2.3 Decode 路径：潜空间注意力

910B decode 的核心思路是**在 latent 空间直接做 attention**，不需要解压 K/V：

```python
# mla_v1.py:1026-1038 — Query 下投影到 latent 空间
def _q_proj_and_k_up_proj(self, x):
    q_nope, q_pe = self.q_proj(x)[0].split(...)
    ql_nope = torch.bmm(q_nope, self.W_UK_T)  # (N,B,P)×(N,P,L) → (N,B,L)
    return ql_nope.transpose(0, 1), q_pe

# mla_v1.py:1006-1013 — Attention 输出上投影
def _v_up_proj(self, x):
    x = x.view(self.num_heads, -1, self.kv_lora_rank)
    x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV)  # latent → v_head_dim
    return x.reshape(-1, self.num_heads * self.v_head_dim)
```

然后调用 FIA v2 做 paged attention，**K 和 V 参数都传 latent cache**，同时传入 rope 部分：

```python
# mla_v1.py:1857-1858
torch_npu.npu_fused_infer_attention_score_v2(
    q_nope,          # query (latent 空间, 512 dim)
    k_nope,          # key   = latent cache (512 dim)
    k_nope,          # value = latent cache (512 dim)  ← K=V=latent
    query_rope=q_pe, # query rope 部分 (64 dim)
    key_rope=k_pe,   # key rope 部分 (64 dim)
    block_table=...,
    block_size=...,
)
# 输出再经 _v_up_proj 上投影回 v_head_dim
```

**关键**：FIA v2 算子内部融合了 nope+rope 的拼接和 attention 计算。

### 2.4 Prefill 路径：解压后全维度 attention

Prefill 是 compute-bound，直接解压做全维度 attention：

```python
# mla_v1.py:1310 — kv_b_proj 解压
kv_nope = self.kv_b_proj(kv_c_normed)[0]  # latent → full K,V
k_nope, v = kv_nope.split([qk_nope_head_dim, v_head_dim], dim=-1)
# 然后用 npu_fused_infer_attention_score 做 TND layout 的全维度 attention
```

### 2.5 融合算子

| 算子 | 用途 | 910B 代码位置 |
|------|------|-------------|
| `npu_kv_rmsnorm_rope_cache` | 融合 RMSNorm + RoPE + Cache写入 | `mla_v1.py:1503` |
| `npu_fused_infer_attention_score_v2` | MLA paged attention (支持 query_rope/key_rope) | `mla_v1.py:1857` |
| `npu_fused_infer_attention_score` | Prefill attention (TND layout) | `mla_v1.py:1400` |
| `npu_mla_prolog_v3` | 全融合 decode 预处理 (W8A8量化时) | `mla_v1.py:1929` |
| `npu_transpose_batchmatmul` | latent 空间的 batch matmul | `mla_v1.py:1010` |
| `npu_interleave_rope` | 交错 RoPE | `mla_v1.py:1615` |

---

## 三、310P 在 vLLM-Ascend 适配 MLA 的 Gap

### 3.1 Gap 总览

6 层阻断，从入口到底层算子：

```
模型加载入口
    │
    ▼ Gap1: model_runner 硬阻断
┌─────────────────────────────────────────────────┐
│ model_runner_310p.py:709  raise ValueError      │
│ worker/v2/model_runner.py:173 raise NotImplemented│
└─────────────────────────────────────────────────┘
    │
    ▼ Gap2: platform 后端路由无 MLA 映射
┌─────────────────────────────────────────────────┐
│ platform.py:261-269  compatibility_backend_map  │
│   只有 (False,False), 无 (True,False) MLA 映射   │
└─────────────────────────────────────────────────┘
    │
    ▼ Gap3: 无 MLA Backend / Impl / MetadataBuilder
┌─────────────────────────────────────────────────┐
│ 不存在 AscendMLABackend310                       │
│ 不存在 AscendMLAImpl310                          │
│ 不存在 AscendMLAMetadataBuilder310               │
└─────────────────────────────────────────────────┘
    │
    ▼ Gap4: NPU 算子缺失（最核心技术阻断）
┌─────────────────────────────────────────────────┐
│ npu_fused_infer_attention_score_v2  ❌          │
│ npu_kv_rmsnorm_rope_cache           ❌          │
│ npu_mla_prolog_v3                   ❌          │
│ npu_transpose_batchmatmul           ❌          │
│ npu_interleave_rope                 ❌          │
└─────────────────────────────────────────────────┘
    │
    ▼ Gap5: KV cache 分配逻辑不支持 MLA 双 cache
┌─────────────────────────────────────────────────┐
│ 无 _get_attention_kv_cache_dims                  │
│ 无 page_strided MLA 分配逻辑                     │
│ 无 AscendMLAAttentionSpec 判断                   │
└─────────────────────────────────────────────────┘
    │
    ▼ Gap6: 310P 特有约束冲突
┌─────────────────────────────────────────────────┐
│ block_size × head_size ≤ 128×128 vs kv_lora=512  │
│ → block_size ≤ 32                                │
│ FRACTAL_NZ 格式 vs MLA latent cache 布局          │
│ head_size ≤ 256 限制                              │
└─────────────────────────────────────────────────┘
```

### 3.2 各 Gap 代码层面详细分析

#### Gap 1: Model Runner 硬阻断

**文件**: `vllm_ascend/_310p/model_runner_310p.py:709-711`

```python
if self.model_config.use_mla:
    logger.error("MLAAttention is not supported.")
    raise ValueError("MLAAttention is not supported for 310P.")
```

**文件**: `vllm_ascend/_310p/worker/v2/model_runner.py:172-174`

```python
if model_config.use_mla:
    raise NotImplementedError("MLA is not supported by model runner v2 on 310P.")
```

**修改方案**: 移除这两处 raise，改为条件判断（当 MLA backend 可用时放行）。

---

#### Gap 2: Platform 后端路由无 MLA 映射

**文件**: `vllm_ascend/platform.py:261-272`

```python
# 910B 的映射表（已实现）
backend_map = {
    (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",  # ← MLA
    (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
}

# 310P 的映射表（只有标准 attention）
compatibility_backend_map = {
    (False, False): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
    # TODO If MLA/SFA is supported in the future...  ← 注释占位，未实现
    # (True, False): "...AscendMLABackend310",
}
```

key 是 `(use_mla, use_sfa)`。310P 只注册了 `(False, False)`，MLA 模型会走默认值，仍然是标准 attention backend，无法处理 MLA。

**修改方案**:

```python
compatibility_backend_map = {
    (False, False): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
    (True, False): "vllm_ascend._310p.attention.mla_v1.AscendMLABackend310",  # ← 新增
}
```

---

#### Gap 3: 不存在 MLA Backend / Impl / MetadataBuilder

910B 有完整的三个类（`vllm_ascend/attention/mla_v1.py`）：

| 910B 类 | 910B 代码位置 | 310P 对应 | 状态 |
|---------|-------------|----------|------|
| `AscendMLABackend` | `mla_v1.py:85` | `AscendMLABackend310` | **不存在** |
| `AscendMLAImpl` | `mla_v1.py:810` | `AscendMLAImpl310` | **不存在** |
| `AscendMLAMetadataBuilder` | `mla_v1.py:256` | `AscendMLAMetadataBuilder310` | **不存在** |

910B 的 `AscendMLAImpl` 是一个 2200 行的类，包含完整的 MLA forward 逻辑。310P 需要新建等价类，但底层算子不同。

**需新建文件**:

```
vllm_ascend/_310p/attention/
    mla_v1.py              ← AscendMLABackend310 + AscendMLAImpl310
    mla_metadata_builder.py ← AscendMLAMetadataBuilder310
```

---

#### Gap 4: NPU 算子缺失（最核心技术阻断）

910B MLA 依赖的算子 vs 310P 可用算子：

| 910B 使用的算子 | 用途 | 910B 代码位置 | 310P 可用替代 |
|---------------|------|-------------|-------------|
| `npu_fused_infer_attention_score_v2` | **Decode latent attention** (支持 query_rope/key_rope) | `mla_v1.py:1857` | ❌ **无替代** |
| `npu_fused_infer_attention_score` | **Prefill attention** (TND layout) | `mla_v1.py:1400` | ❌ 310P 用 `_npu_flash_attention` |
| `npu_kv_rmsnorm_rope_cache` | **融合 RMSNorm+RoPE+Cache写入** | `mla_v1.py:1503` | ❌ 需拆为 3 步 |
| `npu_transpose_batchmatmul` | **Q 下投影 / V 上投影** (latent space bmm) | `mla_v1.py:1010` | ❌ 用 `torch.bmm` 替代 |
| `npu_interleave_rope` | **交错 RoPE** | `mla_v1.py:1615` | ❌ 用 `_npu_rotary_embedding` 替代 |
| `npu_mla_prolog_v3` | **全融合 decode 预处理** | `mla_v1.py:1929` | ❌ 无替代 |

##### 最关键缺失: `npu_fused_infer_attention_score_v2`

910B decode 调用（`mla_v1.py:1857-1858`）:

```python
# K 和 V 都传 latent cache，同时传入 rope 部分
torch_npu.npu_fused_infer_attention_score_v2(
    q_nope,          # query (latent 空间, 512 dim)
    k_nope,          # key   = latent cache (512 dim)
    k_nope,          # value = latent cache (512 dim)  ← K=V=latent
    query_rope=q_pe, # query rope 部分 (64 dim)
    key_rope=k_pe,   # key rope 部分 (64 dim)
    block_table=...,
    block_size=...,
)
```

这个算子**内部融合了**:
1. 从 paged cache 按 block_table 读取 latent KV
2. 从 paged cache 读取 rope KV
3. 计算 `score = Q_latent·K_latent^T + Q_rope·K_rope^T`
4. `softmax(score)·K_latent` → 输出

310P 的 `_npu_paged_attention`（`_310p/attention/attention_v1.py:198`）:

```python
torch_npu._npu_paged_attention(
    query=query,           # 标准 query (无 rope 分离)
    key_cache=self.key_cache,
    value_cache=self.value_cache,
    num_kv_heads=...,
    num_heads=...,
    scale_value=...,
    block_table=...,
    context_lens=...,
)
# ← 不支持 query_rope / key_rope 参数
# ← 不支持 K=V=latent 的 latent 空间 attention
```

##### 可拆解的算子（非硬阻断）

| 算子 | 拆解方案 |
|------|---------|
| `npu_kv_rmsnorm_rope_cache` | 拆为: RMSNorm + `npu_apply_rotary_pos_emb`(支持dim=64) + `_npu_reshape_and_cache` |
| `npu_transpose_batchmatmul` | 用标准 `torch.bmm` 替代，性能有损失但功能等价 |
| `npu_interleave_rope` | 用 `_npu_rotary_embedding` 替代 |

---

#### Gap 5: KV Cache 分配逻辑不支持 MLA 双 cache

**文件**: `vllm_ascend/_310p/model_runner_310p.py`

910B 的 `model_runner_v1.py` 有完整的 MLA KV cache 分配逻辑:

```python
# 910B: model_runner_v1.py:4860-4876 — 获取 MLA cache 维度
def _get_attention_kv_cache_dims(self, layer_name, kv_cache_spec):
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        attn_layer = attn_layers[layer_name]
        if isinstance(attn_layer, MLAAttention):
            return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim  # ← (512, 64)

# 910B: model_runner_v1.py:5454-5469 — MLA 双 cache 分配
if isinstance(current_kv_cache_spec, AscendMLAAttentionSpec):
    k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
    kv_cache_shape_list = [
        (*cache_prefix, k_dim),  # k_cache: latent (512)
        (*cache_prefix, v_dim),  # v_cache: rope (64)
    ]
```

310P 的 `model_runner_310p.py` **完全没有这些逻辑**:

```python
# 310P: model_runner_310p.py:725-838 — 只有标准 attention cache 分配
def _allocate_kv_cache_tensors(self, kv_cache_config):
    # ... 只有标准 AttentionSpec 的处理
    # 没有 AscendMLAAttentionSpec 判断
    # 没有 _get_attention_kv_cache_dims
    # 没有 latent + rope 双 cache 分配
```

**需要新增**:
- `_get_attention_kv_cache_dims` 方法（可直接复用 910B 逻辑）
- MLA `AscendMLAAttentionSpec` 判断分支
- latent cache + rope cache 的双 tensor 分配

---

#### Gap 6: 310P 特有约束冲突

##### block_size × head_size 限制

**文件**: `model_runner_310p.py:67`

```python
_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128  # = 16384
```

MLA 的 latent cache `head_size = kv_lora_rank = 512`:

```python
block_size × head_size ≤ 16384
block_size × 512 ≤ 16384
block_size ≤ 32
```

310P 的虚拟块拆分机制（`model_runner_310p.py:797-815`）:

```python
supported_sizes = [
    s for s in self.attn_backend.get_supported_kernel_block_sizes()
    if s * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
]
# get_supported_kernel_block_sizes() 返回 [128, 64]
# 128 × 512 = 65536 > 16384 → 不满足
# 64 × 512 = 32768 > 16384 → 不满足
# 列表为空！
```

**问题**: 310P 的 `get_supported_kernel_block_sizes` 只返回 `[128, 64]`，都超过限制。需要添加 32:

```python
# 修改 AscendAttentionBackend310.get_supported_kernel_block_sizes
@staticmethod
def get_supported_kernel_block_sizes() -> list[int]:
    return [128, 64, 32]  # ← 新增 32，适配 MLA latent dim
```

##### FRACTAL_NZ 格式

310P 强制 NZ 格式（`hardware_profile.py:296`）:

```python
weight_layout_policy=WeightLayoutPolicy.FORCE_NZ
```

KV cache shape 是 5D NZ（`_310p/attention/attention_v1.py:55-81`）:

```python
# 标准 attention: (2, num_blocks, (num_kv_heads × head_size)//16, block_size, 16)
```

MLA 的 latent cache 需要类似格式:

```python
# MLA latent: head_size=512, num_kv_heads=1 (所有头共享)
# NZ shape: (2, num_blocks, (1×512)//16, block_size, 16) = (2, B, 32, S, 16)
# MLA rope:  head_size=64, num_kv_heads=1
# NZ shape: (2, num_blocks, (1×64)//16, block_size, 16) = (2, B, 4, S, 16)
```

理论上可行（512 和 64 都是 16 的倍数），但 `get_kv_cache_shape` 需要适配 MLA 的 `num_kv_heads=1`。

##### head_size ≤ 256 限制

**文件**: `vllm_ascend/_310p/worker/v2/model_runner.py:822-823`

```python
max_head_size = max(spec.head_size for spec in attention_specs)
if max_head_size > 256:
    raise NotImplementedError(f"310P paged attention requires head_size <= 256")
```

MLA 的 `AscendMLAAttentionSpec` 的 `head_size` 设置方式（910B `model_runner_v1.py:6085-6139`）:

```python
# 910B: MLA spec 的 head_size = kv_lora_rank + qk_rope_head_dim = 576
kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
    block_size=self.block_size,
    num_kv_heads=1,
    head_size=head_size,  # ← 576
)
```

如果 310P 也设 `head_size=576`，会触发 `> 256` 限制。**但 MLA 实际存为两个独立 cache (512 + 64)**，需要修改 spec 创建逻辑使 `head_size` 反映单个 cache 的维度。

### 3.3 310P MLA 适配方案

#### 方案 A：解压存储（最简单，放弃 MLA 内存优势）

**思路**：在写入 KV cache 前，先用 `kv_b_proj` 解压到完整维度，存储标准 K/V。

```
kv_no_split → kv_a_layernorm → kv_b_proj → 完整 K[num_heads, qk_nope_dim], V[num_heads, v_head_dim]
                                                      → 写入标准 paged KV cache
Decode: q_b_proj → q_nope + q_pe → RoPE(q_pe) → concat(q_nope, q_pe)
        → _npu_paged_attention(q, k_cache, v_cache)
```

| 优点 | 缺点 |
|------|------|
| 不改 310P attention backend | KV cache 内存 ×28（丧失 MLA 核心价值） |
| 所有 NPU op 已存在 | 256K 上下文时 OOM 风险极高 |
| 开发量最小 | 需要每层存储 32 heads × 192 dim |

**适用场景**：短期验证、短上下文（<4K）精度对齐。**不适合生产部署**。

#### 方案 B：潜空间 attention + 软件 rope 拼接（推荐）

**思路**：像 910B 一样在 latent 空间做 attention，但用软件方式处理 rope 拼接。

**Decode 路径**：
```
1. q_b_proj(hidden) → q_nope[H, qk_nope_dim], q_pe[H, qk_rope_dim]
2. RoPE(q_pe)  ← 用 npu_apply_rotary_pos_emb (310P 已支持 dim=64)
3. q_nope × W_UK_T → q_latent[H, kv_lora_rank]  ← 用 torch.bmm
4. 从 paged cache 读取 k_latent[kv_lora_rank], k_pe[qk_rope_dim]
5. 将 latent + rope 预拼接为单个 query/key:
   q_combined = concat(q_latent[512], q_pe[64])  # dim=576
   k_combined = concat(k_latent[512], k_pe[64])  # dim=576
6. 用 _npu_paged_attention 做单头 paged attention (num_kv_heads=1, head_size=576)
   但 block_size × 576 > 128×128 → block_size ≤ 28，需更小的块
7. attn_output × W_UV → v_output[v_head_dim]
```

| 优点 | 缺点 |
|------|------|
| 保持 MLA 内存优势 | rope 拼接增加维度，block_size 受限更严重 |
| 复用 `_npu_paged_attention` | 需要额外的 rope 预拼接计算 |
| 不需要自定义算子 | block_size ≤ 28 → 块表更大，调度开销增加 |

#### 方案 C：Ascend C 自定义 MLA 算子（最优性能，最高开发成本）

**思路**：开发 310P 专用的 MLA fused attention 算子，类似 910B 的 FIA v2 但适配 310P 架构。

```
输入: q_latent[H, kv_lora_rank], q_pe[H, qk_rope_dim]
      k_cache_paged[num_blocks, block_size, kv_lora_rank]
      k_pe_paged[num_blocks, block_size, qk_rope_dim]
      block_table, seq_lens

算子内部:
  for each block:
    k_latent = k_cache_paged[block_id]     # [block_size, kv_lora_rank]
    k_pe = k_pe_paged[block_id]            # [block_size, qk_rope_dim]

    score_nope = q_latent @ k_latent.T     # [H, block_size]
    score_rope = q_pe @ k_pe.T             # [H, block_size]
    score = scale * (score_nope + score_rope)

    attn_weights = softmax(score + mask)
    output += attn_weights @ k_latent       # [H, kv_lora_rank]

  # 最后在算子外部做 W_UV 上投影
```

| 优点 | 缺点 |
|------|------|
| 最优性能，一次 kernel launch | 开发周期长（需 Ascend C 或 Triton-Ascend） |
| 完全保持 MLA 优势 | 需要维护 310P 专用算子 |
| 可融合 RoPE + RMSNorm + CacheWrite | 需要处理 NZ 格式 |

#### 方案 D：混合方案（推荐实际落地路径）

分阶段实现：

| 阶段 | 方案 | 目标 | 预估周期 |
|------|------|------|---------|
| 阶段 1 | 方案 A — 解压存储 | 精度验证 | ~2 周 |
| 阶段 2 | 方案 B — latent + rope 预拼接 | 可用部署 | ~4 周 |
| 阶段 3 | 方案 C — 自定义算子 | 高性能 | ~8 周 |

### 3.4 修改清单汇总

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 1 | `_310p/model_runner_310p.py:709-711` | 移除 `raise ValueError("MLAAttention is not supported")` | 低 |
| 2 | `_310p/worker/v2/model_runner.py:172-174` | 移除 `raise NotImplementedError("MLA is not supported")` | 低 |
| 3 | `platform.py:261-269` | `compatibility_backend_map` 新增 `(True, False)` → MLA backend 路径 | 低 |
| 4 | **新建** `_310p/attention/mla_v1.py` | `AscendMLABackend310` + `AscendMLAImpl310` (核心 forward 逻辑) | **高** |
| 5 | **新建** `_310p/attention/mla_metadata_builder.py` | MLA metadata builder (可部分复用 910B) | 中 |
| 6 | `_310p/attention/attention_v1.py:97-99` | `get_supported_kernel_block_sizes` 新增 32 | 低 |
| 7 | `_310p/model_runner_310p.py:725-838` | 新增 `_get_attention_kv_cache_dims` + MLA 双 cache 分配 | 中 |
| 8 | `_310p/worker/v2/model_runner.py:822` | 修改 head_size 限制逻辑，豁免 MLA spec | 低 |
| 9 | `_310p/ops/rotary_embedding.py` | 确认 `npu_apply_rotary_pos_emb` 支持 dim=64（已支持） | 无 |
| 10 | `_310p/attention/attention_v1.py` | `_npu_reshape_and_cache` 适配 MLA 双 cache 写入 | 中 |

**核心难点在 #4**：`AscendMLAImpl310` 的 forward 实现。由于 310P 缺少 `npu_fused_infer_attention_score_v2`（支持 rope 融合的 paged attention），需要用现有 310P 算子拼装出等价的 latent space attention 逻辑，或开发自定义 Ascend C 算子。

---

## 四、mHC 架构原理与 310P 适配 Gap

### 4.1 mHC 核心概念

mHC (multi-step HyperConnection) 是 Xing4.0 独创的**多流残差连接机制**，替代了标准 Transformer 的简单残差加法。它维护 `hc_mult=4` 个并行的 hidden state 流，通过门控和 Sinkhorn 归一化的混合矩阵在流之间动态路由信息。

#### 配置参数

```python
hc_mult = 4              # 并行流数量
hc_sinkhorn_iters = 20   # Sinkhorn 归一化迭代次数
hc_eps = 1e-6            # Sinkhorn epsilon
mhc_h_res_clamp_min = -30   # comb logits 截断下限
mhc_h_res_clamp_max = 30    # comb logits 截断上限
```

#### 标准 Transformer vs mHC 对比

```
标准 Transformer Decoder Layer:
    h ──→ norm(h) ──→ attn(h) ──→ h + attn_out        ← 简单残差加法
    h ──→ norm(h) ──→ ffn(h)  ──→ h + ffn_out           ← 简单残差加法

mHC Decoder Layer:
    h[4 streams] ──→ attn_hc(h) ──→ collapse ──→ norm ──→ attn ──→ expand + mix ──→ h'[4 streams]
                                                                        ↑
                                        post gate + comb matrix (Sinkhorn normalized)
```

### 4.2 mHC 计算流程

#### 第一步：多流扩展

模型入口处，hidden_states 从单流扩展为 4 流：

```python
# Xing4_0Model.forward()
hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
# [batch, seq, 3584] → [batch, seq, 4, 3584]
```

初始时 4 个流完全相同，后续每层通过 mHC 让它们逐渐分化。

#### 第二步：HyperConnection 前向

每个 Decoder Layer 有两个 `Xing4_0HyperConnection`（`attn_hc` 和 `ffn_hc`），它接收 4 流输入，输出三样东西：

```python
# Xing4_0HyperConnection.forward(hidden_streams)
# hidden_streams: [batch, seq, 4, 3584]

hc = 4  # hc_mult
flat = input_norm(hidden_streams.flatten(start_dim=2))  # RMSNorm, [batch, seq, 4*3584=14336]

# 一个大线性投影: 14336 → 24 (= 4 + 4 + 4*4)
pre_w, post_w, comb_w = F.linear(flat, hc_fn).split([4, 4, 16], dim=-1)
```

权重参数：

| 参数名 | 形状 | 含义 | 初始化 |
|--------|------|------|--------|
| `hc_fn` | `[24, 14336]` | 线性投影权重矩阵 | 正态分布 |
| `hc_base` | `[24]` | 偏置向量 | 全 0 |
| `hc_scale` | `[3]` | 缩放因子 | 全 1 |

`24 = 4 + 4 + 4×4`，分别对应三组输出：

```
hc_fn [24, 14336]
    │
    ├── pre_w  [*, 4]      ← 前置门控权重
    ├── post_w [*, 4]      ← 后置门控权重
    └── comb_w [*, 16]     ← 混合矩阵 (4×4 展平)
```

然后加上 bias 和 scale：

```python
pre_b, post_b, comb_b = hc_base.split([4, 4, 16])      # 偏置
pre_scale, post_scale, comb_scale = hc_scale.unbind(0)   # 缩放因子

# 前置门控: sigmoid → [0, 1]
pre = torch.sigmoid(pre_w * pre_scale + pre_b)           # [*, 4]

# 后置门控: 2*sigmoid → [0, 2]
post = 2 * torch.sigmoid(post_w * post_scale + post_b)   # [*, 4]

# 混合矩阵: 4×4，经 Sinkhorn 归一化为双随机矩阵
comb_logits = comb_w.view(..., 4, 4) * comb_scale + comb_b.view(4, 4)
comb_logits = torch.clamp(comb_logits, -30, 30)
comb_max = comb_logits.amax(dim=-1, keepdim=True)
comb = torch.exp(comb_logits - comb_max)                 # [*, 4, 4]
```

#### 第三步：Sinkhorn 归一化

这是 mHC 的核心——将 4×4 混合矩阵归一化为**双随机矩阵**（行和列都归一化）：

```python
for _ in range(20):  # hc_sinkhorn_iters
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # 行归一化
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # 列归一化
```

Sinkhorn 归一化的意义：
- **行归一化**：每个输出流从所有输入流接收的总权重为 1
- **列归一化**：每个输入流向所有输出流发送的总权重为 1
- 结果是一个**双随机矩阵**，保证信息在 4 个流之间均衡流动

#### 第四步：流坍缩

用 `pre` 门控将 4 个流加权求和为 1 个流，作为 attention/FFN 的输入：

```python
collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
# pre: [batch, seq, 4, 1]
# hidden_streams: [batch, seq, 4, 3584]
# collapsed: [batch, seq, 3584]  ← 4流坍缩为1流
```

#### 第五步：LayerNorm + Attention/FFN

```python
collapsed = self.input_layernorm(collapsed)     # RMSNorm
attn_output = self.self_attn(collapsed, ...)     # 在坍缩后的单流上做 attention
# attn_output: [batch, seq, 3584]
```

#### 第六步：流扩展 + 残差混合

将 attention 输出扩展回 4 流，并用 comb 矩阵混合残差：

```python
# post 门控扩展: [batch, seq, 3584] → [batch, seq, 4, 3584]
expanded = post.unsqueeze(-1) * attn_output.unsqueeze(-2)
# [batch, seq, 4, 1] * [batch, seq, 1, 3584] → [batch, seq, 4, 3584]

# comb 矩阵混合历史流: [batch, seq, 4, 4] @ [batch, seq, 4, 3584]
residual = torch.matmul(comb, hidden_states)
# [batch, seq, 4, 4] × [batch, seq, 4, 3584] → [batch, seq, 4, 3584]

# 最终: 扩展的 attention 输出 + 混合的残差
hidden_states = expanded + residual
# [batch, seq, 4, 3584]
```

FFN 部分完全对称，用 `ffn_hc` 做同样的操作。

#### 完整数据流图

```
输入: hidden_states [batch, seq, 4, 3584]
                │
    ┌───────────┴───────────────┐
    │   Xing4_0HyperConnection  │  (attn_hc)
    │                           │
    │  1. RMSNorm + Linear      │
    │     → pre[4], post[4],    │
    │       comb_w[16]          │
    │                           │
    │  2. pre = sigmoid(...)    │  ← 前置门控 [0,1]
    │     post = 2*sigmoid(...) ← 后置门控 [0,2]
    │     comb = Sinkhorn(      │  ← 20次行列归一化
    │       exp(comb_logits))   │     → 双随机矩阵 4×4
    │                           │
    │  3. collapsed =           │  ← 4流加权求和 → 1流
    │     (pre * h).sum(dim=2)  │
    └───────────┬───────────────┘
                │ collapsed [batch, seq, 3584]
                ▼
         input_layernorm
                │
                ▼
         self_attn (MLA)
                │ attn_output [batch, seq, 3584]
                ▼
    ┌───────────┴───────────────┐
    │  post * attn_output       │  ← 门控扩展回4流
    │  + comb @ hidden_states   │  ← Sinkhorn混合残差
    └───────────┬───────────────┘
                │
                ▼
    hidden_states' [batch, seq, 4, 3584]
                │
    ┌───────────┴───────────────┐
    │   Xing4_0HyperConnection  │  (ffn_hc)  ← 完全对称
    │   ... 同上 ...             │
    └───────────┬───────────────┘
                │
                ▼
         post_attention_layernorm
                │
                ▼
         mlp (MoE / Dense)
                │
                ▼
    post * mlp_output + comb @ hidden_states
                │
                ▼
    hidden_states'' [batch, seq, 4, 3584]  ← 传给下一层
```

#### mHC 相比标准残差连接的优势

| 特性 | 标准 Transformer | mHC |
|------|-----------------|-----|
| 残差流数 | 1 | 4（hc_mult） |
| 残差连接 | `h + f(h)` 简单加法 | `post·f(collapse(h)) + comb·h` 门控混合 |
| 信息路由 | 无路由，所有信息走同一通道 | 4 流动态路由，每层不同 |
| 门控 | 无 | pre (输入门控) + post (输出门控) |
| 混合矩阵 | 单位矩阵 | Sinkhorn 双随机矩阵（动态生成） |
| 表达能力 | 有限 | 显著增强（多流 + 动态路由） |
| 计算代价 | 低 | 高（每层 2 次大 Linear + 20 次 Sinkhorn） |

### 4.3 910B 上的 mHC 实现

vllm-ascend 中有**两套** mHC 实现，对应不同模型：

#### 实现 1：原生 Ascend C++ 算子（DeepSeek V4/V4.1）

这是**性能最优**的路径，使用 `torch.ops._C_ascend` 注册的自定义 C++ 算子：

| 算子 | 用途 | 使用方 |
|------|------|-------|
| `npu_hc_pre_v2` | mHC 前置（RMSNorm + Linear + sigmoid + Sinkhorn + 坍缩） | DeepSeek V4 |
| `npu_hc_pre_v3` | mHC 前置 v3（支持 pre_mix 传入，延迟坍缩） | DeepSeek V4.1 |
| `npu_hc_post` | mHC 后置（门控扩展 + comb 矩阵混合残差） | V4 + V4.1 |

DeepSeek V4 的调用（`models/deepseek_v4/model.py:745-755`）：

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    y = torch.ops._C_ascend.npu_hc_pre_v2(
        x, hc_fn, hc_scale, hc_base,
        self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps
    )
    return y  # → (post_mix, comb_mix, layer_input)

def hc_post(self, x, residual, post, comb):
    y = torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    )
    return y.squeeze(0)
```

**关键**：这些 C++ 算子将 Sinkhorn 20 次迭代、sigmoid 门控、RMSNorm、流坍缩**全部融合到一个 kernel launch 中**。

#### 实现 2：PyTorch fallback（GLM-5.3-Flash / Xing4.0）

对于使用上游 vLLM mHC 接口的模型，vllm-ascend 通过 `patch_triton.py` 将 TileLang 算子替换为 PyTorch 实现：

`patch/worker/patch_triton.py:229-317`：

```python
# tilelang mHC pre kernel 在 NPU 上不可用
# 用 mhc_pre_torch (PyTorch 实现) 替代
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch, mhc_post_torch

def _mhc_pre_npu(self, residual, fn, hc_scale, hc_base, ...):
    post_mix, comb_mix, layer_input = _mhc_pre_torch(
        residual, fn, hc_scale, hc_base, ...
    )
    # tilelang 版本融合了 RMSNorm，PyTorch 版本没有，需要补上
    if norm_weight is not None:
        layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
    return post_mix, comb_mix, layer_input

# 替换上游的 forward_oot 和 forward_native
_mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
_mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
_mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
_mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
```

#### 两套实现的对比

```
                    DeepSeek V4/V4.1              GLM-5.3-Flash / Xing4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
算子来源            vllm_ascend C++ 自定义算子     上游 vLLM PyTorch fallback
                    torch.ops._C_ascend.*          vllm.model_executor.kernels.mhc.torch.*

算子名              npu_hc_pre_v2 / v3             mhc_pre_torch
                    npu_hc_post                    mhc_post_torch

Sinkhorn 20次       ✅ 融合在 C++ kernel 内部     ❌ PyTorch for 循环
RMSNorm             ✅ 融合在 C++ kernel 内部     ❌ 需要额外补上 (_mhc_rms_norm)
kernel launch 次数  1 次 (pre) + 1 次 (post)      多次 (多个 PyTorch 算子)

性能                最优                           有性能损失
精度                HF32 模式 (10bit mantissa)    FP32
```

### 4.4 310P 适配 mHC 的 Gap

#### 好消息：底层 C++ kernel 有 310P 代码路径

mHC 的 Ascend C kernel 中**已经写了 310P (950PR/950DT) 的实现**：

**`csrc/moe/hc_pre/op_kernel/hc_pre.cpp:19,49`**：
```cpp
#if defined(__DAV_C310__)     // ← 950PR/950DT (arch35) 代码路径
    #include "hc_pre_m_k_split_core_arch35.h"
    #include "hc_pre_m_split_core_arch35.h"
    #include "hc_pre_base_arch35.h"
#else                          // ← A3 代码路径
    #include "hc_pre_m_k_split_core.h"
    #include "hc_pre_base.h"
#endif
```

**`csrc/moe/hc_post/op_kernel/hc_post.cpp:17,36`**：
```cpp
#if defined(__DAV_C310__)     // ← 950PR/950DT 代码路径
    #include "hc_post_float32.h"
    #include "hc_post_bfloat16.h"
#endif
```

#### 坏消息：三层注册缺失

##### Gap 1: Op 定义未注册 `ascend310p`

**文件**: `csrc/moe/hc_pre/op_host/hc_pre_def.cpp:75-83`

```cpp
this->AICore().AddConfig("ascend910b");      // ✅ 910B
this->AICore().AddConfig("ascend910_93");    // ✅ 910C
this->AICore().AddConfig("ascend950");      // ✅ 950PR/950DT (arch35)
// ❌ 没有 "ascend310p" (arch20/DAV_2002)
```

`ascend950` (arch35) 是 950PR/950DT，不是传统的 310P (arch20)。原始 310P 需要注册 `"ascend310p"` 配置。

##### Gap 2: PyTorch 绑定未在 310P 分支注册

**文件**: `csrc/torch_binding.cpp:2748-2807 vs 2808+`

```cpp
#ifdef ASCEND_PLATFORM_310P    // ← 310P 分支
    // 只注册了:
    //   get_physical_device_id
    //   npu_causal_conv1d_310
    //   npu_recurrent_gated_delta_rule_310
    //   chunk_gated_delta_rule_fwd_h
    //   chunk_fwd_o_vllm
    //   chunk_kda_fwd / kda_gate_cumsum / kda_layout_swap12
    // ❌ 没有 npu_hc_pre_v2 / npu_hc_pre_v3 / npu_hc_post
#else                          // ← 非 310P 分支
    // ✅ 注册了 npu_hc_pre_v2, npu_hc_pre_v3, npu_hc_post
    ops.def("npu_hc_pre_v2(...)");
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);
    ops.def("npu_hc_pre_v3(...)");
    ops.impl("npu_hc_pre_v3", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v3_npu);
    ops.def("npu_hc_post(...)");
    ops.impl("npu_hc_post", torch::kPrivateUse1, &vllm_ascend::npu_hc_post_npu);
#endif
```

310P 平台编译时，`ASCEND_PLATFORM_310P` 被定义，走第一个分支，**mHC 算子完全不会被注册**。

##### Gap 3: kernel 代码路径不兼容原始 310P (arch20)

`__DAV_C310__` 是 950PR/950DT (arch35) 的宏定义，**不是原始 310P (arch20/DAV_2002)**：

```cpp
// hc_pre.cpp
#if defined(__DAV_C310__)   // ← 这是 arch35 (950PR/950DT)，不是 arch20 (310P)
    // arch35 的实现，使用 arch35 专有的 Cube/Vector 指令
#else
    // A3 的实现，使用 lib/matmul_intf.h
#endif
```

原始 310P 是 arch20，既不走 `__DAV_C310__` 分支（那是 950 专用），也不走 A3 分支（那是 A3 专用）。**310P 没有任何代码路径**。

#### PyTorch fallback 路径分析

`patch/worker/patch_triton.py:229-317` 提供了 PyTorch fallback：

```python
# 310P 没有 Triton，HAS_TRITON=False
if not HAS_TRITON:
    # 用 mhc_pre_torch (纯 PyTorch) 替代 TileLang kernel
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch, mhc_post_torch

    def _mhc_pre_npu(self, residual, fn, hc_scale, hc_base, ...):
        post_mix, comb_mix, layer_input = _mhc_pre_torch(...)
        # TileLang 版本融合了 RMSNorm，PyTorch 版本没有，需要补上
        if norm_weight is not None:
            layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
        return post_mix, comb_mix, layer_input

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
```

这条路径**功能上可行**，但有问题：

| 问题 | 影响 |
|------|------|
| Sinkhorn 20 次循环 | PyTorch for 循环，每次 2 个算子（sum + div），20×2=40 个算子 launch |
| 无算子融合 | RMSNorm、Linear、sigmoid、Sinkhorn、坍缩全是独立算子 |
| ACLGraph 不可用 | 40 次循环的算子调度无法被图捕获 |
| 性能 | 相比 C++ 融合算子（1 次 launch），性能差 10-40 倍 |

#### 310P mHC 实现方案

| 方案 | 说明 | 工作量 | 性能 |
|------|------|--------|------|
| 方案 1 | 在 `torch_binding.cpp` 的 310P 分支注册 mHC 算子 | 高（需为 arch20 写 kernel） | 最优 |
| 方案 2 | 使用 PyTorch fallback (`patch_triton.py` 已有) | 低（确保 patch 正确加载） | 差（Sinkhorn 未融合） |
| 方案 3 | 为 310P (arch20) 开发专用 Ascend C mHC kernel | 最高（Ascend C + 编译 + 测试） | 最优 |

**推荐路径**：方案 2 → 方案 3（先验证精度，再优化性能）

#### mHC Gap 汇总

| # | 层级 | 现状 | 修改 |
|---|------|------|------|
| 1 | C++ kernel | `__DAV_C310__` 是 950PR(arch35)，310P(arch20) 无代码路径 | 需为 arch20 写 kernel 或用 PyTorch fallback |
| 2 | Op 定义 | `hc_pre_def.cpp` 只注册 ascend910b/910_93/950 | 需添加 `"ascend310p"` 配置 |
| 3 | PyTorch 绑定 | `torch_binding.cpp` 310P 分支不注册 mHC 算子 | 在 310P 分支添加 mHC 算子注册 |
| 4 | PyTorch fallback | `patch_triton.py` 已有 fallback，Sinkhorn 未融合 | 可直接使用，性能较差 |
| 5 | 310P model runner | 不支持 mHC 模型的多流 hidden_states | 需适配 `[num_tokens, hc_mult, hidden_size]` 形状 |
| 6 | 310P 平台 | `patch_mamba_config_310.py:42` 也阻断 MLA | mHC 和 MLA 阻断互相独立 |

**核心结论**：mHC 的 PyTorch fallback 路径已存在（`patch_triton.py`），功能上可以在 310P 上运行，但性能较差（Sinkhorn 20 次循环未融合）。要获得高性能需要为 310P (arch20) 开发专用 Ascend C kernel，但相比 MLA 的 gap，**mHC 的 gap 更小**——至少功能可用的 fallback 路径是存在的。

---

## 五、根因总结

### Xing4.0 在 310P 无法运行

```
Xing4.0-29B-A4B 在 310P 无法运行

根因链路:
  CANN 310P 缺少 MLA 专用算子 (npu_fused_infer_attention_score_v2 等)
       ↓
  所有上层框架无法实现高效 MLA decode attention
       ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ vLLM-Ascend:  代码硬阻断 (model_runner raise ValueError)    │
  │ MindIE-LLM:   官方支持列表标注 310P "不支持"                │
  │ MindIE Motor: 依赖 vLLM-Ascend，间接不支持                  │
  └─────────────────────────────────────────────────────────────┘

  MLA 的 decode 核心需求:
    score = Q_latent·K_latent^T + Q_rope·K_rope^T   (两部分分数相加)
    需要算子支持: paged attention + query_rope/key_rope 融合

  310P 可用的 _npu_paged_attention:
    只支持标准 Q·K^T，不支持 rope 分离拼接
    block_size × head_size ≤ 128×128 限制使 latent(512) 难以处理

  MLA 可能的实现路径:
    方案A (短期验证): 解压存储，放弃 MLA 内存优势，用标准 attention
    方案B (可用部署): latent attention + rope 预拼接，小 block_size
    方案C (高性能):   开发 310P 专用 Ascend C MLA 算子
    方案D (分阶段):   A→B→C 逐步推进

  mHC 的 gap 更小:
    PyTorch fallback 路径已存在 (patch_triton.py)
    功能上可在 310P 运行，但 Sinkhorn 20 次循环未融合
    高性能需要为 arch20 开发专用 Ascend C kernel

  模型适配 PR 状态:
    vLLM PR #57135: Open / Pending (有精度问题待修复)
    vLLM-Ascend: 无公开 PR
    910B 能跑 = 预构建镜像 (内部 fork 打包)
    所有平台上游主分支均不能直接跑 Xing4.0
```

### 各框架 MLA 支持矩阵

| 框架 | 910B (A2) | 910C (A3) | 310P |
|------|:---------:|:---------:|:----:|
| vLLM-Ascend | ✅ MLA | ✅ MLA | ❌ 代码硬阻断 |
| MindIE-LLM | ✅ MLA | ✅ MLA | ❌ 官方不支持 |
| MindIE Motor | ✅ (依赖 vLLM) | ✅ (依赖 vLLM) | ❌ (依赖 vLLM) |

### 各组件 Gap 优先级

| 组件 | Gap 严重度 | 核心阻断 | 可用 fallback |
|------|----------|---------|-------------|
| MLA | **P0 致命** | CANN 310P 缺少 FIA v2 算子 | 方案 A (解压存储，放弃压缩优势) |
| mHC | **P1 高** | C++ kernel 不支持 arch20 | PyTorch fallback (性能差) |
| MoE | P1 中 | sigmoid scoring 无优化路径 | 通用 fallback 路径可用 |
| MTP | P1 中 | MLA + MTP 组合阻断 | — |
| 模型 PR | **P0 致命** | vLLM PR 未合入 | 使用预构建镜像 |

**根因**：
1. CANN 310P 缺少 `npu_fused_infer_attention_score_v2` 等 MLA 专用算子，这影响了所有上层框架
2. vLLM PR #57135 未合入，所有平台的上游主分支均不支持 Xing4.0
3. mHC 的 PyTorch fallback 路径已存在，gap 比 MLA 小
