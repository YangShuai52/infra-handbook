---
id: flash-attention
title: FlashAttention：IO 复杂度分析与分块算法
module: 04-attention-long-context
tags:
- FlashAttention
- 显存优化
- IO复杂度
- kernel
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: FlashAttention 的核心不是减少 FLOPs，而是通过分块 + online softmax 把 HBM 读写次数从 O(S²) 降到
  O(S²d/M)，在 SRAM 内完成整个 attention，显存从 O(S²) 降到 O(S)。
source:
- https://arxiv.org/abs/2205.14135
questions:
- FlashAttention 减少了 FLOPs 吗？
- 为什么叫"精确"attention？和稀疏 attention 有什么区别？
- online softmax 的数值稳定性怎么保证？
- FlashAttention 在短序列（如 S=512）上还快吗？
- 反向传播为什么可以重算而不存 P？
---

## 一句话结论

FlashAttention 解决的问题不是"算得慢"，而是**"搬得慢"**。

标准 attention 需要把 S×S 的分数矩阵写回 HBM 再读出来做 softmax、再写回、再读出来乘 V。在 A100 上 HBM 带宽 1.5~2 TB/s，SRAM 带宽约 19 TB/s，差 10 倍以上；当 S 很大时，kernel 完全卡在访存上，Tensor Core 大部分时间在空转。

FlashAttention 的做法：**把 Q 分成块、K/V 分成块，每次只把一小块 load 进 SRAM，在 SRAM 内完成该块的 QKᵀ、softmax 修正、乘 V，最后只把输出块写回 HBM**。全程不物化 S×S 矩阵。

结果：

```
显存:  O(S²)  →  O(S)
HBM 访存: O(S²) → O(S²·d / M)      M = SRAM 大小
FLOPs: 略增（online softmax 的重缩放），但远小于访存收益
```

## 原理推导

### 1. 标准 attention 的 IO 复杂度

A100 参数：SRAM（每 SM 的 shared memory）约 192KB 全芯片合计约 20MB 量级可用；HBM 40/80GB，带宽 1.5~2 TB/s。

标准实现的步骤与 HBM 访存（忽略 batch/head，只算单头）：

| 步骤 | 操作 | HBM 读写量 |
|---|---|---|
| 1 | 读 Q,K 写 S=QKᵀ | O(Sd + Sd + S²) |
| 2 | 读 S 写 P=softmax(S) | O(S² + S²) |
| 3 | 读 P,V 写 O=PV | O(S² + Sd + Sd) |

总计 **O(S²)** 级别的 HBM 访问（系数约 4~5 个 S²）。

对比 FLOPs 是 O(S²d) = S² × 128（head_dim=128）。所以：

```
算术强度 (FLOPs / byte) ≈ S²·128 / (5·S²·2字节) = 12.8 FLOP/byte
```

而 A100 bf16 的机器平衡点约 `312e12 / 1.5e12 ≈ 208 FLOP/byte`。**12.8 远低于 208，说明标准 attention 是严重的 memory-bound**，算力利用率通常不到 10%。

### 2. 分块 + Online Softmax

难点：softmax 需要整行的全局信息（max 和 sum），分块后看不到后续块的值。

解决办法是维护**运行时的最大值和归一化因子**，用增量修正。设已处理块 1..i-1 得到：

```
m_{i-1} = 到目前为止的行最大值
ℓ_{i-1} = 到目前为止的 exp 之和（相对于 m_{i-1}）
O_{i-1} = 到目前为止的累加输出（未归一化）
```

处理第 i 块时：

```
S_i = Q_block · K_iᵀ                                  # (B_r, B_c)
m_i = max(m_{i-1}, rowmax(S_i))                       # 更新最大值
p_i = exp(S_i - m_i)                                  # 用新最大值归一化
ℓ_i = exp(m_{i-1} - m_i) · ℓ_{i-1} + rowsum(p_i)      # 修正旧的和
O_i = exp(m_{i-1} - m_i) · O_{i-1} + p_i · V_i        # 修正旧的输出
```

最后 `O = O_last / ℓ_last`。

**为什么正确**：设旧的部分和用旧基准 m_old，新的用 m_new，则

```
exp(s - m_old) = exp(s - m_new) · exp(m_new - m_old)
```

所以只要把旧累加值乘上 `exp(m_old - m_new)` 就换到了新基准。这就是那个修正因子。整个算法是数值稳定的，因为 `m` 始终是当前已见最大值，`exp` 的输入恒 ≤ 0。

### 3. 块大小的选择

SRAM 大小 M，需要同时放 Q 块、K 块、V 块、O 块、分数块：

```
B_r·d (Q) + B_c·d (K) + B_c·d (V) + B_r·d (O) + B_r·B_c (S) ≤ M
```

典型取 `B_c ≈ M/(4d)`，`B_r ≈ min(B_c, d)`。A100 上 d=128、M≈100KB（fp32 累加）时，B_c 约 128~256。

HBM 访存量（外层循环 Q 块，内层遍历所有 K 块）：

```
对每个 Q 块: 读所有 K,V = O(Sd) 次 → 共 S/B_r 个 Q 块
总访存 = O( (S/B_r) · Sd ) = O( S²d / B_r )
```

而标准实现是 O(S²)。当 `B_r ≈ 100` 量级、d=128 时，看起来比值不利，但实际因为标准实现的 S² 项要读写 4~5 次且是 fp32/bf16 混合，而 FlashAttention 的 S²d/B_r 中 B_r 可以到 256，**实测加速 2~4 倍**（S=2k~8k）。

### 4. 反向传播

标准实现需要存 S×S 的 P 矩阵给反向。FlashAttention **不存 P**，而是存 `O` 和统计量 `(m, ℓ)`，反向时**重算** S 和 P：

```
多了 O(S²d) 的 FLOPs（重算前向），但省了 O(S²) 的显存
```

这是典型的 **recompute 换显存**，在 memory-bound 场景下是划算的。

### 5. v1 / v2 / v3 的差异

| 版本 | 核心改进 | 效果 |
|---|---|---|
| v1 | 分块 + online softmax，IO 感知 | 显存 O(S)，相对标准实现 2~4× 加速 |
| v2 | 减少非 matmul FLOPs（少做 rescale）；Q 外层、KV 内层循环调换；并行化到 seq 维度（即使 batch=1 也能占满 SM）；warp 间分工优化 | 相对 v1 再快约 2×，A100 上达到 50~73% MFU |
| v3 | 针对 Hopper：FP8 低精度（新硬件的 FP8 Tensor Core）、TMA 异步拷贝、warp specialization（生产者-消费者分离）、overlap softmax 与 GEMM | H100 上 75%+ MFU，FP8 下接近 1.5 PFLOPS |

## 具体例子

**例 1：显存节省的定量对比**

B=8, H=32, S=8192, d_h=128, bf16：

```
标准实现的分数矩阵:  8 × 32 × 8192² × 2 bytes = 34.4 GB
FlashAttention:      只需 (B_r × B_c) 的分数块 ≈ 128 × 128 × 2 = 32 KB（每 SM）
                     加上输出 O: 8 × 32 × 8192 × 128 × 2 = 0.5 GB
```

**从 34.4GB 降到 0.5GB**，这就是"FlashAttention 让长上下文成为可能"的定量含义。

**例 2：算术强度对比**

S=4096, d_h=128, bf16（2 字节）：

```
标准: FLOPs = 4·S²·d = 4 × 4096² × 128 = 8.59e9
      访存  ≈ 5 × S² × 2 = 1.68e8 bytes
      算术强度 = 8.59e9 / 1.68e8 = 51 FLOP/byte

Flash(B_c=256): 访存 ≈ S²·d/B_c × 2 = 4096²×128/256 × 2 = 1.68e7 bytes
      算术强度 = 8.59e9 / 1.68e7 = 512 FLOP/byte
```

A100 机器平衡点约 208 FLOP/byte → 标准实现 51 < 208（memory bound），FlashAttention 512 > 208（compute bound）。**这就是加速的本质：把 kernel 从带宽受限推到算力受限**。

**例 3：一次 decode 步的分析**

decode 时 S_q = 1，所以 attention 本来就不需要 S×S 矩阵，FlashAttention 的收益主要来自**减少 kernel 启动和 KV 读取次数**（FlashDecoding 沿 KV 长度切分并行）。这时候瓶颈是"读 KV Cache 的带宽"：

```
7B 模型, S=8192: KV 大小 = 2 × 8192 × 32 × 32 × 128 × 2 = 4.3 GB
每生成一个 token 要读 4.3 GB
A100 带宽 1.5 TB/s → 理论最快 2.9 ms/token → 约 345 token/s（单请求上限）
```

这也是为什么 GQA/MLA/量化在 decode 阶段收益巨大——它们直接削减这 4.3GB。

## 代码速记

```python
import torch

def flash_attention_forward(Q, K, V, block_q=64, block_kv=64):
    """
    Q,K,V: (S, d) 单头，fp32。模拟 FlashAttention 的分块 + online softmax。
    显存占用 O(S*d)，不物化 S×S 矩阵。
    """
    S, d = Q.shape
    O = torch.zeros(S, d)
    m = torch.full((S, 1), -torch.inf)      # 运行最大值
    l = torch.zeros(S, 1)                   # 运行 exp 之和
    for i in range(0, S, block_q):
        q_blk = Q[i:i+block_q]              # (B_r, d)
        m_i, l_i = m[i:i+block_q], l[i:i+block_q]
        o_i = torch.zeros_like(q_blk)
        for j in range(0, i + block_q, block_kv):     # causal: 只到对角线
            k_blk, v_blk = K[j:j+block_kv], V[j:j+block_kv]
            s = q_blk @ k_blk.T / (d ** 0.5)          # (B_r, B_c)
            # causal 掩码处理
            if j + block_kv > i:
                rows = torch.arange(i, i + q_blk.shape[0]).unsqueeze(1)
                cols = torch.arange(j, j + k_blk.shape[0]).unsqueeze(0)
                s = s.masked_fill(cols > rows, -torch.inf)
            m_new = torch.maximum(m_i, s.max(dim=1, keepdim=True).values)
            m_new = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_new)
            corr = torch.exp(m_i - m_new)
            l_i = corr * l_i + p.sum(dim=1, keepdim=True)
            o_i = corr * o_i + p @ v_blk
            m_i = m_new
        O[i:i+block_q] = o_i / l_i
        m[i:i+block_q], l[i:i+block_q] = m_i, l_i
    return O

# 正确性验证：与标准实现对比
S, d = 512, 64
Q, K, V = torch.randn(S, d), torch.randn(S, d), torch.randn(S, d)
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
ref = torch.softmax(Q @ K.T / d**0.5 + mask.float() * -1e9, -1) @ V
out = flash_attention_forward(Q, K, V)
print("max abs diff:", (ref - out).abs().max().item())   # ~1e-6
```

## 面试追问

**Q：FlashAttention 减少了 FLOPs 吗？**
A：几乎没有，甚至略微增加（online softmax 的重缩放、反向的重算）。它优化的是 **HBM 访存次数**。很多人把"FlashAttention 更快"归因于"算法改进"，准确说法是"**IO 感知的精确算法**"——数学结果和标准 attention 完全等价（不是近似），只是访存模式不同。

**Q：为什么叫"精确"attention？和稀疏 attention 有什么区别？**
A：FlashAttention 计算的是完整的、数学上等价的 softmax attention，没有任何近似。稀疏/线性 attention 改变的是**计算什么**（少算一些分数或用核函数近似），FlashAttention 改变的是**怎么算**（同样的计算，但访存更少）。所以 FlashAttention 可以和各种稀疏/高效注意力正交叠加。

**Q：online softmax 的数值稳定性怎么保证？**
A：关键是维护运行最大值 m，使得 `exp` 的输入恒为 `s - m ≤ 0`，因此 `exp(s-m) ∈ (0, 1]`，绝不会溢出。而修正因子 `exp(m_old - m_new) ≤ 1`（因为 m 单调不减），所以旧值只会被缩小，不会放大，也不会溢出。边界情况：第一个块时 m 初值是 -inf，`exp(-inf - s) = 0`，修正因子为 0，正好不影响——实现上要把 -inf 特判掉（代码里用 `torch.where` 处理）。

**Q：FlashAttention 在短序列（如 S=512）上还快吗？**
A：收益很小，甚至可能更慢。因为 S² 项不大时，标准实现的 S×S 矩阵也能放进缓存，而 FlashAttention 的分块循环有额外开销和更多 kernel 启动。经验上 **S ≥ 1024 才明显收益，S ≥ 2048 收益显著**。短序列的推理（尤其是 decode 阶段 S_q=1）要用 FlashDecoding 而不是标准的 FlashAttention。

**Q：反向传播为什么可以重算而不存 P？**
A：因为 P 可以由 Q、K 和统计量 m 重算得到（`p = exp(QKᵀ - m)`），而 m 只有 S 个元素（每块一个标量）。重算的代价是额外的 `O(S²d)` FLOPs，但省下 `O(S²)` 显存。在 memory-bound 场景（大模型训练基本都是）算力有余、显存紧张，这笔交易很划算——和 activation checkpointing 是同一个思想。

**Q：v2 说的"减少非 matmul FLOPs"具体是什么？**
A：v1 在每个块都做 `O_i = corr·O_i + p_i·V_i`，即对整个输出块做 rescale，这是逐元素的非矩阵乘运算，不能打 Tensor Core。v2 推迟 rescale——只在块的最后做一次归一化，中间累加不做修正（改为在最后统一用 `exp(m_last - m_i)` 修正），把非 matmul 操作从 O(B_r·d·块数) 降到 O(B_r·d)。另外 v2 交换了循环顺序（Q 在外、KV 在内变成更适合并行的形式），并把序列维度也并行化（v1 只在 batch 和 head 维度并行，长序列 batch=1 时 SM 利用率低）。

## 关联知识点
- Scaled Dot-Product Attention 数学推导（01-transformer）
- KV Cache 与 prefill/decode 瓶颈（02-attention）
- PagedAttention 与 vLLM（02-attention）
- 稀疏与线性注意力、Mamba（02-attention）
- 激活重计算与显存估算（04-distributed）
