---
id: memory-and-checkpointing
title: 显存与激活重计算：四大块逐项估算与能否跑起来
module: 13-distributed-training
tags:
- 显存
- 激活重计算
- 序列并行
- 激活
- 估算
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 训练显存 = 模型 + 优化器 + 梯度 + 激活。激活是随 batch/seq/层数变化的"弹性"大头，激活重计算（梯度检查点）用约 33% 额外前向
  FLOPs 换显存大幅下降；序列并行/上下文并行把激活沿序列维切分。
source:
- https://arxiv.org/abs/2104.04473
questions:
- 训练显存四大块各自随什么变化？哪块是"弹性"的？
- full 重计算多算了多少 FLOPs？为什么不是翻倍？
- selective 重计算相比 full 省了什么、多了什么？为什么更常用？
- 序列并行和张量并行切的维度有什么不同？
- 给定 A100 80G + 7B + ZeRO-3，能不能跑、激活要不要重计算？
---

## 一句话结论

- **四大块**：参数(2B)、梯度(2B)、优化器状态(12B，AdamW)、**激活**（随 B·S·L 变化，最大弹性项）。
- **激活重计算（gradient checkpointing）**：前向不存中间激活，反向时重新算 → full 重计算多约 1/3 前向 FLOPs，但激活显存从 O(B·S·L) 降到 O(B·L)（只存每层输入）。
- **序列并行（Megatron SP）/ 上下文并行（Ring Attention）**：把激活沿序列维 S 切到多卡，进一步压低单卡激活。

## 原理推导

### 单卡显存四大块

设参数量 P、混合精度 fp16：

| 块 | 公式（每参数） | 7B 示例 |
|---|---|---|
| 参数 | 2 B | 14 GB |
| 梯度 | 2 B | 14 GB |
| 优化器(AdamW) | 12 B | 84 GB |
| 激活 | 见下 | 视 B/S 而定 |
| **合计** | **16 B + 激活** | **112 GB + 激活** |

→ 只模型状态就 112 GB，远超单卡；ZeRO/FSDP 解决前三块，**激活只能靠重计算/并行**。

### 激活估算（Transformer 单层，简化）

一层激活 ≈ `B · S · (34·d + 5·S·h)` 字节（fp16），其中 d 隐藏维、h 头数。主导项是 attention 的 `S²`（per-head 分数矩阵 `S²·h`）和 FFN 的 `S·d_ff`。对 7B（d=4096, h=32, d_ff≈11008）：

- 单层激活 ≈ `B·S·(34·4096 + 5·S·32)` ≈ `B·S·(139k + 160·S)` 字节。
- 取 B=1, S=4096, L=32：单层 ≈ 1·4096·(139k+655k)=3.24 MB ×32 ≈ **104 MB/卡**——看似不大，但 B=64、S=8192 时可达数 GB，高并发长序列下激活会反超模型状态。

### 激活重计算

- **Full recompute**：只存每层输入，反向时重跑该层前向。激活显存 ~ 降为 O(B·L·d)（与 S 无关），代价是 **额外一次前向 ≈ +33% 总 FLOPs**（不是翻倍，因为只重算 transformer block 内部）。
- **Selective recompute**：只重算显存贵的部分（如 attention 的 S² 矩阵、dropout mask），保留 LN/attention 输出 → 省大部分显存、只多 ~5~15% 计算，性价比最高。

### 序列并行 / 上下文并行

- **Megatron SP**：把 LayerNorm 和 Dropout 沿序列维切分（它们本就逐 token 独立），激活再降 `1/N`；配合 TP 使用，TP 切隐藏维、SP 切序列维。
- **Ring Attention / Context Parallel**：把长序列 S 切到 N 卡，attention 用分块 + 环交换 KV，使**单卡激活随 S/N 下降**，支持超长上下文训练。

## 具体例子

**给定：A100 80G，7B 模型（fp16），ZeRO-3（8 卡）+ full 重计算，B=4, S=4096。能跑吗？**

- 模型状态（ZeRO-3, 8 卡）：`16/8=2 B/param` ×7B ≈ 14 GB/卡。
- 激活（full 重计算，只存层输入）：≈ `B·L·d·2` × 少量 ≈ 4·32·4096·2 ≈ 1 MB 量级，可忽略。
- 其余临时/碎片：~10 GB。
- 合计 ≈ 14+1+10 ≈ **25 GB/卡 ≪ 80G** → **可以跑**。若关掉重计算（存全部激活）在 B=4,S=4096 下激活可达 ~10+ GB，仍然可跑，但 S 提到 32K 就会爆。

## 代码速记

```python
# 激活重计算（PyTorch）
from torch.utils.checkpoint import checkpoint
def block_forward(block, x):
    return checkpoint(block, x, use_reentrant=False)  # 反向时重算

# 估算一次 step 通信耗时（α-β）
def comm_time(M_bytes, N, a_us=5.0, bw_GBs=200.0):
    steps = 2 * (N - 1)
    return a_us * steps + (2 * M_bytes * (N-1) / N) / (bw_GBs * 1e3)  # ms
```

## 面试追问

1. 训练显存四大块各自随什么变化？哪块是"弹性"的？
2. full 重计算多算了多少 FLOPs？为什么不是翻倍？
3. selective 重计算相比 full 省了什么、多了什么？为什么更常用？
4. 序列并行和张量并行切的维度有什么不同？
5. 给定 A100 80G + 7B + ZeRO-3，能不能跑、激活要不要重计算？

## 关联知识点

- [ZeRO 与 FSDP：优化器状态、梯度与参数的分片](zero-and-fsdp)
- [并行策略：DP / TP / PP 与三维并行选型](parallel-strategies)
- [集合通信原语：all-reduce / all-gather 与 Ring 推导](comm-primitives)
