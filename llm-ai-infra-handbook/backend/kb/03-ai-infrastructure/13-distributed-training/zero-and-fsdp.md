---
id: zero-and-fsdp
title: ZeRO 与 FSDP：优化器状态、梯度与参数的分片
module: 13-distributed-training
tags:
- ZeRO
- FSDP
- 显存
- 优化器状态
- 数据并行
- offload
difficulty: 4
freq: 高
updated: 2026-08-28
summary: ZeRO 把"模型状态"（优化器状态/梯度/参数）沿数据并行维度分片，从 DDP 每卡 16 bytes/param 降到 ZeRO-3 的 16/N
  bytes/param。FSDP 是 PyTorch 对 ZeRO-3 的工程实现（flat parameter + 分片策略 + CPU/NVMe offload）。
source:
- https://arxiv.org/abs/2104.04473
questions:
- 为什么 AdamW 的优化器状态是 12 字节而不是 8 字节？
- ZeRO-2 和 ZeRO-1 相比多分片了什么？为什么 ZeRO-2 不常用 ZeRO-1 省下的那点？
- ZeRO-3 通信量为什么是 DDP 的约 1.5 倍？
- FSDP 的 flat parameter 为什么要"打平"再分片？（减少小 tensor all-gather 开销）
- CPU offload 省了显存但付出了什么？
---

## 一句话结论

- **AdamW 每参数 12 字节优化器状态**：fp32 主权重 4B + 一阶动量 4B + 二阶动量 4B。这是显存大头。
- **ZeRO-1** 只分片优化器状态；**ZeRO-2** 再加梯度；**ZeRO-3** 再加参数。显存随 GPU 数 N 线性下降。
- **FSDP ≈ ZeRO-3**：`FULL_SHARD`=ZeRO-3、`SHARD_GRAD_OP`=ZeRO-2、`NO_SHARD`=DDP；offload 把状态卸到 CPU/NVMe 换时延。

## 原理推导

### 单卡显存分解（以 7.5B、fp16 参数/梯度、AdamW 为例）

| 部分 | 每参数字节 | 说明 |
|---|---|---|
| 参数（fp16） | 2 | 训练时真正参与计算的权重 |
| 梯度（fp16） | 2 | 反向产出 |
| 优化器状态（fp32） | 12 | 4+4+4（主权重/动量/方差） |

→ **DDP 每卡固定 16 bytes/param**，7.5B 模型 = 120 GB/卡，单张 80G 根本放不下。

### ZeRO 各阶段单卡显存（N=64 卡）

把 12 字节优化器状态、2 字节梯度、2 字节参数按比例分片：

| 阶段 | 参数 | 梯度 | 优化器 | 合计 bytes/param | 7.5B 单卡 |
|---|---|---|---|---|---|
| Baseline (DDP) | 2 | 2 | 12 | 16.0 | 120 GB |
| ZeRO-1 | 2 | 2 | 12/64 | 4.19 | 31.4 GB |
| ZeRO-2 | 2 | 2/64 | 12/64 | 2.22 | 16.6 GB |
| ZeRO-3 | 2/64 | 2/64 | 12/64 | 0.25 | 1.88 GB |

> 注：ZeRO-3 在每层前向/反向前需 **all-gather** 该层参数、反向后 **reduce-scatter** 梯度，瞬时峰值略高，但常驻显存如上表。

### 通信量

- DDP：每个 step 一次 `all-reduce(梯度)`，体积 = 2 bytes/param（fp16 梯度）× P。
- ZeRO-3：参数 `all-gather`（前向+反向各一次 ≈ 4P 体积）+ 梯度 `reduce-scatter`（2P）→ 约 **1.5× DDP** 的通信量，换来显存从 16 降到 0.25 bytes/param。

## 具体例子

- **7.5B 模型想塞进单张 A100 80G**：用 ZeRO-3（FSDP FULL_SHARD）在 8 卡上，每卡 ≈ 1.88 GB 模型状态；加上激活 ≈ 可跑，但需激活重计算（见同模块"显存与激活重计算"）。
- **FSDP offload**：把优化器状态卸到 CPU（`cpu_offload=True`），单卡可训远超显存的大模型，但每步要把状态 CPU↔GPU 搬运，训练慢数倍。

## 代码速记

```python
# PyTorch FSDP 分片策略对照 ZeRO
from torch.distributed.fsdp import ShardingStrategy
FULL_SHARD      # ZeRO-3：参数+梯度+优化器都分片
SHARD_GRAD_OP   # ZeRO-2：梯度+优化器分片，参数全量（每层内分片）
NO_SHARD        # ZeRO-0/DDP：不分片（等价于 DDP）

# 典型写法
model = FSDP(model, sharding_strategy=FULL_SHARD,
             cpu_offload=CPUOffload(offload_params=True))  # 卸到 CPU
```

## 面试追问

1. 为什么 AdamW 的优化器状态是 12 字节而不是 8 字节？
2. ZeRO-2 和 ZeRO-1 相比多分片了什么？为什么 ZeRO-2 不常用 ZeRO-1 省下的那点？
3. ZeRO-3 通信量为什么是 DDP 的约 1.5 倍？
4. FSDP 的 flat parameter 为什么要"打平"再分片？（减少小 tensor all-gather 开销）
5. CPU offload 省了显存但付出了什么？

## 关联知识点

- [并行策略：DP / TP / PP 与三维并行选型](parallel-strategies)
- [集合通信原语：all-reduce / all-gather 与 Ring 推导](comm-primitives)
- [显存与激活重计算：四大块逐项估算](memory-and-checkpointing)
