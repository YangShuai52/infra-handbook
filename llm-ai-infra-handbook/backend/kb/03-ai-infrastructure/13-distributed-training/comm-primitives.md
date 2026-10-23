---
id: comm-primitives
title: 集合通信原语：all-reduce / all-gather / reduce-scatter 与 Ring 推导
module: 13-distributed-training
tags:
- 集合通信
- all-reduce
- Ring
- NCCL
- 通信量
- bandwidth
difficulty: 3
freq: 中
updated: 2026-08-28
summary: 分布式训练的通信建立在四个集合原语之上。Ring All-Reduce 把朴素 Parameter Server 的 (N-1)M 通信量压到带宽最优的
  2M(N-1)/N，是理解"训练 step 通信耗时"的核心。
source:
- https://arxiv.org/abs/2104.04473
questions:
- Ring All-Reduce 的总通信量为什么是 2M(N-1)/N？分哪两个阶段？
- 为什么 Ring 比 PS 在大规模下更优？
- α-β 模型里，节点内和跨节点分别由谁主导？
- all-to-all 在什么场景出现？为什么 MoE 通信最重？
- 为什么要在反向时做 reduce-scatter 而不是 all-reduce（ZeRO 的选择）？
---

## 一句话结论

- 四个原语：**all-reduce**（各卡求和后广播）、**reduce-scatter**（先求和再按块分散）、**all-gather**（把分散块拼成完整）、**all-to-all**（MoE 的 token 重排）。
- **Ring All-Reduce** 带宽最优：总传输量 = `2M·(N-1)/N`，当 N 大时趋近 `2M`（M 为模型参数量字节）。
- 通信耗时用 **α-β 模型**估算：`T ≈ α·(跳数) + β·(字节数)`，延迟 α 和带宽 β 由拓扑（NVLink/IB）决定。

## 原理推导

### 朴素 PS 的瓶颈

中心化 Parameter Server：N 个 worker 各把梯度（M 字节）发给 PS 再收回更新 → 单卡 `2(N-1)M`，且 PS 成为带宽瓶颈。

### Ring All-Reduce

把 M 字节切成 N 段。Ring 上每步：节点 i 把第 `i` 段发给 i+1、同时接收 i-1 段并做 **in-place 累加**。

- **Reduce-Scatter 阶段**（N-1 步）：每步传 M/N 字节，结束后第 k 个节点持有"第 k 段"的全局求和结果。
- **All-Gather 阶段**（N-1 步）：再把各自持有的已求和段沿环广播，每步 M/N 字节。

总传输量（每卡）`= 2·(N-1)·(M/N) = 2M·(N-1)/N`。对比 PS 的 `2(N-1)M`：当 N 很大，Ring 把"乘以 N"变成"约 2 倍常数"，**带宽利用率随 N 提升**。

### α-β 模型

```
T_comm ≈ α · steps + β · bytes
```
- α：单跳延迟（节点内 NVLink ~1μs，跨节点 IB ~数 μs~数十 μs）
- β：1/带宽（A100 NVLink ~600 GB/s，IB ~50~200 GB/s）
- Ring 步数 = 2(N-1)，bytes = 2M(N-1)/N

→ 跨节点时 α 主导、应尽量**计算-通信重叠**；节点内 β 主导、要压数据量。

## 具体例子

- 7B 模型 fp16 梯度 M ≈ 14 GB。8 卡 Ring All-Reduce：理论传输 `2·14·7/8 ≈ 24.5 GB/卡`。在 NVLink 600 GB/s 下约 41 ms；若走 IB 50 GB/s 约 490 ms → 这就是为什么同机多卡要 NVLink。
- MoE 的 **all-to-all**：每个 token 按路由发到专家所在卡，是通信量最大的环节，需要 EP（专家并行）+ 通信重叠。

## 代码速记

```python
import torch.distributed as dist
# 四原语（每个都阻塞到完成）
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)      # 各卡求和后每卡都有完整和
dist.reduce_scatter(tensor, op=dist.ReduceOp.SUM)  # 求和后按 rank 分散一段
dist.all_gather(list_of_tensors, tensor)           # 各卡把本地段拼成完整
dist.all_to_all(list_out, list_in)                 # MoE token 重排
```

## 面试追问

1. Ring All-Reduce 的总通信量为什么是 2M(N-1)/N？分哪两个阶段？
2. 为什么 Ring 比 PS 在大规模下更优？
3. α-β 模型里，节点内和跨节点分别由谁主导？
4. all-to-all 在什么场景出现？为什么 MoE 通信最重？
5. 为什么要在反向时做 reduce-scatter 而不是 all-reduce（ZeRO 的选择）？

## 关联知识点

- [ZeRO 与 FSDP：优化器状态、梯度与参数的分片](zero-and-fsdp)
- [并行策略：DP / TP / PP 与三维并行选型](parallel-strategies)
- [显存与激活重计算：四大块逐项估算](memory-and-checkpointing)
