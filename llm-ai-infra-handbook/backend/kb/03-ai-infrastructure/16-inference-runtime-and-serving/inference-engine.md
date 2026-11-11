---
id: inference-engine
title: 推理引擎：Continuous Batching、PD 分离与 vLLM/SGLang 对比
module: 16-inference-runtime-serving
tags:
- 推理引擎
- Continuous Batching
- PagedAttention
- PD分离
- vLLM
- SGLang
- prefix cache
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 推理引擎的目标是把 GPU 算力和显存"喂满"。Continuous Batching 在 token 级动态调度避免静态 batch 的空等；PD
  分离把算力密集的 prefill 与带宽密集的 decode 拆到不同实例；prefix cache(KV 复用) 省重复计算。主流：vLLM / SGLang
  / TensorRT-LLM / LMDeploy。
source:
- https://arxiv.org/abs/2309.06180
questions:
- Continuous Batching 相比静态 batch 解决了什么？
- PagedAttention 为什么能省显存、能共享前缀？
- 为什么要把 prefill 和 decode 分离部署？
- TTFT / TPOT / ITL 分别由什么决定？
- vLLM / SGLang / TRT-LLM 怎么选？
---

## 一句话结论

- **Continuous Batching（迭代级调度）**：不等一个序列生成完再换下一批，而是**每生成一个 token 就动态增删序列**，GPU 利用率从 ~10% 提到 ~90%。
- **Prefill vs Decode**：prefill 是 compute-bound（一次算全 prompt 的 KV），decode 是 memory-bound（每步读全权重）。**PD 分离**让两者各用最适硬件/并行策略。
- **Prefix / RadixAttention**：相同前缀（system prompt、few-shot）的 KV Cache 跨请求复用，避免重复 prefill。
- **核心指标**：TTFT（首 token 延迟）、TPOT（每 token 延迟）、ITL（token 间延迟）、吞吐（tokens/s）。

## 原理推导

### 静态 Batching 的问题

传统做法把请求 pad 到同长、成批处理，直到**整批全结束**才接新请求 → 短请求早结束也得等长的，GPU 大量空转（bubble）。Orca 提出 **iteration-level scheduling**：每个 decoding step 后，结束的序列移出、新序列补入，调度粒度 = 1 个 token。

### PagedAttention（vLLM）

把 KV Cache 像操作系统**虚拟内存**一样分页（block），逻辑上连续、物理上不连续：
- **显存零浪费**：不再为最大长度预留连续块，按需分配页，碎片由 block table 管理。
- **Copy-on-write / 共享**：多请求共享同一前缀（parallel sampling、beam search、共享 system prompt）时只存一份 KV，按引用计数共享页。
- **块大小**：小→碎片少利用率高但 block table 大；大→反之，典型 16。

### PD 分离（Disaggregated）

- **Prefill 实例**：大 batch、高并行（TP 大）、吃算力，追求高吞吐产 KV。
- **Decode 实例**：小 batch、memory-bound、追求低 TPOT，KV 通过高速网络从 prefill 搬过来。
- 好处：互不干扰，prefill 的突发算力需求不打断 decode 的平稳低延迟。

### 关键 CUDA 优化常识

- **算子融合（fused kernel）**：RMSNorm+QKV proj 合并、RoPE 融合进 attention，减少 kernel 启动与 HBM 读写。
- **FlashDecoding**：decode 时对长 KV 做分块并行（类似 FlashAttention 的分块 + 在线 softmax），把 memory-bound 的注意力在 KV 维并行。
- **Tensor Core 与 K 维对齐**：矩阵乘的 K 维需是 8/16 倍数才能吃满 Tensor Core。
- **memory coalescing / shared memory / warp**：优化数据访问模式，避免 bank conflict。

## 具体例子

| 引擎 | 特点 |
|---|---|
| **vLLM** | PagedAttention 开创者，生态成熟，吞吐高，prefix cache 完善 |
| **SGLang** | RadixAttention（前缀树复用 KV）、结构化生成快、支持 PD 分离 |
| **TensorRT-LLM** | NVIDIA 官方，极致 kernel 优化，延迟最低，但编译/部署重 |
| **LMDeploy** | 轻量、量化友好（AWQ 原生）、TurboMind 引擎，国产卡支持好 |

**指标场景**：对话机器人看 TPOT（要 <50ms 才"跟手"）；批量摘要看吞吐；长文档首答看 TTFT（prefill 决定）。

## 代码速记

```python
# 伪：iteration-level 调度循环
while pending or running:
    batch = scheduler.step()                 # 每步动态凑批（增删序列）
    logits = model(batch)                    # prefill/decode 混合批
    for seq in batch:
        if seq.finished: running.remove(seq)
        else: seq.append(sample(logits[seq]))
    if new_request_arrived: pending.append(new_request)   # 不等整批结束
```

## 面试追问

1. Continuous Batching 相比静态 batch 解决了什么？
2. PagedAttention 为什么能省显存、能共享前缀？
3. 为什么要把 prefill 和 decode 分离部署？
4. TTFT / TPOT / ITL 分别由什么决定？
5. vLLM / SGLang / TRT-LLM 怎么选？

## 关联知识点

- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
- [PagedAttention 与高效注意力变体](paged-attention)
- [投机解码：draft-then-verify 与加速比推导](speculative-decoding)
