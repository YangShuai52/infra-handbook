---
id: kv-cache
title: KV Cache：原理、显存公式与两阶段瓶颈
module: 15-inference-algorithms
tags:
- KV Cache
- 推理
- 显存
- prefill
- decode
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: KV Cache 用空间换时间把 decode 从 O(S²) 降到 O(S)，显存公式 2·B·S·L·n_kv·d_h；prefill 是算力瓶颈、decode
  是带宽瓶颈，两者的优化手段完全不同。
source:
- https://arxiv.org/abs/2211.17192
questions:
- KV Cache 会让精度下降吗？
- 为什么不能只缓存 K 不缓存 V？
- batch 推理时不同长度怎么管理 KV Cache？
- Chunked Prefill 是什么，为什么需要？
- 多轮对话的 KV Cache 怎么复用？
---

## 一句话结论

KV Cache 是**自回归推理最重要的一次空间换时间**：缓存已计算的 K、V，让每步只算新 token 的 attention，把单步复杂度从 O(S²) 降到 O(S)。

代价是显存随 `B × S` 线性增长，长上下文高并发时 KV Cache 会超过模型权重成为第一大显存占用。所以：

```
prefill 阶段（处理输入 prompt）：算力瓶颈，优化目标是打满 Tensor Core
decode 阶段（逐 token 生成）  ：带宽瓶颈，优化目标是少读 KV Cache
```

**这两个阶段的优化手段几乎不重叠**，混淆它们是推理优化类问题最常犯的错误。

## 原理推导

### 1. 为什么需要 KV Cache

第 t 步生成时，query 是 `q_t`（只依赖第 t 个 token），但 attention 需要所有历史 K、V：

```
Attention(q_t, K_{1:t}, V_{1:t}) = softmax(q_t · K_{1:t}ᵀ / √d_h) · V_{1:t}
```

关键在于 `k_i = x_i · W_k`、`v_i = x_i · W_v` **只依赖第 i 个 token 及其之前的信息**，与后续生成无关，所以算一次就能永久复用。

不缓存的代价（生成第 t 个 token）：

```
需要重算 K_{1:t}, V_{1:t}:  FLOPs = 2 × t × d × 2 × n_kv × d_h   （K 和 V 两个投影）
加上 attention 本身:         4 × t × d
```

对所有 t = 1..S 求和 → **O(S²)**。

缓存后：

```
每步只需算 1 个 token 的 q,k,v:    FLOPs = 2 × d × 3 × n_kv·d_h（与 S 无关）
每步 attention:                    4 × S × d（读一遍 KV）
总计 O(S) 每步，O(S²) 对全序列（但常数极小，且是读不是算）
```

**注意**：即使有 KV Cache，生成整个序列的总复杂度仍是 O(S²)（每步读 S 个 KV，共 S 步），但**常数从"重算所有投影矩阵"降到"读一遍缓存"**，实际快 10~100 倍。

### 2. 显存公式

每层每 token 要缓存 K 和 V 两个向量，每个维度 `n_kv × d_h`：

```
每层每 token 元素数 = 2 × n_kv × d_h
每 token 总元素数   = 2 × n_kv × d_h × L
全序列全 batch      = 2 × B × S × L × n_kv × d_h × bytes
```

用模型配置重写（MHA 时 n_kv·d_h ≈ d）：

```
KV Cache ≈ 2 × B × S × L × d × bytes = 4 × B × S × N_bytes   （N ≈ L·d²·4 时粗略换算）
```

即 **KV Cache ≈ 4 × batch × seq_len × 模型权重的字节数**（对 MHA 且 d_ff ≈ 8d/3 的配置，这是很好用的心算公式）。

### 3. 两阶段的算术强度分析

**Prefill**：一次处理 S 个 token，类似训练的前向。

```
FLOPs ≈ 2·N·S           （N = 模型参数量）
访存  ≈ 2·N + KV写入
算术强度 ≈ 2·N·S / 2·N = S   → S ≥ 208 时就 compute bound（A100 bf16）
```

结论：**prefill 在任意合理长度下都是算力瓶颈**，要用 FlashAttention、大 batch、chunked prefill 打满算力。

**Decode**：每步只算 1 个 token，但要读全部 KV Cache 和全部权重。

```
FLOPs ≈ 2·N             （每个 token 过一遍所有参数）
访存  ≈ 2·N (权重) + KV Cache 大小
算术强度 ≈ 2·N / (2·N + KV) < 1   → 极端 memory bound
```

结论：**decode 的瓶颈是读权重和读 KV Cache**，优化手段是：

1. 减少 KV Cache 大小 → GQA / MLA / KV 量化 / 窗口注意力
2. 减少权重读取 → 权重量化（INT4/FP8）
3. 提高 batch 让一次读权重服务多个请求 → continuous batching
4. 一次多算几个 token → 投机解码 / 多 token 预测

### 4. KV Cache 优化手段汇总

| 手段 | 原理 | 压缩比 | 质量影响 |
|---|---|---|---|
| GQA / MQA | 减少 KV 头数 | 4~32× | 极小 |
| MLA | 低秩压缩 + 矩阵吸收 | 50~100× | 极小 |
| KV 量化 | FP8 / INT8 / INT4 存 KV | 2~4× | 小（需异常值处理） |
| 滑动窗口 | 只保留最近 W 个 token | S/W | 长依赖任务受损 |
|  StreamingLLM | 保留 sink + 最近窗口 | S/(W+k) | 中等，适合流式 |
| PagedAttention | 消除碎片和预留浪费 | 1.2~4×（实际利用率） | 无 |
| Prefix Caching | 跨请求共享相同前缀 | 视重复率 | 无 |
| 卸载到 CPU/SSD | 换介质 | 显存→内存 | 延迟上升 |

## 具体例子

**例 1：完整算一遍 7B 的 KV Cache**

LLaMA-2-7B: L=32, d=4096, n_h=n_kv=32, d_h=128，bf16。

```
每层每 token:  2 × 32 × 128 = 8192 个元素 = 16384 字节 = 16 KB
32 层:        16 KB × 32 = 512 KB / token
```

| 场景 | token 数 | KV Cache |
|---|---|---|
| 单条 4k 对话 | 4096 | 2.0 GB |
| 单条 32k 长文 | 32768 | 16.0 GB |
| 32 并发 × 8k | 262144 | 128.0 GB |
| 128 并发 × 4k（GQA g=8） | 524288 | 64.0 GB |

对比权重 13.5GB：**32 并发 8k 时 KV Cache 是权重的 9.5 倍**。这就是为什么服务端必须做 KV 优化。

**例 2：decode 速度的理论上限**

7B，batch=1，S=8192，MHA，A100（HBM 1.5 TB/s 实际约 1.2 TB/s）：

```
每步要读: 权重 13.5 GB + KV 4.3 GB = 17.8 GB
理论最快: 17.8 / 1200 GB/s = 14.8 ms/token → 67 token/s
```

这个数字就是**单请求 8k 上下文的速度天花板**，再怎么优化 kernel 也过不去。要提升只能：

```
用 GQA(g=8):  KV → 1.1 GB, 总读 14.6 GB → 82 token/s（+22%）
用 FP8 权重:  权重 → 6.75 GB, 总读 7.85 GB → 153 token/s（+128%）
两者都用:     6.75 + 0.54 = 7.3 GB → 164 token/s
```

注意 batch 增大时，权重读取被分摊，只有 KV 读取随 batch 增长，所以**吞吐（总 token/s）随 batch 增大而提升，但单请求延迟不变**。

**例 3：prefill 与 decode 的时间对比**

7B，S_prompt = 2048，S_output = 512，A100 40% MFU：

```
Prefill FLOPs = 2 × 6.74e9 × 2048 = 2.76e13
Prefill 时间  = 2.76e13 / (312e12 × 0.4) = 0.22 s   → TTFT ≈ 220 ms

Decode 每步读 ≈ 13.5 GB(权重) + 0.5 GB(KV) = 14 GB
Decode 每步时间 = 14 / 1200 = 11.7 ms
512 步 = 6.0 s    → TPOT ≈ 11.7 ms/token
```

**结论：生成 512 个 token 的时间里，96.5% 花在 decode**。所以"生成慢"几乎总是 decode 的问题，优化 TTFT（prefill）对整体体验的改善有限。

## 代码速记

```python
import torch, torch.nn as nn, torch.nn.functional as F

class KVCache:
    """最简 KV Cache：预分配 + 指针写入，避免每步 cat"""
    def __init__(self, max_len, n_kv, d_h, n_layers, dtype=torch.bfloat16, device="cuda"):
        # (layers, 2, 1, n_kv, max_len, d_h)，batch=1 简化
        self.buf = torch.zeros(n_layers, 2, 1, n_kv, max_len, d_h,
                               dtype=dtype, device=device)
        self.pos = 0

    def update(self, layer_idx, k, v):
        """k, v: (1, n_kv, S_new, d_h)"""
        S_new = k.shape[2]
        self.buf[layer_idx, 0, :, :, self.pos:self.pos+S_new] = k
        self.buf[layer_idx, 1, :, :, self.pos:self.pos+S_new] = v
        return self.buf[layer_idx, 0, :, :, :self.pos+S_new], \
               self.buf[layer_idx, 1, :, :, :self.pos+S_new]

    def advance(self, n): self.pos += n

def decode_step(model, cache, token_id, pos):
    """单步 decode：只算 1 个 token，复用缓存"""
    x = model.embed(token_id)                       # (1, 1, d)
    for i, layer in enumerate(model.layers):
        q = layer.q(x)                              # (1, n_h, 1, d_h)
        k, v = layer.k(x), layer.v(x)
        k_full, v_full = cache.update(i, k, v)
        out = F.scaled_dot_product_attention(q, k_full, v_full)  # 无 causal mask
        x = x + layer.o(out)
        x = x + layer.ffn(layer.n2(x))
    cache.advance(1)
    return model.head(model.norm(x))

def kv_cache_report(L, d, n_h, n_kv, d_h, B, S, bytes_per=2):
    total = 2 * B * S * L * n_kv * d_h * bytes_per
    return {
        "per_token_KB": 2 * L * n_kv * d_h * bytes_per / 1024,
        "total_GB": total / 1024**3,
        "weight_GB_est": 13.5,
        "kv_over_weight": total / 1024**3 / 13.5,
    }

print(kv_cache_report(32, 4096, 32, 32, 128, 1, 4096))
# {'per_token_KB': 512.0, 'total_GB': 2.0, 'kv_over_weight': 0.148}
```

## 面试追问

**Q：KV Cache 会让精度下降吗？**
A：数学上不会（缓存的 K/V 和重算的完全等价），但工程上会：缓存通常用 bf16/fp16 甚至 FP8/INT8 存储，比计算精度低；而且长序列累积的舍入误差虽小但存在。实测在 8k 以内影响可忽略，超长上下文（100k+）配合低精度 KV 需要谨慎评估。

**Q：为什么不能只缓存 K 不缓存 V？**
A：V 也可以由缓存重建吗？不行——V 依赖输入 x_i，而 x_i 是前一层的输出，要重算 V 就得从第一层重算整个网络，代价是 O(S) 层的前向，等于回到不缓存。而 K 同理。所以 K 和 V 必须一起缓存。反过来，Q 不需要缓存是因为每步只需要当前 token 的 q，历史 q 从不参与计算。

**Q：batch 推理时不同长度怎么管理 KV Cache？**
A：这是 PagedAttention 的核心动机。朴素做法是"按最大长度预分配"，浪费率可达 60%~80%（短请求占了长请求的坑）。PagedAttention 把缓存切成固定大小的 block（如 16 个 token），按需分配，消除内部碎片；配合 copy-on-write 实现共享前缀（相同 system prompt 的多个请求共享 physical block）。详见 PagedAttention 一篇。

**Q：Chunked Prefill 是什么，为什么需要？**
A：一次处理超长 prompt（如 32k）的 prefill 会独占 GPU 好几秒，期间所有 decode 请求被阻塞（延迟尖刺）。Chunked prefill 把长 prompt 切成小块（如每块 512 token），和 decode 请求混合调度：每轮处理一块 prefill + 一批 decode。这样 decode 请求的延迟不会被长 prefill 拖垮，代价是 prefill 的总时长略微增加（多几次 kernel 启动、无法一次打满算力）。

**Q：多轮对话的 KV Cache 怎么复用？**
A：多轮对话中，历史轮次的 K/V 完全不变，可以缓存下来只对新一轮的增量做 prefill。工程上叫 **prefix caching / RadixAttention**（SGLang）：用 radix tree 组织所有请求的 KV block，公共前缀自动命中。命中率高时能省 50%~90% 的 prefill 计算。注意前提是历史完全一致的字节级前缀，任何 system prompt 的改动都会失效。

**Q：KV Cache 能卸载到 CPU 吗？什么时候划算？**
A：能，但只在高并发、请求间有明显空闲时划算。卸载的代价是 PCIe 带宽（约 20~30 GB/s，比 HBM 慢 50 倍），所以只能卸载"暂时不用"的请求（等待调度、被抢占）。vLLM 的 swap 和 FlexGen 的分层存储都是这个思路。评估标准：如果 KV 太大导致并发上不去，而 GPU 算力还有余量，卸载能提升总吞吐；如果已经算力打满，卸载只会增加延迟。

## 关联知识点
- MHA / MQA / GQA / MLA（02-attention）
- PagedAttention 与 vLLM 显存管理（02-attention）
- FlashAttention 与 IO 复杂度（02-attention）
- Continuous Batching 与推理引擎（07-inference）
- 量化：KV Cache 量化（07-inference）
- 长上下文外推：YaRN / NTK（02-attention）
