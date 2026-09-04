---
id: paged-attention
title: PagedAttention 与高效注意力变体
module: 16-inference-runtime-serving
tags:
- PagedAttention
- vLLM
- 显存管理
- 稀疏注意力
- 线性注意力
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: PagedAttention 用操作系统的分页思想管理 KV Cache，把显存浪费从 60%+ 降到 4% 以下并支持前缀共享；稀疏/线性注意力则从算法侧把
  O(S²) 降下来，两者正交。
source:
- https://arxiv.org/abs/2309.06180
questions:
- PagedAttention 和 FlashAttention 是一回事吗？
- block 不连续，kernel 怎么处理？
- 前缀共享是怎么保证正确性的？
- 为什么线性注意力的检索能力弱？
- 混合架构（线性 + 少量全注意力）为什么有效？
---

## 一句话结论

PagedAttention 解决的问题是**显存碎片和过度预留**，不是算力。

朴素 KV Cache 管理的问题：

1. **内部碎片**：按最大长度预分配，短请求占着长坑
2. **外部碎片**：请求长度不一，释放后留下大小不一的空洞
3. **无法共享**：相同 system prompt 的 N 个请求各存一份

PagedAttention 把 KV Cache 切成固定大小的 **block**（如 16 个 token），用**页表**（logical block → physical block）映射，实现按需分配、几乎零碎片、以及 copy-on-write 的跨请求共享。

实测效果：**显存浪费从 60%~80% 降到 4% 以下，吞吐提升 2~4 倍**。这是 vLLM 成为事实标准的核心原因。

## 原理推导

### 1. 朴素方案的浪费有多大

假设按 `max_seq_len = 2048` 预分配，但实际请求长度服从分布：平均 400 token、最大 2000 token。

```
平均利用率 = 400 / 2048 = 19.5%
浪费率 = 80.5%
```

再考虑"预留"进一步加剧：部分框架会预留到 max_len + output_len，浪费更高。

即使是"按实际 prompt 长度精确分配 + 预留 output 长度"，也有两个问题：
- 输出长度事先未知，必须保守预留
- 分配/释放频繁，产生外部碎片

### 2. PagedAttention 的三层结构

```
逻辑层:  请求的 KV Cache 是 一段连续的 logical blocks [b0, b1, b2, ...]
页表层:  block table: logical_block_id → physical_block_id
物理层:  GPU 显存中固定大小的 physical blocks（非连续）
```

Block 大小的选择（默认 16 tokens）：

```
太小:  页表变大，block table 查找开销上升，kernel 效率低
太大:  内部碎片回升（最后一个 block 平均浪费 B/2）
16~32 是实测最优区间
```

显存浪费的定量分析（block size = B，请求长度 S）：

```
最后一个 block 的浪费 = B/2（平均）
相对浪费率 = (B/2) / S
B=16, S=400 → 2%
```

对比朴素方案 60%+，收益巨大。

### 3. Copy-on-Write 与前缀共享

多个请求共享相同前缀（如同一个 system prompt、同一份文档的多轮问答）时：

```
初始:  所有请求的 logical blocks 指向同一批 physical blocks（引用计数 = N）
生成分歧后:  只复制被改写的那个 block，其余继续共享
```

这就是操作系统的 **copy-on-write**。收益取决于前缀重复率：

```
共享前缀占比 60% 的 N 个请求 → 显存节省接近 60%
```

工程上要注意：只有**字节级完全相同**的前缀才能命中，所以 system prompt 里带时间戳、随机数会导致 100% 失效。

### 4. 调度配合

PagedAttention 与调度器配合才完整：

1. ** admission control**：只有当前显存放得下整个请求（含预留输出）才接受
2. **抢占与重算**：显存不足时，抢占低优先级请求，把它的 block 释放；恢复时重算（或 swap 到 CPU）
3. **Continuous batching**：iteration-level 调度，每个 decode 步重新组批，完成的请求立即退出、新请求立即加入

这三者 + PagedAttention 共同构成 vLLM 的吞吐优势。

### 5. 稀疏与线性注意力（算法侧的另一条路）

PagedAttention 优化的是"存"，稀疏/线性注意力优化的是"算"。

**稀疏注意力**（少算一些分数）：

| 类型 | 模式 | 复杂度 |
|---|---|---|
| 局部窗口 | 只看最近 W 个 | O(S·W) |
| 全局 + 局部（Longformer/BigBird） | 局部窗口 + 若干全局 token + 随机 | O(S·(W+R+G)) |
| 轴向/带状（Sparse Transformer） | 按维度分解 | O(S√S) |
| NSA（DeepSeek） | 压缩 + 选择 + 滑窗三分支 | O(S·W) |

**线性注意力**（改掉 softmax）：

标准 attention 可以写成核形式：

```
Attention_i = Σ_j exp(q_i·k_j) v_j / Σ_j exp(q_i·k_j)
若 exp(q·k) ≈ φ(q)ᵀφ(k)，则
Attention_i = φ(q_i)ᵀ Σ_j φ(k_j) v_jᵀ / φ(q_i)ᵀ Σ_j φ(k_j)
            = φ(q_i)ᵀ · S / φ(q_i)ᵀ · z        其中 S, z 可递推
```

递推形式使复杂度降为 **O(S·d²)**，且解码是 O(1) 每步（类似 RNN）。

代表：Performer（随机特征）、Linear Transformer、RetNet、GLA、Mamba（状态空间 + 选择性门控）。

**代价**：softmax 的指数放大带来"聚焦"能力，核近似丢掉了这一点；线性注意力的**召回/精确检索能力明显弱于 softmax attention**（这是近年多篇论文的共识）。所以纯线性模型在"大海捞针"类任务上表现差，主流方案是**混合架构**（大部分层用线性/滑窗，少量层保留全注意力，如 Jamba、Samba、部分 Mamba-2 混合体）。

### 6. Mamba / 状态空间模型

```
h_t = A·h_{t-1} + B·x_t        （状态更新，A 是结构化矩阵）
y_t = C·h_t                     （输出）
```

加上**选择性**（让 B、C、Δ 依赖输入）后就是 Mamba（S6）。

```
复杂度:  训练 O(S·d)（可用并行扫描，仍是 O(S log S) 但常数小）
         推理 O(1) 每步，显存 O(1)（不需要 KV Cache）
```

这是它相比 Transformer 最大的卖点：**无限上下文、常数显存**。但如上所述，精确检索能力是短板。

## 具体例子

**例 1：显存节省的定量对比**

A100 80GB，7B 模型（权重 13.5GB），剩余 64GB 用于 KV（MHA, d=4096, L=32, bf16，每 token 512KB）：

```
朴素（按 2048 预分配，实际平均 400 token）:
  每请求预留 = 2048 × 512KB = 1.0 GB
  并发数 = 64 / 1.0 = 64 个请求
  实际有效 token = 64 × 400 = 25,600 tokens

PagedAttention（block=16，按需分配）:
  每请求实际 = 400 × 512KB × 1.02 = 0.2 GB
  并发数 = 64 / 0.2 = 320 个请求
  有效 token = 320 × 400 = 128,000 tokens

吞吐提升 ≈ 5×
```

**例 2：前缀共享的收益**

100 个请求，共享 800 token 的 system prompt，各自有 200 token 的问题：

```
不共享:  100 × 1000 × 512KB = 51.2 GB
共享:    800 × 512KB + 100 × 200 × 512KB = 0.4 + 10.2 = 10.6 GB
节省:    79%
```

**例 3：block size 的权衡**

```
请求长度分布: 平均 S=400

B=8:   浪费 (8/2)/400 = 1.0%,  页表条目 50 个
B=16:  浪费 (16/2)/400 = 2.0%, 页表条目 25 个
B=32:  浪费 (32/2)/400 = 4.0%, 页表条目 13 个
B=128: 浪费 (128/2)/400 = 16%, 页表条目 4 个
```

B=16 是"浪费率"和"页表开销"的平衡点，这也是 vLLM 的默认值。

**例 4：稀疏注意力的复杂度对比（S=128k）**

```
全注意力:        S² = 1.64e10    相对 1×
滑窗 W=4096:     S×W = 5.24e8   相对 1/31
NSA:             ≈ S×(W+选择数+压缩) 相对约 1/25
线性注意力:      S×d = 1.28e9（d=1万） 但每步 O(1)
```

## 代码速记

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class BlockAllocator:
    """PagedAttention 的物理块分配器（简化 CPU 版）"""
    num_blocks: int
    block_size: int
    free: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.free = list(range(self.num_blocks))
        self.refcount = [0] * self.num_blocks

    def alloc(self) -> Optional[int]:
        if not self.free:
            return None
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def free_block(self, b: int):
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def share(self, b: int):
        self.refcount[b] += 1

@dataclass
class Sequence:
    """一个请求的逻辑 KV Cache"""
    seq_id: int
    logical: List[int] = field(default_factory=list)   # logical → physical
    token_count: int = 0

class PagedKVCache:
    def __init__(self, num_blocks, block_size):
        self.alloc = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.seqs: Dict[int, Sequence] = {}
        # 前缀哈希 → physical blocks，用于跨请求前缀共享
        self.prefix_index: Dict[int, Tuple[List[int], int]] = {}

    def _n_blocks(self, n_tokens):
        return (n_tokens + self.block_size - 1) // self.block_size

    def append(self, seq_id, tokens, prefix_hash=None):
        seq = self.seqs.setdefault(seq_id, Sequence(seq_id))
        # 1) 尝试命中共享前缀
        if prefix_hash is not None and prefix_hash in self.prefix_index:
            shared, n_shared_tokens = self.prefix_index[prefix_hash]
            for b in shared:
                self.alloc.share(b)
            seq.logical = list(shared)
            seq.token_count = n_shared_tokens
            tokens = tokens[n_shared_tokens:]        # 只需处理剩余部分
        # 2) 按需分配新块
        need = self._n_blocks(seq.token_count + len(tokens)) - len(seq.logical)
        for _ in range(need):
            b = self.alloc.alloc()
            if b is None:
                raise RuntimeError("OOM: 无空闲 block，需要抢占/swap")
            seq.logical.append(b)
        seq.token_count += len(tokens)
        return seq

    def fork_on_write(self, src_id, dst_id):
        """copy-on-write：先共享，写时再复制"""
        src = self.seqs[src_id]
        dst = Sequence(dst_id, list(src.logical), src.token_count)
        for b in dst.logical:
            self.alloc.share(b)
        self.seqs[dst_id] = dst

    def release(self, seq_id):
        seq = self.seqs.pop(seq_id)
        for b in seq.logical:
            self.alloc.free_block(b)

def memory_efficiency(block_size, avg_len, max_len):
    """对比朴素预留 vs 分页的显存效率"""
    naive = avg_len / max_len
    paged = avg_len / (avg_len + block_size / 2)
    return {"naive_util": naive, "paged_util": paged,
            "saving": 1 / paged - 1 / naive}

print(memory_efficiency(16, 400, 2048))
# {'naive_util': 0.195, 'paged_util': 0.980, 'saving': 4.1×}
```

## 面试追问

**Q：PagedAttention 和 FlashAttention 是一回事吗？**
A：完全不是，两者正交、可以叠加。FlashAttention 优化**单次 attention 计算**的访存（不物化 S×S 矩阵）；PagedAttention 优化 **KV Cache 的显存管理**（分页、共享、按需分配）。vLLM 里两者同时存在：kernel 用 FlashAttention/PagedAttention 融合的 kernel，显存管理用分页。

**Q：block 不连续，kernel 怎么处理？**
A：kernel 拿到的是 block table（一个 int 数组），对每个 logical block 先查表得到 physical block id，再定位到实际显存地址。因为 block 内部是连续的，所以每个 block 内仍可以用高效的连续访存和向量化。代价是每次访问多一次间接寻址（查表），但这在 SRAM 里完成，代价远小于节省的 HBM。

**Q：前缀共享是怎么保证正确性的？**
A：靠**哈希 + 引用计数 + copy-on-write**。前缀内容做哈希（对 token id 序列分块哈希，形成链式 hash，保证只有完整 block 才共享），命中则共享；当某个请求要往共享 block 里写入新 token 时，先检查引用计数 > 1，若是则复制一份新的 physical block 再写（copy-on-write），并把自己的页表指向新块。这样已共享的其他请求不受影响。

**Q：为什么线性注意力的检索能力弱？**
A：线性注意力把历史压缩成一个固定大小的状态矩阵 `S = Σ φ(k_j)v_jᵀ`，这是一个**有损压缩**——多个 key-value 对被叠加进同一个矩阵，无法再单独取出某一个。softmax attention 则显式保留每个 (k_j, v_j)，查询时逐一比对，所以能精确定位到某一个 token。这就是为什么"大海捞针"任务上线性模型显著落后，也解释了为什么纯 Mamba 模型在 in-context learning 上弱于 Transformer。

**Q：混合架构（线性 + 少量全注意力）为什么有效？**
A：因为两类任务的需求不同。大部分 token 间的交互是"局部的、可压缩的"（语法、邻近指代），线性/滑窗层处理这部分既省又快；少数任务需要"精确的全局检索"，用插入的全注意力层（如每 6 层插 1 层）承担。实验表明用 1/8 ~ 1/4 的全注意力层就能恢复绝大部分检索能力，同时保留线性层的效率优势。Jamba、Samba、Mamba-2-hybrid 都采用这个思路。

**Q：vLLM 的 swap 和 recompute 怎么选？**
A：抢占时有两种恢复方式：把 KV block swap 到 CPU 内存（换回时 PCIe 传输约 20~30 GB/s），或者重算 prefill（成本 = 一次 prefill 的 FLOPs）。判据是比较两者耗时：短 prompt 重算更快（算力便宜），长 prompt swap 更快（传输比重算便宜）。vLLM 会根据 block 数量和当前带宽动态选择。

## 关联知识点
- KV Cache 原理与显存公式（02-attention）
- FlashAttention 与 IO 复杂度（02-attention）
- MHA / MQA / GQA / MLA（02-attention）
- Continuous Batching 与推理引擎（07-inference）
- RadixAttention / prefix caching（07-inference）
- 长上下文外推：YaRN / NTK（02-attention）
