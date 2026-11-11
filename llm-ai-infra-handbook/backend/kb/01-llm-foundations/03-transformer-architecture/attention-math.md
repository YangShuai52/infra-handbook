---
id: attention-math
title: Scaled Dot-Product Attention 与为什么要除以 √d
module: 03-transformer-architecture
tags:
- 注意力
- 数学推导
- 复杂度
- Attention Sink
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 除以 √d_k 是为了把点积的方差从 d_k 拉回 1，避免 softmax 进入梯度为 0 的饱和区；Attention 的时间复杂度是 O(S²·d)，显存复杂度是
  O(S²)。
source:
- https://arxiv.org/abs/1706.03762
questions:
- 乘性注意力和加性注意力有什么区别？为什么选乘性？
- 多头注意力的每个头到底学到了什么？可以剪掉一部分吗？
- softmax 一定要归一化吗？如果去掉 softmax 会怎样？
- causal mask 是怎么实现的？为什么不影响 GQA/MQA？
- 为什么 attention 的 Q 和 K 要用不同的投影矩阵？
---

## 一句话结论

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

除以 √d_k 不是工程 trick，而是**方差校正**：q·k 是 d_k 个独立随机变量之和，其方差正比于 d_k，不除的话 d_k=128 时点积的标准差约为 √128 ≈ 11.3，softmax 会退化成近似 one-hot，梯度趋近于 0，训练直接停滞。除以 √d_k 后标准差回到 1，softmax 处于梯度健康的区域。

复杂度上要分清两个量：**时间 O(S²·d)**（两个 S×S×d 的矩阵乘），**显存 O(S²)**（那个 S×S 的注意力分数矩阵）。长上下文的所有优化，本质上都是在 O(S²) 这一项上做文章。

## 原理推导

### 1. 为什么必须除以 √d_k

设 q、k 的各分量独立，均值为 0，方差为 1（初始化后大致成立）。则：

```
q·k = Σ_{i=1}^{d_k} q_i · k_i

E[q·k]   = Σ E[q_i]E[k_i] = 0
Var[q·k] = Σ Var[q_i·k_i]        （独立随机变量之和，方差可加）
         = Σ (E[q_i²]E[k_i²] - E[q_i]²E[k_i]²)
         = Σ (1 × 1 - 0) = d_k
```

即 **q·k 的方差 = d_k，标准差 = √d_k**。

再看 softmax 的梯度。对 logits 向量 `z`，softmax 输出 `p`，交叉熵损失 `L`：

```
∂p_i/∂z_j = p_i(δ_ij - p_j)
∂L/∂z = p - y
```

关键在于 `∂p_i/∂z_j` 正比于 `p_i·p_j`。当某个 logit 远大于其他（比如大 11 倍标准差），softmax 输出接近 one-hot，某一维 p→1 其余 p→0，于是所有 `p_i·p_j → 0`，**梯度消失**。

除以 √d_k 后：

```
Var[ (q·k)/√d_k ] = d_k / d_k = 1
```

logits 落在 ±3 范围内，softmax 输出平滑，梯度健康。

**一个常见追问**：如果 q、k 的方差不是 1 而是 σ²，那么缩放因子应该是 σ√d_k。RoPE 不改变向量模长，所以不影响这个结论。

### 2. 时间与显存复杂度

记 S = 序列长度，d = d_model，B = batch，H = 头数，d_h = d/H。

**单头**：

```
QKᵀ          : S × d_h × S  →  FLOPs = 2·S²·d_h
softmax      : O(S²)
× V          : S × S × d_h  →  FLOPs = 2·S²·d_h
```

**全部 H 个头**：

```
Attention FLOPs ≈ 4 · B · H · S² · d_h = 4 · B · S² · d
```

注意这里的 **4**：QKᵀ 是 2 次运算（乘+加）乘 S²d_h，AV 同理，合计 4·S²d_h。

对比 FFN 的 FLOPs（每 token 2·3·d·d_ff，共 B·S 个 token）：

```
FFN FLOPs ≈ 6 · B · S · d · d_ff
```

两者相等时的序列长度：

```
4·B·S²·d = 6·B·S·d·d_ff
S = 1.5 · d_ff
```

LLaMA-7B 的 d_ff=11008，即 **S ≈ 16512** 时 attention 才追平 FFN。所以在常规 4k/8k 长度下，**FFN 才是算力大头**；长到 128k 以上，attention（以及那个 S×S 的显存）才会成为主要矛盾。这是很多人答错的送分题。

**参数与算力的比例也要分清**：参数量上 FFN 占 2/3，但算力占比取决于 S，不能混为一谈。

### 3. 显存复杂度与显存优化

标准实现需要物化 S×S 的分数矩阵：

```
显存 = B · H · S² · (2 或 4 bytes)        # bf16 存 2 字节，softmax 时转 fp32
```

7B 模型，B=8, H=32, S=8192，bf16：

```
8 × 32 × 8192² × 2 = 3.44e10 bytes ≈ 32 GB
```

这比模型权重（13.5GB）还大，所以长序列推理必须做 kernel 优化。FlashAttention 的切入点正是**不物化这个 S×S 矩阵**，详见 `FlashAttention` 一篇。

### 4. Attention Sink（注意力汇聚）

现象：无论输入是什么，第一个 token（通常是 `<bos>` 或第一个图像 patch）都会吸走大量注意力权重，而且它的 value 范数很小、对输出贡献接近 0。

解释：softmax 的性质要求所有位置权重之和为 1。当某一层的所有 query 都"不想"关注任何 key 时，理想输出是"零向量"，但 softmax 无法表达全零。于是模型学会把多余的注意力**倾倒**到一个固定的 token 上，把它的 value 学成接近 0，等效实现了"no-op"。

这个发现的工程意义非常实际：

1. **流式/滑窗推理**（StreamingLLM）只要保留 sink token + 近期窗口，就能在无限长输入上不崩
2. **量化时** sink token 所在的区域对量化误差极其敏感，把它们和 value 一起处理会显著劣化
3. 加一个可学习的 `sink` 参数（软注意力门）能提升长上下文稳定性

## 具体例子

**例 1：手算缩放的效果**

设 d_k = 128，q、k 各分量独立标准正态。随机采样一次，假设 q·k = 14.2（这很常见，因为标准差 ≈ 11.3）。取另一 key 使得 q·k₂ = -3.0。

不缩放时：

```
z = [14.2, -3.0]
e^14.2 = 1.47e6,  e^-3.0 = 0.0498
p = [1.0000000, 0.000000034]      → 几乎 one-hot
∂p_1/∂z_2 = -p_1·p_2 ≈ -3.4e-8    → 梯度消失
```

缩放后：

```
z = [14.2/11.31, -3.0/11.31] = [1.256, -0.265]
e^1.256 = 3.51,  e^-0.265 = 0.767
p = [0.821, 0.179]                 → 平滑分布
∂p_1/∂z_2 = -0.821 × 0.179 = -0.147  → 梯度健康
```

**例 2：注意力分数矩阵显存**

B=8, H=32, S=8192，bf16 存分数 + fp32 做 softmax：

```
bf16 分数矩阵:  8 × 32 × 8192² × 2 = 3.44e10 B = 32.0 GB
fp32 softmax:   8 × 32 × 8192² × 4 = 6.87e10 B = 64.0 GB
```

这就是"为什么一定要 FlashAttention"的定量答案——不是慢，是**放不下**。

**例 3：Attention vs FFN 的 FLOPs 交叉点**

LLaMA-7B（d=4096, d_ff=11008, L=32），batch=1：

```
S = 4096:
  Attn = 4 × 1 × 4096² × 4096 × 32 = 8.80e12 FLOPs
  FFN  = 6 × 1 × 4096 × 4096 × 11008 × 32 = 3.54e13 FLOPs
  → FFN 是 Attention 的 4.0 倍

S = 32768:
  Attn = 4 × 1 × 32768² × 4096 × 32 = 5.63e14 FLOPs
  FFN  = 6 × 1 × 32768 × 4096 × 11008 × 32 = 2.83e14 FLOPs
  → Attention 反超，是 FFN 的 2.0 倍
```

## 代码速记

```python
import torch, torch.nn.functional as F
import math

def attention(q, k, v, mask=None, p_drop=0.0):
    """
    q: (B, H, S_q, d_h)
    k, v: (B, H, S_k, d_h)
    返回 (B, H, S_q, d_h), 以及注意力权重
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # (B,H,S_q,S_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)          # fp32 更稳，PyTorch 内部会 upcast
    attn = F.dropout(attn, p_drop)
    return torch.matmul(attn, v), attn

def attention_flops(B, H, S_q, S_k, d_h, causal=True):
    """两个 matmul 的乘加各算 2 FLOPs，causal 时约一半"""
    base = 4 * B * H * S_q * S_k * d_h
    return base / 2 if causal else base

def score_matrix_bytes(B, H, S, dtype_bytes=2):
    return B * H * S * S * dtype_bytes

# 验证：7B, B=8, S=8192
print(f"{score_matrix_bytes(8, 32, 8192)/1024**3:.1f} GB")  # 32.0 GB
print(f"{attention_flops(1, 32, 4096, 4096, 128)/1e12:.2f} TFLOPs")  # 17.18
```

注意：`torch.finfo(dtype).min` 与 `-inf` 的区别。用 `-inf` 时如果某一行全被 mask（例如 padding 行），softmax 会产生 NaN；用 `min` 至少不会 NaN，但更规范的做法是给每行预留一个永远不被 mask 的位置（如 sink token）。

## 面试追问

**Q：乘性注意力和加性注意力有什么区别？为什么选乘性？**
A：加性注意力用一个前馈网络算分数 `vᵀ tanh(W[q;k])`，乘性直接 `qᵀk`。加性的计算量更大（多一个 d² 的 W），但两者表达能力相当。选乘性的真正原因是**乘性可以写成两个大矩阵乘**，能用 Tensor Core 打满；加性需要先把 q 和 k 拼起来做逐位置的小矩阵乘，无法高效批处理。这也是 Transformer 原论文强调"可以用高度优化的矩阵乘法"的原因。

**Q：多头注意力的每个头到底学到了什么？可以剪掉一部分吗？**
A：经验上头的功能高度分化：有专门看上一个词的、有看句法依存头的、有看标点或专有名词的，但也有大量头是冗余的（多个头学出几乎相同的模式）。所以**推理时剪头可行**（剪掉 20-40% 通常损失很小），但**训练时不能剪**——训练中的冗余头提供了梯度路径的多样性，早期剪掉会损害最终效果。

**Q：softmax 一定要归一化吗？如果去掉 softmax 会怎样？**
A：softmax 提供两件事：非负性和归一化（行和为 1）。去掉后退化成线性注意力（有权重可能为负），可以用核方法改写成 O(N) 复杂度，但会损失"聚焦"能力——softmax 的指数放大让模型能几乎完全忽略无关 token，线性注意力做不到，这就是线性注意力质量下降的核心原因。

**Q：causal mask 是怎么实现的？为什么不影响 GQA/MQA？**
A：实现上就是给 `scores` 加一个下三角为 0、上三角为 -inf 的矩阵。它只作用于 S_q × S_k 的分数矩阵，与 Q/K/V 的头数无关，所以 GQA/MQA 完全不受影响。注意用 KV Cache 增量解码时，query 长度为 1，不需要 mask。

**Q：为什么 attention 的 Q 和 K 要用不同的投影矩阵？**
A：如果共享（Q=K），注意力矩阵会强制对称，即"i 关注 j 的程度"恒等于"j 关注 i 的程度"，这在语言里是不合理的（修饰关系天然有向）。用不同投影打破对称性，是表达能力上的硬需求。

## 关联知识点
- MHA / MQA / GQA / MLA 对比与 KV Cache 压缩（02-attention）
- FlashAttention 与 IO 复杂度分析（02-attention）
- 稀疏与线性注意力、Mamba 状态空间模型（02-attention）
- 手写 Multi-Head Attention 与 KV Cache（10-coding）
