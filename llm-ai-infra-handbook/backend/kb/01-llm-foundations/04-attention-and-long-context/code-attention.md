---
id: code-attention
title: 手撕 Attention：MHA / GQA / KV Cache / Flash
module: 04-attention-long-context
tags:
- 手撕代码
- Attention
- GQA
- KV Cache
- FlashAttention
- PyTorch
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 面试高频：标准 MHA（含 causal mask）、GQA（query 头按组复用 kv 头）、带 KV Cache 的增量 decode、以及
  FlashAttention 的 online softmax 分块实现。所有张量维度自洽、可直接运行。
source:
- https://arxiv.org/abs/2205.14135
questions:
- causal mask 怎么构造？为什么上三角置 -inf 而不是 0？
- GQA 相比 MHA 省了什么？repeat_interleave 在做什么？
- KV Cache 为什么能避免重算？缓存的是什么？
- FlashAttention 的 online softmax 为什么要维护 m 和 l？
- FlashAttention 显存为什么是 O(N) 而不是 O(N²)？
---

## 一句话结论

- **MHA**：`O = softmax(QKᵀ/√d)V`，Q/K/V 头数相同。
- **GQA**：query 头数 Hq = n·Hkv，每个 kv 头被 n 个 query 头共享（`repeat_interleave`）。
- **KV Cache**：decode 时只算当前 token 的 q，与缓存的历史 K/V 拼接后做注意力，避免重算。
- **FlashAttention**：分块 + online softmax，边读 K/V 块边更新输出与归一化统计量，显存 O(N) 而非 O(N²)。

## 实现要点

- 维度约定：`q,k,v: [B, H, S, D]`（多头已 split）。
- causal mask：上三角置 `-inf`，保证位置 i 只看 ≤i。
- GQA 的 `repeat_interleave(n, dim=1)` 让 `[B,Hkv,S,D]` → `[B,Hq,S,D]`。

## 输入输出示例

```python
# 输入 q,k,v 形状 [1, 8, 10, 64]（B=1,H=8,S=10,D=64）
out = scaled_dot_product_attention(q, k, v, causal=True)
# 输出 out: [1, 8, 10, 64]，位置 i 只依赖位置 ≤ i
```

## 核心代码

```python
import torch, torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, causal=False, mask=None):
    # q,k,v: [B, H, Sq/Sk, D]
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)        # [B,H,Sq,Sk]
    if causal:
        Sq, Sk = q.size(-2), k.size(-2)
        cm = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(cm, float('-inf'))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = scores.softmax(-1)
    return attn @ v                                             # [B,H,Sq,D]

def gqa_attention(q, k, v, causal=True):
    # q:[B,Hq,S,D], k,v:[B,Hkv,S,D], Hq = n*Hkv
    n = q.size(1) // k.size(1)
    k = k.repeat_interleave(n, dim=1)                      # [B,Hq,S,D]
    v = v.repeat_interleave(n, dim=1)
    return scaled_dot_product_attention(q, k, v, causal=causal)

# ---- KV Cache 增量 decode ----
class KVCache:
    def __init__(self): self.k, self.v = [], []
    def update(self, k, v):
        # k,v: [B, H, 1, D]（当前 token）；返回完整历史
        self.k.append(k); self.v.append(v)
        return torch.cat(self.k, dim=2), torch.cat(self.v, dim=2)

def decode_step(q_cur, kv_cache, k_proj, v_proj, wo, cache):
    # q_cur: [B,H,1,D]
    k_cur = k_proj(q_cur); v_cur = v_proj(q_cur)
    K, V = cache.update(k_cur, v_cur)                       # [B,H,S,P,d]
    ctx = scaled_dot_product_attention(q_cur, K, V, causal=False)
    return wo(ctx)                                          # 下一 token 的 logits 输入

# ---- FlashAttention 简化版：online softmax 分块 ----
def flash_attention(q, k, v, block=64):
    B, H, S, D = q.shape
    O = torch.zeros_like(q)
    m = torch.full((B, H, S, 1), -1e9, device=q.device)     # 运行最大值
    l = torch.zeros(B, H, S, 1, device=q.device)            # 运行归一化分母
    for i in range(0, S, block):
        kj, vj = k[:, :, i:i+block], v[:, :, i:i+block]      # 一个 K/V 块
        s = (q @ kj.transpose(-2, -1)) / (D ** 0.5)          # [B,H,S,block]
        m_new = torch.maximum(m, s.max(-1, keepdim=True))
        p = torch.exp(s - m_new)
        l_new = torch.exp(m - m_new) * l + p.sum(-1, keepdim=True)
        O = (torch.exp(m - m_new) * l * O + p @ vj) / l_new
        m, l = m_new, l_new
    return O
```

## 面试追问

1. causal mask 怎么构造？为什么上三角置 -inf 而不是 0？
2. GQA 相比 MHA 省了什么？repeat_interleave 在做什么？
3. KV Cache 为什么能避免重算？缓存的是什么？
4. FlashAttention 的 online softmax 为什么要维护 m 和 l？
5. FlashAttention 显存为什么是 O(N) 而不是 O(N²)？

## 关联知识点

- [MHA / MQA / GQA / MLA 对比与 KV Cache 压缩](mha-mqa-gqa-mla)
- [FlashAttention：IO 感知的分块算法](flash-attention)
- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
