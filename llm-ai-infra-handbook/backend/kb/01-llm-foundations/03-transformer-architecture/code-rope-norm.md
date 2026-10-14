---
id: code-rope-norm
title: 手撕 RoPE / RMSNorm / SwiGLU / Pre-LN Block
module: 03-transformer-architecture
tags:
- 手撕代码
- RoPE
- RMSNorm
- SwiGLU
- Pre-LN
- PyTorch
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 现代 LLM 的"标准件"默写题：RoPE 的频率预计算(precompute_freqs_cis)与旋转应用(rotate_half + apply_rotary_emb)、RMSNorm（去均值与
  bias）、SwiGLU FFN、以及 Pre-LN Transformer Block 前向。
source:
- https://arxiv.org/abs/1706.03762
questions:
- RoPE 为什么用"相邻两维当复数对"？half rotation 是什么？
- RMSNorm 相比 LayerNorm 少了什么？为什么更稳？
- SwiGLU 为什么中间维取 8d/3 而不是 4d？
- Pre-LN 和 Post-LN 在代码上差哪一行？
- HF 实现里 `freqs_cis` 为什么要缓存、为什么不每次重算？
---

## 一句话结论

- **RoPE**：给每个位置预计算一组旋转角 `freqs_cis`，把 q/k 的相邻两维当成复数做旋转，使内积只依赖相对位置。
- **RMSNorm**：`x / sqrt(mean(x²)+ε) · γ`，省去减均值与 bias，更稳更快。
- **SwiGLU**：`Swish(W1 x) ⊙ (W2 x)`，中间维取 `4d` 或 `8d/3` 以匹配参数量。
- **Pre-LN Block**：先 RMSNorm 再做注意力/FFN，残差加在归一化之前。

## 实现要点

- `precompute_freqs_cis(dim, end, base)`：生成 `[end, dim/2]` 的角度表。
- `rotate_half(x)`：把后半维与前半维交错成对，供复数乘法。
- SwiGLU 用 `8d/3` 时为保证整数常取 `int(8*d/3)` 且向上对齐到某倍数。

## 输入输出示例

```python
freqs = precompute_freqs_cis(dim=64, end=128, base=10000)
x = torch.randn(1, 4, 128, 64)          # [B,H,S,dim]
out = apply_rotary_emb(x, x, freqs)      # q,k 同形旋转后 [1,4,128,64]
normed = RMSNorm(64)(torch.randn(2, 10, 64))   # 均值≈0、方差≈1*γ
```

## 核心代码

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

# ---------- RoPE ----------
def precompute_freqs_cis(dim, end, base=10000.0, theta=None):
    base = theta or base
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                    # [end, dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # 复数 [end, dim/2]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, freqs_cis):
    # q,k: [B,H,S,dim]; freqs_cis: [S, dim/2] 复数
    def to_cplx(t):
        return torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2).contiguous())
    qc, kc = to_cplx(q), to_cplx(k)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(1)      # [1,1,S,dim/2]
    qo = torch.view_as_real(qc * freqs).flatten(-2).type_as(q)
    ko = torch.view_as_real(kc * freqs).flatten(-2).type_as(k)
    return qo, ko

# ---------- RMSNorm ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.gamma = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.gamma

# ---------- SwiGLU FFN ----------
class SwiGLU(nn.Module):
    def __init__(self, d, ffn_mult=8/3):
        super().__init__()
        hid = int(ffn_mult * d)
        self.w1 = nn.Linear(d, hid, bias=False)
        self.w2 = nn.Linear(hid, d, bias=False)
        self.w3 = nn.Linear(d, hid, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))   # SwiGLU(x)=SiLU(W1x)⊙W3x

# ---------- Pre-LN Block ----------
class PreLNBblock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = SwiGLU(d)
    def forward(self, x, freqs=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)     # 实际需接 RoPE 与 causal mask
        x = x + a
        return x + self.ffn(self.ln2(x))
```

## 面试追问

1. RoPE 为什么用"相邻两维当复数对"？half rotation 是什么？
2. RMSNorm 相比 LayerNorm 少了什么？为什么更稳？
3. SwiGLU 为什么中间维取 8d/3 而不是 4d？
4. Pre-LN 和 Post-LN 在代码上差哪一行？
5. HF 实现里 `freqs_cis` 为什么要缓存、为什么不每次重算？

## 关联知识点

- [RoPE 旋转位置编码：推导、性质与实现陷阱](position-encoding-rope)
- [归一化、残差与激活函数：RMSNorm / Pre-LN / SwiGLU](normalization-activation)
- [手撕 Attention：MHA / GQA / KV Cache / Flash](code-attention)
