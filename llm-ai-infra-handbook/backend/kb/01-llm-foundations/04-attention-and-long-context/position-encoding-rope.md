---
id: position-encoding-rope
title: RoPE 旋转位置编码：推导、性质与实现陷阱
module: 04-attention-long-context
tags:
- 位置编码
- RoPE
- 长上下文
- 手撕代码
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: RoPE 用复数乘法的旋转群把绝对位置编码成旋转角，使 qᵀ_m k_n 天然只依赖相对位置 m-n，同时具备随相对距离增长的远程衰减性；实现上按相邻两维配对旋转。
source:
- https://arxiv.org/abs/2205.14135
questions:
- RoPE 是绝对位置编码还是相对位置编码？
- RoPE 为什么能外推，但外推效果又不好？
- 位置插值（PI）和 NTK-aware 缩放的区别？
- RoPE 的维度数是奇数怎么办？
- RoPE 对 KV Cache 友好吗？
---

## 一句话结论

RoPE 的核心思想只有一句话：**把 query 和 key 的每一对相邻维度看成复平面上的一个二维向量，位置 m 对应把它旋转 m·θ 弧度**。

这样做了之后，两个不同位置的向量做内积时，旋转角会相减：

```
⟨R_m q, R_n k⟩ 只依赖 (m - n)
```

于是"绝对位置编码"（每个位置的 q、k 都被独立地旋转了绝对角度）+ "相对位置语义"（内积只依赖相对距离）**同时成立**。这正是 RoPE 相比 Sinusoidal（绝对）和 ALiBi（纯偏置）最优雅的地方。

工程上要记住的三个数：默认 `base = 10000`，`head_dim = 128`，频率维度共 `d_h/2 = 64` 组，最低频率对应波长 `2π · base^(0/64) / 1 = 2π`… 最高波长为 `2π · 10000^(63/64) ≈ 2π · 8796 ≈ 55268` tokens。**RoPE 的理论外推上限约等于 base 对应的最长波长**。

## 原理推导

### 1. 从二维旋转出发

考虑 d=2 的情形。把 q 写成复数 `q = q₁ + i·q₂`，位置 m 的旋转定义为乘上 `e^{i·m·θ}`：

```
R_m q = e^{i·m·θ} · q
      = (cos mθ + i sin mθ)(q₁ + i q₂)
      = (q₁cos mθ - q₂ sin mθ) + i(q₁ sin mθ + q₂ cos mθ)
```

写成实矩阵：

```
R_m = [ cos(mθ)  -sin(mθ) ]
      [ sin(mθ)   cos(mθ) ]
```

验证内积的相对性（用复数共轭形式更简洁，`⟨a,b⟩ = Re(a·b̄)`）：

```
⟨R_m q, R_n k⟩ = Re( e^{imθ} q · conj(e^{inθ} k) )
               = Re( e^{imθ} · e^{-inθ} · q · k̄ )
               = Re( e^{i(m-n)θ} · q · k̄ )
```

**结果只依赖 (m-n)，与 m、n 的绝对值无关。** 这就是 RoPE 相对性的完整证明。

### 2. 推广到 d 维

把 d_h 维切成 d_h/2 组，每组用一个不同的角频率：

```
θ_j = base^(-2j/d_h),  j = 0, 1, ..., d_h/2 - 1
```

即频率从 `1`（第 0 组，旋转最快）到 `base^(-(d_h-2)/d_h)`（最后一组，旋转最慢）呈几何级数衰减。这与 Sinusoidal 编码的频率设计完全同构。

于是 R_m 是一个**分块对角**矩阵：

```
R_m = diag( R(m·θ_0), R(m·θ_1), ..., R(m·θ_{d_h/2 - 1}) )
```

其中每个 `R(m·θ_j)` 是 2×2 旋转。所以 d 维的旋转是 d_h/2 个独立的二维旋转。

由于 R_m 是正交矩阵（`RᵀR = I`），还顺带得到一个好性质：**RoPE 不改变向量模长**，即 `‖R_m q‖ = ‖q‖`，因此不影响 attention 中的 √d_k 缩放分析。

### 3. 高效实现：为什么不用 d×d 矩阵

直接构造 R_m（d_h × d_h 稀疏）再乘是 O(d_h²)，浪费。实际用逐元素写法：

```
给定 x = (x_0, x_1, ..., x_{d-1})
配对方式一（相邻配对，LLaMA/HF 采用）：
  x'_2j   = x_2j · cos(mθ_j) - x_{2j+1} · sin(mθ_j)
  x'_{2j+1} = x_{2j+1} · cos(mθ_j) + x_2j · sin(mθ_j)

配对方式二（前半后半配对，GPT-NeoX 采用）：
  x'_j      = x_j · cos(mθ_j) - x_{j+d/2} · sin(mθ_j)
  x'_{j+d/2} = x_{j+d/2} · cos(mθ_j) + x_j · sin(mθ_j)
```

两种写法数学等价（差一个置换），但**模型之间不兼容**——加载别人的权重时必须确认用的是哪种。这是实际工程中踩坑最多的点。

`rotate_half` 的实现技巧：

```
rotate_half(x) = concat(-x_{d/2:}, x_{:d/2})     # 相邻配对版
x' = x * cos + rotate_half(x) * sin
```

### 4. 远程衰减性

对随机初始化的 q、k，可以证明：

```
|⟨R_m q, R_n k⟩| 的期望随 |m-n| 增大而衰减（振荡衰减）
```

直觉：高频组（j 小）在距离稍大时相位早已跑满 2π 多圈，内积贡献相互抵消；只有低频组（j 大）在长距离上仍能保持稳定相位。所以**长距离依赖主要由低频维度承载**。

这个性质直接推出两条工程结论：

1. **低频维度承载长距离信息**，做外推（NTK / YaRN）时应该主要缩放低频，而不是一刀切
2. **高频维度决定短距离分辨力**，过度缩放高频会导致近处位置也分不清（这也是 YaRN 要用 ramp 函数的原因）

### 5. base 的选择与外推上限

第 j 组的旋转周期（走完 2π 需要的 token 数）：

```
T_j = 2π / θ_j = 2π · base^(2j/d_h)
```

最大周期（j = d_h/2 - 1）：

```
T_max = 2π · base^((d_h-2)/d_h)
```

- base = 10000, d_h = 128 → T_max = 2π · 10000^(126/128) ≈ 2π · 8140 ≈ 51,100 tokens
- base = 500000, d_h = 128 → T_max = 2π · 500000^(126/128) ≈ 2π · 427,000 ≈ 2,680,000 tokens

**结论**：想原生支持长上下文，直接把 base 调大（LLaMA-3 用 500000 支持 128k），比事后做位置插值更干净。

## 具体例子

**例 1：手算一个小例子**

d_h = 4（2 组），base = 10000，位置 m = 5。

```
θ_0 = 10000^(-0/4) = 1
θ_1 = 10000^(-2/4) = 1/100 = 0.01

m·θ_0 = 5.0 rad,  m·θ_1 = 0.05 rad
```

给定 q = [1, 0, 0, 1]：

```
第 0 组 (x_0=1, x_1=0):
  x'_0 = 1·cos5 - 0·sin5 = 0.2837
  x'_1 = 0·cos5 + 1·sin5 = -0.9589

第 1 组 (x_2=0, x_3=1):
  x'_2 = 0·cos0.05 - 1·sin0.05 = -0.0500
  x'_3 = 1·cos0.05 + 0·sin0.05 = 0.9988

R_5 q = [0.2837, -0.9589, -0.0500, 0.9988]
```

验证模长不变：`1+0+0+1 = 2`，旋转后 `0.0805+0.9195+0.0025+0.9976 = 2.0` ✓

**例 2：相对性数值验证**

取 q = [1,0,0,1], k = [0,1,1,0]，分别算 (m,n) = (5,2) 与 (100,97)，相对距离都是 3：

```
(m-n) = 3 时:  ⟨R_5 q, R_2 k⟩
(m-n) = 3 时:  ⟨R_100 q, R_97 k⟩
两者应完全相等（浮点误差内）
```

下面是可运行验证：

```python
import torch, math
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    ang = torch.outer(torch.tensor([float(pos)]), inv)          # (1, d/2)
    cos, sin = torch.cos(ang).repeat_interleave(2,-1), torch.sin(ang).repeat_interleave(2,-1)
    def rot_half(t):  # 相邻配对
        t = t.view(*t.shape[:-1], -1, 2)
        return torch.stack((-t[...,1], t[...,0]), -1).flatten(-2)
    return x * cos + rot_half(x) * sin

q = torch.tensor([1.,0,0,1]); k = torch.tensor([0.,1,1,0])
a = (rope(q,5) * rope(k,2)).sum()
b = (rope(q,100) * rope(k,97)).sum()
print(a.item(), b.item())     # 两者相等，验证相对性
```

**例 3：base 与外推上限**

```
base=10000,  d_h=128 → T_max ≈ 5.1e4  tokens  （LLaMA-2，训练 4k，外推到 32k 靠 PI/NTK）
base=500000, d_h=128 → T_max ≈ 2.7e6  tokens  （LLaMA-3，原生 8k 训练，128k 微调）
base=1e6,    d_h=128 → T_max ≈ 5.5e6  tokens  （部分长上下文模型）
```

## 代码速记

```python
import torch

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0):
    """返回 cos/sin 缓存，形状 (seq_len, dim)，已 repeat_interleave 到逐维"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (seq_len, dim) 与相邻配对对齐
    return torch.cos(emb), torch.sin(emb)

def rotate_half(x):
    """相邻两维配对: (x0,x1) -> (-x1,x0)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_emb(xq, xk, cos, sin, positions=None):
    """
    xq, xk: (B, H, S, d_h)
    cos, sin: (max_seq, d_h) 预计算缓存
    positions: (B, S) 每个样本每个位置的真实下标，用于打包序列(packing)
    """
    B, H, S, d = xq.shape
    if positions is not None:
        cos = cos[positions].unsqueeze(1)   # (B, 1, S, d)
        sin = sin[positions].unsqueeze(1)
    else:
        cos, sin = cos[:S].unsqueeze(0).unsqueeze(0), sin[:S].unsqueeze(0).unsqueeze(0)
    q_out = xq.float() * cos + rotate_half(xq.float()) * sin
    k_out = xk.float() * cos + rotate_half(xk.float()) * sin
    return q_out.to(xq.dtype), k_out.to(xk.dtype)

# ---- 常见坑的对照 ----
# 1) cos/sin 必须 upcast 到 fp32 再算，bf16 下大位置的角度精度误差会被放大
# 2) packing 时必须传 positions，否则同一 batch 内不同文档的位置索引会串味
# 3) GPT-NeoX 式配对（前半后半）与上式不同，加载权重前必须确认
# 4) 推理时用 KV Cache，k 的旋转只在 prefill/该步做一次，不要重复施加
```

## 面试追问

**Q：RoPE 是绝对位置编码还是相对位置编码？**
A：形式上是绝对的（每个位置的 q、k 都被独立施加了只依赖自身绝对位置的旋转），效果上是相对的（内积结果只依赖相对距离）。准确说法是"**用绝对编码的形式实现了相对位置语义**"，这也是它相比纯相对编码（如 Transformer-XL 的偏置项）的工程量优势——不需要改写 attention 的矩阵乘，只需在 QK 投影后加一步逐元素旋转。

**Q：RoPE 为什么能外推，但外推效果又不好？**
A：RoPE 本身是"周期性"编码，理论上位置可以无限增长，所以能在训练长度外继续算（不像可学习位置编码那样直接越界）。但训练时模型只见过 `|m-n| < L_train` 范围内的相位差，超出后高频维度的相位进入了未见区域，注意力分数分布失真，PPL 迅速恶化。解决办法就是各种缩放（PI / NTK / YaRN），本质都是**压缩相位差，把未见的相位映射回已见区间**。

**Q：位置插值（PI）和 NTK-aware 缩放的区别？**
A：PI 是把位置索引统一除以缩放因子 `m → m/s`，等价于所有频率同时变密，**高频和低频被同等对待**，缺点是短距离分辨力也一起变差（近处位置变得难区分）。NTK-aware 只改 base（`base → base·s^(d/(d-2))`），效果是**高频几乎不动、低频被拉伸**，保留了短距离分辨力、只扩展长距离，所以少量微调就能外推。

**Q：RoPE 的维度数是奇数怎么办？**
A：把最后一个维度留空不旋转即可（d_h 是奇数时 pad 到偶数再截回）。实践中 head_dim 一定是 2 的倍数（通常 64/128/256），不会遇到这个问题。

**Q：RoPE 对 KV Cache 友好吗？**
A：非常友好，这是它胜过 ALiBi 的关键之一。因为旋转是逐位置的、施加在 k 上之后就固定了，增量解码时新算出的 k 直接旋转一次后入缓存即可，历史缓存不需要任何修正。而某些相对位置编码（如 T5 的可学习相对偏置）在生成新 token 时需要访问全部历史相对距离，缓存复用更麻烦。

**Q：为什么有的模型（如部分视觉语言模型）用 multi-axis RoPE / M-RoPE？**
A：图像/视频的位置是二维或三维的。M-RoPE 把 head_dim 切成若干段，每段分别编码一个轴（时间/高度/宽度）。这样既能表达二维空间关系，又保持了旋转的相对性。代价是文本-only 输入时多段必须退化为相同角度，否则会与纯文本预训练分布不一致。

## 关联知识点
- ALiBi / 位置插值 / NTK / YaRN / 长上下文外推（02-attention）
- 长上下文训练与序列并行（02-attention）
- 手写 RoPE / RMSNorm / SwiGLU（10-coding）
- MHA / MQA / GQA / MLA（02-attention）
