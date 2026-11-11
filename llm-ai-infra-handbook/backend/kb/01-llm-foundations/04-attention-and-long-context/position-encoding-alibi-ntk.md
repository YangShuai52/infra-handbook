---
id: position-encoding-alibi-ntk
title: 长上下文外推：ALiBi、位置插值、NTK 与 YaRN
module: 04-attention-long-context
tags:
- 位置编码
- 长上下文
- ALiBi
- NTK
- YaRN
- 外推
- RoPE
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 位置编码处理"超长序列"有两派：ALiBi 用注意力偏置从训练长度就限制感知范围、天然外推；RoPE 系（PI/NTK/YaRN）则通过在"频率轴"上做插值/缩放，把已训练的位置映射拉伸到更长区间。本质是"外推
  vs 插值"的权衡。
source:
- https://arxiv.org/abs/2205.14135
questions:
- 为什么 ALiBi 能"零成本"外推，而 Sinusoidal/RoPE 直接外推会崩？
- NTK-aware 相比 PI 改了什么？为什么说 PI 破坏了高频/低频的不对称性？
- YaRN 的温度修正 t 是在 attention 的哪一步生效？不加会怎样？
- Dynamic NTK 和 YaRN 的一个关键区别（一个推理期自适应、一个需训练）？
- RoPE 的 base 越大，外推上限越高还是越低？为什么 Llama-3 把 base 提到 50 万？
---

## 一句话结论

- **ALiBi**：不在词向量里加位置，而是在 attention 分数上加一个与距离线性负相关、与头相关的偏置（`-m·|i-j|`）。训练时就只看局部 → 推理时天然能外推到更长。
- **位置插值 PI / NTK-aware / YaRN**：RoPE 在"频率维度"上做缩放，把训练位置 `[0, L)` 线性/非线性地压缩到 `[0, L')`（L' > L），让模型在已见过的频率区间内工作，代价是牺牲一点短文本分辨率。

## 原理推导

### ALiBi 的偏置从哪来

标准 attention 分数 `S = QKᵀ/√d`。ALiBi 改为：

```
S_ij = (Q_i K_jᵀ)/√d  +  bias_i,j,    bias_i,j = - m_h · (i - j)
```

- `m_h` 是第 h 个头的斜率（沿 head 维度几何递减，如 `2^(-8h/H)`），使不同头关注不同距离。
- 距离越远偏置越负 → 远距离 token 被天然压制。**训练时上下文窗口固定，但偏置不依赖绝对位置值**，所以推理到更长序列时，偏置公式照常成立 → 零成本外推。
- 对比 Sinusoidal/RoPE：它们把绝对位置编码进向量，训练长度 L 之外的位置在训练分布之外，直接外推会"越界"。

### RoPE 外推为什么难

RoPE 中第 m 个位置的旋转角频率 `θ_i = base^(-2i/d)`，`m·θ_i` 随位置线性增长。训练只见过 `m ∈ [0, L)`，推理到 `m' > L` 时 `m'·θ_i` 进入未训练区间 → 高频维（大 i，小 θ）在训练内已转过多圈、很难外推；低频维（小 i，大 θ）变化慢反而好外推。这就是"高频难外推、低频易外推"。

### 位置插值 PI（Position Interpolation, k=2 with Llama-2）

直接把位置坐标压缩：`f'(m) = f(m · L/L')`。原 RoPE 角 `m·θ` 变成 `(m·L/L')·θ`，即把 `[0,L')` 的测试位置映射回 `[0,L)` 训练区间。等价于把 RoPE 的 base 改为 `base' = base^(L/L')`（因为 `m·base'^(-2i/d) = m·(L'/L)·base^(-2i/d)`）。**代价**：短文本分辨率下降（相邻 token 角度差变小），需要继续训练（SFT/继续预训练）来适应。

### NTK-aware 插值（频域视角）

PI 对所有频率统一缩放，破坏了"高频应少缩放、低频应多缩放"的特性。NTK 思路：把缩放分散到"维度轴"而非"位置轴"——对每维 i 用不同的缩放，使**低频维几乎不缩放（保留外推能力）、高频维缩放更多（避免越界）**。实现上不直接改位置，而是改 `θ_i` 的几何级数间距（或等价的 base 调整），数学上等价于对 RoPE 的旋转频率做"非均匀重采样"。

### Dynamic NTK（推理期自适应）

训练仍用原始 base，推理时根据当前序列长度动态把 base 提升到 `base' = base · (L'/L)^α`（α≈0.5~1），长度越长 base 越大，越偏向"插值"而非"外推"。无需微调，配合已有权重即可跑更长。

### YaRN（Yet another RoPE extensioN）

综合 PI 与 NTK，并解决"缩放后高频维在 bf16 下丢失"的问题：

1. **维度特定的缩放因子** `γ_i`：低频维接近 1（不缩放），高频维接近 `L/L'`（强缩放），中间用 ramp 函数平滑过渡。
2. **温度修正（temperature scaling）**：attention 分数除以 `t = 0.1·log(L'/L)+1`，补偿缩放导致的 logits 幅度变化，防止 softmax 过早饱和。

```
rope_dims 分三段：
  i < dim_low  → γ=1            （保留外推）
  dim_low≤i≤dim_high → ramp 线性插值 (1 → L/L')
  i > dim_high → γ = L/L'       （强插值）
```

## 具体例子

以 Llama-7B（d=128, base=10000, 训练 L=4096）扩展到 L'=32768（8×）：

- **PI**：新 base = 10000^8 ≈ 10^32，所有频率统一大幅压缩。短文本分辨率明显下降。
- **NTK-aware**：base 改为 `10000·8^(1/(d/2-2))` ≈ 10^(...)，仅高频维被压缩，低频维几乎不动。
- **YaRN**：对 128 维分别算 γ_i，并设 t≈1.2，继续训练约 1000 步即可稳定支持 32K。

经验结论：**YaRN ≈ NTK 的外推性 + PI 的稳定性 + 温度修正的数值安全**，是 Llama 系长上下文的事实标准。

## 代码速记

```python
# YaRN 的 per-dim 缩放因子（简化）
import math
def yarn_scales(dim, scale, low_frac=0.25, high_frac=0.75):
    # 返回每个旋转维度的缩放 gamma
    out = []
    for i in range(0, dim, 2):
        idx = i / dim
        if idx < low_frac:
            out.append(1.0)
        elif idx > high_frac:
            out.append(scale)
        else:  # ramp 线性过渡
            out.append(scale + (1 - scale) * (high_frac - idx) / (high_frac - low_frac))
    return out  # 长度 dim/2
```

## 面试追问

1. 为什么 ALiBi 能"零成本"外推，而 Sinusoidal/RoPE 直接外推会崩？
2. NTK-aware 相比 PI 改了什么？为什么说 PI 破坏了高频/低频的不对称性？
3. YaRN 的温度修正 t 是在 attention 的哪一步生效？不加会怎样？
4. Dynamic NTK 和 YaRN 的一个关键区别（一个推理期自适应、一个需训练）？
5. RoPE 的 base 越大，外推上限越高还是越低？为什么 Llama-3 把 base 提到 50 万？

## 关联知识点

- [RoPE 旋转位置编码：推导、性质与实现陷阱](position-encoding-rope)
- [长上下文训练：数据、RoPE 调整与序列并行](long-context)
- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
