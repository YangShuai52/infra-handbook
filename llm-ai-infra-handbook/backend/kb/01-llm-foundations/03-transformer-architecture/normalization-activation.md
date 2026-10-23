---
id: normalization-activation
title: 归一化、残差与激活函数：RMSNorm / Pre-LN / SwiGLU
module: 03-transformer-architecture
tags:
- 归一化
- LayerNorm
- RMSNorm
- 激活函数
- SwiGLU
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: LLM 用 RMSNorm（去掉减均值和 bias，省算力且效果相当）配 Pre-LN（保证梯度可无损回传）；FFN 用 SwiGLU 并把中间维从
  4d 降到 8d/3 以保持参数量不变。
source:
- https://arxiv.org/abs/1706.03762
questions:
- RMSNorm 为什么不减均值了效果还行？
- Pre-LN 有没有缺点？
- 为什么 LLM 里几乎看不到 bias 项？
- SwiGLU 的 SiLU 换成 GELU 行不行？
- 归一化放在 Attention/FFN 之前还是之后，对推理有影响吗？
---

## 一句话结论

三个选择，三个理由：

1. **RMSNorm 取代 LayerNorm**：去掉"减均值"这一步，效果基本无损，但少一次归约、少一个 bias 参数，还能让 kernel 更简单。现代 LLM 几乎全部使用 RMSNorm。
2. **Pre-LN 取代 Post-LN**：残差支路变成从输入直通输出的恒等路径，深层模型不需要精细 warmup 也能训稳。
3. **SwiGLU 取代 ReLU/GELU**：门控机制让 FFN 具备"选择性通过"能力，实验上在同等参数量下 consistently 更好；配套把中间维从 4d 降到 8d/3。

## 原理推导

### 1. BatchNorm → LayerNorm → RMSNorm

**BatchNorm**（在 batch 维归一化）：

```
y = (x - E_B[x]) / sqrt(Var_B[x] + ε) · γ + β
```

对 LLM 不适用的原因根本上是三条：

- 序列长度可变、batch 内 padding，统计量被 padding 污染
- 小 batch（大模型训练 batch 通常只有几到几十）统计量噪声极大
- 推理时需要维护 running mean/var，与自回归生成（batch=1）冲突

**LayerNorm**（在特征维归一化）：

```
μ = (1/d) Σ x_i
σ² = (1/d) Σ (x_i - μ)²
y_i = (x_i - μ) / sqrt(σ² + ε) · γ_i + β_i
```

对每个 token 独立计算，与 batch 无关，天然适配自回归和变长序列。

**RMSNorm**（只按 RMS 缩放）：

```
rms(x) = sqrt( (1/d) Σ x_i² )
y_i = x_i / rms(x) · γ_i
```

注意 RMSNorm 做了两处简化：

- **去掉减均值 μ**：不重新中心化，只做缩放。原论文的解释是 LayerNorm 的成功主要来自"re-scaling invariance"（缩放不变性）而非"re-centering invariance"（平移不变性）。
- **去掉 bias β**：少 d 个参数。

计算量对比（每层每 token）：

```
LayerNorm: 求均值（d 次加）+ 求方差（d 次乘加）+ 归一化（含减法）
RMSNorm:   求平方和（d 次乘加）+ 归一化（无减法）
```

实测在 GPU 上 RMSNorm 比 LayerNorm 快约 10%~40%（取决于实现和 d 大小），主要收益不是省 FLOPs，而是**少一次全局归约同步**，kernel 启动和访存更省。

RMSNorm 的一个隐含假设：输入均值接近 0。这在 Pre-LN 结构下基本成立（残差流经过多次归一化），但在某些位置（如残差分支方差累积很大时）会有偏差，实践中不影响效果。

### 2. Pre-LN vs Post-LN 的梯度分析

**Post-LN**（原始 Transformer）：

```
x_{l+1} = LayerNorm(x_l + F(x_l))
```

**Pre-LN**（现代 LLM）：

```
x_{l+1} = x_l + F(LayerNorm(x_l))
```

关键差异在反向传播时残差支路的梯度。设 `y = LayerNorm(x + F(x))`，对 Post-LN 有：

```
∂y/∂x = ∂LN/∂u · (I + ∂F/∂x)    其中 u = x + F(x)
```

`∂LN/∂u` 的范数约为 `1/σ_u`，而 `σ_u`（残差加之后、归一化之前的方差）在深层会显著大于 1（因为每层都在累加）。于是**靠近输出层的梯度被放大 `1/σ` 倍，靠近输入层的梯度被逐层连乘 `1/σ` 衰减**。

对 Pre-LN，反向传播沿残差支路：

```
∂L/∂x_l = ∂L/∂x_{l+1} · (I + ∂F/∂x_l · ∂LN/∂x_l)
        = ∂L/∂x_{l+1} + （子层贡献）
```

第一项是**恒等映射**，梯度可以无损穿过多层直达输入。这就是为什么 Pre-LN 对 warmup 和学习率鲁棒。

代价：Pre-LN 没有归一化压制残差流的方差，`σ` 会随层数单调增长。解决办法是**缩放初始化**：把残差分支的输出按 `1/sqrt(2L)` 缩放（如 `residual = residual + out / sqrt(2 * n_layers)`，GPT-2 的做法），或者对 W_o 和 W_down 用 `std ∝ 1/sqrt(2L)` 初始化。

DeepNorm 是 Post-LN 的改良：把残差加改成 `x + α·F(x)` 并把 `α` 与层数绑定（如 `α = (2L)^{1/4}`），理论可训到 1000 层，但实践中 Pre-LN 更简单，未成为主流。

### 3. 激活函数：从 ReLU 到 SwiGLU

**ReLU**：`max(0, x)` — 死神经元问题，负半轴梯度为 0
**GELU**：`x · Φ(x) ≈ x · σ(1.702x)` — 平滑，GPT/BERT 时代主流
**SiLU / Swish**：`x · σ(x)` — 比 GELU 计算更简单（一个 sigmoid），效果相当

**GLU 家族**（门控线性单元）：

```
GLU(x)    = (xW_1) ⊗ σ(xW_2)
SwiGLU(x) = (xW_1) ⊗ SiLU(xW_2) = (xW_gate) ⊗ (SiLU(xW_gate') · ...)
```

标准写法（三个矩阵）：

```
FFN_SwiGLU(x) = ( SiLU(x·W_gate) ⊙ (x·W_up) ) · W_down
```

为什么门控有效：SwiGLU 提供了**逐维的选择性**——gate 分支可以学到"这一维该不该通过"，等价于给 FFN 加了一个数据相关的稀疏门。而 ReLU/GELU 的非线性是固定形状的逐元素函数，没有这种数据相关的路由能力。

参数量代价：SwiGLU 有 3 个矩阵，是标准 FFN（2 个）的 1.5 倍。为保持总参数量不变，把中间维从 `4d` 降到：

```
3 · d · d_ff_new = 2 · d · 4d    →    d_ff_new = 8d/3 ≈ 2.67d
```

LLaMA 取 `d_ff = round(8d/3 / 256) * 256`：d=4096 → 10922.67 → 11008。

## 具体例子

**例 1：LayerNorm vs RMSNorm 的手算**

x = [1, 2, 3, 4]，d=4，γ=1，β=0，ε=0（便于手算）：

```
LayerNorm:
  μ = 2.5
  σ² = ((-1.5)² + (-0.5)² + 0.5² + 1.5²) / 4 = (2.25+0.25+0.25+2.25)/4 = 1.25
  y = [-1.342, -0.447, 0.447, 1.342]

RMSNorm:
  rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.739
  y = [0.365, 0.730, 1.095, 1.461]
```

注意 RMSNorm 的输出**均值不为 0**（均值 = 0.913），这是两者的本质差异。

**例 2：Pre-LN 的方差累积**

假设每层 FFN 输出的方差为 σ_F² = 1，残差流方差递推（Pre-LN 后归一化使每层输入方差≈1）：

```
Var[x_L] ≈ Var[x_0] + L · σ_F²
```

L=32 时 `Var ≈ 33`（若 x_0 方差为 1），标准差 ≈ 5.7。这就是为什么需要 scaled 初始化——否则深层输入到 Norm 之前的数值范围过大，bf16 精度不足。

**例 3：SwiGLU 参数量核对（LLaMA-7B）**

```
d = 4096, d_ff = 11008
SwiGLU:   3 × 4096 × 11008 = 135,266,304
标准FFN(4d=16384): 2 × 4096 × 16384 = 134,217,728
差异: 0.78%  → 基本持平，验证了 8d/3 的选择
```

**例 4：RMSNorm 参数量占比**

7B 模型里所有 RMSNorm 的 γ 参数：

```
2 个 Norm/层 × 32 层 × 4096 + 1 个 final norm × 4096 = 65 × 4096 = 266,240
```

占总参数 0.004%——**归一化层参数量完全可以忽略**，它的价值在稳定性而不在容量。

## 代码速记

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))     # 只有 γ，没有 β
    def _norm(self, x):
        # 计算时必须 upcast 到 fp32：bf16 下平方和容易溢出/精度不足
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff=None, multiple_of=256):
        super().__init__()
        # 8d/3 并按 256 对齐，与 LLaMA 一致
        if d_ff is None:
            d_ff = int(8 * d / 3)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up   = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

def scaled_init(module, n_layers, std=0.02):
    """Pre-LN 的残差分支缩放初始化：让 Var[x_L] 不随 L 爆炸"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std / (2 * n_layers) ** 0.5)

# 三种归一化对比（数值）
x = torch.tensor([1., 2., 3., 4.])
print(F.layer_norm(x, (4,)))                       # 均值归零
print(RMSNorm(4)._norm(x.float()))                 # 均值不为零
```

## 面试追问

**Q：RMSNorm 为什么不减均值了效果还行？**
A：LayerNorm 带来两个不变性——re-centering（平移不变）和 re-scaling（缩放不变）。RMSNorm 论文的核心论点是：**起作用的其实是 re-scaling**，re-centering 是附带产物。去掉减均值后模型仍能保持每层输入尺度稳定，这就够了。而且 Pre-LN 结构下残差流本身近似零均值（多层累加的中心极限效应），减均值的收益进一步变小。

**Q：Pre-LN 有没有缺点？**
A：有，主要是两点。(1) 激活值方差随层数累积，深层输出数值范围大，对低精度训练/推理不友好，需要用 scaled 初始化补偿。(2) 有研究工作指出 Pre-LN 的"有效残差深度"比 Post-LN 浅——因为恒等路径太顺畅，深层容易退化成"近似什么都不做"，最终模型的表征能力略逊于训稳后的 Post-LN。但工程上 Pre-LN 的稳定性收益远大于这点损失。

**Q：为什么 LLM 里几乎看不到 bias 项？**
A：三个原因。(1) 与 RMSNorm/LayerNorm 的 β 功能重叠，归一化已经提供了平移自由度；(2) 大模型训练对数值稳定性敏感，去掉 bias 能减少一条梯度路径；(3) 省参数（虽然只占 0.01%）。LLaMA 起带头作用后基本成了行业惯例。

**Q：SwiGLU 的 SiLU 换成 GELU 行不行？**
A：行，叫 GeGLU，效果与 SwiGLU 接近（原论文里在部分任务上 GeGLU 略优）。选 SiLU 的理由纯粹是计算成本：sigmoid 比 GELU 的精确版（含 erf）便宜得多，而 GELU 的 tanh 近似版虽然也便宜但仍比 sigmoid 多几次运算。大模型 FFN 的 FLOPs 占比最高，这点差异会被放大。

**Q：归一化放在 Attention/FFN 之前还是之后，对推理有影响吗？**
A：推理的数值结果完全等价（只是计算顺序不同），但 kernel 实现上有差别：Pre-LN 下 Norm 的输入是上一层的残差输出，可以和前面的 kernel 融合；Post-LN 下 Norm 在残差加之后，多一次访存。另外量化时两者差异明显——Post-LN 保证残差输出始终被归一化，量化误差不会累积；Pre-LN 的残差流方差大，量化需要更谨慎。

**Q：LayerNorm 的 ε 为什么要加在 sqrt 里面？**
A：`(x-μ)/sqrt(σ²+ε)` 而不是 `(x-μ)/(sqrt(σ²)+ε)`。加在根号内是标准做法，作用是当 σ²→0 时避免除零，且 ε 的量纲与方差一致（1e-5 ~ 1e-6）。bf16 训练时 ε 有时要调大到 1e-5，因为 bf16 的尾数只有 8 位，小方差的相对误差大。

## 关联知识点
- Transformer 结构与参数量计算（01-transformer）
- 训练稳定性：混合精度、loss spike、初始化（03-pretrain）
- 量化：SmoothQuant / GPTQ / AWQ 中的归一化处理（07-inference）
- 手写 RoPE / RMSNorm / SwiGLU（10-coding）
