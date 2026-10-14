---
id: lora-and-qlora
title: LoRA / QLoRA 推导与进阶变体（rsLoRA、DoRA、PiSSA）
module: 08-sft-peft
tags:
- LoRA
- QLoRA
- 低秩分解
- 量化
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: LoRA 用 ΔW=BA 且 B 零初始化保证起点等价；scaling 用 α/r 使适配器输出幅度随 r 衰减，r 大时应改用 α/√r（rsLoRA）；QLoRA
  靠 NF4+双重量化把 7B 压到约 3.6GB。
source:
- https://arxiv.org/abs/2106.09685
questions:
- LoRA 推理时能不能合回原权重？合的时候要注意什么？
- LoRA 的 r 是不是越大越好？
- LoRA 和全参数微调，什么时候必须选后者？
- QLoRA 用 4-bit 量化权重训练，为什么精度损失不大？
- QLoRA 的 NF4 和 GPTQ/AWQ 的 int4 有什么区别？
---

## 一句话结论

LoRA（Low-Rank Adaptation）冻结预训练权重 $W_0$，只训练一个低秩增量 $\Delta W = BA$。三条核心结论：

1. **B 必须零初始化、A 用高斯（或 kaiming）初始化**。这样 $t=0$ 时 $\Delta W = 0$，模型输出与预训练模型**逐位相同**，训练是从基座平滑起步而不是从一个随机扰动起步。且此时 $\partial L/\partial B \neq 0$、$\partial L/\partial A = 0$，第一步只动 B，第二步 A 才开始更新——如果两个都零初始化，两个梯度都是 0，适配器直接死掉。
2. **scaling 系数 $\alpha/r$ 让适配器的输出幅度按 $\alpha/\sqrt{r}$ 衰减**，所以 $r$ 从 8 加到 64 时等效学习率在变小。$r \le 64$ 的常用区间内问题不大，$r$ 上百时应改用 rsLoRA 的 $\alpha/\sqrt{r}$。
3. **QLoRA 的内存账**：NF4 权重 4 bit + 分页优化器 + 双重量化，7B 模型权重仅约 3.6 GB，单张 24GB 卡可全参数质量地微调 7B，48GB 卡可微调 65B。

## 原理推导

### 1. 低秩分解与参数量

设原线性层 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将其替换为：

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

其中 $A \in \mathbb{R}^{r \times k}$（降维），$B \in \mathbb{R}^{d \times r}$（升维），$r \ll \min(d, k)$。

参数量对比：

$$\frac{r(d+k)}{d \cdot k}$$

取 $d = k = 4096$、$r = 8$：$\frac{8 \times 8192}{4096 \times 4096} = \frac{65536}{16777216} = 0.39\%$。

### 2. 零初始化的必要性（链式法则展开）

令 $s = \frac{\alpha}{r} B A x$ 为适配器的输出，记上游梯度 $g = \partial L / \partial s \in \mathbb{R}^d$，则：

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r} \cdot g \, (Ax)^\top \in \mathbb{R}^{d \times r}$$
$$\frac{\partial L}{\partial A} = \frac{\alpha}{r} \cdot B^\top g \, x^\top \in \mathbb{R}^{r \times k}$$

三种初始化组合的结果：

| 初始化 | $t=0$ 时 $\partial L/\partial B$ | $t=0$ 时 $\partial L/\partial A$ | 结果 |
|---|---|---|---|
| $B=0$，$A\sim\mathcal{N}$ | $\neq 0$（因为 $Ax \neq 0$） | $=0$（因为 $B^\top = 0$） | 正常：第一步动 B，第二步 A 跟上 |
| $A=0$，$B\sim\mathcal{N}$ | $=0$ | $\neq 0$ | 也能训，对称情况 |
| $A=B=0$ | $=0$ | $=0$ | 死锁，永远学不到东西 |

### 3. scaling 系数 $\alpha/r$ 到底在补偿什么

设 $A, B$ 的元素独立同分布、方差为 $\sigma^2$。则 $\Delta W = BA$ 的单个元素：

$$\Delta W_{ij} = \sum_{p=1}^{r} B_{ip} A_{pj}$$

是 $r$ 个独立乘积之和，其标准差为 $\sqrt{r}\,\sigma^2$（方差 $r\sigma^4$）。

于是适配器对输出的贡献幅度：

$$\left\| \frac{\alpha}{r}\Delta W x \right\| \;\propto\; \frac{\alpha}{r} \cdot \sqrt{r}\,\sigma^2 \|x\| = \frac{\alpha}{\sqrt{r}}\,\sigma^2\|x\|$$

**结论：$\alpha/r$ 这个缩放让适配器输出随 $r$ 增大而按 $1/\sqrt{r}$ 衰减**。也就是说换更大的 $r$ 时，如果不重调 $\alpha$ 或 lr，适配器实际在"变弱"，等效学习率在变小。这正是大 $r$ 下出现 rank collapse（有效秩塌缩、适配器几乎不学）的原因。

rsLoRA 的修正：把缩放改成 $\alpha/\sqrt{r}$，则

$$\left\| \frac{\alpha}{\sqrt{r}}\Delta W x \right\| \;\propto\; \frac{\alpha}{\sqrt{r}} \cdot \sqrt{r}\,\sigma^2\|x\| = \alpha\sigma^2\|x\|$$

与 $r$ 无关，rank 稳定。$r \le 64$ 时两式差异在 8 倍以内（$\sqrt{64}/1 = 8$ vs $64/1=64$，比值 8 倍），实践中通过调 lr 可以吸收；$r \ge 128$ 后差距显著。

### 4. 秩该怎么选

理论依据是**内在维度假设**（Aghajanyan 等，Intrinsic Dimensionality，2020）：微调一个预训练模型，实际只需要在几百维的低维子空间里搜索就能复现全参数微调的效果。RoBERTa-large 在 MNLI 上约 200 维子空间即可达到全微调 90% 的效果。

LoRA 原论文用**子空间相似度**分析给出了实操结论：

- $r=8$ 学到的 top 奇异方向，与 $r=64$ 学到的 top 奇异方向高度重合；
- $r=64$ 多出来的方向主要是噪声方向（随机且与 $r=8$ 的子空间正交）；
- 在 GPT-3 175B 上，$r=1$ 在部分任务上就接近 $r=64$。

所以经验法则：

| 场景 | 建议 $r$ |
|---|---|
| 格式/风格对齐、分类任务 | 4-16 |
| 通用指令跟随 | 16-32 |
| 数学/代码等需要新能力的任务 | 32-128，甚至直接全参数 |
| 需要**注入新知识**（术语、事实） | LoRA 无法胜任，考虑继续预训练或全参数 |

Biderman 等人（LoRA Learns Less and Forgets Less，2024）的实验结论值得背：在代码和数学这类**需要学新东西**的任务上，LoRA 明显落后于全参数微调；但在**灾难性遗忘**上 LoRA 更轻。也就是说 LoRA 更擅长"改变行为"而不是"增加知识"。

### 5. 哪些层加 LoRA

| 方案 | 层 | 特点 |
|---|---|---|
| 原论文 | 仅 $W_q, W_v$ | 最小改动，效果已不错 |
| QLoRA / 主流实践 | 注意力全部 4 个投影 + MLP 全部 2-3 个投影 | 效果最好，参数量仍只有 0.5% 量级 |
| 加 embedding / lm_head | 词表相关 | 一般不加；参数量暴涨且收益小 |
| 归一化层 / bias | LayerNorm | 通常保持可训练（参数量极小，稳定训练） |

### 6. QLoRA 三件套

**（a）NF4（4-bit NormalFloat）**。普通定点量化假设数据均匀分布，对以 0 为中心的正态分布浪费了大量码字在长尾。NF4 做的是**分位数量化（quantile quantization）**：把标准正态分布按等概率分成 16 份，取每份的分位点作为码字。

bitsandbytes 实际使用的 16 个码字（归一化到 $[-1,1]$）：

```
-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0
```

注意它是**非对称**的：8 个正码字 + 0 + 7 个负码字。原因是生成时用 `linspace(offset, 0.5, 9)[:-1]` 取 8 个正分位点，负侧只取 7 个，凑成 $2^4 = 16$ 个。

量化流程（absmax 量化）：按 64 个元素分块，每块存一个 fp32 的 absmax，

$$c = \max_i |w_i|, \qquad q_i = \text{round}\!\left( \frac{w_i}{c} \to \text{最近的 NF4 码字} \right)$$

反量化 $\hat w_i = c \cdot \text{NF4}[q_i]$。

**（b）双重量化（Double Quantization）**。上面每 64 个元素要存一个 fp32 常数，摊薄成本 $32/64 = 0.5$ bit/参数——比 4 bit 本身还占 12.5%。双重量化把这个 fp32 常数再做一次 8-bit 量化：每 256 个常数共享一个二级 fp32 常数。

$$\text{每参数开销} = 4 + \frac{8}{64} + \frac{32}{64 \times 256} = 4 + 0.125 + 0.00195 = 4.127 \text{ bit}$$

从 4.5 bit 降到 4.127 bit，省下 0.373 bit/参数。

**（c）分页优化器（Paged Optimizer）**。用 NVIDIA 统一内存的页式迁移，在显存峰值（长序列 batch）时把优化器状态逐页搬到 CPU 内存，避免 OOM 尖峰。

**计算精度**：权重以 4-bit 存储，做前向时按块**反量化到 bf16** 再算矩阵乘，梯度也只回传到 bf16 的 LoRA 适配器。所以是"4-bit 存储 + bf16 计算"。

## 具体例子

**例 1：Llama-7B 上给所有线性层加 r=16 的 LoRA，算参数量**

Llama-7B：$d = 4096$，$n_{\text{layers}} = 32$，$d_{\text{ff}} = 11008$。每个矩阵的参数量为 $r \times (d_{\text{in}} + d_{\text{out}})$。

| 矩阵 | 形状 | LoRA 参数量 |
|---|---|---|
| $W_q$ | 4096 → 4096 | $16 \times 8192 = 131072$ |
| $W_k$ | 4096 → 4096 | $131072$ |
| $W_v$ | 4096 → 4096 | $131072$ |
| $W_o$ | 4096 → 4096 | $131072$ |
| $W_{gate}$ | 4096 → 11008 | $16 \times 15104 = 241664$ |
| $W_{up}$ | 4096 → 11008 | $241664$ |
| $W_{down}$ | 11008 → 4096 | $241664$ |

单层合计 $4 \times 131072 + 3 \times 241664 = 524288 + 724992 = 1249280$。

32 层：$1249280 \times 32 = 39{,}976{,}960 \approx 4.0\times10^7$，即 **4000 万参数，占 7B 的 0.57%**。

**例 2：QLoRA 微调 7B 的显存账**

| 项 | 计算 | 大小 |
|---|---|---|
| NF4 量化权重 | $7\times10^9 \times 0.5$ B | 3.50 GB |
| 双重量化常数 | $7\times10^9 \times 0.127 / 8$ B | 0.11 GB |
| LoRA 参数（bf16） | $4.0\times10^7 \times 2$ B | 0.08 GB |
| LoRA 的 Adam 状态（fp32 m+v） | $4.0\times10^7 \times 8$ B | 0.32 GB |
| LoRA fp32 主权重 | $4.0\times10^7 \times 4$ B | 0.16 GB |
| 激活（开 gradient checkpointing） | — | 约 1-3 GB |

合计 **约 4.2 GB + 激活**，24GB 单卡跑 7B 绰绰有余。对比全参数 SFT 的 112 GB（不含激活），省了 26 倍。

**例 3：手算一个 NF4 量化**

权重块（64 个里取 4 个示意）：$w = [0.83, -0.21, 0.05, -0.97]$，$c = \max|w_i| = 0.97$。

归一化：$[0.856, -0.216, 0.0515, -1.0]$。

查 NF4 码字表取最近邻：$0.856 \to 0.7229$（而非 1.0）；$-0.216 \to -0.1848$；$0.0515 \to 0.0796$；$-1.0 \to -1.0$。

反量化：$[0.701, -0.179, 0.077, -0.97]$。绝对误差分别为 $0.013, 0.004, 0.001, 0$。

## 代码速记

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0])


class LoRALinear(nn.Module):
    """在冻结的 nn.Linear 上挂一个低秩分支，前向可合并不留延迟。"""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05, rslora=False):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)          # 冻结预训练权重
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / (r ** 0.5 if rslora else r)   # rsLoRA: alpha/sqrt(r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))  # B 必须零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                               # x: (..., in_features)
        z = self.base(x)                                # (..., out_features)
        h = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)  # 先降维再升维
        return z + self.scaling * h

    def merge(self):
        """推理前把 BA 加回原权重，之后前向与原始 Linear 完全等价、零额外延迟。"""
        dW = self.scaling * (self.lora_B @ self.lora_A)   # (out, in)
        self.base.weight.data += dW
        return self.base


def nf4_quantize(w, block=64):
    """absmax + NF4 最近邻。返回 int 索引、每块 absmax、padding 长度。"""
    flat = w.float().flatten()
    pad = (-flat.numel()) % block
    blocks = F.pad(flat, (0, pad)).view(-1, block)          # (N, block)
    c = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)   # (N, 1)
    normed = blocks / c                                     # 归一化到 [-1, 1]
    idx = (normed.unsqueeze(-1) - NF4).abs().argmin(dim=-1)  # (N, block) 最近邻码字下标
    return idx.to(torch.uint8), c, pad                      # 实际部署两个下标打包进 1 byte


def nf4_dequantize(idx, c, pad):
    """4-bit 存储、bf16 计算：反量化后参与矩阵乘。"""
    q = (NF4[idx.long()] * c).flatten()                     # (N*block,)
    return q[:q.numel() - pad] if pad else q


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(2, 64)
    print("起点等价:", torch.allclose(lora(x), base(x)))     # True，因为 B=0
    lora(x).sum().backward()
    print("B 有梯度:", lora.lora_B.grad.abs().sum().item() > 0)
    print("A 梯度为 0:", lora.lora_A.grad.abs().sum().item() == 0)

    w = torch.randn(128) * 0.1
    idx, c, pad = nf4_quantize(w)
    print("每参数 bit:", (idx.numel() * 4 + c.numel() * 32) / w.numel())
    print("反量化 MSE:", ((nf4_dequantize(idx, c, pad) - w) ** 2).mean().item())
```

## 面试追问

**Q：LoRA 推理时能不能合回原权重？合的时候要注意什么？**
A：能，而且这是 LoRA 相比 Adapter / Prefix-Tuning 的最大工程优势——合并后没有任何额外层，零推理延迟。注意三点：（1）合并前要把 `scaling = α/r` 乘进去；（2）合并要在 fp32 下做再转回 bf16，否则 bf16 下 $W_0 + \Delta W$ 的小量更新会被舍入吃掉（$W$ 的量级远大于 $\Delta W$）；（3）多个 LoRA 想在同一基座上热切换时不能真合并，要用 Punica / S-LoRA 这类把 LoRA 当独立 CUDA 算子 batched 执行的方案。

**Q：LoRA 的 r 是不是越大越好？**
A：不是。$r$ 大到一定程度后指标饱和甚至下降，原因是多出来的奇异方向是噪声方向。同时由于 $\alpha/r$ 的缩放，大 $r$ 会稀释适配器输出。实操做法：先用 $r=16$、$\alpha=32$（即 $\alpha = 2r$）跑基线，如果欠拟合（train loss 降不下去）再往上加 $r$ 到 32/64 并同步调 $\alpha$；如果过拟合就降 $r$ 或加 dropout。

**Q：LoRA 和全参数微调，什么时候必须选后者？**
A：需要向模型注入大量新知识时。LoRA 的容量上限是 $r(d+k)$ 个自由度且只能沿低秩方向修改，写入几万条新事实、学一门新语言、大规模词表扩展，LoRA 都会明显欠拟合。Biderman 等人在代码和数学任务上观测到 LoRA 与全参数微调的差距随数据量增大而拉大。此外，如果基座与目标的分布差异极大（比如从通用文本切到生物医学文献），全参数或继续预训练更合适。

**Q：QLoRA 用 4-bit 量化权重训练，为什么精度损失不大？**
A：两个原因。一是 NF4 用在**信息论上最优**的位置：神经网络权重近似零均值正态分布，分位数量化使每个码字被用到的概率相等，量化熵最大，同码率下误差最小。二是训练时真正被更新的是 **bf16 的 LoRA 适配器**，4-bit 只影响**前向输入到适配器的那条通路**，误差是"静态扰动"而不是"梯度被量化"。反过来，如果 4-bit 也用来存梯度和优化器状态，效果会明显变差。

**Q：QLoRA 的 NF4 和 GPTQ/AWQ 的 int4 有什么区别？**
A：定位不同。NF4 是**存储格式**，设计目标是"量化后还能反量化回来继续训练"，配合分页优化器服务于训练场景，且 NF4 反量化查表开销小、可即时反量化。GPTQ / AWQ 是**后训练量化（PTQ）算法**，目标是纯推理，用二阶信息或激活感知来最小化逐层输出误差，并且通常融合 int4 的 GEMM kernel 直接做低精度矩阵乘。NF4 训练出来的权重不能直接拿去做 int4 推理，需要再做一次 PTQ。

**Q：DoRA 和 LoRA 的差别，为什么它能在同参数量下更好？**
A：DoRA 把权重分解成**幅度（magnitude）和方向（direction）**：$W = m \cdot V/\|V\|_c$，其中 $m \in \mathbb{R}^{1\times k}$ 是逐列范数，$V/\|V\|_c$ 是单位方向。LoRA 只更新方向且方向和幅度耦合在一起（$\Delta W$ 会同时改变两者）；DoRA 单独训一个幅度向量 $m$，方向部分才交给 LoRA。观察到 full FT 的方向变化幅度远小于幅度变化（约 1% vs 20%+），所以解耦后各学各的更高效。代价是训练时不能简单合并——要把 $m$ 重新乘回去，推理时仍可合并，无延迟。

## 关联知识点

- SFT 全流程配方（微调与参数高效）
- Prompt Tuning 家族与 Adapter（微调与参数高效）
- 继续预训练与领域适配（微调与参数高效）
- 量化基础与 PTQ/QAT（推理优化与部署）
- DPO 完整推导（对齐与强化学习）
