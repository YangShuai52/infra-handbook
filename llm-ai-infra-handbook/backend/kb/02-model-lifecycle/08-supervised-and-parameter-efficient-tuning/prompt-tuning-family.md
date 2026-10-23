---
id: prompt-tuning-family
title: Prompt Tuning / Prefix Tuning / P-Tuning v2 与 Adapter 家族
module: 08-sft-peft
tags:
- 软提示
- Prefix
- Adapter
- PEFT
- 高频
difficulty: 4
freq: 中
updated: 2026-08-28
summary: 软提示只训 0.01%-0.3% 参数但推理有延迟且小模型上失效；Prefix 的重参数化是为训练稳定性；LoRA 能合并所以部署最省事。
source:
- https://arxiv.org/abs/2106.09685
questions:
- 为什么 Prompt Tuning 在小模型（< 10B）上明显不如 full fine-tuning？
- Prefix Tuning 和 P-Tuning v2 到底有什么区别？
- LoRA 和 Adapter 怎么选？
- 软提示的"可解释性"到底是好还是差？
- 多任务场景下三种方法怎么切换？
---

## 一句话结论

这一类方法（soft prompt / prefix / adapter）共同的思想是：**冻结整个预训练模型，只训练一小撮新增参数**。三者的差别只在于"这一小撮参数插在哪、怎么插"。

1. **Prompt Tuning 只在输入 embedding 前拼 token，Prefix Tuning 在每一层的 K/V 前拼，P-Tuning v2 = deep prefix + 去掉重参数化 + 多任务，把适用范围从 10B+ 下探到 300M。**
2. **Prefix 的重参数化（用一个 MLP 生成前缀，训完丢弃 MLP）是为了训练稳定性**：直接优化前缀对初始化和学习率极敏感，容易不收敛；映射到低维连续空间后优化曲面平滑很多。训完算参数量时只算前缀本身，MLP 不计入。
3. **和 LoRA 比，软提示类方法的最大硬伤是推理有额外开销**（占序列长度、增加 KV Cache、增加 attention 计算），而 LoRA 能合并进原权重做到零延迟。所以今天工程上 LoRA 是默认选择，软提示主要用于多任务热切换和"不想碰模型权重"的场景。

## 原理推导

### 1. Prompt Tuning：最省参数的方案

在输入 embedding 序列前面拼接 $k$ 个**可训练的虚拟 token** $P \in \mathbb{R}^{k \times d}$：

$$\text{input} = [\,P;\; E(x_1), E(x_2), \dots, E(x_n)\,]$$

模型完全冻结，只训 $P$，参数量 $k \times d$。T5-XXL（11B，$d=4096$）取 $k=20$ 只有 81920 个参数，占 $0.00074\%$。

代价：这 $k$ 个位置占掉上下文长度，且推理时要和真实 token 一起做 attention。

### 2. Prefix Tuning：把可训练参数下沉到每一层

对第 $l$ 层，在 Key / Value 前面拼接可训练前缀 $P_k, P_v \in \mathbb{R}^{m \times d}$（$m$ 是前缀长度）：

$$K' = [P_k;\; xW_k], \qquad V' = [P_v;\; xW_v] \in \mathbb{R}^{(m+n) \times d}$$

$$\text{head} = \text{softmax}\!\left(\frac{Q K'^\top}{\sqrt{d_h}}\right) V'$$

参数量 $L \times 2 \times m \times d$（$L$ 为层数）。

**关键推导：前缀如何改变 attention 输出。**
把 softmax 沿拼接维度拆成两段。令 $a = QP_k^\top/\sqrt{d_h} \in \mathbb{R}^{n \times m}$，$b = QK^\top/\sqrt{d_h} \in \mathbb{R}^{n \times n}$，配分函数 $Z = \sum_i e^{a_i} + \sum_j e^{b_j}$。则输出的第 $t$ 行：

$$\text{out}_t = \frac{\sum_i e^{a_{ti}} P_v[i] + \sum_j e^{b_{tj}} V[j]}{Z}
= \lambda_t \cdot \underbrace{\text{softmax}(a_t) P_v}_{\text{前缀项（与内容无关的凸组合）}} \;+\; (1-\lambda_t) \cdot \underbrace{\text{softmax}(b_t)V}_{\text{原本的 attention 输出}}$$

其中 $\lambda_t = \sum_i e^{a_{ti}} / Z \in (0,1)$。

这个式子说明两件事：

- 前缀项 $\text{softmax}(a_t)P_v$ 是 $m$ 个前缀 value 向量**与输入内容无关**的凸组合，本质是一个可学习的"偏置/引导向量"，作用是给标准 attention 输出加一个常数偏移。
- 同时它把原本的 attention 输出按 $(1-\lambda_t)$ **缩小**了。所以 $\lambda_t$ 不能太大，否则内容 attention 被压没。这解释了为什么前缀需要小心初始化（过大 $\lambda$ 直接废掉模型），也解释了为什么需要重参数化。

### 3. 为什么必须重参数化（reparameterization）

直接把 $P$ 当自由参数优化有两个问题：

- **初始化敏感**：随机初始化的 $P$ 会立刻被 attention 看到，$\lambda_t$ 在大值前缀下接近 1，模型输出被前缀主导，loss 从很高处起飞且震荡。
- **优化曲面崎岖**：$P$ 的维度是 $L \times 2 \times m \times d$，在高维空间里直接用 SGD/Adam 找方向，等价于在一个几乎无平滑先验的空间搜索。

做法是引入一个小矩阵 $P' \in \mathbb{R}^{m \times d'}$（$d' \ll d$）和一个 MLP：

$$P[i,:] = \text{MLP}_\theta(P'[i,:]), \qquad \text{MLP}: \mathbb{R}^{d'} \to \mathbb{R}^{d}$$

训练时优化 $(P', \theta)$，**训完只保留 $P = \text{MLP}(P')$ 并丢弃 MLP**。所以 MLP 的参数不算进最终模型大小，只是训练期的"脚手架"。

### 4. P-Tuning v2 相对于 Prefix Tuning 的改动

| 改动 | 原因 |
|---|---|
| 去掉（或弱化）重参数化 MLP | 在小模型和 NLU 任务上 MLP 反而导致不稳定；实测直接优化前缀 + 合适的 lr 更好 |
| 每层都加（deep prompt） | 沿用 prefix 的做法，浅层 prompt 容量不足 |
| 引入多任务预训练 | 小模型上单个任务的 soft prompt 很难训，先用多任务共享 prompt 预热 |
| 学习率显著更大（$10^{-2}$ 量级） | 可训练参数量极小，需要更大步长；与 full FT 的 $10^{-5}$ 完全不同 |

### 5. Adapter：插串行模块

Houlsby 等（2019）在每个 transformer block 内插入一个瓶颈结构：

$$h \leftarrow h + f(h W_{\text{down}}) W_{\text{up}}, \quad W_{\text{down}} \in \mathbb{R}^{d \times r},\; W_{\text{up}} \in \mathbb{R}^{r \times d},\; r \ll d$$

这里非线性 $f$（ReLU/GELU）是**必须的**——如果去掉 $f$，整个结构退化成 $h + h W_{\text{down}}W_{\text{up}}$，就是一个低秩增量，等价于 LoRA。正是这点非线性让 Adapter 与 LoRA 分道扬镳，也是它**无法合并进原权重**、推理必然有延迟的原因。为了保证起点等价，$W_{\text{up}}$ 通常零初始化（与 LoRA 的 $B=0$ 同理）。

## 具体例子

**例 1：三种方法的参数量实测对比**

| 方法 | 基座 | 配置 | 可训练参数量 | 占比 |
|---|---|---|---|---|
| Prompt Tuning | T5-XXL 11B，$d=4096$ | $k=20$ | $20 \times 4096 = 81920$ | 0.00074% |
| Prefix Tuning | GPT-2-Large 774M，$L=36$，$d=1280$ | $m=20$ | $36 \times 2 \times 20 \times 1280 = 1{,}843{,}200$ | 0.24% |
| P-Tuning v2 | BERT-Large 335M，$L=24$，$d=1024$ | $m=20$ | $24 \times 2 \times 20 \times 1024 = 983{,}040$ | 0.29% |
| Adapter | BERT-Base 110M，$d=768$，$L=12$，每层 2 个 | $r=48$ | $12 \times 2 \times (2 \times 768 \times 48) = 1{,}769{,}472$ | 1.6% |
| LoRA | Llama-7B | $r=16$，全线性层 | $39{,}976{,}960$ | 0.57% |

Adapter 的瓶颈维度 $r$ 是唯一旋钮，参数占比约 $2r/d \times 2$（每层两个 adapter），$r=48, d=768$ 时约 25%... 上面算出 1.6%，因为分母是全模型（含 embedding 和 12 层的 attention/FFN，不只是 12 个 $d$ 维向量）。

**例 2：软提示的推理开销**

前缀长度 $m=20$，处理序列长 $n=512$：

- KV Cache 增加：$20/512 = 3.9\%$；
- self-attention 的 FLOPs 正比于 $(m+n)^2$，增加 $(532/512)^2 - 1 = 1.039^2 - 1 = 7.96\%$；
- 由于前缀对同一任务是共享的，可以被 prefix caching 缓存一次，多请求摊薄后实际开销接近 0（但这要求同一批请求同任务）。

对比 LoRA：合并后 FLOPs 增量 0，KV Cache 增量 0。

**例 3：Prefix 的 $\lambda_t$ 有多大**

设 $m=20$，前缀 value 初始化为小随机值（std 0.01），$P_k$ 也小，则 $a_{ti} \approx 0$，$e^{a_{ti}} \approx 1$，$\sum_i e^{a_{ti}} \approx 20$。而内容部分 512 个位置，即使完全均匀 $e^{b_{tj}}$ 的平均量级也是 $O(1)$，$\sum_j e^{b_{tj}} \approx 512$。于是

$$\lambda_t \approx \frac{20}{20 + 512} = 3.8\%$$

即前缀项被缩小到 3.8% 权重。**这就是为什么前缀必须初始化得足够"响亮"**——太小则学不动，太大则破坏内容 attention。这是软提示类方法对初始化和 lr 敏感的根因。

## 代码速记

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixTuning(nn.Module):
    """每层 K/V 前拼接可训练前缀；用 MLP 重参数化，训完只保留前缀本身。"""

    def __init__(self, n_layers, n_heads, head_dim, prefix_len=20, rep_dim=512):
        super().__init__()
        d = n_heads * head_dim
        self.L, self.H, self.Dh, self.m = n_layers, n_heads, head_dim, prefix_len
        # P' 低维自由参数 (L, 2, m, rep_dim)，第 1 维区分 K 前缀 / V 前缀
        self.P_prime = nn.Parameter(torch.randn(n_layers, 2, prefix_len, rep_dim) * 0.01)
        # 每层一个 MLP：rep_dim -> d，训完丢弃，不计入最终参数量
        self.mlp_k = nn.ModuleList([nn.Sequential(nn.Linear(rep_dim, rep_dim),
                                                 nn.Tanh(), nn.Linear(rep_dim, d))
                                   for _ in range(n_layers)])
        self.mlp_v = nn.ModuleList([nn.Sequential(nn.Linear(rep_dim, rep_dim),
                                                 nn.Tanh(), nn.Linear(rep_dim, d))
                                   for _ in range(n_layers)])

    def forward(self, batch):
        """返回 List[(pk, pv)]，每个 (B, H, m, Dh)，供每层 attention 前拼接。"""
        out = []
        for li in range(self.L):
            pk = self.mlp_k[li](self.P_prime[li, 0])          # (m, d)
            pv = self.mlp_v[li](self.P_prime[li, 1])          # (m, d)
            pk = pk.view(self.m, self.H, self.Dh).transpose(0, 1)   # (H, m, Dh)
            pv = pv.view(self.m, self.H, self.Dh).transpose(0, 1)   # (H, m, Dh)
            pk = pk.unsqueeze(0).expand(batch, -1, -1, -1)          # (B, H, m, Dh)
            pv = pv.unsqueeze(0).expand(batch, -1, -1, -1)
            out.append((pk, pv))
        return out


def prepend_prefix(k, v, prefix_k, prefix_v):
    """把前缀拼到 K/V 前面；attention_mask 也要在前面补 m 个 1。"""
    return torch.cat([prefix_k, k], dim=2), torch.cat([prefix_v, v], dim=2)


def attention_with_prefix(q, k, v, prefix_k, prefix_v, attn_mask=None):
    """q/k/v: (B, H, n, Dh)，prefix_*: (B, H, m, Dh)，返回 (B, H, n, Dh)。"""
    B, H, n, Dh = q.shape
    m = prefix_k.shape[2]
    k2, v2 = prepend_prefix(k, v, prefix_k, prefix_v)          # (B, H, m+n, Dh)
    scale = 1.0 / math.sqrt(Dh)
    scores = torch.matmul(q, k2.transpose(-1, -2)) * scale     # (B, H, n, m+n)
    if attn_mask is not None:                                  # 需要在前缀位补 1
        B1, n1 = attn_mask.shape
        pad = attn_mask.new_ones(B1, m)
        attn_mask = torch.cat([pad, attn_mask], dim=1)
        scores = scores.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))
    return torch.matmul(F.softmax(scores, dim=-1), v2)


class Adapter(nn.Module):
    """瓶颈结构 + 残差；up 零初始化保证 t=0 输出等于输入。"""

    def __init__(self, d, r=48):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.up.weight)          # 起点等价：h + 0 = h
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


if __name__ == "__main__":
    B, H, n, Dh = 2, 12, 16, 64
    pt = PrefixTuning(n_layers=2, n_heads=H, head_dim=Dh, prefix_len=20)
    q = torch.randn(B, H, n, Dh)
    k = v = torch.randn(B, H, n, Dh)
    pk, pv = pt.forward(B)[0]
    print("输出形状:", attention_with_prefix(q, k, v, pk, pv).shape)   # (2, 12, 16, 64)
    print("前缀参数量:", pt.P_prime.numel() * 2 * 64 // 512)            # 仅 P 本身（部署时）
    ad = Adapter(768)
    h = torch.randn(2, 768)
    print("Adapter 起点等价:", torch.allclose(ad(h), h))
```

## 面试追问

**Q：为什么 Prompt Tuning 在小模型（< 10B）上明显不如 full fine-tuning？**
A：两个原因。一是容量：可训练参数只有 $k \times d$，T5-Base 上 $k=20, d=768$ 仅 15360 个自由度，根本不足以编码一个任务的变换。二是机制：软提示本质是在"告诉"模型该做什么，依赖模型本身已经具备的 in-context learning 能力；10B+ 模型本身就擅长读 few-shot 示例，soft prompt 相当于把离散示例蒸馏成了连续向量，所以能 work；小模型没有这个能力，必须真正改权重。P-Tuning v2 通过 deep prompt（每层都加）+ 多任务预热把这个门槛下探到了 300M。

**Q：Prefix Tuning 和 P-Tuning v2 到底有什么区别？**
A：结构上几乎一样（都是每层 K/V 前拼前缀，都来自"deep prompt"这个思路）。差别在实现细节：Prefix 保留 MLP 重参数化，P-Tuning v2 去掉了它；Prefix 在生成任务上验证，P-Tuning v2 在 NLU 任务和小模型上验证；P-Tuning v2 强调多任务预训练和更大的 lr。可以认为 P-Tuning v2 = Prefix Tuning 面向 NLU 和小模型的改良版。

**Q：LoRA 和 Adapter 怎么选？**
A：默认选 LoRA。Adapter 有两个硬伤：一是推理时多一个串行模块（含非线性），无法合并，小 batch 场景延迟增加可达 5%-20%（batch 大时可以部分被吞吐掩盖）；二是参数量通常比 LoRA 高一个量级。Adapter 的适用场景是需要更强的非线性变换能力、或者要严格保证基座权重一个字节都不动（比如多租户共享同一份基座权重、只分发 adapter 模块）。

**Q：软提示的"可解释性"到底是好还是差？**
A：比 LoRA 好一点，但远没有宣传的那么好。常见做法是对学到的 soft prompt 向量在词表 embedding 里做最近邻检索，看它"像哪个词"。但检索出的词通常语义不连贯，而且不同层的对应词不一致，很难串成一句人话。本质上 soft prompt 是在连续空间中做优化，它不需要落在任何真实 token 的 embedding 上，所以"翻译成 token"这件事本身就不成立。

**Q：多任务场景下三种方法怎么切换？**
A：软提示和 Adapter 都极易切换——只需替换那一小撮参数，基座权重完全不变，可以在一份基座上挂几百个任务的模块，服务时按请求动态加载。LoRA 如果不合并也可以用 Punica/S-LoRA 这类 batched LoRA kernel 服务多适配器；但如果合并了，切换就要 unmerge（需要保存未合并的原始权重副本，或者用 $W - \frac{\alpha}{r}B_{\text{old}}A_{\text{old}} + \frac{\alpha}{r}B_{\text{new}}A_{\text{new}}$，多次切换会累积数值误差）。

**Q：如果完全不用 PEFT，只靠 few-shot prompting，什么时候够用？**
A：任务可以靠上下文里的示例说清楚、且不需要改变输出分布的长尾特征时够用。判断标准：如果 prompt 里加 5-10 个示例后指标就饱和了，说明模型已经具备该能力，不需要训练；如果加示例只带来边际改善、且错误类型是"系统性"的（格式不对、语气不对、拒绝回答），那就要 SFT/PEFT，因为 few-shot 改不了长期行为。

## 关联知识点

- LoRA 与 QLoRA 原理（微调与参数高效）
- SFT 全流程配方（微调与参数高效）
- 继续预训练与领域适配（微调与参数高效）
- 投机解码与推理加速（推理优化与部署）
- Continuous Batching 与推理引擎（推理优化与部署）
