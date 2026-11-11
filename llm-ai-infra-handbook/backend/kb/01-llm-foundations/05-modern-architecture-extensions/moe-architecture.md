---
id: moe-architecture
title: MoE 混合专家：路由、负载均衡与 DeepSeekMoE
module: 05-modern-architecture
tags:
- MoE
- 稀疏激活
- 路由
- 负载均衡
- DeepSeek
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: MoE 用稀疏激活解耦参数量与计算量（激活 1/8 参数达到同等效果），核心难题是负载均衡（辅助损失 + 容量因子）和 all-to-all 通信；DeepSeekMoE
  用细粒度专家 + 共享专家显著提升专家特化。
source:
- https://arxiv.org/abs/2101.03961
questions:
- MoE 省显存吗？
- 为什么 MoE 训练比稠密模型不稳定？
- top-k 的 k 怎么选？
- MoE 的专家真的学到了专门的功能吗？
- MoE 适合小模型吗？
---

## 一句话结论

MoE 的价值可以用一句话概括：**用稀疏激活把"参数量"和"计算量"解耦**。

```
稠密模型:  参数量 N,  每 token FLOPs ∝ N
MoE 模型:  参数量 8N, 每 token FLOPs ∝ N（只激活 1/8）
```

所以 MoE 是在**同样的推理算力下塞进更多参数（更多知识容量）**。

代价有三个，面试必须能说全：

1. **显存按总参数算**（不是激活参数），所以 MoE 省算力不省显存
2. **训练不稳定**（路由是离散决策，梯度噪声大）
3. **通信开销**（all-to-all，专家并行时跨卡搬运 token）

## 原理推导

### 1. 基本结构

MoE 层替换 FFN 层（不是替换 attention）。结构：

```
输入 x → Router（门控网络）→ 选出 top-k 个专家
       → 把这 k 个专家的输出加权求和
```

数学形式（E 个专家，top-k 路由）：

```
g = Softmax(W_g · x)                      # 门控分数, (E,)
top_k_idx = TopK(g, k)                    # 选分数最高的 k 个
g' = 归一化(top-k 的分数)                  # 通常 renormalize
y = Σ_{i ∈ top_k_idx} g'_i · FFN_i(x)
```

**为什么只替换 FFN 不替换 attention**：
- FFN 占参数量 2/3，是"知识存储"的主要位置（这个观点有实验支持）
- Attention 的 KV Cache 机制与 MoE 的路由冲突（不同 token 走不同专家，KV 无法复用）
- 实践上 MoE-Attention 收益不明显

**层数安排**：不是每层都 MoE。常见做法是**每隔一层**（或每 1/4 层）放 MoE，其余保持稠密，平衡稳定性和容量。

### 2. 为什么稀疏激活可行

关键假设：**不同的输入需要不同的"知识"**。稠密 FFN 对每个 token 都用全部参数，而实际上一个 token 只需要其中一小部分。

定量关系（Switch Transformer 的结论）：在相同 FLOPs 预算下，MoE 的 loss 显著低于稠密模型；在相同参数量下，MoE 的训练速度快数倍。

一个有用的心智模型：

```
稠密 7B:   每次前向用 7B 参数算
MoE 8×7B:  每次前向用 7B 参数算（激活 1/8），但可以从 56B 的"知识库"里挑
```

### 3. 负载均衡（核心难题）

**问题**：路由是"赢者通吃"的——训练初期某些专家偶然被选中更多，就学得更好，然后被选中更多，形成正反馈。最终：

```
少数专家过载（大量 token 被丢弃）
多数专家饿死（几乎收不到 token，学不到东西）
```

这叫**路由坍塌（routing collapse）**，是 MoE 训练的頭号问题。

**对策一：辅助负载均衡损失**

Switch Transformer 的经典形式。设第 i 个专家在一批数据（含 T 个 token）中：

```
f_i = 分配给专家 i 的 token 比例（含路由权重）
P_i = 专家 i 的平均路由概率（softmax 输出）
```

辅助损失：

```
L_aux = α · E · Σ_{i=1}^{E} f_i · P_i
```

**推导为什么它有用**：在均匀分布下（`f_i = 1/E`, `P_i = 1/E`）：

```
L_aux = α · E · Σ (1/E)(1/E) = α · E · E · (1/E²) = α
```

在完全坍塌（所有 token 给专家 1）时：`f_1 = 1, P_1 = 1`，其余为 0：

```
L_aux = α · E · (1×1) = α·E
```

即**坍塌时损失是均匀的 E 倍**，最小化这个损失就是在推动均匀分布。而且这个损失对 `f` 和 `P` 都是可微的（`f` 通过 top-k 的软权重，`P` 通过 softmax），所以梯度能传回 router。

`α` 的典型取值：1e-2 ~ 1e-3（Switch 用 0.01）。太大会压过主 loss，太小不起作用。

**对策二：容量因子（Capacity Factor）**

每个专家每批最多处理：

```
capacity = CF × (T / E) × k
```

CF = 1.0 时正好平均分配；CF = 1.25 是常用值（留余量）。超出容量的 token：

```
方案 A（drop）:  直接丢弃，该 token 走残差分支直通（信息损失）
方案 B（pad）:   补 dummy token（浪费算力）
方案 C（无丢弃）: 用 token 合并/动态容量，现代实现更倾向这个
```

Drop 比例是 MoE 训练的重要监控指标，超过 5%~10% 说明路由严重失衡。

**对策三：专家级 z-loss / router z-loss**

类似 logits 的 z-loss，压制 router logits 的量级漂移：

```
L_z = λ · (log Σ exp(router_logits))²
```

ST-MoE 提出，能显著改善训练稳定性。

**对策四：噪声路由 / 随机路由**

训练时给 router logits 加噪声（如 `g = Softmax(W_g x + ε)`, `ε ~ N(0, σ²)` 或用 Gumbel），鼓励探索。

### 4. DeepSeekMoE 的两个关键改进

**改进一：细粒度专家分割（Fine-grained Expert Segmentation）**

把每个专家的 FFN 中间维缩小到 1/m，同时把专家数扩大 m 倍，并激活 k×m 个：

```
传统:  E=16 专家,  top-2 激活,  每个专家 d_ff
DeepSeek: E=16m 专家, top-2m 激活, 每个专家 d_ff/m
```

总参数量和激活参数量都不变，但**组合数爆炸式增长**：

```
传统组合数: C(16, 2) = 120
DeepSeek(m=4): C(64, 8) ≈ 4.4e9
```

组合数越多，专家越容易分化出专门功能，路由的表达能力越强。

**改进二：共享专家（Shared Expert）**

设 K_s 个专家**永远被激活**（不参与 top-k 选择），所有 token 都经过它们：

```
y = Σ_{i ∈ shared} FFN_i(x) + Σ_{j ∈ top_k} g_j · FFN_j(x)
```

直觉：共享专家承载"通用知识"（所有 token 都用得上的基础处理能力），路由专家承载"专业知识"。这样路由专家不必重复学习通用能力，可以更专注地特化。

DeepSeek-V3 的配置（公开数据）：

```
专家总数:  256 个路由专家 + 1 个共享专家
激活数:    8 个路由专家 + 1 个共享专家
每 token 激活参数: 37B（总参数 671B）
```

### 5. 通信开销（专家并行 EP）

专家并行时，token 需要先送到对应专家所在的 GPU，算完再送回来。这两次搬运就是 **all-to-all**：

```
第一次 all-to-all:  把 token 按路由结果发送到目标专家所在卡
（本地计算）
第二次 all-to-all:  把计算结果送回原卡
```

通信量估算（每 MoE 层，每 token，bf16）：

```
每个 token 激活 k 个专家，每个方向的向量维度 d
发送: k × d × 2 bytes
接收: k × d × 2 bytes
合计: 4 × k × d bytes per token per layer
```

对比稠密模型 TP 的 all-reduce（每 token 每层约 `2 × d × 2 × (某个常数)`），MoE 的通信量明显更大，且 **all-to-all 比 all-reduce 更难优化**（不是规约操作，是数据重排，无法用 ring 算法减少）。

关键结论：**MoE 在节点内（NVLink 高带宽）做 EP 可以，跨节点做 EP 代价极高**。所以实践上限：EP 度数通常不超过单节点 GPU 数（8）或少量跨节点。

**优化手段**：

- 计算-通信重叠（用独立的 CUDA stream，通信与本地计算并行）
- 限制 EP 度数，节点内优先
- DeepSeek 的 **DualPipe**：把前向/反向的 all-to-all 与计算重叠，做到近乎零通信开销（配合特定的流水线调度）

### 6. MoE 的推理特点

**优势**：每 token 只用激活参数，算力需求低。

**劣势**：

1. **显存按总参数算**：8×7B 的 MoE 需要加载 56B 参数的权重（即使只激活 7B）。8×7B MoE 在 bf16 下要 112GB 权重，需要多卡。
2. **batch 小时 MoE 反而慢**：因为 batch 小的时候，每个专家分到的 token 少，GEMM 效率低下（矩阵太小）；MoE 的优势在高 batch 时才体现。
3. **负载不均导致延迟**：如果路由失衡，某些卡成为瓶颈（木桶效应）。

**MoE 量化**：因为显存是瓶颈，MoE 是量化的**最佳场景**——INT4/FP8 权重量化能直接把 112GB 压到 28~56GB，且 MoE 的激活参数少，量化误差的影响相对小。

## 具体例子

**例 1：参数量与激活量的计算（Mixtral 8×7B）**

配置：8 个专家，top-2 路由，FFN 中间维 d_ff，其余同 7B 稠密。

```
每个专家 FFN 参数: 3 × d × d_ff = 3 × 4096 × 14336 = 176 M
8 个专家:          8 × 176 M = 1.41 B
attention:         67 M
单层总计:          1.48 B
32 层:             47.4 B
embedding:         262 M
总参数:            ≈ 46.7 B （官方: 46.7B）

激活参数:
  每 token 2 个专家 FFN: 2 × 176 M = 352 M
  attention:              67 M
  单层:                   419 M
  32 层:                  13.4 B
  embedding:              262 M
激活参数总计:      ≈ 13.7 B
```

**结论**：46.7B 总参数，但每 token 只算 13.7B 参数的量——推理算力相当于 13B 稠密模型，但知识容量是 47B。

**例 2：组合数的爆炸（DeepSeekMoE）**

```
传统 top-2/16专家: C(16,2) = 120 种组合
细粒度 (m=4):      64 专家选 8    = C(64,8) = 4,426,165,368 种
                   提升 3.7e7 倍
```

**例 3：all-to-all 通信量**

Mixtral 8×7B，EP=8，d=4096，k=2，bf16，batch=1024 tokens：

```
每 token 每层发送:  2 × 4096 × 2 = 16,384 bytes
每 token 每层接收:  2 × 4096 × 2 = 16,384 bytes
每 token 每层总计:  32,768 bytes = 32 KB

一个 batch (1024 tokens) 一层: 32 KB × 1024 = 32 MB
32 层:                        1 GB
```

对比：稠密模型 TP=8 的 all-reduce，每 token 每层约 `2 × d × 2 × 2 = 32 KB`（两次规约 × d × 2字节 × 2）—— 量级相近但 MoE 是 all-to-all，无法规约合并，实际延迟更高。

**例 4：显存对比（bf16）**

```
7B 稠密:        13.5 GB 权重
8×7B MoE:       93.4 GB 权重（46.7B × 2 bytes）
8×7B MoE INT4:  23.4 GB 权重
```

这解释了为什么 MoE 模型的部署推荐量化：不量化的话单卡/单机都放不下。

## 代码速记

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoELayer(nn.Module):
    """简化 MoE: E 个专家, top-k 路由, 带负载均衡辅助损失"""
    def __init__(self, d, d_ff, num_experts=8, top_k=2, capacity_factor=1.25,
                 num_shared=0, aux_alpha=0.01):
        super().__init__()
        self.E, self.k, self.CF, self.alpha = num_experts, top_k, capacity_factor, aux_alpha
        self.gate = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.SiLU(),
                          nn.Linear(d_ff, d, bias=False))
            for _ in range(num_experts)])
        self.shared = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.SiLU(),
                          nn.Linear(d_ff, d, bias=False))
            for _ in range(num_shared)])
        self.aux_loss = 0.0

    def forward(self, x):
        B, S, d = x.shape
        xf = x.reshape(-1, d)                                  # T = B*S tokens
        T = xf.shape[0]
        logits = self.gate(xf)                                 # (T, E)
        probs = torch.softmax(logits.float(), dim=-1)          # P, 路由概率

        topk_p, topk_i = torch.topk(probs, self.k, dim=-1)     # (T, k)
        weights = topk_p / (topk_p.sum(-1, keepdim=True) + 1e-9)  # renormalize

        # ---- 负载均衡辅助损失: L_aux = alpha * E * sum(f_i * P_i) ----
        P_mean = probs.mean(0)                                        # (E,)
        onehot = torch.zeros_like(probs).scatter(1, topk_i, 1.0)      # (T, E)
        f = onehot.mean(0)                                            # (E,)
        self.aux_loss = self.alpha * self.E * (f * P_mean).sum()

        # ---- 容量限制，超出丢弃 ----
        cap = int(self.CF * T * self.k / self.E)
        out = torch.zeros_like(xf)
        for e in range(self.E):
            mask = (topk_i == e).any(-1)                       # 选择该专家的 token
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() > cap:
                idx = idx[:cap]                                # 超容量丢弃
            if idx.numel() == 0:
                continue
            w = weights[idx][(topk_i[idx] == e).nonzero(as_tuple=True)[1]]
            out[idx] += self.experts[e](xf[idx]) * w.unsqueeze(-1)

        for sh in self.shared:                                 # 共享专家：全量
            out = out + sh(xf)
        return out.view(B, S, d)

def router_z_loss(logits, lam=1e-3):
    """压制 router logits 的尺度漂移"""
    return lam * (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()

def moe_flops_per_token(d, d_ff, k, num_shared=0):
    """每 token 每 MoE 层的 FLOPs（只算激活的专家）"""
    return 2 * (k + num_shared) * 3 * d * d_ff

def moe_params(d, d_ff, E, num_shared=0):
    return (E + num_shared) * 3 * d * d_ff

# Mixtral 8x7B 核对
print(f"每层参数: {moe_params(4096, 14336, 8)/1e9:.3f} B")          # 1.409 B
print(f"每 token 每层 FLOPs: {moe_flops_per_token(4096,14336,2)/1e6:.1f} M")
```

## 面试追问

**Q：MoE 省显存吗？**
A：**训练时省一部分，推理时不省**。权重必须全部加载（因为不知道下一个 token 会路由到哪个专家），所以推理显存按总参数算。训练时因为只激活部分专家，激活显存和算力按激活参数算，但优化器状态仍按总参数算（所有专家都要更新）。所以准确说法是：**MoE 省算力，不省显存；省训练时间，不省部署成本**。

**Q：为什么 MoE 训练比稠密模型不稳定？**
A：三个来源。(1) **离散路由**：top-k 是不可微的硬选择，梯度只能通过软权重回传，噪声大；(2) **正反馈坍塌**：专家被选中越多学得越好、越好越被选中，容易跑偏；(3) **token 丢弃**：超出容量的 token 被丢掉，等效于这些位置没有 FFN 处理，损失曲面变得不连续。对策就是辅助损失、z-loss、噪声路由、以及更小的 lr + 更长的 warmup。

**Q：top-k 的 k 怎么选？**
A：常见 k=1（Switch）、k=2（Mixtral、DeepSeek-V2）、k=8（DeepSeek-V3，配合 256 专家）。规律：**专家粒度越细，k 要相应增大以保持激活参数量不变**。k=1 最省算力但路由方差大、不稳定；k≥2 能让每个 token 融合多个专家的输出，质量更好且路由更平滑（梯度更稳定）。现代趋势是"多而细"（E 大、k 大、每个专家小）。

**Q：MoE 的专家真的学到了专门的功能吗？**
A：部分特化，但不是"领域专家"那种清晰分工。实证研究显示：专家确实会分化（某些专家偏向处理标点/代码/特定语言/数字），但大多数专家是"语法/位置"层面的特化而非语义主题层面的特化。这也是为什么"按领域手动分配专家"的做法效果不好——让路由自己学更简单。

**Q：MoE 适合小模型吗？**
A：通常不适合。MoE 的收益来自"参数量 >> 激活量"，而在小模型上：(1) 显存本来就不紧，稀疏化的收益小；(2) batch 小时专家利用率低，GEMM 效率差；(3) 路由的不稳定性在参数少时更突出。经验上 MoE 的收益在 10B+ 参数规模才开始明显，主流 MoE 模型都在 40B+ 总参数。

**Q：MoE 和量化能叠加吗？有什么坑？**
A：能，而且是推荐的组合（MoE 显存压力大，量化收益高）。但有坑：**router 不能量化**（路由决策对精度极敏感，量化后路由分布会漂移，导致负载均衡崩溃）。实践做法是 router 保持 fp32/bf16，专家权重可以激进量化到 INT4/FP8。另外要注意量化后的专家输出误差会在 top-k 加权时被放大。

**Q：all-to-all 能不能和计算重叠？**
A：能，是 MoE 训练性能的关键。做法：用独立的 CUDA stream 做通信，同时 GPU 计算已经到达的 token 的专家前向；或者把前向/反向的通信与相邻层的计算做流水。DeepSeek 的 DualPipe 是极致做法——把整个流水线调度的 bubble 时间用来跑 all-to-all，实现接近零的通信开销。工程上，如果不做重叠，EP 的通信开销可能占到训练时间的 30%~50%。

## 关联知识点
- Transformer 结构与参数量（01-transformer）
- 训练稳定性：z-loss 与 loss spike（03-pretrain）
- 并行策略：DP/TP/PP/EP（04-distributed）
- 通信原语与通信量推导（04-distributed）
- 量化：MoE 权重量化（07-inference）
