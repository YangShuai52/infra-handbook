---
id: sparse-and-linear-attention
title: 稀疏与线性注意力：Longformer、BigBird、Gemma 滑动窗口与 Mamba
module: 05-modern-architecture
tags:
- 稀疏注意力
- 线性注意力
- 滑动窗口
- Mamba
- SSM
- 长序列
difficulty: 4
freq: 中
updated: 2026-08-28
summary: 标准 attention 是 O(N²) 的，长序列下不可承受。稀疏注意力（滑窗/全局+局部/扩张）用结构化掩码把复杂度降到近线性；线性注意力用核技巧把
  softmax 改写成 O(N)；Mamba 等 SSM 走另一条"递归/卷积"路线。面试重点是理解各自"牺牲了什么、换来了什么"。
source:
- https://arxiv.org/abs/2101.03961
questions:
- 稀疏注意力和线性注意力分别牺牲了什么来换线性复杂度？
- 为什么线性注意力的核近似质量通常不如 softmax attention？
- Sliding window 会不会导致"看不到远处"？Gemma 怎么缓解？
- Mamba 为什么理论能回忆任意远处？它和Transformer的归纳偏置差异在哪？
- 线性注意力在推理时是真的 O(N) 吗？（要分清"训练可并行、推理可递推"两个视角）
---

## 一句话结论

- **稀疏注意力**：保留 softmax 的全局表达能力，但只让每个 token 关注一个**结构化子集**（滑窗、扩张、全局 token、随机），把显存/算力从 O(N²) 降到 O(N·w)。
- **线性注意力**：用 `φ(Q)φ(K)ᵀ` 近似 softmax，把注意力重排为 `(φ(Q)ᵀφ(K)V)`，复杂度 O(N·d²)。
- **SSM/Mamba**：放弃注意力，用状态空间模型做线性递推，理论 O(N)，且能"回忆"任意远处信息（因果卷积的全局感受野）。

## 原理推导

### 稀疏注意力的掩码模式

设序列长 N、窗口 w：

1. **Sliding Window（滑窗）**：每个位置只看 `[i-w, i]`。Gemma-2 用"交替滑窗"——部分层局部窗口、部分层全局注意力。
2. **Global + Local（Longformer）**：少数预定义的全局 token（如 [CLS]、问答中的 question token）能看到所有人、且被所有人看到；其余 token 用滑窗。复杂度 `O(N·w + g·N)`，g 为全局 token 数。
3. **Dilated（扩张，BigBird）**：滑窗基础上每隔 d 个位置再连一个，扩大感受野而不增窗口；加上全局 token + 随机少数连接，理论上能近似完整注意力的表达能力（"稀疏图也是通用近似器"）。

稀疏注意力**仍是 softmax attention**，只是掩码不同 → 训练/推理代码几乎不变，质量损失小，但只降常数因子（仍 ~O(N·w)），超长序列仍受限。

### 线性注意力：核技巧改写

标准注意力 `O = softmax(QKᵀ/√d) V`。用特征映射 `φ(·)` 与恒等式 `softmax(x)ᵀv ≈ φ(x)ᵀ φ(K) v / Σφ(x)ᵀφ(K)`（或 `softmax(QKᵀ)=softmax(Q)softmax(K)ᵀ` 当 Q、K 独立时），可得：

```
O_i = (φ(q_i)ᵀ Σ_j φ(k_j) v_jᵀ) / (φ(q_i)ᵀ Σ_j φ(k_j))
    = φ(q_i)ᵀ · S / (φ(q_i)ᵀ · z),   S = Σ_j φ(k_j) v_jᵀ,  z = Σ_j φ(k_j)
```

`S ∈ R^{d×d}`、`z ∈ R^d` 可**递推累积**，每步 O(d²)，总复杂度 O(N·d²)，与序列长 N 无关 → 线性。常见 `φ(x)=elu(x)+1`（Performer）或 `φ(x)=max(x,0)`（无 softmax 变体）。

**代价**：核近似有偏差，且缺少 softmax 的"尖峰聚焦"能力，长程依赖建模弱于标准 attention，质量通常略降。

### Mamba / SSM：另一条路

状态空间模型把输入 `x(t)` 经状态 `h(t)` 映射到输出：离散化后 `h_t = Ā h_{t-1} + B x_t; y_t = C h_t`。`Ā` 由输入相关的 Δ 动态决定（选择性机制），使模型能"按内容决定记住/遗忘"。它是**线性递推**（卷积形式 `y = x * K̄`，`K̄` 由 A、B、C 决定），复杂度 O(N)，且卷积核覆盖整段 → 全局感受野。Mamba-2 进一步把 SSM 与"结构化状态空间注意力"对应，可用矩阵乘法高效训练。

## 具体例子

- **Gemma-2 9B/27B**：交替注意力——每连续两层中一层用全局长上下文、另一层用局部滑动窗口，兼顾长程与效率。
- **Longformer**：窗口 w=512 + 全局 token，处理 4K→16K 文档分类，显存从 O(16K²) 降到 O(16K·512)。
- **Mamba-2.8B**：在 Pile 上 per-token 算力低于同尺寸 Transformer，长序列吞吐显著更高；但下游需要"双向/编码"的任务（如填空）弱于 attention。

## 面试追问

1. 稀疏注意力和线性注意力分别牺牲了什么来换线性复杂度？
2. 为什么线性注意力的核近似质量通常不如 softmax attention？
3. Sliding window 会不会导致"看不到远处"？Gemma 怎么缓解？
4. Mamba 为什么理论能回忆任意远处？它和Transformer的归纳偏置差异在哪？
5. 线性注意力在推理时是真的 O(N) 吗？（要分清"训练可并行、推理可递推"两个视角）

## 关联知识点

- [MHA / MQA / GQA / MLA 对比与 KV Cache 压缩](mha-mqa-gqa-mla)
- [FlashAttention：IO 感知的分块算法](flash-attention)
- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
