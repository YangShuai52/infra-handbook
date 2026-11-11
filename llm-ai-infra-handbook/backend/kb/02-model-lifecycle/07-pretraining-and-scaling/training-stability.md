---
id: training-stability
title: 训练稳定性：混合精度、loss spike 与优化器
module: 07-pretraining-scaling
tags:
- 混合精度
- bf16
- loss spike
- 优化器
- warmup
- Muon
- 高频
difficulty: 4
freq: 高
updated: 2026-08-28
summary: LLM 用 bf16（指数位与 fp32 同宽，不溢出）而非 fp16；loss spike 源于特定层的梯度尖峰，对策是梯度裁剪 + z-loss
  + QK-Norm；调度用 warmup + cosine 到 10% 峰值。
source:
- https://arxiv.org/abs/2203.15556
questions:
- bf16 精度这么低，训练不会有问题吗？
- fp16 和 bf16 在实际训练中的表现差异？
- loss spike 之后模型还能救回来吗？
- 为什么要做梯度裁剪？裁剪阈值怎么定？
- Adam 的 β1、β2 怎么设？大模型和小模型有区别吗？
---

## 一句话结论

大模型训练的稳定性问题，本质是**低精度 + 深网络 + 大 batch** 三者的叠加效应：低精度带来数值误差，深网络放大梯度异常，大 batch 让每次更新影响更大且难以回滚。

三个关键判断：

1. **用 bf16 不用 fp16**：bf16 与 fp32 的指数位都是 8 位（动态范围相同），只是尾数从 23 位砍到 7 位，所以**不会溢出，只是精度低**；fp16 只有 5 位指数，梯度/激活很容易溢出。
2. **loss spike 不是随机的**：它通常对应某一层（常见是 attention 的输出投影或第一个 FFN）的梯度突然变大，且与数据批次（某些"异常"样本）强相关。
3. **现代优化器在换**：AdamW 仍是主流，但 Muon/SOAP 等基于矩阵结构的优化器在大 batch 下表现出更好的样本效率和稳定性。

## 原理推导

### 1. 浮点格式对比

| 格式 | 符号 | 指数位 | 尾数位 | 动态范围(约) | 精度(十进制位) |
|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | ±3.4e38 | ~7 |
| fp16 | 1 | 5 | 10 | ±6.5e4 | ~3 |
| bf16 | 1 | 8 | 7 | ±3.4e38 | ~2 |
| fp8 e4m3 | 1 | 4 | 3 | ±448 | ~1 |
| fp8 e5m2 | 1 | 5 | 2 | ±57344 | ~0.5 |

**关键对比**：

```
fp16 最大正常数 = 65504，梯度超过这个值 → Inf → NaN 传播
bf16 最大正常数 ≈ 3.4e38，几乎不可能溢出
```

代价：bf16 只有 7 位尾数（fp32 的 1/16），相对误差约 2^-8 = 0.4%。这对"大数加小数"很致命：

```
1e6 + 1 在 bf16 下 = 1e6（小数被完全吞掉）
```

所以混合精度训练必须保留 **fp32 主权重**（master weights）：前向/反向用 bf16 算，但参数更新发生在 fp32 副本上，避免小更新被舍入抹掉。

### 2. Loss Scaling（fp16 才需要）

fp16 的梯度下溢问题（小梯度 < 6e-8 变成 0）：

```
1. 前向: loss × S（放大因子，如 2^16）
2. 反向: 梯度自动被放大 S 倍
3. 更新前: 梯度 ÷ S 还原
4. 若检测到 Inf/NaN: 跳过这一步，S ← S/2
5. 连续 N 步无 Inf: S ← S × 2
```

bf16 因为动态范围够，**通常不需要 loss scaling**（这是选 bf16 的工程收益之一）。

### 3. Loss Spike 的成因与对策

**现象**：训练中期突然出现 loss 大幅上升（如从 2.1 跳到 3.5），可能几步恢复，也可能永久损伤。

**已确认的机制**（多项研究的共识）：

1. **特定 batch 触发**：某些样本（超长重复、异常字符、极端分布）导致梯度范数突然增大
2. **梯度在层间不均匀**：某些层的梯度范数长期是其他层的 10~100 倍
3. **Adam 的二阶动量滞后**：Adam 用 `v_t`（梯度平方的指数平均）归一化，当梯度突然变大时，`v` 还没跟上，导致 `g/√v` 被放大 → 更新步过大
4. **与 lr schedule 交互**：在 cosine 下降的中段、或 warmup 刚结束时风险最高

**对策清单**（按有效性排序）：

| 对策 | 做法 | 说明 |
|---|---|---|
| 梯度裁剪 | `clip_grad_norm_(1.0)` | 最基础，几乎必须开 |
| 跳过异常步 | 梯度范数超过阈值时 skip 该 step | 简单有效，但有偏 |
| 降低 lr / 延长 warmup | | 治标 |
| **z-loss** | `loss += λ·(log Z)²`，Z 是 softmax 分母 | 稳定 logits 尺度，防止 logits 爆炸 |
| **QK-Norm** | 对 Q、K 做归一化再算 attention | 直接压制 attention logits 的方差，Gemma 使用 |
| logit softcapping | `logits = tanh(logits/t)·t` | Gemma-2 使用，硬约束 logits 范围 |
| 重启 + 换数据顺序 | 从 spike 前的 checkpoint 恢复 | 兜底手段 |

**z-loss 的原理**：softmax 的分母 `Z = Σ exp(z_i)`，若 logits 整体平移 c，`Z` 变成 `Z·e^c` 但 softmax 输出不变——这意味着 **logits 的绝对尺度不受 loss 约束**，可以自由漂移。z-loss 加一项 `λ·(log Z)²` 把它拉回 0 附近，防止数值漂移累积。典型 λ = 1e-4。

### 4. 学习率调度

**Warmup + Cosine Decay** 是事实标准：

```
lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π · t / T))   for t > warmup
lr(t) = lr_max · t / warmup_steps                              for t ≤ warmup
```

为什么要 warmup：
- 训练初期参数是随机的，梯度方向噪声大，大 lr 会让参数"跑飞"
- Adam 的 `m` 和 `v` 在初期估计不准（偏置校正只修正了期望，方差仍大）
- 经验值：warmup 步数取总步数的 0.5%~2%，或固定几千步

**最终 lr 的选择**：cosine 衰减到 0 还是到峰值的 10%？

```
衰减到 0:  训练结束时模型更"收敛"，但最后阶段 lr 太小，数据利用率低
衰减到 10%:  LLaMA 等主流做法，保留一定"活力"，且便于继续训练
```

继续预训练时，通常从上次的 min_lr 重新开始 warmup。

**batch size 与 lr 的关系**：

- **线性缩放**：batch × k → lr × k（在 batch 不太大时成立）
- **平方根缩放**：batch × k → lr × √k（更保守，噪声更大时适用）
- 实践：大模型训练常用**渐进 batch**（先用小 batch 训一部分，再切大 batch），兼顾样本效率和稳定性

### 5. 优化器

**AdamW**（主流）：

```
m_t = β1·m_{t-1} + (1-β1)·g_t
v_t = β2·v_{t-1} + (1-β2)·g_t²
m̂ = m_t/(1-β1^t),  v̂ = v_t/(1-β2^t)
θ_t = θ_{t-1} - lr·( m̂/(√v̂ + ε) + λ·θ_{t-1} )
```

注意 **AdamW 与 Adam + L2 正则的区别**：AdamW 的权重衰减是**解耦**的（直接减 `lr·λ·θ`），不经过 `1/√v` 的缩放。这对 LLM 很重要，因为 `1/√v` 的量级不确定，会让衰减效果失控。

**显存开销**：每参数 4（fp32 主权重）+ 4（m）+ 4（v）= 12 字节，加 bf16 参数 2 + bf16 梯度 2 = **16 字节/参数**。这是 ZeRO 要切分的对象。

**Muon / SOAP / Kron**：新一代基于矩阵结构的优化器。

核心思想：Adam 是**逐元素**的自适应（每个参数一个学习率），但神经网络的参数是矩阵，应该利用矩阵的结构（如奇异值分布）来做更合理的更新。Muon 的做法是对动量做 **Newton-Schulz 迭代正交化**：

```
M_t = μ·M_{t-1} + g_t                      （动量，可用 Nesterov）
O_t = NewtonSchulz(M_t)                    （正交化，替代 M/‖M‖）
W_t = W_{t-1} - lr·O_t                     （更新）
```

Newton-Schulz 迭代（避免 SVD 的开销，纯矩阵乘可在 GPU 上高效跑）：

```
X ← a·X + b·X·XᵀX + c·X·XᵀX·XᵀX     （三次矩阵乘，迭代 5 次左右收敛）
```

典型系数 `a=3.4445, b=-4.7750, c=2.0315`。

**收益**：在相同算力下，Muon 的样本效率比 AdamW 高约 1.3~2 倍（即达到同样 loss 所需的 token 更少），且对大 batch 更鲁棒。代价是实现复杂、需要在矩阵维度上做匹配（如 embedding 层通常仍用 AdamW）。

### 6. 初始化

标准做法（GPT-2/LLaMA 风格）：

```
权重: N(0, 0.02²)
残差分支输出投影 (W_o, W_down): N(0, 0.02² / √(2·L))     ← 关键
embedding: N(0, 1)  或 N(0, 0.02²)
bias: 0
```

为什么要缩放残差分支：Pre-LN 下残差流的方差随层数累加（见 `归一化与激活函数`），把每个残差分支的输出按 `1/√(2L)` 缩放后，L 层的总方差贡献约为 `L × σ²/(2L) = σ²/2`，与输入方差同量级，不会爆炸。

**muP（maximal update parametrization）**：更系统的方案，让最优超参在模型宽度变化时保持不变，从而可以在小模型上调参再迁移到大模型。核心是让各层的初始化方差和 lr 按宽度做精确缩放。

## 具体例子

**例 1：fp16 溢出的实际场景**

```
某层梯度: [1.2e-3, 8.5e-2, 3.1e-4, ..., 2.4e5]     ← 最大值 2.4e5 > 65504
fp16 表示: Inf
→ m、v 变 Inf → 参数更新变 NaN → 整个模型报废
```

bf16 下 `2.4e5` 完全可以表示，只损失尾数精度，不会崩。

**例 2：bf16 的精度损失**

```
a = 1.0,  b = 0.001
fp32: a + b = 1.001
bf16: a + b = 1.0        （0.001 完全丢失）
```

累积效应：如果在 bf16 下做 1000 次 `x += 0.001`，最终结果可能是 1.0 而不是 2.0。这就是为什么**优化器状态、主权重、归约操作必须在 fp32 下做**。

**例 3：显存开销清单（7B，AdamW，bf16 混合精度）**

| 项目 | 每参数字节 | 7B 总量 |
|---|---|---|
| bf16 参数 | 2 | 13.5 GB |
| bf16 梯度 | 2 | 13.5 GB |
| fp32 主权重 | 4 | 27.0 GB |
| Adam m (fp32) | 4 | 27.0 GB |
| Adam v (fp32) | 4 | 27.0 GB |
| **合计** | **16** | **108 GB** |
| + 激活（需重计算或 TP 切分） | — | 数十 GB |

80G A100 单卡放不下 108GB，所以 7B 全参训练**至少需要 ZeRO-2 或 TP**。

**例 4：warmup 步数的经验值**

```
总训练步数 T = 1e6 （1T tokens / 1M tokens per step）
warmup = 0.5% × 1e6 = 5000 步   （或固定 2000~8000 步）

lr_max 经验值（batch 4M tokens）:
  1B 模型:  3e-4
  7B 模型:  3e-4
  70B 模型: 1.5e-4 ~ 2e-4   （更大的模型用更小的 lr）
  min_lr:   lr_max × 0.1
```

注：小模型（<1B）常用更大 lr（1e-3 ~ 3e-3）。

## 代码速记

```python
import torch, torch.nn as nn, math

# ---------- 1. 混合精度训练骨架（bf16） ----------
def train_step(model, batch, opt, scaler=None, grad_clip=1.0, skip_thresh=None):
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):     # 前向在 bf16
        logits = model(batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),   # 关键: CE 用 fp32
            batch["input_ids"][:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # 梯度裁剪前先检查范数（可选：跳过异常步）
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if skip_thresh is not None and gn > skip_thresh:
        print(f"skip step: grad_norm={gn:.1f}")
        opt.zero_grad(set_to_none=True)
        return None
    opt.step()
    return loss.item()

# ---------- 2. z-loss ----------
def z_loss(logits, lambda_z=1e-4):
    """压制 logits 绝对尺度漂移"""
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return lambda_z * (log_z ** 2).mean()

# ---------- 3. QK-Norm ----------
class QKNormAttention(nn.Module):
    def forward(self, x):
        q, k = self.q(self.norm_q(x)), self.k(self.norm_k(x))
        q = torch.nn.functional.rms_norm(q, q.shape[-1:]) * self.q_scale
        k = torch.nn.functional.rms_norm(k, k.shape[-1:]) * self.k_scale
        return torch.nn.functional.scaled_dot_product_attention(q, k, self.v(x), is_causal=True)

# ---------- 4. warmup + cosine ----------
def lr_lambda(step, warmup, total, min_ratio=0.1):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

# ---------- 5. Muon 的 Newton-Schulz 正交化 ----------
@torch.no_grad()
def newton_schulz(G, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-7):
    """把梯度动量矩阵正交化，替代 M/‖M‖；纯 matmul，GPU 友好"""
    X = G / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.T if G.size(0) > G.size(1) else X

# ---------- 6. 缩放初始化 ----------
def init_weights(module, n_layers, std=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def scale_residual(module, n_layers):
    """残差分支的输出投影要按 1/sqrt(2L) 缩放"""
    with torch.no_grad():
        module.weight.mul_(1.0 / math.sqrt(2 * n_layers))
```

## 面试追问

**Q：bf16 精度这么低，训练不会有问题吗？**
A：关键在于**分层混用**：矩阵乘（GEMM）用 bf16（吃 Tensor Core 的算力优势），但以下部分必须 fp32——(1) 主权重和 Adam 状态（小更新不能被舍入吞掉）；(2) 归约操作（softmax、LayerNorm 的求和，累加误差会累积）；(3) loss 计算（cross entropy 的 log_softmax）；(4) 跨 GPU 的 all-reduce（通信规约）。这套"混合"策略在实践上被证明与纯 fp32 训练的差异在噪声范围内。

**Q：fp16 和 bf16 在实际训练中的表现差异？**
A：fp16 需要 loss scaling，且 loss scaling 的动态调整本身会引入训练波动；遇到梯度尖峰时 fp16 更容易 Inf。bf16 基本不用管这些。所以除了在极老的硬件（不支持 bf16 的 V100 及更早）上，现代训练一律用 bf16。唯一例外：某些对精度极其敏感的操作（如小方差的 LayerNorm）可能需要手动提升精度。

**Q：loss spike 之后模型还能救回来吗？**
A：看情况。如果 spike 后 loss 在几十步内回落到原趋势，通常无碍——研究表明这类 spike 往往是"暂时的数值扰动"，模型会自行恢复。但如果 spike 后 loss 稳定在新的更高水平，说明参数被推到了一个更差的区域，需要从 spike 前的 checkpoint 重启，并换数据顺序/降 lr。工程上最佳实践是：**频繁保存 checkpoint（如每 1000 步）+ 自动检测 spike 并回滚**。

**Q：为什么要做梯度裁剪？裁剪阈值怎么定？**
A：梯度裁剪把更新步的"方向"保留、"幅度"限制，防止单步走太远。阈值选择：先跑几百步不裁剪，统计梯度范数的分布，取中位数或 90 分位数的 1~5 倍作为阈值。常用值 1.0（很多开源配置）看起来很小，但注意它裁剪的是**全局梯度范数**（所有参数梯度平方和的平方根），不是单个参数。

**Q：Adam 的 β1、β2 怎么设？大模型和小模型有区别吗？**
A：默认 `(0.9, 0.95)` 或 `(0.9, 0.999)`。大 batch 训练时 β2 取 0.95 更常见（更快的二阶动量适应，对非平稳的梯度分布更鲁棒，也更快从 spike 恢复）。β1 一般不动。ε 通常 1e-8，但在 bf16 下有时调到 1e-6 以避免除零放大误差。

**Q：Muon 这类新优化器值得用吗？**
A：取决于场景。在小规模（<1B）和中等规模上，Muon 的样本效率优势证据充分（多项独立复现显示 1.3~2× 的 token 效率提升）。但在超大规模（100B+）和超长训练上的公开证据还不多，且工程复杂度更高（需要为不同形状的矩阵分别处理、embedding 和 lm_head 通常仍用 AdamW）。面试时的稳妥说法是："AdamW 是可靠基线，Muon/SOAP 在中等规模上有明确收益且值得关注，但迁移到超大规模需要更多验证。"

**Q：学习率调度为什么要 cosine 而不是阶梯下降？**
A：阶梯下降（step decay）在 CV 时代常用，但预训练场景 cosine 更好，原因是：预训练只跑一个 epoch（数据不重复），没有自然的"阶段"边界；cosine 提供了平滑的退火过程，让模型在训练后期逐步收敛；而且 cosine 的最终 lr 可以直接设为峰值的一部分，便于继续训练。另一个原因是 cosine 对"何时停止训练"更鲁棒——中途停止也不会太差。

## 关联知识点
- 归一化、残差与激活函数（01-transformer）
- ZeRO 与 FSDP 的显存切分（04-distributed）
- 训练显存与算力估算（10-coding）
- 激活重计算（04-distributed）
- MoE 训练的负载均衡与 router z-loss（03-pretrain）
