---
id: grpo-and-rlvr
title: GRPO 与 RLVR：组内相对优势与可验证奖励
module: 09-alignment-reasoning
tags:
- GRPO
- RLVR
- 组内优势
- 推理模型
- o1
- DeepSeek-R1
- 强化学习
difficulty: 4
freq: 高
updated: 2026-08-28
summary: GRPO 用"同一 prompt 采样一组回答、组内相对排名"估计优势，从而省掉 PPO 里的价值网络(critic)，显存与稳定性都更好。RLVR
  用"可验证奖励"（答案对错/单元测试）替代学习型 RM，是 o1/R1 类推理模型的训练范式。
source:
- https://arxiv.org/abs/2305.18290
questions:
- GRPO 为什么不需要 value model？它用什么当基线？
- 组内标准化优势的公式？std 为 0 时怎么办（Dynamic Sampling）？
- RLVR 相比 RLHF 的 RM 有什么优势？（不过优化）
- DeepSeek-R1 是怎么用 GRPO 训出推理能力的？
- GRPO 和 RLOO / REINFORCE++ 的关系？
---

## 一句话结论

- **GRPO（Group Relative Policy Optimization）**：对每个 prompt 采样 G 个回答，用组内得分的**均值和标准差**做基线，优势 `Â_i = (r_i - mean(r)) / std(r)`。**不需要 critic / 价值网络**。
- **省 critic 的收益**：PPO 要 4 个模型，GRPO 只要 Actor + Ref（+ 奖励函数），显存大幅下降、训练更稳。
- **RLVR（RL with Verifiable Rewards）**：奖励来自**可自动验证的信号**（最终答案是否匹配、代码能否通过测试），而非训练一个会过优化的 RM——这是 DeepSeek-R1 / o1 类推理模型的核心。

## 原理推导

### 为什么要去掉 value model

PPO 的 GAE 需要 Critic 估计 `V(s)` 才能算优势 `Â`。Critic 本身需要训练、容易不准、且占一份模型显存。GRPO 的洞见：**同一 prompt 的多个采样天然构成"对照组"**，可以用组统计量当基线，不必学 V。

### GRPO 优势估计

对 prompt `q` 采样 `{o_1,...,o_G}`，各自得奖励 `{r_1,...,r_G}`（来自规则/验证器/`r_φ`）：

```
Â_i = ( r_i - mean({r}) ) / std({r})     # 组内标准化
```

- 若某回答比组内平均好 → 正优势 → 提升其概率；
- 比平均差 → 负优势 → 压低。
- 目标仍是带 KL 约束的裁剪似然（同 PPO-Clip 形式），但优势来自组内而非 critic。

### 单步目标（简化）

```
L_GRPO = - E[ (1/G) Σ_i min( ρ_i Â_i, clip(ρ_i,1-ε,1+ε)Â_i ) ]  + β·KL(π||π_ref)
```

### DAPO 等关键改进（来自 recent work）

- **Clip-Higher**：对正/负优势用不同 clip 上界，鼓励探索高奖励回答。
- **Dynamic Sampling**：丢弃"整组全对/全错"（组内方差为 0）的 prompt，避免无效梯度。
- **Token-level loss**：对长回答按 token 归一，避免长样本主导。
- **Overlong reward shaping**：对超长被截断的回答给惩罚，抑制"啰嗦刷分"。

### RLOO / REINFORCE++ 对比

- **RLOO**：用"组内其他样本的奖励均值"当基线（leave-one-out），是 GRPO 的近亲。
- **REINFORCE++**：标准 REINFORCE + 基线 + KL，工程上更简洁；GRPO 可看作带组内标准化基线的 REINFORCE。

## 具体例子

**数学题 RLVR**：

- prompt = 一道题；采样 G=8 个解答（带 CoT）。
- 奖励 = 最终答案与标准答案一致则 +1，否则 0（或用格式/正确性混合）。
- 组内标准化后，正确解答获正优势被强化，错误解答被压低。
- 不需要 RM：答案对错可由 sympy/执行验证 → 没有"RM 过优化"问题。

**DeepSeek-R1 路线**：R1-Zero 证明纯 GRPO + 规则奖励即可涌现长链推理与自我验证（"aha moment"）；R1 增加冷启动 SFT + 拒绝采样提升可读性。

## 代码速记

```python
# GRPO 组内优势（无 critic）
rewards = [score(o) for o in group]          # 规则/验证器打分
mean_r, std_r = mean(rewards), std(rewards) + 1e-6
adv = [(r - mean_r) / std_r for r in rewards]
# 再套 PPO-Clip 形式更新 actor，KL 约束到 ref
```

## 面试追问

1. GRPO 为什么不需要 value model？它用什么当基线？
2. 组内标准化优势的公式？std 为 0 时怎么办（Dynamic Sampling）？
3. RLVR 相比 RLHF 的 RM 有什么优势？（不过优化）
4. DeepSeek-R1 是怎么用 GRPO 训出推理能力的？
5. GRPO 和 RLOO / REINFORCE++ 的关系？

## 关联知识点

- [RLHF 流水线：SFT→RM→RL 与 PPO-Clip](rlhf-pipeline)
- [奖励模型：Bradley-Terry 与 pairwise loss](reward-model)
- [DPO 推导：从 RLHF 最优解反推闭式 loss](dpo-derivation)
