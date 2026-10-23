---
id: dpo-variants
title: DPO 变体：IPO / KTO / ORPO / SimPO / CPO
module: 09-alignment-reasoning
tags:
- DPO
- IPO
- KTO
- ORPO
- SimPO
- 偏好优化
- 变体
difficulty: 4
freq: 中
updated: 2026-08-28
summary: 标准 DPO 依赖"成对的偏好数据 + 参考模型"。变体分别解决：IPO 去掉过拟合、KTO 用单条"欢喜/讨厌"数据、ORPO 完全省掉参考模型、SimPO
  用序列平均概率当隐式奖励、CPO 直接做偏好分类不采样。
source:
- https://arxiv.org/abs/2305.18290
questions:
- 标准 DPO 在什么数据情况下会过拟合？IPO 怎么缓解？
- KTO 相比 DPO 最大的数据优势是什么？
- ORPO 为什么能完全不要参考模型？
- SimPO 的长度归一化项解决了 DPO 的什么问题？
- 这么多变体，实际项目怎么选？
---

## 一句话结论

| 方法 | 核心改动 | 解决 DPO 的什么 |
|---|---|---|
| IPO | 加正则项，去掉隐式"偏好对无穷分离"假设 | DPO 在噪声/重复数据上过拟合 |
| KTO | 用单条样本 + 人类效用函数（欢喜/讨厌） | 不需要成对数据 |
| ORPO | 只在 SFT loss 上加 odds-ratio 惩罚 | **完全不需要参考模型** |
| SimPO | 用序列平均 logp 当隐式奖励，省参考模型 | 更简单、更稳、无需 β 参考 |
| CPO | 直接分类偏好，跳过采样/生成 | 训练更快、更省 |

## 原理推导

### DPO 的隐患（为什么要有变体）

DPO 的最优解在数学上要求"偏好对完全可分"。当数据有噪声或重复时，标准 DPO loss 会鼓励 `π_θ(y_w)` 和 `π_θ(y_l)` 差距无限拉大（logit 趋向无穷），导致过拟合与熵崩塌。

### IPO（Identity Preference Optimization）

在 DPO loss 上加正则，约束策略不会过度偏离：本质是把"最大化偏好差"换成"在合理范围内最大化"。经验上减少过拟合、提升多样性。

### KTO（Kahneman-Tregman Optimization）

不用成对 `(y_w, y_l)`，而用单条 `(x, y, 是否 desirable)`。损失基于"人类效用"：

```
L_KTO = - E[ λ_y · (1 - sigmoid(β·(r_θ(x,y) - τ))) ]   # r_θ 由策略与参考差定义
```

- `λ_y` 对 desirable / undesirable 样本取不同权重（通常 undesirable 权重更高，因为纠正更重要）。
- 好处：**能在只有"好/坏"标注、没有 pairwise 配对的数据上训**，数据更易得。

### ORPO（Odds Ratio Preference Optimization）

完全不要参考模型，也不训 RM。在 SFT 交叉熵上加一项 odds-ratio 惩罚：

```
L_ORPO = L_SFT + λ · L_OR
L_OR   = - log σ( log( odds(π(y_w|x)) / odds(π(y_l|x)) ) )   # odds = π/(1-π)
```

- 直觉：直接拉大"好样本似然 / 坏样本似然"的 odds 比，无需冻结的 ref 占用显存。
- 显存省一半（不用同时存 ref），训练最简单。

### SimPO（Simple Preference Optimization）

发现 DPO 的奖励 `r ∝ log π_θ(y|x) - log π_ref(y|x)` 中参考项拖累且需额外存 ref。SimPO 用**序列级平均 log 概率**当隐式奖励：

```
r_sim(y|x) = (1/|y|) · log π_θ(y|x)  -  τ      # τ 为边距
L_SimPO = - E[ log σ( β · (r_sim(y_w) - r_sim(y_l)) ) ]
```

- 省掉参考模型；用**长度归一化**（除以 |y|）天然缓解 DPO 的长度偏置。
- 实验上常优于 DPO，且更省资源。

### CPO（Contrastive Preference Optimization）

把偏好学习看成**直接分类**：放弃 DPO 里的"生成 rollout + 奖励"过程，直接在给定 (y_w, y_l) 上做对比分类，等价于 DPO 去掉采样/RL 循环，训练快、工程简单。

## 具体例子

- 你有 1 万条 `(prompt, chosen, rejected)` → 用 DPO / SimPO。
- 你只有"这条回复用户点赞/点踩"的单条数据 → 用 KTO。
- 你显存紧张、想省 ref 模型 → 用 ORPO / SimPO。
- 你不想做在线采样、要快 → 用 CPO。

## 面试追问

1. 标准 DPO 在什么数据情况下会过拟合？IPO 怎么缓解？
2. KTO 相比 DPO 最大的数据优势是什么？
3. ORPO 为什么能完全不要参考模型？
4. SimPO 的长度归一化项解决了 DPO 的什么问题？
5. 这么多变体，实际项目怎么选？

## 关联知识点

- [DPO 推导：从 RLHF 最优解反推闭式 loss](dpo-derivation)
- [RLHF 流水线：SFT→RM→RL 与 PPO-Clip](rlhf-pipeline)
- [奖励模型：Bradley-Terry 与 pairwise loss](reward-model)
