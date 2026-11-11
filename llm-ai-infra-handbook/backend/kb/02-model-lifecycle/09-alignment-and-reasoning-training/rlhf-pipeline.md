---
id: rlhf-pipeline
title: RLHF 流水线：SFT→RM→RL 与 PPO-Clip
module: 09-alignment-reasoning
tags:
- RLHF
- PPO
- 强化学习
- 奖励模型
- 对齐
difficulty: 4
freq: 高
updated: 2026-08-28
summary: RLHF 三阶段：SFT 冷启动 → 训练奖励模型(RM) → 用 PPO 在 RM 指导下优化策略并加 KL 约束防跑偏。PPO-Clip 用裁剪的概率比替代无界更新；线上同时跑四个模型（Actor/Critic/Ref/RM）。
source:
- https://arxiv.org/abs/2305.18290
questions:
- RLHF 为什么需要四个模型？可不可以省？
- PPO 的 clip 解决了什么问题？不 clip 会怎样？
- KL 惩罚项作用在哪些 token 上？β 太大/太小的影响？
- GAE 的 γ 和 λ 分别控制什么？
- 为什么 RLHF 训练经常不稳定？常见调参有哪些？
---

## 一句话结论

- **三阶段**：① SFT 得到初始策略 π_SFT；② 用人类偏好数据训奖励模型 r_φ；③ 用 PPO 最大化 `E[r] - β·KL(π||π_ref)`，把模型对齐到"人类喜欢"。
- **PPO-Clip** 目标：`L = E[ min(ρ·A, clip(ρ,1-ε,1+ε)·A) ]`，用裁剪的概率比防止单步更新过大。
- **四个模型**：Actor(被训)、Critic(估优势)、Ref(冻结的 SFT，算 KL)、RM(打分的奖励)。显存与部署是主要成本。

## 原理推导

### 带 KL 约束的优化目标

```
max_π  E_{x~D, y~π(·|x)}[ r_φ(x,y) ]  -  β · KL( π(·|x) || π_ref(·|x) )
```

- `r_φ` 来自奖励模型；`β` 控制"偏离 SFT 多远"，β 大→更保守。
- KL 项防止策略为了刷高 RM 分数而跑出语言模型分布（即奖励黑客）。

### PPO-Clip 目标（逐 token）

设 `ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`（新旧策略概率比），优势 `Â_t`：

```
L^{CLIP} = E[ min( ρ_t · Â_t,  clip(ρ_t, 1-ε, 1+ε) · Â_t ) ]
```

- 当 Â_t > 0（好动作）：ρ 大于 1+ε 的部分被截掉 → 不会因一次好样本过度更新。
- 当 Â_t < 0（坏动作）：ρ 小于 1-ε 的部分被截掉。
- 配合 **GAE** 估计优势：`Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`，`δ_t = r_t + γV(s_{t+1}) - V(s_t)`。

### 完整损失（PPO 一次更新）

```
L = L^{CLIP}  -  c_1·L^{VF}  +  c_2·H[π]  -  β·KL(π||π_ref)
     价值损失       熵奖励         KL 惩罚
```

## 具体例子

**训练一个助手回复**：

1. SFT：用 10k 条人工示范 `{prompt, good_answer}` 微调基座 → π_SFT。
2. RM：收集偏好对 `(prompt, y_w, y_l)`，训 `r_φ` 使 `σ(r(x,y_w)-r(x,y_l))` 尽量大。
3. RL：对每个 prompt 采样 rollout，RM 给整条回复打分 `r`，用 GAE 算 token 级优势，PPO 更新 Actor；Ref 提供 KL 基准；Critic 估 V。

**四个模型显存**（7B 为例，fp16）：Actor 14G + Critic 14G + Ref 14G + RM 14G ≈ 56G，再加优化器与激活 → 通常需多卡或 LoRA/小 RM。

## 代码速记

```python
# PPO-Clip 单步（简化）
ratio = torch.exp(new_logp - old_logp)          # ρ_t
unclipped = ratio * adv
clipped = torch.clipped(ratio, 1-eps, 1+eps) * adv
policy_loss = -torch.min(unclipped, clipped).mean()
value_loss = F.mse_loss(value, returns)
loss = policy_loss + c1*value_loss - c2*entropy
```

## 面试追问

1. RLHF 为什么需要四个模型？可不可以省？
2. PPO 的 clip 解决了什么问题？不 clip 会怎样？
3. KL 惩罚项作用在哪些 token 上？β 太大/太小的影响？
4. GAE 的 γ 和 λ 分别控制什么？
5. 为什么 RLHF 训练经常不稳定？常见调参有哪些？

## 关联知识点

- [奖励模型：Bradley-Terry 与 pairwise loss](reward-model)
- [DPO 推导：从 RLHF 最优解反推闭式 loss](dpo-derivation)
- [GRPO 与 RLVR：省掉 value 的组内相对优势](grpo-and-rlvr)
