---
id: reward-model
title: 奖励模型：Bradley-Terry 与 pairwise ranking loss
module: 09-alignment-reasoning
tags:
- 奖励模型
- Bradley-Terry
- 偏好数据
- 过优化
- 长度偏置
- PRM
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 奖励模型把"人类偏好"蒸馏成标量分数。训练用 Bradley-Terry 模型的 pairwise ranking loss：让赢的样本得分更高。关键难点是过优化（reward
  hacking 的源头）与长度偏置，以及过程奖励(PRM)与结果奖励(ORM)的取舍。
source:
- https://arxiv.org/abs/2305.18290
questions:
- 奖励模型的 loss 是怎么从 Bradley-Terry 来的？
- 什么是过优化？为什么 PPO 分数涨了但人觉得变差了？
- 长度偏置怎么缓解？
- ORM 和 PRM 的区别？PRM 标注为什么贵？
- 奖励模型一般怎么从 SFT 模型改造？（改最后一层）
---

## 一句话结论

- **Bradley-Terry**：人类偏好 `y_w ≻ y_l` 的概率建模为 `σ(r(x,y_w) - r(x,y_l))`；训练目标是最大化该概率的负对数似然。
- **过优化（overoptimization）**：RM 分数越高不代表人类越满意，PPO 会找到 RM 的"盲点"刷分 → 真实胜率反而下降。
- **长度偏置**：RM 容易学会"越长越好"，需用长度归一化/长度惩罚缓解。
- **ORM vs PRM**：结果奖励只看最终答案；过程奖励逐步打分，对推理更有用但标注贵。

## 原理推导

### Pairwise Ranking Loss

给定偏好对 `(x, y_w, y_l)`（y_w 优于 y_l），RM 输出标量 `r_φ(x,y)`：

```
L(φ) = - E[ log σ( r_φ(x, y_w) - r_φ(x, y_l) ) ]
```

- 等价于"让 y_w 的得分比 y_l 高"，差值越大损失越小。
- RM 通常在 SFT 模型上**去掉最后一层 lm_head、接一个标量头**（取最后一个 token 的隐状态过线性层得分数）。
- 推理时 RM 只用于给整条 rollout 打分（或逐 token），不参与生成。

### 过优化

经验规律（[Gao et al. 2023]）：随 PPO 训练，RM 给分持续上升，但**人类对生成质量的真实胜率先升后降**——RM 被"攻破"。根源是 RM 只是偏好的不完美代理，PPO 在代理上过优化。

缓解：早停（按真实胜率曲线）、对 RM 做集成、降低 β 让 KL 更紧、定期用新偏好数据重训 RM。

### 长度偏置

RM 在 pairwise 数据上易学到"长回复 = 好回复"的虚假相关。缓解：

- 训练时**长度归一化**：用 `r'(y) = r(y) - λ·len(y)` 或做长度分桶配对；
- 推理时给奖励减长度项（RLHF 中常见 length penalty）；
- 收集"长度平衡"的偏好对（同长度内比质量）。

## 具体例子

- **InstructGPT RM**：6B 模型，SFT 初始化，最后 token 隐状态 → 标量；用约 50k 人类偏好对训练。
- **PRM（过程奖励）**：对数学题的每一步标注"正确/错误"，训练 PRM 在每一步打分，配合 best-of-n 或 MCTS 搜索；标注成本高，常用"LLM 自评 + 人工校准"或蒙特卡洛估计。
- **ORM**：只对最终答案对错的 rollout 给 0/1 或标量；简单但无法纠正中间错误步骤。

## 代码速记

```python
# pairwise RM loss
score_w = rm(x, y_w)   # [B,1]
score_l = rm(x, y_l)
loss = -F.logsigmoid(score_w - score_l).mean()   # Bradley-Terry

# 长度归一化（推理）
reward = rm(x, y) - length_penalty * y.numel()
```

## 面试追问

1. 奖励模型的 loss 是怎么从 Bradley-Terry 来的？
2. 什么是过优化？为什么 PPO 分数涨了但人觉得变差了？
3. 长度偏置怎么缓解？
4. ORM 和 PRM 的区别？PRM 标注为什么贵？
5. 奖励模型一般怎么从 SFT 模型改造？（改最后一层）

## 关联知识点

- [RLHF 流水线：SFT→RM→RL 与 PPO-Clip](rlhf-pipeline)
- [DPO 推导：从 RLHF 最优解反推闭式 loss](dpo-derivation)
- [奖励黑客：长度攻击、谄媚与缓解](reward-hacking)
