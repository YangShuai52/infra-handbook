---
id: dpo-derivation
title: DPO 推导：从 RLHF 最优解反推闭式 loss
module: 09-alignment-reasoning
tags:
- DPO
- 推导
- KL约束
- Bradley-Terry
- 偏好优化
- 必考
difficulty: 5
freq: 高
updated: 2026-08-28
summary: DPO 的精髓：先写出 RLHF 带 KL 约束的优化问题，求出其闭式最优策略 π*，再用 Bradley-Terry 偏好模型把"奖励 r"消掉，得到只含
  π_θ 与参考模型 π_ref 的似然损失——无需显式奖励模型、无需 RL 采样。
source:
- https://arxiv.org/abs/2305.18290
questions:
- 从 RLHF 目标到 DPO loss 一共几步？哪一步把奖励 r 消掉了？
- 第 3 步的 β·log Z(x) 常数为什么在偏好对里被抵消？
- DPO 为什么不需要显式奖励模型？
- β 的作用？β→0 / β→∞ 分别发生什么？
- DPO 的局限？（分布外负例、对噪声敏感、不学绝对质量）
---

## 一句话结论

DPO loss（闭式）：

```
L_DPO(θ) = - E_{(x,y_w,y_l)}[ log σ( β · (
      log(π_θ(y_w|x)/π_θ(y_l|x))
    - log(π_ref(y_w|x)/π_ref(y_l|x))
) ) ]
```

关键洞察：**最优策略下，奖励 r(x,y) 可以完全用策略与参考策略的对数似然差表示**，于是把"训奖励模型 + RL 优化"两步合一，直接做有监督式的偏好分类。

## 原理推导（逐步，不可跳步）

### 第 1 步：写出 RLHF 目标

```
max_π   E_{x~D, y~π(·|x)}[ r(x,y) ]  -  β · KL( π(y|x) ‖ π_ref(y|x) )
```

记为 `J[π]`，β > 0。这是逐 prompt 独立的问题，对固定 x 求最优 π*(·|x)。

### 第 2 步：求闭式最优策略

对每个 x，最优解满足（拉格朗日/KKT，标准结果）：

```
π*(y|x) =  (1 / Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

其中配分函数 `Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)`，保证归一化。

### 第 3 步：把奖励 r 解出来

对上式取 log 再整理：

```
log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)
⇒  r(x,y)   = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

注意 `β log Z(x)` 只依赖 x、不依赖 y——它是个"常数项"，在偏好比较里会被消掉。

### 第 4 步：代入 Bradley-Terry 偏好模型

人类偏好 `y_w ≻ y_l` 的概率：

```
p*(y_w ≻ y_l) = σ( r(x, y_w) - r(x, y_l) )
```

代入第 3 步的 r（常数项 β log Z(x) 相减抵消）：

```
= σ( β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x)) )
= σ( β · log[ π*(y_w|x) · π_ref(y_l|x)
              / ( π_ref(y_w|x) · π*(y_l|x) ) ] )
```

### 第 5 步：用可训策略 π_θ 替换不可达的 π*

π* 是理论最优，我们用参数化策略 π_θ 去逼近它。把上式中的 π* 换成 π_θ，取负对数似然即 DPO loss：

```
L_DPO(θ) = - E[ log σ( β · log[ π_θ(y_w|x)·π_ref(y_l|x)
                              / ( π_ref(y_w|x)·π_θ(y_l|x) ) ] ) ]
```

整理对数项得到开头给出的等价形式。完成推导。✅

## 梯度解读（面试加分）

对 y_w 的梯度：增大 `log π_θ(y_w|x)`（让模型更倾向赢的样本）；
对 y_l 的梯度：压低 `log π_θ(y_l|x)`。
权重 `σ(...)` 的反面——**模型当前越"搞反"（把 y_l 打得比 y_w 高），梯度越大**；一旦模型已正确区分，梯度趋近 0。β 缩放整个对数差，等价于 KL 约束强度。

## 具体例子

设 β=0.1，某对 (y_w, y_l) 当前：
`log π_θ(y_w|x) - log π_θ(y_l|x) = +0.3`，
`log π_ref(y_w|x) - log π_ref(y_l|x) = -0.5`。
则括号内 `= 0.1·(+0.3 - (-0.5)) = 0.08`，σ(0.08)≈0.52 → 损失 `=-log(0.52)=0.65`。
若模型翻转该对（前者变 -0.3），括号内 `=0.1·(-0.3+0.5)=0.02`？不，应为 0.1·(-0.3-(-0.5))=0.02，σ≈0.505，损失≈0.69，变化不大——说明 β 较小时梯度温和；β 大则对错更敏感（强 KL 约束）。

## 代码速记

```python
def dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    # 都是序列级对数似然之和
    pi_logratio  = pi_logps_w  - pi_logps_l
    ref_logratio = ref_logps_w - ref_logps_l
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()
```

## 面试追问

1. 从 RLHF 目标到 DPO loss 一共几步？哪一步把奖励 r 消掉了？
2. 第 3 步的 β·log Z(x) 常数为什么在偏好对里被抵消？
3. DPO 为什么不需要显式奖励模型？
4. β 的作用？β→0 / β→∞ 分别发生什么？
5. DPO 的局限？（分布外负例、对噪声敏感、不学绝对质量）

## 关联知识点

- [RLHF 流水线：SFT→RM→RL 与 PPO-Clip](rlhf-pipeline)
- [奖励模型：Bradley-Terry 与 pairwise loss](reward-model)
- [DPO 变体：IPO / KTO / ORPO / SimPO](dpo-variants)
