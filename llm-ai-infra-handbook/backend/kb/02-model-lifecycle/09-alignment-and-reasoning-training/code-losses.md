---
id: code-losses
title: 手撕各类 loss：CE / DPO / GRPO / MoE 负载均衡
module: 09-alignment-reasoning
tags:
- 手撕代码
- loss
- cross entropy
- DPO
- GRPO
- MoE
- PyTorch
difficulty: 4
freq: 高
updated: 2026-08-28
summary: 高频 loss 默写：带 label smoothing 的 cross entropy、DPO loss（Bradley-Terry 负对数似然）、GRPO
  loss（组内归一化优势 + 序列级）、MoE 负载均衡辅助损失。维度自洽、可直接运行。
source:
- https://arxiv.org/abs/2305.18290
questions:
- label smoothing 等价于什么正则？α 太大/太小？
- DPO loss 里 β 作用在哪一项？为什么需要 reference 模型？
- GRPO 为什么不需要 value model？优势怎么来的？
- MoE 负载均衡损失为什么要"鼓励专家被选频率均匀"？
- 序列级 logp 计算时为什么要除以 token 数（或 mask 求和）？
---

## 一句话结论

- **CE + label smoothing**：把 one-hot 目标平滑为 `(1-α)·onehot + α/K`，等价于 KL 到均匀。
- **DPO**：`−log σ( β·(logπθ(yw)−logπθ(yl) − (logπref(yw)−logπref(yl))) )`。
- **GRPO**：组内 `(r_i − mean)/std` 当优势，套 PPO-Clip，无需 critic。
- **MoE 负载均衡**：鼓励每个 token 被均匀分到各专家，用专家分配比例的熵/平方误差做辅助损失。

## 实现要点

- 序列级 logp：`logp = Σ_t log_softmax(logits_t)[y_t]`，用 loss mask 只算有效 token。
- DPO 需要"策略与参考模型对同一序列"的对数似然差。
- GRPO 的优势在 group 内标准化，注意 std 加小 epsilon 防零。

## 输入输出示例

```python
logits = torch.randn(2, 5, 100)          # [B,T,V]
target = torch.randint(0,100,(2,5))
ce = cross_entropy_smooth(logits, target, alpha=0.1)   # scalar

# DPO：给定 4 个 logp 序列
loss = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)    # scalar
```

## 核心代码

```python
import torch, torch.nn.functional as F

# ---------- CE + label smoothing ----------
def cross_entropy_smooth(logits, target, alpha=0.1, ignore_index=-100):
    V = logits.size(-1)
    logp = logits.log_softmax(-1)
    onehot = F.one_hot(target, V).float()
    smooth = onehot * (1 - alpha) + alpha / V
    loss = -(smooth * logp).sum(-1)
    mask = (target != ignore_index)
    return (loss * mask).sum() / mask.sum()

# ---------- DPO ----------
def seq_logp(logits, labels, mask):
    # logits:[B,T,V], labels:[B,T], mask:[B,T] 1=有效
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1) / mask.sum(-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # 每个都是序列级 logp 向量 [B]
    pi_logratio  = pi_w - pi_l
    ref_logratio = ref_w - ref_l
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()

# ---------- GRPO ----------
def grpo_loss(logps, rewards, old_logps=None, beta=0.04, eps=0.2):
    # logps:[B,T], rewards:[B]；组内标准化优势
    mean_r, std_r = rewards.mean(), rewards.std() + 1e-6
    adv = (rewards - mean_r) / std_r                       # [B]
    ratio = (logps - old_logps).exp() if old_logps is not None else torch.ones_like(logps)
    # 序列级平均（token 级亦可）
    seq_adv = adv.unsqueeze(-1)
    pol = -(ratio * seq_adv).mean()
    kl = ((old_logps - logps).exp() - 1 - (old_logps - logps)).mean() * beta
    return pol + kl

# ---------- MoE 负载均衡辅助损失 ----------
def moe_aux_loss(gates, expert_mask, n_experts):
    # gates:[B,T,E] 路由权重; expert_mask:[B,T,E] 是否选该专家
    f = expert_mask.float().mean(dim=(0, 1))              # 各专家被选频率 [E]
    P = gates.mean(dim=(0, 1))                            # 路由概率均值 [E]
    return (f * P).sum() * n_experts       # 鼓励 f≈P 均匀
```

## 面试追问

1. label smoothing 等价于什么正则？α 太大/太小？
2. DPO loss 里 β 作用在哪一项？为什么需要 reference 模型？
3. GRPO 为什么不需要 value model？优势怎么来的？
4. MoE 负载均衡损失为什么要"鼓励专家被选频率均匀"？
5. 序列级 logp 计算时为什么要除以 token 数（或 mask 求和）？

## 关联知识点

- [DPO 推导：从 RLHF 最优解反推闭式 loss](dpo-derivation)
- [GRPO 与 RLVR：组内相对优势与可验证奖励](grpo-and-rlvr)
- [MoE 混合专家：路由、负载均衡与 DeepSeekMoE](moe-architecture)
