---
id: decoding-strategy
title: 解码策略：greedy / beam / top-k / top-p / temperature / repetition
module: 15-inference-algorithms
tags:
- 解码
- temperature
- top-p
- top-k
- beam search
- repetition penalty
- min-p
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 自回归生成每步从 logits 采样。temperature 对 logits 缩放改分布"锐度"；top-k 截前 k 个、top-p 截累积概率达
  p 的最小集合（动态大小）；repetition penalty 抑制重复。min-p 是新趋势，按分布熵自适应截断。
source:
- https://arxiv.org/abs/2211.17192
questions:
- temperature 为什么能让分布"变平"？T→0 收敛到什么？
- top-p 相比 top-k 好在哪？它有什么缺陷？
- min-p 解决了 top-p 的什么问题？
- repetition penalty 的公式？为什么不设为很大？
- 为什么对话模型多用采样而非 beam search？
---

## 一句话结论

- **greedy**：每步取 argmax，确定但易退化/重复。
- **temperature T**：logits ← logits/T，T>1 更平（更随机）、T<1 更尖（更确定）。
- **top-k / top-p(nucleus)**：只在前 k 个 / 累积概率≥p 的最小候选集里采样；top-p 自适应、更稳。
- **repetition penalty**：对已出现 token 的 logit 除以 >1 系数，抑制复读。

## 原理推导

### Temperature 的softmax 平滑

原始 `p_i = softmax(z_i/T)`。T 大 → `z_i/T` 差距缩小 → 分布更均匀（随机性↑）；T→0 → 趋近 one-hot（≈greedy）。数学上 T 等价于在能量模型里缩放"温度"，是高斯的方差因子。

### Top-p（nucleus）采样

把词按概率降序排列，取最小前缀使 `Σ_{i≤k} p_i ≥ p`，只在该集合里重新归一化采样。相比 top-k 的**固定大小**，top-p 在"高置信"时集合小（如只 2~3 个词）、"不确定"时集合大，更鲁棒。缺点：p 过大退化成全量、p 过小丢多样性。

### Min-p（新方案）

top-p 的缺陷：分布很尖时仍强制采很多词；分布平时截太少。min-p 改为：只在 `p_i ≥ α·max(p)` 的候选里采样（α 如 0.1）。高置信时只留接近峰值的少数词，低置信时自动放宽 → 比 top-p 更自适应，已在 llama.cpp / vLLM 支持。

### Repetition penalty

对已有 token 集合 R：`z_i ← z_i / θ`（若已出现，θ>1）或 `z_i ← z_i·θ`（若未出现）。θ=1 无作用，θ=1.2~1.5 常用。缓解"复读机"，但过大抑制合理重复（如列表、代码）。

### Beam search

维护 B 条候选序列，每步扩 B×V 再保留 top-B。适合有唯一正确答案的任务（翻译、SQL），但开放生成易产出"安全但无聊"的句；现代对话模型多用采样而非 beam。

## 具体例子

- **代码补全**：T=0（贪心）保证确定性可复现。
- **创意写作**：T=0.8~1.0 + top-p=0.9，保证多样性。
- **长文本复读**：加 repetition penalty 1.2 + no_repeat_ngram_size=3。
- **RAG 答案抽取**：T=0 让答案稳定。

## 代码速记

```python
import torch.nn.functional as F
def sample(logits, T=1.0, top_p=0.9, rep_penalty=1.0, generated=None):
    if rep_penalty != 1.0 and generated:
        logits[generated] /= rep_penalty          # 简化：对出现过的降权
    p = F.softmax(logits / T, dim=-1)
    # top-p
    sorted_p, idx = p.sort(descending=True)
    cum = sorted_p.cumsum(0)
    mask = cum - sorted_p > top_p                 # 超出累积的部分
    sorted_p[mask] = 0
    p = sorted_p.scatter(0, idx, sorted_p); p /= p.sum()
    return torch.multinomial(p, 1)
```

## 面试追问

1. temperature 为什么能让分布"变平"？T→0 收敛到什么？
2. top-p 相比 top-k 好在哪？它有什么缺陷？
3. min-p 解决了 top-p 的什么问题？
4. repetition penalty 的公式？为什么不设为很大？
5. 为什么对话模型多用采样而非 beam search？

## 关联知识点

- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
- [量化：对称/非对称、GPTQ 与 AWQ](quantization)
- [投机解码：draft-then-verify 与加速比推导](speculative-decoding)
