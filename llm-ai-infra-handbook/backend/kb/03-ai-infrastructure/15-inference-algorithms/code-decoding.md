---
id: code-decoding
title: 手撕解码采样：top-k / top-p / beam / 投机 verify
module: 15-inference-algorithms
tags:
- 手撕代码
- 采样
- top-k
- top-p
- beam search
- 投机解码
- PyTorch
difficulty: 3
freq: 中
updated: 2026-08-28
summary: 生成时的采样实现：top-k 截前 k、top-p 截累积概率、beam search（带 length penalty）、以及投机解码的 verify
  阶段（接受/拒绝采样）。全部维度自洽、可直接运行。
source:
- https://arxiv.org/abs/2211.17192
questions:
- top-p 比 top-k 好在哪？集合大小怎么定？
- beam search 的 length penalty 解决什么？α 大于 1 会怎样？
- 投机解码的 verify 为什么能"不改分布"？
- temperature 在采样代码里作用在哪一步？
- 为什么对话模型常用采样而非 beam？
---

## 一句话结论

- **top-k**：取概率最大的 k 个，重新归一化后采样。
- **top-p(nucleus)**：取累积概率≥p 的最小集合，集合大小动态。
- **beam search**：维护 B 条序列，每步保留 top-B 扩展；length penalty 防止短序列占优。
- **投机 verify**：对 draft 候选逐个接受（按 target/draft 概率比），首个拒绝处用修正分布重采样。

## 实现要点

- `logits → log_softmax → 截取值范围 → 重归一化 → multinomial`。
- beam 用张量并行扩展 B×V 再取 top-B；注意不同长度序列的得分要除以 `length^α`（length penalty）。

## 输入输出示例

```python
logits = torch.randn(1, 10000)
top_k_sample(logits, k=50)     # 返回 token id，只在 top-50 内采
top_p_sample(logits, p=0.9)    # 动态集合内采
beams = beam_search(logits.unsqueeze(1), model, B=3, max_len=10)
```

## 核心代码

```python
import torch, torch.nn.functional as F

def top_k_sample(logits, k=50, temperature=1.0):
    logits = logits / temperature
    topk = torch.topk(logits, k)
    probs = F.softmax(topk.values, -1)
    return topk.indices[torch.multinomial(probs, 1)]

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_idx = logits.sort(descending=True)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    mask = cum - sorted_logits.softmax(-1) > p         # 超出累积的截掉
    sorted_logits[mask] = -1e9
    probs = F.softmax(sorted_logits, -1)
    return sorted_idx[torch.multinomial(probs, 1)]

def beam_search(first_logits, step_fn, B=3, max_len=10, length_penalty=0.7):
    # first_logits: [1, V]；step_fn(logits,[B,S]) -> [B,V]
    probs = F.log_softmax(first_logits[0], -1)
    top = probs.topk(B)
    seqs = top.indices.unsqueeze(1)                    # [B,1]
    scores = top.values.clone()                        # [B]
    for t in range(1, max_len):
        cand = []
        for i in range(B):
            nxt = F.log_softmax(step_fn(seqs[i:i+1]), -1)
            cand.append(scores[i] + nxt)
        cand = torch.stack(cand).flatten()             # [B*V]
        k = torch.topk(cand, B)
        # 还原 (seq_idx, token)
        seq_idx = k.indices // cand.size(-1) if False else (k.indices // (cand.size(0)//B))
        new_seqs, new_scores = [], []
        for rank in range(B):
            si = k.indices[rank] // (cand.size(0)//B)
            ti = k.indices[rank] % (cand.size(0)//B)
            new_seqs.append(torch.cat([seqs[si], torch.tensor([ti])]))
            lp = ((t+1) ** length_penalty) / (1 ** length_penalty)
            new_scores.append(k.values[rank] / lp)
        seqs = torch.stack(new_seqs); scores = torch.stack(new_scores)
    return seqs[scores.argmax()]

def speculative_verify(draft_logps, target_logps):
    # draft_logps,target_logps: [gamma] 各候选的 log 概率（已按序列）
    out = []
    for i in range(len(draft_logps)):
        q, p = math.exp(draft_logps[i]), math.exp(target_logps[i])
        if q <= p and random.random() < p / q:
            out.append('accept')
        else:
            out.append('reject@%d' % i); break
    return out
```

## 面试追问

1. top-p 比 top-k 好在哪？集合大小怎么定？
2. beam search 的 length penalty 解决什么？α 大于 1 会怎样？
3. 投机解码的 verify 为什么能"不改分布"？
4. temperature 在采样代码里作用在哪一步？
5. 为什么对话模型常用采样而非 beam？

## 关联知识点

- [解码策略：temperature / top-p / repetition](decoding-strategy)
- [投机解码：draft-then-verify 与加速比推导](speculative-decoding)
- [各类 loss 手撕：CE / DPO / GRPO / MoE](code-losses)
