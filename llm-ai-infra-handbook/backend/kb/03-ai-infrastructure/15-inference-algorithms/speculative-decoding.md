---
id: speculative-decoding
title: 投机解码：draft-then-verify 与加速比推导
module: 15-inference-algorithms
tags:
- 投机解码
- speculative decoding
- 自回归
- 加速
- Medusa
- EAGLE
difficulty: 4
freq: 中
updated: 2026-08-28
summary: LLM decode 每步只产 1 个 token 且受限于显存带宽（memory-bound）。投机解码用"小而快的 draft 模型一次猜 γ
  个 token，大模型一次并行验证"，在不改变输出分布的前提下把每步产出从 1 提升到约 (1-p^{γ+1})/(1-p) 个 token。
source:
- https://arxiv.org/abs/2211.17192
questions:
- 为什么 decode 是 memory-bound 而不是 compute-bound？
- 投机解码为什么能"不改输出分布"？拒绝采样的角色？
- 推导每步期望产出 (1-p^{γ+1})/(1-p)。p 和 γ 怎么影响加速？
- 自投机 / Medusa / EAGLE 各靠什么出草稿？
- 投机解码的加速上限受什么限制？
---

## 一句话结论

- **瓶颈**：自回归 decode 每步只算 1 个 token，但一次前向要读完整模型权重（~百 GB 级），是 **memory-bound**，算力大量闲置。
- **投机解码**：draft 小模型（或同模型浅层）一次猜 γ 个候选，target 大模型**一次前向并行验证**全部；验证通过的保留，首个失败处由 target 重采样。
- **分布不变**：拒绝采样保证输出与"直接逐 token 跑 target"完全同分布。
- **加速比**：每步产出 token 数期望 `E[T] = (1 - p^{γ+1})/(1-p)`（p 为单步接受率），理想 p→1 时趋近 γ+1。

## 原理推导

### 为什么能不改分布

设 draft 分布 `q`、target 分布 `p`。对每个候选位置 i：
- 若 `q(x_i) ≤ p(x_i)`：以概率 `p(x_i)/q(x_i)` 接受；
- 若 `q(x_i) > p(x_i)`：直接接受，并在拒绝时以修正分布 `(p-q)/(1-q)` 重采样。

这套拒绝采样（classic 1847）保证最终样本来自 `p`。关键：**target 只做一次前向**覆盖所有 γ 候选（attention 用 prefill 式并行），所以验证成本≈生成 1 个 token，却可能产出 γ+1 个。

### 期望加速比推导

设每步接受概率为 p（独立近似），候选数 γ。记接受个数 K：

- K=k（k<γ）：前 k 个通过、第 k+1 个被拒 → 产出 k+1 个 token，概率 `p^k(1-p)`。
- K=γ（全通过）：产出 γ+1 个 token，概率 `p^γ`。

```
E[T] = Σ_{k=0}^{γ-1} (k+1)·p^k(1-p) + (γ+1)·p^γ
     = (1 - p^{γ+1}) / (1 - p)
```

- γ=1：E[T]=1+p；γ→∞ 且 p→1：E[T]→γ+1。
- **实际 wall-clock 加速** ≈ E[T] ×（target 单步成本 /（target 单步成本 + γ×draft 单步成本））。draft 越便宜、p 越高，越接近 E[T]。

### 近似可接受率 p

p 取决于 draft 与 target 的分布相似度。若两者接近（如同一模型的不同层、或强小模型），p 可达 0.7~0.9 → γ=5 时 E[T]≈ (1-0.9^6)/(0.1)≈4.7 倍产出。工程上常见 2~3× 端到端加速。

## 具体例子

- **同模型自投机**：用 target 的前几层当 draft（如 12 层中的前 4 层）一次出 γ 个，再用全部层验证 → 零额外模型。
- **Medusa**：在主干上挂多个"预测头"并行预测未来若干 token（head1 预测第 t+1、head2 预测 t+2…），树状验证。
- **EAGLE / EAGLE2**：基于"上层特征可预测下层"做草稿，用一层小 LM 在特征空间自回归，接受率显著高于随机草稿。

## 代码速记

```python
# 投机解码主循环（示意）
for _ in range(steps):
    draft = [draft_model(x) for _ in range(gamma)]      # 小模型快出 γ 个
    # target 一次前向验证整段
    logits_target = target_model(x + draft)             # 并行
    n_accept = 0
    for i, d in enumerate(draft):
        p_i, q_i = softmax(logits_target[i]), softmax(draft_logits[i])
        if q_i[d] <= p_i[d] and rand() < p_i[d]/q_i[d]:
            n_accept += 1
        else:                                            # 拒绝 → 用修正分布重采样
            x = sample_corrected(p_i, q_i); break
    if n_accept == gamma: x = sample(logits_target[gamma])
    output += draft[:n_accept] + [x]
```

## 面试追问

1. 为什么 decode 是 memory-bound 而不是 compute-bound？
2. 投机解码为什么能"不改输出分布"？拒绝采样的角色？
3. 推导每步期望产出 (1-p^{γ+1})/(1-p)。p 和 γ 怎么影响加速？
4. 自投机 / Medusa / EAGLE 各靠什么出草稿？
5. 投机解码的加速上限受什么限制？

## 关联知识点

- [KV Cache：原理、显存公式与两阶段瓶颈](kv-cache)
- [量化：对称/非对称、GPTQ 与 AWQ](quantization)
- [推理引擎：Continuous Batching 与 PD 分离](inference-engine)
