---
id: pretrain-objective
title: 预训练目标：CLM / MLM / Prefix-LM / Span Corruption 与 FIM
module: 02-language-modeling-tokenization
tags:
- 预训练目标
- CLM
- MLM
- Prefix-LM
- Span Corruption
- FIM
- decoder-only
difficulty: 3
freq: 中
updated: 2026-08-28
summary: 预训练目标决定"模型从数据里学什么结构"。MLM（BERT）是双向去噪、CLM（GPT）是单向自回归、Span Corruption（T5）随机掩片段、Prefix-LM
  混合二者。现代 LLM 几乎全是 CLM，因为它与"生成式推理"目标一致、且训推一致。FIM 让同一 CLM 也能做"中间填充"，是代码模型的关键技巧。
source:
- https://arxiv.org/abs/1808.06226
questions:
- CLM 的 loss 为什么要屏蔽第 1 个 token 和 padding？不屏蔽会怎样？
- 为什么现在主流是 decoder-only 而不是 Encoder-Decoder / Prefix-LM？
- FIM 是怎么做到"不改变 CLM 也能填充中间"的？PSM 和 SPM 区别？
- Span Corruption 相比 MLM 解决了什么痛点？
- Prefix-LM 在什么场景下比纯 CLM 好？为什么没成为主流？
---

## 一句话结论

- **CLM（因果语言建模）**：`P(x_t | x_{<t})`，单向，是 GPT/LLaMA/Qwen 等的标准目标，也是当前主流。
- **MLM（掩码语言建模）**：随机遮住 15% token 让模型还原，双向，BERT 用；不适合自回归生成。
- **Span Corruption（T5）**：遮住连续片段并替换为哨兵 token，序列到序列地还原，兼顾双向与生成。
- **为什么 decoder-only 胜出**：CLM 的"训练目标 = 推理方式 = 数据利用效率"三者一致，且 scaling 友好；Encoder-Decoder 的额外结构收益在大规模下被"规模 + 数据"抹平。

## 原理推导

### CLM 的 loss 与有效 token 数

对长度 T 的序列，标准 CLM 只对**非填充**的 token 求交叉熵：

```
L = - (1 / T_eff) · Σ_{t=1}^{T} 1[t 非 padding] · log P(x_t | x_{<t}; θ)
```

- 关键坑：多文档拼接到一个 sequence 时，第 1 个 token 没有前文、不应参与 loss（`-100` 标签屏蔽）；padding token 也要屏蔽，否则稀释梯度。
- **有效 token 数 T_eff** 是计算"训练 token 数"的真正分母，也是 scaling law 里 D 的来源。

### MLM

```
输入:  [CLS] the cat [MASK] on the [MASK]   → 还原 "sat" "mat"
L = - Σ_{被遮 token} log P(x_m | x_{\masked})
```
- 遮 15%（80%→[MASK], 10%→随机词, 10%→原词，防过拟合到 [MASK] 信号）。
- 双向上下文 → 适合理解/分类，但[MASK]在预训练-微调不一致，且无法自然生成。

### Span Corruption（T5）

```
原始:  the cat sat on the mat
遮片段(span): the cat <X> on the <Y>   (X,Y 为哨兵)
目标:  <X> the cat <Y> mat
```
- 模型读"损坏序列"，以自回归方式输出"被遮片段 + 哨兵"，可并行还原多个片段。
- 相比 MLM 更接近 seq2seq 生成，且避免 [MASK] 不一致问题。

### FIM（Fill-In-the-Middle，代码模型关键）

把 `(prefix, suffix, middle)` 重排为 `(suffix → prefix → middle)` 再按 CLM 训练：

```
PSM:  <PRE> prefix <SUF> suffix <MID> middle
```
- 训练时以一定概率（如 50%）做此置换，推理时模型能"给定前后文补全中间"，极大提升代码编辑/插入能力（StarCoder、Code Llama、Qwen-Coder 均用）。

## 具体例子

- **GPT-3 / LLaMA**：纯 CLM，next-token prediction。
- **BERT-base**：MLM + NSP（下一句预测，后被证明作用有限）。
- **T5 / UL2**：Span Corruption；UL2 进一步把"普通去噪、长跨度去噪、前缀 LM"三种目标混合（mode token 切换），统一理解+生成。
- **StarCoder / DeepSeek-Coder**：CLM + FIM（PSM/SPM 两种置换），使代码补全支持中缀填充。

## 面试追问

1. CLM 的 loss 为什么要屏蔽第 1 个 token 和 padding？不屏蔽会怎样？
2. 为什么现在主流是 decoder-only 而不是 Encoder-Decoder / Prefix-LM？
3. FIM 是怎么做到"不改变 CLM 也能填充中间"的？PSM 和 SPM 区别？
4. Span Corruption 相比 MLM 解决了什么痛点？
5. Prefix-LM 在什么场景下比纯 CLM 好？为什么没成为主流？

## 关联知识点

- [Scaling Law：Kaplan、Chinchilla 与推理时扩展](scaling-law)
- [预训练数据工程：从 Common Crawl 到配比](data-engineering)
- [Tokenizer：BPE / WordPiece / Unigram 与词表设计](tokenizer)
