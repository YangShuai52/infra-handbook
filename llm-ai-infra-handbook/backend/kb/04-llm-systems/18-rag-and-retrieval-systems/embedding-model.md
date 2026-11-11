---
id: embedding-model
title: 句向量模型：InfoNCE、难负例与 Matryoshka
module: 18-rag-retrieval
tags:
- 句向量
- 对比学习
- InfoNCE
- 难负例
- Matryoshka
- MTEB
difficulty: 4
freq: 中
updated: 2026-08-28
summary: 句向量模型用双塔 + 对比学习把语义相近的句子映射到相近向量。训练核心是 InfoNCE loss 与难负例挖掘；in-batch negatives
  高效利用批内样本；Matryoshka 表示学习让向量可截断维度；MTEB/C-MTEB 是标准评测。
source:
- https://arxiv.org/abs/2005.11401
questions:
- InfoNCE 为什么叫"在 batch 内做分类"？负例越多越好吗？
- in-batch negatives 是什么？有什么坑（false negative）？
- 难负例为什么比随机负例好？太难会怎样？
- Matryoshka 为什么能截断维度还不掉点？
- 句向量为什么要归一化？用余弦还是点积？
---

## 一句话结论

- **双塔 + 对比学习**：(anchor, positive, negative) 三元组，目标让正例向量更近、负例更远。
- **InfoNCE**：把对比学习写成"在 batch 内做分类"——正确配对的对数似然，负例越多越接近互信息下界。
- **难负例（hard negatives）**：与正例语义接近却错的样本，比随机负例学得更好，但太难会变成噪声。
- **Matryoshka**：同一向量不同前缀维度都可用，按需截断省存储/提速。

## 原理推导

### InfoNCE loss

给定 anchor `a`、正例 `p`、一批负例 `{n_i}`，相似度用点积/余弦 `s(·,·)`、温度 τ：

```
L = - log [ exp(s(a,p)/τ) / ( exp(s(a,p)/τ) + Σ_i exp(s(a,n_i)/τ) ) ]
```

- 分子是"正例对"；分母是"正例 + 所有负例"（即在一个 batch 内做 softmax 分类）。
- **in-batch negatives**：一个 batch 里其他样本的 positive 自动当本样本的 negative → 不额外采负例就获得大量负例，效率极高。
- τ 控制分布锐度：小 τ 更强调难样本。

### 为什么难负例重要

随机负例太"容易"（语义差很远，模型一眼区分），梯度信息少。难负例（同主题不同答案、近义不同义）迫使模型学细粒度区分。但**过难负例**（实为正确但标注噪声）会损害训练 → 常做"中等难"挖掘（如 BM25 召回但非正例的前几名）。

### Matryoshka Representation Learning (MRL)

训练时除主 loss 外，对向量的前 `d'`（d'=8,16,32,...）维也计算对比 loss：

```
L_total = Σ_{d' ∈ Dims} λ_{d'} · L_infoNCE(前 d' 维)
```

→ 推理时可用 256 维（高精度）或 64 维（快检索），同一模型多分辨率。

### 归一化与相似度选择

- 训练末尾常对向量 L2 归一化，使点积 = 余弦，距离只取决于夹角，避免长度主导。
- 检索用余弦更稳；但归一化后极端长度信息丢失，某些任务（长文档 vs 短句）需权衡。

## 具体例子

- **BGE / GTE / E5**：主流开源句向量，多语言/中文强，常配指令前缀（"为检索表示这段passage："）。
- **指令前缀**：同一模型检索 query 和 passage 用不同指令，拉开两者分布，提升对齐。
- **维度截断**：Matryoshka 模型在亿级库里用 128 维做 ANN，省 4× 内存，精度仅微降。

## 代码速记

```python
# InfoNCE（in-batch negatives，简化）
a, p = encode(anchors), encode(positives)      # [B, d]
a, p = F.normalize(a), F.normalize(p)
sim = a @ p.t() / tau                          # [B,B]，对角线为正例
pos = torch.diag(sim)
loss = -torch.log(torch.exp(pos) / torch.exp(sim).sum(-1)).mean()
# 负例 = 同 batch 其他行（in-batch negatives）
```

## 面试追问

1. InfoNCE 为什么叫"在 batch 内做分类"？负例越多越好吗？
2. in-batch negatives 是什么？有什么坑（false negative）？
3. 难负例为什么比随机负例好？太难会怎样？
4. Matryoshka 为什么能截断维度还不掉点？
5. 句向量为什么要归一化？用余弦还是点积？

## 关联知识点

- [向量索引：Flat/IVF/PQ/HNSW 与权衡](vector-index)
- [RAG 流水线：切分、召回、重排与失败归因](rag-pipeline)
- [上下文工程：prompt 设计、压缩与记忆](context-engineering)
