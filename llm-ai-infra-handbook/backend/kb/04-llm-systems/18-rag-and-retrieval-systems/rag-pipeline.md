---
id: rag-pipeline
title: RAG 流水线：切分、召回、重排与失败归因
module: 18-rag-retrieval
tags:
- RAG
- 切分
- 召回
- 重排
- HyDE
- 失败模式
difficulty: 3
freq: 高
updated: 2026-08-28
summary: RAG = 检索 + 生成。链路为：文档解析 → 切分 → 向量化建库 → 查询召回(Bi-Encoder) → 重排(Cross-Encoder)
  → 拼进 prompt 生成。常见失败分三类：召不回、召回但排序错、召回了但生成没用上。
source:
- https://arxiv.org/abs/2005.11401
questions:
- 为什么 RAG 要两阶段（召回+重排）而不是一步到位？
- chunk size 怎么选？为什么 overlap 有用？
- HyDE 为什么能提升召回？
- RRF 为什么比直接加权各检索器分数更鲁棒？
- RAG 答非所问，怎么判断是召回问题还是生成问题？
---

## 一句话结论

- **切分策略**：固定长度/递归/语义/按结构；chunk 太大噪声多、太小丢上下文，经验 256~512 token + 小 overlap。
- **两阶段召回**：Bi-Encoder（双塔，快，做粗召回）→ Cross-Encoder（交互，准，做重排 Top-k）。
- **查询改写 / HyDE**：把用户问题改写成更利于检索的 query，或用"假设性文档"向量检索。
- **多路融合 RRF**：`score_i = Σ 1/(k+rank_i)`，把不同检索器结果合并。

## 原理推导

### 为什么需要两阶段

- **Bi-Encoder**：query 和 doc 各自编码为向量，`sim = f(q)·f(d)`。可离线建库、在线只编码 query → 毫秒级召回百万级文档，但**无交互**信息、精度有限。
- **Cross-Encoder**：把 `[q; d]` 拼一起过一次 transformer 得相关性分数 → 精度高，但每次要对 query×每个候选做前向，**只能用于少量候选重排**（如召回 100 重排到 10）。

→ 标准做法：Bi-Encoder 召回 Top-100，Cross-Encoder 重排 Top-10。

### 切分权衡

| 策略 | 优点 | 缺点 |
|---|---|---|
| 固定长度 | 简单、均匀 | 切断语义（句子/表格被劈） |
| 递归切分 | 按段落/句子边界 | 仍可能跨主题 |
| 语义切分 | 按相似度断点，语义完整 | 计算贵、可能过大 |
| 按结构 | 表格/标题保留 | 依赖结构化源 |

经验：chunk ≈ 256~512 token，overlap ≈ 10~20%（让边界上下文连续），并对表格/代码特殊处理。

### RRF（Reciprocal Rank Fusion）

多个召回源（BM25 + 向量 + 知识图谱）结果融合：

```
RRF_score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d)),   k 常取 60
```

不依赖各源分数的绝对尺度，只看排名 → 鲁棒地合并异构检索器。

### HyDE（Hypothetical Document Embeddings）

用户问"怎么做 X"，让 LLM 先生成一段"理想答案文档"，再用这段假文档的向量去检索 → 假文档的语义空间比短 query 更接近真实文档，召回更准。

## 具体例子

**企业知识库问答**：
1. 解析 PDF（保留标题层级与表格）→ 递归切分 400 token + overlap 64。
2. 建库：Bi-Encoder 向量（如 bge-large）。
3. query 改写（"2024 报销政策" → "公司差旅报销标准与额度"）+ BM25 互补。
4. 召回 50 → Cross-Encoder 重排 8 → 拼进 prompt。
5. 生成时要求"只基于给定资料，标注出处"。

**失败归因**：
- 答案说"不知道"但资料里有 → 召回失败（切分/索引问题）。
- 资料召回了但答案没引用 → 排序或生成 prompt 问题。
- 召回相关但答非所问 → 生成阶段没"用上"（需更强 instruction / 压缩噪声）。

## 代码速记

```python
# 两阶段召回 + RRF 融合（示意）
def rrf(ranks_list, k=60):
    score = defaultdict(float)
    for ranks in ranks_list:                 # 每个检索器给出 doc->rank
        for doc, r in ranks.items():
            score[doc] += 1/(k+r)
    return sorted(score, key=score.get, reverse=True)[:top_n]
# BM25 召回 + 向量召回 各出一列 ranks，融合后送 Cross-Encoder 重排
```

## 面试追问

1. 为什么 RAG 要两阶段（召回+重排）而不是一步到位？
2. chunk size 怎么选？为什么 overlap 有用？
3. HyDE 为什么能提升召回？
4. RRF 为什么比直接加权各检索器分数更鲁棒？
5. RAG 答非所问，怎么判断是召回问题还是生成问题？

## 关联知识点

- [向量索引：Flat/IVF/PQ/HNSW 与权衡](vector-index)
- [句向量模型：InfoNCE、难负例与 Matryoshka](embedding-model)
- [上下文工程：prompt 设计、压缩与记忆](context-engineering)
