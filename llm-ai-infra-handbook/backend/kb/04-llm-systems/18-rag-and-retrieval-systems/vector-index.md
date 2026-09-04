---
id: vector-index
title: 向量索引：Flat / IVF / PQ / HNSW 与权衡
module: 18-rag-retrieval
tags:
- 向量检索
- HNSW
- IVF
- PQ
- ANN
- 召回率
- 延迟
difficulty: 4
freq: 中
updated: 2026-08-28
summary: 向量检索是在高维空间找近邻(ANN)。Flat 暴力精确但 O(Nd)；IVF 用聚类做倒排只搜近簇；PQ 乘积量化压缩向量省内存；HNSW 用多层图实现
  O(log N) 近似；DiskANN 把索引放磁盘支持十亿级。核心是召回率-延迟-QPS 的三角权衡。
source:
- https://arxiv.org/abs/2005.11401
questions:
- IVF 的 nprobe 怎么影响召回和延迟？
- PQ 为什么能压缩内存？代价是什么？
- HNSW 为什么是 O(log N)？M 和 efSearch 怎么调？
- Flat / IVF / HNSW 分别适合什么规模？
- 召回率、延迟、QPS 三者如何权衡？调参顺序？
---

## 一句话结论

- **Flat（暴力）**：精确最近邻，复杂度 O(N·d)，仅适合小库（<10万）。
- **IVF（倒排）**：k-means 聚类，查询只搜最近若干簇（nprobe），速度↑召回↓。
- **PQ（乘积量化）**：把向量切成 m 段各量化，用查表算距离，内存降到 1/m，但引入近似误差。
- **HNSW**：分层可导航小世界图，查询 O(log N)，召回高、内存较大，最常用。
- **DiskANN**：图索引放 SSD，支持十亿级、内存友好。

## 原理推导

### IVF（Inverted File）

1. 对全库做 k-means（k=nlist 个簇心）。
2. 每个向量归入最近簇心，建倒排列表。
3. 查询：先找最近的 nprobe 个簇心，只在那些簇内做精确距离。

- 召回率 ∝ nprobe/nlist；延迟 ∝ nprobe·(簇平均大小)·d。
- 调参：nlist 取 `√N` 量级；nprobe 从 1 逐步调，召回到 95%+ 即可停。

### PQ（Product Quantization）

把 d 维向量切成 m 段（每段 d/m 维），每段独立做 k-means（码本大小 2^8=256）：

- 原向量用 m 个"码字下标"表示，存储从 `d×4B` 降到 `m×1B`。
- 距离用**预计算的码本距离表**查表累加（`d(x,y) ≈ Σ_j table_j[code_x_j][code_y_j]`），无需还原向量。
- 代价：量化误差，召回略降；常与 IVF 组合（IVF-PQ）。

### HNSW（Hierarchical NSW）

- 多层图：顶层稀疏（长边，快速跨区），底层稠密（短边，精确定位）。
- 插入时按概率 `p` 决定"能升到第几层"，形成类似跳表的结构。
- 查询从顶层贪心走到邻居更近的节点，逐层下降，到底层得近邻。
- **复杂度**：每层的候选访问数约 `O(log N)`，总查询 `O(log N)`。
- 参数：`M`（每层最大连接数，越大召回↑内存↑）、`efSearch`（动态候选集大小，越大越准越慢）、`efConstruction`（建图质量）。

### DiskANN / Vamana

- 在 SSD 上存图，内存只放导航结构 → 支持十亿级向量、单机能跑。
- 用"磁盘友好的图遍历顺序"减少随机读。

## 具体例子

| 场景 | 选索引 | 理由 |
|---|---|---|
| 10万条知识库，要精确 | Flat / HNSW | 库小，HNSW 够快且近似无损 |
| 千万级，内存紧张 | IVF-PQ | 压缩 + 倒排，内存降 1/m |
| 亿级在线检索 | HNSW（内存足）或 DiskANN | 高 QPS 或内存受限 |
| 要求 99% 召回 | 调高 efSearch / nprobe | 牺牲一点延迟 |

**调参顺序**（经验）：先定索引类型 → 调 `nprobe`(IVF)/`efSearch`(HNSW) 到召回达标 → 再压内存（PQ）。

## 代码速记

```python
# faiss 三种索引（示意）
import faiss
index_flat = faiss.IndexFlatIP(d)                       # 精确内积
index_ivf  = faiss.index_factory(d, "IVF1024,PQ32")     # 倒排+PQ
index_hnsw = faiss.index_factory(d, "HNSW32")           # 图
index_ivf.train(x); index_ivf.add(x)
# 查询
D, I = index_hnsw.search(q, k=10)      # efSearch 影响召回
```

## 面试追问

1. IVF 的 nprobe 怎么影响召回和延迟？
2. PQ 为什么能压缩内存？代价是什么？
3. HNSW 为什么是 O(log N)？M 和 efSearch 怎么调？
4. Flat / IVF / HNSW 分别适合什么规模？
5. 召回率、延迟、QPS 三者如何权衡？调参顺序？

## 关联知识点

- [RAG 流水线：切分、召回、重排与失败归因](rag-pipeline)
- [句向量模型：InfoNCE、难负例与 Matryoshka](embedding-model)
- [上下文工程：prompt 设计、压缩与记忆](context-engineering)
