---
id: data-engineering
title: 预训练数据工程：从 Common Crawl 到配比
module: 06-data-engineering
tags:
- 数据
- 预训练
- 去重
- 配比
- 消融实验
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 预训练数据是可复现的护城河：Common Crawl 经过 URL 过滤、正文提取、语言识别、MinHash 去重、质量打分后只剩原始体积的百分之几，配比则由小规模消融实验外推决定。
source:
- https://arxiv.org/abs/2406.17557
questions:
- 去重到什么程度合适？会不会去重过头？
- 为什么代码数据能提升数学推理能力？
- 数据配比怎么定，真的是靠消融实验吗？
- 多 epoch 到底能不能用？
- 数据去污染会不会误删？
---

## 一句话结论

现代开源复现（RefinedWeb、Dolma、RedPajama、FineWeb）已经证明一件事：**数据工程的质量比模型结构的改动影响更大**。同样的 7B 架构，用 FineWeb 训出来的模型明显强于用未处理的 Common Crawl。

标准流水线可以概括为一句话：

```
原始网页 → URL 过滤 → 正文提取 → 语言识别 → 去重 → 质量过滤 → 配比混合 → 去污染
```

**定量印象**：Common Crawl 每月原始 WARC 约 300~400 TB 压缩数据，经过完整流水线后，高质量英文子集只剩约几百亿到上万亿 token——**留存率通常在 1%~10%**。

## 原理推导

### 1. 为什么需要这么复杂的流水线

Common Crawl 的原始内容里充满了：

- 机器生成的垃圾页面（SEO 农场、自动生成的内容）
- 导航栏、广告、cookie 提示、页脚（正文提取要去掉）
- 重复内容（同一篇文章被无数站点转载，占比可能高达 30%~60%）
- 成人内容、垃圾邮件、恶意内容
- 多语言混杂（英文模型只保留高质量英文）

不处理的后果不是"学得慢"，而是**学歪**：模型会把高频出现的模板文本（导航、免责声明）当成语言模式的一部分，浪费容量还产生奇怪的生成行为。

### 2. 各步骤的做法

**URL 过滤**：
- 基于黑名单域名（成人、赌博、垃圾站）
- 基于 URL 模式（路径深度过浅、含随机字符串的多半是垃圾）
- 基于域名信誉

**正文提取（Boilerplate Removal）**：
- 规则法：trafilatura、jusText、readability（基于标签密度、链接密度、文本块结构）
- 模型法：用轻量分类器判断每个 DOM 块是否正文
- 关键指标：正文召回率 vs 模板残留率的权衡

**语言识别**：fastText 的 lid.176 分类器，按置信度阈值过滤（英文通常要求 p > 0.8 或 0.99，取决于想要的严格度）。

**去重（Dedup）**：

| 层级 | 方法 | 说明 |
|---|---|---|
| 精确去重 | 对全文或段落做 hash | 只去掉完全相同的，收益有限 |
| 模糊去重 | **MinHash / MinHashLSH** | 核心手段，用 Jaccard 相似度近似 |
| 段落级去重 | 对段落而非全文做 MinHash | 更细，能去掉转载中的公共段落 |
| 跨数据集去重 | 与评测集做 n-gram 重叠 | 去污染 |

MinHash 的原理：用 k 个随机哈希函数把文档映射成 k 个最小值（签名），两个集合的 Jaccard 相似度 ≈ 签名中对应位相等的比例：

```
J(A, B) ≈ |{i : minhash_i(A) = minhash_i(B)}| / k
```

配合 LSH（把签名分成 b 个 band，每个 band r 行，同一 band 全等的对进入同一桶），把 O(n²) 的相似度比较降到近似 O(n)：

```
候选对概率 = 1 - (1 - J^r)^b
```

典型参数：k=256 个哈希、b=32 个 band、每 band r=8 行，相似度阈值约 0.8。

**质量过滤**：

| 方法 | 做法 | 代表 |
|---|---|---|
| 启发式规则 | 词数、标点比例、字母数字比、停用词密度、困惑度阈值 | RefinedWeb、Gopher 规则集 |
| 轻量模型打分 | 用 fastText 或小模型在"高质量语料 vs 随机网页"上训练二分类器 | FineWeb 的 educational 分类器 |
| 模型困惑度 | 用 KenLM 或小 LM 算 PPL，过滤高 PPL（异常文本） | CCNet |
| LLM 打分 | 用大模型标注质量，蒸馏到小分类器 | 成本高，用在关键子集 |

**配比（Mixing Ratio）**：

核心方法：**在小模型上做消融实验，再把结论外推**。步骤：

1. 训练一系列 1B~3B 的小模型，每个用不同的数据配比
2. 在下游评测集上比较（用"每个领域的 loss"或"代表性 benchmark"作代理指标）
3. 选出最优配比，外推到大模型

关键结论（多家论文一致）：

- **代码数据的价值被严重低估**：加入代码能显著提升推理能力（不只是代码能力），现代配比里代码常占 10%~30%
- **数学/科学文本**（arXiv、教材）对推理能力提升明显
- **去重和高质量的收益远超"加更多数据"**
- 多语言数据会略微损害英文能力，但比例合适时（<20%）影响可控

典型配比参考（LLaMA-1 7B 的公开数据）：

```
CommonCrawl ( processed )   67.0%
C4                           15.0%
Github                        4.5%
Wikipedia                     4.5%
Books                         4.5%
ArXiv                         2.5%
StackExchange                 2.0%
```

**去污染（Decontamination）**：

评测集泄漏是"虚假提升"的最大来源。做法：

```
对每个 benchmark 的测试样本，取它的 n-gram 集合（通常 8-gram 或 13-gram）
在训练集中检索，若任一训练文档命中超过 X% 的 n-gram，则删除该文档
```

注意：n-gram 去污染对"改写过的泄漏"无效（如把测试题换个说法），所以还要结合 embedding 相似度、以及人工检查。

### 3. 数据量与 epoch

Chinchilla 最优的结论是**参数量和数据量要同步放大**（每参数约 20 token）。但实际工业界因为推理成本考虑，普遍**过训练**（over-trained）：LLaMA-1 7B 用 1T tokens（约 143 token/参数），LLaMA-2 用 2T，LLaMA-3 8B 用 15T（约 1875 token/参数）。

理由：训练一次、推理无数次，宁可多花训练算力换一个"更小但更强"的模型。

**多 epoch 的问题**：重复数据会导致记忆化（memorization），超过 4 个 epoch 收益急剧衰减，且增加隐私/版权风险。现代做法是尽量单 epoch、靠扩大数据源解决。

## 具体例子

**例 1：留存率的定量感受**

假设某月 Common Crawl：

```
原始 WARC (压缩)       : ~350 TB
解压后 HTML            : ~1.5 PB
URL 过滤后             : 剩 60%
正文提取后             : 剩 30%（去掉模板后 token 大减）
语言识别(英文 p>0.8)   : 剩 35%
MinHash 模糊去重       : 剩 50%（去掉转载）
质量过滤(分类器 top)   : 剩 30%

总留存率 = 0.6 × 0.3 × 0.35 × 0.5 × 0.3 = 0.945%
```

从 1.5 PB HTML → 约 14 TB 文本 → 约 3~4 万亿 token 的高质量子集。数量级符合公开数据集（FineWeb 约 15 万亿 token，是多年多快照累积的结果）。

**例 2：MinHash 的相似度估计**

文档 A 的词集合 {a,b,c,d,e}，文档 B 的词集合 {c,d,e,f,g}：

```
交集 = {c,d,e} → 3
并集 = {a,b,c,d,e,f,g} → 7
Jaccard = 3/7 = 0.4286
```

用 256 个哈希做 MinHash，期望约有 `256 × 0.4286 ≈ 110` 个位置签名相同。LSH 参数 b=32, r=8 时，成为候选对的概率：

```
P = 1 - (1 - 0.4286^8)^32 = 1 - (1 - 0.0011)^32 = 1 - 0.9654 = 0.035
```

即 Jaccard=0.43 的文档对只有 3.5% 概率被检查，而高相似度（J=0.9）的对：

```
P = 1 - (1 - 0.9^8)^32 = 1 - (1 - 0.4305)^32 ≈ 1.0
```

几乎必然被检出。这就是 LSH 的"S 曲线"——**在阈值附近急剧区分**，符合去重需求。

**例 3：配比消融实验的设计**

想决定"代码数据占比"，在 1B 模型上做：

```
固定总 token 数 = 20B（Chinchilla 最优的 20×）
实验组:  代码占比 0%, 5%, 10%, 20%, 40%
评测:    HumanEval(代码) + GSM8K(推理) + MMLU(知识) + HellaSwag(常识)
```

典型结果形态：代码从 0%→10% 时 GSM8K 和 HumanEval 同步大幅提升；10%→20% 时 HumanEval 继续涨但 MMLU 略降；超过 30% 后 MMLU 明显下降。**最优区间通常在 10%~20%**，且这个结论能被外推到 7B/70B。

**例 4：去污染的实际影响**

有研究报告过，某些模型在 GSM8K 上去污染后分数下降 3~8 个百分点。具体机制：

```
GSM8K 题目被大量数学学习网站收录 → 这些网站进了 Common Crawl
→ 模型见过原题或高度相似的改写字
```

n-gram 去污染（13-gram）能去掉直接复制，但去掉不了"参数改写"的版本，所以还要：
- 对 benchmark 做 embedding 近邻检索
- 人工抽样检查高风险样本
- 报告"去污染后的分数"与"原始分数"两个数字

## 代码速记

```python
import re, hashlib, random
from collections import Counter

# ---------- 1. 启发式质量过滤（Gopher/RefinedWeb 风格） ----------
def quality_filters(text: str) -> bool:
    """返回 True 表示保留"""
    words = text.split()
    n = len(words)
    if n < 50 or n > 100_000:                  # 长度
        return False
    # 单词平均长度（机器生成/垃圾常见超长串）
    if sum(len(w) for w in words) / n > 10:
        return False
    # 停用词比例（真正的自然语言有合理密度的 the/and/of）
    stop = {"the", "and", "of", "to", "a", "in", "is", "it", "that", "for"}
    if sum(w.lower() in stop for w in words) / n < 0.02:
        return False
    # 特殊字符比例
    if sum(not c.isalnum() and not c.isspace() for c in text) / len(text) > 0.3:
        return False
    # 重复行/重复 n-gram 比例
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.5:
        return False
    # 必须以句号等终止符结尾（正文提取完整）
    if text.strip()[-1] not in ".!?\"'":
        return False
    return True

# ---------- 2. MinHash + LSH ----------
class MinHash:
    def __init__(self, num_perm=256, seed=42):
        rng = random.Random(seed)
        # 通用哈希: h(x) = (a*x + b) % p
        self.p = (1 << 61) - 1
        self.a = [rng.randrange(1, self.p) for _ in range(num_perm)]
        self.b = [rng.randrange(0, self.p) for _ in range(num_perm)]
        self.k = num_perm

    def signature(self, tokens):
        hv = [int(hashlib.blake2b(t.encode(), digest_size=8).hexdigest(), 16)
              for t in set(tokens)]
        return [min((ai * h + bi) % self.p for h in hv) for ai, bi in zip(self.a, self.b)]

def jaccard_estimate(sig_a, sig_b):
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)

def lsh_bands(sig, b=32):
    """LSH 分桶：把签名切成 b 个 band，每 band 拼成 key"""
    r = len(sig) // b
    return [tuple(sig[i*r:(i+1)*r]) for i in range(b)]

# ---------- 3. n-gram 去污染 ----------
def ngrams(text, n=13):
    toks = re.findall(r"\w+", text.lower())
    return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))

def is_contaminated(doc, test_ngrams, threshold=0.8):
    """若文档包含任一测试样本的 >=80% n-gram，判为污染"""
    dg = ngrams(doc, 13)
    for tg in test_ngrams:
        hit = sum(1 for g in tg if g in dg)
        if hit / max(len(tg), 1) >= threshold:
            return True
    return False

# ---------- 4. 配比采样 ----------
def mix_sample(datasets, weights, total_tokens):
    """datasets: {name: iterator}；weights: {name: float}，按权重分配 token 预算"""
    budget = {k: int(total_tokens * v / sum(weights.values())) for k, v in weights.items()}
    out = Counter()
    for name, it in datasets.items():
        got = 0
        for doc in it:
            if got >= budget[name]:
                break
            out[name] += len(doc.split())
            got += len(doc.split())
    return dict(out)

print(mix_sample({"cc": iter([]), "code": iter([])},
                 {"cc": 0.85, "code": 0.15}, 1_000_000))
```

## 面试追问

**Q：去重到什么程度合适？会不会去重过头？**
A：去重收益存在边际递减，且过度去重的风险是**删掉了真实的重复模式**（如法律条文的固定表述、代码里的常见模板、教科书的标准定义），这些"重复"本身是有信息量的。实践上的做法：全文级去重阈值取 Jaccard 0.8~0.9（只去掉高度重复），段落级去重更激进但保留在每个文档内至少出现一次的段落。另外要保留一个"未去重"的对照做消融。

**Q：为什么代码数据能提升数学推理能力？**
A：有三种主流解释，面试时说"多因素"最稳妥：(1) 代码本身是结构化的多步推理，模型在代码上学会"分解问题→逐步求解"的模式；(2) 代码里包含大量变量名、函数调用、注释，天然带有符号推理和状态追踪的监督信号；(3) 代码数据的噪声低（能编译/能运行的有隐含质量筛选），信息密度高于网页。实证上，加入代码后 GSM8K 提升可达 5~10 个百分点。

**Q：数据配比怎么定，真的是靠消融实验吗？**
A：是的，但消融实验的成本很高（每个配比要训一个小模型），所以工业界会用**代理指标**降低开销：用"各数据源上的 held-out loss"或"小模型在代理 benchmark 上的表现"作指标，而不是完整训练。还有一个常用技巧是 **DoReMi / 分布鲁棒优化**——用一个小模型在 group DRO 目标下自动学习各数据源的权重，把人工搜索变成优化问题。

**Q：多 epoch 到底能不能用？**
A：能用几个 epoch，但收益递减且风险递增。经验数字：2~4 个 epoch 基本无损失，超过 4 个 epoch 后收益很小且开始出现明显的记忆化（模型能逐字背出训练文档）。这也是"数据受限"时的常见妥协——当高质量数据用完了，宁可重复高质量数据也不加低质量数据，但要控制 epoch 数并接受收益递减。

**Q：数据去污染会不会误删？**
A：会。n-gram 去污染对**通用短语**（如常见的固定搭配、标准定义、代码片段）会误伤，因为评测集里也会出现这些。常见缓解：只对"高风险的 n-gram"（长且稀有的 13-gram）做严格匹配，对短 n-gram 不做全文档删除而只做局部替换。另外要注意**不要对训练集做过度去污染导致数据分布偏移**——把包含常见知识的文档全删了，模型反而学不到东西。

**Q：合成数据在预训练里怎么用？**
A：两条路。(1) **教科书式合成**（如 phi 系列）：用 LLM 生成高质量、信息密集的"教科书"文本，证明小模型也能达到大模型的效果，关键是**信息密度和信息覆盖度**，而不是数据量。(2) **数据增强**：对现有数据做改写、翻译、结构化。风险是**模型坍缩（model collapse）**——反复用模型生成的数据训练会导致分布退化、多样性丧失，所以合成数据必须与真实数据混合，且比例要控制。

## 关联知识点
- Tokenizer 与词表设计（03-pretrain）
- Scaling Law 与 Chinchilla（03-pretrain）
- 长上下文训练与序列并行（02-attention）
- 继续预训练与领域适配（05-peft）
- 数据污染与评测可信度（09-eval）
