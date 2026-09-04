---
id: tokenizer
title: Tokenizer：BPE / WordPiece / Unigram 与词表设计
module: 02-language-modeling-tokenization
tags:
- Tokenizer
- BPE
- 词表
- 分词
- 高频
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 主流 LLM 用字节级 BPE（BBPE），词表 10 万~15 万；词表大小影响参数量（2·V·d）、压缩率、多语言公平性；数字必须逐位切分否则算术能力崩。
source:
- https://arxiv.org/abs/1808.06226
questions:
- BPE 和 WordPiece 的本质区别是什么？
- 为什么现在都用 BBPE 而不是 Unigram？
- 多语言模型怎么设计词表？
- 词表越大越好吗？
- 为什么很多模型要扩词表（如 LLaMA 加中文 token）？
---

## 一句话结论

现代 LLM 的 tokenizer 事实标准：**字节级 BPE（BBPE）+ 词表 100k~152k + 数字按位切分**。

需要记住的三个判断：

1. **词表大小影响参数量**：embedding + lm_head 共 `2·V·d`，词表从 32k 提到 128k，7B 模型多约 7.9 亿参数（占 12%）。
2. **压缩率 ≠ 质量**：词表越大，同样文本用的 token 越少（推理更快、上下文更长），但稀有 token 学不好（训练样本不足）。
3. **分词方式直接决定算术能力**：数字若被切成 `1234`（一个 token），模型无法学到逐位进位，算术能力会显著崩坏。

## 原理推导

### 1. 为什么需要子词（Subword）

纯词级（word-level）的问题：
- 词表爆炸（百万级）
- OOV（未登录词）无法处理
- 无法处理形态变化（run/running/ran 各算一个词）

纯字符级（char-level）的问题：
- 序列长度爆炸（同一段文本 token 数多 4~6 倍）
- attention 是 O(S²)，长度代价极高
- 语义单元被拆碎，学起来难

子词的折中：常用词是整体 token，生僻词拆成有意义的片段。

```
"unhappiness" → ["un", "happi", "ness"]
"transformer" → ["transform", "er"]
```

### 2. BPE（Byte Pair Encoding）

训练算法（自底向上合并）：

```
1. 初始化：词表 = 所有基础单元（字符，或字节）
2. 统计训练语料中所有相邻符号对的频次
3. 把频次最高的一对合并成新符号，加入词表
4. 重复 2-3，直到达到目标词表大小
```

**手算示例**：语料 `{"low": 5, "lower": 2, "newest": 6, "widest": 3}`

```
初始（字符级，加 </w> 结束符）:
  l o w </w>       ×5
  l o w e r </w>   ×2
  n e w e s t </w> ×6
  w i d e s t </w> ×3

第1轮统计相邻对:
  (l,o)=7  (o,w)=7  (w,</w>)=5  (w,e)=8  (e,r)=2  (r,</w>)=2
  (n,e)=6  (e,w)=6  (e,s)=9  (s,t)=9  (t,</w>)=9  (w,i)=3  (i,d)=3  (d,e)=3
  → 最高 (e,s)=9 和 (s,t)=9，取 (e,s) 合并为 "es"

第1轮后:
  l o w </w> ×5 | l o w e r </w> ×2 | n e w es t </w> ×6 | w i d es t </w> ×3

第2轮: (es,t)=9 最高 → 合并为 "est"
第3轮: (est,</w>)=9 最高 → 合并为 "est</w>"
...
```

GPT-2 的实现还加了一个正则预分词（pre-tokenization），先把文本切成"词块"再在块内做 BPE，避免跨词合并（如 "dog" 和 "." 不合并）。

### 3. 字节级 BPE（BBPE）

GPT-2 的关键改进：**基础单元不是字符而是 UTF-8 字节**（256 个）。

```
"中" 的 UTF-8 = E4 B8 AD（3 字节）
初始:  ["\xe4", "\xb8", "\xad"]
多轮合并后: ["中"]（一个 token）
```

优势：
- **词表基础只有 256 个，绝无 OOV**（任何输入都能表示）
- 多语言天然统一（不需要为每种语言设计字符集）
- 罕见字符退化为字节级表示，仍可处理

代价：**非英文的压缩率低**。UTF-8 里一个中文字符 3 字节，即使合并成 1 个 token 也"占"了 3 个字节的配额；早期的 GPT-2 词表（50k）里中文 token 极少，导致中文文本 token 数是英文的 2~3 倍（更贵的推理、更短的上下文）。后来的模型（Qwen、LLaMA-3）都专门扩了多语言 token。

### 4. WordPiece（BERT）

与 BPE 的区别在**合并准则**：BPE 选频次最高的对，WordPiece 选**最大化语言模型似然**的对：

```
score(a, b) = freq(ab) / (freq(a) × freq(b))
```

即：两个片段共同出现的频率，相对于它们独立出现的频率的比值。这倾向于合并"强关联"的对，而不是单纯高频的对。

另外 WordPiece 用 `##` 标记非词首的子词（`"unhappiness" → ["un", "##happi", "##ness"]`），BPE 不需要。

### 5. Unigram / SentencePiece（LLaMA-1、T5）

与 BPE 相反，**自顶向下**：

```
1. 初始化一个很大的候选词表（所有子串，或 BPE 学到的词表的超集）
2. 用 EM 算法估计每个 token 的概率（unigram 语言模型）
3. 计算去掉每个 token 后总损失的增加量（loss 增量）
4. 删掉损失增量最小的那批 token（如每次删 10%）
5. 重复 2-4 直到目标词表大小
```

优势：
- 可以给出**多种分词方式及概率**（支持子词正则化，用作数据增强）
- 直接以语言模型似然为目标，理论上更优

SentencePiece 是这套算法的工程封装，额外做了两件重要的事：
- 把空格当作普通字符（`▁`），实现**可逆分词**（detokenize 后能完全还原原文）
- 提供 BPE 和 Unigram 两种实现

### 6. 词表大小的影响

**参数量**（不共享 embedding）：

```
embedding + lm_head = 2 × V × d
```

| 词表 | d=4096 时的参数量 | 占 7B 比例 |
|---|---|---|
| 32,000 | 262 M | 3.9% |
| 65,536 | 537 M | 8.0% |
| 100,000 | 819 M | 12.2% |
| 152,064（Qwen2） | 1,246 M | 18.5%（相对 Qwen 的 d） |

**压缩率**（bytes per token，英文）：

```
32k 词表  ≈ 4.0 bytes/token
100k 词表 ≈ 4.6 bytes/token
128k 词表 ≈ 4.8 bytes/token
```

收益递减明显：从 32k 到 100k 提升 15%，100k 到 200k 提升不足 5%。

**训练充分性**：词表越大，长尾 token 在训练集中出现次数越少。经验上每个 token 至少要出现几百次才能学好，词表超过 200k 后长尾 token 的 embedding 基本是噪声。

### 7. 数字切分（关键细节）

标准 BPE 会把 `1234` 合并成一个 token（因为高频）。后果：

```
模型看到的是 "1234" 这个整体符号，无法访问它的十进制结构
→ 学不会进位，多位数加法/乘法准确率断崖式下降
```

**LLaMA 的做法**：把数字强制拆成单个 digit。

```
"1234" → ["1", "2", "3", "4"]
```

代价：数字文本 token 数变多（如 "3.14159" 要 7 个 token）。收益：算术能力显著提升，且**从左边逐位输出**的结果符合人类读数习惯。

更进一步的做法（如某些数学专用模型）：数字按 3 位分组（从右往左，如 `1,234,567` → `1` `234` `567`），兼顾效率和可解析性。

## 具体例子

**例 1：BPE 手算（完整 3 轮）**

语料：`aaabdaaabac`（经典示例）

```
初始: a a a b d a a a b a c
统计相邻对: (a,a)=4  (a,b)=2  (b,d)=1  (d,a)=1  (b,a)=1  (a,c)=1
最高: (a,a)=4 → 合并为 "Z"(即 aa)

序列: Z a b d Z a b a c
统计: (Z,a)=2  (a,b)=2  (b,d)=1  (d,Z)=1  (b,a)=1  (a,c)=1
最高: (Z,a)=2 → 合并为 "Y"(即 Zaa = aaa)

序列: Y b d Y b a c
统计: (Y,b)=2  (b,d)=1  (d,Y)=1  (b,a)=1  (a,c)=1
最高: (Y,b)=2 → 合并为 "X"(即 Yb = aaab)

最终: X d X a c
词表增量: aa, aaa, aaab
```

**例 2：词表大小对参数量的影响（7B，d=4096）**

```
V=32000:   2 × 32000 × 4096 = 262,144,000  = 262 M
V=128256:  2 × 128256 × 4096 = 1,050,  ... 计算: 2×128256×4096 = 1,050,  (精确) 1,050,  let's compute: 128256 × 4096 = 525,336,576; ×2 = 1,050,673,152 ≈ 1.05 B
```

即词表从 32k 提到 128k，**多出 789 M 参数**（占 6.74B 的 11.7%）。这部分参数不参与"深度计算"（只是查表和投影），所以有些工作会建议共享 embedding 或用更小的词表 + 更深的层数。

**例 3：中文压缩率对比**

同一段中文（1000 字，UTF-8 约 3000 字节）：

```
GPT-2 词表(50k, 以英文为主):  约 1500~2000 tokens  (1.5~2 tokens/字)
LLaMA-1 词表(32k):            约 700~1000 tokens   (0.7~1 tokens/字)
Qwen 词表(152k, 中文优化):     约 500~700 tokens    (0.5~0.7 tokens/字)
```

**3 倍的 token 差异 = 3 倍推理成本 + 3 倍上下文占用**。这是为什么中文模型必须自己训 tokenizer。

**例 4：数字切分的实测影响**

同一道 4 位数加法题：

```
不切分: "1234+5678" → ["1234", "+", "5678"]  → 3 tokens
        模型必须记住 1234+5678 这个"事实"，无法外推到 1235+5678
切分:   "1234+5678" → ["1","2","3","4","+","5","6","7","8"] → 9 tokens
        模型可以学逐位加法，且能从右往左进位
```

实测（LLaMA 论文报告）：数字切分后，多位数加法的准确率从约 30% 提升到接近 100%（在分布内的位数范围内）。

## 代码速记

```python
from collections import Counter, defaultdict
import re

# ---------- BPE 训练（简化可运行版） ----------
def get_stats(vocab):
    """vocab: {token_tuple: freq}"""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[word[i], word[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    out = {}
    bigram = re.escape(" ".join(pair))          # pair 形如 ('a','b')
    pat = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        w = " ".join(word)
        w = pat.sub("".join(pair), w)
        out[tuple(w.split())] = freq
    return out

def train_bpe(corpus, num_merges=100):
    # 预分词 + 字符级初始化
    vocab = Counter()
    for text in corpus:
        for word in text.split():
            vocab[tuple(list(word) + ["</w>"])] += 1
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return vocab, merges

corpus = ["low low low low low", "lower lower", "newest newest newest newest newest newest",
          "widest widest widest"]
vocab, merges = train_bpe(corpus, num_merges=10)
print(merges[:5])   # [('e','s'), ('es','t'), ('est','</w>'), ...]

# ---------- 应用 BPE ----------
def bpe_encode(word, merges):
    word = tuple(list(word) + ["</w>"])
    for a, b in merges:                          # 按学习顺序应用（关键）
        i = 0
        new = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                new.append(a + b); i += 2
            else:
                new.append(word[i]); i += 1
        word = tuple(new)
    return word

# ---------- 字节级 BPE 的基础：UTF-8 ----------
def to_bytes(s):
    return list(s.encode("utf-8"))
print(to_bytes("中"))     # [228, 184, 173]  即 E4 B8 AD

# ---------- 词表参数量估算 ----------
def embed_params(V, d, tie=False):
    return V * d if tie else 2 * V * d
print(f"32k:  {embed_params(32000, 4096)/1e6:.0f} M")
print(f"128k: {embed_params(128256, 4096)/1e9:.2f} B")

# ---------- 数字按位切分 ----------
def split_digits(text):
    return re.sub(r"\d", lambda m: " " + m.group() + " ", text)
print(split_digits("1234+5678=6912"))
```

## 面试追问

**Q：BPE 和 WordPiece 的本质区别是什么？**
A：合并准则不同。BPE 用"共现频次"，WordPiece 用"似然增益" `freq(ab)/(freq(a)·freq(b))`，后者倾向于合并那些"单独出现也不算特别多、但一起出现比例很高"的强关联对，理论上更符合语言模型的建模目标。另外 WordPiece 用 `##` 前缀区分子词位置，BPE 不做位置区分（靠正则预分词实现类似效果）。

**Q：为什么现在都用 BBPE 而不是 Unigram？**
A：主要是工程和历史惯性——GPT 系列用了 BBPE，后来者跟随以保证生态兼容。技术上 BBPE 的"绝无 OOV"是硬优势（字节是完备基）。Unigram 的优势（多分词采样、概率化）在数据增强场景有价值，但对纯预训练不重要。不过 SentencePiece（Unigram 的实现）在 T5/LLaMA-1 里用过，说明两者都能训出好模型，不是决定性的选择。

**Q：多语言模型怎么设计词表？**
A：核心矛盾是"词表预算有限 vs 语言众多"。做法有三步：(1) 按目标语言的目标占比，对各语言语料做**配额采样**（不能按自然分布，否则英语占 90%）；(2) 在这些配额数据上联合训练 BPE；(3) 检查各语言的 **fertility**（平均每个词产生多少 token），若某语言过高（如 >3）就给它追加 token 配额。Qwen 的 152k 词表就是这么为中文/多语言专门扩的。注意：多语言词表会让英文的 token 数略微增加（因为预算被分走了）。

**Q：词表越大越好吗？**
A：不是，收益递减且有三个代价。(1) **参数量**：2·V·d 线性增长，2 倍词表就是多 1B 参数在 7B 模型上；(2) **长尾 token 学不好**：词表 200k+ 时，大量 token 在训练集中只出现几次，embedding 是噪声；(3) **训练不稳定**：超长尾的 logits 会让 softmax 的分布更难拟合。经验最优区在 32k~150k，且要与模型规模匹配（小模型用小词表）。

**Q：为什么很多模型要扩词表（如 LLaMA 加中文 token）？**
A：因为原词表对某语言压缩率低，导致：(1) 同样的钱能训的 token 变少（推理成本高 2~3 倍）；(2) 上下文窗口被浪费（一句中文占的 token 数是英文的 2 倍）；(3) 长尾字符被拆成字节，模型难以建模。扩展做法是：在新语言语料上训练一批新 token，追加到原词表后，**只训练 embedding 和 lm_head 的新增行**（可以冻结主体）做少量适配，然后全参微调。注意扩词表后 lm_head 的维度变了，需要同步扩展并合理初始化（通常用原 token 的 embedding 均值或对应子 token 的组合）。

**Q：tokenizer 会影响模型的推理长度外推吗？**
A：会，间接但明确。位置编码是按 token 位置算的，压缩率低的 tokenizer 意味着**同样的物理文本会消耗更多位置**，所以"支持 128k token"对不同语言的实际文本容量差 2~3 倍。另外，训练时没见过的 token 组合（如超长重复的数字串）会消耗大量位置，导致模型在该输入上的行为异常。

## 关联知识点
- 预训练数据工程与配比（03-pretrain）
- Transformer 结构与参数量计算（01-transformer）
-  Scaling Law 与 Chinchilla（03-pretrain）
- 长上下文与位置编码（02-attention）
- 继续预训练与词表扩展（05-peft）
