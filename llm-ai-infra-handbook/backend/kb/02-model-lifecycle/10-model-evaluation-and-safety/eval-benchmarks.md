---
id: eval-benchmarks
title: 评测榜单：MMLU / GSM8K / HumanEval / Needle 与数据污染
module: 10-model-evaluation-safety
tags:
- 评测
- MMLU
- GSM8K
- HumanEval
- 长上下文
- 数据污染
- Arena
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 评测分多维：知识(MMLU/C-Eval)、推理(GSM8K/MATH)、代码(HumanEval/MBPP)、长上下文(Needle/RULER)、指令遵循(IFEval)、竞技场(Arena)。注意评测陷阱——prompt
  敏感性、答案抽取方式、以及训练数据污染（需去污染）。
source:
- https://arxiv.org/abs/2211.09110
questions:
- MMLU / GSM8K / HumanEval 各测什么能力？
- loglikelihood 评测和生成式评测的差别与适用？
- 什么是 pass@k？为什么报告多个 k？
- 数据污染怎么检测和去除？
- Needle-in-a-Haystack 测的是什么？它的局限？
---

## 一句话结论

- **知识**：MMLU(57科多选)、C-Eval(中文)、CMMLU。
- **推理**：GSM8K(小学应用题)、MATH(竞赛)、BBH(硬推理)。
- **代码**：HumanEval/MBPP（pass@k，看 k 次里能否通过）。
- **长上下文**：Needle-in-a-Haystack（海中捞针）、RULER（多类型长依赖）。
- **指令/竞技**：IFEval(格式遵循)、Chatbot Arena(人类对打败率 ELO)。
- **数据污染**：测试集可能进了预训练语料 → 必须去污染(n-gram/embedding 重叠检测)才可信。

## 原理推导

### 两种评测方式

- **loglikelihood（困惑度式）**：对选择题，比较各选项的条件概率 `P(option|prompt)`，取最高 → 快、确定、但不反映"生成能力"。
- **生成式（open-ended）**：让模型生成答案，再用规则/模型抽取判分 → 更贴近真实使用，但有抽取噪声与 prompt 敏感。

同一题用不同 prompt 模板，得分可差几个点 → 报告结果要固定模板并多次取平均。

### pass@k 与 HumanEval

代码任务用 `pass@k`：采样 k 个程序，任一通过单元测试即算过：

```
pass@k = 1 - C(n-c, k)/C(n, k)     # n 次采样中 c 次通过
```

报告 pass@1（最严）、pass@10、pass@100。

### 数据污染检测与去污染

- **n-gram 重叠**：测试题与预训练语料做 13-gram 匹配，命中即疑似污染。
- **embedding 相似**：语义级去重，抓"换表述"的泄漏。
- **黄金标准**：从预训练语料中**剔除**所有评测集相关文本（如 Phi / OLMo 公开清洗流程）。

### 长上下文评测

- **Needle-in-a-Haystack**：在长文本随机位置插入一句事实（"针"），问该事实 → 测模型是否"看见"。
- **RULER**：扩展为多类型（检索/计数/排序/聚合）+ 多种长度，比单针更全。

## 具体例子

- Llama-3 报告 MMLU 79%、GSM8K 8-shot 83%。
- 某模型在 HumanEval pass@1=40%、pass@10=65% → 说明"会做但一次不一定对"，适合配 sampling/self-consistency。
- 自建业务评测：抽 200 条真实客服对话，人工标注"是否正确"，比刷公开榜更反映落地效果。

## 代码速记

```python
# pass@k 估计（示意）
from math import comb
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n-c, k) / comb(n, k)
# n 次采样通过 c 次 → 至少一次通过的概率
```

## 面试追问

1. MMLU / GSM8K / HumanEval 各测什么能力？
2. loglikelihood 评测和生成式评测的差别与适用？
3. 什么是 pass@k？为什么报告多个 k？
4. 数据污染怎么检测和去除？
5. Needle-in-a-Haystack 测的是什么？它的局限？

## 关联知识点

- [幻觉：分类、检测与缓解](hallucination)
- [LLM-as-Judge：偏差与消除](llm-as-judge)
- [RAG 流水线：切分、召回、重排与失败归因](rag-pipeline)
