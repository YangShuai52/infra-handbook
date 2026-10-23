---
id: context-engineering
title: 上下文工程：prompt 设计、压缩、记忆与 lost-in-the-middle
module: 19-agent-context
tags:
- 上下文工程
- prompt
- 记忆
- 压缩
- lost-in-the-middle
- few-shot
difficulty: 3
freq: 高
updated: 2026-08-28
summary: 会写 prompt 已经从艺术变成工程。要点：系统提示词定角色与约束、结构化标记隔离、few-shot 的精选与排序、长上下文的压缩与摘要、长期记忆的向量/结构化存储，以及缓解
  lost-in-the-middle（模型忽视中段信息）的布局策略。
source:
- https://modelcontextprotocol.io/specification/
questions:
- 什么是 lost-in-the-middle？怎么缓解？
- few-shot 的样例怎么选、怎么排序？
- 为什么要用 XML/Markdown 标记隔离指令和资料？
- 长期记忆为什么不能只靠"把历史全塞进上下文"？
- 上下文压缩会丢信息，怎么权衡？
---

## 一句话结论

- **系统提示词**：定义角色、目标、约束、输出格式、安全边界；放在最前，用清晰指令而非模糊描述。
- **结构化标记**：用 XML/Markdown 把"指令 / 资料 / 用户问题"隔离（`<context>...</context>`），降低模型混淆。
- **few-shot**：精选与目标任务同分布、由易到难的样例，顺序影响表现（难例放最后常更好）。
- **lost-in-the-middle**：模型更重视首尾、忽视中段；关键资料放头尾，或多次重排。
- **记忆**：短期在上下文窗口、长期用向量库 + 结构化摘要，避免无限堆上下文（成本与注意力衰减）。

## 原理推导

### 为什么位置会影响表现（lost-in-the-middle）

Transformer 的注意力在**长序列中段**更容易被首尾的强信号"稀释"：前部的 system prompt 和后部刚生成的 token 获得更多注意力预算，中间检索到的资料若淹没在大段文本里，贡献被弱化。实证（[Lost in the Middle, 2023]）：把关键信息放开头或结尾，模型利用率显著高于放中间。

工程对策：
1. 关键事实/约束放 system prompt 开头与结尾；
2. 检索资料做**重要性重排**，高相关置顶；
3. 超长资料先摘要再注入，而非整篇堆入；
4. 必要时分多次提问（每次聚焦少量资料）。

### 上下文压缩

- **抽取式压缩**：用模型把长对话/文档抽成要点（实体、决策、未决项）。
- **抽象式摘要**：滚动摘要旧轮次，保留最近 N 轮原文。
- 权衡：压缩省 token 但可能丢细节 → 重要信息（如用户偏好、待办）用**结构化记忆**单独存（JSON/KV），不被压缩影响。

### 长期记忆架构

```
用户消息 → 判断是否需要记忆 → 向量化存入 Memory Store
查询时 → 检索相关记忆 + 当前上下文 → 拼入 prompt
```

- 向量记忆：适合模糊回忆（"上次他说过偏好 X"）。
- 结构化记忆：适合精确字段（用户画像、待办列表）。
- 多 Agent 时各 Agent 维护私有记忆 + 共享黑板（blackboard）通信。

## 具体例子

**客服 Agent 上下文模板**：

```
<system>你是XX客服，只能依据<context>回答，不知道就直说。</system>
<context>{检索到的政策片段，按相关度排序}</context>
<user>{用户问题}</user>
```

- 政策片段重排后置顶最相关 3 条，避免中段丢失。
- 用户历史偏好存结构化记忆（"VIP、偏好中文"），每轮注入。

**代码 Agent**：把仓库结构、相关文件摘要存长期记忆，当前文件全文进短期上下文，避免每步重读全仓。

## 代码速记

```python
# 简单上下文压缩（示意）
def build_prompt(system, retrieved, history, user):
    # 重排：相关度高的检索资料放最前
    retrieved = sorted(retrieved, key=lambda x: -x.score)
    ctx = "\n\n".join(f"[资料{i+1}] {r.text}" for i, r in enumerate(retrieved[:5]))
    # 历史做滚动摘要
    summary = summarize(history[:-K]) if len(history) > K else ""
    return f"<system>{system}</system>\n<context>{ctx}</context>\n{summary}\n<user>{user}</user>"
```

## 面试追问

1. 什么是 lost-in-the-middle？怎么缓解？
2. few-shot 的样例怎么选、怎么排序？
3. 为什么要用 XML/Markdown 标记隔离指令和资料？
4. 长期记忆为什么不能只靠"把历史全塞进上下文"？
5. 上下文压缩会丢信息，怎么权衡？

## 关联知识点

- [Agent 基础：ReAct、Function Calling 与规划](agent-fundamentals)
- [RAG 流水线：切分、召回、重排与失败归因](rag-pipeline)
- [句向量模型：InfoNCE、难负例与 Matryoshka](embedding-model)
