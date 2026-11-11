---
id: agent-fundamentals
title: Agent 基础：ReAct、Function Calling 与规划
module: 19-agent-context
tags:
- Agent
- ReAct
- Function Calling
- 工具调用
- 规划
- 反思
difficulty: 3
freq: 高
updated: 2026-08-28
summary: Agent = 模型 + 工具 + 记忆 + 规划。核心循环是 ReAct（Thought→Action→Observation）让模型边想边调工具；Function/Tool
  Calling 把"调哪个工具、参数如何"结构化训练出来；Plan-and-Execute / Reflexion 等范式解决长任务规划与自我修正。
source:
- https://modelcontextprotocol.io/specification/
questions:
- ReAct 相比纯 Chain（一次生成答案）强在哪？
- Function Calling 的训练数据长什么样？运行时怎么处理？
- Plan-and-Execute 和 Reflexion 分别解决什么痛点？
- Agent 的"记忆"一般怎么实现？为什么不能无限塞上下文？
- 多 Agent 协作的通信开销问题怎么理解？
---

## 一句话结论

- **Agent 四件套**：LLM（大脑）、Tools（手）、Memory（记忆）、Planning（规划）。
- **ReAct**：把"推理轨迹"和"行动"交错输出（Thought → Action → Observation → …），是最基础也最有效的范式。
- **Function Calling**：模型输出结构化 `{name, arguments}`，由运行时执行工具并回传结果；通过 SFT/RL 训练模型"何时调、调什么、参数对"。
- **进阶范式**：Plan-and-Execute（先规划再逐步执行）、Reflexion（用失败反思改进）、CodeAct（用代码当动作空间）。

## 原理推导

### ReAct 循环

提示词让模型按固定格式输出：

```
Thought: 我需要先查天气再决定穿什么
Action: search_weather(city="北京")
Observation: 北京今天 22°C 多云
Thought: 温度适宜，无需厚外套
Action: Finish(answer="建议穿薄外套")
```

- 模型一次只决定"下一步动作"，环境执行后把 Observation 拼回上下文，再让模型决定下一步。
- 优势：可解释（看到思考链）、能多步纠错、能调外部工具突破参数化知识边界。

### Function Calling 的数据格式

训练数据形如：

```json
{"role":"user","content":"北京天气怎么样？"}
{"role":"assistant","tool_calls":[
   {"name":"get_weather","arguments":{"city":"北京"}}]}
{"role":"tool","name":"get_weather","content":"22°C 多云"}
{"role":"assistant","content":"北京今天 22°C 多云，建议…"}
```

- 模型被训练成：在需要外部信息时输出 `tool_calls`，运行时解析→执行→把结果以 `role:tool` 回灌，继续生成。
- 关键工程：参数 schema（JSON Schema）约束、类型校验、超时/重试、并行调用。

### 规划与反思

- **Plan-and-Execute**：先让模型产出完整计划（子任务列表），再逐个执行，适合长任务；比纯 ReAct 更不容易"走偏"。
- **Reflexion**：任务失败后，让模型读自己的轨迹做**语言化反思**（"我之前因为没先查 X 而错"），把反思写入记忆，下次避免。
- **CodeAct**：用"写/执行 Python"作为统一动作空间，比自然语言的 Action 更灵活、可组合。

## 具体例子

- **客服 Agent**：ReAct 循环里可调 `query_order / refund / search_kb`，Observation 来自数据库。
- **代码 Agent（如 DeepSeek-Coder / OpenHands）**：CodeAct，模型生成并运行代码，Observation 是 stdout/报错，迭代修 bug。
- **多 Agent**：Planner 拆解 → Worker 执行 → Critic 评审，通信开销随 Agent 数线性增长，需控制拓扑（流水线 vs 星型）。

## 代码速记

```python
# ReAct 最简循环（示意）
ctx = system_prompt + user_msg
while not done:
    out = llm(ctx)                       # 解析出 Thought / Action
    if out.action == "Finish": break
    obs = run_tool(out.action, out.args) # 工具执行
    ctx += f"\nObservation: {obs}"       # 回灌上下文
```

## 面试追问

1. ReAct 相比纯 Chain（一次生成答案）强在哪？
2. Function Calling 的训练数据长什么样？运行时怎么处理？
3. Plan-and-Execute 和 Reflexion 分别解决什么痛点？
4. Agent 的"记忆"一般怎么实现？为什么不能无限塞上下文？
5. 多 Agent 协作的通信开销问题怎么理解？

## 关联知识点

- [上下文工程：prompt 设计、压缩与记忆](context-engineering)
- [RAG 流水线：切分、召回、重排与失败归因](rag-pipeline)
- [句向量模型：InfoNCE、难负例与 Matryoshka](embedding-model)
