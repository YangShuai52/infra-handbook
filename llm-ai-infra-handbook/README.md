# LLM & AI Infra Handbook

一份可运行、可检索、可练习的 LLM 与 AI 基础设施开源知识手册。内容从模型原理延伸到数据与训练、GPU 与分布式系统、推理服务、RAG / Agent 和生产治理，同时保留刷题、间隔重复与面试模拟能力。

当前版本包含 **20 个模块、163 张知识卡片**。项目保持纯本地、一条命令启动；未配置模型 API 时，知识浏览、检索、刷题和自测仍然可用。

## 在线知识手册

仓库发布到 `YangShuai52/infra-handbook` 并启用 GitHub Pages 后，可直接访问：

**[https://yangshuai52.github.io/infra-handbook/](https://yangshuai52.github.io/infra-handbook/)**

在线手册是纯静态版本，提供章节侧栏、全文搜索、上下章导航、代码复制、公式渲染、深浅色主题和移动端阅读。正文仍以 `backend/kb/` 为唯一内容源；GitHub Actions 会在每次知识卡片更新后自动生成并发布网页，不需要维护第二份卡片。

## 知识体系

1. **LLM 基础**：数学与性能模型、语言建模与分词、Transformer、注意力与长上下文、现代架构扩展。
2. **模型生命周期**：数据工程、预训练与 Scaling、SFT / PEFT、对齐与推理训练、模型评估与安全评测。
3. **AI 基础设施**：加速器与服务器、GPU 内核与编译器、分布式训练、训练平台、推理算法、推理运行时、集群运营与成本。
4. **LLM 系统**：RAG 与检索、Agent 与上下文、安全治理与供应链。
5. **实践与参考**：可复现实验、公式与检查单、故障手册、质量校验工具。

完整目录、学习路径和模块边界见 [知识库导航](backend/kb/README.md)，分类依据与主要来源见 [体系研究记录](docs/research/llm-ai-infra-taxonomy.md)。

## 产品能力

- **知识树**：浏览 20 个模块的 Markdown 卡片，每张卡片包含结论、机制、工程权衡、例子、观测排障、来源和追问。
- **检索问答**：中英混合 BM25，索引标题、标签、摘要、常见问法和正文；配置兼容 OpenAI 的接口后可使用带引用的流式 RAG。
- **刷题记忆**：基于 SM-2 的本地间隔重复，含今日队列、错题本和分模块统计。
- **面试模拟**：支持模型追问，也支持无 API Key 的卡片追问自测，结果可回流复习队列。
- **模型档案**：本地种子档案、Hugging Face Trending 与 arXiv 最新论文入口。



## 快速开始

前置要求：Python 3.10+。

```bash
cd llm-ai-infra-handbook
chmod +x run.sh
./run.sh
# 浏览器打开 http://localhost:8000
```

首次运行会创建隔离环境并安装 `backend/requirements.txt` 中的依赖。服务默认绑定 `127.0.0.1`；需要局域网访问时使用：

```bash
HOST=0.0.0.0 ./run.sh
```

启用 RAG 生成：

```bash
LLM_API_KEY=sk-xxx ./run.sh
LLM_API_KEY=sk-xxx LLM_BASE_URL=https://example.com/v1 LLM_MODEL=your-model ./run.sh
```



## 仓库结构

```text
.
├── backend/
│   ├── kb/                 # 20 个模块、163 张 Markdown 知识卡片
│   ├── frontend/index.html # 零构建单页界面
│   ├── app.py              # API 与静态服务
│   ├── kb_lib.py           # 卡片解析与章节切分
│   ├── search.py           # 中英混合 BM25
│   ├── study.py            # SM-2 学习数据与调度
│   └── interview.py        # 面试会话、追问与报告
├── docs/research/          # 分类体系研究与一手来源
├── labs/                   # 可复现实验入口
├── reference/              # 公式、检查单和故障手册入口
├── site/                   # GitHub Pages 的 Docsify 外壳与学习路径
├── tools/                  # 知识库质量工具
└── run.sh
```



## 编写知识卡片

在对应模块目录新增 Markdown 文件。最小 frontmatter：

```yaml
---
id: my-topic
title: 标题
module: 16-inference-runtime-serving
tags: [推理服务, 调度]
difficulty: 3
freq: 中
updated: 2026-09-04
source:
  - https://example.com/primary-source
questions:
  - 这个机制解决了什么瓶颈？
  - 如何验证它在生产环境中的收益？
  - 它的质量、性能和复杂度如何权衡？
summary: 一句话说明核心结论。
---
```

提交前运行：

```bash
python tools/validate_frontmatter.py
python tools/build_handbook.py --output _site
python tools/validate_handbook.py _site
```

校验会检查模块归属、ID 唯一性、来源链接、问法数量、最低卡片规模和静态站点内部链接；GitHub Actions 也会执行同一检查。

本地预览纯知识手册：

```bash
python tools/build_handbook.py --output _site
python -m http.server 4173 --directory _site
# 浏览器打开 http://localhost:4173
```

首次发布时，在 GitHub 仓库的 `Settings → Pages → Build and deployment` 中把 Source 设为 **GitHub Actions**。之后推送到 `main` 即会自动更新网页。

## 数据与开源准备

学习记录仅保存在本地 `backend/data/study.db`。联网内容缓存位于 `backend/data/`，不会改变知识卡片正文。仓库已按计划中的 `infra-handbook` 方向整理，但尚未代替维护者选择开源许可证或执行远端推送。