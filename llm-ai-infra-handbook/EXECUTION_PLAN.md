# LLM & AI Infra Handbook — 执行文档

> 依据 `ROADMAP.md` 拆解 ｜ 更新：2026-09-04 ｜ 状态：Phase B（v0.3）验收完成，下一步 Phase C1
> 原则：**每步可独立验证、可独立提交**；每步结束服务都能正常启动（`PORT=8001 ./run.sh`）。

---

## 总览

```
Phase A（v0.2 学习闭环）  A1 → A2 → A3 → A4 → A5     ✅ 代码完成，真实使用观察中
Phase B（v0.3 面试陪练）  B1 → B2 → B3 → B4          ✅ 浏览器整轮验收通过
Phase C（v0.4 检索+内容） C1 → C2 ∥ C3 → C4          ← 下一阶段，C2/C3 可并行
```

依赖关系：A 是 B 的地基（面试结果写 review_log）；C1 独立，可随时插入。

---

## Phase A：学习闭环（v0.2）✅ 代码完成（2026-09-03），待 3 天真实使用调参

### A0. 基线确认（~5 分钟）✅

- [x] 确认服务可启动：`PORT=8001 ./run.sh`，`/healthz`、`/api/status` 正常
- [x] 确认 `.venv` 已由 run.sh 自动建好（上一轮已修复）
- [x] 提交当前基线（requirements.txt / run.sh / app.py healthz 修复）作为一个 commit

**DoD**：干净 checkout 上一条命令即可跑通。

### A1. 数据层：SQLite 初始化与迁移（新增 `backend/study.py` 前半）✅

- [x] `backend/data/` 下创建/打开 `study.db`
- [x] 建表 `review_card`、`review_log`（DDL 见 ROADMAP 3.2，启动时 `CREATE TABLE IF NOT EXISTS` 幂等执行）
- [x] 通用工具函数：`_now()`（ISO 时间）、`_today()`（ISO 日期）、孤儿卡片清理（card_id 不在当前 CARDS 中的记录跳过）

**验证**：python 一行脚本调 `init_db()`，`sqlite3 study.db .schema` 看到两张表；重复执行不报错。
**DoD**：幂等建表 + 不破坏已有数据。

### A2. SM-2 调度与查询（`backend/study.py` 后半）✅

- [x] `answer(card_id, quality) -> dict`：quality ∈ {1,3,4,5} 否则 raise；按 ROADMAP 3.3 规则更新 review_card、写 review_log；返回 `{next_interval_days, due_at}`
  - 首次刷卡懒创建记录（reps=1, interval=1）
  - quality<3：reps 清零、lapses+1、interval=1、ease-=0.2（下限 1.3）
  - quality>=3：reps==1→interval 1，reps==2→interval 6，reps>=3→interval×ease（四舍五入）；q==3 再减 0.15，q==5 加 0.1（上限 2.8）
- [x] `queue(limit, module) -> list`：due_at ≤ today 按 due_at 升序；不足 limit 用「从未刷过」卡片随机补齐（module 过滤同时作用于两者）
- [x] `quiz_view(card_id)`：标题/summary/tags/headings，**不含正文**
- [x] `stats()`：总卡片 / 已开始 / 已掌握（interval≥21）/ 今日待复习 / 连续学习天数（review_log 按 natural day 去重连续计数）/ 各模块通过率
- [x] `wrong_book(module)`：lapses>0 或近 3 次 quality 均值 ≤ 2.5

**验证**：写一段临时脚本模拟 10 次答题序列，手工核对 interval/ease 演进是否符合上表（例如 q=4 三连：1→6→6×2.5=15）。
**DoD**：SM-2 演进与 ROADMAP 3.3 一致；队列/统计/错题本三个查询返回结构正确。

### A3. API 接入（`backend/app.py`）✅

- [x] 启动时调 `study.init_db()`
- [x] 新增 5 个路由：`GET /api/study/queue`、`GET /api/study/quiz/{card_id}`、`POST /api/study/answer`、`GET /api/study/stats`、`GET /api/study/wrong-book`
- [x] `POST /api/study/answer`：非法 quality → 400；卡片不存在 → 404
- [x] 注意：新路由放在静态目录挂载之前（同 healthz 教训）

**验证**（curl 串）：
```bash
curl -s localhost:8001/api/study/queue | jq           # 有新卡
curl -s -X POST localhost:8001/api/study/answer -H 'Content-Type: application/json' \
     -d '{"card_id":"position-encoding-rope","quality":4}' | jq
curl -s localhost:8001/api/study/answer -d '{"card_id":"x","quality":9}'  # → 400
curl -s localhost:8001/api/study/stats | jq
```
**DoD**：5 个路由全部按契约返回；错误路径返回 400/404。

### A4. 前端刷题页签（`backend/frontend/index.html`）✅

- [x] 顶部导航新增「刷题」页签，页面含：今日队列、自测卡（标题+summary → 揭晓按钮 → 拉正文渲染）、四档自评按钮、结果显示（下次复习 X 天后）
- [x] 顶部常驻：今日进度 / 连续天数；入口：错题本、统计
- [x] 错题本页：列表 + 跳转卡片详情；统计页：模块通过率（简单条形即可，纯 CSS）

**验证**：浏览器手动走通「学卡片 → 自测 → 下一张 → 刷完队列」全流程；刷新页面进度不丢。
**DoD**：ROADMAP 3.5 的交互流程完整可走。

### A5. 收尾与发布（代码/文档完成；3 天真实使用与调参待进行）

- [x] README 增加刷题功能说明（四大模块 + 数据存储 + 访问范围说明）
- [x] ROADMAP / EXECUTION_PLAN 勾掉 v0.2 验收项
- [ ] 用真实学习数据跑 3 天，记录通过率分布，必要时微调 ease 增减幅度（见 ROADMAP 风险表）

**DoD（v0.2 验收）**：连续使用中「学 → 测 → 复习队列 → 标记掌握」闭环成立，stats 数据合理。

---

## Phase B：AI 面试官模拟（v0.3）✅ B1~B4 完成并通过浏览器验收（2026-09-04）

### B1. 会话与出题（新增 `backend/interview.py`）✅

- [x] 会话存储：内存 dict 即可（`{session_id: {module, max_rounds=10, history[], depth}}`），重启丢失可接受
- [x] 出题约束 prompt：仅从模块卡片内容出题；每题最多追问 3 层；不问卡片未覆盖内容；引用卡片 id
- [x] `start(module)`：检索该模块卡片拼上下文 → 生成第一题（无 Key 时返回降级标记）

补充：模型需输出严格 JSON 且 card_id 可溯源到给定卡片，不合法重试一次仍失败即降级（宁降级也不出烂题）。

**DoD**：start 返回合法第一题，且问题可溯源到某张卡片。已验证（7 项 mock）。

### B2. 答题流（SSE）✅

- [x] `POST /api/interview/answer`：复用 generator 的 SSE 通道；流式输出「评价 + 追问 / 下一题 / 结束」
- [x] 每题结束时把该题得分（模型按 0~5 自评）写入 review_log（quality 口径与 P0 一致，数据互通）
- [x] 追加 `POST /api/interview/start` 路由，链路连通

设计补充：评价文本流式转发，末行 `@@DECISION@@` 携带 JSON 决策（score / followup·next·end），
服务端钳制追问深度 ≤3、轮次 ≤10、card_id 必须可溯源。
**重要修复**：同一场面试对同一张卡的后续回答只写流水、不推进 SM-2 间隔
（否则间隔会被层层放大到 3720 天，卡片等于退出复习队列）。

**DoD**：一轮完整对话 10 题不中断；review_log 出现对应记录。
已验证：16 轮（10 道正式题 + 6 次追问）无中断，落库 16 条，间隔 ≤1 天。

### B3. 练习报告 ✅

- [x] `GET /api/interview/report`：逐题得分、薄弱点聚类（按卡片/模块）、推荐复习卡片（直接给出进队列的建议）
- [x] 无 Key 降级：追问链自测模式（按卡片「追问」小节提问 + 自评写回）

实现要点：
- 自测模式直接解析卡片「面试追问」小节成题（当前 48 张卡全部带该小节，兼容编号列表与 Q/A 两种格式），
  每卡最多 3 题、单场上限 10 题；`POST /api/interview/self-answer` 自评写回，quality 即 score
- 报告统一支持两种模式：逐题 score/卡片标题/原回答、按卡片与模块聚类、分数分布、
  薄弱点（均分<3 或最低分≤2）、推荐前 5 条并附原因；每条薄弱点都带 card_id，可点回原文
- 新增 `study.enqueue_today()` + `POST /api/interview/enqueue`：一键把推荐卡片提前到今日队列；
  未学新卡不会被凭空插入（返回 skipped_new —— 它们本来就在队列里）

**DoD**：报告可解释（每条薄弱点都能点回卡片）。
已验证：真实知识库 10 题自测全链路 + 报告 + 一键进队列；LLM 模式报告含非法分数/决策缺失处理；HTTP 层 400/404 齐备。

### B4. 前端面试页 ✅

- [x] 「面试模拟」页签：选模块 → 开始 → 对话流 UI → 结束出报告

浏览器验收补充：
- 在无 Key 降级模式下完成 01-transformer 模块 10/10 题，报告显示 10 个有效评分并正确聚合薄弱卡片
- 报告中的薄弱卡片可点击回到知识卡片原文；「加入今日复习队列」确认回流 1 张卡片
- 验收时发现并修复 Q/A 格式中答案行被误识别为题目的问题；新增 `backend/tests/test_interview.py` 覆盖 Q/A 与编号列表两种格式

**DoD（v0.3 验收）**：任选模块完成一轮 10 题模拟，报告引用卡片且错题进入复习队列。
已验证：真实浏览器整轮通过；测试在临时副本中进行，未修改正式学习数据。

---

## Phase C：检索升级与内容扩容（v0.4）— 进行中（C2 实现、C3 完成）

### C1. 修复中文标签命中 bug（独立，可最先做）

- [ ] `search.py:109`：`query.lower().split()` 改为「查询分词结果与标签的双向包含 + 中文 bigram 匹配」
- [ ] 建立 50 条中文自然语言测试问法集（`backend/tests/search_cases.json`），写一个可重复运行的评估脚本输出命中率

**DoD**：问法集命中率显著提升且有数字基线（当前值 → 修复后值）。

### C2. QA 变体注入

- [x] frontmatter 新增 `questions: [...]`（163 张卡片均为 3~5 条面试官问法）
- [x] `kb_lib.load_cards` 读取；`search.Index` 语料追加变体
- [ ] 用 C1 的问法集回归验证

**DoD**：命中率相对 C1 再提升；无 Key 检索模式同样受益。

### C3. 内容扩容到 150+（与 C2 并行推进）

- [x] 从原 10 模块重构为 20 模块，覆盖 LLM 基础、模型生命周期、AI Infra、RAG / Agent 与安全治理
- [x] 扩容到 163 张卡片；每张卡均带 `source` 与 `questions`
- [x] 保持“结论 → 机制 → 工程权衡 → 例子 → 观测排障 → 追问”的统一结构
- [x] 新增 `tools/validate_frontmatter.py` 与 GitHub Actions 质量门禁

**DoD**：卡片 ≥ 150，全部通过 frontmatter 校验。✅ 2026-09-04：163 卡 / 20 模块 / 163 个唯一 ID。

### C4. 混合检索（可选，最后做）

- [ ] 可选依赖 sentence-transformers + bge-small-zh；未安装自动退回纯 BM25
- [ ] BM25 归一化分数 × 余弦相似度加权融合（权重进环境变量可调）

**DoD**：装与不装都能跑；问法集命中率不回退。

---

## Phase D：GitHub Pages 纯知识手册（v0.4.1）✅

### D1. 静态阅读站 ✅

- [x] 使用 Docsify 建立章节侧栏、全文搜索、上下章导航、代码复制和移动端布局
- [x] 支持 KaTeX 公式渲染与深浅色主题
- [x] 首页提供四条学习路径，模块页提供章节导语和卡片摘要

### D2. 单一内容源与自动构建 ✅

- [x] `backend/kb/` 保持唯一正文来源，不提交重复知识卡片
- [x] `tools/build_handbook.py` 在发布时移除 frontmatter、补齐元数据与来源并重写内部链接
- [x] `tools/validate_handbook.py` 检查 163 张卡片和站内路由完整性

### D3. GitHub Pages 自动发布 ✅

- [x] `.github/workflows/deploy-pages.yml` 在 `main` 更新后生成并发布 `_site`
- [x] 发布前串联知识库校验与静态站点校验
- [x] README 增加在线地址、首次启用步骤和本地预览方式

**DoD**：本地静态产物包含 20 模块 / 163 卡片且无断链；推送目标仓库后只需在 Pages 设置中选择一次 GitHub Actions。

---

## 执行节奏建议

| 周 | 目标 |
|---|---|
| 第 1 周 | A0~A3（数据层 + API 全通） |
| 第 2 周 | A4~A5，开始日常使用收集数据 |
| 第 3 周 | B1~B2 |
| 第 4 周 | B3~B4，v0.3 验收 |
| 第 5 周起 | C1 → C2 ∥ C3，内容持续填充 |

## 提交纪律

- 每个 Step 一个 commit，格式：`A2: SM-2 调度与队列查询`
- 每个 Step 完成即跑一次冒烟：`PORT=8001 ./run.sh` + 对应 curl
- C3 内容卡片按模块分 commit，便于回滚

## 已知环境注意

- 本机 8000 被 Semantica 应用占用，统一用 `PORT=8001`
- `.venv` 由 run.sh 自动维护，不要手动往系统 python 装依赖
