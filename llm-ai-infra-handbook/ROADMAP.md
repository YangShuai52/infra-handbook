# LLM & AI Infra Handbook — 进化路线与 P0 方案细化

> 路线图版本：v0.4.1 ｜ 更新：2026-09-04 ｜ 状态：20 模块 / 163 卡及 GitHub Pages 手册已就绪，下一步完成检索评测
> 定位一句话：一份覆盖 LLM 基础、模型生命周期与 AI Infra 的可运行、可检索、可练习开源手册。

---

## 1. 背景与定位

### 1.1 现状

项目已实现五项核心能力：知识树浏览（163 卡片 / 20 模块）、BM25 检索问答（可升级 RAG）、SM-2 刷题记忆、AI 面试陪练（含无 Key 自测降级）和模型档案聚合。
知识体系已经按 LLM 基础、模型生命周期、AI Infra 与 LLM 系统重构，卡片已带来源与自然语言问法。当前短板转向 v0.4 检索质量：中文自然语言检索尚无 50 条量化基线，中文标签命中仍有已知问题；此外 v0.2 的跨天真实使用调参仍需继续。

### 1.2 市场空隙

| 竞品 | Star | 解决了 | 没解决 |
|---|---|---|---|
| llm-action / Datawhale 系列 | 10k~25k | 八股内容组织 | 只能看，不能练 |
| Dify / RAGFlow / FastGPT | 20k~80k | 查（知识库问答） | 不解决"记住" |
| Anki 系 | 生态巨大 | 记（间隔重复） | 没有 LLM 八股内容，也不懂数学推导 |

**结论：查得到 ≠ 记得住。"内容 × 记忆闭环"是空白点，即本项目定位。**

---

## 2. 北极星目标与阶段规划

**北极星目标：让用户在 30 天内"刷完且记住"全部知识卡片。**

衡量"记住"的两个代理指标：
1. **连续 7 天有复习行为**（说明学习闭环真的转起来了）
2. **卡片通过率随间隔重复轮次上升**（说明记忆在形成）

### 2.1 里程碑

> 状态（2026-09-04）：**v0.2 学习闭环代码完成**，仍在 3 天真实使用观察期；**v0.3 面试陪练已完成并通过浏览器 10 题整轮验收**；下一步进入 v0.4 的 C1 中文检索评测与修复。

| 阶段 | 版本 | 交付物 | 验收标准 |
|---|---|---|---|
| 1 个月 | v0.2 学习闭环 | 刷题模式 + SM-2 间隔重复 + 错题本 | 能完整走通「学卡片 → 自测 → 复习队列 → 标记掌握」循环 |
| 2 个月 | v0.3 面试陪练 | AI 面试官追问模拟 + 练习报告 | 每个模块可模拟一轮 10 题追问，答案引用知识卡片 |
| 3 个月 | v0.4 检索 + 内容 | 混合检索 + 卡片 150+ | 中文自然语言问法命中正确卡片率 > 90% |
| 开源发布 | v0.4.1 纯知识手册 | Docsify + GitHub Pages 自动发布 | 20 模块 / 163 卡片可在线阅读、搜索且无断链 |

### 2.2 不做清单（明确边界）

- ❌ 多用户 / 权限系统 —— Dify 的战场
- ❌ 可视化工作流 —— Dify / FastGPT 的战场
- ❌ 通用文档解析（PDF/扫描件）—— RAGFlow 的战场
- ❌ 团队协作、云同步 —— 与单人学习定位无关
- 原则：**保持"一条命令启动、无 Key 可用、纯本地运行"**

---

## 3. P0：学习闭环（v0.2，细化方案）

### 3.1 功能概述

三个子功能，共用一套复习数据：

1. **刷题模式**：随机抽卡片，隐藏答案自测，自评掌握程度
2. **间隔重复队列**：SM-2 算法决定"今天该复习哪几张"
3. **错题本 + 进度仪表盘**：按模块统计薄弱点

### 3.2 数据模型（SQLite 单文件，`backend/data/study.db`）

保持零外部服务原则，SQLite 标准库 `sqlite3` 即可，不引入 ORM。

```sql
-- 卡片复习状态（一张知识卡片一条记录，首次刷题时懒创建）
CREATE TABLE review_card (
    card_id      TEXT PRIMARY KEY,        -- 对应 kb 卡片 id
    ease         REAL NOT NULL DEFAULT 2.5,   -- EF 因子，下限 1.3
    interval_days REAL NOT NULL DEFAULT 0,     -- 当前间隔（天）
    reps         INTEGER NOT NULL DEFAULT 0,  -- 连续答对次数
    lapses       INTEGER NOT NULL DEFAULT 0,  -- 遗忘（答重来）次数
    due_at       TEXT NOT NULL,               -- ISO 日期，下次应复习时间
    last_review  TEXT,                        -- 最近一次复习时间
    created_at   TEXT NOT NULL
);

-- 复习流水（用于统计与错题本）
CREATE TABLE review_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id       TEXT NOT NULL,
    reviewed_at   TEXT NOT NULL,
    quality       INTEGER NOT NULL,          -- 0~5，见 3.3 映射
    interval_before REAL NOT NULL,
    interval_after  REAL NOT NULL
);
CREATE INDEX idx_review_log_card ON review_log(card_id);
CREATE INDEX idx_review_log_time ON review_log(reviewed_at);
```

- **错题本** = `lapses > 0` 或近期 `quality <= 2` 的卡片视图，无需单独建表
- 单用户设计，无 user 表；将来若做多用户只需加 `user_id` 列

### 3.3 SM-2 算法与交互映射

用户自评四档按钮 → 映射 SM-2 quality（0~5）：

| 前端按钮 | quality | 语义 |
|---|---|---|
| 完全想不起来 | 1 | 黑了，重来 |
| 想起来但有错 | 3 | 有缺口，重来但 EF 不降 |
| 想起来了，有点卡 | 4 | 通过，间隔正常增长 |
| 秒答且完整 | 5 | 通过，间隔加速增长 |

调度规则（标准 SM-2）：

```
quality < 3:  reps=0, lapses+=1, interval=1（明天再来）, ease -= 0.2
quality >= 3: reps+=1
  reps == 1: interval = 1
  reps == 2: interval = 6
  reps >= 3: interval = round(interval * ease)
  quality == 3: ease -= 0.15; quality == 5: ease += 0.1（上限 2.8）
  ease 下限 1.3
due_at = today + interval
```

首次刷一张卡（懒创建 review_card 记录）：`reps=1, interval=1, due_at=明天`。

### 3.4 API 设计（新增，均在 `/api/study` 前缀下）

```
GET  /api/study/queue?limit=20&module=
     → 今日待复习队列：due_at <= today 按 due_at 升序；
       若不足 limit，用"从未刷过 / 随机"的卡片补齐（保证新卡持续进入）
     返回: [{card_id, title, module, is_new, due_at}]

GET  /api/study/quiz/{card_id}
     → 刷题视图：返回题目信息（title/summary/tags）+ 章节标题列表（headings），
       **不返回正文**（避免用户瞄到答案）；正文在提交后由前端调 /api/card/{id} 展示

POST /api/study/answer
     body: {card_id, quality: 1|3|4|5}
     → 写 review_log + 更新 review_card，返回 {next_interval_days, due_at}

GET  /api/study/stats
     → 仪表盘：{总卡片, 已开始, 已掌握(interval>=21), 今日待复习,
               连续学习天数, 按模块的通过率柱状数据}

GET  /api/study/wrong-book?module=
     → 错题本：lapses>0 或近 3 次 quality 均值 <= 2.5 的卡片列表
```

**边界与异常（EARS 式约束）**

- The system shall 在知识库卡片被删除/改名后，把对应 review_card 视为孤儿数据，不进入队列、不计入统计
- When quality 落在 {1,3,4,5} 之外，the system shall 返回 400
- If 当日队列为空，the system shall 返回空列表而非报错，前端展示"今日已刷完"
- While 刷题进行中（quiz 视图已返回），刷新页面或直接调 /api/card 不产生任何 review 记录——只有显式 POST answer 才计数

### 3.5 前端交互（index.html 内新增"刷题"页签）

```
刷题页主流程：
[今日队列 n 张] → 显示问题卡（标题+summary）→ [点击揭晓]
→ 渲染卡片正文 → 四档自评按钮 → 显示"下次复习：X 天后" → 下一张
顶部常驻：今日进度 / 连续天数 / [错题本] [统计] 入口
```

### 3.6 涉及改动点

| 文件 | 改动 |
|---|---|
| `backend/study.py`（新增） | SQLite 初始化、SM-2 调度、队列/统计查询 |
| `backend/app.py` | 挂载 5 个 study 路由；启动时迁移建表 |
| `backend/frontend/index.html` | 新增刷题页签与交互 |
| `backend/requirements.txt` | 无新增（sqlite3 为标准库） |

---

## 4. P1：AI 面试官模拟（v0.3，已完成）

复用现有 RAG 链路（检索 → OpenAI 兼容接口 → SSE）。

**会话模型**：一次模拟 = {module, max_rounds=10, 已问问题列表, 追问深度}。
面试官 prompt 约束：仅从该模块卡片内容出题；每题最多追问 3 层（"为什么→再为什么"）；不得出卡片未覆盖的内容；结束后按卡片内容逐题给分（0~5）+ 生成练习报告（薄弱点 + 推荐复习卡片）。

```
POST /api/interview/start   {module}        → {session_id, first_question}
POST /api/interview/answer  {session_id, answer} → SSE {追问 or 下一题 or 结束}
GET  /api/interview/report  {session_id}    → {逐题得分, 薄弱点, 推荐卡片}
```

无 Key 时降级：退化为"追问链自测模式"——按卡片「追问」小节顺序提问，用户自评，结果直接写入 review_log（与 P0 数据互通）。

**验收状态（2026-09-04）**：B1~B4 全部完成。真实浏览器在无 Key 模式下完成 01-transformer 模块 10 题，自评结果生成可解释报告；薄弱卡片可回跳原文并一键进入今日复习队列。验收同时修复了 Q/A 格式中答案行被误抽为题目的问题，并补充两类题目格式的自动回归测试。

---

## 5. P2：检索升级（v0.4，进行中）

1. **QA 变体注入（已实现，量化回归待 C1 问法集）**
   - 163 张卡片 frontmatter 均包含 3~5 条 `questions`
   - 加载器与 BM25 索引已读取这些问法
2. **修复已知 bug**：`search.py:109` 标签命中用空格切分查询，中文查询永远命不中中文标签——改为对查询词与标签做包含/分词匹配
3. **混合检索（可选升级）**：引入本地 sentence-transformers embedding（中文模型如 bge-small-zh），BM25 分数与余弦相似度加权融合；作为可选依赖，未安装时自动退回纯 BM25，维持"无 Key 可用"

验收：构建 50 条中文自然语言测试问法集，命中率 > 90%。

---

## 6. P3：内容规模化（第一阶段完成，持续维护）

- 已完成 48 → 163 张卡片、10 → 20 个模块的重构与扩容
- 已覆盖多模态、训练/推理 Infra、Agent 进阶、MCP、安全治理与供应链
- 每张卡片均包含可访问的 `source` 和 3~5 条 `questions`
- 新增自动校验与 CI；后续按实际质量审阅持续深化卡片正文
- 扩展方式：在 `backend/kb/<主线>/<模块>/` 下新增 `.md`，重启自动索引

---

## 7. 风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| 自评四档主观性强，间隔参数失真 | 复习排期不准 | 上线后观察通过率分布，必要时收紧 quality==4 的 interval 增速 |
| SQLite 并发写冲突 | 刷题提交偶发失败 | 单用户场景压力极小；uvicorn 单 worker 即可 |
| AI 面试官幻觉（问超出卡片的内容） | 误导学习 | prompt 强约束 + 报告页展示引用卡片供核对 |
| 8000 端口被其他应用占用 | 启动失败 | 支持 PORT 环境变量（已实现） |

---

## 8. 下一步行动

1. 继续完成 v0.2 的 3 天真实使用观察，收集通过率分布并决定是否调整 SM-2 参数
2. 推进 v0.4/C1：建立 50 条中文自然语言问法集与当前命中率基线
3. 修复中文标签命中逻辑，用同一问法集做修复前后对比
4. 用同一问法集量化 QA 变体带来的增益，并持续人工深化新增卡片
5. 仓库推送到 `YangShuai52/infra-handbook` 后，在 Pages 设置中选择 GitHub Actions，完成首次在线发布
