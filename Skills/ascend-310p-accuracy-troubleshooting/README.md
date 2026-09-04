# Ascend 310P 精度问题定位 Skill

用于复现、定位和验证 Ascend 310P 上 vLLM/vLLM-Ascend 推理精度回归。目标不是只找到“某个配置关掉后分数恢复”，而是形成从异常现象到特性、commit、代码路径或首个异常 tensor 的可审计证据链。

## 适用场景

- 模型输出乱码、答非所问或结果不稳定；
- TextVQA、GSM8K、C-Eval、BoolQ、BFCL 等数据集精度下降；
- good/bad branch、日期或 commit 之间出现精度回归；
- 怀疑量化、MTP、图模式、prefix/KV cache、chunked prefill、TP、并发或自定义算子；
- 需要完成特性隔离、commit 二分、tensor dump、代码修复及全量回归。

不用于一般性能调优，也不自动修改 Ascend 驱动、CANN 或 PTA。

## 安装

将整个目录复制到 Codex skills 目录：

```text
<CODEX_HOME>/skills/ascend-310p-accuracy-troubleshooting/
```

必须保留 `SKILL.md`、`references/` 和 `agents/` 的相对目录结构。安装后重新打开任务或刷新 skills 列表。

## 使用

显式调用：

```text
$ascend-310p-accuracy-troubleshooting
```

推荐一次提供：

- 服务器、容器和登录方式；
- vLLM-Ascend 源码路径，或是否需要搭建环境；
- good/bad branch、commit 或日期；
- 模型、权重、量化方式和目标数据集；
- 完整 `vllm serve` 与评测命令；
- 标准分、实测分、允许波动；
- 可用 NPU/TP；
- 是否授权联网、安装普通依赖和修改源码。

示例：

```text
使用 $ascend-310p-accuracy-troubleshooting 定位 Ascend 310P 上的精度回归。
模型为 <model>，目标数据集为 GSM8K；good commit 为 <good>，bad commit 为 <bad>。
标准分为 <expected>，实测为 <actual>，允许波动 <threshold>。
vLLM-Ascend 源码目录为 <path>，服务命令为 <command>。
允许在指定容器内联网、切换诊断 commit、安装 vLLM/vLLM-Ascend、ais_bench 和普通 Python/build 依赖。
请复现、定位根因、修复并完成目标数据集全量回归。
```

缺失信息能从环境、源码或日志取得时，skill 会优先自行检查。

## 工作方式

1. 固化环境、源码、权重、数据集和验收标准。
2. 验证运行时确实加载目标 vLLM/vLLM-Ascend 源码。
3. 先运行结构化 curl、长序列 curl、真实失败样本和固定小样本。
4. 排除端口、旧服务、模板、截断、后处理等伪回归。
5. 单变量隔离特性，分析 good/bad diff；必要时二分 commit。
6. 二分不可行时，通过 tensor dump 找首个稳定异常点。
7. 修复后按低成本测试、目标集子集、目标数据集全量顺序回归。

定位阶段避免反复运行全量数据集。简单 curl 通过也不能证明目标数据集精度恢复。

## 验证等级

| 等级 | 范围 | 可声明结论 |
|---|---|---|
| L0 | 服务和日志 | 服务可达 |
| L1 | 单个确定性请求 | 单例基本生成正确 |
| L2 | 5–10 个确定性请求 | 最小 smoke 通过 |
| L3 | 失败样本及目标数据集固定子集 | 候选修复可进入全量验收 |
| L4 | 用户指定目标数据集全量 | 该目标数据集达到验收标准 |
| L5 | 用户或项目要求的更广评测集合 | 约定范围内综合回归通过 |

L4 不固定为 TextVQA；目标数据集由用户报告决定。

## 关键约束

- `vllm serve`、日志复现和源码调试必须在用户确认的 vLLM-Ascend 源码目录执行。
- `ais_bench` 必须在 benchmark 仓库根目录执行。
- 每个 commit 或配置候选必须重启服务，不复用旧进程、KV cache 或 prefix cache。
- good/bad 只改变目标变量；构建失败、OOM 和 HCCL 异常不能判为精度 bad。
- 安装完成后必须核对 verified vLLM commit、真实 import 路径和 editable install 路径；证据不完整时停止测试。
- 修复完成必须执行用户指定目标数据集的 L4 全量回归；L5 仅在验收范围要求时执行。

## 目录结构

```text
ascend-310p-accuracy-troubleshooting/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
└── references/
    ├── deployment.md
    ├── accuracy-testing.md
    └── triage-workflow.md
```

- `SKILL.md`：入口、路由、核心原则和安全边界。
- `references/deployment.md`：源码安装、版本配对和运行时验收。
- `references/accuracy-testing.md`：服务启动、分级 curl、ais_bench 和 L0–L5。
- `references/triage-workflow.md`：good/bad 对比、权重加载审计、特性隔离、二分和 tensor dump。

## 安全边界

- 不自动升级、降级、重装或卸载 Ascend 驱动。
- CANN、NNAL、PTA/`torch_npu` 变更必须先有定位证据，再取得用户明确确认。
- 不批量删除源码、构建目录、工具链、数据集或测试产物。
- 不终止或抢占其他任务占用的 NPU。

## 输出内容

最终结果包括环境指纹、精确复现、good/bad 矩阵、根因证据、修复内容、最高验证等级、回归结果、产物位置及未解决风险。无法证明根因时，给出已排除项和下一项最有区分力的实验。
