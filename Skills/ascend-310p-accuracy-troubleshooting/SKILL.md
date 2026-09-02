---
name: ascend-310p-accuracy-troubleshooting
description: 定位、复现、二分并验证 Ascend 310P 上 vLLM/vLLM-Ascend 推理精度回归。适用于环境搭建、curl 或 ais_bench 复现、特性隔离、commit/PR 根因定位及修复回归；不用于一般性能调优。
---

# Ascend 310P 精度问题定位

从用户报告的精度异常出发，建立可复现、可审计的证据链，逐步缩小到特性、commit、代码路径或首个异常 tensor。除非用户同时要求，不把性能调优混入精度定位。

## 按任务读取资料

- 搭建、切换或核验 vLLM、vLLM-Ascend 与 310P 环境：读取 [references/deployment.md](references/deployment.md)。使用其中的源码安装方式；安装后必须通过运行时源码一致性验收。
- 启动服务、设计分级 curl、运行 ais_bench：读取 [references/accuracy-testing.md](references/accuracy-testing.md)。
- 对比 good/bad、隔离特性、二分 commit、tensor dump 或验证修复：读取 [references/triage-workflow.md](references/triage-workflow.md)。
- 任务跨阶段时才读取多份资料，不预先加载无关内容。

## 核心原则

1. **先复现，后修改。** 优先建立低成本且能稳定区分 good/bad 的判据。简单健康检查不能证明精度正常。
2. **一次只改变一个变量。** 保持权重、环境、服务参数、生成参数、评测配置和数据集一致。静态 diff 只用于提出假设，不能单独证明根因。
3. **结论不得超过证据。** 按 `accuracy-testing.md` 的 L0–L5 标记验证范围。L1/L2 不能表述为目标数据集精度恢复；完整修复至少需要用户指定目标数据集的 L4 全量回归，除非存在明确外部阻塞。
4. **运行时证据优先。** commit、版本字符串或 `pip install` 成功不足以证明代码生效；必须核对 Python、包元数据、真实 import 路径、vLLM-Ascend HEAD 与 verified vLLM commit。
5. **区分模型输出错误与评测错误。** 同时保存原始输出、提取答案、归一化答案、单样本得分和 evaluator 日志，再决定是否进入算子或 kernel 定位。
6. **根因必须闭环。** 分数变化只是现象。根因需要稳定复现、单变量控制、代码机制解释，以及 revert、补丁、特性开关或首个异常 tensor 等交叉证据。

## 工作流

### 1. 固化问题与验收标准

记录以下信息；可从环境或日志取得时自行检查，不重复询问：

- last-known-good 与 first-known-bad 的 vLLM/vLLM-Ascend branch、commit 和源码路径；
- 模型与权重身份、量化方式、CANN、PTA/`torch_npu`、驱动、固件、镜像及 310P 拓扑；
- 完整服务命令、目标数据集、benchmark commit、评测命令、标准值与允许波动；
- 生成参数、chat template、后处理、预期与实际结果；
- 用户指定的 NPU、TP、容器及已经确认的结论。

目标数据集由用户报告决定，可能是 TextVQA、GSM8K 或其他数据集；不得默认绑定 TextVQA。

### 2. 建立可比较环境

若已有兼容环境，获取 vLLM-Ascend 源码绝对路径、branch 和 commit；若没有，先检查现有安装，再按用户指定分支及其 verified vLLM commit 搭建。每次安装或切换后执行 `deployment.md` 的验收；状态不是 `ready` 时停止服务化测试。

用户未指定设备时，检查健康、占用、可用显存与 TP 拓扑，只选择满足条件的空闲卡。不得终止或抢占其他任务。

### 3. 分级复现

保持报告配置，依次执行：结构化事实 curl、长输入/长输出确定性 curl、真实失败样本、10–100 条目标数据集固定子集。覆盖与问题相关的 decode 长度、prefix reuse、batch 变化、小并发、MTP 或图路径。无区分力时先增强最小测试，不直接跳到昂贵全量评测。

### 4. 定位

先排除端口、model name、旧服务、模板、EOS、截断、后处理和数据集版本造成的伪回归。再分析 good/bad 区间内 vLLM 与 vLLM-Ascend 的代码变化，建立候选假设。

按证据选择实验：

- 单变量隔离图模式、量化/融合、MTP、prefix/KV/mamba cache、chunked prefill、TP、并发、采样及自定义算子；
- 审计权重并发加载、重复原地变换、QKV 重排、量化参数和 TP shard；
- 判据稳定时执行 commit 二分；构建、OOM、HCCL 或启动失败记为 `blocked/skip`，不能判为精度 bad；
- 二分不可行时，对齐 good/bad 执行坐标，从粗粒度 checkpoint 缩小到首个异常 layer/operator/tensor。

### 5. 修复与验收

仅在用户要求时修改代码。依次验证：稳定最小判据、失败样本、固定子集、用户指定目标数据集全量 L4。仅当用户、CI 或发布标准要求时执行更广泛的 L5。补丁涉及自定义算子、图模式或权重加载时，补充对应单元、构建及路径测试。

## 执行不变量

- `vllm serve`、服务日志复现和源码调试必须在用户确认的 **vLLM-Ascend 源码目录**执行；启动前记录 `pwd`、branch、HEAD、实际模块路径、PID、端口和 served model name。
- `ais_bench` 必须在 benchmark 仓库根目录执行，并使用目标版本文档中的配置名和参数。
- 每个 commit 或配置候选必须重启服务并使用干净请求状态，不复用上一候选的旧进程、KV cache 或 prefix cache。
- good/bad 两侧保持目标变量以外条件一致。保存完整命令、环境指纹、请求、原始响应、评分产物、日志及 run manifest。
- 不以任意非零浮点差异作为根因；先测同环境自身波动，再使用 dtype/量化方式合适的误差指标。

## 授权与安全边界

- 用户已授权指定服务器/容器联网和修改诊断环境后，该授权覆盖范围内的 clone/fetch、干净 worktree/clone 切换、submodule、vLLM、vLLM-Ascend、ais_bench 及普通 Python/build 依赖安装，无需逐步重复确认。记录每次 commit、安装命令和依赖变化；保护已有脏工作树。
- 上述授权不包含驱动、固件、CANN、NNAL、PTA/`torch_npu`、批量删除、设备抢占或无关系统包。
- 不升级、降级、重装或卸载 Ascend 驱动。CANN 或 PTA 变更只有在证据已定位到该组件且用户明确确认后才能执行。
- 不自动批量删除源码、构建目录、工具链、数据集或输出。需要清理时列出精确路径，交由用户手动处理或逐文件明确授权。

## 交付结果

给出紧凑、可审计的结论：

- 环境、版本与运行时源码指纹；
- 精确复现命令、目标数据集、期望值、实际值和最高验证等级；
- good/bad 单变量矩阵及原始产物位置；
- 已隔离的特性、commit/PR、代码路径或首个异常 tensor/operator 及证据；
- 修复内容与 L1–L5 回归结果；
- 未解决风险、外部阻塞和下一项最有区分力的实验。
